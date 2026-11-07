"""Fail-closed runtime adapter for the frozen Prod-V16.5 components.

The adapter deliberately owns no camera, robot, planner, or safety decision.
It routes one already-computed ViT feature map through both registered FCPG
correctors, proposes one five-command Actor candidate for the planner's
existing candidate bank, and supplies State/FK LeWM predictions for that bank.
Only the session-frozen anchor authority may affect planner anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from fcpg.prod_v16_5 import (
    ACTOR_CHECKPOINT_FORMAT,
    LEWM_CHECKPOINT_FORMAT,
    StateFKLeWM,
    StructuredGoalFiveCommandActor,
)
from fcpg.so101_fcpg_lewm_bridge import build_so101_fcpg_corrector


ANCHOR_AUTHORITIES = ("nominal", "original_fcpg", "tuned_fcpg")
CHECKPOINT_FORMAT = "so101-prod-v16-5-fixed-final-checkpoint-v1"
PROTOCOL_ID = "so101-prod-v16-5-fk-seeded-e2e-v1"
TUNED_FCPG_RADIUS_PX = 24.0
TUNED_FCPG_MAX_RESIDUAL_PX = 24.0
ORIGINAL_FCPG_RADIUS_PX = 16.0
ORIGINAL_FCPG_MODEL_MAX_RESIDUAL_PX = 20.0
CANONICAL_TO_DEGREES_SCALE = 100.0
RAW_VIT_ENCODER_TENSOR_COUNT = 198
FCPG_CORRECTOR_TENSOR_COUNT = 61
VIEWS = ("wrist", "column")
V165_ACTOR_SOURCE = "v16_5_structured_goal_actor"
NO_CONTACT_CLEARANCE_M = 0.010
NO_CONTACT_JAW_TOLERANCE_DEG = 0.05
NO_CONTACT_SWEEP_INCREMENT_DEG = 0.25
CONTACT_AUTHORIZATION_FORMAT = "so101-prod-v16-5-contact-authorization-v1"
CONTACT_AUTHORIZATION_DECISION = "GO_CONTACT_HARDWARE"
CONTACT_AUTHORIZATION_GATES = (
    "offline_replay",
    "shadow",
    "zero_motion",
    "no_contact_approach",
)
CONTACT_RUNTIME_CONFIG_FIELDS = (
    "projection_calibration_sha256",
    "live_anchor_calibration_sha256",
    "geometry_protocol",
    "source_image_size",
    "planning_image_size",
    "contact_height_m",
    "peg_clearance_m",
    "max_first_command_jump_deg",
    "max_prefix_step_jump_deg",
    "live_action_scale",
    "control_mode",
    "experiment_method",
    "task_class",
    "phase_contract",
    "require_live_column_calibration",
)


class V165RuntimeContractError(RuntimeError):
    """A frozen V16.5 runtime contract is absent, stale, or inconsistent."""


def _checkpoint_digest_map(
    records: tuple[tuple[str, str, str], ...],
) -> dict[str, str]:
    return {name: digest for name, _path, digest in records}


def _canonical_runtime_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(CONTACT_RUNTIME_CONFIG_FIELDS):
        raise V165RuntimeContractError(
            "V16.5 contact authorization runtime config is incomplete"
        )
    try:
        canonical = json.loads(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise V165RuntimeContractError(
            "V16.5 contact authorization runtime config is not finite JSON"
        ) from exc
    for key in ("projection_calibration_sha256",):
        if not isinstance(canonical[key], str) or len(canonical[key]) != 64:
            raise V165RuntimeContractError(
                f"V16.5 contact runtime config has invalid {key}"
            )
    live_digest = canonical["live_anchor_calibration_sha256"]
    if live_digest is not None and (
        not isinstance(live_digest, str) or len(live_digest) != 64
    ):
        raise V165RuntimeContractError(
            "V16.5 contact runtime config has invalid live-anchor identity"
        )
    if canonical["control_mode"] != "phase_target" or canonical[
        "phase_contract"
    ] != "phase_target_state_machine":
        raise V165RuntimeContractError(
            "V16.5 contact runtime config has invalid phase contract"
        )
    if not isinstance(canonical["require_live_column_calibration"], bool):
        raise V165RuntimeContractError(
            "V16.5 contact runtime config has invalid calibration requirement"
        )
    for key in (
        "contact_height_m",
        "peg_clearance_m",
        "max_first_command_jump_deg",
        "max_prefix_step_jump_deg",
        "live_action_scale",
    ):
        raw = canonical[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise V165RuntimeContractError(
                f"V16.5 contact runtime config has invalid {key}"
            )
        if not np.isfinite(float(raw)) or float(raw) <= 0.0:
            raise V165RuntimeContractError(
                f"V16.5 contact runtime config has invalid {key}"
            )
    return canonical


def v165_rgb_view_nonblack(frame: np.ndarray) -> bool:
    """Use the endpoint-cache media rule for a current runtime RGB slot."""

    value = np.asarray(frame)
    if value.shape != (224, 224, 3) or value.dtype != np.uint8:
        raise V165RuntimeContractError(
            "V16.5 current RGB view must be uint8 [224,224,3]"
        )
    return bool(np.any(value != 0))


def validate_contact_authorization_receipt(
    path: Path,
    *,
    bundle: "V165BundleReceipt",
) -> dict[str, Any]:
    """Validate the separately reviewed gate that permits contact hardware.

    Training checkpoints remain explicitly deployment-ineligible.  This small
    runtime receipt does not add a model metric: it binds four already-reviewed
    integration gates to the exact frozen bundle used by this process.
    """

    resolved = _resolved_file(path, label="V16.5 contact authorization receipt")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V165RuntimeContractError(
            "V16.5 contact authorization receipt is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise V165RuntimeContractError(
            "V16.5 contact authorization receipt must contain an object"
        )
    expected_digests = _checkpoint_digest_map(bundle.checkpoint_records)
    if (
        payload.get("format") != CONTACT_AUTHORIZATION_FORMAT
        or payload.get("protocol_id") != bundle.protocol_id
        or payload.get("decision") != CONTACT_AUTHORIZATION_DECISION
        or payload.get("deployment_authorized") is not True
        or payload.get("bundle_id") != bundle.bundle_id
        or payload.get("anchor_authority") != bundle.anchor_authority
        or payload.get("checkpoint_records") != expected_digests
    ):
        raise V165RuntimeContractError(
            "V16.5 contact authorization does not bind the frozen runtime bundle"
        )
    gates = payload.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(CONTACT_AUTHORIZATION_GATES):
        raise V165RuntimeContractError(
            "V16.5 contact authorization must contain the four registered gates"
        )
    runtime_config = _canonical_runtime_config(payload.get("runtime_config"))
    for name in CONTACT_AUTHORIZATION_GATES:
        gate = gates[name]
        gate_runtime_config = _canonical_runtime_config(
            gate.get("runtime_config") if isinstance(gate, dict) else None
        )
        if not isinstance(gate, dict) or (
            gate.get("decision") != "GO"
            or gate.get("protocol_id") != bundle.protocol_id
            or gate.get("bundle_id") != bundle.bundle_id
            or gate.get("anchor_authority") != bundle.anchor_authority
            or gate.get("checkpoint_records") != expected_digests
            or gate_runtime_config != runtime_config
        ):
            raise V165RuntimeContractError(
                f"V16.5 contact authorization gate is absent or stale: {name}"
            )
    if (
        gates["offline_replay"].get("planner_safety_executed") is not True
        or gates["offline_replay"].get("integrated_gate_eligible") is not True
        or gates["offline_replay"].get("hardware") is not False
        or gates["shadow"].get("commands_sent") != 0
        or gates["shadow"].get("hardware") is not True
        or gates["zero_motion"].get("passed") is not True
        or gates["zero_motion"].get("zero_motion") is not True
        or gates["no_contact_approach"].get("hardware") is not True
        or gates["no_contact_approach"].get("v165_no_contact_approach") is not True
        or gates["no_contact_approach"].get("v165_no_contact_consumed") is not True
        or str(gates["no_contact_approach"].get("phase", "")).lower()
        != "approach"
    ):
        raise V165RuntimeContractError(
            "V16.5 contact authorization gate evidence is not deployment-eligible"
        )
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "format": CONTACT_AUTHORIZATION_FORMAT,
        "decision": CONTACT_AUTHORIZATION_DECISION,
        "gates": list(CONTACT_AUTHORIZATION_GATES),
        "runtime_config": runtime_config,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_file(path: Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise V165RuntimeContractError(f"{label} is not a readable file: {resolved}")
    return resolved


def _load_checkpoint(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise V165RuntimeContractError(f"could not load {label}") from exc
    if not isinstance(payload, dict):
        raise V165RuntimeContractError(f"{label} must contain an object")
    return payload


def _checkpoint_action_contract(payload: Mapping[str, Any], *, label: str) -> dict[str, np.ndarray]:
    raw = payload.get("action_contract")
    if not isinstance(raw, Mapping) or set(raw) != {"step_cap", "lower", "upper"}:
        raise V165RuntimeContractError(
            f"{label} action_contract must contain step_cap/lower/upper"
        )
    result = {
        "step_cap": np.asarray(raw["step_cap"], dtype=np.float32),
        "action_lower": np.asarray(raw["lower"], dtype=np.float32),
        "action_upper": np.asarray(raw["upper"], dtype=np.float32),
    }
    if any(value.shape != (6,) for value in result.values()):
        raise V165RuntimeContractError(f"{label} action contract vectors must be [6]")
    if not all(bool(np.isfinite(value).all()) for value in result.values()):
        raise V165RuntimeContractError(f"{label} action contract must be finite")
    if bool((result["step_cap"] <= 0).any()) or bool(
        (result["action_lower"] >= result["action_upper"]).any()
    ):
        raise V165RuntimeContractError(f"{label} action contract is invalid")
    return result


def _same_action_contract(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> bool:
    return all(
        np.array_equal(np.asarray(left[name]), np.asarray(right[name]))
        for name in ("step_cap", "action_lower", "action_upper")
    )


def _state_dict_sha256(state: Mapping[str, Any], *, label: str) -> str:
    """Match the registered FCPG tensor-state receipt, not the container SHA."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise V165RuntimeContractError(f"{label} contains a non-tensor state")
        value = tensor.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _assert_module_action_contract(
    module: torch.nn.Module,
    expected: Mapping[str, np.ndarray],
    *,
    label: str,
) -> None:
    """Check post-strict-load buffers element-for-element."""

    for name in ("step_cap", "action_lower", "action_upper"):
        actual = getattr(module, name, None)
        if not isinstance(actual, torch.Tensor):
            raise V165RuntimeContractError(f"{label} has no {name} buffer")
        reference = torch.as_tensor(
            expected[name], device=actual.device, dtype=actual.dtype
        )
        if not torch.equal(actual, reference):
            raise V165RuntimeContractError(
                f"{label} post-load action buffer drifted: {name}"
            )


def _checkpoint_vit_source(
    payload: Mapping[str, Any], *, label: str
) -> tuple[Path, str]:
    raw = payload.get("vit_source")
    if not isinstance(raw, (str, Path)):
        raise V165RuntimeContractError(f"{label} has no frozen ViT source")
    path = _resolved_file(Path(raw), label=f"{label} frozen ViT source")
    return path, _sha256_file(path)


def _freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.requires_grad_(False)
    module.eval()
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise V165RuntimeContractError("a V16.5 runtime module remained trainable")
    return module


@dataclass(frozen=True)
class V165BundleReceipt:
    """Immutable identity of one runtime bundle and its anchor authority."""

    anchor_authority: str
    bundle_id: str
    protocol_id: str
    checkpoint_records: tuple[tuple[str, str, str], ...]
    original_max_residual_px: float
    vit_source_sha256: str = ""
    action_contract: tuple[tuple[str, tuple[float, ...]], ...] = ()
    runtime_action_contract_deg: tuple[tuple[str, tuple[float, ...]], ...] = ()
    original_local_radius_px: float = ORIGINAL_FCPG_RADIUS_PX
    original_model_max_residual_px: float = ORIGINAL_FCPG_MODEL_MAX_RESIDUAL_PX
    tuned_max_residual_px: float = TUNED_FCPG_MAX_RESIDUAL_PX
    tuned_local_radius_px: float = TUNED_FCPG_RADIUS_PX
    contact_authorization: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.anchor_authority not in ANCHOR_AUTHORITIES:
            raise ValueError(f"unsupported anchor authority: {self.anchor_authority!r}")
        if not self.bundle_id or self.protocol_id != PROTOCOL_ID:
            raise ValueError("V16.5 bundle identity is invalid")
        if not np.isfinite(self.original_max_residual_px) or self.original_max_residual_px <= 0:
            raise ValueError("original FCPG residual envelope must be positive")
        if len(self.vit_source_sha256) not in {0, 64}:
            raise ValueError("V16.5 frozen ViT source SHA is invalid")
        if self.original_local_radius_px != ORIGINAL_FCPG_RADIUS_PX:
            raise ValueError("original FCPG local radius must be exactly 16 px")
        if self.original_model_max_residual_px != ORIGINAL_FCPG_MODEL_MAX_RESIDUAL_PX:
            raise ValueError("original FCPG model residual must be exactly 20 px")
        if self.tuned_max_residual_px != TUNED_FCPG_MAX_RESIDUAL_PX:
            raise ValueError("tuned FCPG max residual must be exactly 24 px")
        if self.tuned_local_radius_px != TUNED_FCPG_RADIUS_PX:
            raise ValueError("tuned FCPG local radius must be exactly 24 px")

    def as_dict(self) -> dict[str, Any]:
        contact_authorization = dict(self.contact_authorization)
        return {
            "format": "so101-prod-v16-5-runtime-bundle-v1",
            "anchor_authority": self.anchor_authority,
            "bundle_id": self.bundle_id,
            "protocol_id": self.protocol_id,
            "checkpoint_records": {
                name: {"path": path, "sha256": digest}
                for name, path, digest in self.checkpoint_records
            },
            "corrector_contract": {
                "original_max_residual_px": self.original_max_residual_px,
                "original_local_radius_px": self.original_local_radius_px,
                "original_model_max_residual_px": self.original_model_max_residual_px,
                "tuned_local_radius_px": self.tuned_local_radius_px,
                "tuned_max_residual_px": self.tuned_max_residual_px,
                "vit_source_sha256": self.vit_source_sha256 or None,
                "shared_raw_vit_forward": True,
                "fcpg_uses_registered_v12_patch_projector": True,
                "actor_and_context_bridge_use_raw_full256": True,
                "context_bridge_is_visual_context_carrier_only": True,
                "context_bridge_is_not_original_corrector": True,
                "inactive_correctors_are_shadow_only": True,
            },
            "action_contract": {
                name: list(values) for name, values in self.action_contract
            },
            "runtime_state_contract": {
                "model_units": "canonical_normalized_joint_command",
                "robot_units": "lerobot_calibration_degrees",
                "canonical_to_degrees_scale": CANONICAL_TO_DEGREES_SCALE,
                "action_contract_deg": {
                    name: list(values)
                    for name, values in self.runtime_action_contract_deg
                },
            },
            "decision_contract": {
                "legacy_observer_sigma_controls_close": False,
                "legacy_observer_excitation_control_influence": False,
                "prod_v13_control_influence": False,
                "global_locator_control_influence": False,
                "contact_hardware_authorized": bool(contact_authorization),
                "contact_authorization": contact_authorization or None,
            },
        }


@dataclass(frozen=True)
class V165AnchorRouting:
    authority: str
    authoritative: np.ndarray
    original_fcpg: np.ndarray
    tuned_fcpg: np.ndarray
    original_residual_px: np.ndarray
    tuned_residual_px: np.ndarray
    authority_within_envelope: bool

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "anchor_authority": self.authority,
            "shared_raw_vit_forward": True,
            "fcpg_feature_map": "registered_v12_patch_projector_of_shared_raw_tokens",
            "original_fcpg_px": self.original_fcpg,
            "tuned_fcpg_px": self.tuned_fcpg,
            "original_residual_px": self.original_residual_px,
            "tuned_residual_px": self.tuned_residual_px,
            "authority_within_envelope": self.authority_within_envelope,
            "inactive_correctors_shadow_only": True,
        }


@dataclass(frozen=True)
class _RoundInputs:
    full256: np.ndarray
    view_valid: np.ndarray
    current_state: np.ndarray
    canonical_current_state: np.ndarray
    past_completed_commands: np.ndarray


class V165RuntimeAdapter:
    """Inference-only adapter; planner safety remains the sole command authority."""

    def __init__(
        self,
        *,
        receipt: V165BundleReceipt,
        original_corrector: torch.nn.Module,
        tuned_corrector: torch.nn.Module,
        actor: StructuredGoalFiveCommandActor,
        lewm: StateFKLeWM,
        fcpg_patch_projector: torch.nn.Module | None = None,
        actor_baseline_kind: str,
        action_contract: Mapping[str, np.ndarray] | None = None,
        device: torch.device,
    ) -> None:
        if actor_baseline_kind not in {"analytic", "repeat", "damped"}:
            raise V165RuntimeContractError("Actor checkpoint has no registered baseline")
        self._receipt = receipt
        self._device = torch.device(device)
        self._original_corrector = _freeze_module(original_corrector)
        self._tuned_corrector = _freeze_module(tuned_corrector)
        self._actor = _freeze_module(actor)
        self._lewm = _freeze_module(lewm)
        self._fcpg_patch_projector = _freeze_module(
            torch.nn.Identity()
            if fcpg_patch_projector is None
            else fcpg_patch_projector
        )
        self._actor_baseline_kind = actor_baseline_kind
        if action_contract is None:
            action_contract = {
                "step_cap": np.full(6, 0.05, dtype=np.float32),
                "action_lower": np.full(6, -1.0, dtype=np.float32),
                "action_upper": np.full(6, 1.0, dtype=np.float32),
            }
        self._action_contract = {
            name: np.asarray(action_contract[name], dtype=np.float32).copy()
            for name in ("step_cap", "action_lower", "action_upper")
        }
        self._round: _RoundInputs | None = None

    @property
    def anchor_authority(self) -> str:
        return self._receipt.anchor_authority

    @property
    def receipt(self) -> dict[str, Any]:
        return self._receipt.as_dict()

    @property
    def contact_hardware_authorized(self) -> bool:
        return bool(self._receipt.contact_authorization)

    @property
    def contact_runtime_config(self) -> dict[str, Any] | None:
        if not self.contact_hardware_authorized:
            return None
        raw = dict(self._receipt.contact_authorization).get("runtime_config")
        return _canonical_runtime_config(raw)

    def assert_contact_runtime_config(self, actual: Mapping[str, Any]) -> None:
        """Reject reusing a reviewed contact Gate under changed safety inputs."""

        expected = self.contact_runtime_config
        if expected is None:
            raise V165RuntimeContractError(
                "V16.5 contact runtime config requires an authorized bundle"
            )
        observed = _canonical_runtime_config(dict(actual))
        if observed != expected:
            raise V165RuntimeContractError(
                "V16.5 contact runtime configuration changed after review"
            )

    @torch.inference_mode()
    def project_fcpg_feature_map(self, raw_full256: torch.Tensor) -> torch.Tensor:
        """Apply the registered 1298 V12 patch projector to shared raw tokens."""

        raw = torch.as_tensor(raw_full256, device=self._device)
        if raw.ndim != 4 or raw.shape[-2:] != (256, 192):
            raise V165RuntimeContractError(
                "shared raw V12 tokens must be [B,T,256,192]"
            )
        projected = self._fcpg_patch_projector(raw)
        if projected.shape != raw.shape or not bool(torch.isfinite(projected).all()):
            raise V165RuntimeContractError("V12 FCPG patch projector output drifted")
        return (
            projected.reshape(-1, 16, 16, 192)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    @classmethod
    def from_checkpoints(
        cls,
        *,
        anchor_authority: str,
        context_bridge_checkpoint: Path,
        original_fcpg_checkpoint: Path,
        v165_source_checkpoint: Path,
        tuned_fcpg_checkpoint: Path,
        lewm_checkpoint: Path,
        actor_checkpoint: Path,
        grounding_model: Any,
        runtime_action_lower_deg: np.ndarray,
        runtime_action_upper_deg: np.ndarray,
        runtime_max_step_deg: np.ndarray,
        original_max_residual_px: float,
        device: torch.device,
        contact_authorization_receipt: Path | None = None,
    ) -> "V165RuntimeAdapter":
        if anchor_authority not in ANCHOR_AUTHORITIES:
            raise V165RuntimeContractError(
                f"anchor_authority must be one of {ANCHOR_AUTHORITIES}"
            )
        if contact_authorization_receipt is not None:
            raise V165RuntimeContractError(
                "V16.5 contact hardware remains sealed until post-training integration review"
            )
        paths = {
            "context_bridge_carrier": _resolved_file(
                context_bridge_checkpoint, label="context bridge checkpoint"
            ),
            "registered_original_fcpg": _resolved_file(
                original_fcpg_checkpoint, label="registered original FCPG checkpoint"
            ),
            "v165_source_checkpoint": _resolved_file(
                v165_source_checkpoint, label="registered V16.5 V12 source checkpoint"
            ),
            "tuned_fcpg": _resolved_file(
                tuned_fcpg_checkpoint, label="tuned FCPG checkpoint"
            ),
            "lewm": _resolved_file(lewm_checkpoint, label="V16.5 LeWM checkpoint"),
            "actor": _resolved_file(actor_checkpoint, label="V16.5 Actor checkpoint"),
        }
        path_sha256 = {name: _sha256_file(path) for name, path in paths.items()}
        original_payload = _load_checkpoint(
            paths["registered_original_fcpg"],
            label="registered original FCPG checkpoint",
        )
        source_payload = _load_checkpoint(
            paths["v165_source_checkpoint"], label="registered V16.5 V12 source checkpoint"
        )
        fcpg_payload = _load_checkpoint(paths["tuned_fcpg"], label="tuned FCPG checkpoint")
        lewm_payload = _load_checkpoint(paths["lewm"], label="V16.5 LeWM checkpoint")
        actor_payload = _load_checkpoint(paths["actor"], label="V16.5 Actor checkpoint")
        for label, payload, stage in (
            ("tuned FCPG", fcpg_payload, "fcpg"),
            ("V16.5 LeWM", lewm_payload, "lewm"),
            ("V16.5 Actor", actor_payload, "actor320k"),
        ):
            if (
                payload.get("format") != CHECKPOINT_FORMAT
                or payload.get("stage") != stage
                or payload.get("protocol_id") != PROTOCOL_ID
                or payload.get("deployment_authorized") is not False
            ):
                raise V165RuntimeContractError(f"{label} checkpoint contract drifted")
        if (
            fcpg_payload.get("corrector_variant") != "independent3"
            or float(fcpg_payload.get("local_radius_px", -1.0)) != TUNED_FCPG_RADIUS_PX
            or float(fcpg_payload.get("max_residual_px", -1.0))
            != TUNED_FCPG_MAX_RESIDUAL_PX
        ):
            raise V165RuntimeContractError(
                "tuned FCPG must be independent3 with an explicit 24 px contract"
            )
        if lewm_payload.get("model_format") != LEWM_CHECKPOINT_FORMAT:
            raise V165RuntimeContractError("V16.5 LeWM model format drifted")
        if actor_payload.get("model_format") != ACTOR_CHECKPOINT_FORMAT:
            raise V165RuntimeContractError("V16.5 Actor model format drifted")
        if lewm_payload.get("metrics", {}).get("gate_pass") is not True:
            raise V165RuntimeContractError("V16.5 LeWM checkpoint did not pass its gate")
        if actor_payload.get("metrics", {}).get("gate_pass") is not True:
            raise V165RuntimeContractError("V16.5 Actor checkpoint did not pass its gate")
        fcpg_metrics = fcpg_payload.get("outer_development_metrics")
        try:
            fcpg_gate = bool(
                isinstance(fcpg_metrics, Mapping)
                and float(fcpg_metrics["improvement_px"]) > 0.0
                and float(fcpg_metrics["in_bounds_fraction"]) == 1.0
                and all(
                    float(fcpg_metrics["per_view"][view]["improvement_px"]) > 0.0
                    for view in VIEWS
                )
            )
        except (KeyError, TypeError, ValueError):
            fcpg_gate = False
        if not fcpg_gate:
            raise V165RuntimeContractError(
                "tuned FCPG checkpoint did not pass its registered development gate"
            )
        initialization = fcpg_payload.get("initialization")
        if not isinstance(initialization, Mapping):
            raise V165RuntimeContractError("tuned FCPG initialization receipt is absent")
        if (
            initialization.get("warmstart_checkpoint_sha256")
            != path_sha256["registered_original_fcpg"]
        ):
            raise V165RuntimeContractError(
                "original runtime FCPG is not the tuned checkpoint warm-start"
            )
        if initialization.get("source_checkpoint_sha256") != path_sha256[
            "v165_source_checkpoint"
        ]:
            raise V165RuntimeContractError(
                "registered V16.5 source is not the tuned FCPG source"
            )
        original_state = original_payload.get("corrector")
        if not isinstance(original_state, Mapping) or len(original_state) != FCPG_CORRECTOR_TENSOR_COUNT:
            raise V165RuntimeContractError("registered original FCPG has no corrector")
        original_state_sha = _state_dict_sha256(
            original_state, label="registered original FCPG"
        )
        if (
            original_payload.get("source_family") != "v12"
            or original_payload.get("corrector_variant") != "independent3"
            or original_payload.get("source_checkpoint_sha256")
            != path_sha256["v165_source_checkpoint"]
            or original_payload.get("corrector_state_sha256_after")
            != original_state_sha
            or initialization.get("warmstart_corrector_state_sha256")
            != original_state_sha
            or original_payload.get("has_physical_labels") is not False
            or original_payload.get("training_scope") != "synthetic_degradation"
        ):
            raise V165RuntimeContractError(
                "registered original FCPG warm-start contract drifted"
            )

        source_model_state = source_payload.get("model")
        if not isinstance(source_model_state, Mapping):
            raise V165RuntimeContractError("registered V16.5 source has no model state")
        source_encoder_state = {
            str(name).removeprefix("vit.encoder."): tensor
            for name, tensor in source_model_state.items()
            if str(name).startswith("vit.encoder.")
        }
        carrier_encoder = getattr(
            getattr(grounding_model, "base", None), "encoder", None
        )
        if not isinstance(carrier_encoder, torch.nn.Module):
            raise V165RuntimeContractError("context carrier has no raw ViT encoder")
        carrier_encoder_state = carrier_encoder.state_dict()
        if (
            len(source_encoder_state) != RAW_VIT_ENCODER_TENSOR_COUNT
            or set(source_encoder_state) != set(carrier_encoder_state)
            or any(
                not torch.equal(
                    source_encoder_state[name].detach().cpu(),
                    carrier_encoder_state[name].detach().cpu(),
                )
                for name in source_encoder_state
            )
        ):
            raise V165RuntimeContractError(
                "context carrier and V16.5 source raw ViT encoder tensors differ"
            )
        v165_vit_path, v165_vit_sha = _checkpoint_vit_source(
            source_payload, label="registered V16.5 source"
        )
        if (
            initialization.get("vit_source_sha256") != v165_vit_sha
            or original_payload.get("vit_source_sha256") != v165_vit_sha
        ):
            raise V165RuntimeContractError(
                "V16.5 raw ViT source identity is not proven across carrier/correctors"
            )
        protocol_sha = fcpg_payload.get("protocol_sha256")
        if not isinstance(protocol_sha, str) or not protocol_sha or any(
            payload.get("protocol_sha256") != protocol_sha
            for payload in (lewm_payload, actor_payload)
        ):
            raise V165RuntimeContractError("V16.5 checkpoint protocol hashes differ")

        fcpg_action_contract = _checkpoint_action_contract(
            fcpg_payload, label="tuned FCPG"
        )
        lewm_contract = _checkpoint_action_contract(lewm_payload, label="LeWM")
        actor_contract = _checkpoint_action_contract(actor_payload, label="Actor")
        if not _same_action_contract(lewm_contract, actor_contract) or not _same_action_contract(
            lewm_contract, fcpg_action_contract
        ):
            raise V165RuntimeContractError(
                "FCPG, Actor, and LeWM action contracts differ"
            )
        runtime_lower = np.asarray(runtime_action_lower_deg, dtype=np.float32)
        runtime_upper = np.asarray(runtime_action_upper_deg, dtype=np.float32)
        runtime_step = np.asarray(runtime_max_step_deg, dtype=np.float32)
        if any(value.shape != (6,) for value in (runtime_lower, runtime_upper, runtime_step)):
            raise V165RuntimeContractError("runtime action contract vectors must be [6]")
        if not all(
            bool(np.isfinite(value).all())
            for value in (runtime_lower, runtime_upper, runtime_step)
        ) or bool((runtime_lower >= runtime_upper).any()) or bool((runtime_step <= 0).any()):
            raise V165RuntimeContractError("runtime action contract is invalid")
        model_lower_deg = (
            lewm_contract["action_lower"] * np.float32(CANONICAL_TO_DEGREES_SCALE)
        )
        model_upper_deg = (
            lewm_contract["action_upper"] * np.float32(CANONICAL_TO_DEGREES_SCALE)
        )
        model_step_deg = (
            lewm_contract["step_cap"] * np.float32(CANONICAL_TO_DEGREES_SCALE)
        )
        # The learned domain must contain every hard-runtime state and every
        # transition allowed by the common planner. Otherwise LeWM would
        # silently project a candidate that runtime could still execute.
        if (
            np.any(model_lower_deg > runtime_lower)
            or np.any(model_upper_deg < runtime_upper)
            or np.any(model_step_deg + np.float32(1.0e-6) < runtime_step)
        ):
            raise V165RuntimeContractError(
                "checkpoint action contract is incompatible with runtime safety"
            )
        tensor_contract = {
            name: torch.from_numpy(value).to(device)
            for name, value in lewm_contract.items()
        }
        lewm_kwargs = lewm_payload.get("model_kwargs")
        if not isinstance(lewm_kwargs, Mapping):
            raise V165RuntimeContractError("LeWM model_kwargs are absent")
        try:
            lewm = StateFKLeWM(
                **tensor_contract,
                residual_cap=torch.as_tensor(
                    lewm_kwargs["residual_cap"], device=device, dtype=torch.float32
                ),
                dimension=int(lewm_kwargs["dimension"]),
                layers=int(lewm_kwargs["layers"]),
            ).to(device)
            actor = StructuredGoalFiveCommandActor(
                **tensor_contract,
                dimension=int(actor_payload.get("model_kwargs", {}).get("dimension", -1)),
            ).to(device)
            lewm.load_state_dict(lewm_payload["state_dict"], strict=True)
            actor.load_state_dict(actor_payload["state_dict"], strict=True)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise V165RuntimeContractError(
                "V16.5 Actor/LeWM modules do not strict-load"
            ) from exc
        _assert_module_action_contract(lewm, lewm_contract, label="V16.5 LeWM")
        _assert_module_action_contract(actor, actor_contract, label="V16.5 Actor")

        # The Actor consumes the raw 16x16 ViT grid. FCPG was trained on that
        # same raw encoder followed by the registered V12 source's learned
        # patch_projector, so the projector is a separate, strict-loaded role.
        try:
            from v10.model import _BoundedResidualProjector

            patch_projector = _BoundedResidualProjector(192).to(device)
            projector_state = {
                str(name).removeprefix("patch_projector."): tensor
                for name, tensor in source_model_state.items()
                if str(name).startswith("patch_projector.")
            }
            patch_projector.load_state_dict(projector_state, strict=True)
        except (ImportError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise V165RuntimeContractError(
                "registered V12 FCPG patch projector does not strict-load"
            ) from exc
        projector_state_sha = _state_dict_sha256(
            patch_projector.state_dict(), label="V12 FCPG patch projector"
        )

        tuned_state = fcpg_payload.get("corrector")
        if not isinstance(tuned_state, Mapping) or len(tuned_state) != FCPG_CORRECTOR_TENSOR_COUNT:
            raise V165RuntimeContractError("tuned FCPG corrector tensor inventory drifted")
        original = build_so101_fcpg_corrector(
            grounding_model,
            camera_context_dim=5,
            variant="independent3",
        ).to(device)
        original.local_radius_px = ORIGINAL_FCPG_RADIUS_PX
        original.max_residual_px = ORIGINAL_FCPG_MODEL_MAX_RESIDUAL_PX
        tuned = build_so101_fcpg_corrector(
            grounding_model,
            camera_context_dim=5,
            variant="independent3",
        ).to(device)
        tuned.local_radius_px = TUNED_FCPG_RADIUS_PX
        tuned.max_residual_px = TUNED_FCPG_MAX_RESIDUAL_PX
        try:
            original.load_state_dict(original_state, strict=True)
            tuned.load_state_dict(tuned_state, strict=True)
        except (KeyError, RuntimeError) as exc:
            raise V165RuntimeContractError(
                "original/tuned FCPG correctors do not strict-load"
            ) from exc
        if (
            float(getattr(original, "local_radius_px", -1.0))
            != ORIGINAL_FCPG_RADIUS_PX
            or float(getattr(original, "max_residual_px", -1.0))
            != ORIGINAL_FCPG_MODEL_MAX_RESIDUAL_PX
            or float(getattr(tuned, "local_radius_px", -1.0))
            != TUNED_FCPG_RADIUS_PX
            or float(getattr(tuned, "max_residual_px", -1.0))
            != TUNED_FCPG_MAX_RESIDUAL_PX
        ):
            raise V165RuntimeContractError("FCPG radius/residual runtime contract drifted")

        records = tuple(
            (name, str(path), path_sha256[name])
            for name, path in sorted(paths.items())
        ) + (
            ("v165_vit_source", str(v165_vit_path), v165_vit_sha),
        )
        identity_payload = {
            "anchor_authority": anchor_authority,
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol_sha,
            "checkpoints": {name: digest for name, _, digest in records},
            "original_max_residual_px": float(original_max_residual_px),
            "original_corrector_state_sha256": original_state_sha,
            "fcpg_patch_projector_state_sha256": projector_state_sha,
            "vit_source_sha256": v165_vit_sha,
            "action_contract": {
                name: values.tolist() for name, values in lewm_contract.items()
            },
            "runtime_action_contract_deg": {
                "lower": runtime_lower.tolist(),
                "upper": runtime_upper.tolist(),
                "max_step": runtime_step.tolist(),
            },
            "tuned_max_residual_px": TUNED_FCPG_MAX_RESIDUAL_PX,
        }
        bundle_id = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest()
        receipt = V165BundleReceipt(
            anchor_authority=anchor_authority,
            bundle_id=bundle_id,
            protocol_id=PROTOCOL_ID,
            checkpoint_records=records,
            original_max_residual_px=float(original_max_residual_px),
            vit_source_sha256=v165_vit_sha,
            action_contract=tuple(
                (name, tuple(float(value) for value in lewm_contract[name]))
                for name in ("step_cap", "action_lower", "action_upper")
            ),
            runtime_action_contract_deg=tuple(
                (name, tuple(float(value) for value in values))
                for name, values in (
                    ("lower", runtime_lower),
                    ("upper", runtime_upper),
                    ("max_step", runtime_step),
                )
            ),
        )
        baseline = str(actor_payload.get("baseline_selection", {}).get("selected", ""))
        return cls(
            receipt=receipt,
            original_corrector=original,
            tuned_corrector=tuned,
            actor=actor,
            lewm=lewm,
            fcpg_patch_projector=patch_projector,
            actor_baseline_kind=baseline,
            action_contract=lewm_contract,
            device=device,
        )

    @torch.inference_mode()
    def route_anchors(
        self,
        *,
        feature_map: torch.Tensor,
        nominal_px: np.ndarray,
        anchors_xyz: np.ndarray,
        observations: np.ndarray,
        image_size: tuple[int, int],
    ) -> V165AnchorRouting:
        nominal = np.asarray(nominal_px, dtype=np.float32)
        xyz = np.asarray(anchors_xyz, dtype=np.float32)
        obs = np.asarray(observations, dtype=np.float32)
        if nominal.ndim != 3 or nominal.shape[1:] != (3, 2):
            raise V165RuntimeContractError("nominal_px must be [T,3,2]")
        if xyz.shape != (len(nominal), 3, 3) or obs.shape[0] != len(nominal):
            raise V165RuntimeContractError("FCPG geometry/history is misaligned")
        if not all(bool(np.isfinite(value).all()) for value in (nominal, xyz, obs)):
            raise V165RuntimeContractError("FCPG inputs must be finite")
        if feature_map.shape[0] != len(nominal):
            raise V165RuntimeContractError("shared feature map history is misaligned")
        tensors = (
            feature_map,
            torch.from_numpy(nominal).to(self._device),
            torch.from_numpy(xyz).to(self._device),
            torch.from_numpy(obs).to(self._device),
        )
        original_output = self._original_corrector(*tensors, image_size=image_size)
        tuned_output = self._tuned_corrector(*tensors, image_size=image_size)
        try:
            original = original_output.q_corrected_px.reshape(nominal.shape)
            tuned = tuned_output.q_corrected_px.reshape(nominal.shape)
        except (AttributeError, RuntimeError, ValueError) as exc:
            raise V165RuntimeContractError("corrector output shape drifted") from exc
        original_np = original.float().cpu().numpy().astype(np.float32)
        tuned_np = tuned.float().cpu().numpy().astype(np.float32)
        if not bool(np.isfinite(original_np).all()) or not bool(np.isfinite(tuned_np).all()):
            raise V165RuntimeContractError("corrector output is non-finite")
        original_residual = np.linalg.norm(original_np - nominal, axis=-1)
        tuned_residual = np.linalg.norm(tuned_np - nominal, axis=-1)
        if self.anchor_authority == "nominal":
            authoritative = nominal.copy()
            within = True
        elif self.anchor_authority == "original_fcpg":
            authoritative = original_np.copy()
            within = bool(
                np.all(original_residual <= self._receipt.original_max_residual_px)
            )
        else:
            authoritative = tuned_np.copy()
            within = bool(np.all(tuned_residual <= TUNED_FCPG_MAX_RESIDUAL_PX))
        return V165AnchorRouting(
            authority=self.anchor_authority,
            authoritative=authoritative,
            original_fcpg=original_np,
            tuned_fcpg=tuned_np,
            original_residual_px=original_residual.astype(np.float32),
            tuned_residual_px=tuned_residual.astype(np.float32),
            authority_within_envelope=within,
        )

    def prepare_round(
        self,
        *,
        full256_by_view: Mapping[str, np.ndarray],
        view_valid: Mapping[str, bool],
        current_state: np.ndarray,
        past_raw_actions: np.ndarray | None,
    ) -> None:
        if set(full256_by_view) != set(VIEWS) or set(view_valid) != set(VIEWS):
            raise V165RuntimeContractError("round tokens require wrist and column")
        scene = np.stack(
            [np.asarray(full256_by_view[view], dtype=np.float32) for view in VIEWS]
        )
        views = np.asarray([bool(view_valid[view]) for view in VIEWS], dtype=bool)
        state = np.asarray(current_state, dtype=np.float32)
        if scene.shape != (2, 256, 192) or state.shape != (6,):
            raise V165RuntimeContractError("V16.5 round tensor shapes drifted")
        if not bool(np.isfinite(scene).all()) or not bool(np.isfinite(state).all()):
            raise V165RuntimeContractError("V16.5 round tensors must be finite")
        if not bool(views.any()):
            raise V165RuntimeContractError("V16.5 needs at least one valid camera view")
        # Training cache semantics: missing/black slots carry exactly zero
        # tokens and cannot leak a learned positional/background signature.
        scene[~views] = 0.0
        canonical_state = state / np.float32(CANONICAL_TO_DEGREES_SCALE)
        if np.any(canonical_state < self._action_contract["action_lower"]) or np.any(
            canonical_state > self._action_contract["action_upper"]
        ):
            raise V165RuntimeContractError(
                "runtime joint state lies outside the V16.5 action contract"
            )
        past = np.repeat(canonical_state[None], 2, axis=0)
        if past_raw_actions is not None:
            raw = np.asarray(past_raw_actions, dtype=np.float32)
            if raw.ndim != 3 or raw.shape[1:] != (5, 6) or len(raw) > 2:
                raise V165RuntimeContractError("past_raw_actions must be [N<=2,5,6]")
            if not bool(np.isfinite(raw).all()):
                raise V165RuntimeContractError("past_raw_actions must be finite")
            endpoints = raw[:, -1] / np.float32(CANONICAL_TO_DEGREES_SCALE)
            if np.any(endpoints < self._action_contract["action_lower"]) or np.any(
                endpoints > self._action_contract["action_upper"]
            ):
                raise V165RuntimeContractError(
                    "past command endpoint lies outside the V16.5 action contract"
                )
            past[-len(endpoints) :] = endpoints
        self._round = _RoundInputs(
            full256=scene.copy(),
            view_valid=views.copy(),
            current_state=state.copy(),
            canonical_current_state=canonical_state.astype(np.float32),
            past_completed_commands=past.astype(np.float32),
        )

    def _require_round(self, current_state: np.ndarray) -> _RoundInputs:
        if self._round is None:
            raise V165RuntimeContractError("V16.5 round inputs were not prepared")
        state = np.asarray(current_state, dtype=np.float32)
        if state.shape != (6,) or not np.array_equal(state, self._round.current_state):
            raise V165RuntimeContractError("V16.5 round state changed after token capture")
        return self._round

    def action_contract_fixed_point_mask(
        self,
        *,
        current_state: np.ndarray,
        five_command_candidates: np.ndarray,
    ) -> np.ndarray:
        """Rows whose commands are unchanged by V16.5's action projector."""

        state = np.asarray(current_state, dtype=np.float32)
        candidates = np.asarray(five_command_candidates, dtype=np.float32)
        if state.shape != (6,) or candidates.ndim != 3 or candidates.shape[1:] != (5, 6):
            raise V165RuntimeContractError("V16.5 action-contract mask shape drifted")
        canonical_state = state / np.float32(CANONICAL_TO_DEGREES_SCALE)
        canonical = candidates / np.float32(CANONICAL_TO_DEGREES_SCALE)
        previous = np.concatenate(
            (
                np.broadcast_to(canonical_state, (len(canonical), 1, 6)),
                canonical[:, :-1],
            ),
            axis=1,
        )
        finite = np.isfinite(canonical).all(axis=(1, 2))
        bounded = (
            (canonical >= self._action_contract["action_lower"][None, None])
            & (canonical <= self._action_contract["action_upper"][None, None])
        ).all(axis=(1, 2))
        stepped = (
            np.abs(canonical - previous)
            <= self._action_contract["step_cap"][None, None] + np.float32(1.0e-6)
        ).all(axis=(1, 2))
        return np.asarray(finite & bounded & stepped, dtype=bool)

    @torch.inference_mode()
    def propose_actor_prefix(
        self,
        *,
        current_state: np.ndarray,
        baseline_prefixes: np.ndarray,
        candidate_ik_safe: np.ndarray,
        selected_phase_distance: np.ndarray,
        horizon_raw: int,
        phase: Any,
    ) -> dict[str, Any]:
        round_inputs = self._require_round(current_state)
        if isinstance(horizon_raw, bool) or int(horizon_raw) < 5:
            raise V165RuntimeContractError("Actor bank horizon must be at least five")
        prefixes = np.asarray(baseline_prefixes, dtype=np.float32)
        safe = np.asarray(candidate_ik_safe, dtype=bool)
        distance = np.asarray(selected_phase_distance, dtype=np.float32)
        if prefixes.ndim != 3 or prefixes.shape[1:] != (int(horizon_raw), 6):
            raise V165RuntimeContractError("baseline candidate bank shape drifted")
        if safe.shape != (len(prefixes),) or distance.shape != safe.shape:
            raise V165RuntimeContractError("baseline candidate metadata is misaligned")
        eligible = np.flatnonzero(safe & np.isfinite(distance))
        if len(eligible) == 0:
            return {"prefixes": np.empty((0, int(horizon_raw), 6), dtype=np.float32), "metadata": []}
        goal_index = int(eligible[np.argmin(distance[eligible])])
        desired_goal = prefixes[goal_index, -1] / np.float32(
            CANONICAL_TO_DEGREES_SCALE
        )
        state = torch.from_numpy(round_inputs.canonical_current_state[None]).to(
            self._device
        )
        past = torch.from_numpy(round_inputs.past_completed_commands[None]).to(self._device)
        goal = torch.from_numpy(desired_goal[None]).to(self._device)
        output = self._actor(
            current_full256=torch.from_numpy(round_inputs.full256[None]).to(self._device),
            view_valid=torch.from_numpy(round_inputs.view_valid[None]).to(self._device),
            current_state=state,
            past_completed_commands=past,
            desired_joint_goal=goal,
            baseline_kind=self._actor_baseline_kind,
            baseline_joint_goal=(goal if self._actor_baseline_kind == "analytic" else None),
        ).action
        chunk = (
            output[0].float().cpu().numpy().astype(np.float32)
            * np.float32(CANONICAL_TO_DEGREES_SCALE)
        )
        if chunk.shape != (5, 6) or not bool(np.isfinite(chunk).all()):
            raise V165RuntimeContractError("V16.5 Actor output must be finite [5,6]")
        prefix = np.repeat(chunk[-1: ], int(horizon_raw), axis=0)
        prefix[:5] = chunk
        phase_value = phase.value if hasattr(phase, "value") else str(phase)
        return {
            "prefixes": prefix[None],
            "metadata": [
                {
                    "source": V165_ACTOR_SOURCE,
                    "variant": "five_command_then_hold",
                    "actor_chunk_steps": 5,
                    "goal_baseline_index": goal_index,
                    "phase": phase_value,
                    "lewm_scored": True,
                    "bundle_id": self._receipt.bundle_id,
                }
            ],
        }

    @torch.inference_mode()
    def predict_candidate_states(
        self,
        *,
        current_state: np.ndarray,
        five_command_candidates: np.ndarray,
    ) -> np.ndarray:
        round_inputs = self._require_round(current_state)
        candidates = np.asarray(five_command_candidates, dtype=np.float32)
        if candidates.ndim != 3 or candidates.shape[1:] != (5, 6):
            raise V165RuntimeContractError("V16.5 LeWM candidates must be [N,5,6]")
        if not bool(np.isfinite(candidates).all()):
            raise V165RuntimeContractError("V16.5 LeWM candidates must be finite")
        fixed = self.action_contract_fixed_point_mask(
            current_state=current_state,
            five_command_candidates=candidates,
        )
        if not bool(fixed.all()):
            raise V165RuntimeContractError(
                "V16.5 LeWM candidate is not an action-projector fixed point"
            )
        canonical_candidates = candidates / np.float32(CANONICAL_TO_DEGREES_SCALE)
        output = self._lewm(
            current_state=torch.from_numpy(
                round_inputs.canonical_current_state[None]
            ).to(
                self._device
            ),
            past_completed_commands=torch.from_numpy(
                round_inputs.past_completed_commands[None]
            ).to(self._device),
            candidate_commands=torch.from_numpy(canonical_candidates[None]).to(
                self._device
            ),
        ).predicted_state_path
        path = (
            output[0].float().cpu().numpy().astype(np.float32)
            * np.float32(CANONICAL_TO_DEGREES_SCALE)
        )
        if path.shape != (len(candidates), 6, 6) or not bool(np.isfinite(path).all()):
            raise V165RuntimeContractError("V16.5 LeWM output must be [N,6,6]")
        if not np.allclose(path[:, 0], round_inputs.current_state, atol=1.0e-5, rtol=0.0):
            raise V165RuntimeContractError("V16.5 LeWM q0 does not match current state")
        return path[:, 1:]


def conservative_joint_sweeps(
    current_state: np.ndarray,
    prefixes: np.ndarray,
    *,
    max_joint_increment_deg: float = NO_CONTACT_SWEEP_INCREMENT_DEG,
) -> tuple[np.ndarray, ...]:
    """Joint-space samples including current FK and every segment interior."""

    state = np.asarray(current_state, dtype=np.float32)
    bank = np.asarray(prefixes, dtype=np.float32)
    if state.shape != (6,) or bank.ndim != 3 or bank.shape[-1] != 6:
        raise ValueError("no-contact sweep received invalid joint shapes")
    if (
        not np.isfinite(max_joint_increment_deg)
        or max_joint_increment_deg <= 0
        or not bool(np.isfinite(state).all())
        or not bool(np.isfinite(bank).all())
    ):
        raise ValueError("no-contact sweep contract is invalid")
    sweeps: list[np.ndarray] = []
    for prefix in bank:
        rows = [state.copy()]
        previous = state
        for endpoint in prefix:
            segments = max(
                1,
                int(
                    np.ceil(
                        float(np.max(np.abs(endpoint - previous)))
                        / float(max_joint_increment_deg)
                    )
                ),
            )
            rows.extend(
                np.linspace(
                    previous,
                    endpoint,
                    segments + 1,
                    dtype=np.float32,
                )[1:]
            )
            previous = endpoint
        sweeps.append(np.stack(rows).astype(np.float32))
    return tuple(sweeps)


def v165_command_reference_states(
    prefixes: np.ndarray,
    *,
    predicted_steps: int,
) -> np.ndarray:
    """Return the raw q1..qK commands matching V16.5's per-command rollout."""

    bank = np.asarray(prefixes, dtype=np.float32)
    steps = int(predicted_steps)
    if (
        bank.ndim != 3
        or bank.shape[-1] != 6
        or steps <= 0
        or steps > 5
        or bank.shape[1] < steps
        or not bool(np.isfinite(bank).all())
    ):
        raise ValueError("V16.5 command reference must be finite [N,H>=K,6], K<=5")
    return bank[:, :steps].copy()


def no_contact_approach_mask(
    *,
    phase: Any,
    current_state: np.ndarray,
    prefixes: np.ndarray,
    swept_xyz: tuple[np.ndarray, ...] | list[np.ndarray],
    contact_height_m: float,
    clearance_m: float = NO_CONTACT_CLEARANCE_M,
) -> tuple[np.ndarray, tuple[tuple[str, ...], ...]]:
    """Pure planner-side hard gate for the one-chunk no-contact approach."""

    phase_value = phase.value if hasattr(phase, "value") else str(phase)
    state = np.asarray(current_state, dtype=np.float32)
    bank = np.asarray(prefixes, dtype=np.float32)
    if state.shape != (6,) or bank.ndim != 3 or bank.shape[-1] != 6:
        raise ValueError("no-contact gate received invalid command shapes")
    if len(swept_xyz) != len(bank):
        raise ValueError("no-contact swept anchor paths must align with candidates")
    if (
        not np.isfinite(contact_height_m)
        or not np.isfinite(clearance_m)
        or clearance_m <= 0
    ):
        raise ValueError("no-contact height contract is invalid")
    result = np.ones(len(bank), dtype=bool)
    reasons: list[tuple[str, ...]] = []
    for index, (prefix, raw_path) in enumerate(zip(bank, swept_xyz, strict=True)):
        path = np.asarray(raw_path, dtype=np.float32)
        if (
            path.ndim != 3
            or path.shape[1:] != (3, 3)
            or len(path) < len(prefix) + 1
            or not bool(np.isfinite(path).all())
        ):
            raise ValueError(
                "no-contact swept anchor path must include current FK and command endpoints"
            )
        row: list[str] = []
        if phase_value != "approach":
            row.append("v165_no_contact_phase")
        jaw_path = np.concatenate((state[5:6], prefix[:, 5]))
        if np.any(np.diff(jaw_path) < -NO_CONTACT_JAW_TOLERANCE_DEG):
            row.append("v165_no_contact_jaw_close")
        if np.any(path[..., 2] < float(contact_height_m) + float(clearance_m)):
            row.append("v165_no_contact_anchor_clearance")
        reasons.append(tuple(row))
        result[index] = not row
    return result, tuple(reasons)


def enforce_no_contact_selected_chunk(
    *,
    phase: Any,
    current_state: np.ndarray,
    selected_chunk: np.ndarray,
    swept_xyz: np.ndarray,
    contact_height_m: float,
) -> None:
    safe, reasons = no_contact_approach_mask(
        phase=phase,
        current_state=current_state,
        prefixes=np.asarray(selected_chunk, dtype=np.float32)[None],
        swept_xyz=(np.asarray(swept_xyz, dtype=np.float32),),
        contact_height_m=contact_height_m,
    )
    if not bool(safe[0]):
        raise V165RuntimeContractError(
            "no-contact approach rejected selected chunk: " + ", ".join(reasons[0])
        )


__all__ = [
    "ANCHOR_AUTHORITIES",
    "CONTACT_AUTHORIZATION_DECISION",
    "CONTACT_AUTHORIZATION_FORMAT",
    "CONTACT_AUTHORIZATION_GATES",
    "NO_CONTACT_CLEARANCE_M",
    "TUNED_FCPG_MAX_RESIDUAL_PX",
    "TUNED_FCPG_RADIUS_PX",
    "V165_ACTOR_SOURCE",
    "V165AnchorRouting",
    "V165BundleReceipt",
    "V165RuntimeAdapter",
    "V165RuntimeContractError",
    "conservative_joint_sweeps",
    "enforce_no_contact_selected_chunk",
    "no_contact_approach_mask",
    "validate_contact_authorization_receipt",
    "v165_command_reference_states",
    "v165_rgb_view_nonblack",
]
