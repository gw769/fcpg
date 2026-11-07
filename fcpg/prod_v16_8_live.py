"""Receipt-bound V16.8 Actor overlay for the existing production planner.

The overlay owns only one proposal.  Existing FCPG supplies the current visual
tokens and the planner supplies one joint-space proxy goal; the frozen V16.8
Actor emits a five-command chunk which is held to the planner horizon.  In
``candidate`` mode that one prefix is appended to the common bank and the
already-loaded frozen State/FK LeWM ranks it with every other candidate.  In
``shadow`` mode the same Actor and LeWM forwards run, but the prefix is never
inserted into the decision bank.

Only chunks confirmed as sent by the common runtime are committed.  Four
completed chunks form one fixed-goal H0--H3 window: H0 has no history, H1--H3
carry one--three chronological ``[5,6]`` chunks, and the fourth completion
atomically resets the next window to H0.  This module has no robot, camera,
serial, CEM, candidate-selection, or watchdog implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from fcpg.prod_v16_5 import (
    COMMAND_STEPS,
    LEWM_CHECKPOINT_FORMAT,
    PATCHES,
    STATE_DIM,
    TOKEN_DIM,
    VIEWS,
    StateFKLeWM,
)
from fcpg import prod_v16_5_runtime as _v165
from fcpg.prod_v16_8 import FullHistoryFiveCommandActor
from fcpg.prod_v16_8_runtime import (
    V168OfflineRuntimeContractError,
    load_registered_pass8_actor,
)


PROTOCOL_ID = "so101-prod-v16-8-live-actor-overlay-v1"
FOUNDATION_FORMAT = "so101-prod-v16-8-original-fcpg-lewm-foundation-v1"
V165_PROTOCOL_ID = "so101-prod-v16-5-fk-seeded-e2e-v1"
REGISTERED_ORIGINAL_FCPG_SHA256 = (
    "38dfc187f0b9c35b195692005df32f518fcb4685aaa3c880a8a7ee32facafae5"
)
REGISTERED_CONTEXT_BRIDGE_SHA256 = (
    "89faf2583ba4d66260f4a8b497f5a18fba51680c921d426cdf2d2a9d2f845846"
)
REGISTERED_V165_SOURCE_SHA256 = (
    "1298b2579e317af3b628f02022abb640dec2bcb6bbe6559e28457ed0717e25c8"
)
REGISTERED_LEWM_SHA256 = (
    "28a1e331ecf3ff35ea994164be242b202828f7751ae3dfddee9768e5569769bd"
)
REGISTERED_LEWM_STATE_SHA256 = (
    "923cf7a8ca0151c1b0d1bc56a968271a093c7d32b2b19666382cd4707b2e9426"
)
REGISTERED_LEWM_RECEIPT_SHA256 = (
    "38d42a0df0c1cca93fbdae7fdb2b2989acb567eee1a9f5b7fde5c301cead2caa"
)
REGISTERED_VIT_SOURCE_SHA256 = (
    "fcd004d1a01cfa3c376cc060dfa738defb68385a19f71e16bdde0f207b6a59b7"
)
HISTORY_CHUNKS = 3
WINDOW_CHUNKS = 4
WINDOW_COMMANDS = WINDOW_CHUNKS * COMMAND_STEPS
PROXY_GOAL_COMMAND_INDEX = WINDOW_COMMANDS - 1
CANONICAL_TO_DEGREES_SCALE = 100.0
ACTOR_SOURCE = "v16_8_full_main_pass8_actor"
MODES = ("shadow", "candidate")
Mode = Literal["shadow", "candidate"]

ACTION_CONTRACT = {
    "step_cap": [0.04, 0.04, 0.04, 0.04, 0.04, 0.08],
    "lower": [-1.0, -1.0, -1.0, -1.0, -1.0, 0.0],
    "upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}
class V168LiveContractError(RuntimeError):
    """The frozen artifact or live causal-window contract drifted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V168LiveContractError(f"cannot read {label}") from exc
    if not isinstance(value, Mapping):
        raise V168LiveContractError(f"{label} must contain an object")
    return value


def _validate_foundation_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the overlay to original FCPG plus the formally-GO State/FK LeWM."""

    receipt = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if (
        receipt.get("format") != FOUNDATION_FORMAT
        or receipt.get("protocol_id") != V165_PROTOCOL_ID
        or not isinstance(receipt.get("bundle_id"), str)
        or not receipt["bundle_id"]
        or receipt.get("anchor_authority") != "original_fcpg"
        or receipt.get("tuned_fcpg_present") is not False
        or receipt.get("legacy_actor_present") is not False
    ):
        raise V168LiveContractError("V16.8 foundation receipt identity drifted")
    records = receipt.get("checkpoint_records")
    required = {
        "context_bridge_carrier",
        "original_fcpg",
        "v165_source_checkpoint",
        "lewm",
        "lewm_receipt",
        "v165_vit_source",
    }
    if not isinstance(records, Mapping) or set(records) != required:
        raise V168LiveContractError("V16.8 foundation checkpoint receipt is incomplete")
    for name in required:
        row = records[name]
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
        ):
            raise V168LiveContractError(
                f"V16.8 foundation checkpoint identity is invalid: {name}"
            )
    runtime = receipt.get("runtime_state_contract")
    if not isinstance(runtime, Mapping) or (
        runtime.get("model_units") != "canonical_normalized_joint_command"
        or runtime.get("robot_units") != "lerobot_calibration_degrees"
        or float(runtime.get("canonical_to_degrees_scale", float("nan")))
        != CANONICAL_TO_DEGREES_SCALE
    ):
        raise V168LiveContractError("V16.8 foundation unit contract drifted")
    corrector = receipt.get("corrector_contract")
    if not isinstance(corrector, Mapping) or (
        corrector.get("fcpg_uses_registered_v12_patch_projector") is not True
        or corrector.get("shared_raw_vit_forward") is not True
        or corrector.get("authority") != "original_fcpg"
        or corrector.get("has_physical_labels") is not False
    ):
        raise V168LiveContractError("V16.8 original FCPG receipt drifted")
    if receipt.get("lewm_gate_pass") is not True:
        raise V168LiveContractError("V16.8 foundation LeWM is not formally GO")
    return receipt


@dataclass(frozen=True)
class _PreparedRound:
    full256: np.ndarray
    view_valid: np.ndarray
    current_state_deg: np.ndarray
    current_state: np.ndarray


@dataclass(frozen=True)
class V168FoundationReceipt:
    """Immutable original-FCPG + formally-GO State/FK-LeWM identity."""

    anchor_authority: str
    bundle_id: str
    protocol_id: str
    checkpoint_records: tuple[tuple[str, str, str], ...]
    original_max_residual_px: float
    vit_source_sha256: str
    action_contract: tuple[tuple[str, tuple[float, ...]], ...]
    runtime_action_contract_deg: tuple[tuple[str, tuple[float, ...]], ...]
    contact_authorization: tuple[tuple[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": FOUNDATION_FORMAT,
            "anchor_authority": self.anchor_authority,
            "bundle_id": self.bundle_id,
            "protocol_id": self.protocol_id,
            "checkpoint_records": {
                name: {"path": path, "sha256": digest}
                for name, path, digest in self.checkpoint_records
            },
            "tuned_fcpg_present": False,
            "legacy_actor_present": False,
            "lewm_gate_pass": True,
            "corrector_contract": {
                "authority": "original_fcpg",
                "has_physical_labels": False,
                "training_scope": "synthetic_degradation",
                "original_local_radius_px": _v165.ORIGINAL_FCPG_RADIUS_PX,
                "original_model_max_residual_px": (
                    _v165.ORIGINAL_FCPG_MODEL_MAX_RESIDUAL_PX
                ),
                "planner_residual_envelope_px": self.original_max_residual_px,
                "vit_source_sha256": self.vit_source_sha256,
                "shared_raw_vit_forward": True,
                "fcpg_uses_registered_v12_patch_projector": True,
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
            "deployment_evidence_boundary": {
                "original_fcpg_has_physical_labels": False,
                "original_fcpg_formal_physical_receipt_present": False,
                "contact_hardware_authorized": False,
                "shadow_only_until_separate_integration_review": True,
            },
        }


@dataclass(frozen=True)
class _OriginalAnchorRouting:
    authoritative: np.ndarray
    original_fcpg: np.ndarray
    original_residual_px: np.ndarray
    authority_within_envelope: bool

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "anchor_authority": "original_fcpg",
            "original_fcpg_px": self.original_fcpg,
            "original_residual_px": self.original_residual_px,
            "authority_within_envelope": self.authority_within_envelope,
            "tuned_fcpg_present": False,
            "legacy_actor_present": False,
            "shared_raw_vit_forward": True,
            "fcpg_feature_map": "registered_v12_patch_projector_of_shared_raw_tokens",
        }


class V168FoundationRuntimeAdapter(_v165.V165RuntimeAdapter):
    """V16.5-compatible foundation without any tuned FCPG or legacy Actor.

    Only methods used by the common planner are inherited: raw-token patch
    projection, State/FK LeWM round preparation, action fixed-point checks, and
    candidate prediction.  The Actor proposal method is explicitly refused.
    """

    @classmethod
    def from_registered_components(
        cls,
        *,
        context_bridge_checkpoint: Path,
        original_fcpg_checkpoint: Path,
        v165_source_checkpoint: Path,
        lewm_checkpoint: Path,
        lewm_receipt: Path,
        grounding_model: Any,
        runtime_action_lower_deg: np.ndarray,
        runtime_action_upper_deg: np.ndarray,
        runtime_max_step_deg: np.ndarray,
        original_max_residual_px: float,
        device: torch.device,
    ) -> "V168FoundationRuntimeAdapter":
        paths = {
            "context_bridge_carrier": _v165._resolved_file(
                context_bridge_checkpoint, label="context bridge checkpoint"
            ),
            "original_fcpg": _v165._resolved_file(
                original_fcpg_checkpoint, label="registered original FCPG checkpoint"
            ),
            "v165_source_checkpoint": _v165._resolved_file(
                v165_source_checkpoint, label="registered V16.5 V12 source checkpoint"
            ),
            "lewm": _v165._resolved_file(
                lewm_checkpoint, label="registered V16.5 State/FK LeWM checkpoint"
            ),
            "lewm_receipt": _v165._resolved_file(
                lewm_receipt, label="registered V16.5 State/FK LeWM receipt"
            ),
        }
        digests = {name: _sha256_file(path) for name, path in paths.items()}
        expected = {
            "context_bridge_carrier": REGISTERED_CONTEXT_BRIDGE_SHA256,
            "original_fcpg": REGISTERED_ORIGINAL_FCPG_SHA256,
            "v165_source_checkpoint": REGISTERED_V165_SOURCE_SHA256,
            "lewm": REGISTERED_LEWM_SHA256,
            "lewm_receipt": REGISTERED_LEWM_RECEIPT_SHA256,
        }
        if any(digests[name] != digest for name, digest in expected.items()):
            raise V168LiveContractError(
                "V16.8 foundation received an unregistered FCPG/LeWM artifact"
            )
        receipt_payload = _read_json(paths["lewm_receipt"], label="State/FK LeWM receipt")
        receipt_checkpoint = receipt_payload.get("checkpoint")
        exposures = receipt_payload.get("exposures")
        if (
            receipt_payload.get("format") != "so101-prod-v16-5-stage-receipt-v1"
            or receipt_payload.get("protocol_id") != V165_PROTOCOL_ID
            or receipt_payload.get("stage") != "lewm"
            or receipt_payload.get("decision") != "GO"
            or receipt_payload.get("deployment_authorized") is not False
            or not isinstance(receipt_checkpoint, Mapping)
            or receipt_checkpoint.get("sha256") != REGISTERED_LEWM_SHA256
            or receipt_checkpoint.get("state_sha256") != REGISTERED_LEWM_STATE_SHA256
            or not isinstance(exposures, Mapping)
            or int(exposures.get("task_test_reads", -1)) != 0
            or int(exposures.get("hardware_actions", -1)) != 0
        ):
            raise V168LiveContractError("State/FK LeWM GO receipt drifted")
        original_payload = _v165._load_checkpoint(
            paths["original_fcpg"], label="registered original FCPG checkpoint"
        )
        source_payload = _v165._load_checkpoint(
            paths["v165_source_checkpoint"], label="registered V16.5 V12 source"
        )
        lewm_payload = _v165._load_checkpoint(
            paths["lewm"], label="registered V16.5 State/FK LeWM"
        )
        if (
            lewm_payload.get("format") != _v165.CHECKPOINT_FORMAT
            or lewm_payload.get("stage") != "lewm"
            or lewm_payload.get("protocol_id") != V165_PROTOCOL_ID
            or lewm_payload.get("model_format") != LEWM_CHECKPOINT_FORMAT
            or lewm_payload.get("deployment_authorized") is not False
            or lewm_payload.get("state_sha256") != REGISTERED_LEWM_STATE_SHA256
            or lewm_payload.get("metrics", {}).get("gate_pass") is not True
        ):
            raise V168LiveContractError("registered State/FK LeWM checkpoint drifted")
        original_state = original_payload.get("corrector")
        if not isinstance(original_state, Mapping) or len(original_state) != _v165.FCPG_CORRECTOR_TENSOR_COUNT:
            raise V168LiveContractError("registered original FCPG has no corrector")
        original_state_sha = _v165._state_dict_sha256(
            original_state, label="registered original FCPG"
        )
        if (
            original_payload.get("source_family") != "v12"
            or original_payload.get("corrector_variant") != "independent3"
            or original_payload.get("source_checkpoint_sha256")
            != REGISTERED_V165_SOURCE_SHA256
            or original_payload.get("corrector_state_sha256_after")
            != original_state_sha
            or original_payload.get("has_physical_labels") is not False
            or original_payload.get("training_scope") != "synthetic_degradation"
        ):
            raise V168LiveContractError("registered original FCPG contract drifted")
        source_model_state = source_payload.get("model")
        if not isinstance(source_model_state, Mapping):
            raise V168LiveContractError("registered V16.5 source has no model state")
        source_encoder_state = {
            str(name).removeprefix("vit.encoder."): tensor
            for name, tensor in source_model_state.items()
            if str(name).startswith("vit.encoder.")
        }
        carrier_encoder = getattr(
            getattr(grounding_model, "base", None), "encoder", None
        )
        if not isinstance(carrier_encoder, torch.nn.Module):
            raise V168LiveContractError("context carrier has no raw ViT encoder")
        carrier_state = carrier_encoder.state_dict()
        if (
            len(source_encoder_state) != _v165.RAW_VIT_ENCODER_TENSOR_COUNT
            or set(source_encoder_state) != set(carrier_state)
            or any(
                not torch.equal(
                    source_encoder_state[name].detach().cpu(),
                    carrier_state[name].detach().cpu(),
                )
                for name in source_encoder_state
            )
        ):
            raise V168LiveContractError("context carrier and source ViT differ")
        vit_path, vit_sha = _v165._checkpoint_vit_source(
            source_payload, label="registered V16.5 source"
        )
        if (
            vit_sha != REGISTERED_VIT_SOURCE_SHA256
            or original_payload.get("vit_source_sha256") != vit_sha
        ):
            raise V168LiveContractError("registered FCPG ViT source drifted")
        lewm_contract = _v165._checkpoint_action_contract(
            lewm_payload, label="State/FK LeWM"
        )
        expected_contract = {
            "step_cap": np.asarray(ACTION_CONTRACT["step_cap"], dtype=np.float32),
            "action_lower": np.asarray(ACTION_CONTRACT["lower"], dtype=np.float32),
            "action_upper": np.asarray(ACTION_CONTRACT["upper"], dtype=np.float32),
        }
        if not _v165._same_action_contract(lewm_contract, expected_contract):
            raise V168LiveContractError("V16.8 Actor and State/FK LeWM contracts differ")
        runtime_lower = np.asarray(runtime_action_lower_deg, dtype=np.float32)
        runtime_upper = np.asarray(runtime_action_upper_deg, dtype=np.float32)
        runtime_step = np.asarray(runtime_max_step_deg, dtype=np.float32)
        if any(value.shape != (STATE_DIM,) for value in (runtime_lower, runtime_upper, runtime_step)):
            raise V168LiveContractError("runtime action vectors must be [6]")
        model_lower = lewm_contract["action_lower"] * np.float32(CANONICAL_TO_DEGREES_SCALE)
        model_upper = lewm_contract["action_upper"] * np.float32(CANONICAL_TO_DEGREES_SCALE)
        model_step = lewm_contract["step_cap"] * np.float32(CANONICAL_TO_DEGREES_SCALE)
        if (
            not all(bool(np.isfinite(value).all()) for value in (runtime_lower, runtime_upper, runtime_step))
            or np.any(model_lower > runtime_lower)
            or np.any(model_upper < runtime_upper)
            or np.any(model_step + np.float32(1.0e-6) < runtime_step)
        ):
            raise V168LiveContractError(
                "runtime safety permits states/steps outside the V16.8 foundation"
            )
        tensor_contract = {
            name: torch.from_numpy(value).to(device)
            for name, value in lewm_contract.items()
        }
        kwargs = lewm_payload.get("model_kwargs")
        try:
            if not isinstance(kwargs, Mapping):
                raise TypeError("missing LeWM model_kwargs")
            lewm = StateFKLeWM(
                **tensor_contract,
                residual_cap=torch.as_tensor(
                    kwargs["residual_cap"], device=device, dtype=torch.float32
                ),
                dimension=int(kwargs["dimension"]),
                layers=int(kwargs["layers"]),
            ).to(device)
            lewm.load_state_dict(lewm_payload["state_dict"], strict=True)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise V168LiveContractError("State/FK LeWM does not strict-load") from exc
        _v165._assert_module_action_contract(
            lewm, lewm_contract, label="V16.8 foundation LeWM"
        )
        try:
            from v10.model import _BoundedResidualProjector

            projector = _BoundedResidualProjector(TOKEN_DIM).to(device)
            projector_state = {
                str(name).removeprefix("patch_projector."): tensor
                for name, tensor in source_model_state.items()
                if str(name).startswith("patch_projector.")
            }
            projector.load_state_dict(projector_state, strict=True)
        except (ImportError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise V168LiveContractError("V12 FCPG patch projector does not strict-load") from exc
        original = _v165.build_so101_fcpg_corrector(
            grounding_model, camera_context_dim=5, variant="independent3"
        ).to(device)
        original.local_radius_px = _v165.ORIGINAL_FCPG_RADIUS_PX
        original.max_residual_px = _v165.ORIGINAL_FCPG_MODEL_MAX_RESIDUAL_PX
        try:
            original.load_state_dict(original_state, strict=True)
        except (KeyError, RuntimeError) as exc:
            raise V168LiveContractError("original FCPG does not strict-load") from exc
        records = tuple(
            (name, str(path), digests[name]) for name, path in sorted(paths.items())
        ) + (("v165_vit_source", str(vit_path), vit_sha),)
        identity = {
            "protocol_id": V165_PROTOCOL_ID,
            "anchor_authority": "original_fcpg",
            "records": {name: digest for name, _path, digest in records},
            "original_state_sha256": original_state_sha,
            "action_contract": {
                name: value.tolist() for name, value in lewm_contract.items()
            },
            "runtime_step_deg": runtime_step.tolist(),
        }
        bundle_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        receipt = V168FoundationReceipt(
            anchor_authority="original_fcpg",
            bundle_id=bundle_id,
            protocol_id=V165_PROTOCOL_ID,
            checkpoint_records=records,
            original_max_residual_px=float(original_max_residual_px),
            vit_source_sha256=vit_sha,
            action_contract=tuple(
                (name, tuple(float(item) for item in lewm_contract[name]))
                for name in ("step_cap", "action_lower", "action_upper")
            ),
            runtime_action_contract_deg=tuple(
                (name, tuple(float(item) for item in value))
                for name, value in (
                    ("lower", runtime_lower),
                    ("upper", runtime_upper),
                    ("max_step", runtime_step),
                )
            ),
        )
        result = cls.__new__(cls)
        result._receipt = receipt
        result._device = torch.device(device)
        result._original_corrector = _v165._freeze_module(original)
        result._tuned_corrector = None
        result._actor = None
        result._lewm = _v165._freeze_module(lewm)
        result._fcpg_patch_projector = _v165._freeze_module(projector)
        result._actor_baseline_kind = "absent"
        result._action_contract = {
            name: np.asarray(value, dtype=np.float32).copy()
            for name, value in lewm_contract.items()
        }
        result._round = None
        return result

    @torch.inference_mode()
    def route_anchors(
        self,
        *,
        feature_map: torch.Tensor,
        nominal_px: np.ndarray,
        anchors_xyz: np.ndarray,
        observations: np.ndarray,
        image_size: tuple[int, int],
    ) -> _OriginalAnchorRouting:
        nominal = np.asarray(nominal_px, dtype=np.float32)
        xyz = np.asarray(anchors_xyz, dtype=np.float32)
        obs = np.asarray(observations, dtype=np.float32)
        if (
            nominal.ndim != 3
            or nominal.shape[1:] != (3, 2)
            or xyz.shape != (len(nominal), 3, 3)
            or obs.shape[0] != len(nominal)
            or feature_map.shape[0] != len(nominal)
            or not all(bool(np.isfinite(value).all()) for value in (nominal, xyz, obs))
        ):
            raise V168LiveContractError("original FCPG round inputs are misaligned")
        output = self._original_corrector(
            feature_map,
            torch.from_numpy(nominal).to(self._device),
            torch.from_numpy(xyz).to(self._device),
            torch.from_numpy(obs).to(self._device),
            image_size=image_size,
        )
        try:
            original = output.q_corrected_px.reshape(nominal.shape)
        except (AttributeError, RuntimeError, ValueError) as exc:
            raise V168LiveContractError("original FCPG output shape drifted") from exc
        points = original.float().cpu().numpy().astype(np.float32)
        if not bool(np.isfinite(points).all()):
            raise V168LiveContractError("original FCPG output is non-finite")
        residual = np.linalg.norm(points - nominal, axis=-1).astype(np.float32)
        return _OriginalAnchorRouting(
            authoritative=points.copy(),
            original_fcpg=points,
            original_residual_px=residual,
            authority_within_envelope=bool(
                np.all(residual <= self._receipt.original_max_residual_px)
            ),
        )

    def propose_actor_prefix(self, **_kwargs: Any) -> dict[str, Any]:
        raise V168LiveContractError(
            "legacy V16.5 Actor is absent; V16.8 owns the sole learned proposal"
        )


class V168LiveActorOverlay:
    """One frozen V16.8 proposal source layered over the frozen foundation."""

    def __init__(
        self,
        *,
        actor: FullHistoryFiveCommandActor,
        actor_binding: Mapping[str, Any],
        foundation_receipt: Mapping[str, Any],
        mode: Mode,
        device: torch.device,
    ) -> None:
        if mode not in MODES:
            raise V168LiveContractError("V16.8 live mode must be shadow or candidate")
        self._actor = actor
        self._actor_binding = json.loads(json.dumps(dict(actor_binding)))
        self._foundation_receipt = _validate_foundation_receipt(
            foundation_receipt
        )
        self._mode: Mode = mode
        self._device = torch.device(device)
        self._round: _PreparedRound | None = None
        self._goal: np.ndarray | None = None
        self._goal_source: dict[str, Any] | None = None
        self._history = np.zeros(
            (HISTORY_CHUNKS, COMMAND_STEPS, STATE_DIM), dtype=np.float32
        )
        self._transition = 0
        self._window_index = 0
        self._completed_total = 0
        self._last_proposal: dict[str, Any] | None = None
        self._last_shadow_lewm: dict[str, Any] | None = None
        self._last_commit: dict[str, Any] | None = None
        self._last_reset: dict[str, Any] | None = None

    @classmethod
    def from_registered_pass8(
        cls,
        *,
        receipt_path: str | Path,
        checkpoint_path: str | Path,
        foundation_receipt: Mapping[str, Any],
        mode: Mode,
        device: str | torch.device,
    ) -> "V168LiveActorOverlay":
        resolved = torch.device(device)
        if resolved.type not in {"cpu", "cuda"}:
            raise V168LiveContractError("V16.8 device must be cpu or cuda")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise V168LiveContractError("requested V16.8 CUDA device is unavailable")
        # One public content-addressed loader is shared by offline and live
        # entrypoints.  A reviewed byte-identical checkpoint may be relocated;
        # receipt, container, and state identities remain exact.
        try:
            actor, offline_binding = load_registered_pass8_actor(
                receipt_path, checkpoint_path, device=resolved
            )
        except V168OfflineRuntimeContractError as exc:
            raise V168LiveContractError(
                "cannot load the registered V16.8 pass8 Actor"
            ) from exc
        binding = {
            "receipt": {
                "path": offline_binding.receipt_path,
                "sha256": offline_binding.receipt_sha256,
            },
            "checkpoint": {
                "path": offline_binding.checkpoint_path,
                "sha256": offline_binding.checkpoint_sha256,
                "receipt_recorded_path": offline_binding.training_milestone_path,
                "state_sha256": offline_binding.state_sha256,
                "completed_pass": offline_binding.completed_pass,
            },
            "parameters": offline_binding.parameters,
        }
        return cls(
            actor=actor,
            actor_binding=binding,
            foundation_receipt=foundation_receipt,
            mode=mode,
            device=resolved,
        )

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def decision_influence(self) -> bool:
        return self._mode == "candidate"

    @property
    def transition(self) -> int:
        return int(self._transition)

    @property
    def receipt(self) -> dict[str, Any]:
        records = self._foundation_receipt["checkpoint_records"]
        return {
            "format": PROTOCOL_ID,
            "mode": self._mode,
            "actor": self._actor_binding,
            "foundation": {
                "bundle_id": self._foundation_receipt["bundle_id"],
                "protocol_id": self._foundation_receipt["protocol_id"],
                "anchor_authority": self._foundation_receipt.get("anchor_authority"),
                "active_fcpg_sha256": records["original_fcpg"]["sha256"],
                "active_lewm_sha256": records["lewm"]["sha256"],
                "lewm_receipt_sha256": records["lewm_receipt"]["sha256"],
                "tuned_fcpg_present": False,
                "legacy_v165_actor_present": False,
                "legacy_v165_actor_candidate_enabled": False,
            },
            "actor_source": ACTOR_SOURCE,
            "candidate_contract": "one_five_command_then_hold_prefix",
            "ranking_authority": "frozen_GO_v16_5_state_fk_lewm_common_ranker",
            "cem_present": False,
            "direct_selection_present": False,
            "performance_gate_present": False,
            "history": "exact_completed_sent_chunks_3x5x6_H0_H3",
            "goal": "planner_proxy_fixed_for_four_chunk_window",
            "training_h4_equivalence_verified": False,
            "model_units": "canonical_normalized_joint_command",
            "robot_units": "lerobot_calibration_degrees",
            "canonical_to_degrees_scale": CANONICAL_TO_DEGREES_SCALE,
            "shadow_commands_sent": 0 if self._mode == "shadow" else None,
        }

    def status(self) -> dict[str, Any]:
        valid = np.arange(HISTORY_CHUNKS) < self._transition
        invalid = ~valid[:, None, None]
        return {
            **self.receipt,
            "window_index": self._window_index,
            "transition": f"H{self._transition}",
            "history_valid": valid.tolist(),
            "future_history_exact_zero": bool(
                np.all(self._history[np.broadcast_to(invalid, self._history.shape)] == 0)
            ),
            "fixed_goal_present": self._goal is not None,
            "completed_chunks_total": self._completed_total,
            "last_proposal": self._last_proposal,
            "last_shadow_lewm": self._last_shadow_lewm,
            "last_commit": self._last_commit,
            "last_reset": self._last_reset,
        }

    def reset_window(self, reason: str) -> dict[str, Any]:
        """Atomically discard history and the fixed goal at a session boundary."""

        if not isinstance(reason, str) or not reason.strip():
            raise V168LiveContractError("V16.8 reset reason must be non-empty")
        before = {
            "window_index": self._window_index,
            "transition": f"H{self._transition}",
            "fixed_goal_present": self._goal is not None,
        }
        self._transition = 0
        self._history.fill(0.0)
        self._goal = None
        self._goal_source = None
        self._round = None
        self._last_proposal = None
        self._last_shadow_lewm = None
        self._window_index += 1
        self._last_reset = {
            "reason": reason.strip(),
            "before": before,
            "after": {
                "window_index": self._window_index,
                "transition": "H0",
                "fixed_goal_present": False,
            },
        }
        return dict(self._last_reset)

    def prepare_round(
        self,
        *,
        full256_by_view: Mapping[str, np.ndarray],
        view_valid: Mapping[str, bool],
        current_state_deg: np.ndarray,
    ) -> None:
        if set(full256_by_view) != {"wrist", "column"} or set(view_valid) != {
            "wrist",
            "column",
        }:
            raise V168LiveContractError("V16.8 round requires wrist and column")
        scene = np.stack(
            [np.asarray(full256_by_view[view], dtype=np.float32) for view in ("wrist", "column")]
        )
        views = np.asarray(
            [bool(view_valid[view]) for view in ("wrist", "column")], dtype=bool
        )
        state_deg = np.asarray(current_state_deg, dtype=np.float32)
        if scene.shape != (VIEWS, PATCHES, TOKEN_DIM) or state_deg.shape != (STATE_DIM,):
            raise V168LiveContractError("V16.8 live round tensor shapes drifted")
        if not bool(np.isfinite(scene).all()) or not bool(np.isfinite(state_deg).all()):
            raise V168LiveContractError("V16.8 live round tensors must be finite")
        if not bool(views.any()):
            raise V168LiveContractError("V16.8 needs at least one valid RGB view")
        scene[~views] = 0.0
        state = state_deg / np.float32(CANONICAL_TO_DEGREES_SCALE)
        lower = np.asarray(ACTION_CONTRACT["lower"], dtype=np.float32)
        upper = np.asarray(ACTION_CONTRACT["upper"], dtype=np.float32)
        if bool(((state < lower) | (state > upper)).any()):
            raise V168LiveContractError("current robot state leaves V16.8 bounds")
        self._round = _PreparedRound(
            full256=scene.copy(),
            view_valid=views.copy(),
            current_state_deg=state_deg.copy(),
            current_state=state.astype(np.float32),
        )

    def _condensed_past(self, current: np.ndarray) -> np.ndarray:
        if self._transition == 0:
            return np.repeat(current[None], 2, axis=0).astype(np.float32)
        endpoints = self._history[: self._transition, -1]
        if self._transition == 1:
            return np.stack((current, endpoints[-1])).astype(np.float32)
        return endpoints[-2:].astype(np.float32, copy=True)

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
        if self._round is None:
            raise V168LiveContractError("V16.8 round tokens were not prepared")
        state_deg = np.asarray(current_state, dtype=np.float32)
        if not np.array_equal(state_deg, self._round.current_state_deg):
            raise V168LiveContractError("V16.8 round state changed after token capture")
        prefixes = np.asarray(baseline_prefixes, dtype=np.float32)
        safe = np.asarray(candidate_ik_safe, dtype=bool)
        distance = np.asarray(selected_phase_distance, dtype=np.float32)
        if (
            isinstance(horizon_raw, bool)
            or int(horizon_raw) < WINDOW_COMMANDS
            or prefixes.ndim != 3
            or prefixes.shape[1:] != (int(horizon_raw), STATE_DIM)
            or safe.shape != (len(prefixes),)
            or distance.shape != safe.shape
        ):
            raise V168LiveContractError("V16.8 planner candidate bank is misaligned")
        eligible = np.flatnonzero(safe & np.isfinite(distance))
        if len(eligible) == 0:
            self._last_proposal = {
                "generated": False,
                "reason": "no_safe_planner_proxy_goal",
                "transition": f"H{self._transition}",
            }
            return {
                "prefixes": np.empty((0, int(horizon_raw), STATE_DIM), dtype=np.float32),
                "metadata": [],
            }
        if self._goal is None:
            goal_index = int(eligible[np.argmin(distance[eligible])])
            # Training's desired H4 endpoint is the measured state after four
            # five-command chunks.  The live proxy therefore reads command 19,
            # never the end of a longer (normally 40-step) planner horizon.
            goal = prefixes[
                goal_index, PROXY_GOAL_COMMAND_INDEX
            ] / np.float32(CANONICAL_TO_DEGREES_SCALE)
            lower = np.asarray(ACTION_CONTRACT["lower"], dtype=np.float32)
            upper = np.asarray(ACTION_CONTRACT["upper"], dtype=np.float32)
            if not bool(np.isfinite(goal).all()) or bool(((goal < lower) | (goal > upper)).any()):
                raise V168LiveContractError("planner-proxy goal leaves V16.8 bounds")
            self._goal = goal.astype(np.float32, copy=True)
            self._goal_source = {
                "baseline_index_at_h0": goal_index,
                "phase_at_h0": phase.value if hasattr(phase, "value") else str(phase),
                "proxy_command_index": PROXY_GOAL_COMMAND_INDEX,
                "window_chunks": WINDOW_CHUNKS,
                "window_commands": WINDOW_COMMANDS,
                "planner_horizon_raw": int(horizon_raw),
            }
        valid = np.arange(HISTORY_CHUNKS) < self._transition
        invalid = ~valid[:, None, None]
        if not bool(
            np.all(self._history[np.broadcast_to(invalid, self._history.shape)] == 0)
        ):
            raise V168LiveContractError("V16.8 future history is not exact zero")
        current_tensor = torch.from_numpy(self._round.current_state[None]).to(self._device)
        output = self._actor(
            current_full256=torch.from_numpy(self._round.full256[None]).to(self._device),
            view_valid=torch.from_numpy(self._round.view_valid[None]).to(self._device),
            current_state=current_tensor,
            past_completed_commands=torch.from_numpy(
                self._condensed_past(self._round.current_state)[None]
            ).to(self._device),
            completed_command_history=torch.from_numpy(self._history[None]).to(self._device),
            history_valid=torch.from_numpy(valid[None]).to(self._device),
            transition=torch.tensor([self._transition], dtype=torch.int64, device=self._device),
            desired_joint_goal=torch.from_numpy(self._goal[None]).to(self._device),
            baseline_kind="repeat",
            baseline_joint_goal=None,
        )
        canonical = output.action[0].float().cpu().numpy().astype(np.float32)
        chunk = canonical * np.float32(CANONICAL_TO_DEGREES_SCALE)
        if chunk.shape != (COMMAND_STEPS, STATE_DIM) or not bool(np.isfinite(chunk).all()):
            raise V168LiveContractError("V16.8 Actor output is not finite [5,6]")
        prefix = np.repeat(chunk[-1:], int(horizon_raw), axis=0)
        prefix[:COMMAND_STEPS] = chunk
        phase_value = phase.value if hasattr(phase, "value") else str(phase)
        metadata = {
            "source": ACTOR_SOURCE,
            "variant": "five_command_then_hold",
            "actor_chunk_steps": COMMAND_STEPS,
            "phase": phase_value,
            "transition": f"H{self._transition}",
            "history_valid": valid.tolist(),
            "goal_fixed_window": True,
            "goal_source": dict(self._goal_source or {}),
            "training_h4_equivalence_verified": False,
            "lewm_scored": True,
            "v168_mode": self._mode,
            "foundation_bundle_id": self._foundation_receipt["bundle_id"],
        }
        self._last_proposal = {
            "generated": True,
            "transition": f"H{self._transition}",
            "phase": phase_value,
            "candidate_inserted": self.decision_influence,
            "chunk_deg": chunk.tolist(),
        }
        return {"prefixes": prefix[None], "metadata": [metadata]}

    def record_shadow_lewm(self, predicted_states_deg: np.ndarray) -> None:
        if self._mode != "shadow":
            raise V168LiveContractError("shadow LeWM receipt requires shadow mode")
        states = np.asarray(predicted_states_deg, dtype=np.float32)
        if states.shape != (1, COMMAND_STEPS, STATE_DIM) or not bool(np.isfinite(states).all()):
            raise V168LiveContractError("V16.8 shadow LeWM output must be [1,5,6]")
        self._last_shadow_lewm = {
            "ran": True,
            "commands_sent": 0,
            "decision_influence": False,
            "predicted_states_deg": states[0].tolist(),
        }

    def commit_completed_chunk(
        self,
        sent_chunk_deg: np.ndarray,
        *,
        round_id: str,
        selected_source: str | None,
    ) -> dict[str, Any]:
        """Advance only from the common runtime's verified ``result.sent``."""

        if self._mode == "shadow":
            raise V168LiveContractError("shadow mode must never commit sent commands")
        sent = np.asarray(sent_chunk_deg, dtype=np.float32)
        if sent.shape != (COMMAND_STEPS, STATE_DIM) or not bool(np.isfinite(sent).all()):
            raise V168LiveContractError("completed sent chunk must be finite [5,6]")
        canonical = sent / np.float32(CANONICAL_TO_DEGREES_SCALE)
        lower = np.asarray(ACTION_CONTRACT["lower"], dtype=np.float32)
        upper = np.asarray(ACTION_CONTRACT["upper"], dtype=np.float32)
        if bool(((canonical < lower) | (canonical > upper)).any()):
            raise V168LiveContractError("completed sent chunk leaves canonical bounds")
        transition_before = self._transition
        window_before = self._window_index
        if transition_before < HISTORY_CHUNKS:
            self._history[transition_before] = canonical
            self._transition += 1
        else:
            # H3's command is the fourth chunk. It completes, but is never a
            # history input inside the finished window; the next call is H0.
            self.reset_window("four_chunk_window_complete")
        self._completed_total += 1
        self._last_commit = {
            "round_id": str(round_id),
            "selected_source": selected_source,
            "transition_before": f"H{transition_before}",
            "transition_after": f"H{self._transition}",
            "window_before": window_before,
            "window_after": self._window_index,
            "commands": COMMAND_STEPS,
            "provenance": "common_runtime_verified_result_sent",
        }
        return dict(self._last_commit)


__all__ = [
    "ACTION_CONTRACT",
    "ACTOR_SOURCE",
    "CANONICAL_TO_DEGREES_SCALE",
    "MODES",
    "PROTOCOL_ID",
    "PROXY_GOAL_COMMAND_INDEX",
    "V168FoundationRuntimeAdapter",
    "V168LiveActorOverlay",
    "V168LiveContractError",
    "WINDOW_COMMANDS",
]
