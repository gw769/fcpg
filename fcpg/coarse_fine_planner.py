"""Planner-facing CoarseFine gates, fusion consumption, and 2x2 mode helpers.

This module does not train, does not touch LeWM loss, and never infers
``coarse_fine`` from state-dict keys. Old payloads stay on the bridge
``independent3`` / ``two_tip_structured`` path.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from fcpg.grounding_confidence_fusion import (
    COARSE_FINE_VARIANT,
    FALLBACK_LAST_ACCEPTED,
    GROUNDING_GATED_FCPG,
    GROUNDING_NOMINAL,
    POLICY_UNVALIDATED,
    RANKING_OFF,
    RANKING_ON,
    VIEW_WRIST,
    GroundingFusionState,
    GroundingRankingContract,
    corrected_under_contract,
    envelope_policy_for_variant,
    fuse_from_coarse_fine_output,
    next_correction_seed_px,
)
from fcpg.so101_fcpg_lewm_bridge import infer_so101_fcpg_corrector_variant


GROUNDING_ARM_NOMINAL = "nominal"
GROUNDING_ARM_FCPG = "fcpg"
GROUNDING_ARM_FCPG_LEWM = "fcpg_lewm"

_METADATA_KEYS = (
    "coarse_fine_metadata",
    "corrector_metadata",
)
_WEIGHT_KEYS = ("corrector", "bridge", "state_dict", "model")
_CALIBRATION_FLAGS = ("has_physical_labels", "uncertainty_calibrated")


def peek_bridge_corrector_variant(source: Mapping[str, Any] | Path | str) -> str:
    """Read the explicit/inferred variant without constructing a planner."""

    payload: Mapping[str, Any]
    if isinstance(source, (str, Path)):
        import torch

        loaded = torch.load(Path(source), map_location="cpu", weights_only=False)
        if not isinstance(loaded, Mapping):
            raise TypeError(f"{source} is not a checkpoint mapping")
        payload = loaded
    else:
        payload = source
    return infer_so101_fcpg_corrector_variant(dict(payload))


def _looks_like_weight_payload(payload: Mapping[str, Any]) -> bool:
    return any(key in payload for key in _WEIGHT_KEYS)


def extract_coarse_fine_checkpoint_metadata(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Pull physical/calibration flags. Missing metadata is unready, not inferred.

    Dedicated ``coarse_fine_metadata`` / ``corrector_metadata`` win. A full
    checkpoint that merely carries a generic ``metadata`` dict, stray
    top-level flags, or the same nested keys inside ``args`` is unready:
    those collide with old ``vars(args)`` training dumps and must not stamp
    live-execute readiness.
    """

    raw: Mapping[str, Any] | None = None
    for key in _METADATA_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            raw = candidate
            break
    # Weight payloads save ``args: vars(args)``. Nested dedicated keys there
    # are leftover CLI/config, not a live-ready stamp.
    if raw is None and not _looks_like_weight_payload(payload):
        args = payload.get("args")
        if isinstance(args, Mapping):
            for key in _METADATA_KEYS:
                candidate = args.get(key)
                if isinstance(candidate, Mapping):
                    raw = candidate
                    break
    if raw is None and not _looks_like_weight_payload(payload):
        generic = payload.get("metadata")
        if isinstance(generic, Mapping) and any(
            flag in generic for flag in _CALIBRATION_FLAGS
        ):
            raw = generic
        elif any(flag in payload for flag in _CALIBRATION_FLAGS):
            raw = payload
    if raw is None:
        return {
            "present": False,
            "has_physical_labels": False,
            "uncertainty_calibrated": False,
        }
    return {
        "present": True,
        "has_physical_labels": bool(raw.get("has_physical_labels", False)),
        "uncertainty_calibrated": bool(raw.get("uncertainty_calibrated", False)),
        "training_data_kind": raw.get("training_data_kind"),
        "predictor_id": raw.get("predictor_id"),
        "format": raw.get("format"),
    }


def coarse_fine_ready_for_live_execute(metadata: Mapping[str, Any] | None) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    return bool(metadata.get("has_physical_labels")) and bool(
        metadata.get("uncertainty_calibrated")
    )


def grounding_arm_name(grounding_mode: str, ranking_mode: str) -> str:
    """Named three-way view of the 2x2 grounding x ranking grid."""

    if grounding_mode == GROUNDING_NOMINAL:
        return GROUNDING_ARM_NOMINAL
    if ranking_mode == RANKING_ON:
        return GROUNDING_ARM_FCPG_LEWM
    return GROUNDING_ARM_FCPG


def planner_grounding_ranking_contract(
    *,
    grounding_mode: str = GROUNDING_GATED_FCPG,
    ranking_mode: str = RANKING_ON,
) -> GroundingRankingContract:
    return GroundingRankingContract(
        grounding_mode=str(grounding_mode),
        ranking_mode=str(ranking_mode),
    )


def resolve_global_tip_checkpoint_for_variant(
    variant: str,
    global_tip_checkpoint: Path | str | None,
) -> tuple[Path | None, dict[str, Any]]:
    """Fail-closed if both are requested. Never chain coarse_fine with global tip."""

    checkpoint = None if global_tip_checkpoint is None else Path(global_tip_checkpoint)
    record = {
        "requested": checkpoint is not None,
        "disabled_for_coarse_fine": False,
        "path": None if checkpoint is None else str(checkpoint),
    }
    if str(variant) != COARSE_FINE_VARIANT:
        return checkpoint, record
    record["disabled_for_coarse_fine"] = True
    if checkpoint is not None:
        raise ValueError(
            "coarse_fine cannot chain with the planner global-tip seed; "
            "pass disable_global_tip_locator / global_tip_checkpoint=None"
        )
    return None, record


def assert_coarse_fine_planner_policy(
    *,
    variant: str,
    metadata: Mapping[str, Any] | None,
    recursive_grounding: bool = False,
    global_tip_checkpoint: Path | str | None = None,
    request_live_execute: bool = False,
    allow_uncalibrated_coarse_fine_shadow: bool = False,
) -> dict[str, Any]:
    """Refuse illegal CoarseFine deployments. Old variants are a no-op."""

    variant = str(variant)
    if variant != COARSE_FINE_VARIANT:
        return {
            "variant": variant,
            "coarse_fine": False,
            "live_execute_allowed": True,
            "shadow_allowed": True,
            "global_tip_disabled": False,
            "wrist_correction_enabled": None,
        }
    if recursive_grounding:
        raise ValueError(
            "recursive_grounding cannot be combined with coarse_fine; "
            "next-frame seed must stay nominal"
        )
    resolve_global_tip_checkpoint_for_variant(variant, global_tip_checkpoint)
    meta = extract_coarse_fine_checkpoint_metadata(metadata or {})
    ready = coarse_fine_ready_for_live_execute(meta)
    if request_live_execute and not ready:
        raise ValueError(
            "coarse_fine live execute is refused: checkpoint has no physical "
            "labels or uncertainty is uncalibrated; random/unvalidated "
            "architectures cannot be deployed"
        )
    if not ready and not allow_uncalibrated_coarse_fine_shadow:
        raise ValueError(
            "coarse_fine without physical labels / calibrated uncertainty "
            "requires allow_uncalibrated_coarse_fine_shadow for offline/shadow only"
        )
    return {
        "variant": variant,
        "coarse_fine": True,
        "live_execute_allowed": bool(ready),
        "shadow_allowed": True,
        "has_physical_labels": bool(meta.get("has_physical_labels")),
        "uncertainty_calibrated": bool(meta.get("uncertainty_calibrated")),
        "metadata_present": bool(meta.get("present")),
        "global_tip_disabled": True,
        "wrist_correction_enabled": False,
        "next_seed_policy": "nominal",
        "allow_uncalibrated_coarse_fine_shadow": bool(
            allow_uncalibrated_coarse_fine_shadow
        ),
        "request_live_execute": bool(request_live_execute),
    }


def apply_coarse_fine_observe_points(
    native: Any,
    *,
    seed_px: Any,
    nominal_px: Any,
    view: str,
    view_valid: Any | None,
    contract: GroundingRankingContract,
    fusion_state: GroundingFusionState | None = None,
    column_wrist_conflict: Any = False,
) -> tuple[np.ndarray, Any]:
    """Fuse sidecar residuals and REPLACE planner points. Do not add again."""

    seed = np.asarray(seed_px, dtype=np.float32)
    nominal = np.asarray(nominal_px, dtype=np.float32)
    fusion = fuse_from_coarse_fine_output(
        native,
        nominal_px=seed,
        view_valid=view_valid,
        view=view,
        state=fusion_state,
        column_wrist_conflict=column_wrist_conflict,
    )
    gated = np.asarray(fusion.corrected_px, dtype=np.float32)
    if gated.shape != seed.shape:
        gated = gated.reshape(seed.shape)
    points = corrected_under_contract(nominal, gated, contract, view=view)
    if (
        fusion_state is not None
        and str(view) != VIEW_WRIST
        and contract.grounding_mode == GROUNDING_GATED_FCPG
    ):
        last_fallback = _last_fallback(fusion.fallback)
        if last_fallback is None:
            last = gated[-1] if gated.ndim == 3 else gated
            fusion_state.record_accepted(last, view=str(view))
    _ = next_correction_seed_px(nominal)
    return np.asarray(points, dtype=np.float32), fusion


def coarse_fine_observe_diagnostics(
    native: Any,
    fusion: Any,
    *,
    view: str,
    variant: str,
    contract: GroundingRankingContract,
    global_tip_disabled: bool,
    wrist_correction_enabled: bool,
) -> dict[str, Any]:
    """Planner-facing sidecar record. Reliability/sigma stay unvalidated."""

    fusion_diag = dict(getattr(fusion, "diagnostics", {}) or {})
    return {
        "corrector_variant": str(variant),
        "delta_coarse_px": _jsonable_array(getattr(native, "delta_coarse_px", None)),
        "delta_fine_px": _jsonable_array(getattr(native, "delta_fine_px", None)),
        "delta_total_px": _jsonable_array(getattr(native, "delta_px", None)),
        "accept_coarse": _jsonable_array(getattr(native, "accept_coarse", None)),
        "coarse_score": _jsonable_array(getattr(native, "coarse_score", None)),
        "sigma_fine_px": _jsonable_array(getattr(native, "sigma_fine_px", None)),
        "reliability": _jsonable_array(getattr(native, "reliability", None)),
        "reliability_status": "unvalidated",
        "sigma_status": "unvalidated",
        "uncertainty_calibrated": bool(
            getattr(native, "uncertainty_calibrated", False)
        ),
        "policy": fusion_diag.get("policy", POLICY_UNVALIDATED),
        "tier": _jsonable_value(getattr(fusion, "tier", fusion_diag.get("tier"))),
        "applied_scale": _jsonable_value(
            getattr(fusion, "applied_scale", fusion_diag.get("applied_scale"))
        ),
        "fallback": _jsonable_value(
            getattr(fusion, "fallback", fusion_diag.get("fallback"))
        ),
        "fusion_reasons": tuple(getattr(fusion, "reasons", ()) or ()),
        "envelope_policy": envelope_policy_for_variant(variant),
        "envelope_degraded": _jsonable_value(fusion_diag.get("envelope_degraded")),
        "global_tip_disabled": bool(global_tip_disabled),
        "grounding_mode": contract.grounding_mode,
        "ranking_mode": contract.ranking_mode,
        "grounding_arm": grounding_arm_name(
            contract.grounding_mode, contract.ranking_mode
        ),
        "wrist_correction_enabled": bool(wrist_correction_enabled),
        "wrist_primary": contract.wrist_primary,
        "next_seed_policy": fusion_diag.get("next_seed_policy", "nominal"),
        "next_seed_is_previous_residual": False,
        "last_accepted_used_as_seed": False,
        "last_accepted_fallback_only": True,
        "fusion_replaces_points": True,
        "fusion_added_to_model_points": False,
        "view": str(view),
        "fallback_last_accepted": FALLBACK_LAST_ACCEPTED,
    }


def disabled_global_tip_diagnostics() -> dict[str, Any]:
    return {
        "global_locator_enabled": False,
        "global_locator_accepted": False,
        "global_locator_rejection_reasons": ("disabled_for_coarse_fine",),
        "global_tip_disabled": True,
    }


def _last_fallback(fallback: Any) -> str | None:
    if fallback is None:
        return None
    if isinstance(fallback, np.ndarray):
        if fallback.size == 0:
            return None
        value = fallback.reshape(-1)[-1]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return None if value is None else str(value) if value else None
    text = str(fallback)
    return None if text in {"", "None", "nan"} else text


def _jsonable_array(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        return array.tolist()
    return array.tolist()


def _jsonable_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_jsonable_value(item) for item in value.tolist()]
        if value.shape == ():
            item = value.item()
            return None if item is None else item
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value
