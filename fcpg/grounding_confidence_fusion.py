"""Consume CoarseFine uncertainty before planner-facing corrected anchors.

Sigma, scores, and view conflict stay out of LeWM dynamics, residual-only
``LeWMGroundingOutput``, raw CLS, and training loss. This module is the
sidecar that turns coarse/fine residuals into a fused corrected point, a
confidence tier, and an explicit fallback.

``fcpg.coarse_fine_fcpg`` is owned by another module. This file imports that
API by convention (``CoarseFineGroundingOutput``, ``as_lewm_output``,
``CoarseFineFCPG``) and otherwise types against the protocol below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Protocol, runtime_checkable

import numpy as np


FCPG_TRAINING_ENVELOPE_PX = 18.0

TIER_HIGH = "high"
TIER_MID = "mid"
TIER_LOW = "low"

FALLBACK_NOMINAL = "nominal"
FALLBACK_LAST_ACCEPTED = "last_accepted"

GROUNDING_NOMINAL = "nominal"
GROUNDING_GATED_FCPG = "gated_fcpg"
RANKING_OFF = "off"
RANKING_ON = "on"

ENVELOPE_POLICY_BLOCK = "block"
ENVELOPE_POLICY_DEGRADE = "degrade"

COARSE_FINE_VARIANT = "coarse_fine"
LEGACY_CORRECTOR_VARIANTS = ("independent3", "two_tip_structured")

VIEW_COLUMN = "column"
VIEW_WRIST = "wrist"

POLICY_UNVALIDATED = "policy_unvalidated"
NEXT_SEED_POLICY_NOMINAL = "nominal"

REASON_COARSE_REJECTED = "coarse_rejected"
REASON_FINE_GATED = "fine_gated"
REASON_COLLAPSE = "collapse"
REASON_EDGE = "edge"
REASON_INVALID = "invalid"
REASON_VIEW_INVALID = "view_invalid"
REASON_CONFLICT = "column_wrist_conflict"
REASON_WRIST_FALLBACK_FORBIDDEN = "wrist_fallback_forbidden"
REASON_ENVELOPE_DEGRADED = "envelope_degraded"
REASON_POLICY_UNVALIDATED = POLICY_UNVALIDATED
REASON_NONFINITE = "nonfinite"
REASON_LOW_MARGIN = "low_peak_margin"
REASON_HIGH_ENTROPY = "high_entropy"

_TIP_SLICES = (1, 2)
_CENTER_INDEX = 0


class FallbackSource(str, Enum):
    """Allowed fallback destinations. Wrist is intentionally absent."""

    NOMINAL = FALLBACK_NOMINAL
    LAST_ACCEPTED = FALLBACK_LAST_ACCEPTED


class ConfidenceTier(str, Enum):
    HIGH = TIER_HIGH
    MID = TIER_MID
    LOW = TIER_LOW


@runtime_checkable
class CoarseFineGroundingLike(Protocol):
    """Convention imported from ``fcpg.coarse_fine_fcpg`` when that file exists.

    Expected fields (another agent owns the concrete dataclass):

    ``delta_coarse_px``, ``delta_fine_px``, compatible ``delta_px``,
    ``q_coarse_px``, ``q_corrected_px`` / ``p_hat_px``,
    ``coarse_score`` / ``peak_margin`` / ``entropy`` / ``accept_coarse``,
    ``sigma_fine_px``, ``reliability``, ``uncertainty_calibrated=False``,
    ``keypoint_latent``. Optional ``as_lewm_output`` method; module-level
    ``as_lewm_output`` is also accepted. Residual-only adaptation must not
    copy reliability or sigma onto ``LeWMGroundingOutput``.
    """

    delta_coarse_px: Any
    delta_fine_px: Any
    accept_coarse: Any


@dataclass(frozen=True)
class ConfidenceFusionConfig:
    """Preregistered consumption thresholds. Mid scale must never amplify."""

    high_min_score: float
    mid_min_score: float
    high_max_sigma_px: float
    mid_max_sigma_px: float
    mid_scale: float
    envelope_px: float = FCPG_TRAINING_ENVELOPE_PX
    min_peak_margin: float | None = 0.15
    max_entropy: float | None = 2.5
    uncertainty_calibrated: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < float(self.mid_scale) <= 1.0:
            raise ValueError("mid_scale must lie in (0, 1]; amplification is forbidden")
        if not np.isfinite(self.envelope_px) or float(self.envelope_px) <= 0.0:
            raise ValueError("envelope_px must be a finite positive scalar")
        if self.high_min_score < self.mid_min_score:
            raise ValueError("high_min_score must be >= mid_min_score")
        if self.high_max_sigma_px > self.mid_max_sigma_px:
            raise ValueError("high_max_sigma_px must be <= mid_max_sigma_px")
        for name in (
            "high_min_score",
            "mid_min_score",
            "high_max_sigma_px",
            "mid_max_sigma_px",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative scalar")


@dataclass(frozen=True)
class GroundingRankingContract:
    """2×2 planner mode: grounding × ranking. Wrist primary is forced off."""

    grounding_mode: str
    ranking_mode: str
    wrist_primary: str = "off"

    def __post_init__(self) -> None:
        if self.grounding_mode not in {GROUNDING_NOMINAL, GROUNDING_GATED_FCPG}:
            raise ValueError(
                f"grounding_mode must be {GROUNDING_NOMINAL!r} or "
                f"{GROUNDING_GATED_FCPG!r}"
            )
        if self.ranking_mode not in {RANKING_OFF, RANKING_ON}:
            raise ValueError(
                f"ranking_mode must be {RANKING_OFF!r} or {RANKING_ON!r}"
            )
        if self.wrist_primary != "off":
            raise ValueError("wrist primary comparison is forced off")


@dataclass
class GroundingFusionState:
    """Last accepted same-view pose. Never a next-round seed. Never wrist."""

    last_accepted_px: np.ndarray | None = None
    last_accepted_view: str | None = None

    def record_accepted(self, corrected_px: Any, *, view: str) -> bool:
        if str(view) == VIEW_WRIST:
            return False
        points = _as_numpy(corrected_px, dtype=np.float32)
        if points is None or points.shape[-2:] != (3, 2):
            return False
        if not np.all(np.isfinite(points)):
            return False
        self.last_accepted_px = np.asarray(points, dtype=np.float32).copy()
        self.last_accepted_view = str(view)
        return True

    def next_seed_px(self, nominal_px: Any) -> np.ndarray:
        return next_correction_seed_px(nominal_px)


@dataclass(frozen=True)
class GroundingConfidenceFusion:
    corrected_px: np.ndarray
    applied_scale: float | np.ndarray
    tier: str | np.ndarray
    fallback: str | None | np.ndarray
    reasons: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return False


@dataclass(frozen=True)
class GroundingConfidenceSidecar:
    """Parallel diagnostic payload. Not a LeWM input."""

    fusion: GroundingConfidenceFusion
    lewm_consumed_sigma: bool = False
    raw_context_depends_on_fusion: bool = False


def resolve_fusion_config(*, uncertainty_calibrated: bool = False) -> ConfidenceFusionConfig:
    """Return preregistered thresholds. Uncalibrated is the conservative default."""

    if uncertainty_calibrated:
        return ConfidenceFusionConfig(
            high_min_score=0.70,
            mid_min_score=0.40,
            high_max_sigma_px=3.0,
            mid_max_sigma_px=8.0,
            mid_scale=0.70,
            envelope_px=FCPG_TRAINING_ENVELOPE_PX,
            min_peak_margin=0.10,
            max_entropy=3.5,
            uncertainty_calibrated=True,
        )
    return ConfidenceFusionConfig(
        high_min_score=0.85,
        mid_min_score=0.55,
        high_max_sigma_px=2.0,
        mid_max_sigma_px=5.0,
        mid_scale=0.50,
        envelope_px=FCPG_TRAINING_ENVELOPE_PX,
        min_peak_margin=0.15,
        max_entropy=2.5,
        uncertainty_calibrated=False,
    )


def envelope_policy_for_variant(variant: str) -> str:
    """Old variants still whole-round block; coarse_fine degrades instead."""

    if str(variant) == COARSE_FINE_VARIANT:
        return ENVELOPE_POLICY_DEGRADE
    return ENVELOPE_POLICY_BLOCK


def planner_envelope_should_block(
    variant: str,
    residual_norm_px: float,
    envelope_px: float = FCPG_TRAINING_ENVELOPE_PX,
) -> bool:
    """Planner drop-in: coarse_fine never whole-round-blocks on the 18 px cap."""

    if envelope_policy_for_variant(variant) == ENVELOPE_POLICY_DEGRADE:
        return False
    return float(residual_norm_px) > float(envelope_px)


def iter_grounding_ranking_grid() -> Iterator[GroundingRankingContract]:
    for grounding in (GROUNDING_NOMINAL, GROUNDING_GATED_FCPG):
        for ranking in (RANKING_OFF, RANKING_ON):
            yield GroundingRankingContract(
                grounding_mode=grounding,
                ranking_mode=ranking,
            )


def corrected_under_contract(
    nominal_px: Any,
    gated_corrected_px: Any,
    contract: GroundingRankingContract,
    *,
    view: str = VIEW_COLUMN,
) -> np.ndarray:
    """Select planner-facing anchors. Ranking must not change the result."""

    _ = contract.ranking_mode
    nominal = np.asarray(_as_numpy(nominal_px, dtype=np.float32), dtype=np.float32)
    gated = np.asarray(
        _as_numpy(gated_corrected_px, dtype=np.float32), dtype=np.float32
    )
    if str(view) == VIEW_WRIST:
        return nominal.copy()
    if contract.grounding_mode == GROUNDING_NOMINAL:
        return nominal.copy()
    return gated.copy()


def next_correction_seed_px(
    nominal_px: Any,
    fused_corrected_px: Any | None = None,
    last_residual_px: Any | None = None,
) -> np.ndarray:
    """Next-round seed is always nominal. Previous residual is never a seed."""

    _ = fused_corrected_px, last_residual_px
    seed = _as_numpy(nominal_px, dtype=np.float32)
    if seed is None:
        raise ValueError("nominal_px is required")
    return np.asarray(seed, dtype=np.float32).copy()


def is_coarse_fine_output(output: Any) -> bool:
    if output is None:
        return False
    type_name = type(output).__name__
    if type_name == "LeWMGroundingOutput":
        return False
    return all(
        hasattr(output, name)
        for name in ("delta_coarse_px", "delta_fine_px", "accept_coarse")
    )


def as_residual_only_lewm_output(output: Any) -> Any:
    """Feed old bridge context. Never copies reliability or sigma."""

    from fcpg.transformer_grounding import LeWMGroundingOutput

    if isinstance(output, LeWMGroundingOutput):
        _assert_residual_only_lewm_output(output)
        return output

    method = getattr(output, "as_lewm_output", None)
    if callable(method):
        try:
            adapted = method()
        except TypeError:
            adapted = method(output)
        if isinstance(adapted, LeWMGroundingOutput):
            _assert_residual_only_lewm_output(adapted)
            return adapted

    module_fn = _optional_coarse_fine_as_lewm_output()
    if module_fn is not None:
        try:
            adapted = module_fn(output)
        except TypeError:
            adapted = None
        if isinstance(adapted, LeWMGroundingOutput):
            _assert_residual_only_lewm_output(adapted)
            return adapted

    if all(
        hasattr(output, name)
        for name in ("delta_px", "q_corrected_px", "keypoint_latent")
    ):
        adapted = LeWMGroundingOutput(
            delta_px=output.delta_px,
            q_corrected_px=output.q_corrected_px,
            keypoint_latent=output.keypoint_latent,
            feature_map=getattr(output, "feature_map", None),
            global_feature=getattr(output, "global_feature", None),
        )
        _assert_residual_only_lewm_output(adapted)
        return adapted
    raise TypeError("corrector output cannot be adapted to residual-only LeWMGroundingOutput")


def fuse_from_coarse_fine_output(
    output: Any,
    *,
    nominal_px: Any,
    view_valid: Any | None = None,
    last_accepted_px: Any | None = None,
    last_accepted_view: str | None = None,
    column_wrist_conflict: Any = False,
    view: str = VIEW_COLUMN,
    state: GroundingFusionState | None = None,
    config: ConfidenceFusionConfig | None = None,
) -> GroundingConfidenceFusion:
    """Extract the CoarseFine protocol fields and run the pure consumer."""

    if not is_coarse_fine_output(output):
        raise TypeError("output does not look like CoarseFineGroundingOutput")
    q_corrected = getattr(output, "q_corrected_px", None)
    if q_corrected is None:
        q_corrected = getattr(output, "p_hat_px", None)
    _ = q_corrected
    calibrated = bool(getattr(output, "uncertainty_calibrated", False))
    return fuse_grounding_confidence(
        nominal_px=nominal_px,
        delta_coarse_px=output.delta_coarse_px,
        delta_fine_px=output.delta_fine_px,
        accept_coarse=output.accept_coarse,
        coarse_score=getattr(output, "coarse_score", None),
        sigma_fine_px=getattr(output, "sigma_fine_px", None),
        view_valid=view_valid,
        last_accepted_px=last_accepted_px,
        last_accepted_view=last_accepted_view,
        column_wrist_conflict=column_wrist_conflict,
        collapse=getattr(output, "collapse", False),
        edge=getattr(output, "edge", False),
        invalid=getattr(output, "invalid", False),
        peak_margin=getattr(output, "peak_margin", None),
        entropy=getattr(output, "entropy", None),
        view=view,
        uncertainty_calibrated=calibrated,
        config=config,
        state=state,
    )


def fuse_grounding_confidence(
    *,
    nominal_px: Any,
    delta_coarse_px: Any,
    delta_fine_px: Any,
    accept_coarse: Any,
    coarse_score: Any | None = None,
    sigma_fine_px: Any | None = None,
    view_valid: Any | None = None,
    last_accepted_px: Any | None = None,
    last_accepted_view: str | None = None,
    column_wrist_conflict: Any = False,
    collapse: Any = False,
    edge: Any = False,
    invalid: Any = False,
    peak_margin: Any | None = None,
    entropy: Any | None = None,
    view: str = VIEW_COLUMN,
    uncertainty_calibrated: bool = False,
    config: ConfidenceFusionConfig | None = None,
    state: GroundingFusionState | None = None,
) -> GroundingConfidenceFusion:
    """Fuse coarse+fine residuals with a three-tier consume / fallback policy."""

    if config is None:
        config = resolve_fusion_config(uncertainty_calibrated=uncertainty_calibrated)
    elif config.uncertainty_calibrated != bool(uncertainty_calibrated):
        config = ConfidenceFusionConfig(
            high_min_score=config.high_min_score,
            mid_min_score=config.mid_min_score,
            high_max_sigma_px=config.high_max_sigma_px,
            mid_max_sigma_px=config.mid_max_sigma_px,
            mid_scale=config.mid_scale,
            envelope_px=config.envelope_px,
            min_peak_margin=config.min_peak_margin,
            max_entropy=config.max_entropy,
            uncertainty_calibrated=bool(uncertainty_calibrated),
        )

    nominal = _require_anchor_points(nominal_px, "nominal_px")
    coarse = _require_anchor_points(delta_coarse_px, "delta_coarse_px")
    fine = _require_anchor_points(delta_fine_px, "delta_fine_px")
    if coarse.shape != nominal.shape or fine.shape != nominal.shape:
        raise ValueError("nominal, delta_coarse, and delta_fine must share shape")

    leading = nominal.shape[:-2]
    count = int(np.prod(leading, dtype=np.int64)) if leading else 1
    nom = nominal.reshape(count, 3, 2)
    coarse_b = coarse.reshape(count, 3, 2)
    fine_b = fine.reshape(count, 3, 2)

    accept = _broadcast_bool(accept_coarse, leading, count, name="accept_coarse")
    collapse_b = _broadcast_bool(collapse, leading, count, name="collapse")
    edge_b = _broadcast_bool(edge, leading, count, name="edge")
    invalid_b = _broadcast_bool(invalid, leading, count, name="invalid")
    conflict_b = _broadcast_bool(
        column_wrist_conflict, leading, count, name="column_wrist_conflict"
    )
    view_ok = _broadcast_view_valid(view_valid, leading, count)
    scores = _reduce_score(coarse_score, leading, count)
    sigmas = _reduce_sigma(sigma_fine_px, leading, count)
    margins = _optional_reduce_min(peak_margin, leading, count)
    entropies = _optional_reduce_max(entropy, leading, count)

    last = last_accepted_px
    last_view = last_accepted_view
    if state is not None:
        if last is None:
            last = state.last_accepted_px
            last_view = state.last_accepted_view
    last_usable, wrist_forbidden = _usable_last_accepted(
        last,
        last_accepted_view=last_view,
        view=str(view),
        leading=leading,
        count=count,
        template=nom,
    )

    corrected = np.empty_like(nom)
    scales = np.zeros(count, dtype=np.float64)
    tiers = np.empty(count, dtype=object)
    fallbacks = np.empty(count, dtype=object)
    reason_rows: list[list[str]] = []
    envelope_flags = np.zeros(count, dtype=bool)
    candidate_norms = np.zeros(count, dtype=np.float64)
    applied_norms = np.zeros(count, dtype=np.float64)
    fine_applied = np.zeros(count, dtype=bool)
    coarse_applied = np.zeros(count, dtype=bool)

    for index in range(count):
        reasons: list[str] = []
        if not config.uncertainty_calibrated:
            reasons.append(REASON_POLICY_UNVALIDATED)
        if wrist_forbidden:
            reasons.append(REASON_WRIST_FALLBACK_FORBIDDEN)

        force_low = False
        if not bool(accept[index]):
            force_low = True
            reasons.extend((REASON_COARSE_REJECTED, REASON_FINE_GATED))
        if bool(collapse_b[index]):
            force_low = True
            reasons.append(REASON_COLLAPSE)
        if bool(edge_b[index]):
            force_low = True
            reasons.append(REASON_EDGE)
        if bool(invalid_b[index]):
            force_low = True
            reasons.append(REASON_INVALID)
        if not bool(view_ok[index]):
            force_low = True
            reasons.append(REASON_VIEW_INVALID)
        if bool(conflict_b[index]):
            force_low = True
            reasons.append(REASON_CONFLICT)
        if not np.all(np.isfinite(nom[index])) or not np.all(
            np.isfinite(coarse_b[index])
        ) or not np.all(np.isfinite(fine_b[index])):
            force_low = True
            reasons.append(REASON_NONFINITE)
        if (
            margins is not None
            and config.min_peak_margin is not None
            and np.isfinite(margins[index])
            and float(margins[index]) < float(config.min_peak_margin)
        ):
            force_low = True
            reasons.append(REASON_LOW_MARGIN)
        if (
            entropies is not None
            and config.max_entropy is not None
            and np.isfinite(entropies[index])
            and float(entropies[index]) > float(config.max_entropy)
        ):
            force_low = True
            reasons.append(REASON_HIGH_ENTROPY)

        accept_i = bool(accept[index]) and not force_low
        candidate_fine = fine_b[index] if accept_i else np.zeros((3, 2), dtype=np.float64)
        candidate_coarse = (
            coarse_b[index] if accept_i else np.zeros((3, 2), dtype=np.float64)
        )
        if not bool(accept[index]):
            candidate_fine = np.zeros((3, 2), dtype=np.float64)
            candidate_coarse = np.zeros((3, 2), dtype=np.float64)
        candidate = candidate_coarse + candidate_fine
        max_norm = float(np.max(_anchor_norms(candidate)))
        candidate_norms[index] = max_norm

        score_i = float(scores[index])
        sigma_i = float(sigmas[index])
        if force_low:
            tier = TIER_LOW
        elif (
            score_i >= config.high_min_score
            and sigma_i <= config.high_max_sigma_px
        ):
            tier = TIER_HIGH
        elif (
            score_i >= config.mid_min_score
            and sigma_i <= config.mid_max_sigma_px
        ):
            tier = TIER_MID
        else:
            tier = TIER_LOW

        scale = 0.0
        fallback: str | None = None
        if tier == TIER_HIGH:
            scale = 1.0
        elif tier == TIER_MID:
            scale = float(config.mid_scale)
        else:
            last_i = None if last_usable is None else last_usable[index]
            if last_i is not None and not np.all(np.isfinite(last_i)):
                last_i = None
            if last_i is not None:
                corrected[index] = last_i
                fallback = FALLBACK_LAST_ACCEPTED
            else:
                corrected[index] = nom[index]
                fallback = FALLBACK_NOMINAL
            scale = 0.0

        if tier != TIER_LOW:
            if max_norm * scale > float(config.envelope_px) + 1.0e-12:
                if max_norm > 0.0:
                    scale = min(scale, float(config.envelope_px) / max_norm)
                else:
                    scale = 0.0
                envelope_flags[index] = True
                reasons.append(REASON_ENVELOPE_DEGRADED)
            scale = min(float(scale), 1.0)
            if scale < 0.0:
                scale = 0.0
            applied = scale * candidate
            if _anchor_norms(applied).max(initial=0.0) > max_norm + 1.0e-9:
                raise RuntimeError("fusion amplified residual norm")
            corrected[index] = nom[index] + applied
            coarse_applied[index] = bool(accept[index]) and scale > 0.0
            fine_applied[index] = bool(accept[index]) and scale > 0.0

        corrected[index, _CENTER_INDEX] = 0.5 * (
            corrected[index, 1] + corrected[index, 2]
        )
        applied_norms[index] = float(
            np.max(_anchor_norms(corrected[index] - nom[index]))
        )
        scales[index] = scale
        tiers[index] = tier
        fallbacks[index] = fallback
        reasons.append(f"tier_{tier}")
        if fallback is not None:
            reasons.append(f"fallback_{fallback}")
        reason_rows.append(_unique_preserve(reasons))

    corrected = _rederive_centers(corrected).reshape(nominal.shape).astype(np.float32)
    union_reasons = _unique_preserve(code for row in reason_rows for code in row)
    diagnostics = {
        "tier": tiers.reshape(leading) if leading else str(tiers[0]),
        "applied_scale": scales.reshape(leading) if leading else float(scales[0]),
        "fallback": fallbacks.reshape(leading) if leading else fallbacks[0],
        "uncertainty_calibrated": bool(config.uncertainty_calibrated),
        "policy": (
            POLICY_UNVALIDATED
            if not config.uncertainty_calibrated
            else "preregistered_thresholds_only"
        ),
        "calibration_claim": (
            "none" if not config.uncertainty_calibrated else "not_ece_calibrated"
        ),
        "envelope_px": float(config.envelope_px),
        "envelope_policy": ENVELOPE_POLICY_DEGRADE,
        "envelope_degraded": (
            envelope_flags.reshape(leading) if leading else bool(envelope_flags[0])
        ),
        "blocked": False,
        "candidate_max_norm_px": (
            candidate_norms.reshape(leading) if leading else float(candidate_norms[0])
        ),
        "applied_max_norm_px": (
            applied_norms.reshape(leading) if leading else float(applied_norms[0])
        ),
        "fine_applied": (
            fine_applied.reshape(leading) if leading else bool(fine_applied[0])
        ),
        "coarse_applied": (
            coarse_applied.reshape(leading) if leading else bool(coarse_applied[0])
        ),
        "next_seed_policy": NEXT_SEED_POLICY_NOMINAL,
        "next_seed_is_previous_residual": False,
        "column_wrist_residuals_summed": False,
        "wrist_fallback_used": False,
        "view": str(view),
        "mid_scale": float(config.mid_scale),
        "reasons_per_item": _object_rows(reason_rows, leading, count),
    }
    return GroundingConfidenceFusion(
        corrected_px=corrected,
        applied_scale=scales.reshape(leading) if leading else float(scales[0]),
        tier=tiers.reshape(leading) if leading else str(tiers[0]),
        fallback=fallbacks.reshape(leading) if leading else fallbacks[0],
        reasons=tuple(union_reasons),
        diagnostics=diagnostics,
    )


def _optional_coarse_fine_as_lewm_output() -> Any | None:
    try:
        from fcpg.coarse_fine_fcpg import as_lewm_output as module_fn
    except ImportError:
        return None
    return module_fn if callable(module_fn) else None


def _assert_residual_only_lewm_output(output: Any) -> None:
    fields = set(getattr(type(output), "__dataclass_fields__", {}))
    forbidden = {"reliability", "reliability_logit", "sigma_fine_px", "sigma"}
    leaked = fields & forbidden
    if leaked:
        raise TypeError(
            "LeWMGroundingOutput must remain residual-only; "
            f"found {sorted(leaked)}"
        )


def _as_numpy(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _require_anchor_points(value: Any, name: str) -> np.ndarray:
    array = _as_numpy(value, dtype=np.float64)
    if array is None or array.ndim < 2 or array.shape[-2:] != (3, 2):
        raise ValueError(f"{name} must have shape [..., 3, 2]")
    return np.asarray(array, dtype=np.float64)


def _broadcast_bool(
    value: Any,
    leading: tuple[int, ...],
    count: int,
    *,
    name: str,
) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    array = _as_numpy(value)
    if array is None:
        return np.zeros(count, dtype=bool)
    if array.shape == ():
        return np.full(count, bool(array), dtype=bool)
    if leading and array.shape == leading:
        return array.astype(bool, copy=False).reshape(count)
    if array.shape == leading + (3,) or array.shape == leading + (2,):
        return np.all(array.astype(bool, copy=False), axis=-1).reshape(count)
    if array.size == count:
        return array.astype(bool, copy=False).reshape(count)
    if array.size == 1:
        return np.full(count, bool(array.reshape(())), dtype=bool)
    raise ValueError(f"{name} has an incompatible shape {array.shape}")


def _broadcast_view_valid(
    value: Any,
    leading: tuple[int, ...],
    count: int,
) -> np.ndarray:
    if value is None:
        return np.ones(count, dtype=bool)
    array = _as_numpy(value)
    if array is None:
        return np.ones(count, dtype=bool)
    if array.shape == ():
        return np.full(count, bool(array), dtype=bool)
    if leading and array.shape == leading:
        return array.astype(bool, copy=False).reshape(count)
    if array.shape == leading + (3,):
        tips = array[..., 1] & array[..., 2]
        return tips.astype(bool, copy=False).reshape(count)
    if array.shape[-1] == 3 and array.reshape(-1, 3).shape[0] == count:
        flat = array.reshape(count, 3)
        return np.asarray(flat[:, 1] & flat[:, 2], dtype=bool)
    return _broadcast_bool(value, leading, count, name="view_valid")


def _reduce_score(value: Any, leading: tuple[int, ...], count: int) -> np.ndarray:
    array = _as_numpy(value, dtype=np.float64)
    if array is None:
        return np.zeros(count, dtype=np.float64)
    return _reduce_leading(array, leading, count, reducer=np.min, fill=0.0)


def _reduce_sigma(value: Any, leading: tuple[int, ...], count: int) -> np.ndarray:
    array = _as_numpy(value, dtype=np.float64)
    if array is None:
        return np.full(count, np.inf, dtype=np.float64)
    return _reduce_leading(array, leading, count, reducer=np.max, fill=np.inf)


def _optional_reduce_min(
    value: Any, leading: tuple[int, ...], count: int
) -> np.ndarray | None:
    if value is None:
        return None
    array = _as_numpy(value, dtype=np.float64)
    if array is None:
        return None
    return _reduce_leading(array, leading, count, reducer=np.min, fill=np.inf)


def _optional_reduce_max(
    value: Any, leading: tuple[int, ...], count: int
) -> np.ndarray | None:
    if array_is_missing(value):
        return None
    array = _as_numpy(value, dtype=np.float64)
    if array is None:
        return None
    return _reduce_leading(array, leading, count, reducer=np.max, fill=-np.inf)


def array_is_missing(value: Any) -> bool:
    return value is None


def _reduce_leading(
    array: np.ndarray,
    leading: tuple[int, ...],
    count: int,
    *,
    reducer,
    fill: float,
) -> np.ndarray:
    if array.shape == ():
        return np.full(count, float(array), dtype=np.float64)
    if leading and array.shape == leading:
        return array.astype(np.float64, copy=False).reshape(count)
    if array.size == count:
        return array.astype(np.float64, copy=False).reshape(count)
    if leading and len(array.shape) > len(leading) and array.shape[: len(leading)] == leading:
        flat = array.reshape(count, -1)
        return reducer(flat, axis=-1)
    if array.size == 1:
        return np.full(count, float(array.reshape(())), dtype=np.float64)
    return np.full(count, fill, dtype=np.float64)


def _usable_last_accepted(
    last_accepted_px: Any,
    *,
    last_accepted_view: str | None,
    view: str,
    leading: tuple[int, ...],
    count: int,
    template: np.ndarray,
) -> tuple[np.ndarray | None, bool]:
    wrist_forbidden = False
    if last_accepted_px is None:
        return None, False
    if last_accepted_view == VIEW_WRIST and view != VIEW_WRIST:
        return None, True
    if last_accepted_view is not None and last_accepted_view != view:
        if last_accepted_view == VIEW_WRIST:
            return None, True
        return None, False
    points = _as_numpy(last_accepted_px, dtype=np.float64)
    if points is None or points.shape[-2:] != (3, 2):
        return None, wrist_forbidden
    if points.shape == (3, 2):
        stacked = np.broadcast_to(points, (count, 3, 2)).copy()
    elif leading and points.shape == leading + (3, 2):
        stacked = points.reshape(count, 3, 2)
    elif points.shape == template.shape:
        stacked = points.reshape(count, 3, 2)
    else:
        raise ValueError("last_accepted_px must match nominal_px or be [3, 2]")
    usable = np.empty_like(stacked)
    any_valid = False
    for index in range(count):
        if np.all(np.isfinite(stacked[index])):
            usable[index] = stacked[index]
            any_valid = True
        else:
            usable[index] = np.nan
    if not any_valid:
        return None, wrist_forbidden
    return usable, wrist_forbidden


def _anchor_norms(delta: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(delta, dtype=np.float64), axis=-1)


def _rederive_centers(points: np.ndarray) -> np.ndarray:
    out = np.asarray(points, dtype=np.float64).copy()
    out[..., _CENTER_INDEX, :] = 0.5 * (
        out[..., _TIP_SLICES[0], :] + out[..., _TIP_SLICES[1], :]
    )
    return out


def _object_rows(
    rows: list[list[str]],
    leading: tuple[int, ...],
    count: int,
) -> Any:
    packed = np.empty(count, dtype=object)
    for index, row in enumerate(rows):
        packed[index] = tuple(row)
    if leading:
        return packed.reshape(leading)
    return tuple(rows[0])


def _unique_preserve(codes: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered


def residual_only_lewm_field_names() -> frozenset[str]:
    from fcpg.transformer_grounding import LeWMGroundingOutput

    return frozenset(LeWMGroundingOutput.__dataclass_fields__)


def jsonable_fusion_value(value: Any) -> Any:
    """JSON-safe view of fusion diagnostics. Tensors never leak."""

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {str(key): jsonable_fusion_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_fusion_value(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [jsonable_fusion_value(item) for item in value.tolist()]
        if value.shape == ():
            item = value.item()
            return None if item is None else jsonable_fusion_value(item)
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


__all__ = [
    "COARSE_FINE_VARIANT",
    "ConfidenceFusionConfig",
    "ConfidenceTier",
    "CoarseFineGroundingLike",
    "ENVELOPE_POLICY_BLOCK",
    "ENVELOPE_POLICY_DEGRADE",
    "FALLBACK_LAST_ACCEPTED",
    "FALLBACK_NOMINAL",
    "FCPG_TRAINING_ENVELOPE_PX",
    "FallbackSource",
    "GROUNDING_GATED_FCPG",
    "GROUNDING_NOMINAL",
    "GroundingConfidenceFusion",
    "GroundingConfidenceSidecar",
    "GroundingFusionState",
    "GroundingRankingContract",
    "LEGACY_CORRECTOR_VARIANTS",
    "POLICY_UNVALIDATED",
    "RANKING_OFF",
    "RANKING_ON",
    "TIER_HIGH",
    "TIER_LOW",
    "TIER_MID",
    "VIEW_COLUMN",
    "VIEW_WRIST",
    "as_residual_only_lewm_output",
    "corrected_under_contract",
    "envelope_policy_for_variant",
    "fuse_from_coarse_fine_output",
    "fuse_grounding_confidence",
    "is_coarse_fine_output",
    "iter_grounding_ranking_grid",
    "jsonable_fusion_value",
    "next_correction_seed_px",
    "planner_envelope_should_block",
    "residual_only_lewm_field_names",
    "resolve_fusion_config",
]
