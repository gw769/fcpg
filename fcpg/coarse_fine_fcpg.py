"""Offline coarse-to-fine FCPG: global pair match plus local residual.

This module is grounding-only. It does not touch LeWM dynamics, does not take a
structured goal, and does not treat M1/FK projections as pixel GT. Physical
labels are not produced here.

Composition
-----------
``StructuredGlobalCoarseMatcher`` scores two fingertips on a dense feature map
and decodes them as a pair. ``LeWMAnchorResidualCorrector(num_keypoints=2)``
then predicts a *local* residual around ``stopgrad(q_coarse_tips)``. Center is
always the fingertip midpoint. Fine sigma/reliability live in
``FineUncertaintySidecar`` and are never written onto ``LeWMGroundingOutput``.

Identity init
-------------
Coarse queries start at zero, so score maps are uniform and the pair is
rejected. The fine residual head keeps its last-layer zeros. Rejected writes
are gated to zero, so an untrained module returns exact nominal tips.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.modules import sample_feature_map_at_pixels
from fcpg.so101_global_tip_locator import feature_patch_centers
from fcpg.transformer_grounding import LeWMAnchorResidualCorrector, LeWMGroundingOutput


CENTER_INDEX = 0
FIXED_INDEX = 1
MOVING_INDEX = 2
TIP_INDICES = (FIXED_INDEX, MOVING_INDEX)
LEFT_INDEX = FIXED_INDEX
RIGHT_INDEX = MOVING_INDEX
ANCHOR_NAMES = ("center", "fixed", "moving")
ANCHOR_ALIASES = ("center", "left", "right")

_FORBIDDEN_PROVENANCE_KEYS = frozenset(
    {
        "global_tip_checkpoint",
        "global_tip_path",
        "global_tip_head",
        "locator_checkpoint",
        "locator_path",
        "planner_seed",
        "correction_seed_px",
        "provenance",
        "global_tip_state_dict",
    }
)
_GLOBAL_TIP_FORMAT = "so101-global-tip-heatmap-v1"
# Pair scores are [B, K, K]. Cap K at construction so memory cannot explode.
_MAX_TOP_K_PAIRS = 512


@dataclass(frozen=True)
class CoarseFineGroundingOutput:
    """Coarse pair match plus fine residual, with an uncalibrated sidecar."""

    delta_coarse_px: torch.Tensor
    delta_fine_px: torch.Tensor
    delta_px: torch.Tensor
    q_coarse_px: torch.Tensor
    q_corrected_px: torch.Tensor
    p_hat_px: torch.Tensor
    coarse_score: torch.Tensor
    peak_margin: torch.Tensor
    entropy: torch.Tensor
    accept_coarse: torch.Tensor
    sigma_fine_px: torch.Tensor
    reliability: torch.Tensor
    uncertainty_calibrated: bool
    keypoint_latent: torch.Tensor
    score_maps: torch.Tensor
    feature_map: torch.Tensor | None = None
    global_feature: torch.Tensor | None = None


@dataclass(frozen=True)
class CoarseMatchResult:
    """Two-tip global match. Center is never an independent channel."""

    tips_px: torch.Tensor
    score: torch.Tensor
    peak_margin: torch.Tensor
    entropy: torch.Tensor
    accept: torch.Tensor
    left_score: torch.Tensor
    right_score: torch.Tensor
    score_maps: torch.Tensor


class FineUncertaintySidecar(nn.Module):
    """Uncalibrated fine-stage sigma and reliability.

    This is a wrapper around fine keypoint latents. It does not extend
    ``LeWMGroundingOutput`` and must not be reported as ECE-calibrated.
    """

    calibrated = False

    def __init__(self, model_dim: int, min_sigma_px: float = 1.0) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if min_sigma_px <= 0:
            raise ValueError("min_sigma_px must be positive")
        self.model_dim = int(model_dim)
        self.min_sigma_px = float(min_sigma_px)
        hidden = max(self.model_dim // 2, 1)
        self.sigma_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.sigma_head[-1].weight)
        nn.init.zeros_(self.sigma_head[-1].bias)
        nn.init.zeros_(self.reliability_head[-1].weight)
        nn.init.zeros_(self.reliability_head[-1].bias)

    def forward(
        self, tip_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tip_latent.ndim != 3 or tip_latent.shape[-1] != self.model_dim:
            raise ValueError(
                f"tip_latent must have shape [B,2,{self.model_dim}], "
                f"got {tuple(tip_latent.shape)}"
            )
        if tip_latent.shape[1] != 2:
            raise ValueError("sidecar consumes two fingertip latents only")
        sigma_tips = F.softplus(self.sigma_head(tip_latent).squeeze(-1))
        sigma_tips = sigma_tips + self.min_sigma_px
        reliability_tips = torch.sigmoid(self.reliability_head(tip_latent).squeeze(-1))
        sigma_tips = torch.nan_to_num(
            sigma_tips,
            nan=self.min_sigma_px,
            posinf=1.0e6,
            neginf=self.min_sigma_px,
        ).clamp_min(self.min_sigma_px)
        reliability_tips = torch.nan_to_num(
            reliability_tips, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        return sigma_tips, reliability_tips


class StructuredGlobalCoarseMatcher(nn.Module):
    """Full-image / large-range matcher that predicts two fingertips only.

    M1/FK projections are a query hint, left/right order, and a soft jaw-axis
    prior. They are never pixel GT. Separation is not regressed onto FK; the
    only hard geometry gate is a configurable pulse-derived floor plus
    same-cell collapse / edge / low-peak rejection.

    Independent three-channel heatmap decode is intentionally absent: a joint
    top-k pair search scores (fixed, moving) together. Pair translation of
    tens of pixels is allowed when appearance and axis/order agree.
    """

    def __init__(
        self,
        visual_dim: int,
        top_k_pairs: int = 24,
        axis_prior_weight: float = 0.25,
        order_prior_weight: float = 0.25,
        separation_floor_px: float = 8.0,
        edge_margin_px: float = 2.0,
        min_tip_score: float = 0.35,
        min_peak_margin: float = 0.12,
        max_entropy: float | None = None,
        entropy_uniform_fraction: float = 0.85,
        entropy_temperature: float = 0.08,
    ) -> None:
        super().__init__()
        if visual_dim <= 0:
            raise ValueError("visual_dim must be positive")
        if top_k_pairs <= 0:
            raise ValueError("top_k_pairs must be positive")
        if int(top_k_pairs) > _MAX_TOP_K_PAIRS:
            raise ValueError(
                f"top_k_pairs must be <= {_MAX_TOP_K_PAIRS} to bound K^2 pair memory"
            )
        if separation_floor_px <= 0:
            raise ValueError("separation_floor_px must be positive")
        if edge_margin_px < 0:
            raise ValueError("edge_margin_px must be non-negative")
        if min_tip_score < 0 or min_peak_margin < 0:
            raise ValueError("score gates must be non-negative")
        if max_entropy is not None and max_entropy <= 0:
            raise ValueError("max_entropy must be positive when set")
        if not 0.0 < entropy_uniform_fraction < 1.0:
            raise ValueError("entropy_uniform_fraction must be in (0, 1)")
        if entropy_temperature <= 0:
            raise ValueError("entropy_temperature must be positive")
        self.visual_dim = int(visual_dim)
        self.top_k_pairs = int(top_k_pairs)
        self.axis_prior_weight = float(axis_prior_weight)
        self.order_prior_weight = float(order_prior_weight)
        self.separation_floor_px = float(separation_floor_px)
        self.edge_margin_px = float(edge_margin_px)
        self.min_tip_score = float(min_tip_score)
        self.min_peak_margin = float(min_peak_margin)
        self.max_entropy = None if max_entropy is None else float(max_entropy)
        self.entropy_uniform_fraction = float(entropy_uniform_fraction)
        self.entropy_temperature = float(entropy_temperature)
        self.tip_queries = nn.Parameter(torch.zeros(2, self.visual_dim))
        self.nominal_query_proj = nn.Linear(self.visual_dim, self.visual_dim)
        nn.init.zeros_(self.nominal_query_proj.weight)
        nn.init.zeros_(self.nominal_query_proj.bias)

    def set_separation_floor_px(self, value: float) -> None:
        if value <= 0:
            raise ValueError("separation_floor_px must be positive")
        self.separation_floor_px = float(value)

    def _appearance_queries(
        self,
        feature_map: torch.Tensor,
        q_nominal_tips_px: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        sampled = sample_feature_map_at_pixels(
            feature_map,
            q_nominal_tips_px,
            image_size=image_size,
            padding_mode="border",
        )
        learned = self.tip_queries.to(
            device=feature_map.device, dtype=feature_map.dtype
        ).unsqueeze(0)
        learned = learned.expand(feature_map.shape[0], -1, -1)
        return learned + self.nominal_query_proj(sampled)

    def _score_maps(self, feature_map: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        features = F.normalize(feature_map, dim=1, eps=1.0e-6)
        query_norm = queries.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        queries = queries / query_norm
        return torch.einsum("bnc,bchw->bnhw", queries, features)

    def forward(
        self,
        feature_map: torch.Tensor,
        q_nominal_tips_px: torch.Tensor,
        image_size: tuple[int, int],
    ) -> CoarseMatchResult:
        if feature_map.ndim != 4:
            raise ValueError(
                f"feature_map must have shape [B,C,H,W], got {tuple(feature_map.shape)}"
            )
        if feature_map.shape[1] != self.visual_dim:
            raise ValueError(
                f"feature_map channel {feature_map.shape[1]} != visual_dim {self.visual_dim}"
            )
        if feature_map.shape[-1] < 2 or feature_map.shape[-2] < 2:
            raise ValueError("feature map spatial size must be at least 2x2")
        batch = feature_map.shape[0]
        if q_nominal_tips_px.shape != (batch, 2, 2):
            raise ValueError(
                "q_nominal_tips_px must have shape [B,2,2] as [fixed,moving]"
            )
        if q_nominal_tips_px.device != feature_map.device:
            raise ValueError("q_nominal_tips_px must share device with feature_map")
        if not torch.isfinite(q_nominal_tips_px).all():
            raise ValueError("q_nominal_tips_px must be finite")
        image_size = _validate_image_size(image_size)

        queries = self._appearance_queries(feature_map, q_nominal_tips_px, image_size)
        score_maps = self._score_maps(feature_map, queries)
        decoded = self._decode_pairs(
            score_maps, q_nominal_tips_px, image_size=image_size
        )
        return decoded

    def _decode_pairs(
        self,
        score_maps: torch.Tensor,
        q_nominal_tips_px: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> CoarseMatchResult:
        batch, _, feat_h, feat_w = score_maps.shape
        cells = feat_h * feat_w
        top_k = min(self.top_k_pairs, cells)
        centers = feature_patch_centers(
            feat_h,
            feat_w,
            image_size=image_size,
            device=score_maps.device,
            dtype=score_maps.dtype,
        ).reshape(cells, 2)

        left_flat = score_maps[:, 0].reshape(batch, cells)
        right_flat = score_maps[:, 1].reshape(batch, cells)
        left_scores, left_index = left_flat.topk(top_k, dim=-1)
        right_scores, right_index = right_flat.topk(top_k, dim=-1)
        left_points = centers[left_index]
        right_points = centers[right_index]

        appearance = left_scores.unsqueeze(2) + right_scores.unsqueeze(1)
        offsets = right_points.unsqueeze(1) - left_points.unsqueeze(2)
        separation = torch.linalg.vector_norm(offsets, dim=-1)
        same_cell = left_index.unsqueeze(2) == right_index.unsqueeze(1)

        fk_offset = q_nominal_tips_px[:, 1] - q_nominal_tips_px[:, 0]
        fk_axis = F.normalize(fk_offset, dim=-1, eps=1.0e-6)
        pair_axis = F.normalize(offsets, dim=-1, eps=1.0e-6)
        axis_cos = (pair_axis * fk_axis[:, None, None, :]).sum(dim=-1)
        pair_score = (
            appearance
            + self.axis_prior_weight * axis_cos.abs()
            + self.order_prior_weight * axis_cos
        )

        width, height = image_size
        left_edge = _near_image_edge(left_points, width, height, self.edge_margin_px)
        right_edge = _near_image_edge(right_points, width, height, self.edge_margin_px)
        invalid = (
            same_cell
            | (separation < self.separation_floor_px)
            | left_edge.unsqueeze(2)
            | right_edge.unsqueeze(1)
        )
        neg_large = torch.finfo(pair_score.dtype).min / 8.0
        masked = pair_score.masked_fill(invalid, neg_large)
        flat = masked.reshape(batch, top_k * top_k)
        best_values, best_flat = flat.max(dim=-1)
        second = _second_best_pair_values(flat, best_values, neg_large)
        has_second = second > (neg_large / 2.0)
        # A uniquely valid pair is a sure winner. Zero margin would fail-closed
        # on the only identifiable pair (including top_k=1).
        unique_margin = best_values.new_full(
            best_values.shape, torch.finfo(best_values.dtype).max / 8.0
        )
        peak_margin = torch.where(has_second, best_values - second, unique_margin)

        valid_mask = flat > (neg_large / 2.0)
        # Pair-softmax entropy is a poor uniform detector: many half-correct
        # pairs keep it high even when each tip map is a single peak. Gate on
        # spatial entropy of the two tip maps instead; report that value.
        temperature = self.entropy_temperature
        left_log = torch.log_softmax(left_flat / temperature, dim=-1)
        right_log = torch.log_softmax(right_flat / temperature, dim=-1)
        entropy = 0.5 * (
            -(left_log.exp() * left_log).sum(dim=-1)
            + -(right_log.exp() * right_log).sum(dim=-1)
        )

        best_left = torch.div(best_flat, top_k, rounding_mode="floor")
        best_right = best_flat.remainder(top_k)
        batch_index = torch.arange(batch, device=score_maps.device)
        tips = torch.stack(
            (
                left_points[batch_index, best_left],
                right_points[batch_index, best_right],
            ),
            dim=1,
        )
        left_score = left_scores[batch_index, best_left]
        right_score = right_scores[batch_index, best_right]
        pair_appearance = left_score + right_score

        decoded_collapse = best_left_cell_collapse(
            left_index[batch_index, best_left],
            right_index[batch_index, best_right],
        )
        decoded_edge = _near_image_edge(tips, width, height, self.edge_margin_px).any(
            dim=-1
        )
        decoded_sep = torch.linalg.vector_norm(tips[:, 1] - tips[:, 0], dim=-1)
        n_cells = entropy.new_tensor(float(cells))
        if self.max_entropy is None:
            entropy_limit = self.entropy_uniform_fraction * n_cells.clamp_min(2).log()
        else:
            entropy_limit = entropy.new_full((batch,), self.max_entropy)
        has_valid = valid_mask.any(dim=-1)
        # Collapse is pair-level only. Independent argmax coincidence is the
        # near-closed similar-finger case the top-k pair search is for.
        accept = (
            has_valid
            & (~decoded_collapse)
            & (~decoded_edge)
            & (decoded_sep >= self.separation_floor_px)
            & (left_score >= self.min_tip_score)
            & (right_score >= self.min_tip_score)
            & (peak_margin >= self.min_peak_margin)
            & (entropy <= entropy_limit)
            & torch.isfinite(pair_appearance)
            & torch.isfinite(left_score)
            & torch.isfinite(right_score)
            & torch.isfinite(entropy)
            & torch.isfinite(tips).all(dim=(-2, -1))
        )
        tips = torch.where(accept[:, None, None], tips, q_nominal_tips_px)
        return CoarseMatchResult(
            tips_px=tips,
            score=pair_appearance,
            peak_margin=peak_margin,
            entropy=entropy,
            accept=accept,
            left_score=left_score,
            right_score=right_score,
            score_maps=score_maps,
        )


def _allclose_fp32(
    left: torch.Tensor, right: torch.Tensor, *, atol: float
) -> bool:
    return bool(torch.allclose(left.float(), right.float(), atol=atol, rtol=0.0))


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("image_size must be (width, height)")
    width = int(image_size[0])
    height = int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    return width, height


def _second_best_pair_values(
    flat: torch.Tensor,
    best_values: torch.Tensor,
    neg_large: float,
) -> torch.Tensor:
    if flat.shape[-1] >= 2:
        return flat.topk(2, dim=-1).values[:, 1]
    return best_values.new_full(best_values.shape, neg_large)


def _near_image_edge(
    points_px: torch.Tensor,
    width: int,
    height: int,
    margin_px: float,
) -> torch.Tensor:
    x = points_px[..., 0]
    y = points_px[..., 1]
    return (
        (x < margin_px)
        | (y < margin_px)
        | (x > float(width) - 1.0 - margin_px)
        | (y > float(height) - 1.0 - margin_px)
    )


def best_left_cell_collapse(left_cell: torch.Tensor, right_cell: torch.Tensor) -> torch.Tensor:
    return left_cell == right_cell


def derive_center_from_tips(tips_px: torch.Tensor) -> torch.Tensor:
    """Replace / build the center as the exact midpoint of two tips."""

    if tips_px.ndim < 2 or tips_px.shape[-1] != 2 or tips_px.shape[-2] != 2:
        raise ValueError("tips_px must end with shape [2,2]")
    center = tips_px.mean(dim=-2, keepdim=True)
    return torch.cat((center, tips_px), dim=-2)


def _triple_from_tips(tips_px: torch.Tensor) -> torch.Tensor:
    return derive_center_from_tips(tips_px)


def as_lewm_output(output: CoarseFineGroundingOutput) -> LeWMGroundingOutput:
    """Adapt to the residual-only dataclass. Old fields only."""

    return LeWMGroundingOutput(
        delta_px=output.delta_px,
        q_corrected_px=output.q_corrected_px,
        keypoint_latent=output.keypoint_latent,
        feature_map=output.feature_map,
        global_feature=output.global_feature,
    )


def _reject_forbidden_kwargs(kwargs: dict[str, Any]) -> None:
    bad = _FORBIDDEN_PROVENANCE_KEYS.intersection(kwargs)
    if bad:
        raise ValueError(
            "No-Go: global-tip checkpoint/provenance cannot be injected: "
            + ", ".join(sorted(bad))
        )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected CoarseFineFCPG arguments: {unknown}")


class CoarseFineFCPG(nn.Module):
    """Global two-tip matcher plus local two-tip residual corrector.

    ``forward(feature_map, q_nominal_px, keypoints_xyz, robot_obs, image_size)``
    keeps the existing corrector contract. Coarse may move more than 16 px.
    Fine still uses ``local_radius_px=16`` and ``max_residual_px=20``.
    """

    def __init__(
        self,
        visual_dim: int,
        obs_dim: int,
        model_dim: int = 256,
        local_grid_size: int = 5,
        local_radius_px: float = 16.0,
        local_layers: int = 4,
        global_layers: int = 2,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_residual_px: float = 20.0,
        top_k_pairs: int = 24,
        axis_prior_weight: float = 0.25,
        order_prior_weight: float = 0.25,
        separation_floor_px: float = 8.0,
        edge_margin_px: float = 2.0,
        min_tip_score: float = 0.35,
        min_peak_margin: float = 0.12,
        max_entropy: float | None = None,
        entropy_temperature: float = 0.08,
        min_sigma_px: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _reject_forbidden_kwargs(kwargs)
        self.visual_dim = int(visual_dim)
        self.obs_dim = int(obs_dim)
        self.num_keypoints = 3
        self.model_dim = int(model_dim)
        self.local_radius_px = float(local_radius_px)
        self.max_residual_px = float(max_residual_px)
        self.coarse = StructuredGlobalCoarseMatcher(
            visual_dim=self.visual_dim,
            top_k_pairs=top_k_pairs,
            axis_prior_weight=axis_prior_weight,
            order_prior_weight=order_prior_weight,
            separation_floor_px=separation_floor_px,
            edge_margin_px=edge_margin_px,
            min_tip_score=min_tip_score,
            min_peak_margin=min_peak_margin,
            max_entropy=max_entropy,
            entropy_temperature=entropy_temperature,
        )
        self.fine_corrector = LeWMAnchorResidualCorrector(
            visual_dim=self.visual_dim,
            obs_dim=self.obs_dim,
            num_keypoints=2,
            model_dim=self.model_dim,
            local_grid_size=local_grid_size,
            local_radius_px=self.local_radius_px,
            local_layers=local_layers,
            global_layers=global_layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_residual_px=self.max_residual_px,
        )
        self.fine_uncertainty = FineUncertaintySidecar(
            model_dim=self.model_dim,
            min_sigma_px=min_sigma_px,
        )
        self._assert_nogo_construction()

    def _assert_nogo_construction(self) -> None:
        residual_heads = [
            module
            for module in self.modules()
            if type(module) is LeWMAnchorResidualCorrector
        ]
        if len(residual_heads) != 1:
            raise RuntimeError(
                "No-Go: two full residual heads must not be composed; "
                "fine is the only LeWMAnchorResidualCorrector"
            )
        if self.fine_corrector.num_keypoints != 2:
            raise RuntimeError(
                "No-Go: independent three-anchor / center prediction is forbidden"
            )
        if isinstance(self.coarse, LeWMAnchorResidualCorrector):
            raise RuntimeError("No-Go: coarse must not be a second full residual head")
        if hasattr(self, "center_head") or hasattr(self.coarse, "center_head"):
            raise RuntimeError("No-Go: independent center prediction is forbidden")
        if hasattr(self.coarse, "fk_separation_head"):
            raise RuntimeError("No-Go: FK separation hard regression is forbidden")

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[no-untyped-def]
        if isinstance(state_dict, dict) and state_dict.get("format") == _GLOBAL_TIP_FORMAT:
            raise ValueError(
                "No-Go: global-tip checkpoint/provenance cannot be injected"
            )
        keys = state_dict.keys() if isinstance(state_dict, dict) else ()
        if any("global_tip" in str(key) for key in keys):
            raise ValueError(
                "No-Go: global-tip checkpoint/provenance cannot be injected"
            )
        return super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        feature_map: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> CoarseFineGroundingOutput:
        if feature_map.ndim != 4:
            raise ValueError(
                f"feature_map must have shape [B,C,H,W], got {tuple(feature_map.shape)}"
            )
        batch = feature_map.shape[0]
        if q_nominal_px.shape != (batch, 3, 2):
            raise ValueError(
                "q_nominal_px must use order [center,fixed/left,moving/right] "
                "with shape [B,3,2]"
            )
        if keypoints_xyz.shape != (batch, 3, 3):
            raise ValueError("keypoints_xyz must have shape [B,3,3]")
        if robot_obs.shape != (batch, self.obs_dim):
            raise ValueError(
                f"robot_obs must have shape {(batch, self.obs_dim)}, "
                f"got {tuple(robot_obs.shape)}"
            )
        if q_nominal_px.device != feature_map.device:
            raise ValueError("q_nominal_px must share device with feature_map")
        if keypoints_xyz.device != feature_map.device:
            raise ValueError("keypoints_xyz must share device with feature_map")
        if robot_obs.device != feature_map.device:
            raise ValueError("robot_obs must share device with feature_map")
        if not torch.isfinite(q_nominal_px).all():
            raise ValueError("q_nominal_px must be finite")
        image_size = _validate_image_size(image_size)

        nominal_tips = q_nominal_px[:, list(TIP_INDICES)]
        coarse = self.coarse(feature_map, nominal_tips, image_size)
        accept = coarse.accept
        q_coarse_tips = torch.where(accept[:, None, None], coarse.tips_px, nominal_tips)
        q_coarse_px = torch.where(
            accept[:, None, None],
            _triple_from_tips(q_coarse_tips),
            q_nominal_px,
        )
        delta_coarse_px = q_coarse_px - q_nominal_px

        # Fine always runs on the full batch (LayerNorm, no BatchNorm).
        # Residual writes are accept-gated below so reject cannot train fine
        # through p_hat. Sidecar/latents stay computed; consumers must mask.
        fine_centers = q_coarse_tips.detach()
        fine = self.fine_corrector(
            feature_map,
            fine_centers,
            keypoints_xyz[:, list(TIP_INDICES)],
            robot_obs,
            image_size,
        )
        fine_delta_tips = torch.where(
            accept[:, None, None],
            fine.delta_px,
            torch.zeros_like(fine.delta_px),
        )
        delta_fine_px = torch.where(
            accept[:, None, None],
            _triple_from_tips(fine_delta_tips),
            torch.zeros(batch, 3, 2, device=q_nominal_px.device, dtype=q_nominal_px.dtype),
        )
        p_hat_tips = nominal_tips + delta_coarse_px[:, list(TIP_INDICES)] + fine_delta_tips
        p_hat_px = torch.where(
            accept[:, None, None],
            _triple_from_tips(p_hat_tips),
            q_nominal_px,
        )
        delta_px = p_hat_px - q_nominal_px

        tip_latent = fine.keypoint_latent
        keypoint_latent = torch.cat(
            (tip_latent.mean(dim=1, keepdim=True), tip_latent), dim=1
        )
        sigma_tips, reliability_tips = self.fine_uncertainty(tip_latent)
        sigma_fine_px = torch.cat(
            (sigma_tips.mean(dim=1, keepdim=True), sigma_tips), dim=1
        )
        reliability = torch.cat(
            (reliability_tips.mean(dim=1, keepdim=True), reliability_tips), dim=1
        )
        coarse_score = torch.stack(
            (
                coarse.score,
                coarse.left_score,
                coarse.right_score,
            ),
            dim=-1,
        )
        output = CoarseFineGroundingOutput(
            delta_coarse_px=delta_coarse_px,
            delta_fine_px=delta_fine_px,
            delta_px=delta_px,
            q_coarse_px=q_coarse_px,
            q_corrected_px=p_hat_px,
            p_hat_px=p_hat_px,
            coarse_score=coarse_score,
            peak_margin=coarse.peak_margin,
            entropy=coarse.entropy,
            accept_coarse=accept,
            sigma_fine_px=sigma_fine_px,
            reliability=reliability,
            uncertainty_calibrated=False,
            keypoint_latent=keypoint_latent,
            score_maps=coarse.score_maps,
            feature_map=feature_map,
            global_feature=None,
        )
        _assert_nogo_output(output, q_nominal_px, fine_centers)
        return output


def _assert_nogo_output(
    output: CoarseFineGroundingOutput,
    q_nominal_px: torch.Tensor,
    fine_centers: torch.Tensor,
) -> None:
    recomposed = q_nominal_px + output.delta_coarse_px + output.delta_fine_px
    if not _allclose_fp32(output.p_hat_px, recomposed, atol=1.0e-4):
        raise RuntimeError(
            "No-Go: two full residual heads must not be added; "
            "p_hat = p_nominal + delta_coarse + delta_fine"
        )
    if not _allclose_fp32(
        output.delta_fine_px,
        output.p_hat_px - output.q_coarse_px,
        atol=1.0e-4,
    ):
        raise RuntimeError(
            "No-Go: fine must be a residual around coarse, not a second full-r"
        )
    accept = output.accept_coarse
    if bool(accept.any()):
        accepted = output.q_corrected_px[accept]
        midpoint = accepted[:, 1:].mean(dim=1)
        if not _allclose_fp32(accepted[:, 0], midpoint, atol=1.0e-5):
            raise RuntimeError(
                "No-Go: independent center prediction is forbidden; "
                "center is the tip midpoint"
            )
        accepted_coarse = output.q_coarse_px[accept]
        if not _allclose_fp32(
            accepted_coarse[:, 0],
            accepted_coarse[:, 1:].mean(dim=1),
            atol=1.0e-5,
        ):
            raise RuntimeError("No-Go: coarse center must be the tip midpoint")
        if not _allclose_fp32(
            fine_centers[accept],
            output.q_coarse_px[accept][:, 1:],
            atol=1.0e-5,
        ):
            raise RuntimeError("No-Go: fine sampling center must be stopgrad(q_coarse)")
        if fine_centers.requires_grad:
            raise RuntimeError("No-Go: fine sampling center must be stopgrad(q_coarse)")
    if bool((~accept).any()):
        rejected = ~accept
        if not _allclose_fp32(
            output.q_corrected_px[rejected],
            q_nominal_px[rejected],
            atol=0.0,
        ):
            raise RuntimeError(
                "No-Go: coarse reject must return nominal; fine must not write"
            )
        if not torch.equal(
            output.delta_fine_px[rejected],
            torch.zeros_like(output.delta_fine_px[rejected]),
        ):
            raise RuntimeError("No-Go: coarse reject gates fine writes to zero")
    if output.uncertainty_calibrated:
        raise RuntimeError("No-Go: sidecar sigma is uncalibrated and must not claim ECE")


__all__ = [
    "ANCHOR_ALIASES",
    "ANCHOR_NAMES",
    "CENTER_INDEX",
    "CoarseFineFCPG",
    "CoarseFineGroundingOutput",
    "CoarseMatchResult",
    "FIXED_INDEX",
    "FineUncertaintySidecar",
    "LEFT_INDEX",
    "MOVING_INDEX",
    "RIGHT_INDEX",
    "StructuredGlobalCoarseMatcher",
    "TIP_INDICES",
    "as_lewm_output",
    "derive_center_from_tips",
]
