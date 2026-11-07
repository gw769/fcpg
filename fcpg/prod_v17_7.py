"""Prod-V17.7 camera-aware, geometry-anchored predictive belief controller.

V17.7 keeps the five V17.6.2 operations (``measure``, ``assign_identity``,
``predict``, ``update``, ``act``) and their tensor contracts.  It changes what
each operation is allowed to know, in response to the V17.6.2 design review:

* ``predict`` (the Grounding Dynamics Model, module name ``lewm``) receives the
  camera-rig identity of the row (archive dataset id, the fish rig, or ``0``
  for an unknown rig) and, when a calibration exists, the analytic FK-projected
  displacement of both anchors over the command chunk.  With an analytic mean
  the network emits only a bounded residual; without one it emits the full
  learned shift exactly as V17.6.2 did.
* ``update`` may fuse a *broadened* geometric prior at H0 when a calibration
  exists.  The geometry is mixed with the factorized-neutral distribution so it
  can bias, but never veto, the current image.
* Belief transport preserves probability mass under process blur.  Only a
  displacement that leaves the frame converts visible mass into ``absent``.
* The measurement optimizer may add a bounded separation penalty on the
  Bhattacharyya overlap of the two hypotheses when their labels are distinct.
* The Actor has three matched modes that share one decoder and one parameter
  set: ``full`` (belief + innovation), ``memoryless`` (scene + q only) and
  ``raw_history`` (scene + q + pooled past scenes/commands/readback).  The
  modes exist so that "structured belief helps" can be separated from "any
  memory helps".

Transformer layers, DINOv2, dual views, histogram filtering and set losses are
standard tools, not individual novelty claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.prod_v17_6 import (
    ABSENT_INDEX,
    ACTOR_DIM,
    ANCHORS,
    BELIEF_CLASSES,
    COMMAND_STEPS,
    COMPACT_DIM,
    FEATURE_DIM,
    GRID_CELLS,
    GRID_SIZE,
    HYPOTHESES,
    IMAGE_SIZE,
    PATH_STEPS,
    STATE_DIM,
    VIEWS,
    VIEW_ORDER,
    VIT_DIM,
    VIT_PATCHES,
    V176Actor,
    V176ActorOutput,
    V176AssignedMeasurement,
    V176BeliefLeWM,
    V176Config,
    V176ContractError,
    V176Core,
    V176ExchangeableMeasurement,
    V176FilterUpdate,
    V176GroundedState,
    V176GroundingEncoder,
    V176Innovation,
    V176InnovationEncoder,
    V176LeWMPrediction,
    V176Measurement,
    _anisotropic_gaussian_blur,
    _bool,
    _finite,
    _forward_splat,
    _normalized_log_belief,
    _project_action_chunk,
    _shape,
    action_chunk_l1,
    belief_prediction_nll,
    belief_statistics,
    compute_innovation,
    exchangeable_pair_nll,
    fuse_measurement_prior,
    identity_or_track_assignment,
    normalize_actor_scene_tokens,
    posterior_weighted_pool,
    propagate_belief,
    state_path_smooth_l1,
    uniform_spatial_prior,
)


ACTOR_MODES: Final = ("full", "memoryless", "raw_history")
SOURCE_UNKNOWN: Final = 0
HISTORY_STEPS: Final = 3
HISTORY_TOKENS_PER_STEP: Final = VIEWS + 1


class V177ContractError(V176ContractError):
    """Raised when a tensor violates the public V17.7 contract."""


# The measurement, grounding encoder and innovation encoder are unchanged.
V177ExchangeableMeasurement = V176ExchangeableMeasurement
V177GroundingEncoder = V176GroundingEncoder
V177InnovationEncoder = V176InnovationEncoder
V177Measurement = V176Measurement
V177AssignedMeasurement = V176AssignedMeasurement
V177GroundedState = V176GroundedState
V177LeWMPrediction = V176LeWMPrediction
V177Innovation = V176Innovation
V177FilterUpdate = V176FilterUpdate
V177ActorOutput = V176ActorOutput


@dataclass(frozen=True)
class V177Config(V176Config):
    source_vocabulary: int = 1024
    source_dropout: float = 0.2
    analytic_transport: bool = True
    analytic_residual_max_px: float = 24.0
    geometry_prior_fusion: bool = True
    geometry_prior_sigma_px: float = 24.0
    geometry_prior_floor: float = 0.25
    mass_preserving_blur: bool = True
    separation_weight: float = 0.1
    actor_mode: str = "full"
    raw_history_steps: int = HISTORY_STEPS

    def validate(self) -> None:
        super().validate()
        if int(self.source_vocabulary) < 2:
            raise V177ContractError("source_vocabulary must hold unknown plus one rig")
        if not 0.0 <= float(self.source_dropout) < 1.0:
            raise V177ContractError("source_dropout must lie in [0, 1)")
        if not 0.0 < float(self.analytic_residual_max_px) <= float(
            self.max_transport_shift_px
        ):
            raise V177ContractError(
                "analytic_residual_max_px must be positive and within the "
                "transport shift bound"
            )
        if float(self.geometry_prior_sigma_px) <= 0.0:
            raise V177ContractError("geometry_prior_sigma_px must be positive")
        if not 0.0 < float(self.geometry_prior_floor) < 1.0:
            raise V177ContractError("geometry_prior_floor must lie in (0, 1)")
        if float(self.separation_weight) < 0.0:
            raise V177ContractError("separation_weight must be non-negative")
        if self.actor_mode not in ACTOR_MODES:
            raise V177ContractError(f"actor_mode must be one of {ACTOR_MODES}")
        if not 1 <= int(self.raw_history_steps) <= HISTORY_STEPS:
            raise V177ContractError(
                f"raw_history_steps must lie in [1, {HISTORY_STEPS}]"
            )


@dataclass(frozen=True)
class V177RawHistory:
    """Completed past transitions for the ``raw_history`` Actor baseline.

    Slots are ordered oldest to newest and padded to ``HISTORY_STEPS``.  A
    padded slot has ``valid == False`` and arbitrary finite content.
    """

    commands: torch.Tensor
    paths: torch.Tensor
    scene_pooled: torch.Tensor
    valid: torch.Tensor


def empty_raw_history(
    batch: int, *, device: torch.device | str, dtype: torch.dtype = torch.float32
) -> V177RawHistory:
    return V177RawHistory(
        commands=torch.zeros(
            int(batch), HISTORY_STEPS, COMMAND_STEPS, STATE_DIM, device=device, dtype=dtype
        ),
        paths=torch.zeros(
            int(batch), HISTORY_STEPS, PATH_STEPS, STATE_DIM, device=device, dtype=dtype
        ),
        scene_pooled=torch.zeros(
            int(batch), HISTORY_STEPS, VIEWS, VIT_DIM, device=device, dtype=dtype
        ),
        valid=torch.zeros(int(batch), HISTORY_STEPS, dtype=torch.bool, device=device),
    )


def append_raw_history(
    history: V177RawHistory,
    commands: torch.Tensor,
    path: torch.Tensor,
    scene_pooled: torch.Tensor,
) -> V177RawHistory:
    """Shift the oldest slot out and append one completed transition."""

    batch = len(history.valid)
    commands = _shape("history commands", commands, (batch, COMMAND_STEPS, STATE_DIM))
    path = _shape("history path", path, (batch, PATH_STEPS, STATE_DIM))
    scene_pooled = _shape("history scene", scene_pooled, (batch, VIEWS, VIT_DIM))
    return V177RawHistory(
        commands=torch.cat(
            (history.commands[:, 1:], commands[:, None].to(history.commands)), dim=1
        ),
        paths=torch.cat((history.paths[:, 1:], path[:, None].to(history.paths)), dim=1),
        scene_pooled=torch.cat(
            (history.scene_pooled[:, 1:], scene_pooled[:, None].to(history.scene_pooled)),
            dim=1,
        ),
        valid=torch.cat(
            (
                history.valid[:, 1:],
                torch.ones(batch, 1, dtype=torch.bool, device=history.valid.device),
            ),
            dim=1,
        ),
    )


def pool_scene_tokens(scene_tokens: torch.Tensor, view_valid: torch.Tensor) -> torch.Tensor:
    """Mean-pool normalized DINO tokens per view for the raw-history baseline."""

    tokens = _finite("scene_tokens", scene_tokens)
    batch = len(tokens)
    tokens = _shape("scene_tokens", tokens, (batch, VIEWS, VIT_PATCHES, VIT_DIM)).float()
    valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(tokens.device)
    return tokens.mean(dim=2) * valid[:, :, None].to(tokens)


def propagate_belief_v177(
    log_belief: torch.Tensor,
    shift_px: torch.Tensor,
    sigma_px: torch.Tensor,
    *,
    uniform_mix: float = 0.05,
    blur_radius: int = 8,
    mass_preserving: bool = True,
) -> torch.Tensor:
    """Transport probability by forward splat, mass-preserving blur and 5% mix.

    V17.6.2 used a zero-padded blur whose boundary leakage was booked as
    ``absent`` at every step, so a stationary anchor near the image border
    slowly vanished.  Here the blur kernel is renormalized per cell and the
    blurred field is rescaled to the exact post-shift mass.  Only the shift can
    move mass out of the frame.
    """

    if not mass_preserving:
        return propagate_belief(
            log_belief,
            shift_px,
            sigma_px,
            uniform_mix=uniform_mix,
            blur_radius=blur_radius,
        )
    if not math.isclose(float(uniform_mix), 0.05, abs_tol=1.0e-12):
        raise V177ContractError("V17.7 propagation requires a 5 percent mixture")
    belief = _normalized_log_belief("log_belief", log_belief)
    if belief.ndim != 4 or belief.shape[1:3] != (VIEWS, ANCHORS):
        raise V177ContractError("log_belief must be [B,2,2,3137]")
    batch = len(belief)
    shift = _shape("shift_px", shift_px, (batch, VIEWS, ANCHORS, 2)).float().to(belief)
    sigma = _shape("sigma_px", sigma_px, (batch, VIEWS, ANCHORS, 2)).float().to(belief)
    probability = belief.exp()
    count = batch * VIEWS * ANCHORS
    visible = probability[..., :GRID_CELLS].reshape(count, 1, GRID_SIZE, GRID_SIZE)
    pixels_per_cell = float(IMAGE_SIZE) / float(GRID_SIZE)
    shifted, shift_loss = _forward_splat(visible, shift.reshape(count, 2) / pixels_per_cell)
    sigma_cells = sigma.reshape(count, 2) / pixels_per_cell
    blurred, _ = _anisotropic_gaussian_blur(shifted, sigma_cells, radius=int(blur_radius))
    support, _ = _anisotropic_gaussian_blur(
        torch.ones_like(shifted), sigma_cells, radius=int(blur_radius)
    )
    blurred = blurred / support.clamp_min(1.0e-6)
    before = shifted.sum(dim=(1, 2, 3))
    after = blurred.sum(dim=(1, 2, 3))
    blurred = blurred * (before / after.clamp_min(1.0e-12))[:, None, None, None]
    absent = (probability[..., ABSENT_INDEX].reshape(count) + shift_loss).clamp(0.0, 1.0)
    transported = torch.cat((blurred.reshape(count, GRID_CELLS), absent[:, None]), dim=-1)
    transported = transported / transported.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    neutral = torch.cat(
        (
            transported.new_full((count, GRID_CELLS), 0.5 / GRID_CELLS),
            transported.new_full((count, 1), 0.5),
        ),
        dim=-1,
    )
    transported = (1.0 - float(uniform_mix)) * transported + float(uniform_mix) * neutral
    transported = transported / transported.sum(dim=-1, keepdim=True)
    return transported.reshape(batch, VIEWS, ANCHORS, BELIEF_CLASSES).clamp_min(1.0e-12).log()


def broaden_geometry_prior(
    geometry_log_prior: torch.Tensor, *, floor: float
) -> torch.Tensor:
    """Mix a geometric prior with the factorized-neutral distribution.

    The floor guarantees every visible cell and the absent class keep at least
    ``floor`` of their neutral mass, so a wrong calibration can bias but cannot
    delete the image measurement.
    """

    if not 0.0 < float(floor) < 1.0:
        raise V177ContractError("geometry floor must lie in (0, 1)")
    geometry = _normalized_log_belief("geometry_log_prior", geometry_log_prior)
    if geometry.ndim != 4 or geometry.shape[1:] != (VIEWS, ANCHORS, BELIEF_CLASSES):
        raise V177ContractError("geometry_log_prior must be [B,2,2,3137]")
    neutral = uniform_spatial_prior(
        len(geometry), device=geometry.device, dtype=geometry.dtype
    ).exp()
    mixed = (1.0 - float(floor)) * geometry.exp() + float(floor) * neutral
    mixed = mixed / mixed.sum(dim=-1, keepdim=True)
    return mixed.clamp_min(1.0e-12).log()


def fuse_geometry_prior(
    log_belief: torch.Tensor,
    geometry_log_prior: torch.Tensor,
    view_valid: torch.Tensor,
    *,
    floor: float,
) -> torch.Tensor:
    """Product-of-experts between a belief and a broadened geometric prior.

    Views without a current image keep the belief untouched: geometry is a
    prior over the anchor location, and in the absence of a measurement the
    caller decides whether it should become the state.
    """

    belief = _normalized_log_belief("log_belief", log_belief)
    batch = len(belief)
    _shape("log_belief", belief, (batch, VIEWS, ANCHORS, BELIEF_CLASSES))
    broadened = broaden_geometry_prior(geometry_log_prior, floor=floor).to(belief)
    _shape("geometry_log_prior", broadened, (batch, VIEWS, ANCHORS, BELIEF_CLASSES))
    valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(belief.device)
    all_valid = torch.ones(batch, VIEWS, dtype=torch.bool, device=belief.device)
    fused = fuse_measurement_prior(broadened, belief, all_valid)
    return torch.where(valid[:, :, None, None], fused, belief)


def hypothesis_overlap(anchor_hypothesis_log_likelihood: torch.Tensor) -> torch.Tensor:
    """Bhattacharyya coefficient between the two visible-conditional maps."""

    hypotheses = _normalized_log_belief(
        "anchor_hypothesis_log_likelihood", anchor_hypothesis_log_likelihood
    )
    if hypotheses.ndim != 4 or hypotheses.shape[1:] != (VIEWS, HYPOTHESES, BELIEF_CLASSES):
        raise V177ContractError("anchor_hypothesis_log_likelihood must be [B,2,2,3137]")
    # Work in log space.  Adding an epsilon independently to every grid cell
    # before ``sqrt`` creates a resolution-dependent overlap floor (about
    # 0.0031 for a 56x56 map), so even disjoint hypotheses are penalized.
    visible_log = hypotheses[..., :GRID_CELLS]
    conditional_log = visible_log - torch.logsumexp(
        visible_log, dim=-1, keepdim=True
    )
    coefficient = torch.exp(
        0.5 * (conditional_log[:, :, 0] + conditional_log[:, :, 1])
    ).sum(dim=-1)
    return coefficient.clamp(0.0, 1.0)


def hypothesis_separation_penalty(
    anchor_hypothesis_log_likelihood: torch.Tensor,
    target_belief: torch.Tensor,
    view_valid: torch.Tensor,
) -> torch.Tensor:
    """Mean excess overlap on views whose two labels are distinct and visible.

    The exchangeable soft-target CE charges only ``log 2`` when both hypotheses
    output the same bimodal map.  This penalty makes duplicated hypotheses cost
    more without touching the set-invariant CE itself.  The target's own
    Bhattacharyya overlap is subtracted before penalization so adjacent or
    closed-jaw labels are not artificially pushed farther apart.
    """

    hypotheses = _normalized_log_belief(
        "anchor_hypothesis_log_likelihood", anchor_hypothesis_log_likelihood
    )
    batch = len(hypotheses)
    target = _finite("target_belief", target_belief).float().to(hypotheses)
    _shape("target_belief", target, (batch, VIEWS, ANCHORS, BELIEF_CLASSES))
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(hypotheses.device)
    visible_mass = target[..., :GRID_CELLS].sum(dim=-1)
    both_visible = (visible_mass[:, :, 0] > 0.5) & (visible_mass[:, :, 1] > 0.5)
    distinct = (target[:, :, 0] - target[:, :, 1]).abs().amax(dim=-1) > 1.0e-7
    weight = (both_visible & distinct & valid).to(hypotheses)
    predicted_overlap = hypothesis_overlap(hypotheses)
    target_overlap = hypothesis_overlap(target.clamp_min(1.0e-30).log()).detach()
    excess_overlap = (predicted_overlap - target_overlap).clamp_min(0.0)
    return (excess_overlap * weight).sum() / weight.sum().clamp_min(1.0)


def _belief_mean_px_v177(log_belief: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(log_belief.float(), dim=-1)[..., :GRID_CELLS]
    mass = probability.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    axis = torch.arange(GRID_SIZE, device=probability.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    center = torch.stack((xx, yy), dim=-1).reshape(GRID_CELLS, 2) * 4.0 + 1.5
    return torch.einsum("...c,cd->...d", probability / mass, center)


def credible_coverage(
    log_belief: torch.Tensor,
    target_px: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    levels: tuple[float, ...] = (0.5, 0.9),
) -> dict[str, torch.Tensor]:
    """Highest-density credible-set coverage of unordered point labels.

    Hypotheses are aligned to labels by the assignment with the smaller summed
    EPE.  Coverage is measured on the visible-conditional map, so it reports
    spatial calibration given visibility; the absent probability is returned
    separately.  A label is covered at level ``L`` when the mass of cells at
    least as probable as its own cell (ties counted against the model,
    excluding the cell itself) is below ``L``.  ``needed_mass`` is that mass
    plus the label cell's own mass, i.e. the smallest tie-inclusive HPD set
    that contains the label.  Every output is ``[B,2,2]`` (view, anchor).
    """

    belief = _normalized_log_belief("log_belief", log_belief)
    batch = len(belief)
    _shape("log_belief", belief, (batch, VIEWS, ANCHORS, BELIEF_CLASSES))
    points = _finite("target_px", target_px).float().to(belief)
    _shape("target_px", points, (batch, VIEWS, ANCHORS, 2))
    active = _bool("target_valid", target_valid, (batch, VIEWS, ANCHORS)).to(belief.device)
    inside = (
        (points[..., 0] >= 0.0)
        & (points[..., 0] < float(IMAGE_SIZE))
        & (points[..., 1] >= 0.0)
        & (points[..., 1] < float(IMAGE_SIZE))
    )
    active = active & inside
    mean = _belief_mean_px_v177(belief)
    direct = torch.linalg.vector_norm(mean - points, dim=-1).sum(dim=-1)
    swapped = torch.linalg.vector_norm(mean - points.flip(2), dim=-1).sum(dim=-1)
    use_swap = swapped < direct
    aligned = torch.where(use_swap[:, :, None, None], belief.flip(2), belief)
    probability = aligned.exp()
    visible = probability[..., :GRID_CELLS]
    mass = visible.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    conditional = visible / mass
    safe = torch.where(active[..., None], points, torch.zeros_like(points))
    cell = ((safe + 0.5) / 4.0 - 0.5).round().clamp(0.0, GRID_SIZE - 1.0).long()
    index = cell[..., 1] * GRID_SIZE + cell[..., 0]
    target_mass = conditional.gather(-1, index[..., None]).squeeze(-1)
    at_least = (conditional * (conditional >= target_mass[..., None])).sum(dim=-1)
    before_target = (at_least - target_mass).clamp_min(0.0)
    result: dict[str, torch.Tensor] = {
        "needed_mass": at_least,
        "absent_probability": probability[..., ABSENT_INDEX],
        "active": active,
    }
    for level in levels:
        result[f"coverage_at_{int(round(float(level) * 100))}"] = (
            before_target < float(level)
        ).to(belief)
    return result


def analytic_shift_from_pixels(
    start_px: torch.Tensor,
    start_valid: torch.Tensor,
    end_px: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic anchor displacement from projected start and end pixels.

    ``start_valid`` is ``[B,2,2]`` per anchor.  A view is analytically usable
    only when both of its anchors start inside the frame and both projected end
    points are finite; an end point outside the frame is a legitimate shift
    whose mass will leave the image.
    """

    raw_start = torch.as_tensor(start_px).float()
    batch = len(raw_start)
    if tuple(raw_start.shape) != (batch, VIEWS, ANCHORS, 2):
        raise V177ContractError("start_px must be [B,2,2,2]")
    raw_end = torch.as_tensor(end_px).float().to(raw_start)
    if tuple(raw_end.shape) != (batch, VIEWS, ANCHORS, 2):
        raise V177ContractError("end_px must be [B,2,2,2]")
    anchor_valid = _bool("start_valid", start_valid, (batch, VIEWS, ANCHORS)).to(
        raw_start.device
    )
    start_finite = torch.isfinite(raw_start).all(dim=-1)
    end_finite = torch.isfinite(raw_end).all(dim=-1)
    usable = (anchor_valid & start_finite & end_finite).all(dim=-1)
    start = torch.nan_to_num(raw_start, nan=0.0, posinf=0.0, neginf=0.0)
    end = torch.nan_to_num(raw_end, nan=0.0, posinf=0.0, neginf=0.0)
    shift = torch.where(usable[:, :, None, None], end - start, torch.zeros_like(start))
    # ``pixels`` runs under inference mode; clone so the caller's autograd
    # graph never saves an inference tensor.
    return shift.clone(), usable.clone()


class V177BeliefLeWM(V176BeliefLeWM):
    """Camera-aware SO-101 belief transport with an optional analytic mean.

    Six-joint contract and scales are deployment-specific; no cross-arm claim.
    """

    def __init__(self, config: V177Config = V177Config()) -> None:
        if not isinstance(config, V177Config):
            raise V177ContractError("V177BeliefLeWM requires a V177Config")
        super().__init__(config)
        dimension = config.feature_dim
        self.source_embedding = nn.Embedding(int(config.source_vocabulary), dimension)
        self.analytic_status_embedding = nn.Parameter(torch.empty(2, dimension))
        self.analytic_shift_projection = nn.Sequential(
            nn.Linear(2, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        nn.init.normal_(self.source_embedding.weight, std=0.02)
        nn.init.normal_(self.analytic_status_embedding, std=0.02)
        with torch.no_grad():
            self.source_embedding.weight[SOURCE_UNKNOWN].zero_()

    def forward(  # type: ignore[override]
        self,
        grounded: V176GroundedState,
        current_q: torch.Tensor,
        commands: torch.Tensor,
        recent_readback_deltas: torch.Tensor | None = None,
        recent_readback_errors: torch.Tensor | None = None,
        *,
        source_index: torch.Tensor | None = None,
        analytic_shift_px: torch.Tensor | None = None,
        analytic_valid: torch.Tensor | None = None,
    ) -> V176LeWMPrediction:
        config: V177Config = self.config  # type: ignore[assignment]
        grounding = _finite("dynamics_grounding", grounded.dynamics_grounding)
        batch = len(grounding)
        grounding = _shape(
            "dynamics_grounding",
            grounding,
            (batch, VIEWS, ANCHORS, config.compact_dim),
        )
        belief = _normalized_log_belief(
            "grounded.log_posterior", grounded.log_posterior
        ).to(grounding)
        _bool("grounded.view_valid", grounded.view_valid, (batch, VIEWS))
        state = _shape("current_q", current_q, (batch, STATE_DIM)).to(grounding)
        action = _shape("commands", commands, (batch, COMMAND_STEPS, STATE_DIM)).to(grounding)

        if source_index is None:
            source = torch.full((batch,), SOURCE_UNKNOWN, dtype=torch.long, device=grounding.device)
        else:
            source = torch.as_tensor(source_index, device=grounding.device)
            if tuple(source.shape) != (batch,) or source.dtype not in (
                torch.int64,
                torch.int32,
                torch.int16,
                torch.uint8,
            ):
                raise V177ContractError("source_index must be an integer [B] tensor")
            source = source.long()
            in_range = ((source >= 0) & (source < int(config.source_vocabulary))).all()
            if source.device.type == "cuda":
                torch._assert_async(in_range, "source_index outside vocabulary")
            elif not bool(in_range):
                raise V177ContractError("source_index outside vocabulary")
        source_token = self.source_embedding(source)[:, None]

        if analytic_shift_px is None:
            shift_in = grounding.new_zeros(batch, VIEWS, ANCHORS, 2)
            valid_in = torch.zeros(batch, VIEWS, dtype=torch.bool, device=grounding.device)
        else:
            raw_shift = torch.as_tensor(analytic_shift_px).to(grounding)
            if tuple(raw_shift.shape) != (batch, VIEWS, ANCHORS, 2):
                raise V177ContractError("analytic_shift_px must be [B,2,2,2]")
            if analytic_valid is None:
                raise V177ContractError("analytic_shift_px requires analytic_valid")
            valid_in = _bool("analytic_valid", analytic_valid, (batch, VIEWS)).to(grounding.device)
            finite = torch.isfinite(raw_shift).all(dim=-1).all(dim=-1)
            valid_in = valid_in & finite
            shift_in = torch.nan_to_num(raw_shift, nan=0.0, posinf=0.0, neginf=0.0)
        if not bool(config.analytic_transport):
            valid_in = torch.zeros_like(valid_in)
        valid_tracks = valid_in[:, :, None].expand(-1, -1, ANCHORS)
        shift_in = torch.where(valid_tracks[..., None], shift_in, torch.zeros_like(shift_in))

        readback_delta = None
        if recent_readback_deltas is not None:
            readback_delta = _shape(
                "recent_readback_deltas",
                recent_readback_deltas,
                (batch, COMMAND_STEPS, STATE_DIM),
            ).to(grounding)
        readback_error = None
        if recent_readback_errors is not None:
            readback_error = _shape(
                "recent_readback_errors",
                recent_readback_errors,
                (batch, COMMAND_STEPS, STATE_DIM),
            ).to(grounding)
        if readback_delta is None and readback_error is None:
            readback_tokens = (
                self.readback_null_embedding.reshape(1, 1, -1) + self.readback_time_embedding
            ).expand(batch, -1, -1)
        else:
            readback_tokens = self.readback_present_embedding.reshape(1, 1, -1)
            readback_tokens = readback_tokens + self.readback_time_embedding
            if readback_delta is not None:
                readback_tokens = readback_tokens + self.readback_delta_projection(
                    readback_delta / self.joint_step_scale
                )
            if readback_error is not None:
                readback_tokens = readback_tokens + self.readback_error_projection(
                    readback_error / self.servo_error_scale
                )

        anchors = self.grounding_projection(grounding) + self.view_embedding
        # The analytic mean enters as a per-track token feature so the network
        # can learn a residual around it.  No anchor-index embedding: tracks
        # remain an exchangeable set.
        anchors = anchors + self.analytic_shift_projection(
            shift_in / float(config.max_transport_shift_px)
        )
        anchors = anchors + self.analytic_status_embedding[valid_tracks.long()]
        anchors = anchors.reshape(batch, VIEWS * ANCHORS, config.feature_dim)

        state_normalized = (state - self.joint_center) / self.joint_half_range
        previous_command = torch.cat((state[:, None], action[:, :-1]), dim=1)
        command_delta = (action - previous_command) / self.joint_step_scale
        motion = torch.cat(
            (
                self.q_projection(state_normalized)[:, None],
                self.command_projection(command_delta),
            ),
            dim=1,
        ) + self.time_embedding
        sequence = torch.cat((anchors, source_token, readback_tokens, motion), dim=1)
        traces = [sequence] if config.trace_level == "full" else None
        for layer in self.layers:
            sequence = layer(sequence)
            if traces is not None:
                traces.append(sequence)
        sequence = self.output_norm(sequence)
        anchor_hidden = sequence[:, : VIEWS * ANCHORS].reshape(
            batch, VIEWS, ANCHORS, config.feature_dim
        )
        residual = torch.tanh(self.shift_head(anchor_hidden))
        bound = torch.where(
            valid_tracks[..., None],
            residual.new_full((), float(config.analytic_residual_max_px)),
            residual.new_full((), float(config.max_transport_shift_px)),
        )
        shift = residual * bound + shift_in
        sigma_fraction = torch.sigmoid(self.sigma_head(anchor_hidden))
        sigma = config.minimum_transport_sigma_px + sigma_fraction * (
            config.maximum_transport_sigma_px - config.minimum_transport_sigma_px
        )
        prior = propagate_belief_v177(
            belief,
            shift,
            sigma,
            uniform_mix=config.transport_uniform_mix,
            blur_radius=config.blur_radius,
            mass_preserving=bool(config.mass_preserving_blur),
        )
        motion_offset = VIEWS * ANCHORS + 1 + COMMAND_STEPS
        motion_hidden = sequence[:, motion_offset:]
        baseline = torch.cat((state[:, None], action), dim=1)
        predicted_q = baseline + self.q_residual_head(motion_hidden) * self.joint_step_scale
        predicted_q = torch.cat((state[:, None], predicted_q[:, 1:]), dim=1)
        return V176LeWMPrediction(
            prior_log_belief=prior,
            predicted_q_path=predicted_q,
            transport_shift_px=shift,
            transport_sigma_px=sigma,
            tokens_by_layer=(torch.stack(traces, dim=1) if traces is not None else None),
        )


class V177Actor(V176Actor):
    """One decoder, three matched context modes.

    Every mode instantiates the same parameters so checkpoints are
    interchangeable; only ``self.mode`` selects which context tokens the five
    action queries may attend to.
    """

    def __init__(self, config: V177Config = V177Config()) -> None:
        if not isinstance(config, V177Config):
            raise V177ContractError("V177Actor requires a V177Config")
        super().__init__(config)
        dimension = config.actor_dim
        self.mode = str(config.actor_mode)
        self.no_grounding_token = nn.Parameter(torch.empty(1, 1, dimension))
        self.history_scene_projection = nn.Linear(VIT_DIM, dimension)
        self.history_motion_projection = nn.Sequential(
            nn.Linear(2 * COMMAND_STEPS * STATE_DIM, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.history_slot_embedding = nn.Parameter(torch.empty(1, HISTORY_STEPS, 1, dimension))
        self.history_kind_embedding = nn.Parameter(
            torch.empty(1, 1, HISTORY_TOKENS_PER_STEP, dimension)
        )
        self.history_null_token = nn.Parameter(torch.empty(1, 1, 1, dimension))
        for value in (
            self.no_grounding_token,
            self.history_slot_embedding,
            self.history_kind_embedding,
            self.history_null_token,
        ):
            nn.init.normal_(value, std=0.02)

    def set_mode(self, mode: str) -> None:
        if mode not in ACTOR_MODES:
            raise V177ContractError(f"actor mode must be one of {ACTOR_MODES}")
        self.mode = str(mode)

    def _history_tokens(self, history: V177RawHistory | None, batch: int) -> torch.Tensor:
        if history is None:
            raise V177ContractError("raw_history Actor requires a V177RawHistory")
        dimension = self.config.actor_dim
        reference = self.step_cap
        commands = _shape(
            "history.commands", history.commands, (batch, HISTORY_STEPS, COMMAND_STEPS, STATE_DIM)
        ).to(reference)
        paths = _shape(
            "history.paths", history.paths, (batch, HISTORY_STEPS, PATH_STEPS, STATE_DIM)
        ).to(reference)
        pooled = _shape(
            "history.scene_pooled", history.scene_pooled, (batch, HISTORY_STEPS, VIEWS, VIT_DIM)
        ).to(reference)
        valid = _bool("history.valid", history.valid, (batch, HISTORY_STEPS)).to(reference.device)
        steps = int(self.config.raw_history_steps)  # type: ignore[attr-defined]
        if steps < HISTORY_STEPS:
            # Keep only the newest ``steps`` completed transitions.
            keep = torch.zeros(HISTORY_STEPS, dtype=torch.bool, device=valid.device)
            keep[HISTORY_STEPS - steps :] = True
            valid = valid & keep[None]
        start = paths[:, :, 0]
        command_delta = (commands - start[:, :, None]) / (self.step_cap * float(COMMAND_STEPS))
        path_delta = (paths[:, :, 1:] - paths[:, :, :-1]) / self.step_cap
        motion = self.history_motion_projection(
            torch.cat((command_delta, path_delta), dim=-1).reshape(
                batch, HISTORY_STEPS, 2 * COMMAND_STEPS * STATE_DIM
            )
        )
        scene = self.history_scene_projection(F.layer_norm(pooled, (VIT_DIM,)))
        tokens = torch.cat((scene, motion[:, :, None]), dim=2)
        tokens = tokens + self.history_slot_embedding + self.history_kind_embedding
        tokens = tokens + self.source_embedding[1]
        null = self.history_null_token + self.history_slot_embedding + self.source_embedding[1]
        tokens = torch.where(valid[:, :, None, None], tokens, null.expand_as(tokens))
        return tokens.reshape(batch, HISTORY_STEPS * HISTORY_TOKENS_PER_STEP, dimension)

    def forward(  # type: ignore[override]
        self,
        scene: torch.Tensor,
        dynamics_grounding: torch.Tensor,
        ordered_tokens: torch.Tensor,
        innovation: torch.Tensor,
        current_q: torch.Tensor,
        appearance_available: torch.Tensor,
        *,
        history: V177RawHistory | None = None,
    ) -> V176ActorOutput:
        if self.mode not in ACTOR_MODES:
            raise V177ContractError(f"actor mode must be one of {ACTOR_MODES}")
        scene = _finite("scene", scene)
        batch = len(scene)
        scene = _shape("scene", scene, (batch, VIEWS, VIT_PATCHES, VIT_DIM)).to(
            dtype=self.scene_projection.weight.dtype
        )
        grounded = _shape(
            "dynamics_grounding",
            dynamics_grounding,
            (batch, VIEWS, ANCHORS, self.config.compact_dim),
        ).to(scene)
        appearance = _shape(
            "ordered_tokens",
            ordered_tokens,
            (batch, VIEWS, ANCHORS, self.config.feature_dim),
        ).to(scene)
        error = _shape(
            "innovation", innovation, (batch, VIEWS, ANCHORS, self.config.compact_dim)
        ).to(scene)
        state = _shape("current_q", current_q, (batch, STATE_DIM)).to(scene)
        available = _bool(
            "appearance_available", appearance_available, (batch, VIEWS)
        ).to(scene.device)
        available_tracks = available[:, :, None]
        scene_group = (
            self.scene_projection(scene)
            + self.scene_view_embedding
            + self.source_embedding[0]
        ).reshape(batch, VIEWS * VIT_PATCHES, self.config.actor_dim)
        if self.mode == "full":
            context_group = (
                self.track_projection(torch.cat((grounded, error), dim=-1))
                + self.appearance_projection(appearance)
                * available_tracks[..., None].to(appearance)
                + self.appearance_status_embedding[available_tracks.long()]
                + self.track_view_embedding
                + self.source_embedding[1]
            ).reshape(batch, VIEWS * ANCHORS, self.config.actor_dim)
        elif self.mode == "memoryless":
            context_group = (self.no_grounding_token + self.source_embedding[1]).expand(
                batch, -1, -1
            ).to(scene)
        else:
            context_group = self._history_tokens(history, batch).to(scene)
        q_group = (
            self.q_projection((state - self.joint_center) / self.joint_half_range)
            + self.source_embedding[2]
        )[:, None]
        memory = torch.cat((scene_group, context_group, q_group), dim=1)
        query = self.action_queries.expand(batch, -1, -1) + self.action_time_embedding
        causal = torch.triu(
            torch.ones(COMMAND_STEPS, COMMAND_STEPS, dtype=torch.bool, device=query.device),
            diagonal=1,
        )
        traces = [query] if self.config.trace_level == "full" else None
        for layer in self.layers:
            query = layer(query, memory, causal)
            if traces is not None:
                traces.append(query)
        query = self.output_norm(query)
        increments = torch.tanh(self.action_head(query)) * self.step_cap
        desired_rows: list[torch.Tensor] = []
        previous = state
        for step in range(COMMAND_STEPS):
            previous = previous + increments[:, step]
            desired_rows.append(previous)
        desired = torch.stack(desired_rows, dim=1)
        commands = _project_action_chunk(
            state, desired, self.step_cap, self.action_lower, self.action_upper
        )
        return V176ActorOutput(
            commands=commands,
            desired_commands=desired,
            increments=increments,
            action_queries_by_layer=(torch.stack(traces, dim=1) if traces is not None else None),
        )


class V177Core(V176Core):
    """Public FCPG -> camera-aware GDM -> predict/update -> matched Actor core."""

    def __init__(self, config: V177Config = V177Config()) -> None:  # noqa: D107
        if not isinstance(config, V177Config):
            raise V177ContractError("V177Core requires a V177Config")
        nn.Module.__init__(self)
        config.validate()
        self.config = config
        self.measurement = V176ExchangeableMeasurement(config)
        self.grounding_encoder = V176GroundingEncoder(config)
        self.lewm = V177BeliefLeWM(config)
        self.innovation_encoder = V176InnovationEncoder(config)
        self.actor = V177Actor(config)

    def predict(  # type: ignore[override]
        self,
        grounded: V176GroundedState,
        current_q: torch.Tensor,
        commands: torch.Tensor,
        recent_readback_deltas: torch.Tensor | None = None,
        recent_readback_errors: torch.Tensor | None = None,
        *,
        source_index: torch.Tensor | None = None,
        analytic_shift_px: torch.Tensor | None = None,
        analytic_valid: torch.Tensor | None = None,
    ) -> V176LeWMPrediction:
        return self.lewm(
            grounded,
            current_q,
            commands,
            recent_readback_deltas=recent_readback_deltas,
            recent_readback_errors=recent_readback_errors,
            source_index=source_index,
            analytic_shift_px=analytic_shift_px,
            analytic_valid=analytic_valid,
        )

    def h0_prior(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        geometry_log_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the H0 prior: neutral, or the broadened geometry when allowed."""

        config: V177Config = self.config  # type: ignore[assignment]
        neutral = uniform_spatial_prior(batch, device=device, dtype=dtype)
        if geometry_log_prior is None or not bool(config.geometry_prior_fusion):
            return neutral
        return broaden_geometry_prior(
            geometry_log_prior, floor=float(config.geometry_prior_floor)
        ).to(neutral)

    def update(  # type: ignore[override]
        self,
        assigned_measurement: V176AssignedMeasurement,
        prediction: V176LeWMPrediction | None = None,
        measured_q_path: torch.Tensor | None = None,
        *,
        geometry_log_prior: torch.Tensor | None = None,
    ) -> V176FilterUpdate:
        batch = len(assigned_measurement.view_valid)
        if prediction is None:
            if measured_q_path is not None:
                raise V177ContractError(
                    "H0 update cannot accept a measured path without prediction"
                )
            prior = self.h0_prior(
                batch,
                device=assigned_measurement.ordered_log_likelihood.device,
                dtype=assigned_measurement.ordered_log_likelihood.dtype,
                geometry_log_prior=geometry_log_prior,
            )
        else:
            if geometry_log_prior is not None:
                raise V177ContractError(
                    "geometry enters only the H0 prior; later steps carry it "
                    "through the analytic transport mean"
                )
            prior = prediction.prior_log_belief
        posterior = fuse_measurement_prior(
            prior,
            assigned_measurement.ordered_log_likelihood,
            assigned_measurement.view_valid,
        )
        ordered_tokens = posterior_weighted_pool(
            assigned_measurement.dense_features,
            posterior,
            self.measurement.absent_embedding,
        )
        ordered_tokens = ordered_tokens * assigned_measurement.view_valid[:, :, None, None].to(
            ordered_tokens
        )
        dynamics_grounding, statistics = self.grounding_encoder(
            posterior, assigned_measurement.view_valid
        )
        grounded = V176GroundedState(
            log_posterior=posterior,
            ordered_tokens=ordered_tokens,
            dynamics_grounding=dynamics_grounding,
            belief_statistics=statistics,
            view_valid=assigned_measurement.view_valid,
            appearance_available=assigned_measurement.view_valid,
        )
        if prediction is None:
            compact = self.innovation_encoder.null(
                batch,
                assigned_measurement.view_valid,
                device=dynamics_grounding.device,
                dtype=dynamics_grounding.dtype,
            )
            residual = dynamics_grounding.new_zeros(batch, PATH_STEPS, STATE_DIM)
            surprise = dynamics_grounding.new_zeros(batch, VIEWS, ANCHORS)
            present = torch.zeros(batch, dtype=torch.bool, device=dynamics_grounding.device)
        else:
            if measured_q_path is None:
                raise V177ContractError("predicted update requires the realized q path")
            signals = compute_innovation(
                posterior,
                prediction.prior_log_belief,
                assigned_measurement.view_valid,
                measured_q_path,
                prediction.predicted_q_path,
            )
            compact = self.innovation_encoder(signals)
            residual = signals.state_residual
            surprise = signals.surprise
            present = torch.ones(batch, dtype=torch.bool, device=dynamics_grounding.device)
        innovation = V176Innovation(
            innovation=compact,
            belief_surprise=surprise,
            state_residual=residual,
            view_valid=assigned_measurement.view_valid,
            has_prediction=present,
        )
        return V176FilterUpdate(grounded=grounded, innovation=innovation)

    def act(  # type: ignore[override]
        self,
        scene_tokens: torch.Tensor,
        grounded: V176GroundedState,
        innovation: torch.Tensor,
        current_q: torch.Tensor,
        *,
        history: V177RawHistory | None = None,
    ) -> V176ActorOutput:
        return act_with(self.actor, scene_tokens, grounded, innovation, current_q, history=history)


def act_with(
    actor: V177Actor,
    scene_tokens: torch.Tensor,
    grounded: V176GroundedState,
    innovation: torch.Tensor,
    current_q: torch.Tensor,
    *,
    history: V177RawHistory | None = None,
) -> V176ActorOutput:
    """Apply the V17.6.2 appearance/scene gating and run one Actor module."""

    batch = len(grounded.ordered_tokens)
    valid = _bool("grounded.view_valid", grounded.view_valid, (batch, VIEWS)).to(
        grounded.ordered_tokens.device
    )
    appearance_available = _bool(
        "grounded.appearance_available", grounded.appearance_available, (batch, VIEWS)
    ).to(grounded.ordered_tokens.device)
    actor_tokens = grounded.ordered_tokens * appearance_available[:, :, None, None].to(
        grounded.ordered_tokens
    )
    actor_scene = scene_tokens * valid[:, :, None, None].to(scene_tokens)
    return actor(
        actor_scene,
        grounded.dynamics_grounding,
        actor_tokens,
        innovation,
        current_q,
        appearance_available,
        history=history,
    )


__all__ = [
    "ABSENT_INDEX",
    "ACTOR_DIM",
    "ACTOR_MODES",
    "ANCHORS",
    "BELIEF_CLASSES",
    "COMMAND_STEPS",
    "COMPACT_DIM",
    "FEATURE_DIM",
    "GRID_CELLS",
    "GRID_SIZE",
    "HISTORY_STEPS",
    "HISTORY_TOKENS_PER_STEP",
    "HYPOTHESES",
    "IMAGE_SIZE",
    "PATH_STEPS",
    "SOURCE_UNKNOWN",
    "STATE_DIM",
    "VIEWS",
    "VIEW_ORDER",
    "VIT_DIM",
    "VIT_PATCHES",
    "V177Actor",
    "V177ActorOutput",
    "V177AssignedMeasurement",
    "V177BeliefLeWM",
    "V177Config",
    "V177ContractError",
    "V177Core",
    "V177ExchangeableMeasurement",
    "V177FilterUpdate",
    "V177GroundedState",
    "V177GroundingEncoder",
    "V177Innovation",
    "V177InnovationEncoder",
    "V177LeWMPrediction",
    "V177Measurement",
    "V177RawHistory",
    "act_with",
    "action_chunk_l1",
    "analytic_shift_from_pixels",
    "append_raw_history",
    "belief_prediction_nll",
    "belief_statistics",
    "broaden_geometry_prior",
    "compute_innovation",
    "credible_coverage",
    "empty_raw_history",
    "exchangeable_pair_nll",
    "fuse_geometry_prior",
    "fuse_measurement_prior",
    "hypothesis_overlap",
    "hypothesis_separation_penalty",
    "identity_or_track_assignment",
    "normalize_actor_scene_tokens",
    "pool_scene_tokens",
    "posterior_weighted_pool",
    "propagate_belief_v177",
    "state_path_smooth_l1",
    "uniform_spatial_prior",
]
