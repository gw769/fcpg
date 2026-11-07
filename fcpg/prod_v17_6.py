"""Prod-V17.6.2 exchangeable visual belief controller core.

The module deliberately separates five operations:

``measure``
    Extract two exchangeable anchor-hypothesis likelihoods from the current
    image only.
``assign_identity``
    Associate the two exchangeable hypotheses by one hard direct/swap choice.
``predict``
    Transport the ordered spatial belief with the proposed action chunk.
``update``
    Fuse the transported prior with the next current-image likelihood.
``act``
    Predict one five-command chunk from current scene, grounding, innovation,
    and robot state.  Raw history and hard image coordinates are not inputs.

Geometry may construct an ordered prior outside ``measure``.  It never enters
the image measurement network and can only select the direct/swap ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F


VIEWS: Final = 2
VIEW_ORDER: Final = ("wrist", "column")
HYPOTHESES: Final = 2
ANCHORS: Final = 2
GRID_SIZE: Final = 56
GRID_CELLS: Final = GRID_SIZE * GRID_SIZE
BELIEF_CLASSES: Final = GRID_CELLS + 1
ABSENT_INDEX: Final = GRID_CELLS
IMAGE_SIZE: Final = 224
VIT_GRID: Final = 16
VIT_PATCHES: Final = VIT_GRID * VIT_GRID
VIT_DIM: Final = 384
FEATURE_DIM: Final = 384
COMPACT_DIM: Final = 256
ACTOR_DIM: Final = 512
STATE_DIM: Final = 6
COMMAND_STEPS: Final = 5
PATH_STEPS: Final = COMMAND_STEPS + 1


class V176ContractError(ValueError):
    """Raised when a tensor violates the public V17.6.2 contract."""


def _finite(name: str, value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    finite = torch.isfinite(tensor).all()
    if tensor.device.type == "cuda":
        # Keep the contract without synchronizing CPU and GPU at every module
        # boundary.  Training loops still check the scalar loss explicitly.
        torch._assert_async(finite, f"{name} must be finite")
    elif not bool(finite):
        raise V176ContractError(f"{name} must be finite")
    return tensor


def _shape(
    name: str, value: torch.Tensor, expected: tuple[int, ...]
) -> torch.Tensor:
    tensor = _finite(name, value)
    if tuple(tensor.shape) != tuple(expected):
        raise V176ContractError(
            f"{name} shape differs: observed={tuple(tensor.shape)} "
            f"expected={tuple(expected)}"
        )
    return tensor


def _bool(
    name: str, value: torch.Tensor, expected: tuple[int, ...]
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.dtype != torch.bool or tuple(tensor.shape) != tuple(expected):
        raise V176ContractError(
            f"{name} must be bool with shape {tuple(expected)}"
        )
    return tensor


def _normalized_log_belief(name: str, value: torch.Tensor) -> torch.Tensor:
    tensor = _finite(name, value).float()
    if tensor.shape[-1] != BELIEF_CLASSES:
        raise V176ContractError(
            f"{name} must end in {BELIEF_CLASSES} classes"
        )
    return F.log_softmax(tensor, dim=-1)


def _grid_coordinates(
    *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, GRID_SIZE, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(GRID_CELLS, 2)


def belief_statistics(log_belief: torch.Tensor) -> torch.Tensor:
    """Return visible mass, absent mass, entropy, mean and covariance terms."""

    belief = _normalized_log_belief("log_belief", log_belief)
    probability = belief.exp()
    visible = probability[..., :GRID_CELLS]
    coordinate = _grid_coordinates(device=belief.device, dtype=belief.dtype)
    mass = visible.sum(dim=-1)
    conditional = visible / mass[..., None].clamp_min(1.0e-12)
    mean = torch.einsum("...c,cd->...d", conditional, coordinate)
    centered = coordinate.reshape(
        *((1,) * (conditional.ndim - 1)), GRID_CELLS, 2
    ) - mean[..., None, :]
    variance = torch.einsum("...c,...cd->...d", conditional, centered.square())
    covariance = (
        conditional * centered[..., 0] * centered[..., 1]
    ).sum(dim=-1)
    entropy = -(probability * belief).sum(dim=-1) / math.log(BELIEF_CLASSES)
    return torch.cat(
        (
            mass[..., None],
            probability[..., ABSENT_INDEX, None],
            entropy[..., None],
            mean,
            variance,
            covariance[..., None],
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class V176Config:
    feature_dim: int = FEATURE_DIM
    compact_dim: int = COMPACT_DIM
    actor_dim: int = ACTOR_DIM
    heads: int = 8
    measurement_layers: int = 3
    lewm_layers: int = 4
    actor_layers: int = 6
    ffn_multiplier: int = 4
    dropout: float = 0.0
    max_transport_shift_px: float = 64.0
    minimum_transport_sigma_px: float = 0.35
    maximum_transport_sigma_px: float = 8.0
    transport_uniform_mix: float = 0.05
    blur_radius: int = 8
    servo_error_scale_deg: tuple[float, ...] = (
        0.838503,
        1.786527,
        1.967177,
        1.386477,
        0.714411,
        4.626208,
    )
    trace_level: str = "none"

    def validate(self) -> None:
        dimensions = (
            self.feature_dim,
            self.compact_dim,
            self.actor_dim,
            self.heads,
            self.measurement_layers,
            self.lewm_layers,
            self.actor_layers,
            self.ffn_multiplier,
            self.blur_radius,
        )
        if min(dimensions) <= 0:
            raise V176ContractError("all V17.6.2 dimensions must be positive")
        if self.feature_dim % self.heads or self.actor_dim % self.heads:
            raise V176ContractError("attention heads must divide model widths")
        if self.dropout != 0.0:
            raise V176ContractError("V17.6.2 dropout is fixed at zero")
        if not (
            0.0 < self.minimum_transport_sigma_px
            <= self.maximum_transport_sigma_px
        ):
            raise V176ContractError("transport sigma bounds are invalid")
        if self.max_transport_shift_px <= 0.0:
            raise V176ContractError("transport shift bound must be positive")
        if not math.isclose(self.transport_uniform_mix, 0.05, abs_tol=1.0e-12):
            raise V176ContractError(
                "V17.6.2 transport mixture is fixed at 5 percent"
            )
        if len(self.servo_error_scale_deg) != STATE_DIM or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.servo_error_scale_deg
        ):
            raise V176ContractError(
                "servo_error_scale_deg must contain six positive finite values"
            )
        if self.trace_level not in {"none", "full"}:
            raise V176ContractError("trace_level must be 'none' or 'full'")


@dataclass(frozen=True)
class V176Measurement:
    """Unordered current-image measurement; no physical identity is attached."""

    scene_tokens: torch.Tensor
    dense_features: torch.Tensor
    anchor_hypothesis_log_likelihood: torch.Tensor
    view_valid: torch.Tensor
    query_traces: torch.Tensor | None


@dataclass(frozen=True)
class V176AssignedMeasurement:
    """Hard-ordered physical anchors or anonymous tracks after association."""

    scene_tokens: torch.Tensor
    dense_features: torch.Tensor
    ordered_log_likelihood: torch.Tensor
    assignment: torch.Tensor
    view_valid: torch.Tensor
    query_traces: torch.Tensor | None


@dataclass(frozen=True)
class V176GroundedState:
    log_posterior: torch.Tensor
    ordered_tokens: torch.Tensor
    dynamics_grounding: torch.Tensor
    belief_statistics: torch.Tensor
    view_valid: torch.Tensor
    appearance_available: torch.Tensor


@dataclass(frozen=True)
class V176LeWMPrediction:
    prior_log_belief: torch.Tensor
    predicted_q_path: torch.Tensor
    transport_shift_px: torch.Tensor
    transport_sigma_px: torch.Tensor
    tokens_by_layer: torch.Tensor | None


@dataclass(frozen=True)
class V176InnovationSignals:
    probability_delta: torch.Tensor
    statistic_delta: torch.Tensor
    surprise: torch.Tensor
    state_residual: torch.Tensor
    view_valid: torch.Tensor


@dataclass(frozen=True)
class V176Innovation:
    innovation: torch.Tensor
    belief_surprise: torch.Tensor
    state_residual: torch.Tensor
    view_valid: torch.Tensor
    has_prediction: torch.Tensor


@dataclass(frozen=True)
class V176FilterUpdate:
    grounded: V176GroundedState
    innovation: V176Innovation


@dataclass(frozen=True)
class V176ActorOutput:
    commands: torch.Tensor
    desired_commands: torch.Tensor
    increments: torch.Tensor
    action_queries_by_layer: torch.Tensor | None


def normalize_actor_scene_tokens(
    dino_tokens: torch.Tensor,
    view_valid: torch.Tensor,
) -> torch.Tensor:
    """Apply the one canonical Actor scene-token normalization and mask."""

    dino = _finite("dino_tokens", dino_tokens)
    batch = len(dino)
    dino = _shape(
        "dino_tokens", dino, (batch, VIEWS, VIT_PATCHES, VIT_DIM)
    ).float()
    valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(dino.device)
    return F.layer_norm(dino, (VIT_DIM,)) * valid[:, :, None, None].to(dino)


def uniform_spatial_prior(
    batch: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the factorized-neutral H0 prior.

    Visibility is Bernoulli-neutral while conditional visible location is
    uniform: P(absent)=0.5 and P(pixel)=0.5/GRID_CELLS.
    """

    if int(batch) <= 0:
        raise V176ContractError("uniform prior batch must be positive")
    spatial = torch.full(
        (int(batch), VIEWS, ANCHORS, GRID_CELLS),
        0.5 / GRID_CELLS,
        device=device,
        dtype=dtype,
    )
    absent = torch.full(
        (int(batch), VIEWS, ANCHORS, 1),
        0.5,
        device=device,
        dtype=dtype,
    )
    return torch.cat((spatial, absent), dim=-1).log()


def posterior_weighted_pool(
    dense_features: torch.Tensor,
    log_belief: torch.Tensor,
    absent_token: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pool spatial features without reducing a belief to a hard coordinate."""

    dense = _finite("dense_features", dense_features)
    if dense.ndim != 4 or dense.shape[1:3] != (VIEWS, GRID_CELLS):
        raise V176ContractError(
            "dense_features must be [B,2,3136,D]"
        )
    batch, _, _, dimension = dense.shape
    belief = _normalized_log_belief("log_belief", log_belief).to(dense)
    _shape(
        "log_belief",
        belief,
        (batch, VIEWS, ANCHORS, BELIEF_CLASSES),
    )
    probability = belief.exp()
    pooled = torch.einsum(
        "bvac,bvcd->bvad", probability[..., :GRID_CELLS], dense
    )
    if absent_token is not None:
        absent = _finite("absent_token", absent_token).to(dense)
        if absent.shape == (dimension,):
            absent = absent.reshape(1, 1, 1, dimension)
        elif absent.shape != (batch, VIEWS, ANCHORS, dimension):
            raise V176ContractError(
                "absent_token must be [D] or [B,2,2,D]"
            )
        pooled = pooled + probability[..., ABSENT_INDEX, None] * absent
    return pooled


def identity_or_track_assignment(
    anchor_hypothesis_log_likelihood: torch.Tensor,
    ordered_prior_log_belief: torch.Tensor | None = None,
    view_valid: torch.Tensor | None = None,
    *,
    direct_assignment: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Associate two hypotheses by one hard direct/swap permutation.

    Rows are maintained tracks (or physical anchors when external geometry
    identifies them); columns are exchangeable visual hypotheses.  A soft
    marginal would collapse an ambiguous two-component set into two identical
    marginals, so ambiguity deliberately keeps an anonymous hard track order.
    The operation only permutes likelihoods and never moves spatial mass.
    """

    hypotheses = _normalized_log_belief(
        "anchor_hypothesis_log_likelihood",
        anchor_hypothesis_log_likelihood,
    )
    if hypotheses.ndim != 4 or hypotheses.shape[1:3] != (VIEWS, HYPOTHESES):
        raise V176ContractError(
            "anchor_hypothesis_log_likelihood must be [B,2,2,3137]"
        )
    batch = len(hypotheses)
    prior = (
        uniform_spatial_prior(
            batch, device=hypotheses.device, dtype=hypotheses.dtype
        )
        if ordered_prior_log_belief is None
        else _normalized_log_belief(
            "ordered_prior_log_belief", ordered_prior_log_belief
        ).to(hypotheses)
    )
    _shape(
        "ordered_prior_log_belief",
        prior,
        (batch, VIEWS, ANCHORS, BELIEF_CLASSES),
    )
    valid = (
        torch.ones(batch, VIEWS, dtype=torch.bool, device=hypotheses.device)
        if view_valid is None
        else _bool("view_valid", view_valid, (batch, VIEWS)).to(
            hypotheses.device
        )
    )
    if direct_assignment is None:
        # Log probability overlap between each physical prior and image
        # hypothesis.  This scores only the two identity hypotheses; it never
        # translates or otherwise edits either spatial measurement.
        score = torch.logsumexp(
            prior[:, :, :, None, :] + hypotheses[:, :, None, :, :], dim=-1
        )
        same = score[:, :, 0, 0] + score[:, :, 1, 1]
        swapped = score[:, :, 0, 1] + score[:, :, 1, 0]
        direct = same >= swapped
    else:
        supplied = torch.as_tensor(
            direct_assignment, device=hypotheses.device
        )
        if tuple(supplied.shape) != (batch, VIEWS):
            raise V176ContractError(
                "direct_assignment must be [B,2]"
            )
        if supplied.dtype == torch.bool:
            direct = supplied
        else:
            supplied = _finite("direct_assignment", supplied).to(hypotheses)
            exact = ((supplied == 0.0) | (supplied == 1.0)).all()
            if supplied.device.type == "cuda":
                torch._assert_async(
                    exact, "direct_assignment must contain only 0 or 1"
                )
            elif not bool(exact):
                raise V176ContractError(
                    "direct_assignment must contain only 0 or 1"
                )
            direct = supplied.bool()
    # Invalid current pixels carry no association evidence.  Keeping the raw
    # order is harmless because the likelihood is uniform and avoids creating
    # a fractional assignment that would merge the two maintained tracks.
    direct = torch.where(valid, direct, torch.ones_like(direct))
    weight = direct.to(hypotheses)
    assignment = torch.stack(
        (
            torch.stack((weight, 1.0 - weight), dim=-1),
            torch.stack((1.0 - weight, weight), dim=-1),
        ),
        dim=-2,
    )
    ordered_probability = torch.einsum(
        "bvas,bvsc->bvac", assignment, hypotheses.exp()
    )
    ordered_log = ordered_probability.clamp_min(1.0e-12).log()
    ordered_log = F.log_softmax(ordered_log, dim=-1)
    return ordered_log, assignment


def exchangeable_pair_nll(
    anchor_hypothesis_log_likelihood: torch.Tensor,
    target_belief: torch.Tensor,
    view_valid: torch.Tensor,
) -> torch.Tensor:
    """Exchangeable two-permutation soft-target cross-entropy surrogate.

    For one-hot targets this equals the exact joint set NLL.  Distributed
    targets retain assignment invariance but should not be interpreted as a
    calibrated likelihood of a set-valued event.
    """

    hypotheses = _normalized_log_belief(
        "anchor_hypothesis_log_likelihood",
        anchor_hypothesis_log_likelihood,
    )
    batch = len(hypotheses)
    target = _finite("target_belief", target_belief).float().to(hypotheses)
    _shape(
        "target_belief",
        target,
        (batch, VIEWS, ANCHORS, BELIEF_CLASSES),
    )
    non_negative = (target >= 0.0).all()
    if target.device.type == "cuda":
        torch._assert_async(non_negative, "target_belief must be non-negative")
    elif not bool(non_negative):
        raise V176ContractError("target_belief must be non-negative")
    mass = target.sum(dim=-1, keepdim=True)
    positive_mass = (mass > 0.0).all()
    if target.device.type == "cuda":
        torch._assert_async(positive_mass, "target_belief must have positive mass")
    elif not bool(positive_mass):
        raise V176ContractError("target_belief must have positive mass")
    target = target / mass
    valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(
        hypotheses.device
    )
    cost = -torch.einsum("bvac,bvsc->bvas", target, hypotheses)
    same = cost[:, :, 0, 0] + cost[:, :, 1, 1]
    swapped = cost[:, :, 0, 1] + cost[:, :, 1, 0]
    # For distinct one-hot labels the two assignments are mutually exclusive
    # set events and their probabilities are summed exactly.  When both target
    # elements are identical (most importantly both absent), direct and swap
    # denote the same surrogate event and must be counted only once.
    per_view = -torch.logsumexp(torch.stack((-same, -swapped), -1), -1)
    duplicate_target = (target[:, :, 0] - target[:, :, 1]).abs().amax(-1) <= 1.0e-7
    per_view = per_view + duplicate_target.to(per_view) * math.log(2.0)
    return (per_view * valid).sum() / valid.sum().clamp_min(1)


def _forward_splat(
    visible: torch.Tensor, shift_px: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably translate probability mass; return grid and lost mass."""

    count = len(visible)
    if visible.shape != (count, 1, GRID_SIZE, GRID_SIZE):
        raise V176ContractError("visible belief grid shape differs")
    shift = _shape("shift_px", shift_px, (count, 2)).to(visible)
    axis = torch.arange(GRID_SIZE, device=visible.device, dtype=visible.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    target_x = xx[None] + shift[:, 0, None, None]
    target_y = yy[None] + shift[:, 1, None, None]
    x0 = target_x.floor()
    y0 = target_y.floor()
    dx = target_x - x0
    dy = target_y - y0
    source = visible[:, 0].reshape(count, -1)
    output = visible.new_zeros(count, GRID_CELLS)
    retained = visible.new_zeros(count)
    for x_offset, y_offset, weight in (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    ):
        xi = x0 + x_offset
        yi = y0 + y_offset
        inside = (xi >= 0) & (xi < GRID_SIZE) & (yi >= 0) & (yi < GRID_SIZE)
        index = (
            yi.clamp(0, GRID_SIZE - 1).long() * GRID_SIZE
            + xi.clamp(0, GRID_SIZE - 1).long()
        ).reshape(count, -1)
        contribution = source * weight.reshape(count, -1) * inside.reshape(
            count, -1
        )
        output.scatter_add_(1, index, contribution)
        retained = retained + contribution.sum(dim=-1)
    original = source.sum(dim=-1)
    lost = (original - retained).clamp_min(0.0)
    return output.reshape(count, 1, GRID_SIZE, GRID_SIZE), lost


def _anisotropic_gaussian_blur(
    value: torch.Tensor, sigma_px: torch.Tensor, *, radius: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Axis-aligned anisotropic blur with per-belief differentiable scales."""

    count = len(value)
    sigma = _shape("sigma_px", sigma_px, (count, 2)).to(value).clamp_min(0.05)
    offsets = torch.arange(
        -int(radius), int(radius) + 1, device=value.device, dtype=value.dtype
    )
    weight_x = torch.exp(
        -0.5 * (offsets[None] / sigma[:, 0, None]).square()
    )
    weight_y = torch.exp(
        -0.5 * (offsets[None] / sigma[:, 1, None]).square()
    )
    weight_x = weight_x / weight_x.sum(dim=-1, keepdim=True)
    weight_y = weight_y / weight_y.sum(dim=-1, keepdim=True)
    kernel = 2 * int(radius) + 1
    horizontal = F.unfold(value, (1, kernel), padding=(0, int(radius)))
    horizontal = (
        horizontal * weight_x[:, :, None]
    ).sum(dim=1).reshape(count, 1, GRID_SIZE, GRID_SIZE)
    vertical = F.unfold(
        horizontal, (kernel, 1), padding=(int(radius), 0)
    )
    vertical = (
        vertical * weight_y[:, :, None]
    ).sum(dim=1).reshape(count, 1, GRID_SIZE, GRID_SIZE)
    before = value.sum(dim=(1, 2, 3))
    after = vertical.sum(dim=(1, 2, 3))
    scale = torch.minimum(
        torch.ones_like(after), before / after.clamp_min(1.0e-12)
    )
    vertical = vertical * scale[:, None, None, None]
    after = vertical.sum(dim=(1, 2, 3))
    lost = (before - after).clamp_min(0.0)
    return vertical, lost


def propagate_belief(
    log_belief: torch.Tensor,
    shift_px: torch.Tensor,
    sigma_px: torch.Tensor,
    *,
    uniform_mix: float = 0.05,
    blur_radius: int = 8,
) -> torch.Tensor:
    """Transport probability by forward splat, blur, and exact 5% mixture."""

    if not math.isclose(float(uniform_mix), 0.05, abs_tol=1.0e-12):
        raise V176ContractError(
            "V17.6.2 propagation requires a 5 percent mixture"
        )
    belief = _normalized_log_belief("log_belief", log_belief)
    if belief.ndim != 4 or belief.shape[1:3] != (VIEWS, ANCHORS):
        raise V176ContractError("log_belief must be [B,2,2,3137]")
    batch = len(belief)
    shift = _shape(
        "shift_px", shift_px, (batch, VIEWS, ANCHORS, 2)
    ).float().to(belief)
    sigma = _shape(
        "sigma_px", sigma_px, (batch, VIEWS, ANCHORS, 2)
    ).float().to(belief)
    probability = belief.exp()
    count = batch * VIEWS * ANCHORS
    visible = probability[..., :GRID_CELLS].reshape(
        count, 1, GRID_SIZE, GRID_SIZE
    )
    pixels_per_cell = float(IMAGE_SIZE) / float(GRID_SIZE)
    shifted, shift_loss = _forward_splat(
        visible, shift.reshape(count, 2) / pixels_per_cell
    )
    blurred, blur_loss = _anisotropic_gaussian_blur(
        shifted,
        sigma.reshape(count, 2) / pixels_per_cell,
        radius=int(blur_radius),
    )
    absent = probability[..., ABSENT_INDEX].reshape(count)
    absent = (absent + shift_loss + blur_loss).clamp(0.0, 1.0)
    transported = torch.cat(
        (blurred.reshape(count, GRID_CELLS), absent[:, None]), dim=-1
    )
    transported = transported / transported.sum(dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )
    neutral = torch.cat(
        (
            transported.new_full(
                (count, GRID_CELLS), 0.5 / GRID_CELLS
            ),
            transported.new_full((count, 1), 0.5),
        ),
        dim=-1,
    )
    transported = (
        (1.0 - float(uniform_mix)) * transported
        + float(uniform_mix) * neutral
    )
    transported = transported / transported.sum(dim=-1, keepdim=True)
    return transported.reshape(
        batch, VIEWS, ANCHORS, BELIEF_CLASSES
    ).clamp_min(1.0e-12).log()


def fuse_measurement_prior(
    prior_log_belief: torch.Tensor,
    ordered_measurement_log_likelihood: torch.Tensor,
    view_valid: torch.Tensor,
) -> torch.Tensor:
    """Normalized product-of-experts; invalid views are exact prior-only."""

    raw_prior = _finite("prior_log_belief", prior_log_belief).float()
    if raw_prior.ndim != 4 or raw_prior.shape[1:] != (
        VIEWS,
        ANCHORS,
        BELIEF_CLASSES,
    ):
        raise V176ContractError(
            "prior_log_belief must be [B,2,2,3137]"
        )
    normalization_error = torch.logsumexp(raw_prior, dim=-1).abs().max()
    normalized = normalization_error <= 1.0e-4
    if raw_prior.device.type == "cuda":
        torch._assert_async(normalized, "prior_log_belief must be normalized")
    elif not bool(normalized):
        raise V176ContractError("prior_log_belief must be normalized")
    prior = F.log_softmax(raw_prior, dim=-1)
    measurement = _normalized_log_belief(
        "ordered_measurement_log_likelihood",
        ordered_measurement_log_likelihood,
    ).to(prior)
    batch = len(prior)
    _shape(
        "ordered_measurement_log_likelihood",
        measurement,
        (batch, VIEWS, ANCHORS, BELIEF_CLASSES),
    )
    valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(prior.device)
    prior_visible = torch.logsumexp(prior[..., :GRID_CELLS], dim=-1)
    measurement_visible = torch.logsumexp(
        measurement[..., :GRID_CELLS], dim=-1
    )
    visibility = F.log_softmax(
        torch.stack(
            (
                prior_visible + measurement_visible,
                prior[..., ABSENT_INDEX] + measurement[..., ABSENT_INDEX],
            ),
            dim=-1,
        ),
        dim=-1,
    )
    conditional_xy = F.log_softmax(
        (prior[..., :GRID_CELLS] - prior_visible[..., None])
        + (
            measurement[..., :GRID_CELLS]
            - measurement_visible[..., None]
        ),
        dim=-1,
    )
    updated = torch.cat(
        (
            conditional_xy + visibility[..., 0, None],
            visibility[..., 1, None],
        ),
        dim=-1,
    )
    # `where` is intentional: an invalid image is a neutral factor bit-for-bit.
    return torch.where(valid[:, :, None, None], updated, raw_prior)


def compute_innovation(
    posterior_log_belief: torch.Tensor,
    prior_log_belief: torch.Tensor,
    view_valid: torch.Tensor,
    measured_q_path: torch.Tensor | None = None,
    predicted_q_path: torch.Tensor | None = None,
) -> V176InnovationSignals:
    """Return distributional and proprioceptive execution mismatch signals."""

    posterior = _normalized_log_belief(
        "posterior_log_belief", posterior_log_belief
    )
    prior = _normalized_log_belief("prior_log_belief", prior_log_belief).to(
        posterior
    )
    batch = len(posterior)
    _shape(
        "prior_log_belief",
        prior,
        (batch, VIEWS, ANCHORS, BELIEF_CLASSES),
    )
    valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(
        posterior.device
    )
    posterior_probability = posterior.exp()
    prior_probability = prior.exp()
    probability_delta = posterior_probability - prior_probability
    log_ratio = (posterior - prior).clamp(-20.0, 20.0)
    forward_kl = (posterior_probability * log_ratio).sum(dim=-1)
    reverse_kl = (prior_probability * -log_ratio).sum(dim=-1)
    surprise = 0.5 * (forward_kl + reverse_kl)
    statistic_delta = belief_statistics(posterior) - belief_statistics(prior)
    mask = valid[:, :, None]
    probability_delta = probability_delta * mask[..., None]
    statistic_delta = statistic_delta * mask[..., None]
    surprise = surprise * mask
    if (measured_q_path is None) != (predicted_q_path is None):
        raise V176ContractError(
            "measured and predicted q paths must be supplied together"
        )
    if measured_q_path is None:
        state_residual = posterior.new_zeros(batch, PATH_STEPS, STATE_DIM)
    else:
        measured = _shape(
            "measured_q_path",
            measured_q_path,
            (batch, PATH_STEPS, STATE_DIM),
        ).to(posterior)
        predicted = _shape(
            "predicted_q_path",
            predicted_q_path,
            (batch, PATH_STEPS, STATE_DIM),
        ).to(posterior)
        state_residual = measured - predicted
    return V176InnovationSignals(
        probability_delta=probability_delta,
        statistic_delta=statistic_delta,
        surprise=surprise,
        state_residual=state_residual,
        view_valid=valid,
    )


def belief_prediction_nll(
    prior_log_belief: torch.Tensor,
    target_log_belief: torch.Tensor,
    view_valid: torch.Tensor,
) -> torch.Tensor:
    prior = _normalized_log_belief("prior_log_belief", prior_log_belief)
    target = _normalized_log_belief("target_log_belief", target_log_belief).to(
        prior
    )
    batch = len(prior)
    valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(prior.device)
    per_anchor = -(target.exp() * prior).sum(dim=-1)
    active = valid[:, :, None].expand_as(per_anchor)
    return (per_anchor * active).sum() / active.sum().clamp_min(1)


def state_path_smooth_l1(
    predicted_q_path: torch.Tensor,
    target_q_path: torch.Tensor,
    *,
    scale: float = 4.0,
) -> torch.Tensor:
    predicted = _finite("predicted_q_path", predicted_q_path).float()
    target = _shape("target_q_path", target_q_path, tuple(predicted.shape)).to(
        predicted
    )
    residual = (predicted[:, 1:] - target[:, 1:]) / float(scale)
    return F.smooth_l1_loss(residual, torch.zeros_like(residual))


def action_chunk_l1(
    commands: torch.Tensor,
    target_commands: torch.Tensor,
    step_cap: torch.Tensor,
) -> torch.Tensor:
    output = _finite("commands", commands).float()
    target = _shape(
        "target_commands", target_commands, tuple(output.shape)
    ).to(output)
    cap = _shape("step_cap", step_cap, (STATE_DIM,)).to(output)
    return ((output - target).abs() / cap[None, None]).mean()


class _EncoderLayer(nn.Module):
    def __init__(self, dimension: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn),
            nn.GELU(),
            nn.Linear(ffn, dimension),
        )

    def forward(
        self, value: torch.Tensor, padding: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = self.attention_norm(value)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding,
            need_weights=False,
        )
        value = value + attended
        return value + self.ffn(self.ffn_norm(value))


class _MeasurementLayer(nn.Module):
    def __init__(self, dimension: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dimension)
        self.self_attention = nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.cross_attention = nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn), nn.GELU(), nn.Linear(ffn, dimension)
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(query)
        attended, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        query = query + attended
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(
            self.cross_norm(query), normalized_memory, normalized_memory,
            need_weights=False,
        )
        query = query + attended
        return query + self.ffn(self.ffn_norm(query))


class _ActorLayer(nn.Module):
    def __init__(self, dimension: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dimension)
        self.self_attention = nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.cross_attention = nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn), nn.GELU(), nn.Linear(ffn, dimension)
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.self_norm(query)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        query = query + attended
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(
            self.cross_norm(query), normalized_memory, normalized_memory,
            need_weights=False,
        )
        query = query + attended
        return query + self.ffn(self.ffn_norm(query))


class V176ExchangeableMeasurement(nn.Module):
    """Current-image-only exchangeable anchor-hypothesis measurement."""

    def __init__(self, config: V176Config = V176Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dimension = config.feature_dim
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, dimension, 3, padding=1),
            nn.GELU(),
        )
        self.dino_projection = nn.Conv2d(VIT_DIM, dimension, 1)
        self.rgb_dino_fusion = nn.Sequential(
            nn.Conv2d(2 * dimension, dimension, 1),
            nn.GELU(),
            nn.Conv2d(dimension, dimension, 3, padding=1),
        )
        self.position_projection = nn.Sequential(
            nn.Linear(2, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.camera_embedding = nn.Parameter(
            torch.empty(1, VIEWS, 1, dimension)
        )
        self.anchor_hypothesis_queries = nn.Parameter(
            torch.empty(1, 1, HYPOTHESES, dimension)
        )
        self.layers = nn.ModuleList(
            _MeasurementLayer(
                dimension, config.heads, config.ffn_multiplier * dimension
            )
            for _ in range(config.measurement_layers)
        )
        self.query_norm = nn.LayerNorm(dimension)
        self.pixel_norm = nn.LayerNorm(dimension)
        self.pixel_projection = nn.Linear(dimension, dimension)
        self.absent_head = nn.Linear(dimension, 1)
        self.absent_embedding = nn.Parameter(torch.empty(dimension))
        with torch.no_grad():
            self.dino_projection.weight.zero_()
            diagonal = min(
                self.dino_projection.weight.shape[0],
                self.dino_projection.weight.shape[1],
            )
            index = torch.arange(diagonal)
            self.dino_projection.weight[index, index, 0, 0] = 1.0
            if self.dino_projection.bias is not None:
                self.dino_projection.bias.zero_()
        nn.init.normal_(self.camera_embedding, std=0.02)
        nn.init.normal_(self.anchor_hypothesis_queries, std=0.02)
        nn.init.normal_(self.absent_embedding, std=0.02)

    def forward(
        self,
        dino_tokens: torch.Tensor,
        view_valid: torch.Tensor,
        rgb: torch.Tensor | None = None,
    ) -> V176Measurement:
        dino = _finite("dino_tokens", dino_tokens)
        batch = len(dino)
        dino = _shape(
            "dino_tokens", dino, (batch, VIEWS, VIT_PATCHES, VIT_DIM)
        ).float()
        valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(dino.device)
        normalized_dino = normalize_actor_scene_tokens(dino, valid)
        dino_map = normalized_dino.reshape(
            batch * VIEWS, VIT_GRID, VIT_GRID, VIT_DIM
        ).permute(0, 3, 1, 2)
        dino_map = F.interpolate(
            dino_map,
            size=(GRID_SIZE, GRID_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        dino_map = self.dino_projection(dino_map)
        if rgb is None:
            fused = dino_map
        else:
            image = _finite("rgb", rgb)
            image = _shape(
                "rgb", image, (batch, VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE)
            )
            if image.dtype == torch.uint8:
                image = image.float().div(255.0)
            elif image.is_floating_point():
                image = image.float()
                in_range = ((image >= 0.0) & (image <= 1.0)).all()
                if image.device.type == "cuda":
                    torch._assert_async(in_range, "floating RGB must lie in [0,1]")
                elif not bool(in_range):
                    raise V176ContractError("floating RGB must lie in [0,1]")
            else:
                raise V176ContractError("RGB must be uint8 or floating point")
            mean = image.mean(dim=(-1, -2), keepdim=True)
            std = image.std(dim=(-1, -2), keepdim=True).clamp_min(1.0e-4)
            rgb_map = self.rgb_stem(
                ((image - mean) / std).reshape(
                    batch * VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE
                )
            )
            fused = dino_map + self.rgb_dino_fusion(
                torch.cat((dino_map, rgb_map.to(dino_map)), dim=1)
            )
        coordinate = _grid_coordinates(device=fused.device, dtype=fused.dtype)
        position = self.position_projection(coordinate)
        fused = fused + position.T.reshape(
            1, self.config.feature_dim, GRID_SIZE, GRID_SIZE
        )
        fused = fused.reshape(
            batch, VIEWS, self.config.feature_dim, GRID_SIZE, GRID_SIZE
        )
        fused = fused * valid[:, :, None, None, None]
        dense = fused.flatten(start_dim=3).transpose(2, 3)
        query = (
            self.anchor_hypothesis_queries + self.camera_embedding
        ).expand(batch, -1, -1, -1).reshape(
            batch * VIEWS, HYPOTHESES, self.config.feature_dim
        )
        memory = dense.reshape(
            batch * VIEWS, GRID_CELLS, self.config.feature_dim
        )
        traces = (
            [query.reshape(batch, VIEWS, HYPOTHESES, -1)]
            if self.config.trace_level == "full"
            else None
        )
        for layer in self.layers:
            query = layer(query, memory)
            if traces is not None:
                traces.append(query.reshape(batch, VIEWS, HYPOTHESES, -1))
        query = self.query_norm(query).reshape(
            batch, VIEWS, HYPOTHESES, self.config.feature_dim
        )
        pixel = self.pixel_projection(self.pixel_norm(dense))
        spatial_logits = torch.einsum(
            "bvsd,bvcd->bvsc", query, pixel
        ) / math.sqrt(self.config.feature_dim)
        absent_logits = self.absent_head(query)
        logits = torch.cat((spatial_logits, absent_logits), dim=-1).float()
        log_likelihood = F.log_softmax(logits, dim=-1)
        neutral = uniform_spatial_prior(
            batch, device=log_likelihood.device, dtype=log_likelihood.dtype
        )
        log_likelihood = torch.where(
            valid[:, :, None, None], log_likelihood, neutral
        )
        scene = normalized_dino
        return V176Measurement(
            scene_tokens=scene,
            dense_features=dense,
            anchor_hypothesis_log_likelihood=log_likelihood,
            view_valid=valid,
            query_traces=(
                torch.stack(traces, dim=1) if traces is not None else None
            ),
        )


class V176GroundingEncoder(nn.Module):
    def __init__(self, config: V176Config = V176Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.belief_projection = nn.Sequential(
            nn.Linear(64 + 8, config.compact_dim),
            nn.GELU(),
            nn.Linear(config.compact_dim, config.compact_dim),
        )
        self.measurement_status_embedding = nn.Parameter(
            torch.empty(2, config.compact_dim)
        )
        self.dynamics_norm = nn.LayerNorm(config.compact_dim)
        nn.init.normal_(self.measurement_status_embedding, std=0.02)

    @staticmethod
    def statistics(log_belief: torch.Tensor) -> torch.Tensor:
        return belief_statistics(log_belief)

    def forward(
        self,
        log_posterior: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        posterior = _normalized_log_belief("log_posterior", log_posterior)
        batch = len(posterior)
        posterior = _shape(
            "log_posterior",
            posterior,
            (batch, VIEWS, ANCHORS, BELIEF_CLASSES),
        )
        posterior = posterior.to(
            device=self.measurement_status_embedding.device,
            dtype=self.belief_projection[0].weight.dtype,
        )
        valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(
            posterior.device
        )
        visible = posterior.exp()[..., :GRID_CELLS].reshape(
            batch * VIEWS * ANCHORS, 1, GRID_SIZE, GRID_SIZE
        )
        coarse = F.adaptive_avg_pool2d(visible, (8, 8)).reshape(
            batch, VIEWS, ANCHORS, 64
        )
        statistics = self.statistics(posterior)
        # The transported posterior remains a valid state estimate even when
        # current pixels are missing.  Status prevents a propagated prior from
        # masquerading as a fresh measurement.  No appearance feature enters
        # this dynamics representation.
        status_index = valid.long()
        status = self.measurement_status_embedding[status_index][..., None, :]
        belief_state = self.belief_projection(
            torch.cat((coarse, statistics.to(coarse)), dim=-1)
        ) + status
        dynamics_grounding = self.dynamics_norm(belief_state)
        return dynamics_grounding, statistics


class V176BeliefLeWM(nn.Module):
    """SO-101 action-conditioned belief transport without dense features.

    The six-joint contract and scales below are deployment-specific; this
    implementation does not make a cross-arm morphology claim.
    """

    def __init__(self, config: V176Config = V176Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dimension = config.feature_dim
        self.grounding_projection = nn.Linear(config.compact_dim, dimension)
        self.q_projection = nn.Sequential(
            nn.Linear(STATE_DIM, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.command_projection = nn.Sequential(
            nn.Linear(STATE_DIM, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.readback_delta_projection = nn.Sequential(
            nn.Linear(STATE_DIM, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.readback_error_projection = nn.Sequential(
            nn.Linear(STATE_DIM, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.readback_present_embedding = nn.Parameter(torch.empty(dimension))
        self.readback_null_embedding = nn.Parameter(torch.empty(dimension))
        self.readback_time_embedding = nn.Parameter(
            torch.empty(1, COMMAND_STEPS, dimension)
        )
        self.view_embedding = nn.Parameter(torch.empty(1, VIEWS, 1, dimension))
        self.time_embedding = nn.Parameter(torch.empty(1, PATH_STEPS, dimension))
        self.layers = nn.ModuleList(
            _EncoderLayer(
                dimension, config.heads, config.ffn_multiplier * dimension
            )
            for _ in range(config.lewm_layers)
        )
        self.output_norm = nn.LayerNorm(dimension)
        self.shift_head = nn.Linear(dimension, 2)
        self.sigma_head = nn.Linear(dimension, 2)
        self.q_residual_head = nn.Linear(dimension, STATE_DIM)
        cap, lower, upper = _action_contract()
        self.register_buffer("joint_step_scale", cap.float(), persistent=True)
        self.register_buffer(
            "servo_error_scale",
            torch.tensor(config.servo_error_scale_deg, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "joint_center", ((lower + upper) * 0.5).float(), persistent=True
        )
        self.register_buffer(
            "joint_half_range", ((upper - lower) * 0.5).float(), persistent=True
        )
        for value in (
            self.view_embedding,
            self.time_embedding,
            self.readback_present_embedding,
            self.readback_null_embedding,
            self.readback_time_embedding,
        ):
            nn.init.normal_(value, std=0.02)
        for head in (self.shift_head, self.sigma_head, self.q_residual_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        grounded: V176GroundedState,
        current_q: torch.Tensor,
        commands: torch.Tensor,
        recent_readback_deltas: torch.Tensor | None = None,
        recent_readback_errors: torch.Tensor | None = None,
    ) -> V176LeWMPrediction:
        grounding = _finite(
            "dynamics_grounding", grounded.dynamics_grounding
        )
        batch = len(grounding)
        grounding = _shape(
            "dynamics_grounding",
            grounding,
            (batch, VIEWS, ANCHORS, self.config.compact_dim),
        )
        belief = _normalized_log_belief(
            "grounded.log_posterior", grounded.log_posterior
        ).to(grounding)
        valid = _bool(
            "grounded.view_valid", grounded.view_valid, (batch, VIEWS)
        ).to(grounding.device)
        state = _shape("current_q", current_q, (batch, STATE_DIM)).to(grounding)
        action = _shape(
            "commands", commands, (batch, COMMAND_STEPS, STATE_DIM)
        ).to(grounding)
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
                self.readback_null_embedding.reshape(1, 1, -1)
                + self.readback_time_embedding
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
        anchors = self.grounding_projection(grounding)
        # No anchor-index embedding: the two hypotheses are an exchangeable
        # set until physics establishes identity.  This keeps archive training
        # equivariant when camera geometry is unavailable.
        anchors = anchors + self.view_embedding
        anchors = anchors.reshape(batch, VIEWS * ANCHORS, self.config.feature_dim)
        state_normalized = (
            state - self.joint_center
        ) / self.joint_half_range
        previous_command = torch.cat((state[:, None], action[:, :-1]), dim=1)
        command_delta = (action - previous_command) / self.joint_step_scale
        motion = torch.cat(
            (
                self.q_projection(state_normalized)[:, None],
                self.command_projection(command_delta),
            ),
            dim=1,
        ) + self.time_embedding
        # Only the most recently completed chunk is retained.  Its measured
        # motion and servo-error rows remain ordered, so factual and reversed
        # readback are distinct.
        # This process history is available to LeWM only, never to the Actor.
        sequence = torch.cat((anchors, readback_tokens, motion), dim=1)
        traces = [sequence] if self.config.trace_level == "full" else None
        for layer in self.layers:
            # Current view validity is already encoded in G.  A missing image
            # does not invalidate the maintained prior state, so anchor tokens
            # must not be key-padded here.
            sequence = layer(sequence)
            if traces is not None:
                traces.append(sequence)
        sequence = self.output_norm(sequence)
        anchor_hidden = sequence[:, : VIEWS * ANCHORS].reshape(
            batch, VIEWS, ANCHORS, self.config.feature_dim
        )
        shift = torch.tanh(self.shift_head(anchor_hidden))
        shift = shift * self.config.max_transport_shift_px
        sigma_fraction = torch.sigmoid(self.sigma_head(anchor_hidden))
        sigma = self.config.minimum_transport_sigma_px + sigma_fraction * (
            self.config.maximum_transport_sigma_px
            - self.config.minimum_transport_sigma_px
        )
        # Missing current pixels do not mean that the robot stops moving.  A
        # view may be invalid for measurement while its belief prior must still
        # propagate under the executed action.
        prior = propagate_belief(
            belief,
            shift,
            sigma,
            uniform_mix=self.config.transport_uniform_mix,
            blur_radius=self.config.blur_radius,
        )
        motion_hidden = sequence[:, VIEWS * ANCHORS + COMMAND_STEPS :]
        baseline = torch.cat((state[:, None], action), dim=1)
        predicted_q = baseline + (
            self.q_residual_head(motion_hidden) * self.joint_step_scale
        )
        predicted_q = torch.cat((state[:, None], predicted_q[:, 1:]), dim=1)
        return V176LeWMPrediction(
            prior_log_belief=prior,
            predicted_q_path=predicted_q,
            transport_shift_px=shift,
            transport_sigma_px=sigma,
            tokens_by_layer=(
                torch.stack(traces, dim=1) if traces is not None else None
            ),
        )


class V176InnovationEncoder(nn.Module):
    def __init__(self, config: V176Config = V176Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.belief_projection = nn.Sequential(
            nn.Linear(64 + 8 + 1, config.compact_dim),
            nn.GELU(),
            nn.Linear(config.compact_dim, config.compact_dim),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(PATH_STEPS * STATE_DIM, config.compact_dim),
            nn.GELU(),
            nn.Linear(config.compact_dim, config.compact_dim),
        )
        self.present_embedding = nn.Parameter(torch.empty(config.compact_dim))
        self.null_embedding = nn.Parameter(torch.empty(config.compact_dim))
        self.output_norm = nn.LayerNorm(config.compact_dim)
        self.register_buffer(
            "servo_error_scale",
            torch.tensor(config.servo_error_scale_deg, dtype=torch.float32),
            persistent=True,
        )
        nn.init.normal_(self.present_embedding, std=0.02)
        nn.init.normal_(self.null_embedding, std=0.02)

    def forward(self, signals: V176InnovationSignals) -> torch.Tensor:
        delta = _finite("probability_delta", signals.probability_delta)
        batch = len(delta)
        _shape(
            "probability_delta",
            delta,
            (batch, VIEWS, ANCHORS, BELIEF_CLASSES),
        )
        residual = _shape(
            "state_residual",
            signals.state_residual,
            (batch, PATH_STEPS, STATE_DIM),
        ).to(delta)
        valid = _bool(
            "view_valid", signals.view_valid, (batch, VIEWS)
        ).to(delta.device)
        visible = delta[..., :GRID_CELLS].reshape(
            batch * VIEWS * ANCHORS, 1, GRID_SIZE, GRID_SIZE
        )
        coarse = F.adaptive_avg_pool2d(visible, (8, 8)).reshape(
            batch, VIEWS, ANCHORS, 64
        )
        belief = self.belief_projection(
            torch.cat(
                (
                    coarse,
                    signals.statistic_delta.to(delta),
                    signals.surprise[..., None].to(delta),
                ),
                dim=-1,
            )
        )
        state = self.state_projection(
            (residual / self.servo_error_scale).reshape(batch, -1)
        )[:, None, None]
        belief = belief * valid[:, :, None, None]
        # State-path mismatch remains usable when camera measurement is absent.
        return self.output_norm(belief + state + self.present_embedding)

    def null(
        self,
        batch: int,
        view_valid: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        _bool("view_valid", view_valid, (int(batch), VIEWS))
        value = self.output_norm(self.null_embedding).to(
            device=device, dtype=dtype
        )
        return value.reshape(1, 1, 1, -1).expand(
            int(batch), VIEWS, ANCHORS, -1
        )


def _action_contract(
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor((4.0, 4.0, 4.0, 4.0, 4.0, 8.0), device=device),
        torch.tensor((-100.0, -100.0, -100.0, -100.0, -100.0, 0.0), device=device),
        torch.full((STATE_DIM,), 100.0, device=device),
    )


def _project_action_chunk(
    current_q: torch.Tensor,
    desired: torch.Tensor,
    step_cap: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    state = current_q
    rows: list[torch.Tensor] = []
    for step in range(COMMAND_STEPS):
        target = desired[:, step].clamp(min=lower, max=upper)
        state = state + (target - state).clamp(min=-step_cap, max=step_cap)
        rows.append(state)
    return torch.stack(rows, dim=1)


class V176Actor(nn.Module):
    """Current scene plus paired grounding/innovation action chunk policy."""

    def __init__(self, config: V176Config = V176Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dimension = config.actor_dim
        self.scene_projection = nn.Linear(VIT_DIM, dimension)
        track_state_dim = 2 * config.compact_dim
        self.track_projection = nn.Sequential(
            nn.LayerNorm(track_state_dim),
            nn.Linear(track_state_dim, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.appearance_projection = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        self.q_projection = nn.Sequential(
            nn.Linear(STATE_DIM, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.scene_view_embedding = nn.Parameter(
            torch.empty(1, VIEWS, 1, dimension)
        )
        self.track_view_embedding = nn.Parameter(
            torch.empty(1, VIEWS, 1, dimension)
        )
        self.appearance_status_embedding = nn.Parameter(
            torch.empty(2, dimension)
        )
        self.source_embedding = nn.Parameter(torch.empty(3, dimension))
        self.action_queries = nn.Parameter(
            torch.empty(1, COMMAND_STEPS, dimension)
        )
        self.action_time_embedding = nn.Parameter(
            torch.empty(1, COMMAND_STEPS, dimension)
        )
        self.layers = nn.ModuleList(
            _ActorLayer(
                dimension, config.heads, config.ffn_multiplier * dimension
            )
            for _ in range(config.actor_layers)
        )
        self.output_norm = nn.LayerNorm(dimension)
        self.action_head = nn.Linear(dimension, STATE_DIM)
        cap, lower, upper = _action_contract()
        self.register_buffer("step_cap", cap.float(), persistent=True)
        self.register_buffer("action_lower", lower.float(), persistent=True)
        self.register_buffer("action_upper", upper.float(), persistent=True)
        self.register_buffer(
            "joint_center", ((lower + upper) * 0.5).float(), persistent=True
        )
        self.register_buffer(
            "joint_half_range", ((upper - lower) * 0.5).float(), persistent=True
        )
        for value in (
            self.scene_view_embedding,
            self.track_view_embedding,
            self.appearance_status_embedding,
            self.source_embedding,
            self.action_queries,
            self.action_time_embedding,
        ):
            nn.init.normal_(value, std=0.02)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(
        self,
        scene: torch.Tensor,
        dynamics_grounding: torch.Tensor,
        ordered_tokens: torch.Tensor,
        innovation: torch.Tensor,
        current_q: torch.Tensor,
        appearance_available: torch.Tensor,
    ) -> V176ActorOutput:
        scene = _finite("scene", scene)
        batch = len(scene)
        scene = _shape(
            "scene", scene, (batch, VIEWS, VIT_PATCHES, VIT_DIM)
        ).to(dtype=self.scene_projection.weight.dtype)
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
            "innovation",
            innovation,
            (batch, VIEWS, ANCHORS, self.config.compact_dim),
        ).to(scene)
        state = _shape("current_q", current_q, (batch, STATE_DIM)).to(scene)
        available = _bool(
            "appearance_available",
            appearance_available,
            (batch, VIEWS),
        ).to(scene.device)
        available_tracks = available[:, :, None]
        scene_group = (
            self.scene_projection(scene)
            + self.scene_view_embedding
            + self.source_embedding[0]
        ).reshape(batch, VIEWS * VIT_PATCHES, self.config.actor_dim)
        # Pair each track's current state with its own innovation before set
        # processing.  Jointly permuting anonymous tracks leaves the action
        # unchanged; swapping innovation alone changes the pairing.
        track_group = (
            self.track_projection(torch.cat((grounded, error), dim=-1))
            + self.appearance_projection(appearance)
            * available_tracks[..., None].to(appearance)
            + self.appearance_status_embedding[available_tracks.long()]
            + self.track_view_embedding
            + self.source_embedding[1]
        ).reshape(batch, VIEWS * ANCHORS, self.config.actor_dim)
        q_group = (
            self.q_projection(
                (state - self.joint_center) / self.joint_half_range
            )
            + self.source_embedding[2]
        )[:, None]
        memory = torch.cat((scene_group, track_group, q_group), dim=1)
        query = self.action_queries.expand(batch, -1, -1)
        query = query + self.action_time_embedding
        causal = torch.triu(
            torch.ones(
                COMMAND_STEPS,
                COMMAND_STEPS,
                dtype=torch.bool,
                device=query.device,
            ),
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
            state,
            desired,
            self.step_cap,
            self.action_lower,
            self.action_upper,
        )
        return V176ActorOutput(
            commands=commands,
            desired_commands=desired,
            increments=increments,
            action_queries_by_layer=(
                torch.stack(traces, dim=1) if traces is not None else None
            ),
        )


class V176Core(nn.Module):
    """Public FCPG -> LeWM -> predict/update -> Actor core."""

    def __init__(self, config: V176Config = V176Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.measurement = V176ExchangeableMeasurement(config)
        self.grounding_encoder = V176GroundingEncoder(config)
        self.lewm = V176BeliefLeWM(config)
        self.innovation_encoder = V176InnovationEncoder(config)
        self.actor = V176Actor(config)

    def measure(
        self,
        dino_tokens: torch.Tensor,
        view_valid: torch.Tensor,
        rgb: torch.Tensor | None = None,
    ) -> V176Measurement:
        return self.measurement(dino_tokens, view_valid, rgb)

    def measure_from_tokens(
        self, dino_tokens: torch.Tensor, view_valid: torch.Tensor
    ) -> V176Measurement:
        return self.measure(dino_tokens, view_valid, rgb=None)

    def assign_identity(
        self,
        measurement: V176Measurement,
        ordered_prior_log_belief: torch.Tensor | None = None,
        *,
        direct_assignment: torch.Tensor | None = None,
    ) -> V176AssignedMeasurement:
        ordered, assignment = identity_or_track_assignment(
            measurement.anchor_hypothesis_log_likelihood,
            ordered_prior_log_belief,
            measurement.view_valid,
            direct_assignment=direct_assignment,
        )
        return V176AssignedMeasurement(
            scene_tokens=measurement.scene_tokens,
            dense_features=measurement.dense_features,
            ordered_log_likelihood=ordered,
            assignment=assignment,
            view_valid=measurement.view_valid,
            query_traces=measurement.query_traces,
        )

    def predict(
        self,
        grounded: V176GroundedState,
        current_q: torch.Tensor,
        commands: torch.Tensor,
        recent_readback_deltas: torch.Tensor | None = None,
        recent_readback_errors: torch.Tensor | None = None,
    ) -> V176LeWMPrediction:
        return self.lewm(
            grounded,
            current_q,
            commands,
            recent_readback_deltas=recent_readback_deltas,
            recent_readback_errors=recent_readback_errors,
        )

    def update(
        self,
        assigned_measurement: V176AssignedMeasurement,
        prediction: V176LeWMPrediction | None = None,
        measured_q_path: torch.Tensor | None = None,
    ) -> V176FilterUpdate:
        batch = len(assigned_measurement.view_valid)
        if prediction is None:
            if measured_q_path is not None:
                raise V176ContractError(
                    "H0 update cannot accept a measured path without prediction"
                )
            prior = uniform_spatial_prior(
                batch,
                device=assigned_measurement.ordered_log_likelihood.device,
                dtype=assigned_measurement.ordered_log_likelihood.dtype,
            )
        else:
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
        # Invalid current pixels may not leak appearance into the Actor.  The
        # maintained posterior still reaches LeWM through dynamics_grounding.
        ordered_tokens = ordered_tokens * assigned_measurement.view_valid[
            :, :, None, None
        ].to(ordered_tokens)
        dynamics_grounding, statistics = self.grounding_encoder(
            posterior,
            assigned_measurement.view_valid,
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
            residual = dynamics_grounding.new_zeros(
                batch, PATH_STEPS, STATE_DIM
            )
            surprise = dynamics_grounding.new_zeros(
                batch, VIEWS, ANCHORS
            )
            present = torch.zeros(
                batch, dtype=torch.bool, device=dynamics_grounding.device
            )
        else:
            if measured_q_path is None:
                raise V176ContractError(
                    "predicted update requires the realized q path"
                )
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
            present = torch.ones(
                batch, dtype=torch.bool, device=dynamics_grounding.device
            )
        innovation = V176Innovation(
            innovation=compact,
            belief_surprise=surprise,
            state_residual=residual,
            view_valid=assigned_measurement.view_valid,
            has_prediction=present,
        )
        return V176FilterUpdate(grounded=grounded, innovation=innovation)

    def act(
        self,
        scene_tokens: torch.Tensor,
        grounded: V176GroundedState,
        innovation: torch.Tensor,
        current_q: torch.Tensor,
    ) -> V176ActorOutput:
        batch = len(grounded.ordered_tokens)
        valid = _bool(
            "grounded.view_valid", grounded.view_valid, (batch, VIEWS)
        ).to(grounded.ordered_tokens.device)
        appearance_available = _bool(
            "grounded.appearance_available",
            grounded.appearance_available,
            (batch, VIEWS),
        ).to(grounded.ordered_tokens.device)
        actor_tokens = grounded.ordered_tokens * appearance_available[
            :, :, None, None
        ].to(grounded.ordered_tokens)
        actor_scene = scene_tokens * valid[:, :, None, None].to(scene_tokens)
        return self.actor(
            actor_scene,
            grounded.dynamics_grounding,
            actor_tokens,
            innovation,
            current_q,
            appearance_available,
        )


V176PredictionErrorController = V176Core


__all__ = [
    "ABSENT_INDEX",
    "ACTOR_DIM",
    "ANCHORS",
    "BELIEF_CLASSES",
    "COMMAND_STEPS",
    "COMPACT_DIM",
    "FEATURE_DIM",
    "GRID_CELLS",
    "GRID_SIZE",
    "IMAGE_SIZE",
    "PATH_STEPS",
    "HYPOTHESES",
    "STATE_DIM",
    "VIEWS",
    "VIEW_ORDER",
    "VIT_DIM",
    "VIT_PATCHES",
    "V176Actor",
    "V176ActorOutput",
    "V176AssignedMeasurement",
    "V176BeliefLeWM",
    "V176Config",
    "V176ContractError",
    "V176Core",
    "V176ExchangeableMeasurement",
    "V176FilterUpdate",
    "V176GroundedState",
    "V176GroundingEncoder",
    "V176Innovation",
    "V176InnovationEncoder",
    "V176InnovationSignals",
    "V176LeWMPrediction",
    "V176Measurement",
    "V176PredictionErrorController",
    "action_chunk_l1",
    "belief_prediction_nll",
    "compute_innovation",
    "exchangeable_pair_nll",
    "fuse_measurement_prior",
    "posterior_weighted_pool",
    "propagate_belief",
    "identity_or_track_assignment",
    "normalize_actor_scene_tokens",
    "state_path_smooth_l1",
    "uniform_spatial_prior",
]
