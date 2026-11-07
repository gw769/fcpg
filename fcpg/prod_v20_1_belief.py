"""Joint jaw-state Gaussian mixtures for V20.1.

The continuous state is ``(cx, cy, dx, dy)`` with F=c-d and M=c+d.
Covariance is a full 4x4 matrix, so common translation and uncertain aperture
are different random variables.  Zero projected opening remains legal.

Visual networks produce discriminative evidence, not a generative image
likelihood.  The Gaussian-product update below is a robust evidence fusion
operator requiring calibration validation, NOT exact Bayesian filtering. It retains four components
by explicitly moment matching all products belonging to one visual component.
This approximation preserves their first two moments, not every sub-mode.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch

IMAGE_SIZE = 224
VIEW_ORDER = ("wrist", "column")
PAIR_COMPONENTS = 4
FEATURE_DIM = 384
BELIEF_DTYPE = torch.float64


def _endpoint_matrix(reference):
    return reference.new_tensor([[1, 0, -1, 0], [0, 1, 0, -1],
                                 [1, 0, 1, 0], [0, 1, 0, 1]])


def _state_matrix(reference):
    return reference.new_tensor([[.5, 0, .5, 0], [0, .5, 0, .5],
                                 [-.5, 0, .5, 0], [0, -.5, 0, .5]])


def _symmetrize(covariance):
    return (covariance + covariance.transpose(-1, -2)) * .5


def _small_matrix_product(left, right):
    """Full-precision product for tiny covariance/coordinate systems.

    ``autocast(False)`` alone does not disable TF32 GEMM. On RTX 6000D,
    the two 4x4 endpoint/state congruence transforms can otherwise destroy
    positive definiteness when an off-image prior makes mixture covariance
    highly anisotropic. The probability chain retains fp64 from covariance
    construction onward, including backward, without changing neural GEMM policy.
    This is the same matrix product, not eigenvalue clipping or extra noise.
    """
    with torch.autocast(device_type=left.device.type, enabled=False):
        return (left.to(BELIEF_DTYPE).unsqueeze(-1) * right.to(BELIEF_DTYPE).unsqueeze(-3)).sum(-2)


@dataclass(frozen=True)
class JointPairBelief:
    log_weights: torch.Tensor  # [B,2,4]
    state_mean: torch.Tensor  # [B,2,4,4], centre and half-separation
    state_covariance: torch.Tensor  # [B,2,4,4,4], full SPD
    visible_probability: torch.Tensor  # [B,2,4,2]
    identity_probability: torch.Tensor | None = None  # P(first endpoint is F)

    def __post_init__(self):
        if self.identity_probability is None:
            object.__setattr__(self, "identity_probability", torch.full_like(self.log_weights, .5))
        # A broad out-of-frame mode can coexist with a subpixel visible mode.
        # Keeping only the final inverse in fp64 is too late: fp32 congruence
        # transforms can already erase the smaller mode's positive variance.
        for field in fields(self):
            object.__setattr__(self, field.name, getattr(self, field.name).to(BELIEF_DTYPE))

    @property
    def weights(self):
        return self.log_weights.exp()

    @property
    def center_px(self):
        return self.state_mean[..., :2]

    @property
    def means_px(self):
        center, half = self.state_mean[..., :2], self.state_mean[..., 2:]
        return torch.stack((center - half, center + half), -2)

    @property
    def opening_px(self):
        return 2 * self.state_mean[..., 2:].norm(dim=-1)

    @property
    def direction(self):
        half = self.state_mean[..., 2:]
        norm = half.norm(dim=-1, keepdim=True)
        default = torch.zeros_like(half)
        default[..., 0] = 1
        return torch.where(norm > 1e-8, half / norm.clamp_min(1e-8), default)

    @property
    def endpoint_joint_covariance(self):
        covariance = self.state_covariance
        matrix = _endpoint_matrix(covariance)
        with torch.autocast(device_type=covariance.device.type, enabled=False):
            return _symmetrize(_small_matrix_product(_small_matrix_product(matrix, covariance), matrix.T))

    @property
    def tip_covariance_px2(self):
        covariance = self.endpoint_joint_covariance
        return torch.stack((covariance[..., :2, :2], covariance[..., 2:, 2:]), -3)

    @property
    def covariance_px2(self):
        """Compatibility diagnostic only; never used for joint likelihood."""
        return self.tip_covariance_px2.mean(-3)

    def detach(self):
        return JointPairBelief(**{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def to(self, *args, **kwargs):
        # Device movement must never round a stored distribution via a neural
        # fp16/bf16/fp32 dtype request. Resolve only the requested destination
        # with an empty tensor; the real belief remains probability precision.
        destination = self.state_mean.new_empty(0).to(*args, **kwargs)
        return JointPairBelief(**{f.name: getattr(self, f.name).to(
            device=destination.device, dtype=BELIEF_DTYPE,
            non_blocking=kwargs.get("non_blocking", False), copy=kwargs.get("copy", False))
            for f in fields(self)})

    def validate(self):
        batch = self.log_weights.shape[0]
        shapes = {"log_weights": (batch, 2, 4), "state_mean": (batch, 2, 4, 4),
                  "state_covariance": (batch, 2, 4, 4, 4),
                  "visible_probability": (batch, 2, 4, 2),
                  "identity_probability": (batch, 2, 4)}
        for name, shape in shapes.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid joint pair belief {name}, expected {shape}")
        if not torch.allclose(self.log_weights.logsumexp(-1),
                              torch.zeros_like(self.log_weights[..., 0]), atol=2e-4):
            raise ValueError("joint pair weights must normalize")
        if ((self.visible_probability < 0) | (self.visible_probability > 1)).any():
            raise ValueError("visibility probability outside [0,1]")
        if ((self.identity_probability < 0) | (self.identity_probability > 1)).any():
            raise ValueError("identity probability outside [0,1]")
        covariance = self.state_covariance
        if not torch.allclose(covariance, covariance.transpose(-1, -2), atol=1e-5):
            raise ValueError("joint covariance must be symmetric")
        if (torch.linalg.cholesky_ex(covariance).info != 0).any():
            raise ValueError("joint covariance must be positive definite")


def pair_from_means(means_px, covariance_px2=None, visible_probability=None,
                    log_weights=None, *, joint_covariance=None, identity_probability=None):
    """Explicit endpoint-to-state adapter; full endpoint covariance preferred.

    ``covariance_px2`` accepts shared [...,2,2] or per-tip [...,2,2,2]
    marginals.  In that compatibility case cross-tip covariance is zero, not
    guessed from a mechanical model.  ``joint_covariance`` is [...,4,4] in
    (Fx,Fy,Mx,My) order and must not be supplied alongside marginals.
    """
    if means_px.shape[-2:] != (2, 2):
        raise ValueError("pair means must end in [2,2]")
    means_px = means_px.to(BELIEF_DTYPE)
    if joint_covariance is not None and covariance_px2 is not None:
        raise ValueError("provide either joint covariance or tip marginals")
    if joint_covariance is None:
        if covariance_px2 is None:
            raise ValueError("pair covariance is required")
        base_shape = means_px.shape[:-2]
        joint_covariance = means_px.new_zeros(*base_shape, 4, 4)
        if covariance_px2.shape == (*base_shape, 2, 2):
            a = b = covariance_px2
        elif covariance_px2.shape == (*base_shape, 2, 2, 2):
            a, b = covariance_px2.unbind(-3)
        else:
            raise ValueError("tip covariance shape differs from pair means")
        joint_covariance[..., :2, :2] = a
        joint_covariance[..., 2:, 2:] = b
    if joint_covariance.shape != (*means_px.shape[:-2], 4, 4):
        raise ValueError("endpoint joint covariance must be [...,4,4]")
    joint_covariance = joint_covariance.to(BELIEF_DTYPE)
    matrix = _state_matrix(means_px)
    with torch.autocast(device_type=means_px.device.type, enabled=False):
        state = _small_matrix_product(means_px.flatten(-2).unsqueeze(-2), matrix.T).squeeze(-2)
        covariance = _symmetrize(_small_matrix_product(_small_matrix_product(matrix, joint_covariance), matrix.T))
    if visible_probability is None:
        visible_probability = torch.full_like(means_px[..., 0], .5)
    if log_weights is None:
        log_weights = torch.full_like(means_px[..., 0, 0], -math.log(PAIR_COMPONENTS))
    return JointPairBelief(log_weights.log_softmax(-1), state, covariance, visible_probability, identity_probability)


def broad_pair_prior(batch, device=None, dtype=BELIEF_DTYPE):
    means = torch.zeros(batch, 2, 4, 4, device=device, dtype=dtype)
    means[..., :2] = 111.5
    covariance = torch.eye(4, device=device, dtype=dtype).expand(batch, 2, 4, 4, 4).clone() * IMAGE_SIZE**2
    return JointPairBelief(torch.full((batch, 2, 4), -math.log(4), device=device, dtype=dtype),
                           means, covariance, torch.full((batch, 2, 4, 2), .5, device=device, dtype=dtype))


def select_pair_belief(mask, yes, no):
    return JointPairBelief(**{f.name: torch.where(
        mask.reshape(*mask.shape, *([1] * (getattr(yes, f.name).ndim - mask.ndim))),
        getattr(yes, f.name), getattr(no, f.name)) for f in fields(yes)})


def pair_component_maps(belief, size=56):
    """Per-tip map from the true marginal, without an explicit matrix inverse.

    Solving the tiny SPD system in fp64 preserves small visible-tip variance
    alongside large off-image uncertainty. CNN fields remain bf16/fp32.
    """
    means = belief.means_px
    axis = (torch.arange(size, device=means.device, dtype=means.dtype) + .5) * (IMAGE_SIZE / size) - .5
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((x, y), -1).reshape(-1, 2)
    delta = grid - means[..., None, :]
    with torch.autocast(device_type=means.device.type, enabled=False):
        chol = torch.linalg.cholesky(belief.tip_covariance_px2)
        whitened = torch.linalg.solve_triangular(chol, delta.transpose(-1, -2), upper=False)
        mahalanobis = whitened.square().sum(-2)
        return (-.5 * mahalanobis).softmax(-1)


def posterior_weighted_pair_pool(belief, feature_map):
    """Full-map conditional component integral; never midpoint sampling."""
    if feature_map.ndim != 5 or feature_map.shape[:2] != belief.log_weights.shape[:2] or feature_map.shape[-2] != feature_map.shape[-1]:
        raise ValueError("pair pooling requires square [B,2,C,S,S] feature maps")
    maps = (pair_component_maps(belief, feature_map.shape[-1]) * belief.visible_probability[..., None]).mean(-2)
    # The normalized spatial integral crosses back to the neural precision
    # boundary here, never the stored means/covariances or Gaussian solves.
    with torch.autocast(device_type=feature_map.device.type, enabled=False):
        return torch.einsum("bvkn,bvcn->bvkc", maps.float(), feature_map.float().flatten(-2))


def _gaussian_log_density(delta, covariance):
    # Probability precision starts before construction/moment reduction, not
    # merely at factorization. No repair is attempted for invalid covariance.
    delta, covariance = delta.to(BELIEF_DTYPE), covariance.to(BELIEF_DTYPE)
    with torch.autocast(device_type=delta.device.type, enabled=False):
        chol = torch.linalg.cholesky(covariance)
        solve = torch.cholesky_solve(delta[..., None], chol).squeeze(-1)
        logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -.5 * ((delta * solve).sum(-1) + logdet + delta.shape[-1] * math.log(2 * math.pi))


def pair_nll(belief, target_fm_px, valid, ordered=True, *, visibility=None,
             visibility_valid=None, reduction="mean", identity_aware=False):
    """Proper full-pair or observed-subset marginal Gaussian mixture NLL.

    Unordered labels use log[p(F,M)+p(M,F)]-log(2), the normalized
    symmetrized density on a labeled pair.  This is exact summation, not hard
    min matching.  Subtracting log(2) only fixes the density convention.
    Missing points are marginalized, never set to a guessed coordinate.
    Unknown visibility contributes no Bernoulli supervision.
    ``identity_aware=True`` makes ordered labels physical F/M labels: each
    component marginalizes direct/swap using its identity_probability before
    mixing components.  False scores the incoming geometric gauge (raw-head
    compatibility).  Unordered density is independent of identity probability.
    """
    target = target_fm_px.to(BELIEF_DTYPE)
    if target.shape != (*belief.log_weights.shape[:2], 2, 2):
        raise ValueError("pair labels must be [B,2,2,2]")
    point_valid = torch.as_tensor(valid, device=target.device, dtype=torch.bool)
    if point_valid.ndim == 2:
        point_valid = point_valid[..., None].expand(-1, -1, 2)
    if point_valid.shape != target.shape[:-1]:
        raise ValueError("valid must be [B,2] or [B,2,2]")
    has_visibility = visibility is not None
    visible = torch.ones_like(point_valid) if visibility is None else torch.as_tensor(visibility, device=target.device, dtype=torch.bool)
    known = torch.ones_like(point_valid) if visibility_valid is None else torch.as_tensor(visibility_valid, device=target.device, dtype=torch.bool)
    if visible.shape != point_valid.shape or known.shape != point_valid.shape:
        raise ValueError("visibility masks must be [B,2,2]")
    finite = torch.isfinite(target).all(-1)
    if (point_valid & visible & ~finite).any():
        raise ValueError("visible supervised tips require finite labels")
    target = torch.where(finite[..., None], target, torch.zeros_like(target))
    means, covariance = belief.means_px, belief.endpoint_joint_covariance
    probability = belief.visible_probability.clamp(1e-6, 1 - 1e-6)

    def assigned(order):
        assigned_target = target[..., order, :]
        assigned_valid, assigned_visible = point_valid[..., order], visible[..., order]
        spatial = assigned_valid & assigned_visible
        # All four possible observed subsets are exact marginal densities.
        score = torch.zeros_like(belief.log_weights)
        for indices, tip_mask in (((0, 1), (True, False)), ((2, 3), (False, True)),
                                  ((0, 1, 2, 3), (True, True))):
            mask = (spatial == spatial.new_tensor(tip_mask)).all(-1)
            delta = assigned_target.flatten(-2)[..., None, list(indices)] - means.flatten(-2)[..., list(indices)]
            subcov = covariance[..., list(indices), :][..., list(indices)]
            density = _gaussian_log_density(delta, subcov)
            score = score + torch.where(mask[..., None], density, torch.zeros_like(density))
        if has_visibility:
            bernoulli = torch.where(assigned_visible[..., None, :], probability.log(), torch.log1p(-probability))
            visibility_mask = assigned_valid & known[..., order]
            score = score + torch.where(visibility_mask[..., None, :], bernoulli, torch.zeros_like(bernoulli)).sum(-1)
        return belief.log_weights + score

    direct_component, swapped_component = assigned([0, 1]), assigned([1, 0])
    direct, swapped = direct_component.logsumexp(-1), swapped_component.logsumexp(-1)
    ordered = torch.as_tensor(ordered, dtype=torch.bool, device=target.device)
    if ordered.ndim == 1:
        ordered = ordered[:, None]
    symmetric = torch.logaddexp(direct, swapped) - math.log(2)
    if identity_aware:
        probability = belief.identity_probability
        def safe_log(value):
            return torch.where(value > 0, value.clamp_min(torch.finfo(value.dtype).tiny).log(),
                               torch.full_like(value, -float("inf")))
        direct = torch.logaddexp(direct_component + safe_log(probability),
                                 swapped_component + safe_log(1 - probability)).logsumexp(-1)
    loss = -torch.where(ordered, direct, symmetric)
    view_valid = point_valid.any(-1)
    loss = torch.where(view_valid, loss, torch.zeros_like(loss))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError("reduction must be mean, sum or none")
    counts = view_valid.sum(0)
    return ((loss.sum(0) / counts.clamp_min(1)) * (counts > 0)).sum() / (counts > 0).sum().clamp_min(1)


def pair_log_prob(belief, target_fm_px, ordered=True, *, identity_aware=False):
    """Endpoint density [B,2], with exact marginalization over mixture modes."""
    valid = torch.ones(target_fm_px.shape[:2], dtype=torch.bool, device=target_fm_px.device)
    return -pair_nll(belief, target_fm_px, valid, ordered, reduction="none", identity_aware=identity_aware)


def fuse_pair_beliefs(measurement, prior, view_valid, reacquire_mass=.25, *,
                     identity_ambiguous=True, prior_floor=None):
    """Continuous joint Gaussian products and explicit four-component reduction.

    The two ends define a *geometric gauge*, not a hard physical F/M identity.
    Each prior component is aligned to the incoming visual component's gauge
    by greater Gaussian overlap; ties preserve the incoming gauge.  The two
    orientations are summed for evidence scoring, but the higher-overlap
    orientation is used for Gaussian products.  This is an explicit MAP-gauge
    approximation, not exact marginalization of product moments.  Crucially,
    opposing +/-d identities are NEVER averaged into a closed-jaw mean.

    Physical F/M probability travels separately in ``identity_probability``;
    equal identity probabilities stay ambiguous, without moving the pair.
    Aligned prior products are moment matched within each visual component,
    including between-product covariance.  The measurement-only contamination
    branch can reacquire far from an incorrect narrow prior.  No hard span
    floor, distance rejection gate, or appearance identity classifier exists.
    ``identity_ambiguous=False`` is only for already ordered evidence/tests.
    """
    if prior_floor is not None:
        reacquire_mass = prior_floor
    if not 0 <= reacquire_mass < 1:
        raise ValueError("reacquire_mass must be in [0,1)")
    view_valid = torch.as_tensor(view_valid, device=measurement.log_weights.device, dtype=torch.bool)
    if view_valid.shape != measurement.log_weights.shape[:2]:
        raise ValueError("view_valid must be [B,2]")
    if prior is None:
        unknown = broad_pair_prior(measurement.log_weights.shape[0], measurement.log_weights.device)
        return select_pair_belief(view_valid, measurement, unknown)
    m, cm = measurement.state_mean, measurement.state_covariance
    p, cp = prior.state_mean, prior.state_covariance
    visibility = measurement.visible_probability
    # Axes [B,V,Kmeasurement,Kprior,(state,state)].
    m_exp, cm_exp = m[..., None, :], cm[..., None, :, :]
    p_exp, cp_exp = p[:, :, None, :, :], cp[:, :, None, :, :, :]
    with torch.autocast(device_type=m.device.type, enabled=False):
        direct = _gaussian_log_density(m_exp - p_exp, cm_exp + cp_exp)
        prior_identity = prior.identity_probability[:, :, None, :]
        if identity_ambiguous:
            sign = m.new_tensor([1, 1, -1, -1])
            swapped_p = p_exp * sign
            swapped_cp = cp_exp * sign[:, None] * sign[None, :]
            swapped = _gaussian_log_density(m_exp - swapped_p, cm_exp + swapped_cp)
            swap = swapped > direct
            p_exp = torch.where(swap[..., None], swapped_p, p_exp)
            cp_exp = torch.where(swap[..., None, None], swapped_cp, cp_exp)
            # F/M confidence is soft even though the geometric gauge is MAP.
            same_probability = torch.stack((direct, swapped), -1).softmax(-1)[..., 0]
            aligned_identity = same_probability * prior_identity + (1 - same_probability) * (1 - prior_identity)
            overlap = torch.logaddexp(direct, swapped) - math.log(2)
        else:
            overlap, aligned_identity = direct, prior_identity.expand_as(direct)
        summed_covariance = cm_exp + cp_exp
        gain = torch.linalg.solve(summed_covariance, cm_exp).transpose(-1, -2)
        fused_mean = m_exp + (gain @ (p_exp - m_exp)[..., None]).squeeze(-1)
        # Joseph form avoids subtractive cancellation for a broad measurement
        # multiplied by a very narrow prior (common during reacquisition).
        complement = torch.eye(4, device=m.device, dtype=BELIEF_DTYPE) - gain
        fused_cov = _symmetrize(complement @ cm_exp @ complement.transpose(-1, -2)
                                + gain @ cp_exp @ gain.transpose(-1, -2))
        log_branch = overlap + prior.log_weights[:, :, None, :] + math.log1p(-reacquire_mass)
        # Preserve the full covariance, including dependence across tips.
        means, covariances, branch_logs = fused_mean, fused_cov, log_branch
        v = visibility[..., None, :].expand(*fused_mean.shape[:-1], 2)
        identities = aligned_identity
        if reacquire_mass:
            means = torch.cat((means, m[..., None, :]), -2)
            covariances = torch.cat((covariances, cm[..., None, :, :]), -3)
            v = torch.cat((v, visibility[..., None, :]), -2)
            identities = torch.cat((identities, measurement.identity_probability[..., None]), -1)
            diffuse = torch.full_like(m[..., :1], math.log(reacquire_mass) - 4 * math.log(IMAGE_SIZE))
            branch_logs = torch.cat((branch_logs, diffuse), -1)
        responsibilities = branch_logs.softmax(-1)
        mean = (responsibilities[..., None] * means).sum(-2)
        offset = means - mean[..., None, :]
        covariance = (responsibilities[..., None, None] *
                      (covariances + offset[..., :, None] * offset[..., None, :])).sum(-3)
        covariance = _symmetrize(covariance) + torch.eye(4, device=m.device, dtype=BELIEF_DTYPE) * 1e-5
        visible = (responsibilities[..., None] * v).sum(-2)
        identity = (responsibilities * identities).sum(-1)
        weights = (measurement.log_weights + branch_logs.logsumexp(-1)).log_softmax(-1)
        updated = JointPairBelief(weights, mean, covariance, visible, identity)
    return select_pair_belief(view_valid, updated, prior)
