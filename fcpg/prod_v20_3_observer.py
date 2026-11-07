"""V20.3 visibility-conditioned paired observation update.

The learned RGB/DINO measurement is unchanged. Current image visibility mixes
four observation models (neither endpoint, first, second, both); a model uses
only its observed-coordinate Gaussian marginal. There is no confidence cutoff,
policy router or extra neural module. Visibility is discriminatively predicted,
not presumed calibrated, so this is a robust evidence update, not exact Bayes.

Four-component correspondence is a small MAP mixture approximation. It aligns
visual hypotheses with prior modes before within-component moment reduction;
it is not action-candidate ranking. Keeping prior component identities allows
low-visibility updates to approach the complete prior mixture continuously.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import permutations
import math

import torch

from .prod_v20_1_belief import (
    IMAGE_SIZE, JointPairBelief, _gaussian_log_density, _small_matrix_product,
    broad_pair_prior, pair_from_means, select_pair_belief,
)
from .prod_v20_1_observer import ObserverOutput, POOL_CHANNELS, V201PairObserver


COMPONENT_PERMUTATIONS = tuple(permutations(range(4)))


def _gather_components(value, indices):
    suffix = value.shape[3:]
    index = indices.reshape(*indices.shape, *([1] * len(suffix))).expand(*indices.shape, *suffix)
    return value.gather(2, index)


def _align_measurement(measurement, prior):
    """One-to-one component and endpoint-gauge MAP correspondence.

    Matching costs use visible endpoint marginal overlap only. A hidden
    endpoint's location, covariance or appearance cannot choose the match.
    Discrete assignments have no gradients; selected Gaussian updates do.
    """
    m, p = measurement.means_px, prior.means_px
    cm, cp = measurement.tip_covariance_px2, prior.tip_covariance_px2
    v = measurement.visible_probability
    # [B,V,Kprior,Kmeasurement,endpoint,xy].
    direct = -_gaussian_log_density(m[:, :, None] - p[:, :, :, None],
                                    cm[:, :, None] + cp[:, :, :, None])
    swapped = -_gaussian_log_density(m[:, :, None] - p.flip(-2)[:, :, :, None],
                                     cm[:, :, None] + cp.flip(-3)[:, :, :, None])
    direct = (direct * v[:, :, None]).sum(-1)
    swapped = (swapped * v[:, :, None]).sum(-1)
    swap = swapped < direct
    cost = torch.minimum(direct, swapped)
    choices = torch.tensor(COMPONENT_PERMUTATIONS, device=m.device, dtype=torch.long)
    candidate_cost = cost[..., torch.arange(4, device=m.device), choices].sum(-1)
    assignment = choices[candidate_cost.argmin(-1)]  # prior component -> measurement component
    flip = swap.gather(-1, assignment[..., None]).squeeze(-1)
    means = _gather_components(m, assignment)
    covariance = _gather_components(measurement.endpoint_joint_covariance, assignment)
    visibility = _gather_components(v, assignment)
    identity = _gather_components(measurement.identity_probability, assignment)
    weights = _gather_components(measurement.log_weights, assignment)
    permutation = [2, 3, 0, 1]
    means = torch.where(flip[..., None, None], means.flip(-2), means)
    covariance = torch.where(flip[..., None, None], covariance[..., permutation, :][..., permutation], covariance)
    visibility = torch.where(flip[..., None], visibility.flip(-1), visibility)
    identity = torch.where(flip, 1 - identity, identity)
    return means.flatten(-2), covariance, visibility, identity, weights


def _observed_update(p, cp, m, cm, prior_log_weights, prior_identity,
                     measurement_identity, coordinates, reacquire_mass):
    """Gaussian marginal update plus contamination in observed coordinates only.

    With a block-independent prior, a hidden endpoint is unchanged. With
    cross-endpoint covariance it can move only via the prior's conditional
    correlation, never by a fabricated hidden-endpoint visual observation.
    """
    identity4 = torch.eye(4, device=p.device, dtype=p.dtype)
    selector = identity4[list(coordinates)]
    # Tiny systems must not use TF32 even when neural GEMMs do. Merely
    # disabling bf16 autocast is insufficient on the production GPU.
    mm = _small_matrix_product
    observed_m = mm(m.unsqueeze(-2), selector.T).squeeze(-2)
    observed_p = mm(p.unsqueeze(-2), selector.T).squeeze(-2)
    observation_covariance = mm(mm(selector, cm), selector.T)
    innovation_covariance = mm(mm(selector, cp), selector.T) + observation_covariance
    gain = torch.linalg.solve(innovation_covariance, mm(cp, selector.T).transpose(-1, -2)).transpose(-1, -2)
    mean = p + mm(gain, (observed_m - observed_p)[..., None]).squeeze(-1)
    complement = identity4 - mm(gain, selector)
    covariance = (mm(mm(complement, cp), complement.transpose(-1, -2))
                  + mm(mm(gain, observation_covariance), gain.transpose(-1, -2)))
    overlap = _gaussian_log_density(observed_m - observed_p, innovation_covariance)
    log_regular = prior_log_weights + math.log1p(-reacquire_mass) + overlap
    if reacquire_mass == 0:
        return mean, covariance, log_regular, prior_identity
    # Broad/no-information prior only for the observed subspace. For a partial
    # observation, the other endpoint keeps its prior marginal; its fictional
    # image coordinate never enters even the reacquisition branch.
    unobserved = identity4 - mm(selector.T, selector)
    broad_mean = mm(p.unsqueeze(-2), unobserved.T).squeeze(-2) + mm(observed_m.unsqueeze(-2), selector).squeeze(-2)
    broad_covariance = mm(mm(unobserved, cp), unobserved.T) + mm(mm(selector.T, observation_covariance), selector)
    log_broad = torch.full_like(log_regular, math.log(reacquire_mass) - len(coordinates) * math.log(IMAGE_SIZE))
    logs = torch.stack((log_regular, log_broad), -1)
    responsibility = logs.softmax(-1)
    means = torch.stack((mean, broad_mean), -2)
    covariances = torch.stack((covariance, broad_covariance), -3)
    fused_mean = (responsibility[..., None] * means).sum(-2)
    offset = means - fused_mean[..., None, :]
    fused_covariance = (responsibility[..., None, None] *
        (covariances + offset[..., :, None] * offset[..., None, :])).sum(-3)
    fused_identity = responsibility[..., 0] * prior_identity + responsibility[..., 1] * measurement_identity
    return fused_mean, fused_covariance, logs.logsumexp(-1), fused_identity


def fuse_v203_pair_beliefs(measurement: JointPairBelief, prior: JointPairBelief | None,
                           view_valid, reacquire_mass=.25, *, prior_floor=None):
    """Continuous visibility mixture, followed by four-mode moment reduction.

    Visibility pattern probabilities are taken from the current image model
    (independent Bernoulli endpoints, an explicit approximation), not multiplied
    by a second visibility likelihood. Geometric overlap reweights components
    *within* each pattern; it cannot turn a .01 visibility estimate into a full
    spatial overwrite merely because a conditional density is sharp.

    No-information probability multiplies the complete prior mixture. Exactly
    absent current observations retain prior spatial means/covariances/weights;
    their current visibility becomes zero. Invalid camera views preserve every
    prior field exactly. No future frames, synthetic actual states or online
    parameter adaptation are introduced.
    """
    if prior_floor is not None:
        reacquire_mass = prior_floor
    if not 0 <= reacquire_mass < 1:
        raise ValueError("reacquire_mass must be in [0,1)")
    valid = torch.as_tensor(view_valid, device=measurement.log_weights.device, dtype=torch.bool)
    if valid.shape != measurement.log_weights.shape[:2]:
        raise ValueError("view_valid must be [B,2]")
    if prior is None:
        prior = broad_pair_prior(len(measurement.log_weights), measurement.log_weights.device)
    # Even NaN pixels in an invalid camera must never enter covariance algebra.
    safe_measurement = select_pair_belief(valid, measurement, prior)
    v = safe_measurement.visible_probability
    if not torch.isfinite(v).all() or ((v < 0) | (v > 1)).any():
        raise ValueError("measurement visibility must be finite probabilities")
    with torch.autocast(device_type=measurement.state_mean.device.type, enabled=False):
        m, cm, visibility, measured_identity, log_measurement_weight = _align_measurement(safe_measurement, prior)
        p, cp = prior.means_px.flatten(-2), prior.endpoint_joint_covariance
        prior_weight, prior_identity = prior.weights, prior.identity_probability
        vf, vm = visibility.unbind(-1)
        # Pattern axis: neither, first only, second only, both.
        probability = torch.stack(((1-vf)*(1-vm), vf*(1-vm), (1-vf)*vm, vf*vm), -1)
        image_weight = log_measurement_weight.exp()
        pattern_mass = (image_weight[..., None] * probability).sum(-2)
        component_masses = [pattern_mass[..., :1] * prior_weight]
        means, covariances = [p], [cp]
        identities = [prior_identity]
        pattern_visibility = [p.new_zeros(*p.shape[:-1], 2)]
        visible_logits = []
        for pattern, coordinates in enumerate(((0, 1), (2, 3), (0, 1, 2, 3)), start=1):
            mean, covariance, evidence, identity = _observed_update(
                p, cp, m, cm, prior.log_weights, prior_identity,
                measured_identity, coordinates, reacquire_mass)
            # Clamp only logs of zero-probability patterns. Their total mass is
            # still exactly zero; no hard confidence threshold is introduced.
            logits = log_measurement_weight + probability[..., pattern].clamp_min(1e-30).log() + evidence
            visible_logits.append(logits)
            means.append(mean)
            covariances.append(covariance)
            identities.append(identity)
            mask = p.new_tensor([float(0 in coordinates), float(2 in coordinates)])
            pattern_visibility.append(mask.expand(*p.shape[:-1], 2))
        # A geometric endpoint rename swaps the two one-visible patterns.
        # Normalize them jointly (K x 2), not separately across components;
        # otherwise a per-component gauge rename could change the posterior.
        one_logits = torch.stack(visible_logits[:2], -1)
        one_weights = one_logits.flatten(-2).softmax(-1).reshape_as(one_logits)
        # A numerical log floor is not observation probability. Restore exact
        # zero patterns before normalization, so a hidden endpoint contributes
        # neither invented visibility nor a spatial/innovation observation.
        one_weights = one_weights * (probability[..., 1:3] > 0)
        one_weights = one_weights / one_weights.sum((-1, -2), keepdim=True).clamp_min(1e-30)
        one_mass = pattern_mass[..., 1:3].sum(-1, keepdim=True)
        both_weights = visible_logits[2].softmax(-1) * (probability[..., 3] > 0)
        both_weights = both_weights / both_weights.sum(-1, keepdim=True).clamp_min(1e-30)
        component_masses.extend((one_mass * one_weights[..., 0], one_mass * one_weights[..., 1],
                                 pattern_mass[..., 3:4] * both_weights))
        masses = torch.stack(component_masses, -1)
        component_mass = masses.sum(-1)
        responsibility = masses / component_mass[..., None].clamp_min(1e-30)
        branches = torch.stack(means, -2)
        mean = (responsibility[..., None] * branches).sum(-2)
        offset = branches - mean[..., None, :]
        covariance = (responsibility[..., None, None] *
            (torch.stack(covariances, -3) + offset[..., :, None] * offset[..., None, :])).sum(-3)
        covariance = .5 * (covariance + covariance.transpose(-1, -2)) + torch.eye(4, device=p.device, dtype=p.dtype) * 1e-5
        current_visibility = (responsibility[..., None] * torch.stack(pattern_visibility, -2)).sum(-2)
        identity = (responsibility * torch.stack(identities, -1)).sum(-1)
        updated = pair_from_means(mean.reshape(*mean.shape[:-1], 2, 2),
            joint_covariance=covariance, log_weights=component_mass.clamp_min(1e-30).log(),
            visible_probability=current_visibility, identity_probability=identity)
        absent = (safe_measurement.visible_probability == 0).all((-1, -2))
        exact_no_spatial_evidence = replace(prior, visible_probability=torch.zeros_like(prior.visible_probability))
        updated = select_pair_belief(absent, exact_no_spatial_evidence, updated)
    return select_pair_belief(valid, updated, prior)


class V203PairObserver(V201PairObserver):
    """Same learned measurement architecture, one visibility-aware assimilation."""

    def assimilate(self, measurement, prior, view_valid):
        if prior is None:
            # H0 without geometry uses the same raw observation as cached
            # training initialization; do not add a runtime-only broad update.
            return self.repool(measurement, measurement.belief, view_valid)
        belief = fuse_v203_pair_beliefs(measurement.belief, prior, view_valid, prior_floor=self.prior_floor)
        # A moved distribution requires current-field re-pooling, not old
        # features carried along with the old mean.
        return self.repool(measurement, belief, view_valid)


__all__ = ["V203PairObserver", "fuse_v203_pair_beliefs", "ObserverOutput", "POOL_CHANNELS"]
