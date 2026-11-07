"""V20.1 mode-preserving image proposals and joint-pair observation.

The four proposals are alternative explanations of ONE jaw pair, not four
objects. Global image-local maxima seed differentiable local readouts; no
nominal crop and no artificial minimum physical jaw separation is used.
Continuous belief fusion requires re-integrating the CURRENT spatial field.
The eight-channel field is deliberately retained in cache (not just its old
pooled features); its reconstruction capacity and storage cost are explicit
engineering tradeoffs, not claims of a lossless feature compression.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.prod_v20_1_belief import (
    BELIEF_DTYPE, JointPairBelief, broad_pair_prior, fuse_pair_beliefs, pair_nll, _small_matrix_product,
    posterior_weighted_pair_pool,
)
from fcpg.prod_v20_observer import _grid


IMAGE_SIZE = 224
VIEW_ORDER = ("wrist", "column")
PAIR_COMPONENTS = 4
FEATURE_DIM = 384
POOL_CHANNELS = 8


@dataclass(frozen=True)
class ObserverOutput:
    belief: JointPairBelief
    component_features: torch.Tensor  # [B,2,4,384]
    scene_tokens: torch.Tensor  # [B,2,256,384], detached DINO
    coarse_center_log_posterior: torch.Tensor  # [B,2,4,3136], shared global response
    pooling_map: torch.Tensor  # [B,2,8,56,56], current-frame field, mandatory


@dataclass(frozen=True)
class RawObserverMeasurement:
    """Cache-only fields, not a ready-to-use Actor/runtime observation.

    A frozen raw cache stores the visual belief and spatial field. Its reader
    integrates that field against the current posterior later; computing the
    pre-fusion integral here would be discarded. Keeping a separate type
    avoids representing uncomputed component features as real appearance.
    """

    belief: JointPairBelief
    pooling_map: torch.Tensor


def _select_belief(mask, yes, no):
    return JointPairBelief(**{
        field.name: torch.where(
            mask.reshape(*mask.shape, *((1,) * (getattr(yes, field.name).ndim - mask.ndim))),
            getattr(yes, field.name), getattr(no, field.name),
        ) for field in fields(yes)
    })


def global_mode_proposals(logits: torch.Tensor, *, size: int = 56, count: int = 4,
                          suppression_radius: int = 4) -> torch.Tensor:
    """Discrete image proposal indices [...,K], never a posterior expectation.

    Local-max eligibility and proposal selection are intentionally discrete.
    Global center cross entropy trains all image positions; subsequent local
    soft readouts train the selected visual evidence. Suppression separates
    proposal seeds only, NOT output tips or geometric projection overlap.
    """
    if logits.shape[-1] != size * size or count < 1 or size < 2:
        raise ValueError("global proposal logits must have a square spatial grid")
    original = logits.shape[:-1]
    scores = logits.detach().float().reshape(-1, size * size)
    maxima = scores.reshape(-1, 1, size, size)
    maxima = (maxima == F.max_pool2d(maxima, 3, stride=1, padding=1)).flatten(1)
    available = torch.ones_like(maxima)
    y, x = torch.meshgrid(torch.arange(size, device=logits.device),
                          torch.arange(size, device=logits.device), indexing="ij")
    xy = torch.stack((x.flatten(), y.flatten()), -1)
    selected = []
    for _ in range(count):
        # Images can have fewer than K local maxima. A finite unselected image
        # location is then an explicit low-scoring alternative, not a repeat.
        eligible = maxima & available
        eligible = torch.where(eligible.any(-1, keepdim=True), eligible, available)
        index = scores.masked_fill(~eligible, -torch.inf).argmax(-1)
        selected.append(index)
        distance = (xy[None] - xy[index][:, None]).square().sum(-1)
        available = available & (distance > suppression_radius**2)
    return torch.stack(selected, -1).reshape(*original, count)


def _local_soft_readout(logits, centers, *, size, radius_px):
    """Finite-support soft readout around CURRENT image-evidence proposals."""
    xy = _grid(size, logits)
    squared = (xy - centers[..., None, :]).square().sum(-1)
    # Detached support membership has the same piecewise smooth nature as
    # proposal selection. Inside it, both center and all response logits have
    # gradients. At least the selected pixel is always in support.
    support = squared.detach() <= radius_px**2
    nearest = squared.detach().argmin(-1, keepdim=True)
    support = support.scatter(-1, nearest, True)
    score = logits - squared / (2 * (radius_px / 2)**2)
    probability = score.masked_fill(~support, -torch.inf).softmax(-1)
    # Image coordinates and covariance algebra remain fp32 even when visual
    # convolutions use bf16; otherwise means near x=224 round to whole pixels.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        mean = torch.einsum("...kn,nd->...kd", probability.float(), xy.float())
    return mean, probability


class V201PairObserver(nn.Module):
    """Shared RGB/DINO frontend, current-q soft conditioning, one update.

    Measurement consumes current actual-q geometry, never predicted-q history.
    The historical process prior enters only ``assimilate``. This remains a
    discriminatively trained conditional density with calibrated-fusion work
    still required; the implementation does not claim exact Bayesian evidence.
    """

    def __init__(self, channels: int = 64, prior_floor: float = 0.25):
        super().__init__()
        if channels < 8 or channels % 8 or not 0 < prior_floor < 1:
            raise ValueError("channels must be a multiple of 8; prior floor in (0,1)")
        self.prior_floor = float(prior_floor)
        self.rgb112 = nn.Sequential(
            nn.Conv2d(3, channels // 2, 5, 2, 2), nn.GroupNorm(4, channels // 2),
            nn.GELU(), nn.Conv2d(channels // 2, channels // 2, 3, 1, 1), nn.GELU())
        self.rgb56 = nn.Sequential(nn.Conv2d(channels // 2, channels, 3, 2, 1),
                                   nn.GroupNorm(8, channels), nn.GELU())
        self.dino_projection = nn.Conv2d(FEATURE_DIM, channels, 1)
        self.fusion = nn.Sequential(nn.Conv2d(channels * 2 + 2, channels, 3, 1, 1),
                                   nn.GroupNorm(8, channels), nn.GELU())
        self.center_head = nn.Conv2d(channels, 1, 1)
        self.fine_center_head = nn.Conv2d(channels // 2 + channels, 1, 1)
        self.mechanical_query = nn.Sequential(nn.Linear(10, channels), nn.GELU(),
                                             nn.Linear(channels, channels), nn.Tanh())
        # Half-vector (2), full Cholesky lower triangle (10), visibility (2).
        self.pose_head = nn.Sequential(nn.Linear(channels, channels), nn.GELU(),
                                       nn.Linear(channels, 14))
        self.component_weight_head = nn.Linear(channels, 1)
        self.pool_projection = nn.Conv2d(channels + channels // 2, POOL_CHANNELS, 1)
        self.feature_projection = nn.Linear(POOL_CHANNELS, FEATURE_DIM, bias=False)
        self.distillation_head = nn.Conv2d(channels, FEATURE_DIM, 1)

    def _visual(self, rgb, dino):
        rgb = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb
        if (rgb < 0).any() or (rgb > 1).any():
            raise ValueError("V20.1 RGB floats must be in [0,1]; uint8 is accepted")
        high = self.rgb112(rgb)
        low = self.rgb56(high)
        frozen = dino.detach().to(self.dino_projection.weight).transpose(1, 2).reshape(-1, FEATURE_DIM, 16, 16)
        frozen = F.interpolate(self.dino_projection(frozen), (56, 56), mode="bilinear", align_corners=False)
        xy = ((_grid(56, low) + 0.5) / IMAGE_SIZE * 2 - 1).T.reshape(1, 2, 56, 56)
        return self.fusion(torch.cat((low, frozen, xy.expand(low.shape[0], -1, -1, -1)), 1)), high

    def _pool_field(self, fused, high):
        return self.pool_projection(torch.cat((fused, F.avg_pool2d(high, 2)), 1))

    def appearance_loss(self, rgb, dino_tokens, view_valid):
        valid = view_valid.bool()
        rgb = torch.where(valid[..., None, None, None], rgb, torch.zeros_like(rgb))
        dino_tokens = torch.where(valid[..., None, None], dino_tokens, torch.zeros_like(dino_tokens))
        fused, high = self._visual(rgb.reshape(-1, 3, 224, 224), dino_tokens.reshape(-1, 256, 384))
        dense = F.adaptive_avg_pool2d(self.distillation_head(fused), (16, 16)).flatten(2).transpose(1, 2)
        compact = F.adaptive_avg_pool2d(self._pool_field(fused, high), (16, 16)).flatten(2).transpose(1, 2)
        compact = self.feature_projection(compact)
        target = F.layer_norm(dino_tokens.detach().reshape(-1, 256, 384).float(), (384,))
        # The compact field and its learned readout are trained before freeze;
        # an unsupervised random 8->384 projection would be an interface bug.
        loss = sum((F.layer_norm(p.float(), (384,)) - target).square().mean((-1, -2))
                   for p in (dense, compact)).reshape_as(valid) * 0.5
        return (loss * valid).sum() / valid.sum().clamp_min(1)

    appearance_distillation = appearance_loss

    def _mechanical_context(self, prior, reference, batch):
        if prior is None:
            return reference.new_zeros(batch, 2, reference.shape[1])
        prior = prior.to(device=reference.device)
        center, half = prior.state_mean[..., :2], prior.state_mean[..., 2:]
        features = torch.cat((
            ((center - 111.5) / 112).tanh(), (half / 112).tanh(),
            (prior.state_covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt() / 224).tanh(),
            prior.visible_probability * 2 - 1,
        ), -1)
        encoded = self.mechanical_query(features.to(reference.dtype))
        # Mixture-order invariant physical query; it never specifies a crop.
        return (encoded * prior.weights.to(encoded.dtype)[..., None]).sum(-2)

    def _measure_raw(self, rgb, dino_tokens, view_valid, mechanical_prior=None):
        batch = rgb.shape[0]
        if rgb.shape != (batch, 2, 3, 224, 224) or dino_tokens.shape != (batch, 2, 256, 384) or view_valid.shape != (batch, 2):
            raise ValueError("V20.1 observer requires RGB[B,2,3,224,224], DINO[B,2,256,384], valid[B,2]")
        valid = view_valid.bool()
        clean_rgb = torch.where(valid[..., None, None, None], rgb, torch.zeros_like(rgb))
        clean_dino = torch.where(valid[..., None, None], dino_tokens, torch.zeros_like(dino_tokens))
        if not torch.isfinite(clean_rgb).all() or not torch.isfinite(clean_dino).all():
            raise ValueError("valid observations must be finite")
        fused, high = self._visual(clean_rgb.reshape(-1, 3, 224, 224), clean_dino.reshape(-1, 256, 384))
        spatial = fused.float().reshape(batch, 2, fused.shape[1], 56 * 56)
        context = self._mechanical_context(mechanical_prior, fused, batch)
        correlation = torch.einsum("bvc,bvcn->bvn", context.float(), spatial) / math.sqrt(fused.shape[1])
        global_logits = self.center_head(fused).reshape(batch, 2, -1).float() + 2 * correlation.tanh()
        indices = global_mode_proposals(global_logits)
        seeds = _grid(56, global_logits)[indices]
        local_logits = global_logits[..., None, :].expand(-1, -1, 4, -1)
        coarse_center, local_probability = _local_soft_readout(local_logits, seeds, size=56, radius_px=12)
        fine_input = torch.cat((high, F.interpolate(fused, (112, 112), mode="bilinear", align_corners=False)), 1)
        fine_logits = self.fine_center_head(fine_input).reshape(batch, 2, 1, -1).float().expand(-1, -1, 4, -1)
        center, _ = _local_soft_readout(fine_logits, coarse_center, size=112, radius_px=12)
        center_features = torch.einsum("bvkn,bvcn->bvkc", local_probability, spatial)
        pose = self.pose_head((center_features + .1 * context.float()[..., None, :]).to(fused.dtype)).float()
        state_mean = torch.cat((center.to(BELIEF_DTYPE), pose[..., :2].to(BELIEF_DTYPE) * 16), -1)
        tril = torch.tril_indices(4, 4, device=pose.device)
        cholesky = torch.zeros(batch, 2, 4, 4, 4, device=pose.device, dtype=BELIEF_DTYPE)
        raw = pose[..., 2:12].to(BELIEF_DTYPE)
        values = torch.where(tril[0] == tril[1], F.softplus(raw) * 4 + .5, raw * 2)
        cholesky[..., tril[0], tril[1]] = values
        with torch.autocast(device_type=pose.device.type, enabled=False):
            covariance = _small_matrix_product(cholesky, cholesky.transpose(-1, -2)) + torch.eye(4, device=pose.device, dtype=BELIEF_DTYPE) * 1e-4
        # Global response confidence contributes a differentiable weight even
        # though peak indices are discrete. It does not replace center CE.
        peak_score = global_logits.gather(-1, indices)
        weights = (self.component_weight_head(center_features.to(fused.dtype)).squeeze(-1).float() + peak_score).log_softmax(-1)
        raw_belief = JointPairBelief(weights, state_mean, covariance, pose[..., 12:14].sigmoid())
        belief = _select_belief(valid, raw_belief, broad_pair_prior(batch, device=rgb.device))
        field = self._pool_field(fused, high).reshape(batch, 2, POOL_CHANNELS, 56, 56)
        field = torch.where(valid[..., None, None, None], field, torch.zeros_like(field))
        coarse = global_logits.log_softmax(-1)[..., None, :].expand(-1, -1, 4, -1)
        coarse = torch.where(valid[..., None, None], coarse, torch.full_like(coarse, -math.log(56 * 56)))
        output = ObserverOutput(belief, field.new_zeros(batch, 2, 4, FEATURE_DIM), clean_dino.detach(), coarse, field)
        return output

    def measure(self, rgb, dino_tokens, view_valid, mechanical_prior=None):
        output = self._measure_raw(rgb, dino_tokens, view_valid, mechanical_prior)
        return self.repool(output, output.belief, view_valid)

    measurement = measure

    def measurement_for_cache(self, rgb, dino_tokens, view_valid, mechanical_prior=None):
        """Same raw measurement, without the unused pre-fusion feature integral.

        Only raw-cache builders should use this API. ``measure``/``measurement``
        and ``forward`` retain their full posterior-pooled feature behavior.
        No precision, parameters, input preprocessing or cache fields change.
        """
        output = self._measure_raw(rgb, dino_tokens, view_valid, mechanical_prior)
        return RawObserverMeasurement(output.belief, output.pooling_map)

    def repool(self, measurement, belief, view_valid):
        if measurement.pooling_map.shape != (*view_valid.shape, POOL_CHANNELS, 56, 56):
            raise ValueError("V20.1 requires the current compact spatial field; old pooled-only caches are incompatible")
        pooled = posterior_weighted_pair_pool(belief, measurement.pooling_map)
        features = self.feature_projection(pooled.to(self.feature_projection.weight)).float()
        valid = view_valid.bool()
        return replace(measurement, belief=belief,
                       component_features=torch.where(valid[..., None, None], features, torch.zeros_like(features)),
                       scene_tokens=torch.where(valid[..., None, None], measurement.scene_tokens, torch.zeros_like(measurement.scene_tokens)))

    def assimilate(self, measurement, prior, view_valid):
        if prior is None:
            return self.repool(measurement, measurement.belief, view_valid)
        belief = fuse_pair_beliefs(measurement.belief, prior, view_valid,
                                  prior_floor=self.prior_floor)
        return self.repool(measurement, belief, view_valid)

    def forward(self, rgb, dino_tokens, view_valid, prior=None):
        return self.assimilate(self.measure(rgb, dino_tokens, view_valid, mechanical_prior=prior), prior, view_valid)

    @staticmethod
    def pair_nll(belief, target_fm_px, valid, ordered=True, **kwargs):
        return pair_nll(belief, target_fm_px, valid, ordered, **kwargs)

    @staticmethod
    def center_loss(observation, target_fm_px, valid, *, view_counts=None):
        """View-balanced bilinear center CE, only when BOTH tips are labeled.

        This explicit full-image loss trains proposals missed by top-K. It is
        not claimed to be exact continuous likelihood or extra point labels.
        With ``view_counts`` the denominator belongs to the whole effective
        batch, while this call contributes only the current microbatch sum.
        """
        point_valid = valid.bool()
        if point_valid.ndim == 2:
            point_valid = point_valid[..., None].expand(-1, -1, 2)
        known = point_valid.all(-1) & torch.isfinite(target_fm_px).all((-1, -2))
        target = torch.nan_to_num(target_fm_px.float()).mean(-2)
        known = known & (target >= -.5).all(-1) & (target <= IMAGE_SIZE - .5).all(-1)
        coordinate = ((target + .5) / 4 - .5).clamp(0, 55)
        lower = coordinate.floor().long()
        upper = (lower + 1).clamp_max(55)
        fraction = coordinate - lower
        logp = observation.coarse_center_log_posterior.float().mean(-2)
        loss = torch.zeros_like(target[..., 0])
        for use_x in (0, 1):
            for use_y in (0, 1):
                x = upper[..., 0] if use_x else lower[..., 0]
                y = upper[..., 1] if use_y else lower[..., 1]
                weight = (fraction[..., 0] if use_x else 1 - fraction[..., 0]) * (fraction[..., 1] if use_y else 1 - fraction[..., 1])
                loss = loss - weight * logp.gather(-1, (y * 56 + x)[..., None]).squeeze(-1)
        counts = known.sum(0) if view_counts is None else torch.as_tensor(
            view_counts, device=loss.device, dtype=loss.dtype)
        if counts.shape != (2,):
            raise ValueError("center view_counts must contain the two effective-batch view counts")
        means = (loss * known).sum(0) / counts.clamp_min(1)
        return (means * (counts > 0)).sum() / (counts > 0).sum().clamp_min(1)

    def spatial_loss(self, observation, target_fm_px, valid, ordered=True, *,
                     visibility=None, visibility_valid=None, normalization=None):
        """Joint NLL plus center CE with optional exact microbatch reduction.

        ``normalization`` contains CPU-prevalidated counts for the *whole*
        effective batch: ``pair_view_counts``, ``center_view_counts``, and
        ``effective_rows``. Pair labels and observed centers have distinct
        validity, so a single row/valid-view weight cannot normalize both.
        Returned loss compensates the trainer's row-count microbatch weight.
        Omitting this metadata preserves the original standalone API exactly.
        """
        loss = pair_nll(observation.belief, target_fm_px, valid, ordered,
                        visibility=visibility, visibility_valid=visibility_valid,
                        identity_aware=True,
                        reduction="mean" if normalization is None else "none")
        # An explicitly invisible annotated tip has no supervised center.
        center_valid = valid.bool()
        if visibility is not None:
            if center_valid.ndim == 2:
                center_valid = center_valid[..., None].expand(-1, -1, 2)
            center_valid = center_valid & visibility.bool()
        if normalization is None:
            return loss + .1 * self.center_loss(observation, target_fm_px, center_valid)
        required = {"pair_view_counts", "center_view_counts", "effective_rows"}
        if not required.issubset(normalization):
            raise ValueError("spatial normalization requires pair/center view counts and effective_rows")
        effective_rows = int(normalization["effective_rows"])
        micro_rows = len(target_fm_px)
        if micro_rows < 1 or effective_rows < micro_rows:
            raise ValueError("effective_rows cannot be smaller than the current microbatch")
        counts = torch.as_tensor(normalization["pair_view_counts"], device=loss.device, dtype=loss.dtype)
        if counts.shape != (2,):
            raise ValueError("pair_view_counts must contain the two effective-batch view counts")
        means = loss.sum(0) / counts.clamp_min(1)
        pair_loss = (means * (counts > 0)).sum() / (counts > 0).sum().clamp_min(1)
        center = self.center_loss(observation, target_fm_px, center_valid,
                                  view_counts=normalization["center_view_counts"])
        return (pair_loss + .1 * center) * (effective_rows / micro_rows)


V20_1PairObserver = V201PairObserver
