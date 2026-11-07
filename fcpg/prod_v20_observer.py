"""V20 structured, mechanically informed *pair* observer.

The public unit is ``lerobot_calibrated_position``.  Physical conversion is
owned by ``DifferentiableRobotKinematics``, never by the image network.  A
component describes one complete jaw pair; independent fingertip queries are
deliberately absent.  Four components preserve alternative pair explanations.

The DINO backbone is external and frozen.  This module consumes its detached
features alongside RGB, so cache and runtime use precisely the same observer.
No camera/dataset ID can substitute for an actual camera transform.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.robot_kinematic_graph import (
    DifferentiableRobotKinematics,
    project_reference_points,
)


IMAGE_SIZE = 224
VIEW_ORDER = ("wrist", "column")
PAIR_COMPONENTS = 4
FEATURE_DIM = 384


def _grid(size: int, reference: torch.Tensor) -> torch.Tensor:
    # Pixel-centre coordinates agree with align_corners=False resampling.
    axis = (torch.arange(size, device=reference.device, dtype=reference.dtype) + 0.5) * (IMAGE_SIZE / size) - 0.5
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x, y), -1).reshape(-1, 2)


@dataclass(frozen=True)
class PairBelief:
    """A four-component mixture over ordered *pairs*, not two marginals.

    Covariance is a shared 2-D endpoint uncertainty within a component; the
    shared mixture index also retains pair identity and multimodal dependence.
    Two equal marginal means are never interpreted as a physical jaw pair.
    ``means_px`` is only a derived readout for loss, geometry and diagnostics.
    """

    log_weights: torch.Tensor  # [B,2,4]
    center_px: torch.Tensor  # [B,2,4,2]
    direction: torch.Tensor  # [B,2,4,2], unit vector F -> M
    opening_px: torch.Tensor  # [B,2,4], zero projected span is legal
    covariance_px2: torch.Tensor  # [B,2,4,2,2], symmetric positive definite
    visible_probability: torch.Tensor  # [B,2,4,2], F/M visibility

    @property
    def means_px(self) -> torch.Tensor:
        offset = self.direction * self.opening_px.unsqueeze(-1) * 0.5
        return torch.stack((self.center_px - offset, self.center_px + offset), -2)

    @property
    def weights(self) -> torch.Tensor:
        return self.log_weights.exp()

    def detach(self) -> "PairBelief":
        return PairBelief(**{item.name: getattr(self, item.name).detach() for item in fields(self)})

    def to(self, *args, **kwargs) -> "PairBelief":
        return PairBelief(**{item.name: getattr(self, item.name).to(*args, **kwargs) for item in fields(self)})

    def validate(self) -> None:
        batch = self.log_weights.shape[0]
        expected = {
            "log_weights": (batch, 2, 4), "center_px": (batch, 2, 4, 2),
            "direction": (batch, 2, 4, 2), "opening_px": (batch, 2, 4),
            "covariance_px2": (batch, 2, 4, 2, 2),
            "visible_probability": (batch, 2, 4, 2),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid pair belief {name}, expected {shape}")
        if not torch.allclose(self.log_weights.float().logsumexp(-1), torch.zeros_like(self.log_weights[..., 0]).float(), atol=1e-4):
            raise ValueError("pair mixture weights must normalize")
        if not torch.allclose(self.direction.float().norm(dim=-1), torch.ones_like(self.opening_px).float(), atol=1e-4):
            raise ValueError("pair directions must have unit norm")
        if (self.opening_px < 0).any() or (self.visible_probability < 0).any() or (self.visible_probability > 1).any():
            raise ValueError("invalid opening or visibility")
        covariance = self.covariance_px2.float()
        if not torch.allclose(covariance, covariance.transpose(-1, -2), atol=1e-5) or (torch.linalg.eigvalsh(covariance) <= 0).any():
            raise ValueError("pair covariance must be symmetric positive definite")


def pair_from_means(
    means_px: torch.Tensor,
    covariance_px2: torch.Tensor,
    visible_probability: torch.Tensor,
    log_weights: torch.Tensor | None = None,
) -> PairBelief:
    """Geometry/data adapter only; the learned head predicts a shared pose."""
    delta = means_px[..., 1, :] - means_px[..., 0, :]
    opening = delta.norm(dim=-1)
    fallback = torch.zeros_like(delta)
    fallback[..., 0] = 1.0
    direction = torch.where(opening[..., None] > 1e-6, delta / opening[..., None].clamp_min(1e-6), fallback)
    if log_weights is None:
        log_weights = torch.full_like(opening, -math.log(PAIR_COMPONENTS))
    return PairBelief(log_weights.log_softmax(-1), means_px.mean(-2), direction, opening, covariance_px2, visible_probability)


def broad_pair_prior(batch: int, device=None, dtype=torch.float32) -> PairBelief:
    """Uninformative placeholder for genuinely unavailable geometry."""
    center = torch.full((batch, 2, 4, 2), 111.5, device=device, dtype=dtype)
    direction = torch.zeros_like(center)
    direction[..., 0] = 1.0
    return PairBelief(
        torch.full((batch, 2, 4), -math.log(4), device=device, dtype=dtype), center,
        direction, torch.zeros((batch, 2, 4), device=device, dtype=dtype),
        torch.eye(2, device=device, dtype=dtype).expand(batch, 2, 4, 2, 2).clone() * 224**2,
        torch.full((batch, 2, 4, 2), 0.5, device=device, dtype=dtype),
    )


def _select_belief(mask: torch.Tensor, yes: PairBelief, no: PairBelief) -> PairBelief:
    values = {}
    for item in fields(yes):
        a, b = getattr(yes, item.name), getattr(no, item.name)
        values[item.name] = torch.where(mask.reshape(*mask.shape, *((1,) * (a.ndim - mask.ndim))), a, b)
    return PairBelief(**values)


def pair_component_maps(belief: PairBelief, size: int = 56) -> torch.Tensor:
    """Normalized spatial distributions [B,V,K,2,HW], no point sampling.

    Maps are conditional on a tip being visible; visibility is carried
    separately.  Off-screen Gaussian tails do not pretend to establish
    visibility.  All covariance algebra runs in fp32 under mixed precision.
    """
    means = belief.means_px.float()
    xy = _grid(size, means)
    delta = xy - means.unsqueeze(-2)
    inverse = torch.linalg.inv(belief.covariance_px2.float())
    distance = torch.einsum("bvkani,bvkij,bvkanj->bvkan", delta, inverse, delta)
    return (-0.5 * distance).softmax(-1)


def posterior_weighted_pair_pool(belief: PairBelief, feature_map: torch.Tensor) -> torch.Tensor:
    """Exact conditional pair-feature integral [B,V,K,C].

    Multiply returned component features by ``belief.weights`` only when an
    unconditional mixture summary is wanted.  Keeping K separate is preferred
    by the Actor and avoids erasing alternative whole-pair explanations.
    """
    if feature_map.ndim != 5 or feature_map.shape[:2] != belief.log_weights.shape[:2] or feature_map.shape[-2] != feature_map.shape[-1]:
        raise ValueError("pair pooling requires square [B,2,C,S,S] feature maps")
    # Missing tips contribute the explicit zero/null visual value.  Do not
    # normalize absent mass away and pretend an invisible tip was observed.
    maps = (pair_component_maps(belief, feature_map.shape[-1]) * belief.visible_probability.float()[..., None]).mean(-2)
    return torch.einsum("bvkn,bvcn->bvkc", maps, feature_map.float().flatten(-2))


def pair_nll(
    belief: PairBelief,
    target_fm_px: torch.Tensor,
    valid: torch.Tensor,
    ordered: torch.Tensor | bool = True,
    *,
    visibility: torch.Tensor | None = None,
    visibility_valid: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Ordered pair NLL or hard two-permutation set NLL, view balanced.

    ``valid`` means a supervised view, [B,2], or supervised point [B,2,2].
    Optional ``visibility`` [B,2,2] supplies explicit absent/visible labels.
    ``visibility_valid`` distinguishes known visibility from geometric
    in-frame validity: CAD projection validity alone cannot prove a tip is
    not occluded.  Unknown visibility adds no Bernoulli supervision.
    Missing labels are not silently converted to invisible labels.  Invalid
    target coordinates may be NaN; they are removed before Gaussian algebra.
    """
    target = target_fm_px.float()
    if target.shape != (*belief.log_weights.shape[:2], 2, 2):
        raise ValueError("pair labels must be [B,2,2,2]")
    valid = torch.as_tensor(valid, device=target.device, dtype=torch.bool)
    point_valid = valid[..., None].expand(*valid.shape, 2) if valid.ndim == 2 else valid
    if point_valid.shape != target.shape[:-1]:
        raise ValueError("valid must be [B,2] or [B,2,2]")
    finite = torch.isfinite(target).all(-1)
    if visibility is None:
        visible = torch.ones_like(point_valid)
    else:
        visible = torch.as_tensor(visibility, device=target.device, dtype=torch.bool)
        if visible.shape != point_valid.shape:
            raise ValueError("visibility must be [B,2,2]")
    if (point_valid & visible & ~finite).any():
        raise ValueError("visible supervised tips require finite labels")
    known_visibility = torch.ones_like(point_valid) if visibility_valid is None else torch.as_tensor(visibility_valid, device=target.device, dtype=torch.bool)
    if known_visibility.shape != point_valid.shape:
        raise ValueError("visibility_valid must be [B,2,2]")
    target = torch.where(finite[..., None], target, torch.zeros_like(target))
    inverse = torch.linalg.inv(belief.covariance_px2.float())
    logdet = torch.linalg.slogdet(belief.covariance_px2.float()).logabsdet
    probability = belief.visible_probability.float().clamp(1e-6, 1 - 1e-6)

    def assigned(order: tuple[int, int]) -> torch.Tensor:
        delta = target[..., list(order), :].unsqueeze(2) - belief.means_px.float()
        squared = torch.einsum("bvkai,bvkij,bvkaj->bvka", delta, inverse, delta)
        log_position = -0.5 * (squared + logdet[..., None] + 2 * math.log(2 * math.pi))
        assigned_visible = visible[..., list(order)].unsqueeze(2)
        assigned_valid = point_valid[..., list(order)].unsqueeze(2)
        assigned_visibility_known = known_visibility[..., list(order)].unsqueeze(2)
        log_visibility = torch.where(assigned_visible, probability.log(), torch.log1p(-probability))
        log_score = torch.where(assigned_visible, log_position, torch.zeros_like(log_position)) + torch.where(assigned_visibility_known, log_visibility, torch.zeros_like(log_visibility))
        per_component = torch.where(assigned_valid, log_score, torch.zeros_like(log_score)).sum(-1)
        return -(belief.log_weights.float() + per_component).logsumexp(-1)

    direct, swapped = assigned((0, 1)), assigned((1, 0))
    ordered_tensor = torch.as_tensor(ordered, dtype=torch.bool, device=target.device)
    if ordered_tensor.ndim == 1:
        ordered_tensor = ordered_tensor[:, None]
    loss = torch.where(ordered_tensor, direct, torch.minimum(direct, swapped))
    view_valid = point_valid.any(-1)
    loss = torch.where(view_valid, loss, torch.zeros_like(loss))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError("reduction must be mean, sum or none")
    # Equal views, natural sampling within each view; no span-stratified boost.
    counts = view_valid.sum(0)
    per_view = loss.sum(0) / counts.clamp_min(1)
    return (per_view * (counts > 0)).sum() / (counts > 0).sum().clamp_min(1)


@dataclass(frozen=True)
class ObserverOutput:
    belief: PairBelief
    component_features: torch.Tensor  # [B,2,4,384]
    scene_tokens: torch.Tensor  # [B,2,256,384], frozen DINO
    coarse_center_log_posterior: torch.Tensor  # [B,2,4,3136]


class V20PairObserver(nn.Module):
    """One 56 -> 112 full-image observation, one soft-prior assimilation.

    The same visual weights process both cameras independently.  Current
    actual-q mechanical information conditions full-image paired queries.
    Predicted/history priors are incorporated later, exactly once, by
    ``assimilate``; they must not condition the current measurement queries.
    Assimilation changes mixture responsibilities and physical orientation;
    it cannot fabricate visual features at a new coordinate.  This is a
    finite-mixture importance update, not an exact continuous Bayesian filter.
    Raw component feature integrals can therefore be cached without changing
    the inference path.  A uniform prior floor leaves all image hypotheses
    possible.  RGB is uint8 [0,255] or floating point [0,1].
    """

    def __init__(self, channels: int = 64, prior_floor: float = 0.25):
        super().__init__()
        if channels < 8 or channels % 8 or not 0 < prior_floor < 1:
            raise ValueError("channels must be a multiple of 8; prior floor in (0,1)")
        self.prior_floor = float(prior_floor)
        self.rgb112 = nn.Sequential(nn.Conv2d(3, channels // 2, 5, 2, 2), nn.GroupNorm(4, channels // 2), nn.GELU(), nn.Conv2d(channels // 2, channels // 2, 3, 1, 1), nn.GELU())
        self.rgb56 = nn.Sequential(nn.Conv2d(channels // 2, channels, 3, 2, 1), nn.GroupNorm(8, channels), nn.GELU())
        self.dino_projection = nn.Conv2d(FEATURE_DIM, channels, 1)
        self.fusion = nn.Sequential(nn.Conv2d(channels * 2 + 2, channels, 3, 1, 1), nn.GroupNorm(8, channels), nn.GELU())
        self.center_head = nn.Conv2d(channels, 4, 1)
        self.fine_center_head = nn.Conv2d(channels // 2 + channels, 4, 1)
        self.mechanical_query = nn.Sequential(nn.Linear(10, channels), nn.GELU(), nn.Linear(channels, channels), nn.Tanh())
        # Every head row describes a COMPLETE pair: pose, covariance, visibility.
        self.pose_head = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, 8))
        self.component_weight_head = nn.Linear(channels, 1)
        self.feature_projection = nn.Linear(channels + channels // 2, FEATURE_DIM)
        self.distillation_head = nn.Conv2d(channels, FEATURE_DIM, 1)

    def _visual(self, rgb: torch.Tensor, dino: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb
        if (rgb < 0).any() or (rgb > 1).any():
            raise ValueError("V20 RGB floats must be in [0,1]; uint8 is accepted")
        n = rgb.shape[0]
        high = self.rgb112(rgb)
        low = self.rgb56(high)
        frozen_tokens = dino.detach().to(self.dino_projection.weight).transpose(1, 2).reshape(n, FEATURE_DIM, 16, 16)
        frozen = F.interpolate(self.dino_projection(frozen_tokens), size=(56, 56), mode="bilinear", align_corners=False)
        xy = (_grid(56, low) + 0.5) / IMAGE_SIZE * 2 - 1
        xy = xy.T.reshape(1, 2, 56, 56).expand(n, -1, -1, -1)
        return self.fusion(torch.cat((low, frozen, xy), 1)), high

    def appearance_loss(self, rgb: torch.Tensor, dino_tokens: torch.Tensor, view_valid: torch.Tensor) -> torch.Tensor:
        """Dense appearance initialization only; never fingertip supervision."""
        batch = rgb.shape[0]
        valid = view_valid.bool()
        rgb = torch.where(valid[..., None, None, None], rgb, torch.zeros_like(rgb))
        dino_tokens = torch.where(valid[..., None, None], dino_tokens, torch.zeros_like(dino_tokens))
        fused, _ = self._visual(rgb.reshape(-1, 3, 224, 224), dino_tokens.reshape(-1, 256, 384))
        prediction = F.adaptive_avg_pool2d(self.distillation_head(fused), (16, 16)).flatten(2).transpose(1, 2)
        target = dino_tokens.detach().reshape(-1, 256, 384).float()
        loss = (F.layer_norm(prediction.float(), (384,)) - F.layer_norm(target, (384,))).square().mean((-1, -2)).reshape(batch, 2)
        weight = view_valid.to(loss)
        return (loss * weight).sum() / weight.sum().clamp_min(1)

    appearance_distillation = appearance_loss

    def _mechanical_context(self, mechanical_prior: PairBelief | None, reference: torch.Tensor, batch: int) -> torch.Tensor:
        channels = reference.shape[1]
        if mechanical_prior is None:
            return reference.new_zeros(batch, 2, 4, channels)
        belief = mechanical_prior.to(device=reference.device, dtype=torch.float32)
        # Bounded physical pose summary, not a pixel crop, attention mask or
        # another log-prior product. Invalid geometry is supplied explicitly
        # as None/broad_pair_prior by the geometry owner, never a dataset ID.
        features = torch.cat((
            ((belief.center_px - 111.5) / 112).tanh(), belief.direction,
            (belief.opening_px[..., None] / 224).tanh(),
            (belief.covariance_px2.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt() / 224).tanh(),
            belief.visible_probability * 2 - 1, belief.weights[..., None],
        ), -1)
        return self.mechanical_query(features.to(reference.dtype))

    @staticmethod
    def _mechanical_correlation(query: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        batch = query.shape[0]
        spatial = fused.float().reshape(batch, 2, fused.shape[1], 56 * 56)
        response = torch.einsum("bvkc,bvcn->bvkn", query.float(), spatial) / math.sqrt(fused.shape[1])
        # A wrong mechanical prior cannot suppress any global image position.
        # This bounded feature-conditioning term is NOT a Bayesian update.
        return 2.0 * torch.tanh(response)

    def measure(self, rgb: torch.Tensor, dino_tokens: torch.Tensor, view_valid: torch.Tensor, mechanical_prior: PairBelief | None = None) -> ObserverOutput:
        batch = rgb.shape[0]
        if rgb.shape != (batch, 2, 3, 224, 224) or dino_tokens.shape != (batch, 2, 256, 384) or view_valid.shape != (batch, 2):
            raise ValueError("V20 observer requires RGB[B,2,3,224,224], DINO[B,2,256,384], valid[B,2]")
        valid = view_valid.bool()
        # Missing-view bytes may be NaN and must not enter convolutions.
        clean_rgb = torch.where(valid[..., None, None, None], rgb, torch.zeros_like(rgb))
        clean_dino = torch.where(valid[..., None, None], dino_tokens, torch.zeros_like(dino_tokens))
        if not torch.isfinite(clean_rgb).all() or not torch.isfinite(clean_dino).all():
            raise ValueError("valid observations must be finite")
        prior = broad_pair_prior(batch, rgb.device)
        fused, high = self._visual(clean_rgb.reshape(-1, 3, 224, 224), clean_dino.reshape(-1, 256, 384))
        logits = self.center_head(fused).reshape(batch, 2, 4, -1).float()
        mechanical_context = self._mechanical_context(mechanical_prior, fused, batch)
        logits = logits + self._mechanical_correlation(mechanical_context, fused)
        coarse = logits.log_softmax(-1)
        coarse_mean = torch.einsum("bvkn,nd->bvkd", coarse.exp(), _grid(56, logits))
        fine_input = torch.cat((high, F.interpolate(fused, (112, 112), mode="bilinear", align_corners=False)), 1)
        fine_logits = self.fine_center_head(fine_input).reshape(batch, 2, 4, -1).float()
        # Local fine measurement is centred by the CURRENT global observation,
        # never a nominal search window.  No prior is assimilated a second time.
        fine_delta = _grid(112, logits) - coarse_mean.unsqueeze(-2)
        fine_logits = fine_logits - fine_delta.square().sum(-1) / (2 * 12.0**2)
        fine_probability = fine_logits.softmax(-1)
        center = torch.einsum("bvkn,nd->bvkd", fine_probability, _grid(112, logits))
        center_features = torch.einsum("bvkn,bvcn->bvkc", coarse.exp(), fused.float().reshape(batch, 2, -1, 56 * 56))
        # One shared center/orientation/opening state receives the physical
        # query. Neither tip has an independent nominal-relative residual.
        pose = self.pose_head((center_features + 0.1 * mechanical_context.float()).to(fused.dtype)).float()
        angle = math.pi * torch.tanh(pose[..., 0])
        direction = torch.stack((angle.cos(), angle.sin()), -1)
        opening = F.softplus(pose[..., 1]) * 16.0  # no minimum physical separation
        scale = F.softplus(pose[..., 2:4]) * 4 + 0.5
        rho = 0.95 * torch.tanh(pose[..., 4])
        cross = rho * scale[..., 0] * scale[..., 1]
        covariance = torch.stack((scale[..., 0].square(), cross, cross, scale[..., 1].square()), -1).reshape(batch, 2, 4, 2, 2)
        visibility = pose[..., 5:7].sigmoid()
        weights = (self.component_weight_head(center_features.to(fused.dtype)).squeeze(-1).float() + pose[..., 7]).log_softmax(-1)
        observed = PairBelief(weights, center, direction, opening, covariance, visibility)
        posterior = _select_belief(valid, observed, prior)
        pooled56 = posterior_weighted_pair_pool(posterior, fused.reshape(batch, 2, -1, 56, 56))
        pooled112 = posterior_weighted_pair_pool(posterior, high.reshape(batch, 2, -1, 112, 112))
        features = self.feature_projection(torch.cat((pooled56, pooled112), -1).to(fused.dtype)).float()
        features = torch.where(valid[..., None, None], features, torch.zeros_like(features))
        coarse = torch.where(valid[..., None, None], coarse, torch.full_like(coarse, -math.log(56 * 56)))
        return ObserverOutput(posterior, features, clean_dino.detach(), coarse)

    measurement = measure

    def assimilate(self, measurement: ObserverOutput, prior: PairBelief | None, view_valid: torch.Tensor) -> ObserverOutput:
        """Exactly one current-frame update, using complete-pair overlap.

        Both components and fingertips retain their correspondence.  The
        geometry prior may choose F/M orientation but cannot move either raw
        visual component.  Features remain exact integrals under the cached
        conditional distributions, with updated posterior mixture weights.
        The uniform branch prevents a confident wrong prior from deleting a
        strong newly observed hypothesis.
        """
        observed = measurement.belief
        valid = view_valid.bool()
        if prior is None:
            return measurement
        prior = prior.to(device=observed.center_px.device, dtype=torch.float32)
        covariance = observed.covariance_px2[:, :, :, None].float() + prior.covariance_px2[:, :, None].float()
        inverse = torch.linalg.inv(covariance)
        logdet = torch.linalg.slogdet(covariance).logabsdet

        def overlap(swapped: bool) -> torch.Tensor:
            means = observed.means_px.float()
            if swapped:
                means = means.flip(-2)
            delta = means[:, :, :, None] - prior.means_px[:, :, None].float()
            distance = torch.einsum("bvkmai,bvkmij,bvkmaj->bvkma", delta, inverse, delta).sum(-1)
            log_density = -0.5 * (distance + 2 * logdet + 4 * math.log(2 * math.pi))
            return (log_density + prior.log_weights[:, :, None]).logsumexp(-1)

        same, swapped = overlap(False), overlap(True)
        swap_identity = swapped > same
        score = torch.maximum(same, swapped)
        uniform = torch.full_like(score, math.log(self.prior_floor) - 4 * math.log(IMAGE_SIZE))
        evidence = torch.logaddexp(score + math.log(1 - self.prior_floor), uniform)
        weights = (observed.log_weights.float() + evidence).log_softmax(-1)
        direction = torch.where(swap_identity[..., None], -observed.direction, observed.direction)
        visibility = torch.where(swap_identity[..., None], observed.visible_probability.flip(-1), observed.visible_probability)
        assimilated = PairBelief(weights, observed.center_px, direction, observed.opening_px, observed.covariance_px2, visibility)
        posterior = _select_belief(valid, assimilated, prior)
        features = torch.where(valid[..., None, None], measurement.component_features, torch.zeros_like(measurement.component_features))
        scenes = torch.where(valid[..., None, None], measurement.scene_tokens, torch.zeros_like(measurement.scene_tokens))
        return ObserverOutput(posterior, features, scenes, measurement.coarse_center_log_posterior)

    def forward(self, rgb: torch.Tensor, dino_tokens: torch.Tensor, view_valid: torch.Tensor, prior: PairBelief | None = None) -> ObserverOutput:
        # H0 convenience: current mechanical prior conditions queries and is
        # assimilated once. H1+ callers use measurement(current_q_geometry),
        # then assimilate(predicted_prior); predicted q is never substituted
        # for the current image's actual q in measurement.
        return self.assimilate(self.measure(rgb, dino_tokens, view_valid, mechanical_prior=prior), prior, view_valid)

    @staticmethod
    def pair_nll(belief: PairBelief, target_fm_px: torch.Tensor, valid: torch.Tensor, ordered: torch.Tensor | bool = True, **kwargs) -> torch.Tensor:
        return pair_nll(belief, target_fm_px, valid, ordered, **kwargs)


class V20MechanicalProjection(nn.Module):
    """Shared FK camera readout for observation and LeWM predicted readback.

    Camera transforms are relative to each RobotSpec view reference: tool
    frame for wrist, base frame for column.  Position-to-radian conversion is
    exclusively inside the RobotSpec adapter.  Gripper mapping is approximate
    and therefore never imposes a minimum visual span or hard pixel gate.
    """

    def __init__(self, kinematics: DifferentiableRobotKinematics, camera_from_reference: torch.Tensor, intrinsics: torch.Tensor, image_transform: torch.Tensor | None = None, camera_valid: torch.Tensor | None = None):
        super().__init__()
        if tuple(kinematics.spec.view_names) != VIEW_ORDER:
            raise ValueError("mechanical projection view order must be wrist,column")
        self.kinematics = kinematics
        self.register_buffer("camera_from_reference", torch.as_tensor(camera_from_reference, dtype=torch.float32))
        self.register_buffer("intrinsics", torch.as_tensor(intrinsics, dtype=torch.float32))
        self.register_buffer("image_transform", None if image_transform is None else torch.as_tensor(image_transform, dtype=torch.float32))
        self.register_buffer("camera_valid", torch.ones(2, dtype=torch.bool) if camera_valid is None else torch.as_tensor(camera_valid, dtype=torch.bool))

    @staticmethod
    def _batch(value: torch.Tensor, batch: int, ndim: int) -> torch.Tensor:
        return value.unsqueeze(0).expand(batch, *value.shape) if value.ndim == ndim - 1 else value

    def project(self, q_position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = q_position.float()
        # FK cancellation in the wrist reference and pixel projection remain
        # fp32 even when the learned visual/LeWM layers train under bf16 AMP.
        with torch.autocast(device_type=q.device.type, enabled=False):
            reference = self.kinematics.anchors_in_view_references(q)
            index = [self.kinematics.spec.fixed_tip_index, self.kinematics.spec.moving_tip_index]
            reference = reference[:, :, index]
            batch = q.shape[0]
            transform = self._batch(self.camera_from_reference.to(q), batch, 4)
            intrinsic = self._batch(self.intrinsics.to(q), batch, 4)
            image = None if self.image_transform is None else self._batch(self.image_transform.to(q), batch, 4)
            pixels, _ = project_reference_points(reference, transform, intrinsic, image)
            homogeneous = torch.cat((reference, torch.ones_like(reference[..., :1])), -1)
            camera_points = torch.einsum("bvij,bvaj->bvai", transform, homogeneous)
        camera_valid = self._batch(self.camera_valid, batch, 2)
        # Off-screen motion remains geometrically meaningful for propagation.
        valid = camera_valid[..., None] & (camera_points[..., 2] > 1e-6) & torch.isfinite(pixels).all(-1)
        return torch.where(valid[..., None], pixels, torch.zeros_like(pixels)), valid

    forward = project

    def make_prior(self, q_position: torch.Tensor, sigma_px: float = 36.0) -> PairBelief:
        pixels, valid = self.project(q_position)
        means = pixels[:, :, None].expand(-1, -1, 4, -1, -1)
        covariance = torch.eye(2, device=pixels.device).expand(*means.shape[:3], 2, 2) * sigma_px**2
        # Visibility cannot be inferred from being in frame: geometry is broad.
        prior = pair_from_means(means, covariance, torch.full_like(means[..., 0], 0.5))
        unknown = broad_pair_prior(pixels.shape[0], pixels.device)
        return _select_belief(valid.all(-1), prior, unknown)

    def propagate(self, previous_actual: PairBelief, q_previous: torch.Tensor, q_predicted: torch.Tensor, q_std: torch.Tensor | None = None) -> PairBelief:
        before, before_valid = self.project(q_previous)
        after, after_valid = self.project(q_predicted)
        valid = before_valid.all(-1) & after_valid.all(-1)
        delta = after - before
        means = previous_actual.means_px + delta[:, :, None]
        covariance = previous_actual.covariance_px2.float() + torch.eye(2, device=means.device) * 4.0
        if q_std is not None:
            if q_std.shape != q_predicted.shape or (q_std < 0).any():
                raise ValueError("q_std must be non-negative calibrated-position uncertainty")
            # Jacobian uncertainty readout, not a learned second motion model.
            jacobian = []
            for joint in range(q_predicted.shape[-1]):
                shift = torch.zeros_like(q_predicted)
                shift[:, joint] = 0.01
                plus, _ = self.project(q_predicted + shift)
                minus, _ = self.project(q_predicted - shift)
                jacobian.append((plus - minus) / 0.02)
            jacobian = torch.stack(jacobian, -1)
            propagated = torch.einsum("bvaiq,bvajq,bq->bvaij", jacobian, jacobian, q_std.float().square()).mean(2)
            covariance = covariance + propagated[:, :, None]
        inside = ((means >= 0) & (means < IMAGE_SIZE)).all(-1)
        visibility = previous_actual.visible_probability * inside.to(means)
        predicted = pair_from_means(means, covariance, visibility, previous_actual.log_weights)
        # No camera means no invented pixel displacement; carry uncertainty.
        unknown = PairBelief(previous_actual.log_weights, previous_actual.center_px, previous_actual.direction, previous_actual.opening_px, previous_actual.covariance_px2 + torch.eye(2, device=means.device) * 16, previous_actual.visible_probability)
        return _select_belief(valid, predicted, unknown)
