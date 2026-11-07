"""Current image fields -> one actual-q/history conditioned dual-view update.

The fixed tool camera uses mechanical Wrist means; Column uses four paired
queries and two cross-attention layers. A conditioned result is a posterior,
not an independent likelihood to multiply by the prior a second time.
"""
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .prod_v20_1_belief import JointPairBelief, broad_pair_prior, select_pair_belief
from .prod_v20_1_observer import ObserverOutput, RawObserverMeasurement, V201PairObserver, POOL_CHANNELS
from .prod_v20_policy import empty_history, _validate_history
from .prod_v20_observer import _grid
from .prod_v20_3_2_history import ObserverMotionHistory


@dataclass(frozen=True)
class V2032ObserverOutput(ObserverOutput):
    update_applied: bool = False
    spatial_update_valid: torch.Tensor | None = None
    out_of_support: torch.Tensor | None = None
    geometry_available: torch.Tensor | None = None
    candidate_belief: JointPairBelief | None = None
    raw_column_center_px: torch.Tensor | None = None
    raw_column_fm_px: torch.Tensor | None = None
    nominal_conflict: torch.Tensor | None = None


def output_like(raw, belief, **metadata):
    return V2032ObserverOutput(belief, raw.component_features, raw.scene_tokens,
        raw.coarse_center_log_posterior, raw.pooling_map, **metadata)


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_norm, self.kv_norm, self.ff_norm = (nn.LayerNorm(192) for _ in range(3))
        self.self_attention = nn.MultiheadAttention(192, 6, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(192, 6, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(192, 768), nn.GELU(), nn.Linear(768, 192))

    def forward(self, query, context, padding):
        q = self.q_norm(query)
        query = query + self.self_attention(q, q, q, need_weights=False)[0]
        query = query + self.cross_attention(self.q_norm(query), self.kv_norm(context),
            self.kv_norm(context), key_padding_mask=padding, need_weights=False)[0]
        return query + self.ff(self.ff_norm(query))


class V2032PairObserver(V201PairObserver):
    analytic_wrist = True
    raw_feature_cache = True
    reject_nominal_conflict = True

    def __init__(self, channels=64, prior_floor=.25, *, attention_dim=192,
                 attention_heads=6, attention_layers=2, evidence_threshold=.5,
                 max_column_residual_px=16.):
        super().__init__(channels, prior_floor)
        if (attention_dim, attention_heads, attention_layers) != (192, 6, 2):
            raise ValueError("V20.3.2 attention is fixed at 192D/6 heads/2 layers")
        if not 0 < evidence_threshold < 1 or max_column_residual_px != 16.:
            raise ValueError("invalid observer evidence/residual configuration")
        self.evidence_threshold = evidence_threshold
        self.max_column_residual_px = float(max_column_residual_px)
        for name in ("center_head", "fine_center_head", "mechanical_query", "pose_head", "component_weight_head"):
            delattr(self, name)
        # The high-resolution RGB field is supervised directly downstream.
        # Its eight-channel cache is an explicit storage bottleneck, not a
        # claim of lossless compression. DINO supplies additional context.
        self.spatial_projection = nn.Conv2d(POOL_CHANNELS, 192, 1)
        self.xy_projection = nn.Linear(2, 192)
        self.dino_context = nn.Linear(384, 192)
        self.joint_projection = nn.Linear(2, 192)
        self.joint_embeddings = nn.Parameter(torch.randn(1, 6, 192) * .02)
        self.history_projection = nn.Sequential(nn.Linear(19, 192), nn.GELU(), nn.Linear(192, 192))
        self.history_positions = nn.Parameter(torch.randn(1, 3, 192) * .02)
        self.prior_projection = nn.Linear(10, 192)
        self.column_queries = nn.Parameter(torch.randn(1, 4, 192) * .02)
        self.column_attention = nn.ModuleList([CrossAttention(), CrossAttention()])
        self.column_state_head = nn.Linear(192, 4)  # local centre, rotation, log span
        self.column_scale_head = nn.Linear(192, 4)
        self.column_visibility_head = nn.Linear(192, 2)
        self.column_weight_head = nn.Linear(192, 1)
        self.wrist_visibility_head = nn.Linear(POOL_CHANNELS, 2)
        self.register_buffer("joint_lower", torch.tensor([-100.] * 5 + [0.]))
        self.register_buffer("joint_upper", torch.tensor([100.] * 6))
        nn.init.normal_(self.column_state_head.weight, std=.005)
        nn.init.zeros_(self.column_state_head.bias)

    def measure(self, rgb, dino_tokens, view_valid, mechanical_prior=None):
        batch = len(rgb)
        if rgb.shape != (batch, 2, 3, 224, 224) or dino_tokens.shape != (batch, 2, 256, 384):
            raise ValueError("raw dual-view RGB/DINO shapes differ")
        valid = torch.as_tensor(view_valid, device=rgb.device, dtype=torch.bool)
        if valid.shape != (batch, 2):
            raise ValueError("view_valid must be [B,2]")
        clean_rgb = torch.where(valid[..., None, None, None], rgb, torch.zeros_like(rgb))
        clean_dino = torch.where(valid[..., None, None], dino_tokens, torch.zeros_like(dino_tokens))
        if not torch.isfinite(clean_rgb).all() or not torch.isfinite(clean_dino).all():
            raise ValueError("valid RGB/DINO must be finite")
        fused, high = self._visual(clean_rgb.reshape(-1, 3, 224, 224), clean_dino.reshape(-1, 256, 384))
        field = self._pool_field(fused, high).reshape(batch, 2, POOL_CHANNELS, 56, 56)
        field = torch.where(valid[..., None, None, None], field, torch.zeros_like(field))
        # Stored belief fields are neutral sentinels for the established cache
        # layout. They never stand in for a q-conditioned model observation.
        return ObserverOutput(broad_pair_prior(batch, rgb.device),
            field.new_zeros(batch, 2, 4, 384), clean_dino.detach(),
            field.new_full((batch, 2, 4, 3136), -math.log(3136)), field)

    measurement = measure

    def _column_center_output(self, center, nominal_center, geometric):
        delta = center - nominal_center
        limited = nominal_center + delta.clamp(-self.max_column_residual_px, self.max_column_residual_px)
        return torch.where(geometric[:, None, None], limited, center)

    def _column_shape_reference(self, prior, nominal):
        return prior.state_mean[:, 1].float()

    def _column_visibility(self, query):
        return self.column_visibility_head(query).sigmoid()

    def _column_correspondence(self, query, spatial, half):
        return torch.einsum("bkd,bnd->bkn", query.float(), spatial.float()) / math.sqrt(192)

    def _column_center_readout(self, logp):
        return torch.einsum("bkn,nd->bkd", logp.exp(), _grid(56, logp))

    def _column_query_prior(self, prior, nominal):
        return prior

    def measurement_for_cache(self, rgb, dino_tokens, view_valid, mechanical_prior=None):
        raw = self.measure(rgb, dino_tokens, view_valid)
        return RawObserverMeasurement(raw.belief, raw.pooling_map)

    def _context(self, raw, q, history, available):
        field = raw.pooling_map[:, 1].to(self.spatial_projection.weight)
        spatial = self.spatial_projection(field).flatten(2).transpose(1, 2)
        xy = (_grid(56, spatial) + .5) / 112. - 1
        spatial = spatial + self.xy_projection(xy.to(self.xy_projection.weight))[None]
        scene = self.dino_context(F.layer_norm(raw.scene_tokens[:, 1].to(spatial), (384,)))
        span = (self.joint_upper - self.joint_lower).to(q)
        normalized = (q - self.joint_lower.to(q)) / span * 2 - 1
        normalized = torch.where(available[:, None], normalized, torch.zeros_like(normalized))
        joints = self.joint_projection(torch.stack((normalized, available[:, None].expand_as(q).to(q)), -1).to(spatial))
        joints = joints + self.joint_embeddings.to(spatial)
        if isinstance(history, ObserverMotionHistory):
            history.validate(len(q))
        else:
            _validate_history(history, len(q))
        hb = history.readbacks_position.to(q)
        hc = history.commands_position.to(q)
        h = torch.cat(((hb[:, :, 0] - self.joint_lower.to(q)) / span * 2 - 1,
            (hb[:, :, -1] - hb[:, :, 0]) / span,
            (hc[:, :, -1] - hb[:, :, -1]) / span,
            history.time_deltas.to(q).sum(-1, keepdim=True)), -1)
        h = torch.where(history.valid.to(q.device)[..., None], h, torch.zeros_like(h))
        historic = self.history_projection(h.to(spatial)) + self.history_positions.to(spatial)
        context = torch.cat((spatial, scene, joints, historic), 1)
        padding = torch.cat((torch.zeros(len(q), 3136 + 256, device=q.device, dtype=torch.bool),
            ~available[:, None].expand(-1, 6), ~history.valid.to(q.device)), -1)
        return spatial, context, padding

    def update(self, measurement, prior, view_valid, *, q_position, actual_history=None,
               q_available=None, mechanical_prior=None):
        if getattr(measurement, "update_applied", False):
            raise ValueError("observation already updated; reuse raw image features")
        q = torch.as_tensor(q_position, device=measurement.pooling_map.device, dtype=torch.float32)
        if q.shape != (len(measurement.pooling_map), 6) or not torch.isfinite(q).all():
            raise ValueError("q_position must be finite calibrated [B,6]")
        valid = torch.as_tensor(view_valid, device=q.device, dtype=torch.bool)
        if valid.shape != (len(q), 2):
            raise ValueError("view_valid must be [B,2]")
        available = torch.ones(len(q), device=q.device, dtype=torch.bool) if q_available is None else torch.as_tensor(q_available, device=q.device, dtype=torch.bool).expand(len(q))
        history = empty_history(len(q), device=q.device) if actual_history is None else actual_history
        geometric = torch.zeros_like(valid)
        if prior is not None:
            prior.validate()
            # Unknown geometry is represented by identity .5 in the broad
            # prior; known projections use physical F/M identity 1.
            geometric = (prior.identity_probability - .5).abs().amax(-1) > 1e-8
        else:
            prior = broad_pair_prior(len(q), q.device)
        nominal = prior if mechanical_prior is None else mechanical_prior
        nominal.validate()
        geometric = (nominal.identity_probability - .5).abs().amax(-1) > 1e-8
        spatial, context, padding = self._context(measurement, q, history, available)
        query_prior = self._column_query_prior(prior, nominal)
        p = query_prior.state_mean[:, 1].float()
        stats = torch.cat((p / 224., query_prior.state_covariance[:, 1].diagonal(dim1=-2, dim2=-1).float().sqrt()/224.,
            query_prior.weights[:, 1, :, None].float(), geometric[:, 1, None, None].expand(-1, 4, 1).float()), -1)
        query = self.column_queries.to(spatial).expand(len(q), -1, -1) + self.prior_projection(stats.to(spatial))
        for layer in self.column_attention:
            query = layer(query, context, padding)
        # Every paired hypothesis has its OWN all-image correspondence map.
        state = self.column_state_head(query).float()
        shape_reference = self._column_shape_reference(prior, nominal)
        prior_angle = torch.atan2(shape_reference[..., 3], shape_reference[..., 2])
        angle = prior_angle + state[..., 2].tanh() * math.pi
        old_span = 2 * shape_reference[..., 2:].norm(dim=-1)
        span = torch.where(geometric[:, 1, None], old_span * state[..., 3].clamp(-2, 2).exp(),
                           F.softplus(state[..., 3]) * 32.)
        half = .5 * span[..., None] * torch.stack((angle.cos(), angle.sin()), -1)
        logits = self._column_correspondence(query, spatial, half)
        logp = logits.log_softmax(-1)
        center = self._column_center_readout(logp)
        center = center + state[..., :2].tanh() * 2.
        mean = prior.state_mean.clone()
        mean[:, 1] = torch.cat((center, half), -1).to(mean)
        # Bound the actual spatial output, not only the memory write. Always
        # reference this endpoint's nominal geometry; repeated refreshes may
        # not accumulate another 16 px relative to an already corrected prior.
        raw_delta = mean[:, 1, :, :2] - nominal.state_mean[:, 1, :, :2]
        mean[:, 1, :, :2] = self._column_center_output(
            mean[:, 1, :, :2], nominal.state_mean[:, 1, :, :2], geometric[:, 1])
        mean[:, 0] = nominal.state_mean[:, 0]
        covariance = prior.state_covariance.clone()
        covariance[:, 0] = nominal.state_covariance[:, 0]
        std = F.softplus(self.column_scale_head(query).float()) * 4 + .5
        covariance[:, 1] = torch.diag_embed(std.double().square())
        visibility = prior.visible_probability.clone()
        visibility[:, 1] = self._column_visibility(query).to(visibility)
        wrist_field = measurement.pooling_map[:, 0].to(self.wrist_visibility_head.weight).mean((-1, -2))
        visibility[:, 0] = self.wrist_visibility_head(wrist_field).sigmoid()[:, None].to(visibility)
        weights = prior.log_weights.clone()
        weights[:, 0] = nominal.log_weights[:, 0]
        weights[:, 1] = self.column_weight_head(query).float().squeeze(-1).log_softmax(-1).to(weights)
        identity = prior.identity_probability.clone()
        identity[:, 0] = nominal.identity_probability[:, 0]
        candidate = JointPairBelief(weights, mean, covariance, visibility, identity)
        confidence = (candidate.weights[..., None] * visibility).sum(-2).amin(-1)
        evidence = valid & (confidence > self.evidence_threshold)
        evidence[:, 0] &= geometric[:, 0]
        # A threshold expresses observation availability; it is never an
        # actuator gate. Its calibration must be measured on real occlusions.
        updated = select_pair_belief(evidence, candidate, prior)
        updated = replace(updated, visible_probability=torch.where(valid[..., None, None],
            visibility, torch.zeros_like(visibility)))
        fields = {}
        for name in ("state_mean", "state_covariance", "log_weights", "identity_probability"):
            value = getattr(updated, name).clone()
            value[:, 0] = getattr(nominal, name)[:, 0]
            fields[name] = value
        updated = replace(updated, **fields)
        out = torch.zeros_like(valid)
        selected = candidate.log_weights[:, 1].argmax(-1)
        map_delta = raw_delta[torch.arange(len(q), device=q.device), selected]
        out[:, 1] = geometric[:, 1] & (map_delta.abs().amax(-1) > self.max_column_residual_px)
        coarse = measurement.coarse_center_log_posterior.new_full((len(q), 2, 4, 3136), -math.log(3136))
        coarse[:, 1] = logp.to(coarse)
        raw = replace(measurement, coarse_center_log_posterior=coarse)
        result = output_like(raw, updated, update_applied=True, spatial_update_valid=evidence,
            out_of_support=out if self.reject_nominal_conflict else torch.zeros_like(out),
            nominal_conflict=out, geometry_available=geometric, candidate_belief=candidate,
            raw_column_center_px=center.detach(),
            raw_column_fm_px=torch.stack((center - half, center + half), -2).detach())
        return self.repool(result, updated, valid)

    def assimilate(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def forward(self, rgb, dino_tokens, view_valid, prior=None, *, q_position=None,
                actual_history=None, q_available=None, mechanical_prior=None):
        if q_position is None:
            raise ValueError("q_position must be explicit; use measure() to cache image fields")
        return self.update(self.measure(rgb, dino_tokens, view_valid), prior, view_valid,
            q_position=q_position, actual_history=actual_history, q_available=q_available,
            mechanical_prior=mechanical_prior)

    def training_observation(self, rgb, dino_tokens, view_valid, *, prior=None, q_position=None,
                             q_available=None, actual_history=None):
        return self.forward(rgb, dino_tokens, view_valid, prior, q_position=q_position,
                            q_available=q_available, actual_history=actual_history)

    def spatial_loss(self, observation, *args, **kwargs):
        # Supervise the current image candidate even when observation evidence
        # is insufficient to update runtime memory. No gradient through a
        # threshold is required to teach visibility or spatial correspondence.
        supervised = replace(observation, belief=observation.candidate_belief) if getattr(observation, "candidate_belief", None) is not None else observation
        return super().spatial_loss(supervised, *args, **kwargs)

    @staticmethod
    def center_loss(observation, target_fm_px, valid, *, view_counts=None):
        # A mixture is scored as a mixture; averaging the four log scores
        # would train every candidate to use the same peak.
        logp = observation.coarse_center_log_posterior
        mixed = torch.logsumexp(logp + observation.belief.log_weights[..., None].to(logp), dim=-2, keepdim=True)
        return V201PairObserver.center_loss(replace(observation,
            coarse_center_log_posterior=mixed), target_fm_px, valid, view_counts=view_counts)
