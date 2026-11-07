"""Explicit spatial auxiliary supervision; no task policy or FG pseudo-GT leak."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FG_ARCHITECTURE = 'v21-scene-fg-pair-readout-v1'


class SceneFGReadout(nn.Module):
    """Per-camera readout of scene384; shared for real and predicted scenes."""
    def __init__(self):
        super().__init__()
        self.views = nn.ModuleList([nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 384),
            nn.GELU(), nn.Linear(384, 4)) for _ in range(2)])

    def forward(self, scene):
        if scene.shape[-2:] != (2, 384):
            raise ValueError('F/G readout requires [...,2,384] scenes')
        return torch.stack([head(scene[..., c, :]).reshape(*scene.shape[:-2], 2, 2)
                            for c, head in enumerate(self.views)], dim=-3)


def enable_fg_readout(model):
    if hasattr(model, 'fg_readout'):
        raise ValueError('F/G head already configured')
    model.fg_readout = SceneFGReadout().to(model.scene_prediction.weight)
    model.scene_fg_architecture = FG_ARCHITECTURE


def fg_checkpoint_mode(payload):
    value = payload.get('scene_fg_architecture')
    has_weights = any(k.startswith('lewm.fg_readout.') for k in payload['model'])
    if value not in (None, FG_ARCHITECTURE) or has_weights != (value == FG_ARCHITECTURE):
        raise ValueError('F/G checkpoint requires matching explicit spatial architecture')
    protocol = payload.get('native40_protocol', {})
    if 'scene_fg_architecture' in protocol and protocol['scene_fg_architecture'] != value:
        raise ValueError('F/G checkpoint/protocol metadata differ')
    return value


@dataclass(frozen=True)
class FGBatch:
    xy: torch.Tensor  # [...,2,2,2], letterbox224 /224, zero-filled only under false mask
    valid: torch.Tensor  # [...,2], actual image and admitted pseudo-label
    ordered: torch.Tensor  # [...,2], unordered Column pair uses assignment-invariant loss


def fg_batch(table, identities, *, device, image_valid):
    identities = np.asarray(identities)
    mask = identities >= 0
    xy = torch.zeros(*identities.shape, 2, 2, 2, device=device)
    valid = torch.zeros(*identities.shape, 2, device=device, dtype=torch.bool)
    ordered = torch.zeros_like(valid)
    if mask.any():
        result = table.gather(identities[mask])
        available = result['pair_available'] & result['current_image_valid']
        # Missing NaNs never enter the differentiable loss, even with a zero mask.
        clean = np.where(available[..., None, None], result['fg_px'], 0.)/224.
        if not np.isfinite(clean).all():
            raise ValueError('nonfinite admitted F/G label')
        at = torch.as_tensor(mask, device=device)
        xy[at] = torch.as_tensor(clean, device=device)
        valid[at] = torch.as_tensor(available, device=device)
        ordered[at] = torch.as_tensor(result['ordered_identity_available'], device=device)
    if image_valid.shape != valid.shape:
        raise ValueError('F/G camera/scene mask shape differs')
    return FGBatch(xy, valid & image_valid, ordered)


def memory_fg_batch(table, identities, capture_times, measurement_available, memory):
    """Label the actual last retained exposure separately for each camera.

    Missing measurements do not advance FCPG visual memory. An image-only
    readout of that memory must therefore retain its exposure's label, rather
    than silently using the latest robot/chain endpoint's F/G. Only the factual
    prefix used to build memory is supplied; future labels never enter here.
    """
    identities = np.asarray(identities)
    b, views, _ = memory.valid.shape
    if (views != 2 or identities.ndim != 2 or identities.shape[0] != b
            or identities.shape[1] < 1 or identities.dtype.kind not in 'iu'
            or (identities < 0).any()
            or capture_times.shape != (*identities.shape, 2)
            or capture_times.dtype != torch.float64
            or measurement_available.shape != capture_times.shape
            or measurement_available.dtype != torch.bool):
        raise ValueError('F/G memory labels require identified causal per-view exposures')
    device = memory.valid.device
    times = capture_times.to(device)
    available = measurement_available.to(device)
    latest_valid = memory.valid[:, :, -1]
    latest_times = memory.image_times[:, :, -1]
    # These timestamps were cloned into memory; exact equality identifies the
    # exposure, not a nearest-frame tolerance or a fabricated fresh timestamp.
    matched = (times.transpose(1, 2) == latest_times[:, :, None]) & available.transpose(1, 2)
    if (latest_valid & ~matched.any(-1)).any():
        raise ValueError('retained F/G exposure is absent from the factual input prefix')
    slots = torch.arange(identities.shape[1], device=device)[None, None, :]
    chosen = torch.where(matched, slots, -1).amax(-1).clamp_min(0)
    labels = fg_batch(table, identities, device=device, image_valid=available)
    batch = torch.arange(b, device=device)[:, None]
    view = torch.arange(2, device=device)[None, :]
    valid = labels.valid[batch, chosen, view] & latest_valid
    xy = labels.xy[batch, chosen, view]
    xy = torch.where(valid[..., None, None], xy, torch.zeros_like(xy))
    return FGBatch(xy, valid, labels.ordered[batch, chosen, view] & valid)


def pair_loss(prediction, labels):
    """Equal camera weights; preserve identity only when explicitly supported."""
    if prediction.shape != labels.xy.shape:
        raise ValueError('predicted and labelled F/G shapes differ')
    terms, reports = [], []
    for c in range(2):
        mask = labels.valid[..., c]
        if not mask.any():
            reports.append(dict(pairs=0, pixel_error_sum=0., mean_pixel_error=None))
            continue
        p, y = prediction[..., c, :, :].float()[mask], labels.xy[..., c, :, :][mask]
        same = F.smooth_l1_loss(p, y, beta=5/224., reduction='none').mean((-1,-2))
        swap = F.smooth_l1_loss(p, y.flip(-2), beta=5/224., reduction='none').mean((-1,-2))
        ordered = labels.ordered[..., c][mask]
        chosen = torch.where(ordered, same, torch.minimum(same, swap))
        terms.append(chosen.mean())
        direct_px = (p-y).norm(dim=-1).mean(-1)*224
        swapped_px = (p-y.flip(-2)).norm(dim=-1).mean(-1)*224
        distance = torch.where(ordered, direct_px, torch.minimum(direct_px, swapped_px))
        reports.append(dict(pairs=int(mask.sum()), pixel_error_sum=float(distance.detach().sum()),
                            mean_pixel_error=float(distance.detach().mean())))
    loss = torch.stack(terms).mean() if terms else prediction.float().sum()*0
    return loss, reports


def spatial_supervision(model, response, encoded_future, future, current_labels, future_labels):
    """All three paths train: current observed, real future encoder, prediction.

    Neither the real future encoder nor the spatial head is detached. Labels
    are not passed to model inputs, memory, actor or CEM. Archive has no fake
    fish geometry labels; caller only invokes this on the admitted fish domain.
    """
    head = model.fg_readout
    # Explicit FP32 head/loss after BF16 scene encoder/predictor matmuls.
    current = head(response.observation_scene[:, :, -1].float())
    target = head(encoded_future.float())
    predicted = head(response.predicted_scene[:, future.query_indices].float())
    pieces = [pair_loss(current, current_labels), pair_loss(target, future_labels), pair_loss(predicted, future_labels)]
    loss = sum(p[0] for p in pieces)/3
    persistence = current[:, None].expand_as(predicted)
    # A cold/missing camera has no observed scene to persist. Keep its true
    # future supervision, but never make an all-zero latent a fake baseline.
    paired_valid = future_labels.valid & response.observation_valid[:, None, :, -1]
    paired_labels = FGBatch(future_labels.xy, paired_valid, future_labels.ordered)
    _, baseline = pair_loss(persistence, paired_labels)
    _, paired_prediction = pair_loss(predicted, paired_labels)
    return loss, dict(scene_fg_loss=float(loss.detach()),
        scene_fg_current=pieces[0][1], scene_fg_future_encoding=pieces[1][1],
        scene_fg_prediction=pieces[2][1], scene_fg_persistence=baseline,
        scene_fg_prediction_same_mask=paired_prediction,
        scene_fg_persistence_mask_source='actual-latest-view-and-future-label-v1',
        scene_fg_target_kind='FK/fused-pseudo-label; not independent physical ground truth')


def audit_spatial_gradients(loss, model, response, encoded_future, future, current_labels, future_labels):
    """Canary only: isolate this auxiliary from the existing scene/SIGReg loss."""
    gradients = torch.autograd.grad(loss, (response.observation_scene, encoded_future,
        response.predicted_scene, *model.fg_readout.parameters()), retain_graph=True)
    paths = (('current_encoding', gradients[0][:, :, -1], current_labels.valid),
             ('future_encoding', gradients[1], future_labels.valid),
             ('future_prediction', gradients[2][:, future.query_indices], future_labels.valid))
    result = {}
    for name, gradient, mask in paths:
        magnitude = gradient.float().abs().sum(-1)
        if not mask.any() or not torch.isfinite(gradient).all() or not (magnitude[mask] > 0).all():
            raise ValueError('spatial canary: missing/nonfinite F/G gradient on '+name)
        result[name] = dict(valid_pairs=int(mask.sum()),
            gradient_l1=float(magnitude[mask].sum()),
            valid_pairs_by_time_and_camera=mask.sum(0).tolist())
    if not all(torch.isfinite(g).all() for g in gradients[3:]) or not sum(
            float(g.abs().sum()) for g in gradients[3:]) > 0:
        raise ValueError('spatial canary: F/G head has no finite gradient')
    return dict(spatial_only=True, head_gradient_verified=True,
        query_indices_one_based=(future.query_indices+1).tolist(), paths=result)
