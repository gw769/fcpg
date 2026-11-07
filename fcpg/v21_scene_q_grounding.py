"""Archive proprioceptive grounding: read joint angles back from scene384.

The fish half already has an explicit spatial readout (F/G fingertips) and its
within-episode latent motion recovered once that readout was trained. Archive
has no F/G labels, but every exposure stores the joint angles recorded at its
capture time. This head mirrors the F/G readout on the same three paths
(current observed scene, real future encoder, predicted scene) with the label
replaced by capture-time q. Labels never enter model inputs, memory, Actor or
CEM, and the scene encoder still sees only DINO patches, so q can only be
recovered from the image.
"""
import torch
from torch import nn
from torch.nn import functional as F

Q_ARCHITECTURE = 'v21-scene-q-readout-v1'
Q_TARGET_KIND = 'recorded capture-time joint angles of each exposure; never a model input'
Q_LOSS_BETA = .02  # normalized joint units (q_half_range ~ 100-150 deg, so ~2-3 deg)


class SceneQReadout(nn.Module):
    """Per-camera q readout of scene384; shared for real and predicted scenes."""
    def __init__(self):
        super().__init__()
        self.views = nn.ModuleList([nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 384),
            nn.GELU(), nn.Linear(384, 6)) for _ in range(2)])

    def forward(self, scene):
        if scene.shape[-2:] != (2, 384):
            raise ValueError('q readout requires [...,2,384] scenes')
        return torch.stack([head(scene[..., c, :]) for c, head in enumerate(self.views)], dim=-2)


def enable_q_readout(model, *, seed=0):
    """Add the head without consuming the caller's CPU or CUDA training RNG."""
    if hasattr(model, 'q_readout'):
        raise ValueError('q readout already configured')
    if type(seed) is not int or seed < 0:
        raise ValueError('q readout initialization needs a nonnegative integer seed')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = SceneQReadout()
    model.q_readout = head.to(model.scene_prediction.weight)
    model.scene_q_architecture = Q_ARCHITECTURE


def q_checkpoint_mode(payload):
    value = payload.get('scene_q_architecture')
    has_weights = any(k.startswith('lewm.q_readout.') for k in payload['model'])
    if value not in (None, Q_ARCHITECTURE) or has_weights != (value == Q_ARCHITECTURE):
        raise ValueError('q-readout checkpoint requires matching explicit q architecture')
    protocol = payload.get('native40_protocol', {})
    if 'scene_q_architecture' in protocol and protocol['scene_q_architecture'] != value:
        raise ValueError('q-readout checkpoint/protocol metadata differ')
    return value


def q_loss(model, prediction, q, valid):
    """Equal camera weights; smooth-L1 in normalized joint units, MAE in degrees."""
    if prediction.shape != q.shape or valid.shape != q.shape[:-1] or q.shape[-2:] != (2, 6):
        raise ValueError('q readout, label and mask shapes differ')
    if valid.dtype != torch.bool:
        raise ValueError('q mask must be bool')
    terms, reports = [], []
    half = model.position_half_range.float()
    for c in range(2):
        mask = valid[..., c]
        if not mask.any():
            reports.append(dict(pairs=0, mae_deg=None))
            continue
        labels = q[..., c, :][mask].float()
        # Invalid slots may hold zeros or NaN; only admitted labels are read.
        if not torch.isfinite(labels).all():
            raise ValueError('nonfinite admitted capture q')
        p = prediction[..., c, :][mask].float()
        y = model.normalize_position(labels)
        terms.append(F.smooth_l1_loss(p, y, beta=Q_LOSS_BETA))
        error = ((p - y).abs() * half.to(p)).detach()
        reports.append(dict(pairs=int(mask.sum()), mae_deg=float(error.mean()),
                            mae_deg_by_joint=error.mean(0).tolist()))
    loss = torch.stack(terms).mean() if terms else prediction.float().sum() * 0
    return loss, reports


def q_grounding(model, response, encoded_future, future, current_q, current_valid):
    """All three paths train: current observed, real future encoder, prediction.

    Neither the real future encoder nor the head is detached, as in F/G.
    current_q is the capture q of the retained memory exposure each current
    scene was encoded from (memory.q_at_image[:, :, -1]).
    """
    head = model.q_readout
    future_q = getattr(future, 'capture_q', None)
    if future_q is None:
        raise ValueError('q grounding requires recorded capture q for every future target')
    current = head(response.observation_scene[:, :, -1].float())
    target = head(encoded_future.float())
    predicted = head(response.predicted_scene[:, future.query_indices].float())
    pieces = [q_loss(model, current, current_q, current_valid),
              q_loss(model, target, future_q, future.valid),
              q_loss(model, predicted, future_q, future.valid)]
    loss = sum(p[0] for p in pieces) / 3
    # Diagnostic only: persist the current readout. A cold camera has no
    # observed scene, so it never becomes a fake all-zero baseline.
    paired = future.valid & current_valid[:, None]
    _, baseline = q_loss(model, current[:, None].expand_as(predicted), future_q, paired)
    _, same_mask = q_loss(model, predicted, future_q, paired)
    return loss, dict(scene_q_loss=float(loss.detach()), scene_q_current=pieces[0][1],
        scene_q_future_encoding=pieces[1][1], scene_q_prediction=pieces[2][1],
        scene_q_persistence=baseline, scene_q_prediction_same_mask=same_mask,
        scene_q_target_kind=Q_TARGET_KIND)


def audit_q_gradients(loss, model, response, encoded_future, future, current_valid):
    """Canary only: the q objective alone reaches all three paths and the head."""
    gradients = torch.autograd.grad(loss, (response.observation_scene, encoded_future,
        response.predicted_scene, *model.q_readout.parameters()), retain_graph=True)
    paths = (('current_encoding', gradients[0][:, :, -1], current_valid),
             ('future_encoding', gradients[1], future.valid),
             ('future_prediction', gradients[2][:, future.query_indices], future.valid))
    result = {}
    for name, gradient, mask in paths:
        magnitude = gradient.float().abs().sum(-1)
        if not mask.any() or not torch.isfinite(gradient).all() or not (magnitude[mask] > 0).all():
            raise ValueError('q canary: missing/nonfinite q gradient on ' + name)
        result[name] = dict(valid_pairs=int(mask.sum()), gradient_l1=float(magnitude[mask].sum()))
    if not all(torch.isfinite(g).all() for g in gradients[3:]) or not sum(
            float(g.abs().sum()) for g in gradients[3:]) > 0:
        raise ValueError('q canary: q head has no finite gradient')
    return dict(q_only=True, head_gradient_verified=True, paths=result)
