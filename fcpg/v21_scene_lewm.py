"""LeWM-style learned scene targets, alongside (not instead of) full patches.

The scene encoder sees only observed DINO patches and patch positions. It has
no q/action/clock inputs that could satisfy SIGReg without looking at images.
Current and future observations share this trainable encoder: no EMA teacher,
stop-gradient target, RGB reconstruction or future DINO-patch reconstruction.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .v21_full_patch_lewm import DIM, HEADS, V21FullPatchLeWM
from .v21_models import ARMOutput
from .v21_command_memory import LEGACY_COMMANDS

SCENE_ARCHITECTURE = 'v21-full-patch-learned-scene-v1'
SCENE_ARCHITECTURES = {
    'batchnorm': SCENE_ARCHITECTURE,
    'layernorm': 'v21-full-patch-learned-scene-layernorm-v1',
}
ABSOLUTE_SCENE = 'absolute-v1'
RESIDUAL_SCENE = 'current-residual-v1'
FP32_RESIDUAL_SCENE = 'current-residual-fp32-v2'
SCENE_PREDICTION_MODES = (ABSOLUTE_SCENE, RESIDUAL_SCENE, FP32_RESIDUAL_SCENE)
RESIDUAL_SCENE_ARCHITECTURES = {
    'batchnorm': 'v21-full-patch-residual-scene-v1',
    'layernorm': 'v21-full-patch-residual-scene-layernorm-v1',
}
FP32_RESIDUAL_SCENE_ARCHITECTURES = {
    'batchnorm': 'v21-full-patch-residual-scene-fp32-v2',
    'layernorm': 'v21-full-patch-residual-scene-layernorm-fp32-v2',
}
SCENE_SIGREG_SOURCE = 'observation-only-scene-encoder-per-view-time-batch-v1'
SCENE_BRANCH_PREFIXES = ('scene_encoder.', 'scene_context.', 'scene_prediction.')
SCENE_BRANCH_RESET = 'fresh-seeded-scene-branch-v1'


def scene_architecture(normalization, prediction=ABSOLUTE_SCENE):
    if prediction not in SCENE_PREDICTION_MODES or normalization not in SCENE_ARCHITECTURES:
        raise ValueError('unsupported scene normalization or prediction mode')
    architectures = {ABSOLUTE_SCENE: SCENE_ARCHITECTURES,
                     RESIDUAL_SCENE: RESIDUAL_SCENE_ARCHITECTURES,
                     FP32_RESIDUAL_SCENE: FP32_RESIDUAL_SCENE_ARCHITECTURES}
    return architectures[prediction][normalization]


def scene_settings_from_checkpoint(payload):
    """Old BN/LN mean absolute prediction; new residual semantics are explicit."""
    architecture = payload.get('scene_lewm_architecture')
    settings = next(((normalization, prediction) for normalization in SCENE_ARCHITECTURES
                     for prediction in SCENE_PREDICTION_MODES
                     if scene_architecture(normalization, prediction) == architecture), None)
    if settings is None:
        raise ValueError('unsupported learned scene architecture')
    normalization, prediction = settings
    if prediction != ABSOLUTE_SCENE and payload.get('scene_prediction_mode') != prediction:
        raise ValueError('residual scene checkpoint requires explicit prediction mode')
    for metadata in (payload, payload.get('native40_protocol', {})):
        if (metadata.get('scene_normalization', normalization) != normalization
                or metadata.get('scene_lewm_architecture', architecture) != architecture):
            raise ValueError('scene normalization metadata disagrees with checkpoint architecture')
        if metadata.get('scene_prediction_mode', prediction) != prediction:
            raise ValueError('scene prediction metadata disagrees with checkpoint architecture')
    return settings


def scene_normalization_from_checkpoint(payload):
    normalization, _ = scene_settings_from_checkpoint(payload)
    return normalization


def scene_prediction_from_checkpoint(payload):
    _, prediction = scene_settings_from_checkpoint(payload)
    return prediction


def reset_scene_output(model):
    """Identical zero head for absolute/residual controls; no RNG consumption."""
    nn.init.zeros_(model.scene_prediction.weight)
    nn.init.zeros_(model.scene_prediction.bias)


class ObservationSceneEncoder(nn.Module):
    """Learned spatial readout per camera; never replaces the 1536-patch bank."""

    def __init__(self, *, normalization='batchnorm'):
        super().__init__()
        if normalization not in SCENE_ARCHITECTURES:
            raise ValueError('unsupported scene normalization')
        self.position = nn.Parameter(torch.randn(1, 256, DIM) * .02)
        self.query = nn.Parameter(torch.randn(1, 1, DIM) * .02)
        self.norm = nn.LayerNorm(DIM)
        self.attention = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
        self.project_in = nn.Linear(DIM, DIM * 4)
        # BN is the unchanged baseline. LN normalizes each observation on its
        # own channels, so current/future batches and online batch=1 share the
        # same mapping without running-statistics drift.
        self.project_norm = (nn.BatchNorm1d(DIM * 4) if normalization == 'batchnorm'
                             else nn.LayerNorm(DIM * 4))
        self.project_out = nn.Linear(DIM * 4, DIM)

    def forward(self, patches, valid):
        if patches.shape != (*valid.shape, 256, DIM) or valid.dtype != torch.bool:
            raise ValueError('scene observations require patches [...,256,384] and bool validity')
        selected = patches[valid].to(self.position)
        result = self.position.new_zeros(*valid.shape, DIM)
        if not len(selected):
            return result
        if not torch.isfinite(selected).all():
            raise ValueError('nonfinite scene observation')
        tokens = self.norm(selected + self.position)
        query = self.query.expand(len(selected), -1, -1)
        readout = self.attention(query, tokens, tokens, need_weights=False)[0][:, 0]
        hidden = self.project_in(readout)
        # An isolated valid camera is legal at inference or in a masked tail.
        # It must not update BN with a fabricated duplicate observation.
        if isinstance(self.project_norm, nn.BatchNorm1d) and self.training and len(selected) == 1:
            bn = self.project_norm
            hidden = F.batch_norm(hidden, bn.running_mean, bn.running_var,
                                  bn.weight, bn.bias, training=False, eps=bn.eps)
        else:
            hidden = self.project_norm(hidden)
        encoded = self.project_out(F.gelu(hidden))
        return result.index_put((valid,), encoded.to(result))


@dataclass(frozen=True)
class SceneARMOutput(ARMOutput):
    predicted_scene: torch.Tensor | None = None  # [B,H,2,384]
    observation_scene: torch.Tensor | None = None  # [B,2,3,384]
    observation_valid: torch.Tensor | None = None
    scene_anchor_valid: torch.Tensor | None = None  # [B,2], not a prediction-success flag


class V21SceneLeWM(V21FullPatchLeWM):
    architecture = SCENE_ARCHITECTURE

    def __init__(self, *, increment_scale=None, horizon=40, normalization='batchnorm',
                 command_conditioning=LEGACY_COMMANDS, scene_prediction_mode=ABSOLUTE_SCENE):
        super().__init__(increment_scale=increment_scale, horizon=horizon)
        self.scene_encoder = ObservationSceneEncoder(normalization=normalization)
        self.scene_normalization = normalization
        self.scene_prediction_mode = scene_prediction_mode
        self.architecture = scene_architecture(normalization, scene_prediction_mode)
        self.scene_context = nn.Linear(2 * 3 * DIM, DIM, bias=False)
        # Exact q-predictor warm start. The new latent input learns a residual;
        # none of the original visual patches/FiLM/cross-attention are removed.
        nn.init.zeros_(self.scene_context.weight)
        self.scene_prediction = nn.Linear(DIM, 2 * DIM)
        # Create additions LAST so a fixed seed leaves every baseline scene
        # and dynamics parameter identical in the command-input ablation.
        self.enable_command_conditioning(command_conditioning)

    def forward(self, q_position, trace, history, query_offsets, *, patches=None,
                patch_valid=None, memory=None, now_time=None):
        encoded = valid = context = None
        if patches is not None:
            valid = (torch.ones(patches.shape[:3], device=patches.device, dtype=torch.bool)
                     if patch_valid is None else patch_valid)
            encoded = self.scene_encoder(patches, valid)
            context = self.scene_context(encoded.flatten(1))
        response = super().forward(q_position, trace, history, query_offsets,
            patches=patches, patch_valid=patch_valid, memory=memory,
            now_time=now_time, scene_context=context)
        predicted = self.scene_prediction(response.hidden).reshape(
            len(q_position), query_offsets.shape[1], 2, DIM)
        anchor_valid = (valid[:, :, -1] if valid is not None else
                        torch.zeros(len(q_position), 2, dtype=torch.bool, device=q_position.device))
        if self.scene_prediction_mode != ABSOLUTE_SCENE and encoded is not None:
            # Missing observations are zero with explicit false validity; never
            # substitute a future image or relabel an old invalid slot as current.
            # Keep the anchor and the future encoder in the gradient graph.
            anchor = torch.where(anchor_valid[..., None], encoded[:, :, -1], 0.)
            if self.scene_prediction_mode == FP32_RESIDUAL_SCENE:
                # BF16 matmuls are retained, but rounding anchor+delta to BF16
                # can erase small predicted motion before the FP32 loss sees it.
                # Cast BEFORE the sum, without detaching either gradient path.
                # Legacy v1 checkpoints keep their original forward semantics.
                predicted = predicted.float() + anchor[:, None].float()
            else:
                predicted = predicted + anchor[:, None].to(predicted)
        return SceneARMOutput(response.predicted_q_path, response.q_uncertainty,
            response.hidden, response.regularization_hidden, predicted, encoded, valid, anchor_valid)


def reset_scene_branch(model, *, seed):
    """Fresh scene target/context/output only, without consuming training RNG.

    Copy into existing parameters before optimizer creation. Fresh default
    encoder/output initialization and zero context preserve the architecture,
    full visual bank, joint heads and causal command backbone. This does not
    promise identical q predictions: the learned scene context is being reset.
    """
    if (not isinstance(model, V21SceneLeWM) or model.scene_normalization != 'layernorm'
            or model.scene_prediction_mode != ABSOLUTE_SCENE):
        raise ValueError('scene-branch reset requires the existing absolute LN architecture')
    if type(seed) is not int or seed < 0:
        raise ValueError('scene-branch reset requires a nonnegative integer seed')
    # Only the CPU generator is seeded; CUDA dropout/SIGReg streams are not
    # touched. fork_rng restores CPU state after the independent initialization.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        encoder = ObservationSceneEncoder(normalization='layernorm')
        context = nn.Linear(2 * 3 * DIM, DIM, bias=False)
        nn.init.zeros_(context.weight)
        prediction = nn.Linear(DIM, 2 * DIM)
    model.scene_encoder.load_state_dict(encoder.state_dict(), strict=True)
    model.scene_context.load_state_dict(context.state_dict(), strict=True)
    model.scene_prediction.load_state_dict(prediction.state_dict(), strict=True)


def masked_observation_sigreg(observations, valid, sigreg):
    """Original Epps-Pulley statistic across B, separately for each view/time.

    Do not flatten time into B: the original statistic scales by batch size.
    Masking missing exposures must change that count rather than supply zeros.
    """
    values = observations.flatten(1, 2).transpose(0, 1).float()  # [view/time,B,D]
    mask = valid.flatten(1, 2).transpose(0, 1)
    count = mask.sum(1)
    usable = count >= 2
    if not usable.any():
        return values.sum() * 0
    values, mask, count = values[usable], mask[usable], count[usable]
    projection = torch.randn(values.shape[-1], sigreg.num_proj, device=values.device)
    projection = projection / projection.norm(dim=0, keepdim=True).clamp_min(1e-6)
    phase = (values @ projection)[..., None] * sigreg.t
    weight = mask[..., None, None].to(values)
    mean_cos = (phase.cos() * weight).sum(1) / count[:, None, None]
    mean_sin = (phase.sin() * weight).sum(1) / count[:, None, None]
    error = (mean_cos - sigreg.phi).square() + mean_sin.square()
    return ((error @ sigreg.weights) * count[:, None]).mean()


def scene_prediction_loss(model, response, future, sigreg=None, *, encoded_target=None):
    """Future encoder remains in the gradient graph, as in original LeWM."""
    target = model.scene_encoder(future.features, future.valid) if encoded_target is None else encoded_target
    prediction = response.predicted_scene[:, future.query_indices].float()
    valid = future.valid
    squared = (prediction - target.float()).square().mean(-1)
    zero = response.hidden.float().sum() * 0
    mse = squared[valid].mean() if valid.any() else zero
    regularization, stds = zero, []
    if response.observation_scene is not None:
        slots = future.regularization_slots()
        all_observed = torch.cat((response.observation_scene.transpose(1, 2), target[:, slots]), 1)
        all_valid = torch.cat((response.observation_valid.transpose(1, 2), valid[:, slots]), 1)
        if sigreg is not None:
            regularization = masked_observation_sigreg(all_observed, all_valid, sigreg)
        for camera in range(2):
            current = response.observation_scene[:, camera][response.observation_valid[:, camera]]
            observed_future = target[:, :, camera][valid[:, :, camera]]
            observed = torch.cat((current, observed_future), 0).float()
            if len(observed) >= 2:
                stds.append(observed.detach().std(0, unbiased=False).mean())
    # Persistence baseline uses the SAME encoder at the SAME checkpoint. Raw
    # latent MSE across independently learned representations is not comparable.
    current_valid = response.observation_valid[:, :, -1]
    persistence_valid = valid & current_valid[:, None]
    persistence = (target.float() - response.observation_scene[:, :, -1, :][:, None].float()).square().mean(-1)
    paired_mse = float(squared[persistence_valid].mean().detach()) if persistence_valid.any() else None
    paired_baseline = float(persistence[persistence_valid].mean().detach()) if persistence_valid.any() else None
    report = dict(scene_mse=float(mse.detach()), scene_sigreg=float(regularization.detach()),
        scene_targets=int(valid.sum()), scene_persistence_targets=int(persistence_valid.sum()),
        scene_persistence_mse=paired_baseline, scene_same_mask_mse=paired_mse,
        scene_same_mask_relative_error=(paired_mse/paired_baseline if paired_baseline else None),
        scene_std_mean=float(torch.stack(stds).mean()) if stds else None)
    return mse, regularization, report
