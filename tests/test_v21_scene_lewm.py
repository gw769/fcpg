"""Small CPU contracts for the new training path, without a robot or dataset."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.prod_v20_policy import empty_history
from fcpg.regularization import SIGReg
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_full_patch_lewm import V21FullPatchLeWM
from fcpg.v21_models import CommandTrace
from fcpg.v21_native_horizon import core_from_parent
from fcpg.v21_scene_lewm import (SCENE_ARCHITECTURE, ObservationSceneEncoder, V21SceneLeWM,
    masked_observation_sigreg, scene_prediction_loss)
from fcpg.v21_scene_targets import FutureSceneTargets, SceneTargets


def inputs(batch=2):
    torch.set_num_threads(2)
    q = torch.tensor([[0., -50., 50., 50., 0., 30.]]).repeat(batch, 1)
    dt = torch.full((batch, 5), .1)
    trace = CommandTrace.policy(q[:, None].repeat(1, 5, 1), dt)
    offsets = (torch.arange(40, dtype=torch.float64)[None] + 1).repeat(batch, 1) * .1
    patches = torch.randn(batch, 2, 3, 256, 384)
    return q, trace, empty_history(batch), offsets, patches


@pytest.mark.parametrize('normalization', ['batchnorm', 'layernorm'])
def test_q_warm_start_is_exact_and_patch_bank_stays_full(normalization):
    torch.manual_seed(2)
    q, trace, history, offsets, patches = inputs()
    baseline = V21FullPatchLeWM().eval()
    model = V21SceneLeWM(normalization=normalization).eval()
    missing, extra = model.load_state_dict(baseline.state_dict(), strict=False)
    assert not extra
    assert all(k.startswith(('scene_encoder.', 'scene_context.', 'scene_prediction.')) for k in missing)
    bank_shapes = []
    handle = model.layers[0].cross_attn.register_forward_pre_hook(
        lambda module, args: bank_shapes.append(tuple(args[1].shape)))
    with torch.no_grad():
        expected = baseline(q, trace, history, offsets, patches=patches)
        actual = model(q, trace, history, offsets, patches=patches)
    handle.remove()
    torch.testing.assert_close(actual.predicted_q_path, expected.predicted_q_path, atol=0, rtol=0)
    assert bank_shapes == [(2, 1536, 384)]
    assert actual.predicted_scene.shape == (2, 40, 2, 384)


@pytest.mark.parametrize('normalization', ['batchnorm', 'layernorm'])
def test_future_observation_and_encoder_receive_gradients_not_stop_gradient(normalization):
    torch.manual_seed(3)
    q, trace, history, offsets, patches = inputs()
    model = V21SceneLeWM(normalization=normalization).train()
    response = model(q, trace, history, offsets, patches=patches)
    features = torch.randn(2, 8, 2, 256, 384, requires_grad=True)
    future = SceneTargets(features, torch.ones(2, 8, 2, dtype=torch.bool), torch.arange(4, 40, 5))
    target = model.scene_encoder(features, future.valid)
    target.retain_grad()
    loss, reg, report = scene_prediction_loss(model, response, future, SIGReg(num_proj=16), encoded_target=target)
    (loss + .02 * reg).backward()
    for gradient in (target.grad, features.grad, model.scene_encoder.attention.in_proj_weight.grad,
                     model.scene_prediction.weight.grad, model.layers[0].cross_attn.in_proj_weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert report['scene_targets'] == 32


def test_observation_sigreg_preserves_original_batch_scaling_and_masks_padding():
    torch.set_num_threads(2)
    values = torch.randn(4, 3, 2, 16, requires_grad=True)
    valid = torch.ones(4, 3, 2, dtype=torch.bool)
    regularizer = SIGReg(num_proj=32)
    torch.manual_seed(44)
    expected = regularizer(values.flatten(1, 2).transpose(0, 1))
    torch.manual_seed(44)
    actual = masked_observation_sigreg(values, valid, regularizer)
    torch.testing.assert_close(actual, expected)
    valid[0] = False
    torch.manual_seed(44)
    masked = masked_observation_sigreg(values, valid, regularizer)
    torch.manual_seed(44)
    removed = regularizer(values[1:].flatten(1, 2).transpose(0, 1))
    torch.testing.assert_close(masked, removed)
    masked.backward()
    assert torch.count_nonzero(values.grad[0]) == 0


def test_future_lookup_requires_exact_cached_frame_and_does_not_cross_episode():
    # Endpoint 6 is in the following episode; it must never supply a tail label.
    local = np.array([0, 15, 30, 45, 60, 74, 0])
    archive = SimpleNamespace(row_endpoint_index=np.array([[0, 1, 2, 3, 4]]),
        endpoint_episode_index=np.array([0, 0, 0, 0, 0, 0, 1]),
        endpoint_local_frame_index=local, fps=np.array([30., 30.]))
    source = SimpleNamespace(archive=archive, offsets=np.array([0, 76, 300]),
        counts={'archive': 1}, archive_dino=[SimpleNamespace(endpoint_index=np.arange(7))])
    targets = FutureSceneTargets(source)
    actual = targets.identities(np.array([0]), 'archive', 0)
    np.testing.assert_array_equal(actual, [[1, 2, 3, 4, -1, -1, -1, -1]])
    # Frame 74 is NOT relabelled as the 75th-frame image.
    assert 5 not in actual and 6 not in actual


@pytest.mark.parametrize('normalization', ['batchnorm', 'layernorm'])
def test_scene_checkpoint_roundtrip_is_strict_and_restores_all_modules(normalization):
    torch.set_num_threads(2)
    core = V21ResearchCore(V21Config(4, prediction_horizon=40))
    core.lewm = V21SceneLeWM(normalization=normalization)
    payload = dict(config=asdict(core.config), model=core.state_dict(),
                   scene_lewm_architecture=core.lewm.architecture)
    # Existing BN checkpoints have no explicit normalization metadata.
    if normalization == 'layernorm':
        payload['scene_normalization'] = normalization
    loaded = core_from_parent(payload)
    assert isinstance(loaded.lewm, V21SceneLeWM)
    assert loaded.lewm.scene_normalization == normalization
    assert isinstance(loaded.lewm.scene_encoder.project_norm,
                      torch.nn.BatchNorm1d if normalization == 'batchnorm' else torch.nn.LayerNorm)
    for name, value in core.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])


def test_layernorm_scene_encoding_is_independent_of_mode_and_peer_observations():
    torch.set_num_threads(2)
    torch.manual_seed(21)
    model = ObservationSceneEncoder(normalization='layernorm')
    current = torch.randn(2, 2, 3, 256, 384)
    valid = torch.ones(2, 2, 3, dtype=torch.bool)
    with torch.no_grad():
        trained = model.train()(current, valid)
        evaluated = model.eval()(current, valid)
        # Current/future calls and different batch peers must not redefine the
        # latent target of an unchanged image. Nothing in this encoder drops out.
        alone = model.train()(current[:1, :1, :1], valid[:1, :1, :1])
        peers = torch.randn_like(current) * 9 + 4
        peers[0, 0, 0] = current[0, 0, 0]
        with_other_peers = model(peers, valid)
        joined = model(torch.cat((current, peers)), torch.cat((valid, valid)))
    for actual in (evaluated, joined[:2]):
        torch.testing.assert_close(actual, trained, rtol=1e-5, atol=1e-6)
    for actual in (alone[0, 0, 0], with_other_peers[0, 0, 0]):
        torch.testing.assert_close(actual, trained[0, 0, 0], rtol=1e-5, atol=1e-6)
    assert not any('running_' in name or 'num_batches_tracked' in name
                   for name in model.state_dict())


def test_norm_ablation_preserves_all_initial_parameters_and_the_absolute_scene_head():
    torch.set_num_threads(2)
    torch.manual_seed(21004)
    baseline = V21SceneLeWM(normalization='batchnorm')
    torch.manual_seed(21004)
    variant = V21SceneLeWM(normalization='layernorm')
    before, after = dict(baseline.named_parameters()), dict(variant.named_parameters())
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert torch.count_nonzero(variant.scene_context.weight) == 0
    assert torch.count_nonzero(variant.scene_prediction.weight) > 0


@pytest.mark.parametrize('normalization', ['batchnorm', 'layernorm'])
def test_scene_encoder_handles_single_valid_and_all_missing_views(normalization):
    torch.set_num_threads(2)
    model = ObservationSceneEncoder(normalization=normalization).train()
    patches = torch.randn(1, 2, 3, 256, 384, requires_grad=True)
    valid = torch.zeros(1, 2, 3, dtype=torch.bool)
    empty = model(patches, valid)
    assert empty.shape == (1, 2, 3, 384) and torch.count_nonzero(empty) == 0
    valid[0, 1, 2] = True
    encoded = model(patches, valid)
    assert torch.isfinite(encoded).all() and torch.count_nonzero(encoded[~valid]) == 0
    encoded.square().sum().backward()
    assert patches.grad[valid].abs().sum() > 0
    assert torch.count_nonzero(patches.grad[~valid]) == 0


def test_scene_loader_rejects_mislabelled_normalization_and_bn_weights_as_ln():
    torch.set_num_threads(2)
    core = V21ResearchCore(V21Config(4, prediction_horizon=40))
    core.lewm = V21SceneLeWM()
    payload = dict(config=asdict(core.config), model=core.state_dict(),
                   scene_lewm_architecture=SCENE_ARCHITECTURE, scene_normalization='layernorm')
    with pytest.raises(ValueError, match='normalization'):
        core_from_parent(payload)
    payload['scene_lewm_architecture'] = V21SceneLeWM(normalization='layernorm').architecture
    with pytest.raises(RuntimeError, match='running_mean'):
        core_from_parent(payload)
    payload['scene_lewm_architecture'] = 'unknown-scene-architecture'
    with pytest.raises(ValueError, match='architecture'):
        core_from_parent(payload)
