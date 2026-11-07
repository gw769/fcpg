from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_native_horizon import core_from_parent
from fcpg.v21_scene_fg import FG_ARCHITECTURE, enable_fg_readout
from fcpg.v21_scene_lewm import FP32_RESIDUAL_SCENE, V21SceneLeWM
from fcpg.v21_scene_q_grounding import (Q_ARCHITECTURE, audit_q_gradients, enable_q_readout,
    q_checkpoint_mode, q_grounding, q_loss)
from fcpg.v21_scene_targets import FutureSceneTargets


def scene_model():
    torch.set_num_threads(2)
    return V21SceneLeWM(normalization='layernorm', scene_prediction_mode=FP32_RESIDUAL_SCENE)


def labels(model, *shape):
    """Valid recorded joint angles inside the arm's bounds, in degrees."""
    u = torch.rand(*shape, 2, 6) * 1.6 - .8
    return model.position_center + u * model.position_half_range


def test_q_head_uses_its_own_seed_and_leaves_training_rng_untouched():
    a, b, c = scene_model(), scene_model(), scene_model()
    state = torch.random.get_rng_state()
    enable_q_readout(a, seed=7)
    assert torch.equal(state, torch.random.get_rng_state())
    enable_q_readout(b, seed=7)
    enable_q_readout(c, seed=8)
    for (k, x), y, z in zip(a.q_readout.state_dict().items(), b.q_readout.state_dict().values(),
                            c.q_readout.state_dict().values()):
        assert torch.equal(x, y)
        if k.endswith('weight') and x.ndim == 2:
            assert not torch.equal(x, z)
    assert a.scene_q_architecture == Q_ARCHITECTURE
    with pytest.raises(ValueError, match='already configured'):
        enable_q_readout(a, seed=7)


def test_invalid_or_nan_labels_never_reach_loss_and_mae_is_in_degrees():
    model = scene_model()
    q = labels(model, 3)
    valid = torch.ones(3, 2, dtype=torch.bool)
    valid[2, 1] = False
    q[2, 1] = float('nan')  # a missing exposure may carry NaN; it must stay masked
    delta = torch.zeros(3, 2, 6)
    delta[..., 0] = .01
    prediction = (model.normalize_position(torch.nan_to_num(q)) + delta).requires_grad_()
    loss, report = q_loss(model, prediction, q, valid)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()
    assert prediction.grad[2, 1].abs().sum() == 0
    assert [r['pairs'] for r in report] == [3, 2]
    expected = .01 * float(model.position_half_range[0]) / 6
    assert report[0]["mae_deg"] == pytest.approx(expected, rel=1e-3)
    q[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite admitted capture q'):
        q_loss(model, prediction, q, valid)


def test_q_grounding_trains_current_future_encoder_and_predictor_paths():
    model = scene_model()
    enable_q_readout(model, seed=3)
    current = torch.randn(2, 2, 3, 384, requires_grad=True)
    target = torch.randn(2, 40, 2, 384, requires_grad=True)
    predicted = torch.randn(2, 40, 2, 384, requires_grad=True)
    observed = torch.ones(2, 2, 3, dtype=torch.bool)
    response = SimpleNamespace(observation_scene=current, predicted_scene=predicted, observation_valid=observed)
    future = SimpleNamespace(query_indices=torch.arange(40), valid=torch.ones(2, 40, 2, dtype=torch.bool),
                             capture_q=labels(model, 2, 40))
    loss, report = q_grounding(model, response, target, future, labels(model, 2), observed[:, :, -1])
    audit = audit_q_gradients(loss, model, response, target, future, observed[:, :, -1])
    assert audit['head_gradient_verified']
    loss.backward()
    for x in (target, predicted):
        assert x.grad is not None and (x.grad.abs().sum((0, 2, 3)) > 0).all()
    assert current.grad[:, :, -1].abs().sum() > 0 and current.grad[:, :, :-1].abs().sum() == 0
    assert [r['pairs'] for r in report['scene_q_prediction']] == [80, 80]
    with pytest.raises(ValueError, match='recorded capture q'):
        q_grounding(model, response, target, SimpleNamespace(query_indices=future.query_indices,
                    valid=future.valid), labels(model, 2), observed[:, :, -1])


def test_cold_camera_has_no_fake_persistence_but_keeps_future_supervision():
    model = scene_model()
    enable_q_readout(model, seed=3)
    current = torch.zeros(2, 2, 3, 384, requires_grad=True)
    target = torch.randn(2, 40, 2, 384, requires_grad=True)
    predicted = torch.randn(2, 40, 2, 384, requires_grad=True)
    observed = torch.ones(2, 2, 3, dtype=torch.bool)
    observed[:, 1] = False
    response = SimpleNamespace(observation_scene=current, predicted_scene=predicted, observation_valid=observed)
    future = SimpleNamespace(query_indices=torch.arange(40), valid=torch.ones(2, 40, 2, dtype=torch.bool),
                             capture_q=labels(model, 2, 40))
    loss, report = q_grounding(model, response, target, future, labels(model, 2), observed[:, :, -1])
    assert [r['pairs'] for r in report['scene_q_current']] == [2, 0]
    assert [r['pairs'] for r in report['scene_q_prediction']] == [80, 80]
    for key in ('scene_q_persistence', 'scene_q_prediction_same_mask'):
        assert [r['pairs'] for r in report[key]] == [80, 0] and report[key][1]['mae_deg'] is None
    loss.backward()
    assert (target.grad[:, :, 1].abs().sum((0, 2)) > 0).all()


def test_q_checkpoint_needs_explicit_version_and_exact_head_restore():
    torch.set_num_threads(2)
    core = V21ResearchCore(V21Config(4, prediction_horizon=40))
    core.lewm = scene_model()
    enable_fg_readout(core.lewm)
    before = {k: v.clone() for k, v in core.lewm.state_dict().items()}
    enable_q_readout(core.lewm, seed=11)
    for k, v in before.items():
        torch.testing.assert_close(v, core.lewm.state_dict()[k], rtol=0, atol=0)
    payload = dict(config=asdict(core.config), model=core.state_dict(), scene_lewm_architecture=core.lewm.architecture,
        scene_normalization='layernorm', scene_prediction_mode=FP32_RESIDUAL_SCENE,
        scene_fg_architecture=FG_ARCHITECTURE, scene_q_architecture=Q_ARCHITECTURE)
    restored = core_from_parent(payload)
    for k, v in payload['model'].items():
        torch.testing.assert_close(v, restored.state_dict()[k], rtol=0, atol=0)
    assert q_checkpoint_mode(payload) == Q_ARCHITECTURE
    unlabelled = dict(payload)
    del unlabelled['scene_q_architecture']
    with pytest.raises(ValueError, match='explicit q architecture'):
        core_from_parent(unlabelled)
    headless = deepcopy(payload)
    headless['model'] = {k: v for k, v in payload['model'].items() if not k.startswith('lewm.q_readout.')}
    with pytest.raises(ValueError, match='explicit q architecture'):
        q_checkpoint_mode(headless)
    disagreeing = dict(payload, native40_protocol=dict(scene_q_architecture=None))
    with pytest.raises(ValueError, match='protocol metadata differ'):
        q_checkpoint_mode(disagreeing)


def test_future_targets_carry_capture_q_of_the_exact_target_exposure():
    """Endpoint e is frame 3e at 30 fps; its capture q is a value unique to e and the camera."""
    frames, endpoints = 76, 26
    local = np.arange(endpoints) * 3
    archive = SimpleNamespace(row_endpoint_index=np.array([[0, 1, 2, 3, 4]]),
        endpoint_episode_index=np.zeros(endpoints, np.int64), endpoint_local_frame_index=local,
        fps=np.array([30.]))
    times = np.arange(frames) / 30.
    capture_q = np.zeros((endpoints, 2, 6), np.float32)
    capture_q[:, 0] = np.arange(endpoints)[:, None] + .25
    capture_q[:, 1] = -np.arange(endpoints)[:, None] - .5
    capture_times = np.repeat(times[local][:, None], 2, 1)
    capture_times[2, 1] = 0.  # an old retained exposure: not a causal future label

    class Clock:
        def times(self, kind):
            return times

        def observations(self, ids, kind):
            ids = np.asarray(ids)
            return np.ones((len(ids), 2), bool), capture_times[ids], capture_q[ids]

    source = SimpleNamespace(archive=archive, offsets=np.array([0, frames]), counts={'archive': 1},
        archive_dino=[SimpleNamespace(endpoint_index=np.arange(endpoints))], recorded_clock=Clock())
    source._get_archive_dino = lambda ids: (np.zeros((len(ids), 2, 256, 384), np.float32),
                                            np.ones((len(ids), 2), bool))
    provider = FutureSceneTargets(source, supervision='dense40')
    ids = provider.identities(np.array([0]), 'archive', 0)
    future = provider.load(np.array([0]), 'archive', 0, 'cpu')
    assert future.capture_q.shape == (1, 40, 2, 6)
    for k in range(40):
        for c in range(2):
            if future.valid[0, k, c]:
                assert torch.equal(future.capture_q[0, k, c], torch.as_tensor(capture_q[ids[0, k], c]))
            else:
                assert future.capture_q[0, k, c].abs().sum() == 0
    assert ids[0, 1] == 2 and not future.valid[0, 1, 1] and future.valid[0, 1, 0]
    assert future.valid[0, :, 0].sum() == 25  # frames 3..75; no label past the recording


def test_q_grounding_options_and_resume_guard():
    from fcpg.v21_scene_lewm import SCENE_ARCHITECTURE, SCENE_SIGREG_SOURCE
    from scripts.train_v21_horizon40 import (TRAINING_MODE_POLICY, canonical_protocol,
        validate_q_grounding_options, validate_training_semantics)
    ok = dict(archive_q_weight=.1, scene_latent_lewm=True, lewm_only=True, from_scratch=False,
              lewm_mixed_only=True, lewm_archive_only_diagnostic=False, reset_scene_head=False,
              reset_scene_branch=False)
    validate_q_grounding_options(SimpleNamespace(**ok))
    validate_q_grounding_options(SimpleNamespace(**dict(ok, archive_q_weight=0., lewm_only=False)))
    for change in (dict(scene_latent_lewm=False), dict(lewm_only=False), dict(from_scratch=True),
                   dict(lewm_mixed_only=False), dict(reset_scene_branch=True), dict(reset_scene_head=True)):
        with pytest.raises(ValueError, match='q grounding requires'):
            validate_q_grounding_options(SimpleNamespace(**dict(ok, **change)))
    for weight in (-.1, float('inf'), float('nan')):
        with pytest.raises(ValueError, match='finite and nonnegative'):
            validate_q_grounding_options(SimpleNamespace(**dict(ok, archive_q_weight=weight)))
    old = dict(training_mode_policy=TRAINING_MODE_POLICY, sigreg_feature_source=SCENE_SIGREG_SOURCE,
               scene_lewm_architecture=SCENE_ARCHITECTURE, schedule=dict(scene_latent_lewm=True), device={})
    new = deepcopy(old)
    new['schedule']['archive_q_weight'] = 0.
    for key in ('scene_q_architecture', 'archive_q_weight', 'scene_q_domains', 'scene_q_target',
                'scene_q_initialization'):
        new[key] = None
    assert canonical_protocol(old) == canonical_protocol(new)  # legacy runs are unaffected
    with pytest.raises(ValueError, match='q grounding changed'):
        validate_training_semantics(old, full_patch_lewm=False, scene_latent_lewm=True, archive_q_weight=.1)
    grounded = deepcopy(old)
    grounded['schedule']['archive_q_weight'] = .1
    with pytest.raises(ValueError, match='q grounding changed'):
        validate_training_semantics(grounded, full_patch_lewm=False, scene_latent_lewm=True)
    validate_training_semantics(grounded, full_patch_lewm=False, scene_latent_lewm=True, archive_q_weight=.1)
