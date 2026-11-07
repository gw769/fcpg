"""Scene-only coefficient compatibility; no data, optimizer, or hardware."""
import pytest
import torch

from fcpg.v21_scene_lewm import SCENE_ARCHITECTURE, SCENE_SIGREG_SOURCE
from scripts.train_v21_horizon40 import (TRAINING_MODE_POLICY, canonical_protocol,
    main, parser, scene_sigreg_coefficient, scene_training_loss,
    validate_resource_continuation, validate_training_semantics)


def protocol():
    return dict(training_mode_policy=TRAINING_MODE_POLICY, sigreg_feature_source=SCENE_SIGREG_SOURCE,
        scene_lewm_architecture=SCENE_ARCHITECTURE, schedule=dict(scene_latent_lewm=True), device={})


def test_default_expression_gradient_and_rng_are_exact():
    torch.manual_seed(14)
    x = torch.randn(32, requires_grad=True)
    scene, reg, joint = x.square().mean(), x.sin().mean(), x.cos().mean()
    state = torch.get_rng_state().clone()
    old = scene + .02 * reg + joint
    new = scene_training_loss(scene, reg, joint)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    a = torch.autograd.grad(old, x, retain_graph=True)[0]
    b = torch.autograd.grad(new, x)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state())


def test_only_sigreg_gradient_scales():
    scene, reg, joint = (torch.tensor(2., requires_grad=True) for _ in range(3))
    for weight in (.02, .09):
        grads = torch.autograd.grad(scene_training_loss(scene, reg, joint, sigreg_weight=weight),
                                    (scene, reg, joint))
        assert [float(x) for x in grads] == pytest.approx([1., weight, 1.])


def test_legacy_scene_protocol_canonicalizes_to_point_zero_two():
    old, new = protocol(), protocol()
    new.update(sigreg_weight=.02)
    new['schedule']['scene_sigreg_weight'] = .02
    assert canonical_protocol(old) == canonical_protocol(new)
    validate_training_semantics(old, full_patch_lewm=False, scene_latent_lewm=True)
    validate_resource_continuation(old, new)
    assert scene_sigreg_coefficient({}) == .02


def test_changed_coefficient_cannot_resume_or_hide_as_resource_change():
    old, new = protocol(), protocol()
    new['sigreg_weight'] = .09
    new['schedule']['scene_sigreg_weight'] = .09
    with pytest.raises(ValueError, match='SIGReg weight changed'):
        validate_training_semantics(old, full_patch_lewm=False, scene_latent_lewm=True, scene_sigreg_weight=.09)
    with pytest.raises(ValueError, match='cannot change'):
        validate_resource_continuation(old, new)
    validate_training_semantics(new, full_patch_lewm=False, scene_latent_lewm=True, scene_sigreg_weight=.09)
    new['schedule']['scene_sigreg_weight'] = .02
    with pytest.raises(ValueError, match='disagrees'):
        canonical_protocol(new)


@pytest.mark.parametrize('weight', [0., -1., float('nan'), float('inf')])
def test_invalid_coefficient_rejected_before_reading_data(tmp_path, weight):
    with pytest.raises(ValueError, match='finite and positive'):
        main(['--parent', 'absent', '--source-config', 'absent', '--output', str(tmp_path/'absent'),
              '--scene-sigreg-weight', str(weight)])


def test_option_does_not_change_non_scene_training(tmp_path):
    assert parser().parse_args(['--parent', 'x', '--source-config', 'x', '--output', 'x']).scene_sigreg_weight == .02
    with pytest.raises(ValueError, match='requires --scene-latent-lewm'):
        main(['--parent', 'x', '--source-config', 'x', '--output', str(tmp_path/'x'), '--scene-sigreg-weight', '.09'])
