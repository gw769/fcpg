"""CPU-only mode/gradient checks: no data loading or optimizer steps."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from fcpg.prod_v20_policy import empty_history
from fcpg.regularization import SIGReg
from fcpg.v21_full_patch_lewm import V21FullPatchLeWM
from fcpg.v21_models import ARMOutput, CommandTrace
from fcpg.v21_scene_lewm import SCENE_ARCHITECTURE, SCENE_SIGREG_SOURCE, V21SceneLeWM
from scripts.train_v21_horizon40 import (
    SIGREG_FEATURE_SOURCE, TRAINING_MODE_POLICY, canonical_protocol, configure_training_stage,
    development_mode, lewm_sigreg_loss, temporary_eval,
    main, parser, validate_resource_continuation, validate_training_semantics,
)


class ModeCore(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('lewm', 'actor', 'reader', 'dino'):
            setattr(self, name, nn.Sequential(nn.Linear(8, 8), nn.Dropout(.1)))
        self._temporal_collect_training_coverage = True


@pytest.mark.parametrize('stage,scratch,active', [
    ('lewm_archive', False, {'lewm'}),
    ('lewm_fish', True, {'lewm'}),
    ('actor_fish', False, {'actor'}),
    ('actor_archive', True, {'actor', 'reader'}),
])
def test_training_modes_match_optimized_modules(stage, scratch, active):
    core = ModeCore().train()
    configure_training_stage(core, stage, from_scratch=scratch)
    for name in ('lewm', 'actor', 'reader', 'dino'):
        module = getattr(core, name)
        assert all(child.training == (name in active) for child in module.modules())
        assert all(p.requires_grad == (name in active) for p in module.parameters())


@pytest.mark.parametrize('failure', [False, True])
def test_development_restores_mixed_modes_and_coverage_even_on_failure(failure):
    core = ModeCore()
    configure_training_stage(core, 'lewm_archive')
    modes = [module.training for module in core.modules()]
    gradients = [p.requires_grad for p in core.parameters()]
    weights = deepcopy(core.state_dict())

    @torch.no_grad()
    @development_mode(core)
    def evaluate():
        assert not torch.is_grad_enabled()
        assert not any(module.training for module in core.modules())
        assert not core._temporal_collect_training_coverage
        with temporary_eval(core.lewm):
            assert not core.lewm.training
        if failure:
            raise RuntimeError('evaluation failed')

    for _ in range(2):
        if failure:
            with pytest.raises(RuntimeError, match='evaluation failed'):
                evaluate()
        else:
            evaluate()
        assert [module.training for module in core.modules()] == modes
        assert [p.requires_grad for p in core.parameters()] == gradients
        assert core._temporal_collect_training_coverage
    for name, tensor in core.state_dict().items():
        assert torch.equal(tensor, weights[name])


def test_history_forecasts_are_deterministic_and_training_dropout_recovers():
    torch.manual_seed(3)
    core = ModeCore()
    configure_training_stage(core, 'lewm_archive')
    x = torch.ones(64, 8)
    assert not torch.equal(core.lewm(x), core.lewm(x))
    with temporary_eval(core.lewm):
        assert all(p.requires_grad for p in core.lewm.parameters())
        assert torch.equal(core.lewm(x), core.lewm(x))
    assert core.lewm.training and core.lewm[1].training
    assert not core.actor.training and not core.reader.training


@pytest.mark.parametrize('horizon', [5, 15, 40])
def test_full_patch_exposes_pre_norm_queries_without_changing_prediction_head(horizon):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = V21FullPatchLeWM().eval()
    q = torch.tensor([[0., -50., 50., 50., 0., 30.]]).repeat(2, 1)
    dt = torch.full((2, 5), .1)
    trace = CommandTrace.policy(q[:, None].repeat(1, 5, 1), dt)
    offsets = (torch.arange(horizon, dtype=torch.float64)[None] + 1).repeat(2, 1) * .1
    captured = []
    hook = model.norm.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        with torch.no_grad():
            result = model(q, trace, empty_history(2), offsets)
    finally:
        hook.remove()
    assert result.regularization_hidden is captured[0]
    assert result.regularization_hidden.shape == (2, horizon, 384)
    assert not torch.equal(result.regularization_hidden, result.hidden)
    torch.testing.assert_close(result.hidden, model.norm(captured[0]), rtol=0, atol=0)
    expected = trace.active_at(offsets, q) + model.response(result.hidden) * model.increment_scale
    torch.testing.assert_close(result.predicted_q_path[:, 1:], expected, rtol=0, atol=0)
    assert torch.equal(result.predicted_q_path[:, 0], q)


def test_pre_norm_sigreg_reaches_visual_attention_and_keeps_full_patch_bank():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    model = V21FullPatchLeWM().train()
    q = torch.tensor([[0., -50., 50., 50., 0., 30.]]).repeat(2, 1)
    dt = torch.full((2, 5), .1)
    trace = CommandTrace.policy(q[:, None].repeat(1, 5, 1), dt)
    patches = torch.randn(2, 2, 3, 256, 384, requires_grad=True)
    banks = []
    hook = model.layers[0].cross_attn.register_forward_pre_hook(
        lambda module, args: banks.append(args[1].shape))
    try:
        result = model(q, trace, empty_history(2), dt.double().cumsum(-1), patches=patches)
    finally:
        hook.remove()
    result.regularization_hidden.retain_grad()
    # Production uses 1024 projections; a smaller sketch suffices to check connectivity.
    loss = lewm_sigreg_loss(result, SIGReg(knots=17, num_proj=32))
    loss.backward()
    assert banks == [torch.Size([2, 1536, 384])]
    for gradient in (result.regularization_hidden.grad, patches.grad,
                     model.layers[0].cross_attn.in_proj_weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert model.norm.weight.grad is None  # SIGReg bypasses the final LayerNorm.


def test_legacy_output_stays_compatible_but_cannot_silently_supply_post_norm_sigreg():
    response = ARMOutput(torch.zeros(2, 6, 6), torch.ones(2, 5, 6), torch.ones(2, 5, 384))
    assert response.regularization_hidden is None
    with pytest.raises(ValueError, match='pre-final-LayerNorm'):
        lewm_sigreg_loss(response, SIGReg())


def test_resume_cannot_hide_changed_training_modes_or_sigreg_target_as_cache_change():
    protocol = dict(training_mode_policy=TRAINING_MODE_POLICY,
                    sigreg_feature_source=SIGREG_FEATURE_SOURCE,
                    schedule=dict(precision='bf16', feature_bank_gib=48), device={})
    validate_training_semantics(protocol, full_patch_lewm=True)
    for field in ('training_mode_policy', 'sigreg_feature_source'):
        previous = deepcopy(protocol)
        previous.pop(field)
        with pytest.raises(ValueError, match='new output directory'):
            validate_training_semantics(previous, full_patch_lewm=True)
        with pytest.raises(ValueError, match='cannot change'):
            validate_resource_continuation(previous, protocol)


def test_scene_norm_ablation_cannot_resume_as_a_resource_change():
    previous = dict(training_mode_policy=TRAINING_MODE_POLICY,
                    sigreg_feature_source=SCENE_SIGREG_SOURCE,
                    scene_lewm_architecture=SCENE_ARCHITECTURE,
                    schedule=dict(scene_latent_lewm=True), device={})
    # The existing BN run can still be resumed without inventing metadata.
    validate_training_semantics(previous, full_patch_lewm=False,
                                scene_latent_lewm=True, scene_normalization='batchnorm')
    with pytest.raises(ValueError, match='normalization.*new output directory'):
        validate_training_semantics(previous, full_patch_lewm=False,
                                    scene_latent_lewm=True, scene_normalization='layernorm')
    current = deepcopy(previous)
    current['scene_normalization'] = 'batchnorm'
    current['schedule']['scene_normalization'] = 'batchnorm'
    assert canonical_protocol(current) == canonical_protocol(previous)
    current['scene_normalization'] = 'layernorm'
    current['schedule']['scene_normalization'] = 'layernorm'
    current['scene_lewm_architecture'] = V21SceneLeWM(normalization='layernorm').architecture
    validate_training_semantics(current, full_patch_lewm=False,
                                scene_latent_lewm=True, scene_normalization='layernorm')
    with pytest.raises(ValueError, match='cannot change'):
        validate_resource_continuation(previous, current)


def test_layernorm_flag_requires_scene_experiment_before_opening_any_files(tmp_path):
    args = ['--parent', str(tmp_path/'missing.pt'), '--source-config', str(tmp_path/'missing.json'),
            '--output', str(tmp_path/'unused'), '--scene-normalization', 'layernorm']
    assert parser().parse_args(args).scene_normalization == 'layernorm'
    with pytest.raises(ValueError, match='scene-normalization.*scene-latent-lewm'):
        main(args)
    assert not (tmp_path/'unused').exists()


def test_complete_command_mode_cannot_silently_resume_a_sampled_command_run():
    from fcpg.v21_command_memory import FULL_COMMANDS
    previous = dict(training_mode_policy=TRAINING_MODE_POLICY, sigreg_feature_source=SIGREG_FEATURE_SOURCE)
    with pytest.raises(ValueError, match='command conditioning.*new output directory'):
        validate_training_semantics(previous, full_patch_lewm=True, command_conditioning=FULL_COMMANDS)
    previous['lewm_command_conditioning'] = FULL_COMMANDS
    validate_training_semantics(previous, full_patch_lewm=True, command_conditioning=FULL_COMMANDS)
