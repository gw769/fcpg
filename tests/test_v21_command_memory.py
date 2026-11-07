"""Command/observation contracts, no data files or hardware required."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from fcpg.prod_v20_policy import empty_history
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_full_patch_lewm import V21FullPatchLeWM
from fcpg.v21_models import CommandTrace
from fcpg.v21_native_horizon import core_from_parent
from fcpg.v21_scene_lewm import V21SceneLeWM

FULL = 'causal-command-tokens-v1'


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(21004)


def case():
    q = torch.tensor([[0., -50., 50., 50., 0., 30.]])
    offsets = torch.arange(1, 41, dtype=torch.float64)[None] / 10
    commands = q[:, None].expand(-1, 120, -1).clone()
    trace = CommandTrace(commands, torch.arange(120, dtype=torch.float64)[None] / 30,
                         torch.ones(1, 120, dtype=torch.bool), 'recorded', True)
    patches = torch.randn(1, 2, 3, 256, 384)
    return q, trace, empty_history(1), offsets, patches


def opened_model():
    model = V21SceneLeWM(normalization='layernorm', command_conditioning=FULL).eval()
    # A trained branch has nonzero gates. Initialization separately preserves
    # the old predictor exactly; these fixtures exercise the resulting path.
    with torch.no_grad():
        for layer in model.command_readers:
            layer.gate.fill_(.2)
    return model


def test_all_missing_visual_values_cannot_change_prediction():
    q, tr, hist, offsets, patches = case()
    model = V21SceneLeWM(normalization='layernorm').eval()
    mask = torch.zeros(1, 2, 3, dtype=torch.bool)
    with torch.no_grad():
        a = model(q, tr, hist, offsets, patches=patches, patch_valid=mask)
        b = model(q, tr, hist, offsets, patches=patches*3+4, patch_valid=mask)
    torch.testing.assert_close(a.predicted_q_path, b.predicted_q_path, atol=0, rtol=0)
    torch.testing.assert_close(a.predicted_scene, b.predicted_scene, atol=0, rtol=0)


def test_image_age_subtracts_float64_clocks_before_casting():
    q, _, _, _, patches = case()
    model = V21FullPatchLeWM().eval()
    captured = []
    handle = model.film.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    for origin in (0., 1_000_000., 1_700_000_000.):
        now = torch.tensor([origin+1.], dtype=torch.float64)
        memory = SimpleNamespace(q_at_image=q[:, None, None].expand(-1, 2, 3, -1),
                                 image_times=torch.full((1, 2, 3), origin+.9, dtype=torch.float64))
        with torch.no_grad():
            model._visual_bank(patches, torch.ones(1, 2, 3, dtype=torch.bool), q, memory, now)
        expected = torch.log1p(now[:, None, None]-memory.image_times).float()
        torch.testing.assert_close(captured[-1][..., -1], expected, atol=1e-7, rtol=1e-6)
    handle.remove()


def test_full_command_branch_warm_starts_without_changing_existing_weights_or_outputs():
    q, tr, hist, offsets, patches = case()
    torch.manual_seed(9)
    old = V21SceneLeWM(normalization='layernorm').eval()
    torch.manual_seed(9)
    new = V21SceneLeWM(normalization='layernorm', command_conditioning=FULL).eval()
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    with torch.no_grad():
        a = old(q, tr, hist, offsets, patches=patches)
        b = new(q, tr, hist, offsets, patches=patches)
    torch.testing.assert_close(a.predicted_q_path, b.predicted_q_path, atol=0, rtol=0)
    torch.testing.assert_close(a.predicted_scene, b.predicted_scene, atol=0, rtol=0)


def test_intermediate_commands_and_their_times_are_not_discarded():
    q, tr, hist, offsets, patches = case()
    model = opened_model()
    values = tr.positions.clone()
    values[:, 0::3, 0] += 1
    values[:, 1::3, 0] -= 1
    changed = replace(tr, positions=values)
    assert torch.equal(tr.active_at(offsets, q), changed.active_at(offsets, q))
    with torch.no_grad():
        a = model(q, tr, hist, offsets, patches=patches)
        b = model(q, changed, hist, offsets, patches=patches)
    assert (a.predicted_q_path-b.predicted_q_path).abs().max() > 1e-6
    assert (a.predicted_scene-b.predicted_scene).abs().max() > 1e-6
    shifted = changed.offsets.clone()
    shifted[:, 1::3] += .01  # same active-at-query commands, different durations
    with torch.no_grad():
        c = model(q, replace(changed, offsets=shifted), hist, offsets, patches=patches)
    assert not torch.equal(b.predicted_scene, c.predicted_scene)


def test_future_and_exactly_tied_writes_cannot_change_earlier_predictions():
    q, tr, hist, offsets, patches = case()
    model = opened_model()
    values = tr.positions.clone()
    values[:, 15:, 0] += 20  # at 0.5 s or later; q(0.5) precedes the tied write
    with torch.no_grad():
        a = model(q, tr, hist, offsets, patches=patches)
        b = model(q, replace(tr, positions=values), hist, offsets, patches=patches)
    torch.testing.assert_close(a.predicted_scene[:, :5], b.predicted_scene[:, :5], atol=0, rtol=0)
    torch.testing.assert_close(a.predicted_q_path[:, :6], b.predicted_q_path[:, :6], atol=0, rtol=0)
    assert not torch.equal(a.predicted_scene[:, 5:], b.predicted_scene[:, 5:])


def test_padding_and_no_command_yet_queries_remain_finite_and_causal():
    q, tr, hist, offsets, patches = case()
    model = opened_model()
    padded = replace(tr, positions=torch.cat((tr.positions, torch.full((1, 5, 6), float('nan'))), 1),
                     offsets=torch.cat((tr.offsets, torch.full((1, 5), float('nan'), dtype=torch.float64)), 1),
                     valid=torch.cat((tr.valid, torch.zeros(1, 5, dtype=torch.bool)), 1))
    with torch.no_grad():
        a = model(q, tr, hist, offsets, patches=patches)
        b = model(q, padded, hist, offsets, patches=patches)
        delayed = replace(tr, positions=tr.positions[:, :2], offsets=torch.tensor([[.4, .5]], dtype=torch.float64),
                          valid=tr.valid[:, :2])
        c = model(q, delayed, hist, offsets, patches=patches)
    torch.testing.assert_close(a.predicted_scene, b.predicted_scene, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(c.predicted_scene).all()


def test_new_command_weights_receive_gradients_after_gate_initialization():
    q, tr, hist, offsets, patches = case()
    model = V21SceneLeWM(normalization='layernorm', command_conditioning=FULL).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = model(q, tr, hist, offsets, patches=patches)
        (out.predicted_scene.square().mean()+out.predicted_q_path.square().mean()).backward()
        assert model.command_readers[0].gate.grad.abs().sum() > 0
        if step:
            grad = model.command_projection[0].weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        optimizer.step()


def test_command_checkpoint_metadata_is_strict_and_old_models_still_load():
    core = V21ResearchCore(V21Config(4, prediction_horizon=40))
    core.lewm = V21SceneLeWM(normalization='layernorm', command_conditioning=FULL)
    payload = dict(config=asdict(core.config), model=core.state_dict(),
                   scene_lewm_architecture=core.lewm.architecture, lewm_command_conditioning=FULL)
    restored = core_from_parent(payload)
    assert restored.lewm.command_conditioning == FULL
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in core.state_dict().items())
    for invalid in (None, 'sampled-active-v1', 'unknown'):
        bad = dict(payload)
        if invalid is None:
            bad.pop('lewm_command_conditioning')
        else:
            bad['lewm_command_conditioning'] = invalid
        with pytest.raises(ValueError, match='command'):
            core_from_parent(bad)
