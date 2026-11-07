"""17.164 (11/01; James 「dino是凍結的」「你不要陷入誤區」「推理不用sam」): the full-patch LeWM's own picture encoder
(fcpg/v21_full_patch_lewm.py patch_encoder_layers=N): N pre-LN self-attention layers over every readable token of the
picture bank before the query layers read it. A new encoder changes no forecast (its outputs start at zero) and takes
nothing from the RNG stream the rest of the LeWM and the trainer see; a trained one mixes the patches of both cameras
and all three pictures, never reads a picture that was not taken, leaves a row without pictures as it was, trains, and
every loader rebuilds it. The recipe is a-control's (17.162: control setting, balanced inputs, symlog).
nn.MultiheadAttention's inference fast path is off, as in the trainer and the evaluations."""
from dataclasses import asdict

import pytest
import torch

from fcpg.v21_encoder import DIM as FEATURE_DIM, PATCHES
from fcpg.v21_full_patch_lewm import V21FullPatchLeWM, architecture_from_state
from fcpg.v21_native_horizon import core_from_parent
from v21_lewm_helpers import setup
from test_v21_lewm_controller import call

RECIPE = dict(controller_input=True, balanced_inputs=True, command_symlog=True)


@pytest.fixture(autouse=True)
def no_attention_fast_path():
    before = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(before)


def extra(state):
    return dict(controller=torch.tensor([[0., 0.], [1., 1.]])[:len(state.q)])


def forecast(lewm, state, targets, **kwargs):
    with torch.no_grad():
        return call(lewm, state, targets, **kwargs).predicted_q_path


def as_trained(lewm, seed=5):
    """Every encoder weight drawn, the zero-started outputs included (a trained encoder is not the identity)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, weight in lewm.patch_encoder.named_parameters():
            if 'norm' not in name:
                weight.copy_(torch.randn(weight.shape, generator=generator) * .05)
    return lewm


def test_a_new_encoder_changes_nothing_and_takes_nothing_from_the_rng():
    core, state, _, targets, _ = setup(**RECIPE)
    after_plain = torch.rand(4)
    plain = core.lewm
    coded_core, _, _, _, _ = setup(**RECIPE, patch_encoder_layers=2)
    after_coded = torch.rand(4)
    coded = coded_core.lewm
    assert torch.equal(after_plain, after_coded)                    # the same stream after the constructor
    assert 'patch_encoder_layers' not in plain.architecture()
    assert coded.architecture()['patch_encoder_layers'] == 2 and len(coded.patch_encoder) == 2
    old, new = plain.state_dict(), coded.state_dict()
    assert set(new) - set(old) == {k for k in new if k.startswith('patch_encoder.')} and set(old) <= set(new)
    assert all(torch.equal(new[k], v) for k, v in old.items())     # every other weight is the same
    for layer in coded.patch_encoder:
        assert not any(w.any() for w in (layer.attn.out_proj.weight, layer.attn.out_proj.bias,
                                         layer.ff[2].weight, layer.ff[2].bias))
    kwargs = extra(state)
    assert torch.equal(forecast(coded, state, targets, **kwargs), forecast(plain, state, targets, **kwargs))


def pictures(b, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(b, 2, 3, PATCHES, FEATURE_DIM, generator=generator)


def test_a_trained_encoder_mixes_cameras_and_pictures_and_never_reads_a_missing_one():
    core, state, _, _, _ = setup(**RECIPE, patch_encoder_layers=2)
    lewm = as_trained(core.lewm).eval()
    q = state.q[:1]
    patches = pictures(1)
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    with torch.no_grad():
        bank, pad = lewm._visual_bank(patches, valid, q, None, None)
        base = lewm._encode_patches(bank, pad).unflatten(1, (2, 3, PATCHES))
        moved = patches.clone()
        # the other camera's oldest picture, changed in its content (a constant shift would vanish in the LayerNorm)
        moved[:, 1, 0] += torch.randn(moved[:, 1, 0].shape, generator=torch.Generator().manual_seed(4))
        bank2, pad2 = lewm._visual_bank(moved, valid, q, None, None)
        mixed = lewm._encode_patches(bank2, pad2).unflatten(1, (2, 3, PATCHES))
        assert not torch.allclose(mixed[:, 0, 2], base[:, 0, 2], atol=1e-4)      # read by the first camera's newest
        missing = valid.clone()
        missing[:, 1, 0] = False                                   # that picture not taken
        bank3, pad3 = lewm._visual_bank(patches, missing, q, None, None)
        bank4, pad4 = lewm._visual_bank(moved, missing, q, None, None)
        assert torch.equal(pad3, pad4)
        out3, out4 = lewm._encode_patches(bank3, pad3), lewm._encode_patches(bank4, pad4)
        torch.testing.assert_close(out3[~pad3], out4[~pad4], atol=0, rtol=0)


def test_a_row_without_pictures_keeps_its_one_zero_token_and_its_forecast():
    core, state, _, targets, _ = setup(**RECIPE)
    plain = core.lewm
    coded_core, _, _, _, _ = setup(**RECIPE, patch_encoder_layers=2)
    coded = as_trained(coded_core.lewm).eval()
    b = len(state.q)
    valid = torch.ones(b, 2, 3, dtype=torch.bool)
    valid[0] = False
    patches = pictures(b)
    with torch.no_grad():
        bank, pad = coded._visual_bank(patches, valid, state.q, None, None)
        encoded = coded._encode_patches(bank, pad)
    assert not encoded[0].any() and int((~pad[0]).sum()) == 1
    kwargs = dict(extra(state), patches=patches, patch_valid=valid, memory=state.memory, now_time=state.time)
    with torch.no_grad():
        with_encoder = coded(state.q, targets.trace, state.history, targets.readback_offsets, **kwargs)
        without = plain(state.q, targets.trace, state.history, targets.readback_offsets, **kwargs)
    torch.testing.assert_close(with_encoder.predicted_q_path[0], without.predicted_q_path[0], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(with_encoder.predicted_q_path[1:], without.predicted_q_path[1:], atol=1e-5)


def test_the_encoder_trains_from_its_first_update():
    core, state, _, targets, _ = setup(**RECIPE, patch_encoder_layers=2)
    lewm = core.lewm.train()
    call(lewm, state, targets, **extra(state)).predicted_q_path.square().sum().backward()
    for layer in lewm.patch_encoder:                               # the zero-started outputs move first
        assert layer.attn.out_proj.weight.grad.abs().sum() > 0 and layer.ff[2].weight.grad.abs().sum() > 0
    lewm.zero_grad()
    as_trained(lewm)
    call(lewm, state, targets, **extra(state)).predicted_q_path.square().sum().backward()
    for layer in lewm.patch_encoder:                               # then everything inside
        assert layer.attn.in_proj_weight.grad.abs().sum() > 0 and layer.ff[0].weight.grad.abs().sum() > 0


def test_every_loader_rebuilds_the_encoder():
    core, state, _, targets, _ = setup(**RECIPE, patch_encoder_layers=2)
    as_trained(core.lewm)
    whole = core.state_dict()
    assert architecture_from_state(whole, 'lewm.') == dict(width=384, layers=4, controller_input=True,
                                                           balanced_inputs=True, command_symlog=True,
                                                           patch_encoder_layers=2)
    rebuilt = core_from_parent(dict(config=asdict(core.config), model=whole)).eval()
    assert rebuilt.lewm.architecture()['patch_encoder_layers'] == 2
    kwargs = extra(state)
    torch.testing.assert_close(forecast(rebuilt.lewm, state, targets, **kwargs),
                               forecast(core.lewm.eval(), state, targets, **kwargs), atol=0, rtol=0)
    with pytest.raises(RuntimeError):
        V21FullPatchLeWM(horizon=40, **RECIPE).load_state_dict(core.lewm.state_dict(), strict=True)


@pytest.mark.parametrize('layers', [True, -1, 9, 2., '2'])
def test_the_encoder_size_is_a_whole_number_from_0_to_8(layers):
    with pytest.raises(ValueError, match='whole number'):
        V21FullPatchLeWM(horizon=40, patch_encoder_layers=layers)
