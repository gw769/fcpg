"""17.160 (11/01, after the 17.159 audit): a LeWM whose inputs are on the scale of each other (fcpg/v21_full_patch_lewm.py
balanced_inputs=True): the picture bank passes a LayerNorm in every layer before the picture keys and values, the query
input also reads (active command - joints) / increment, and the times are read as t / TIME_SCALE and dt / STEP_SCALE.
A LeWM without it is the one before; the query input is what it says; the picture attention no longer depends on the
bank's scale; both attention paths agree; it trains; every loader rebuilds it.
nn.MultiheadAttention's inference fast path is off, as in the trainer and the evaluations."""
from dataclasses import asdict

import pytest
import torch

from fcpg.v21_full_patch_lewm import (BALANCED_QUERY_INPUTS, QUERY_INPUTS, STEP_SCALE, TIME_SCALE, V21FullPatchLeWM,
                                      _QueryLayer, architecture_from_state)
from fcpg.v21_native_horizon import core_from_parent
from v21_lewm_helpers import setup
from test_v21_lewm_controller import call


@pytest.fixture(autouse=True)
def no_attention_fast_path():
    before = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(before)


def forecast(lewm, state, targets, **kwargs):
    with torch.no_grad():
        return call(lewm, state, targets, **kwargs).predicted_q_path


def test_a_lewm_without_it_is_the_one_before():
    core, state, _, targets, _ = setup()
    plain = core.lewm
    assert plain.query_in.in_features == QUERY_INPUTS and not any(hasattr(layer, 'norm_m') for layer in plain.layers)
    assert 'balanced_inputs' not in plain.architecture()
    assert 'balanced_inputs' not in architecture_from_state(plain.state_dict())
    torch.manual_seed(21004)
    explicit = V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40,
                                balanced_inputs=False).eval()
    state_plain = plain.state_dict()
    assert explicit.state_dict().keys() == state_plain.keys()
    assert all(torch.equal(v, state_plain[k]) for k, v in explicit.state_dict().items())
    assert torch.equal(forecast(explicit, state, targets), forecast(plain, state, targets))
    with pytest.raises(ValueError, match='explicit booleans'):
        V21FullPatchLeWM(horizon=40, balanced_inputs=1)


def test_the_query_input_reads_the_command_distance_and_times_on_a_fixed_scale():
    core, state, _, targets, _ = setup(balanced_inputs=True)
    lewm = core.lewm
    assert lewm.query_in.in_features == BALANCED_QUERY_INPUTS and all(hasattr(layer, 'norm_m') for layer in lewm.layers)
    assert lewm.architecture()['balanced_inputs'] is True
    kept = []
    handle = lewm.query_in.register_forward_hook(lambda module, inputs, output: kept.append(inputs[0]))
    try:
        forecast(lewm, state, targets)
    finally:
        handle.remove()
    q = state.q.float()
    qt = targets.readback_offsets.float()
    active = targets.trace.active_at(targets.readback_offsets, state.q).float()
    h = qt.shape[1]
    qdt = torch.diff(torch.cat((qt.new_zeros(len(q), 1), qt), -1), dim=1)
    expected = torch.cat((lewm.normalize_position(q)[:, None].expand(-1, h, -1), lewm.normalize_position(active),
                          (active - q[:, None]) / lewm.increment_scale, (qt / TIME_SCALE)[..., None],
                          (qdt / STEP_SCALE)[..., None]), -1)
    torch.testing.assert_close(kept[0], expected, atol=0, rtol=0)
    assert kept[0][..., 18].max() <= 1.0 + 1e-6 and kept[0][..., 18].min() > 0         # 40 read-backs up to 4 s


def layer_pair(memory_norm):
    torch.manual_seed(7)
    module = _QueryLayer(64, 8, direct=False, memory_norm=memory_norm).eval()
    direct = _QueryLayer(64, 8, direct=True, memory_norm=memory_norm).eval()
    direct.load_state_dict(module.state_dict(), strict=True)
    return module, direct


def test_the_picture_attention_no_longer_depends_on_the_banks_scale_on_either_path():
    torch.manual_seed(3)
    query, memory = torch.randn(2, 5, 64), torch.randn(2, 30, 64)
    pad = torch.zeros(2, 30, dtype=torch.bool)
    pad[1, 20:] = True
    causal = torch.ones(5, 5, dtype=torch.bool).triu(1)
    module, direct = layer_pair(True)
    with torch.no_grad():
        module.norm_m.weight.uniform_(.5, 1.5)
        module.norm_m.bias.normal_(0, .1)
        direct.load_state_dict(module.state_dict(), strict=True)
        out = module(query, memory, causal, pad)
        torch.testing.assert_close(direct(query, memory, causal, pad), out, atol=2e-6, rtol=1e-5)
        for scale in (.28, 3.5):        # the 17.159 bank's ~0.28 per element, and larger (LayerNorm's eps aside)
            torch.testing.assert_close(module(query, memory * scale, causal, pad), out, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(direct(query, memory * scale, causal, pad), out, atol=1e-4, rtol=1e-4)
        plain, _ = layer_pair(False)
        assert not torch.allclose(plain(query, memory * .28, causal, pad), plain(query, memory, causal, pad), atol=1e-3)


def test_both_attention_paths_give_a_balanced_lewm_the_same_forecasts():
    core, state, _, targets, _ = setup(balanced_inputs=True)
    with torch.no_grad():
        for layer in core.lewm.layers:
            layer.norm_m.weight.uniform_(.5, 1.5)
    expected = forecast(core.lewm, state, targets)
    for layer in core.lewm.layers:
        layer.direct = True
    torch.testing.assert_close(forecast(core.lewm, state, targets), expected, atol=1e-5, rtol=1e-5)


def test_a_balanced_lewm_trains_its_new_parts():
    core, state, _, targets, _ = setup(balanced_inputs=True, controller_input=True)
    lewm = core.lewm.train()
    call(lewm, state, targets, controller=torch.zeros(len(state.q), 2)).predicted_q_path.square().sum().backward()
    assert all(layer.norm_m.weight.grad is not None and layer.norm_m.weight.grad.abs().sum() > 0
               for layer in lewm.layers)
    assert lewm.query_in.weight.grad[:, 12:20].abs().sum() > 0          # the command distance and the scaled times


def test_every_loader_rebuilds_a_balanced_lewm():
    core, state, _, targets, _ = setup(balanced_inputs=True, controller_input=True)
    with torch.no_grad():
        core.lewm.controller_weight.normal_()
        for layer in core.lewm.layers:
            layer.norm_m.weight.uniform_(.5, 1.5)
            layer.norm_m.bias.normal_(0, .1)
    whole = core.state_dict()
    assert architecture_from_state(whole, 'lewm.') == dict(width=384, layers=4, controller_input=True,
                                                           balanced_inputs=True)
    rebuilt = core_from_parent(dict(config=asdict(core.config), model=whole)).eval()
    assert rebuilt.lewm.architecture()['balanced_inputs'] is True
    setting = torch.tensor([[0., 0.], [1., 1.]])[:len(state.q)]
    torch.testing.assert_close(forecast(rebuilt.lewm, state, targets, controller=setting),
                               forecast(core.lewm, state, targets, controller=setting), atol=0, rtol=0)
    with pytest.raises(RuntimeError):
        V21FullPatchLeWM(horizon=40, controller_input=True).load_state_dict(core.lewm.state_dict(), strict=True)
