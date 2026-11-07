"""17.161 (11/01, after the 17.160 audit): a balanced LeWM whose command distance passes DreamerV3's symlog
(fcpg/v21_full_patch_lewm.py command_symlog=True): the query input reads sign(x) log(1 + |x|) of (active command -
joints) / increment, a 17.160 balanced LeWM is the one before, every loader finds the marker and rebuilds it, and it
needs balanced inputs. nn.MultiheadAttention's inference fast path is off, as in the trainer and the evaluations."""
from dataclasses import asdict

import pytest
import torch

from fcpg.v21_full_patch_lewm import V21FullPatchLeWM, architecture_from_state
from fcpg.v21_native_horizon import core_from_parent
from v21_lewm_helpers import setup
from test_v21_lewm_controller import call


@pytest.fixture(autouse=True)
def no_attention_fast_path():
    before = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(before)


def query_input(lewm, state, targets, **kwargs):
    kept = []
    handle = lewm.query_in.register_forward_hook(lambda module, inputs, output: kept.append(inputs[0]))
    try:
        with torch.no_grad():
            out = call(lewm, state, targets, **kwargs).predicted_q_path
    finally:
        handle.remove()
    return kept[0], out


def test_the_command_distance_passes_symlog_and_nothing_else_changes():
    core, state, _, targets, _ = setup(balanced_inputs=True)
    plain_input, _ = query_input(core.lewm, state, targets)
    torch.manual_seed(21004)
    lewm = V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40,
                            balanced_inputs=True, command_symlog=True).eval()
    state_plain = core.lewm.state_dict()
    assert set(lewm.state_dict()) - set(state_plain) == {'command_symlog'}
    assert all(torch.equal(v, state_plain[k]) for k, v in lewm.state_dict().items() if k != 'command_symlog')
    got, _ = query_input(lewm, state, targets)
    distance = plain_input[..., 12:18]
    torch.testing.assert_close(got[..., 12:18], torch.sign(distance) * torch.log1p(distance.abs()), atol=0, rtol=0)
    torch.testing.assert_close(got[..., :12], plain_input[..., :12], atol=0, rtol=0)
    torch.testing.assert_close(got[..., 18:], plain_input[..., 18:], atol=0, rtol=0)
    assert lewm.architecture()['command_symlog'] is True
    assert 'command_symlog' not in core.lewm.architecture()


def test_every_loader_rebuilds_a_symlog_lewm_and_it_needs_balanced_inputs():
    core, state, _, targets, _ = setup(balanced_inputs=True, controller_input=True, command_symlog=True)
    with torch.no_grad():
        core.lewm.controller_weight.normal_()
    whole = core.state_dict()
    assert architecture_from_state(whole, 'lewm.') == dict(width=384, layers=4, controller_input=True,
                                                           balanced_inputs=True, command_symlog=True)
    rebuilt = core_from_parent(dict(config=asdict(core.config), model=whole)).eval()
    assert rebuilt.lewm.architecture()['command_symlog'] is True
    setting = torch.tensor([[0., 0.], [1., 1.]])[:len(state.q)]
    _, expected = query_input(core.lewm, state, targets, controller=setting)
    _, got = query_input(rebuilt.lewm, state, targets, controller=setting)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    with pytest.raises(RuntimeError):         # a 17.160 balanced LeWM does not take its weights
        V21FullPatchLeWM(horizon=40, controller_input=True, balanced_inputs=True).load_state_dict(
            core.lewm.state_dict(), strict=True)
    with pytest.raises(ValueError, match='balanced_inputs'):
        V21FullPatchLeWM(horizon=40, command_symlog=True)
    with pytest.raises(ValueError, match='explicit booleans'):
        V21FullPatchLeWM(horizon=40, balanced_inputs=True, command_symlog=1)
