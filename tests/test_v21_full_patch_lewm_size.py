"""17.106: the full-patch LeWM's width and depth are options; every loader rebuilds the size its weights have."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from fcpg.v21_command_memory import FULL_COMMANDS
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_full_patch_lewm import DIM, LAYERS, V21FullPatchLeWM, architecture_from_state
from fcpg.v21_horizon_training import factual_state, horizon_targets
from fcpg.v21_native_horizon import core_from_parent
from test_v21_scratch_features import actual_chain_source


def forecast(**size):
    torch.manual_seed(21004)
    source = actual_chain_source()
    source.offsets = np.array([0, len(source.raw_q)])
    source.archive.endpoint_episode_index = np.zeros(8, dtype=np.int64)
    core = V21ResearchCore(V21Config(len(source.vocabulary.words), prediction_horizon=40, object_relative=True)).eval()
    core.lewm = V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40, **size).eval()
    rows = np.array([0, 1])
    chain = source.chain(rows, 'archive', images=True)
    with torch.no_grad():
        s = factual_state(core, chain, 2, images=True)
        targets = horizon_targets(source, rows, 'archive', 2)
        out = core.lewm(s.q, targets.trace, s.history, targets.readback_offsets, patches=s.memory.features,
                        patch_valid=s.memory.valid, memory=s.memory, now_time=s.time)
    return core, out


def test_the_default_size_is_the_deployed_lewm():
    lewm = V21FullPatchLeWM(horizon=40)
    assert lewm.architecture() == dict(width=DIM, layers=LAYERS, heads=8) == dict(width=384, layers=4, heads=8)
    assert architecture_from_state(lewm.state_dict()) == dict(width=384, layers=4)
    assert architecture_from_state({}) == dict(width=384, layers=4)       # nothing to read: the default


@pytest.mark.parametrize('width, layers', [(64, 2), (768, 4)])
def test_a_resized_lewm_forecasts_and_every_loader_rebuilds_it_from_its_weights(width, layers):
    _, base = forecast()
    core, out = forecast(width=width, layers=layers)
    assert out.predicted_q_path.shape == base.predicted_q_path.shape and out.hidden.shape[-1] == width
    assert torch.isfinite(out.predicted_q_path).all() and torch.isfinite(out.q_uncertainty).all()
    state = core.state_dict()                                             # a whole core, as checkpoints keep it
    assert architecture_from_state(state, 'lewm.') == dict(width=width, layers=layers)
    rebuilt = core_from_parent(dict(config=asdict(core.config), model=state))    # fcpg/v21_native_horizon.py
    assert rebuilt.lewm.architecture() == dict(width=width, layers=layers, heads=8)
    assert all(torch.equal(a, b) for a, b in zip(rebuilt.lewm.state_dict().values(), core.lewm.state_dict().values()))
    with pytest.raises(RuntimeError):                                     # the default size cannot take them
        V21FullPatchLeWM(horizon=40).load_state_dict(core.lewm.state_dict(), strict=True)


def test_full_command_tokens_follow_the_width_and_depth():
    lewm = V21FullPatchLeWM(horizon=40, width=64, layers=2, command_conditioning=FULL_COMMANDS)
    assert len(lewm.command_readers) == 2 and lewm.command_projection[-1].out_features == 64


def test_resized_lewms_attend_directly_and_it_is_the_modules_function():
    """17.106: a resized LeWM's layers use _attend() on nn.MultiheadAttention's weights; with the modules' own path
    switched back on, the same weights forecast the same values (picture bank and q-only, whose empty bank is one
    readable token in the direct path); the deployed default size never attends directly."""
    assert not any(layer.direct for layer in V21FullPatchLeWM(horizon=40).layers)
    core, out = forecast(width=64, layers=2)
    assert all(layer.direct for layer in core.lewm.layers)
    source = actual_chain_source()
    source.offsets = np.array([0, len(source.raw_q)])
    source.archive.endpoint_episode_index = np.zeros(8, dtype=np.int64)
    rows = np.array([0, 1])
    with torch.no_grad():
        s = factual_state(core, source.chain(rows, 'archive', images=True), 2, images=True)
        targets = horizon_targets(source, rows, 'archive', 2)
        answers = {}
        for direct in (True, False):
            for layer in core.lewm.layers:
                layer.direct = direct
            answers[direct] = (
                core.lewm(s.q, targets.trace, s.history, targets.readback_offsets, patches=s.memory.features,
                          patch_valid=s.memory.valid, memory=s.memory, now_time=s.time),
                core.lewm(s.q, targets.trace, s.history, targets.readback_offsets))
    for a, b in zip(answers[True], answers[False]):
        torch.testing.assert_close(a.predicted_q_path, b.predicted_q_path, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a.q_uncertainty, b.q_uncertainty, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a.hidden, b.hidden, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(answers[True][0].predicted_q_path, out.predicted_q_path)


@pytest.mark.parametrize('width, layers', [(100, 4), (0, 4), (True, 4), (384, 0), (384, 2.0)])
def test_sizes_that_cannot_be_built_are_refused(width, layers):
    with pytest.raises(ValueError):
        V21FullPatchLeWM(horizon=40, width=width, layers=layers)
