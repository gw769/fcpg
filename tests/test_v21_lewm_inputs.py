"""The LeWM-inputs shortcut (fcpg/v21_lewm_inputs.py) hands the LeWM exactly what factual_state does."""
import dataclasses

import numpy as np
import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_full_patch_lewm import V21FullPatchLeWM
from fcpg.v21_horizon_training import factual_state, horizon_targets
from fcpg.v21_lewm_inputs import lewm_inputs
from test_v21_scratch_features import actual_chain_source


def same(a, b, path='state'):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, path
        assert torch.equal(a, b), path
    elif dataclasses.is_dataclass(a):
        assert type(a) is type(b), path
        for field in dataclasses.fields(a):
            same(getattr(a, field.name), getattr(b, field.name), f'{path}.{field.name}')
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f'{path}[{i}]')
    else:
        assert a == b, path


@pytest.mark.parametrize('full_patch', [False, True])
@pytest.mark.parametrize('h', [0, 1, 2, 3])
def test_lewm_inputs_match_factual_state_and_give_the_same_forecast(h, full_patch):
    torch.manual_seed(21004 + h)
    source = actual_chain_source()
    source.offsets = np.array([0, len(source.raw_q)])
    source.archive.endpoint_episode_index = np.zeros(8, dtype=np.int64)
    core = V21ResearchCore(V21Config(len(source.vocabulary.words), prediction_horizon=40, object_relative=True)).eval()
    if full_patch:
        core.lewm = V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40).eval()
    rows = np.array([0, 1])
    chain = source.chain(rows, 'archive', images=True)
    with torch.no_grad():
        full = factual_state(core, chain, h, images=True)
        fast = lewm_inputs(core, chain, h)
        for name in ('q', 'history', 'memory', 'time'):
            same(getattr(full, name), getattr(fast, name), name)
        targets = horizon_targets(source, rows, 'archive', h)
        outputs = [core.lewm(s.q, targets.trace, s.history, targets.readback_offsets, patches=s.memory.features,
                             patch_valid=s.memory.valid, memory=s.memory, now_time=s.time) for s in (full, fast)]
    assert torch.equal(outputs[0].predicted_q_path, outputs[1].predicted_q_path)
    assert type(core.reader).__name__ == 'ReadbackMemoryReader'          # the real core is untouched
    assert type(core.lewm).__name__ == ('V21FullPatchLeWM' if full_patch else 'V21LeWM')
