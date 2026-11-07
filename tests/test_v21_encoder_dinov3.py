"""The v21 core under V21_ENCODER=dinov3_vitl16 (14 x 14 patches of 1024): state machine, the LeWM-inputs shortcut,
both LeWMs and the Actor run forward and backward; a DINOv2 model is refused there. The encoder is fixed per process
(fcpg/v21_encoder.py), so the DINOv3-L half runs in a child process."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

REPO = Path(__file__).resolve().parents[1]

CHILD = textwrap.dedent('''
    import torch
    from fcpg.v21_encoder import DIM, ENCODER, GRID, PATCHES
    assert (ENCODER, GRID, PATCHES, DIM) == ('dinov3_vitl16', 14, 196, 1024)
    from fcpg.v21_core import V21Config, V21ResearchCore
    from fcpg.v21_full_patch_lewm import V21FullPatchLeWM
    from fcpg.v21_horizon_training import factual_state
    from fcpg.v21_lewm_inputs import lewm_inputs
    from fcpg.v21_mirror import flipped_features
    from fcpg.v21_training import synthetic_chain
    from fcpg.v21_data import TaskVocabulary
    torch.manual_seed(0)
    vocabulary = TaskVocabulary.build(['pick red block'])
    try:
        V21ResearchCore(V21Config(len(vocabulary.words), prediction_horizon=40))
        raise SystemExit('a DINOv2 model was built in a DINOv3-L process')
    except ValueError as error:
        assert 'dinov2_vits14' in str(error)
    core = V21ResearchCore(V21Config(len(vocabulary.words), prediction_horizon=40, object_relative=True,
                                     encoder='dinov3_vitl16')).eval()
    assert core.reader.feature_projection.in_features == 1024 and tuple(core.reader.patch_xy.shape) == (196, 2)
    assert core.lewm.feature_in.in_features == 1024
    chain = synthetic_chain(vocabulary, batch=2)
    assert tuple(chain.features.shape[2:]) == (2, 196, 1024)
    flipped = flipped_features(chain.features, torch.tensor([True, False]))
    assert torch.equal(flipped[0, :, :, :14], chain.features[0, :, :, :14].flip(2)) and torch.equal(flipped[1], chain.features[1])
    for h in range(4):
        with torch.no_grad():
            full = factual_state(core, chain, h, images=True)
            fast = lewm_inputs(core, chain, h)
        assert torch.equal(full.q, fast.q) and torch.equal(full.memory.features, fast.memory.features)
        assert tuple(full.memory.features.shape[1:]) == (2, 3, 196, 1024)
    state = full
    trace, offsets = chain.traces[0], chain.readback_offsets[:, 0, 1:]
    for lewm in (core.lewm, V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40)):
        lewm.train()
        response = lewm(state.q, trace, state.history, offsets, patches=state.memory.features,
                        patch_valid=state.memory.valid, memory=state.memory, now_time=state.time)
        response.predicted_q_path.square().mean().backward()
        assert lewm.feature_in.weight.grad is not None and torch.isfinite(lewm.feature_in.weight.grad).all()
    offsets40 = (torch.arange(40, dtype=torch.float64) * .1)[None].expand(2, -1).contiguous()
    action = core.act(state, chain.text_ids, offsets40).commands_position
    assert tuple(action.shape) == (2, 40, 6) and torch.isfinite(action).all()
    print('DINOV3-L CORE OK')
''')


def test_the_core_runs_on_dinov3_l_features_and_refuses_a_dinov2_model():
    env = dict(os.environ, V21_ENCODER='dinov3_vitl16', CUDA_VISIBLE_DEVICES='',
               PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get('PYTHONPATH', '')]))
    done = subprocess.run([sys.executable, '-c', CHILD], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0 and 'DINOV3-L CORE OK' in done.stdout, done.stdout[-2000:] + done.stderr[-4000:]


def test_the_default_process_keeps_dinov2_s_sizes():
    from fcpg.v21_encoder import DIM, ENCODER, GRID, PATCHES
    if os.environ.get('V21_ENCODER', 'dinov2_vits14') != 'dinov2_vits14':
        pytest.skip('this pytest process runs another encoder')
    assert (ENCODER, GRID, PATCHES, DIM) == ('dinov2_vits14', 16, 256, 384)


def test_the_dinov2_extractor_version_is_the_one_every_dinov2_cache_declares():
    from fcpg.v21_encoder import EXTRACTOR_VERSIONS
    from fcpg.v21_recorded_clock import EXTRACTOR_VERSION
    assert EXTRACTOR_VERSIONS['dinov2_vits14'] == EXTRACTOR_VERSION
    assert EXTRACTOR_VERSIONS['dinov3_vitl16'] != EXTRACTOR_VERSION
