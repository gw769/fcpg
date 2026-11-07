"""V21FullPatchLeWM(pose_position=True) (11/03, James 「AOK阿」): a concatenating LeWM's bank also joins each patch's
position basis times its picture's joints as taken and their change since (POSE_POSITION), so in the keys (one linear
layer) the joints as taken can weigh the patches of a picture, not only add the same amount to all of them."""
from dataclasses import asdict, replace

import pytest
import torch

from fcpg.v21_encoder import DIM, PATCHES
from fcpg.v21_full_patch_lewm import BANK_INPUTS, POSE_POSITION, V21FullPatchLeWM, architecture_from_state
from fcpg.v21_memory import pose_position_basis
from fcpg.v21_native_horizon import core_from_parent
from test_v21_concat_inputs import CONCAT
from test_v21_lewm_picture_memory import N, forecast, memory_for
from v21_lewm_helpers import setup


@pytest.fixture(autouse=True)
def no_attention_fast_path():
    before = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(before)


def test_the_bank_joins_the_products_of_each_patch_and_its_pictures_joints():
    core, state, _, targets, _ = setup(**CONCAT, pose_position=True)
    lewm = core.lewm.eval()
    width = BANK_INPUTS + POSE_POSITION
    assert width % 8 == 0 and lewm.architecture()['pose_position'] is True
    assert all(layer.cross_attn.k_proj_weight.shape == (lewm.width, width) for layer in lewm.layers)
    memory = memory_for(state)
    b = len(state.q)
    with torch.no_grad():
        bank, pad = lewm._memory_bank(memory.features, memory.valid, state.q, memory, state.time)
        moved_q = memory.q_at_image.clone()
        moved_q[:, 0, -1] += .4
        moved = replace(memory, q_at_image=moved_q)
        other, _ = lewm._memory_bank(moved.features, moved.valid, state.q, moved, state.time)
    assert bank.shape == (b, 2 * N * PATCHES, width)
    patches, changed = (x.reshape(b, 2, N, PATCHES, width) for x in (bank, other))
    products = patches[..., BANK_INPUTS:]
    # basis x [joints as taken, their change since], as the 19 numbers carry them (normalized)
    numbers = patches[:, 0, -1, 0, DIM + 64:DIM + 64 + 19]
    pose = torch.cat((numbers[:, :6], numbers[:, 12:18]), -1)
    expected = (pose_position_basis()[None, :, :, None] * pose[:, None, None, :]).flatten(-2)
    torch.testing.assert_close(products[:, 0, -1], expected.to(products), atol=1e-6, rtol=1e-5)
    # moving one picture's joints as taken changes its patches' products by different amounts (not one shift)
    delta = (changed[:, 0, -1, :, BANK_INPUTS:] - products[:, 0, -1]).abs().sum(-1)
    assert delta.std() > 1e-3 and torch.equal(changed[:, 0, :-1], patches[:, 0, :-1])
    assert torch.isfinite(forecast(lewm, state, targets, memory)).all()


def test_every_loader_rebuilds_a_pose_position_lewm_and_it_needs_concat():
    core, state, _, targets, _ = setup(**CONCAT, pose_position=True)
    whole = core.state_dict()
    found = architecture_from_state(whole, 'lewm.')
    assert found['pose_position'] is True and found['concat_inputs'] is True
    rebuilt = core_from_parent(dict(config=asdict(core.config), model=whole)).eval()
    memory = memory_for(state)
    torch.testing.assert_close(forecast(rebuilt.lewm, state, targets, memory),
                               forecast(core.lewm.eval(), state, targets, memory), atol=0, rtol=0)
    plain, _, _, _, _ = setup(**CONCAT)
    assert 'pose_position' not in architecture_from_state(plain.state_dict(), 'lewm.')
    with pytest.raises(ValueError, match='pose_position'):
        V21FullPatchLeWM(horizon=40, controller_input=True, balanced_inputs=True, memory_pictures=3, pose_position=True)
    with pytest.raises(RuntimeError):
        plain.lewm.load_state_dict(core.lewm.state_dict(), strict=True)
