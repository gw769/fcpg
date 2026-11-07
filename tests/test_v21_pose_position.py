"""V21Config.pose_position (11/03, James 「AOK阿」): the concatenating FCPG reader's keys also join each patch's position
basis times its picture's joints at capture and their change since (fcpg/v21_memory.pose_position). Without them the
joints at capture enter a key through one linear layer and add the same amount to every patch of a picture: with one
picture in the memory the read does not depend on where the arm was when it was taken at all (the softmax over that
picture's patches is unchanged); the release Actor's commands moved by 0.0000 when they were swapped."""
from dataclasses import replace

import pytest
import torch

from fcpg import v21_compiled
from fcpg.prod_v20_policy import empty_history
from fcpg.v21_compiled import regions
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_encoder import DIM, PATCHES, patch_xy
from fcpg.v21_memory import ADDRESS, POSE_POSITION, ReadbackMemoryReader, ReadLayer, pose_position
from test_v21_compiled import memory, read


@pytest.fixture(autouse=True)
def no_attention_fast_path():
    before = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    v21_compiled.reset()
    yield
    v21_compiled.reset()
    torch.backends.mha.set_fastpath_enabled(before)


@pytest.fixture
def uncompiled(monkeypatch):
    """torch.compile replaced by the identity: the regions' code paths without Inductor."""
    monkeypatch.setattr(torch, 'compile', lambda fn, **kwargs: fn)


def one_picture(seed=0):
    """A memory whose only picture is the newest wrist one, for two batch rows with the same picture."""
    values = memory(2, wrist_only=True, seed=seed)
    valid = torch.zeros_like(values.valid)
    valid[:, 0, -1] = True
    features = values.features.clone()
    features[:, 0, -1] = torch.randn(PATCHES, DIM, generator=torch.Generator().manual_seed(seed + 1))
    features = features * valid[..., None, None]
    q_at = values.q_at_image.clone()
    q_at[:, 0, -1] = 0.
    q_at[1, 0, -1] = torch.tensor([.4, -.3, .2, .1, -.2, 0.])      # row 1: the same picture taken elsewhere
    return replace(values, features=features, valid=valid, q_at_image=q_at)


def reads(reader, values):
    q = torch.zeros(len(values.features), 6)
    now = torch.full((len(q),), 20., dtype=torch.float64)
    return reader(values, q, now, empty_history(len(q)))


@pytest.mark.parametrize('with_products', [False, True])
def test_one_picture_is_read_by_where_it_was_taken_only_with_the_products(with_products):
    torch.manual_seed(3)
    reader = ReadbackMemoryReader(history_address=False, jaw_address=False, pose_queries=True, concat=True,
                                  pose_position=with_products).eval()
    with torch.no_grad():
        for layer in reader.layers:             # weights a trained reader could have: the keys read the address
            layer.key.weight[:, DIM + 2:] *= 30.
        out = reads(reader, one_picture())
    difference = float((out[0, 0] - out[1, 0]).abs().max())
    if with_products:
        assert difference > 1e-3            # the joints at capture move where in the picture it reads
    else:
        assert difference < 1e-6            # one picture: its joints at capture change nothing


def test_the_products_are_the_position_basis_times_the_joints_and_zero_without_a_picture():
    torch.manual_seed(0)
    metadata = torch.randn(4, 3 * PATCHES, ADDRESS + 2)
    metadata[..., ADDRESS:] = patch_xy().repeat(3, 1)
    metadata[1] = 0.
    out = pose_position(metadata)
    assert out.shape == (4, 3 * PATCHES, POSE_POSITION) and not out[1].any()
    xy = metadata[0, 5, ADDRESS:]
    basis = torch.cat((xy, torch.sin(torch.pi * xy), torch.cos(torch.pi * xy)))
    torch.testing.assert_close(out[0, 5], (basis[:, None] * metadata[0, 5, :12][None]).flatten())


def test_the_region_read_with_the_products_is_its_plain_read(uncompiled):
    torch.manual_seed(0)
    reader = ReadbackMemoryReader(history_address=False, jaw_address=False, queries=32, pose_queries=True,
                                  concat=True, pose_position=True).train()
    values = memory(5, wrist_only=False)
    plain = read(reader, values)
    plain.square().sum().backward()
    expected = {name: p.grad.clone() for name, p in reader.named_parameters() if p.grad is not None}
    reader.zero_grad()
    with regions():
        same = read(reader, values)
        same.square().sum().backward()
    assert torch.equal(same, plain)
    assert all(torch.equal(p.grad, expected[name]) for name, p in reader.named_parameters() if p.grad is not None)


def test_the_products_need_the_concatenating_reader_and_change_only_the_keys():
    with pytest.raises(ValueError, match='pose_position'):
        ReadbackMemoryReader(pose_position=True)
    with pytest.raises(ValueError, match='concat_inputs'):
        V21Config(8, pose_position=True)
    with pytest.raises(ValueError, match='concatenating'):
        ReadLayer(False, False, True)
    plain, products = ReadbackMemoryReader(concat=True), ReadbackMemoryReader(concat=True, pose_position=True)
    a, b = plain.state_dict(), products.state_dict()
    assert set(a) == set(b)
    for name in a:
        if name.endswith('key.weight'):
            assert b[name].shape == (192, DIM + 2 + ADDRESS + POSE_POSITION) and a[name].shape == (192, DIM + 2 + ADDRESS)
        else:
            assert a[name].shape == b[name].shape
    core = V21ResearchCore(V21Config(8, object_relative=True, reader_task=True, task_text=False, concat_inputs=True,
                                     actor_history=False, gripper_separate=True, pose_position=True))
    assert core.reader.pose_position and V21Config(8).pose_position is False
