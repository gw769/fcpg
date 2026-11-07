"""Actor and world model as one model (fcpg/v21_world_head.py; training overhaul doc section 17.113): the world head
starts at "nothing changes", its loss is measured against the change itself, the fixed basis is fitted from the
patches, its gradient reaches the reader tokens it reads, and adding it changes no other module's start."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_encoder import DIM, ENCODER, PATCHES
from fcpg.v21_world_head import WorldHead, change_loss


def head(components=16):
    torch.manual_seed(3)
    return WorldHead(components=components, width=64, layers=1, heads=4)


def test_the_head_starts_at_nothing_changes():
    world = head()
    tokens, q = torch.randn(3, 8, 192), torch.randn(3, 6) * 10
    commands = q[:, None] + torch.randn(3, 5, 6)
    change, joints, hidden = world(tokens, q, commands)
    assert change.shape == (3, PATCHES, 16) and torch.equal(change, torch.zeros_like(change))
    assert joints.shape == (3, 5, 6) and torch.allclose(joints, q[:, None].expand(-1, 5, -1))
    assert hidden.shape == (3, PATCHES + 1, 64)


def test_the_change_loss_is_measured_against_the_change_itself():
    target = torch.randn(4, PATCHES, 16)
    valid = torch.tensor([True, True, False, True])
    loss, explained = change_loss(target.clone(), target, valid)
    assert float(loss) == pytest.approx(0.) and float(explained) == pytest.approx(1.)
    loss, explained = change_loss(torch.zeros_like(target), target, valid)
    assert float(loss) == pytest.approx(1.) and float(explained) == pytest.approx(0.)
    wrong = target.clone()
    wrong[2] += 100.                                      # an invalid row does not count
    assert float(change_loss(wrong, target, valid)[0]) == pytest.approx(0.)
    predicted = torch.zeros_like(target, requires_grad=True)
    loss, explained = change_loss(predicted, target, torch.zeros(4, dtype=torch.bool))
    loss.backward()
    assert float(loss) == 0. and np.isnan(float(explained)) and predicted.grad is not None


def test_the_basis_holds_the_main_directions_of_the_patches():
    world = head(components=16)
    generator = torch.Generator().manual_seed(0)
    directions = torch.linalg.qr(torch.randn(DIM, 3, generator=generator))[0]
    patches = (torch.randn(5000, 3, generator=generator) * torch.tensor([9., 6., 4.])) @ directions.T
    patches = patches + .01 * torch.randn(5000, DIM, generator=generator) + 2.
    kept = world.fit_basis(patches)
    assert kept > .99
    assert torch.allclose(world.basis_mean, patches.mean(0))
    span = world.basis[:, :3]
    assert torch.allclose((directions.T @ span).abs().max(1).values, torch.ones(3), atol=1e-3)
    projected = world.project(patches[:2, None].expand(-1, PATCHES, -1))
    assert projected.shape == (2, PATCHES, 16)


def test_the_world_loss_reaches_the_reader_tokens():
    world = head()
    tokens = torch.randn(2, 8, 192, requires_grad=True)
    q = torch.randn(2, 6)
    with torch.no_grad():                                 # past the zero start, as after a few updates
        world.change.weight.normal_(std=.1)
    change, joints, _ = world(tokens, q, q[:, None] + torch.randn(2, 5, 6))
    loss, _ = change_loss(change, torch.randn_like(change), torch.ones(2, dtype=torch.bool))
    loss.backward()
    assert tokens.grad is not None and float(tokens.grad.abs().sum()) > 0


def test_adding_the_world_head_changes_no_other_module():
    vocabulary = TaskVocabulary.build(['pick up the red fish toy and place it in the cardboard box'])
    common = dict(object_relative=True, prediction_horizon=40, actor_history=False, reader_task=True,
                  gripper_separate=True, wrist_only=True, encoder=ENCODER)
    torch.manual_seed(11)
    plain = V21ResearchCore(V21Config(len(vocabulary.words), **common))
    torch.manual_seed(11)
    joint = V21ResearchCore(V21Config(len(vocabulary.words), world_head=True, **common))
    mine = joint.state_dict()
    assert any(key.startswith('world.') for key in mine)
    assert all(torch.equal(value, mine[key]) for key, value in plain.state_dict().items())
    assert not hasattr(plain, 'world')
    with pytest.raises(ValueError, match='world_components'):
        V21Config(len(vocabulary.words), world_head=True, world_components=48, **common)


def test_the_token_head_starts_at_the_tokens_now_and_learns_the_next():
    torch.manual_seed(5)
    world = WorldHead(width=64, layers=1, heads=4, target='tokens', queries=8)
    tokens, q = torch.randn(3, 8, 192), torch.randn(3, 6) * 10
    commands = q[:, None] + torch.randn(3, 5, 6)
    predicted, joints, hidden = world(tokens, q, commands)
    assert torch.equal(predicted, tokens) and torch.allclose(joints, q[:, None].expand(-1, 5, -1))
    assert hidden.shape == (3, 9, 64) and not hasattr(world, 'change') and not hasattr(world, 'patch_query')
    nxt = tokens + torch.randn_like(tokens)
    loss, explained = change_loss(predicted, nxt, torch.ones(3, dtype=torch.bool), current=tokens)
    assert float(loss) == pytest.approx(1.) and float(explained) == pytest.approx(0.)   # "nothing changes"
    loss, _ = change_loss(nxt.clone(), nxt, torch.ones(3, dtype=torch.bool), current=tokens)
    assert float(loss) == pytest.approx(0.)
    live = tokens.clone().requires_grad_(True)
    with torch.no_grad():
        world.token_out.weight.normal_(std=.1)
    predicted, _, _ = world(live, q, commands)
    change_loss(predicted, nxt, torch.ones(3, dtype=torch.bool), current=live)[0].backward()
    assert live.grad is not None and float(live.grad.abs().sum()) > 0
    with pytest.raises(ValueError, match='world target'):
        WorldHead(target='pixels')


def test_the_token_world_head_changes_no_other_module_either():
    vocabulary = TaskVocabulary.build(['pick up the red fish toy and place it in the cardboard box'])
    common = dict(object_relative=True, prediction_horizon=40, actor_history=False, reader_task=True,
                  gripper_separate=True, wrist_only=True, encoder=ENCODER)
    torch.manual_seed(11)
    plain = V21ResearchCore(V21Config(len(vocabulary.words), **common))
    torch.manual_seed(11)
    joint = V21ResearchCore(V21Config(len(vocabulary.words), world_head=True, world_target='tokens', **common))
    mine = joint.state_dict()
    assert joint.world.target == 'tokens' and 'world.token_out.weight' in mine
    assert all(torch.equal(value, mine[key]) for key, value in plain.state_dict().items())
    with pytest.raises(ValueError, match='world_target'):
        V21Config(len(vocabulary.words), world_head=True, world_target='pixels', **common)
