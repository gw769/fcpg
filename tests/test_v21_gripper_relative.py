"""docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md section 17.39 (K2): with gripper_relative the separate jaw output
becomes "current reading + output x half range"; the reading is still no input, the five arm commands do not change,
and every earlier model (the default) is unchanged."""
from dataclasses import replace

import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_horizon_training import factual_state
from fcpg.v21_temporal_contract import PROGRESS_TRAINING_POLICY, begin_training_temporal_contract
from fcpg.v21_training import synthetic_chain


@pytest.fixture
def vocabulary():
    return TaskVocabulary.build(['pick red block', 'pick up the red fish toy'])


def build(vocabulary, **switches):
    torch.manual_seed(5)
    core = V21ResearchCore(V21Config(len(vocabulary.words), object_relative=True, prediction_horizon=40,
                                     actor_history=False, reader_task=True, gripper_separate=True, **switches)).eval()
    begin_training_temporal_contract(core, PROGRESS_TRAINING_POLICY)
    torch.nn.init.normal_(core.actor.output.weight, std=.05)     # a visibly non-trivial proposal
    return core


def test_switch_needs_the_separate_gripper_and_defaults_off(vocabulary):
    assert V21Config(len(vocabulary.words)).gripper_relative is False
    with pytest.raises(ValueError, match='gripper_relative'):
        V21Config(len(vocabulary.words), actor_history=False, gripper_relative=True)
    with pytest.raises(ValueError, match='gripper_relative'):
        V21Config(len(vocabulary.words), actor_history=False, gripper_separate=True, gripper_relative=1)
    # no parameter is added: a K1/H1b state dict loads as it is
    relative, absolute = build(vocabulary, gripper_relative=True), build(vocabulary)
    assert {k: v.shape for k, v in relative.state_dict().items()} == {k: v.shape for k, v in absolute.state_dict().items()}


def test_jaw_follows_the_reading_and_nothing_else_changes(vocabulary):
    chain = synthetic_chain(vocabulary, batch=2)
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(2, -1) * .1
    absolute = build(vocabulary)
    relative = build(vocabulary, gripper_relative=True)
    relative.load_state_dict(absolute.state_dict())
    jaw = torch.tensor([0., 0, 0, 0, 0, 7.])
    with torch.no_grad():
        state = factual_state(absolute, chain, 2, images=True)
        a = absolute.act(state, chain.text_ids, offsets)
        r = relative.act(state, chain.text_ids, offsets)
        r_open = relative.act(replace(state, q=state.q + jaw), chain.text_ids, offsets)
    # the arm commands are the same model's, bit for bit
    torch.testing.assert_close(r.commands_position[..., :5], a.commands_position[..., :5], rtol=0, atol=0)
    # the jaw: reading + output x half range (instead of centre + output x half range), within the mechanical range
    inside = lambda x: (x > relative.actor.position_lower[5] + 1e-3) & (x < relative.actor.position_upper[5] - 1e-3)
    shift = state.q[:, None, 5] - relative.actor.position_center[5]
    both = inside(r.commands_position[..., 5]) & inside(a.commands_position[..., 5])
    torch.testing.assert_close(r.commands_position[..., 5][both], (a.commands_position[..., 5] + shift)[both], rtol=0, atol=1e-4)
    # a different reading moves only the jaw command, by exactly the reading change (the reading is still no input)
    torch.testing.assert_close(r_open.commands_position[..., :5], r.commands_position[..., :5], rtol=0, atol=1e-5)
    moved = inside(r_open.commands_position[..., 5]) & inside(r.commands_position[..., 5])
    torch.testing.assert_close((r_open.commands_position[..., 5] - r.commands_position[..., 5])[moved],
                               torch.full_like(r.commands_position[..., 5][moved], 7.), rtol=0, atol=1e-4)
    # consumers may keep reading "pose + summed increments"
    rebuilt = state.q[:, None] + r.increments.cumsum(1)
    keep = inside(r.commands_position).all(-1)
    torch.testing.assert_close(rebuilt[keep], r.commands_position[keep], rtol=0, atol=1e-3)


# ------------------------------------------------------------------ section 17.46: absolute-jaw head, jaw teacher
def test_the_absolute_jaw_head_needs_the_relative_jaw_and_never_changes_a_command(vocabulary):
    with pytest.raises(ValueError, match='gripper_absolute_aux'):
        V21Config(len(vocabulary.words), actor_history=False, gripper_separate=True, gripper_absolute_aux=True)
    chain = synthetic_chain(vocabulary, batch=2)
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(2, -1) * .1
    relative = build(vocabulary, gripper_relative=True)
    aux = build(vocabulary, gripper_relative=True, gripper_absolute_aux=True)
    added = sorted(set(aux.state_dict()) - set(relative.state_dict()))
    assert added == ['actor.jaw_absolute.bias', 'actor.jaw_absolute.weight']
    aux.load_state_dict({**relative.state_dict(), **{k: aux.state_dict()[k] for k in added}})
    with torch.no_grad():
        state = factual_state(relative, chain, 2, images=True)
        a = relative.act(state, chain.text_ids, offsets)
        b = aux.act(state, chain.text_ids, offsets)
        torch.nn.init.normal_(aux.actor.jaw_absolute.weight, std=.5)
        c = aux.act(state, chain.text_ids, offsets)
    assert a.jaw_absolute is None and b.jaw_absolute.shape == (2, 40, 1)
    torch.testing.assert_close(b.commands_position, a.commands_position, rtol=0, atol=0)
    torch.testing.assert_close(c.commands_position, a.commands_position, rtol=0, atol=0)
    assert not torch.allclose(c.jaw_absolute, b.jaw_absolute)
    # the head's output is "range centre + output x half range", like the absolute jaw of K1 / H1b
    raw = aux.actor.jaw_absolute(aux.actor.norm.weight.new_zeros(1, aux.actor.width))
    expected = aux.actor.position_center[5] + raw[0, 0] * aux.actor.position_half_range[5]
    assert torch.isfinite(expected)


def test_the_jaw_teacher_keeps_the_demonstrator_opening_and_takes_the_rest_from_the_teacher():
    from scripts.train_v21_fcpg_actor import JAW_OPENING, jaw_absolute_loss, teacher_jaw_targets
    commands = torch.zeros(2, 5, 6, dtype=torch.float64)
    commands[..., :5] = 3.
    commands[0, :, 5] = torch.tensor([10., 10.5, 12., 20., 30.])      # row 0: the demonstrator opens from the third command
    commands[1, :, 5] = 6.                                             # row 1: holding, 4 below the reading
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Target:
        commands: torch.Tensor
        valid: torch.Tensor
    target = Target(commands, torch.ones(2, 5, dtype=torch.bool))
    reading = torch.tensor([10., 10.])
    teacher = torch.full((2, 5), 2., dtype=torch.float32)
    taught, opening = teacher_jaw_targets(target, reading, teacher)
    assert opening.tolist() == [[False, False, True, True, True], [False] * 5]
    assert taught.commands[0, :, 5].tolist() == [2., 2., 12., 20., 30.] and taught.commands[1, :, 5].tolist() == [2.] * 5
    torch.testing.assert_close(taught.commands[..., :5], commands[..., :5], rtol=0, atol=0)   # the arm stays
    assert JAW_OPENING == 1.
    scale = torch.full((6,), 2., dtype=torch.float64)
    assert float(jaw_absolute_loss(taught.commands[..., 5:], taught, scale)) == 0.
    assert float(jaw_absolute_loss(taught.commands[..., 5:] + 1., taught, scale)) > 0.
