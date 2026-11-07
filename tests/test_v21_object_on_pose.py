"""V21Config.object_on_pose (11/03): the Actor adds the wrist grasp cue to the pose token while it is seen, as it adds the
place cue, instead of reading it as a token of its own. In the release Actor (concat_inputs, no task words) the place
cue steers the pan (13 of 70 robot states) and the grasp cue does not (0 of 267). Same parameters either way; the
default builds every earlier model."""
from dataclasses import asdict, replace

import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_horizon_training import factual_state
from fcpg.v21_temporal_contract import PROGRESS_TRAINING_POLICY, begin_training_temporal_contract
from fcpg.v21_training import synthetic_chain


@pytest.fixture(autouse=True)
def no_attention_fast_path():
    before = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(before)


@pytest.fixture
def vocabulary():
    return TaskVocabulary.build(['pick red block', 'pick up the red fish toy'])


def build(vocabulary, **switches):
    torch.manual_seed(5)
    core = V21ResearchCore(V21Config(len(vocabulary.words), object_relative=True, prediction_horizon=40,
                                     actor_history=False, reader_task=True, gripper_separate=True, task_text=False,
                                     concat_inputs=True, place_relative=True, **switches)).eval()
    begin_training_temporal_contract(core, PROGRESS_TRAINING_POLICY)
    return core


def proposal(core, state, ids, cue):
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(len(state.q), -1) * .1
    return core.act(replace(state, object_relative=cue), ids, offsets).commands_position


def test_the_cue_on_the_pose_token_has_the_same_parameters_and_steers(vocabulary):
    token, on_pose = build(vocabulary), build(vocabulary, object_on_pose=True)
    assert on_pose.config.object_on_pose and asdict(on_pose.config)['object_on_pose'] is True
    assert {k: v.shape for k, v in token.state_dict().items()} == {k: v.shape for k, v in on_pose.state_dict().items()}
    on_pose.load_state_dict(token.state_dict())
    with torch.no_grad():                         # the cue's output layer starts at zero: make it read something
        for core in (token, on_pose):
            torch.nn.init.normal_(core.actor.object_relative[2].weight, std=.5)
        chain = synthetic_chain(vocabulary, batch=2)
        cue = torch.zeros(2, 5, 8)
        cue[..., 0], cue[..., 1], cue[..., 4], cue[..., 5] = 1., 1., .3, .1
        chain = replace(chain, object_relative=cue)
        state = factual_state(on_pose, chain, 2, images=True)
        seen = state.object_relative.clone()
        moved = seen.clone()
        moved[:, 2] += .1
        base = proposal(on_pose, state, chain.text_ids, seen)
        assert (proposal(on_pose, state, chain.text_ids, moved) - base).abs().max() > 1e-4       # the cue steers
        unseen, other = seen.clone(), moved.clone()
        unseen[:, 0] = other[:, 0] = 0.
        off = proposal(on_pose, state, chain.text_ids, unseen)
        torch.testing.assert_close(proposal(on_pose, state, chain.text_ids, other), off, rtol=0, atol=0)
        # a token of its own and the cue on the pose token are different models from the same parameters
        assert (proposal(token, state, chain.text_ids, seen) - base).abs().max() > 1e-4


def test_the_cue_on_the_pose_token_needs_the_cue():
    with pytest.raises(ValueError, match='object_on_pose'):
        V21Config(8, object_on_pose=True)
    with pytest.raises(ValueError, match='explicit booleans'):
        V21Config(8, object_relative=True, object_on_pose=1)
    assert V21Config(8).object_on_pose is False
