"""V21Config.task_text=False (James 11/02 「fcpg和actor 不要处理了文字了 没用 它看不懂的」): neither the FCPG reader nor the
Actor reads the task words, the read queries still follow the current joints, and the default still builds every
earlier model."""
import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_horizon_training import factual_state
from fcpg.v21_temporal_contract import PROGRESS_TRAINING_POLICY, begin_training_temporal_contract
from fcpg.v21_training import synthetic_chain

WORDS = ('reader.task_words', 'reader.task_queries', 'actor.text')


def build(vocabulary, **switches):
    torch.manual_seed(5)
    core = V21ResearchCore(V21Config(len(vocabulary.words), object_relative=True, prediction_horizon=40,
                                     actor_history=False, reader_task=True, **switches)).eval()
    begin_training_temporal_contract(core, PROGRESS_TRAINING_POLICY)
    return core


@pytest.fixture
def vocabulary():
    return TaskVocabulary.build(['pick red block', 'pick up the red fish toy'])


def test_default_reads_the_task_words(vocabulary):
    core = build(vocabulary)
    assert core.config.task_text
    assert {'reader.task_words.weight', 'actor.text.weight', 'actor.text_position'} <= set(core.state_dict())


def test_text_free_model_has_no_word_weights(vocabulary):
    core, words = build(vocabulary, task_text=False), build(vocabulary)
    names = set(core.state_dict())
    assert not any(name.startswith(WORDS) for name in names)
    assert any(name.startswith('reader.pose_queries') for name in names)
    shapes = {k: v.shape for k, v in words.state_dict().items() if not k.startswith(WORDS)}
    assert {k: v.shape for k, v in core.state_dict().items()} == shapes
    with pytest.raises(ValueError, match='task_text'):
        V21Config(len(vocabulary.words), task_text=1)


def test_text_free_reader_follows_the_joints_not_the_words(vocabulary):
    chain = synthetic_chain(vocabulary, batch=2)
    core = build(vocabulary, task_text=False)
    with torch.no_grad():
        state = factual_state(core, chain, 2, images=True)
        read = core.reader(state.memory, state.q, state.time, state.history, task_ids=chain.text_ids)
        fish = core.reader(state.memory, state.q, state.time, state.history,
                           task_ids=vocabulary.encode(['pick up the red fish toy'] * 2))
        none = core.reader(state.memory, state.q, state.time, state.history)
        moved = core.reader(state.memory, state.q + 5., state.time, state.history)
    torch.testing.assert_close(read, fish, rtol=0, atol=0)
    torch.testing.assert_close(read, none, rtol=0, atol=0)
    assert (moved - read).abs().max() > 1e-5


@pytest.mark.parametrize('task_text', (False, True))
def test_only_a_word_reading_actor_reads_the_words(vocabulary, task_text):
    chain = synthetic_chain(vocabulary, batch=2)
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(2, -1) * .1
    core = build(vocabulary, task_text=task_text)
    with torch.no_grad():
        state = factual_state(core, chain, 2, images=True)
        base = core.act(state, chain.text_ids, offsets).commands_position
        fish = core.act(state, vocabulary.encode(['pick up the red fish toy'] * 2), offsets).commands_position
    assert ((fish - base).abs().max() > 1e-6) == task_text
