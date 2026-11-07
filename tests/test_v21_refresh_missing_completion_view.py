"""A view that missed the completion frame must not acquire fabricated feedback on refresh.

Training: ``complete`` clears visual feedback; ``observe`` on the completion
image produces E_visual only for views whose completion frame arrived; the
Actor then sees ``visual_feedback_available=False`` and zero error for a view
without a completion frame. A resident refresh before the next plan is not a
completed transaction, so it must not turn a still-frame delta into E_visual
for that view either.
"""
from dataclasses import replace

import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_memory import VisualObservation
from fcpg.v21_runtime import EncodedEndpoint, V21ResearchRuntime


def image(q, now, seed, valid=(True, True)):
    features = torch.randn(1, 2, 256, 384, generator=torch.Generator().manual_seed(seed))
    obs = VisualObservation(features, q[:, None].expand(-1, 2, -1),
                            torch.full((1, 2), now, dtype=torch.float64), torch.tensor([valid]))
    return EncodedEndpoint(q, torch.tensor([now], dtype=torch.float64), obs)


@pytest.fixture
def runtime():
    torch.set_num_threads(2)
    torch.manual_seed(21)
    vocab = TaskVocabulary.build(["pick red fish"])
    core = V21ResearchCore(V21Config(len(vocab.words)))
    now = [10.]
    value = V21ResearchRuntime(core, vocab.encode(["pick red fish"]),
                               physical_check=lambda q, u: None, lease_check=lambda: None,
                               clock_fn=lambda: now[0])
    value.observe(image(torch.zeros(1, 6), now[0], 1))
    return value, now


def complete_motion(value, now, seed, completion_valid):
    value.plan()
    source = value.state.q.clone()
    commands = value.begin_execution(source, torch.tensor([now[0]], dtype=torch.float64))
    trace = replace(value.pending.trace, source="recorded")
    path = torch.cat((source[:, None], commands), 1)
    times = torch.arange(6, dtype=torch.float64)[None] * .1 + now[0]
    now[0] += .52
    value.complete(trace, path, times, image(commands[:, -1], now[0], seed, completion_valid))
    value.stop()


@pytest.mark.parametrize("operation", ["observe", "plan"])
def test_view_without_completion_frame_gets_no_feedback_from_a_later_refresh(runtime, operation):
    value, now = runtime
    complete_motion(value, now, 2, completion_valid=(True, False))
    previous = value.state
    # Column had an earlier frame in memory but no completion frame: training
    # semantics are "no visual feedback for column" on this event.
    assert previous.memory.available[0, 1]
    assert not previous.visual_feedback_available[0, 1]
    assert previous.visual_error[0, 1].abs().max() == 0
    assert previous.visual_feedback_available[0, 0]
    now[0] += .01
    fresh = image(previous.q, now[0], 3, valid=(True, True))
    if operation == "observe":
        value.observe(fresh)
        candidate = value.state
    else:
        value.plan(endpoint=fresh)
        candidate = value._pending_state
    # Wrist keeps its completed event.
    torch.testing.assert_close(candidate.visual_error[:, 0], previous.visual_error[:, 0], atol=0, rtol=0)
    assert candidate.visual_feedback_available[0, 0]
    # Column must NOT acquire a still-frame delta labelled as execution feedback.
    assert not candidate.visual_feedback_available[0, 1], "refresh fabricated column visual feedback"
    assert candidate.visual_error[0, 1].abs().max() == 0, "refresh injected a still-frame delta as E_visual"
    # The refresh itself is fine for grounding/memory.
    assert candidate.memory.image_times[0, 1, -1] == now[0]

