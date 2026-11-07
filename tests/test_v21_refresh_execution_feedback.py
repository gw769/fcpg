"""Synthetic IO only: fresh cameras must not erase the last execution event."""
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
    torch.set_num_threads(2); torch.manual_seed(21)
    vocab = TaskVocabulary.build(['pick red fish'])
    core = V21ResearchCore(V21Config(len(vocab.words)))
    now = [10.]
    value = V21ResearchRuntime(core, vocab.encode(['pick red fish']),
        physical_check=lambda q, u: None, lease_check=lambda: None, clock_fn=lambda: now[0])
    value.observe(image(torch.zeros(1, 6), now[0], 1))
    return value, now


def complete_motion(value, now, seed):
    value.plan()
    source = value.state.q.clone()
    commands = value.begin_execution(source, torch.tensor([now[0]], dtype=torch.float64))
    trace = replace(value.pending.trace, source='recorded')
    path = torch.cat((source[:, None], commands), 1)
    times = torch.arange(6, dtype=torch.float64)[None] * .1 + now[0]
    now[0] += .52
    value.complete(trace, path, times, image(commands[:, -1], now[0], seed))
    value.stop()


@pytest.mark.parametrize('operation', ['observe', 'preview', 'plan'])
def test_fresh_image_updates_grounding_but_retains_completed_event(runtime, operation):
    value, now = runtime
    complete_motion(value, now, 2)
    previous = value.state
    assert previous.visual_feedback_available.all() and previous.visual_error.norm() > 0
    now[0] += .01
    # A refresh at a NEW pose (e.g. after HOME) commits like a training exposure;
    # an unmoved pose would not be committed (see test_v21_refresh_duplicate_pose).
    fresh = image(previous.q + 5., now[0], 3)
    if operation == 'observe':
        value.observe(fresh); candidate = value.state
    elif operation == 'plan':
        value.plan(endpoint=fresh); candidate = value._pending_state
        assert value.state is previous
    else:
        seen = []
        handle = value.core.actor.register_forward_pre_hook(lambda m, args: seen.append(args[1].clone()))
        try: value.preview(fresh)
        finally: handle.remove()
        assert value.state is previous and value.pending is None and value.stopped
        torch.testing.assert_close(seen[-1], previous.visual_error, atol=0, rtol=0)
        candidate = value._refresh_endpoint(previous, fresh)
    torch.testing.assert_close(candidate.visual_error, previous.visual_error, atol=0, rtol=0)
    torch.testing.assert_close(candidate.visual_feedback_time, previous.visual_feedback_time, atol=0, rtol=0)
    assert candidate.grounding.sub(previous.grounding).abs().max() > 1e-5
    assert candidate.memory.image_times[0, 0, -1] == now[0]
    assert previous.memory.image_times[0, 0, -1] < now[0]
    assert candidate.feedback_time.item() == previous.feedback_time.item()


def test_event_is_replaced_by_next_real_completion_not_frozen_forever(runtime):
    value, now = runtime
    complete_motion(value, now, 2)
    old = value.state
    now[0] += .01
    value.observe(image(old.q, now[0], 3))
    complete_motion(value, now, 4)
    assert value.state.visual_feedback_available.all()
    assert (value.state.visual_feedback_time > old.visual_feedback_time).all()
    assert not torch.equal(value.state.visual_error, old.visual_error)
    assert value.state.history.valid.sum() == 2


def test_no_completed_execution_does_not_latch_camera_only_event(runtime):
    value, now = runtime
    now[0] += .01; value.observe(image(value.state.q, now[0], 2))
    old = value.state
    assert not old.feedback_available.any()
    now[0] += .01
    fresh = image(old.q + 5., now[0], 3)
    with torch.inference_mode():
        expected = value.core.observe(old, fresh.observation, fresh.q, fresh.time)
    value.observe(fresh)
    torch.testing.assert_close(value.state.visual_error, expected.visual_error, atol=0, rtol=0)
    assert (value.state.visual_feedback_time > old.visual_feedback_time).all()


def test_missing_view_preserves_old_exposure_and_feedback(runtime):
    value, now = runtime
    complete_motion(value, now, 2)
    old = value.state; now[0] += .01
    value.observe(image(old.q, now[0], 3, valid=(True, False)))
    assert not value.state.measurement_available[0, 1]
    torch.testing.assert_close(value.state.memory.image_times[:, 1], old.memory.image_times[:, 1], atol=0, rtol=0)
    torch.testing.assert_close(value.state.visual_error, old.visual_error, atol=0, rtol=0)


def test_future_endpoint_still_rejected(runtime):
    value, now = runtime
    complete_motion(value, now, 2); old = value.state
    with pytest.raises(RuntimeError, match='future'):
        value.observe(image(old.q, now[0] + 1., 3))
    assert value.state is old and value.stopped
