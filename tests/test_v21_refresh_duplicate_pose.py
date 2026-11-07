"""A same-pose refresh replaces the newest exposure; a moved pose appends.

Training memory holds one exposure per transaction with the newest at age
zero. These tests check that resident captures keep that shape.
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
    torch.set_num_threads(2); torch.manual_seed(21)
    vocab = TaskVocabulary.build(["pick red fish"])
    core = V21ResearchCore(V21Config(len(vocab.words)))
    now = [10.]
    value = V21ResearchRuntime(core, vocab.encode(["pick red fish"]),
                               physical_check=lambda q, u: None, lease_check=lambda: None, clock_fn=lambda: now[0])
    return value, now


def complete_motion(value, now, seed):
    value.plan()
    source = value.state.q.clone()
    commands = value.begin_execution(source, torch.tensor([now[0]], dtype=torch.float64))
    trace = replace(value.pending.trace, source="recorded")
    path = torch.cat((source[:, None], commands), 1)
    times = torch.arange(6, dtype=torch.float64)[None] * .1 + now[0]
    now[0] += .52
    value.complete(trace, path, times, image(commands[:, -1], now[0], seed))
    value.stop()


def test_still_refreshes_replace_newest_slot_and_keep_it_fresh(runtime):
    value, now = runtime
    value.observe(image(torch.zeros(1, 6), now[0], 1))
    assert value.state.memory.valid[0, 0].tolist() == [False, False, True]
    for k in range(4):
        now[0] += .3
        value.observe(image(torch.zeros(1, 6), now[0], 10 + k))
    m = value.state.memory
    assert m.valid[0, 0].tolist() == [False, False, True], "still refreshes must not append"
    assert m.image_times[0, 0, -1] == now[0], "newest slot must carry the fresh clock"
    assert value.state.measurement_available.all() and value.state.measurement_pending.all()
    now[0] += .3
    value.observe(image(torch.ones(1, 6) * 5., now[0], 20))  # e.g. after HOME
    assert value.state.memory.valid[0, 0].tolist() == [False, True, True]


def test_memory_holds_one_frame_per_transaction_like_training(runtime):
    value, now = runtime
    value.observe(image(torch.zeros(1, 6), now[0], 1))
    for seed in (2, 3, 4):
        now[0] += .2
        value.observe(image(value.state.q, now[0], 40 + seed))  # resident refresh, no motion
        complete_motion(value, now, seed)
    m = value.state.memory
    assert m.valid[0, 0].all()
    poses = m.q_at_image[0, 0]
    assert (poses[1:] - poses[:-1]).abs().amax(-1).gt(.5).all(), "three slots, three distinct poses"
    assert (m.image_times[0, 0, 1:] > m.image_times[0, 0, :-1]).all()


def test_plan_capture_at_still_pose_refreshes_clock_without_evicting_history(runtime):
    value, now = runtime
    value.observe(image(torch.zeros(1, 6), now[0], 1))
    complete_motion(value, now, 2)
    before = value.state.memory
    now[0] += .1
    value.plan(endpoint=image(value.state.q, now[0], 7))
    after = value._pending_state.memory
    torch.testing.assert_close(after.image_times[0, :, :-1], before.image_times[0, :, :-1])
    assert (after.image_times[0, :, -1] == now[0]).all()
    torch.testing.assert_close(after.q_at_image, before.q_at_image)
    value.stop()
