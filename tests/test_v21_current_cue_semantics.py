"""Current wrist object cues are not historical camera memory.

These tests open no hardware. They reproduce stale-cue leakage and protect the
same-pose refresh case where a fresh exposure replaces, rather than appends to,
the newest memory slot. Clearing a current cue must not delete the real bank.
"""

import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_memory import VisualObservation
from fcpg.v21_runtime import EncodedEndpoint, V21ResearchRuntime


CUE = torch.tensor([[1., .8, .3, -.2, .4, .15, -.8, .6]])


def observation(q, stamp, *, valid=(True, True), cue=CUE, marker=1.):
    batch = len(q)
    mask = torch.as_tensor(valid, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask[None].repeat(batch, 1)
    return VisualObservation(
        torch.full((batch, 2, 256, 384), float(marker)),
        q[:, None].repeat(1, 2, 1),
        torch.full((batch, 2), float(stamp), dtype=torch.float64), mask,
        object_relative=None if cue is None else cue.clone())


@pytest.fixture
def setup():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    vocab = TaskVocabulary.build(['pick red fish'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        core = V21ResearchCore(V21Config(len(vocab.words), object_relative=True)).eval()
    now = [10.]
    runtime = V21ResearchRuntime(core, vocab.encode(['pick red fish']),
        physical_check=lambda q, commands: None, lease_check=lambda: None,
        clock_fn=lambda: now[0])
    q = torch.zeros(1, 6)
    obs = observation(q, now[0])
    runtime.observe(EncodedEndpoint(q, torch.tensor(now, dtype=torch.float64), obs))
    yield core, runtime, now
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize('valid,cue', [((False, True), CUE),
                                     ((False, True), None),
                                     ((False, False), CUE),
                                     ((False, False), None),
                                     ((True, True), None)])
def test_core_current_absence_clears_cue_without_deleting_past_bank(setup, valid, cue):
    core, runtime, _ = setup
    previous = runtime.state
    before = previous.memory
    obs = observation(previous.q, 11., valid=valid, cue=cue, marker=2.)
    with torch.inference_mode():
        after = core.observe(previous, obs, previous.q, torch.tensor([11.], dtype=torch.float64))
    torch.testing.assert_close(after.object_relative, torch.zeros_like(CUE))
    assert after.memory.available.all(), 'camera outage must preserve actual history'
    if not valid[0]:
        torch.testing.assert_close(after.memory.features[:, 0], before.features[:, 0])
        torch.testing.assert_close(after.memory.q_at_image[:, 0], before.q_at_image[:, 0])
        torch.testing.assert_close(after.memory.image_times[:, 0], before.image_times[:, 0])
    torch.testing.assert_close(previous.object_relative, CUE)


def test_readback_only_clears_current_cue_and_keeps_historical_exposure(setup):
    core, runtime, _ = setup
    previous = runtime.state
    with torch.inference_mode():
        after = core.observe(previous, None, previous.q + 1., torch.tensor([11.], dtype=torch.float64))
    torch.testing.assert_close(after.object_relative, torch.zeros_like(CUE))
    assert after.memory is previous.memory
    torch.testing.assert_close(after.memory.image_times, previous.memory.image_times)
    assert not after.measurement_available.any()


def test_core_duplicate_exposure_can_have_current_cue_without_new_insertion(setup):
    core, runtime, _ = setup
    previous = runtime.state
    duplicate = observation(previous.q, 10., cue=CUE)
    with torch.inference_mode():
        after = core.observe(previous, duplicate, previous.q, torch.tensor([10.], dtype=torch.float64))
    assert not after.measurement_available.any(), 'duplicate is not a newly inserted frame'
    torch.testing.assert_close(after.object_relative, CUE)
    assert after.memory is previous.memory


@pytest.mark.parametrize('valid', [(True, True), (True, False)])
def test_runtime_same_pose_fresh_wrist_cue_survives_no_append_refresh(setup, valid):
    _, runtime, now = setup
    previous = runtime.state
    replacement = CUE.clone()
    replacement[:, 2] = -.4
    now[0] = 11.
    obs = observation(previous.q, now[0], valid=valid, cue=replacement, marker=2.)
    runtime.observe(EncodedEndpoint(previous.q, torch.tensor(now, dtype=torch.float64), obs))
    after = runtime.state
    torch.testing.assert_close(after.object_relative, replacement)
    assert after.memory.valid[0, 0].tolist() == [False, False, True]
    assert after.memory.image_times[0, 0, -1] == now[0]
    torch.testing.assert_close(after.memory.features[:, 0, -1], obs.features[:, 0])
    assert after.measurement_available[0, 0]


@pytest.mark.parametrize('valid,cue', [((False, True), CUE), ((False, False), CUE),
                                     ((True, True), None)])
def test_runtime_never_carries_old_cue_through_current_absence(setup, valid, cue):
    _, runtime, now = setup
    previous = runtime.state
    now[0] = 11.
    obs = observation(previous.q, now[0], valid=valid, cue=cue, marker=2.)
    runtime.observe(EncodedEndpoint(previous.q, torch.tensor(now, dtype=torch.float64), obs))
    torch.testing.assert_close(runtime.state.object_relative, torch.zeros_like(CUE))
    assert runtime.state.memory.available.all()
    if not valid[0]:
        torch.testing.assert_close(runtime.state.memory.image_times[:, 0], previous.memory.image_times[:, 0])


def test_runtime_readback_only_does_not_call_old_cue_current(setup):
    _, runtime, now = setup
    previous = runtime.state
    now[0] = 11.
    runtime.observe(EncodedEndpoint(previous.q, torch.tensor(now, dtype=torch.float64), None))
    torch.testing.assert_close(runtime.state.object_relative, torch.zeros_like(CUE))
    assert runtime.state.memory is previous.memory


def test_batch_mask_is_per_row_and_does_not_mutate_supplied_coordinates(setup):
    core, _, _ = setup
    q = torch.zeros(2, 6)
    cues = CUE.repeat(2, 1)
    before = cues.clone()
    obs = observation(q, 10., valid=((False, True), (True, False)), cue=cues)
    with torch.inference_mode():
        after = core.observe(None, obs, q, torch.tensor([10., 10.], dtype=torch.float64))
    torch.testing.assert_close(after.object_relative[0], torch.zeros(8))
    torch.testing.assert_close(after.object_relative[1], CUE[0])
    torch.testing.assert_close(cues, before)
    torch.testing.assert_close(obs.object_relative, before)


def test_invalid_available_cue_is_rejected_not_hidden_by_history(setup):
    core, runtime, _ = setup
    invalid = CUE.clone()
    invalid[:, 2] = torch.nan
    obs = observation(runtime.state.q, 11., cue=invalid)
    with pytest.raises(ValueError, match='object.relative'):
        core.observe(runtime.state, obs, runtime.state.q, torch.tensor([11.], dtype=torch.float64))
