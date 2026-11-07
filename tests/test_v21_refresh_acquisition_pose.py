"""Acquisition poses select image slots; later actual q still addresses the reader.

All observations, clocks and physical callbacks are synthetic. These tests do
not open hardware or establish physical grasp performance.
"""
from dataclasses import replace

import pytest
import torch

from fcpg.v21_memory import VisualObservation
from fcpg.v21_runtime import EncodedEndpoint
from test_v21_planning_memory import runtime  # noqa: F401


def exposure(value, now, acquisition_q, current_q, *, available=(True, True)):
    now[0] = 10.2
    q_at = torch.tensor(acquisition_q, dtype=torch.float32)[None, :, None].expand(1, 2, 6).clone()
    observation = VisualObservation(
        value.state.memory.features[:, :, -1].clone() + .25,
        q_at, torch.tensor([[10.1, 10.15]], dtype=torch.float64),
        torch.tensor([available], dtype=torch.bool))
    return EncodedEndpoint(torch.full((1, 6), float(current_q)),
                           torch.tensor([now[0]], dtype=torch.float64), observation)


def candidate(value, endpoint, operation, monkeypatch):
    if operation == 'observe':
        value.observe(endpoint)
        return value.state
    if operation == 'plan':
        value.plan(endpoint=endpoint)
        return value._pending_state
    seen = []
    original = value._planning_state

    def planning_state(endpoint):
        result = original(endpoint)
        seen.append(result)
        return result

    monkeypatch.setattr(value, '_planning_state', planning_state)
    value.preview(endpoint)
    assert len(seen) == 1
    return seen[0]


@pytest.mark.parametrize('operation', ['observe', 'plan', 'preview'])
@pytest.mark.parametrize('acquisition_q,current_q,expected', [
    ([4., 3.], 3., [[2., 3., 4.], [1., 2., 3.]]),
    ([3., 3.], 4., [[1., 2., 3.], [1., 2., 3.]]),
])
def test_slot_choice_uses_each_exposure_pose_not_later_current_q(
        runtime, monkeypatch, operation, acquisition_q, current_q, expected):
    value, now = runtime
    before = value.state
    endpoint = exposure(value, now, acquisition_q, current_q)
    state = candidate(value, endpoint, operation, monkeypatch)
    assert state.memory.q_at_image[0, :, :, 0].tolist() == expected
    torch.testing.assert_close(state.q, endpoint.q, atol=0, rtol=0)
    torch.testing.assert_close(state.time, endpoint.time, atol=0, rtol=0)
    torch.testing.assert_close(state.memory.q_at_image[:, :, -1], endpoint.observation.q_at_image)
    torch.testing.assert_close(state.memory.image_times[:, :, -1], endpoint.observation.image_times)
    with torch.inference_mode():
        rebuilt = value.core.reader(state.memory, endpoint.q, endpoint.time, state.history)
    torch.testing.assert_close(state.grounding, rebuilt)
    assert state.history is before.history
    if operation != 'observe':
        assert value.state is before
    assert before.memory.q_at_image[0, :, :, 0].tolist() == [[1., 2., 3.], [1., 2., 3.]]


@pytest.mark.parametrize('delta,append', [(.5, False), (-.5, False), (.501, True), (-.501, True)])
def test_acquisition_pose_keeps_existing_half_unit_threshold(runtime, delta, append):
    value, now = runtime
    endpoint = exposure(value, now, [3. + delta, 3.], 5.)
    value.observe(endpoint)
    expected_prefix = [2., 3.] if append else [1., 2.]
    assert value.state.memory.q_at_image[0, 0, :2, 0].tolist() == expected_prefix
    assert value.state.memory.q_at_image[0, 1, :, 0].tolist() == [1., 2., 3.]


def test_missing_view_cannot_select_or_poison_its_slots(runtime):
    value, now = runtime
    before = value.state.memory
    endpoint = exposure(value, now, [4., float('nan')], 3., available=(True, False))
    endpoint.observation.features[:, 1] = float('nan')
    endpoint.observation.image_times[:, 1] = float('nan')
    value.observe(endpoint)
    assert value.state.memory.q_at_image[0, 0, :, 0].tolist() == [2., 3., 4.]
    for field in ('features', 'q_at_image', 'image_times', 'valid'):
        torch.testing.assert_close(getattr(value.state.memory, field)[:, 1], getattr(before, field)[:, 1])
    assert not value.state.measurement_available[0, 1]
    assert torch.isfinite(value.state.grounding).all()


@pytest.mark.parametrize('changed_identity', [False, True])
def test_duplicate_exposure_cannot_become_new_motion_from_later_q(runtime, changed_identity):
    value, now = runtime
    before = value.state
    endpoint = exposure(value, now, [3., 3.], 4.)
    memory = before.memory
    observation = replace(endpoint.observation,
        features=memory.features[:, :, -1].clone(),
        image_times=memory.image_times[:, :, -1].clone())
    if changed_identity:
        observation.q_at_image[0, 0, 0] += 1.
    endpoint = replace(endpoint, observation=observation)
    if changed_identity:
        with pytest.raises(ValueError, match='same exposure identity'):
            value.observe(endpoint)
        assert value.state is before
    else:
        value.observe(endpoint)
        for field in ('features', 'q_at_image', 'image_times', 'valid'):
            torch.testing.assert_close(getattr(value.state.memory, field), getattr(memory, field))
        torch.testing.assert_close(value.state.q, endpoint.q)


@pytest.mark.parametrize('future', ['image', 'endpoint'])
def test_future_clocks_never_enter_memory_during_pose_classification(runtime, future):
    value, now = runtime
    before = value.state
    endpoint = exposure(value, now, [4., 3.], 3.)
    if future == 'image':
        endpoint.observation.image_times[0, 0] = now[0] + .1
    else:
        endpoint = replace(endpoint, time=endpoint.time + .1)
    with pytest.raises((ValueError, RuntimeError), match='future'):
        value.observe(endpoint)
    assert value.state is before
    assert before.memory.q_at_image[0, :, :, 0].tolist() == [[1., 2., 3.], [1., 2., 3.]]
