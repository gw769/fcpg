"""Pure-model checks for assisted preview; no camera, robot, serial or GPU I/O."""
from dataclasses import replace

import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_memory import VisualObservation
from fcpg.v21_runtime import EncodedEndpoint, V21ResearchRuntime


def endpoint(stamp: float, marker: int) -> EncodedEndpoint:
    generator = torch.Generator().manual_seed(marker)
    q = torch.full((1, 6), float(marker))
    observation = VisualObservation(
        torch.randn(1, 2, 256, 384, generator=generator),
        q[:, None].expand(-1, 2, -1),
        torch.full((1, 2), stamp, dtype=torch.float64),
        torch.ones(1, 2, dtype=torch.bool),
    )
    return EncodedEndpoint(q, torch.tensor([stamp], dtype=torch.float64), observation)


@pytest.fixture
def runtime():
    torch.set_num_threads(2)
    torch.manual_seed(211)
    vocabulary = TaskVocabulary.build(["pick red fish"])
    core = V21ResearchCore(V21Config(len(vocabulary.words)))
    now = [10.0]
    audits = []
    value = V21ResearchRuntime(
        core,
        vocabulary.encode(["pick red fish"]),
        physical_check=lambda q, commands: audits.append((q.clone(), commands.clone())),
        lease_check=lambda: None,
        clock_fn=lambda: now[0],
        max_plan_age_s=300.0,
    )
    for stamp, marker in ((8.0, 1), (9.0, 2), (10.0, 3)):
        value.observe(endpoint(stamp, marker))
    return value, now, audits


def test_default_preview_is_the_unchanged_raw_actor(runtime):
    value, now, audits = runtime
    before = value.state
    image = endpoint(now[0], 3)
    dt = torch.full((1, 5), 0.1)
    offsets = torch.cat((dt.new_zeros(1, 1).double(), dt.double().cumsum(-1)[:, :-1]), -1)
    with torch.inference_mode():
        candidate = value._planning_state(image)
        expected = value.core.act(candidate, value.text_ids, offsets).commands_position
        commands, forecast, admission = value.preview(image)
    torch.testing.assert_close(commands, expected)
    torch.testing.assert_close(forecast.trace.positions, expected)
    assert admission == {"allowed": True, "reason": None}
    assert len(audits) == 1
    assert value.state is before and value.pending is None and value._pending_state is None
    assert value.stopped and not value.executing


def test_assisted_builder_receives_exact_offsets_and_forecast_uses_returned_commands(runtime):
    value, now, audits = runtime
    before = value.state
    seen = {}
    def builder(state, command_offsets):
        seen["state"] = state
        seen["offsets"] = command_offsets.detach().clone()
        ramp = torch.arange(5, device=state.q.device, dtype=state.q.dtype)[None, :, None]
        return state.q[:, None] + ramp.expand(-1, -1, 6) * 0.25
    builder.wants_command_offsets = True

    commands, forecast, admission = value.preview(endpoint(now[0], 3), command_builder=builder)
    expected_dt = torch.full((1, 5), 0.1)
    expected_offsets = torch.cat(
        (expected_dt.new_zeros(1, 1).double(), expected_dt.double().cumsum(-1)[:, :-1]), -1
    )
    torch.testing.assert_close(seen["offsets"], expected_offsets, atol=0, rtol=0)
    torch.testing.assert_close(commands, forecast.trace.positions, atol=0, rtol=0)
    torch.testing.assert_close(audits[-1][1], commands, atol=0, rtol=0)
    assert forecast.trace.source == "planned" and forecast.trace.complete is True
    assert admission["allowed"]
    assert value.state is before and value.pending is None and value._pending_state is None
    assert value.stopped and not value.executing


def test_physical_rejection_is_a_nonmutating_audit_and_forecast_stays_exact(runtime):
    value, now, _ = runtime
    before = value.state
    before_features = before.memory.features.clone()
    def builder(state):
        return state.q[:, None].expand(-1, 5, -1).clone() + 0.5
    clean = builder(before)
    def reject_and_try_to_mutate(q, commands):
        q.add_(100)
        commands.add_(100)
        raise RuntimeError("clearance")
    value.physical_check = reject_and_try_to_mutate

    commands, forecast, admission = value.preview(endpoint(now[0], 3), command_builder=builder)
    assert admission == {"allowed": False, "reason": "clearance"}
    torch.testing.assert_close(commands, clean, atol=0, rtol=0)
    torch.testing.assert_close(forecast.trace.positions, clean, atol=0, rtol=0)
    assert value.state is before and value.pending is None and value._pending_state is None
    torch.testing.assert_close(value.state.memory.features, before_features, atol=0, rtol=0)
    assert value.stopped and not value.executing


@pytest.mark.parametrize("bad", [torch.zeros(1, 5, 5), torch.full((1, 5, 6), float("nan"))])
def test_bad_assisted_commands_fail_before_physical_audit_or_forecast(runtime, monkeypatch, bad):
    value, now, audits = runtime
    before = value.state
    monkeypatch.setattr(value.core, "predict", lambda *args, **kwargs: pytest.fail("forecast accepted bad commands"))
    with pytest.raises(ValueError, match="preview commands"):
        value.preview(endpoint(now[0], 3), command_builder=lambda _state: bad)
    assert not audits
    assert value.state is before and value.pending is None and value._pending_state is None


def test_preview_does_not_consume_existing_feedback_or_commit_fresh_endpoint(runtime):
    value, now, _ = runtime
    value.state = replace(
        value.state,
        visual_error=torch.ones_like(value.state.visual_error),
        visual_feedback_available=torch.ones_like(value.state.visual_feedback_available),
        visual_feedback_time=torch.full_like(value.state.visual_feedback_time, 9.0),
    )
    before = value.state
    now[0] = 10.1
    fresh = endpoint(now[0], 4)
    value.preview(fresh, command_builder=lambda state: state.q[:, None].expand(-1, 5, -1).clone())
    assert value.state is before and value.pending is None and value._pending_state is None
    assert value.state.visual_feedback_available.all()
    torch.testing.assert_close(value.state.visual_error, torch.ones_like(value.state.visual_error), atol=0, rtol=0)
    assert value.state.memory.image_times[0, 0, -1] < fresh.time.item()
