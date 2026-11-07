"""Regression tests for V21 command time, feedback lifetime and fair information access."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore, VERSION, CONTRACT
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_memory import VisualObservation
from fcpg.v21_training import (V21Schedule, synthetic_chain, chain_forward,
                              save_checkpoint, reuse_frozen_lewm, load_checkpoint, V21Trainer)


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(21)


def fixture(mode="readback"):
    vocab = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocab.words), memory_mode=mode)).eval()
    batch = synthetic_chain(vocab, batch=1)
    return core, vocab, batch


def image(batch, h):
    return VisualObservation(batch.features[:, h], batch.q[:, h, None].expand(-1, 2, -1),
        batch.endpoint_times[:, h, None].expand(-1, 2), batch.view_valid[:, h])


def start(core, batch):
    return core.observe(None, image(batch, 0), batch.q[:, 0], batch.endpoint_times[:, 0])


def finish(core, batch, state, h, observation=None):
    prediction = core.predict(state, batch.traces[h], batch.readback_offsets[:, h, 1:])
    return core.complete(state, prediction, batch.traces[h], batch.paths[:, h],
                         state.time[:, None]+batch.readback_offsets[:, h], observation)


def test_actor_times_are_explicit_causal_and_affect_output():
    core, _, batch = fixture()
    with torch.no_grad():
        state = start(core, batch)
        times = batch.readback_offsets[:, 0, :5]
        one = core.act(state, batch.text_ids, times).commands_position
        two = core.act(state, batch.text_ids, times*2).commands_position
        assert (one-two).abs().max() > 1e-6
        late = times.clone(); late[:, -1] *= 2
        three = core.act(state, batch.text_ids, late).commands_position
        torch.testing.assert_close(one[:, :4], three[:, :4], atol=1e-6, rtol=1e-6)
        with pytest.raises(TypeError):
            core.act(state, batch.text_ids)


@pytest.mark.parametrize("bad", ["float32", "reverse", "nonzero-start"])
def test_actor_rejects_ambiguous_command_clock(bad):
    core, _, batch = fixture()
    times = batch.readback_offsets[:, 0, :5].clone()
    if bad == "float32": times = times.float()
    if bad == "reverse": times = times.flip(-1)
    if bad == "nonzero-start": times += .1
    with torch.no_grad(), pytest.raises(ValueError):
        core.act(start(core, batch), batch.text_ids, times)


def test_training_passes_each_horizons_recorded_command_offsets():
    core, _, batch = fixture()
    observed = []
    handle = core.actor.register_forward_pre_hook(
        lambda _module, _args, kwargs: observed.append(kwargs["command_offsets"].clone()), with_kwargs=True)
    try:
        with torch.no_grad(): chain_forward(core, batch, "actor")
    finally:
        handle.remove()
    assert len(observed) == 4
    for h in range(4):
        torch.testing.assert_close(observed[h], batch.readback_offsets[:, h, :5], atol=0, rtol=0)


def test_completion_retimes_only_from_source_context(monkeypatch):
    core, _, batch = fixture()
    with torch.no_grad():
        source = start(core, batch)
        forecast = core.predict(source, batch.traces[0], batch.readback_offsets[:, 0, 1:])
        trace = replace(batch.traces[0], offsets=batch.traces[0].offsets*2)
        actual_times = source.time[:, None]+batch.readback_offsets[:, 0]*2
        target = batch.paths[:, 0].clone(); target[:, 1:] += 10
        expected = core.lewm(source.q, trace, source.history, batch.readback_offsets[:, 0, 1:]*2)
        calls = []
        original = core.lewm.forward
        def record(q, commands, history, times):
            calls.append((q.clone(), history, times.clone()))
            return original(q, commands, history, times)
        monkeypatch.setattr(core.lewm, "forward", record)
        end = core.complete(source, forecast, trace, target, actual_times)
    assert len(calls) == 1 and calls[0][1] is source.history
    torch.testing.assert_close(calls[0][0], source.q, atol=0, rtol=0)
    torch.testing.assert_close(end.q_error, target[:, 1:]-expected.predicted_q_path[:, 1:], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(end.feedback_offsets, batch.readback_offsets[:, 0, 1:]*2, atol=1e-9, rtol=0)
    assert end.feedback_retimed.all() and not end.q_error.requires_grad


def test_matching_clock_reuses_prediction_without_extra_arm_call(monkeypatch):
    core, _, batch = fixture()
    with torch.no_grad():
        state = start(core, batch)
        forecast = core.predict(state, batch.traces[0], batch.readback_offsets[:, 0, 1:])
        monkeypatch.setattr(core.lewm, "forward", lambda *args: pytest.fail("unneeded ARM forward"))
        result = core.complete(state, forecast, batch.traces[0], batch.paths[:, 0], batch.readback_offsets[:, 0])
    assert not result.feedback_retimed.any()


def test_duplicate_refresh_does_not_change_feedback_or_action():
    core, _, batch = fixture()
    with torch.no_grad():
        state = finish(core, batch, start(core, batch), 0, image(batch, 1))
        before = core.act(state, batch.text_ids, batch.readback_offsets[:, 1, :5]).commands_position
        assert state.visual_feedback_available.all() and state.visual_error.norm() > 0
        after = core.observe(state, image(batch, 1), state.q, state.time)
        repeated = core.act(after, batch.text_ids, batch.readback_offsets[:, 1, :5]).commands_position
    assert not after.measurement_available.any() and after.measurement_pending.all()
    torch.testing.assert_close(after.visual_error, state.visual_error, atol=0, rtol=0)
    torch.testing.assert_close(before, repeated, atol=0, rtol=0)


def test_readback_only_refresh_preserves_event_age_and_completed_action_consumes_it():
    core, _, batch = fixture()
    with torch.no_grad():
        state = finish(core, batch, start(core, batch), 0, image(batch, 1))
        refresh = core.observe(state, None, state.q, state.time+.2)
        torch.testing.assert_close(refresh.visual_error, state.visual_error, atol=0, rtol=0)
        torch.testing.assert_close(refresh.visual_feedback_time, state.visual_feedback_time, atol=0, rtol=0)
        assert refresh.visual_feedback_available.all()
        end = finish(core, batch, state, 1)
    assert not end.measurement_pending.any() and not end.visual_feedback_available.any()
    assert end.visual_error.count_nonzero() == 0


def test_new_wrist_does_not_erase_unconsumed_column_event():
    core, _, batch = fixture()
    with torch.no_grad():
        state = finish(core, batch, start(core, batch), 0, image(batch, 1))
        obs = replace(image(batch, 2), measurement_available=torch.tensor([[True, False]]))
        refreshed = core.observe(state, obs, batch.q[:, 2], batch.endpoint_times[:, 2])
    torch.testing.assert_close(refreshed.visual_error[:, 1], state.visual_error[:, 1], atol=0, rtol=0)
    assert refreshed.visual_feedback_time[0, 1] == state.visual_feedback_time[0, 1]
    assert refreshed.visual_feedback_time[0, 0] > state.visual_feedback_time[0, 0]


def test_temporal_actor_receives_same_motion_information_and_responds_to_history():
    temporal, _, batch = fixture("temporal")
    addressed, _, _ = fixture("readback"); addressed.load_state_dict(temporal.state_dict())
    with torch.no_grad():
        state = finish(temporal, batch, start(temporal, batch), 0, image(batch, 1))
        path = state.history.readbacks_position.clone()
        path[:, -1, 1, 0] += 30; path[:, -1, 2, 0] -= 30
        changed = replace(state, history=replace(state.history, readbacks_position=path))
        a = temporal.reader.frame_metadata(state.memory, state.q, state.time, state.history)
        b = addressed.reader.frame_metadata(state.memory, state.q, state.time, state.history)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        original = temporal.act(state, batch.text_ids, batch.readback_offsets[:, 1, :5]).commands_position
        altered = temporal.act(changed, batch.text_ids, batch.readback_offsets[:, 1, :5]).commands_position
    assert (original-altered).abs().max() > 1e-6
    assert sum(p.numel() for p in temporal.parameters()) == sum(p.numel() for p in addressed.parameters())


def test_only_compatible_frozen_arm_is_imported(tmp_path):
    core, vocabulary, _ = fixture(); schedule = V21Schedule()
    old, _, _ = fixture(); old.set_loss_scale(core.q_loss_scale)
    with torch.no_grad():
        old.lewm.response.bias.fill_(.7)
        old.actor.output.bias.fill_(float("nan"))  # deliberately unusable old policy
    path = tmp_path/"source"/"lewm"/"final.pt"
    source_paths = {"endpoint_root": "/test/actual"}
    metrics = dict(updates=20000, pair_exposures=5120000, complete=True)
    save_checkpoint(path, old, vocabulary, schedule, stage="lewm", step=20000, complete=True,
                    smoke=False, metrics=metrics)
    payload = torch.load(path, weights_only=False)
    payload.update(version="21.0.0", contract="v21-readback-addressed-visual-memory-v1")
    torch.save(payload, path)
    (path.parent.parent/"config.json").write_text(json.dumps(dict(source_paths=source_paths,
                                                               smoke=False, schedule=asdict(schedule))))
    initial_actor = {key: value.clone() for key, value in core.actor.state_dict().items()}
    result = reuse_frozen_lewm(path, core, vocabulary, schedule, source_paths)
    assert result["additional_training_updates"] == 0 and not result["actor_or_reader_imported"]
    assert core.lewm.response.bias[0] == pytest.approx(.7)
    for key, value in core.actor.state_dict().items():
        torch.testing.assert_close(value, initial_actor[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="checkpoint version"):
        load_checkpoint(path, core, vocabulary, schedule)
    with pytest.raises(ValueError, match="data"):
        reuse_frozen_lewm(path, core, vocabulary, schedule, {})


def test_resume_repairs_missing_stage_report_without_new_updates(tmp_path):
    core, vocabulary, _ = fixture()
    trainer = V21Trainer(core, vocabulary, tmp_path, smoke=True)
    optimizer = trainer.configure("actor")
    metrics = dict(updates=2, pair_exposures=32, complete=True)
    save_checkpoint(tmp_path/"actor"/"final.pt", core, vocabulary, trainer.schedule,
                    stage="actor", step=2, complete=True, smoke=True, optimizer=optimizer, metrics=metrics)
    result = trainer.run("actor", [], lambda _: pytest.fail("completed stage must not train"), 2)
    assert result == metrics
    assert json.loads((tmp_path/"metrics.json").read_text())["actor"] == metrics
