"""V21 research contracts. CPU only; no camera, serial or source-data access."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from fcpg.v21_memory import VisualMemory, VisualObservation, ReadbackMemoryReader
from fcpg.v21_models import CommandTrace, V21LeWM, V21Actor
from fcpg.v21_core import V21Config, V21ResearchCore, CONTRACT
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_runtime import V21ResearchRuntime, EncodedEndpoint, FrozenDINOFrames
from fcpg.v21_training import (V21Trainer, V21Schedule, synthetic_chain,
                              save_checkpoint, load_checkpoint, rng_state)
from fcpg.prod_v20_policy import empty_history


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2); torch.manual_seed(21)


def test_runtime_rejects_future_actual_clock():
    vocabulary = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocabulary.words)))
    runtime = V21ResearchRuntime(core, vocabulary.encode(["pick red block"]),
        physical_check=lambda q, u: None, lease_check=lambda: None, clock_fn=lambda: 10.)
    q = torch.zeros(1, 6)
    with pytest.raises(RuntimeError, match="future"):
        runtime.observe(EncodedEndpoint(q, torch.tensor([11.], dtype=torch.float64), None))
    runtime.observe(EncodedEndpoint(q, torch.tensor([10.], dtype=torch.float64), observation()))
    runtime.plan()
    with pytest.raises(RuntimeError, match="future"):
        runtime.begin_execution(q, torch.tensor([10.05], dtype=torch.float64))
    assert runtime.stopped and runtime.pending is None


def test_completed_history_cannot_invent_a_command_before_first_dispatch():
    vocabulary = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocabulary.words)))
    q = torch.zeros(1, 6)
    state = core.observe(None, observation(), q, torch.tensor([10.], dtype=torch.float64))
    trace = CommandTrace.policy(torch.zeros(1, 5, 6), torch.full((1, 5), .1))
    prediction = core.predict(state, trace, torch.tensor([[.1, .2, .3, .4, .5]], dtype=torch.float64))
    actual = replace(trace, source="recorded", offsets=trace.offsets+.11)
    times = torch.tensor([[10., 10.1, 10.2, 10.3, 10.4, 10.6]], dtype=torch.float64)
    with pytest.raises(ValueError, match="preceding recorded command"):
        core.complete(state, prediction, actual, torch.zeros(1, 6, 6), times)


def observation(t=10., *, valid=(True, True), batch=1):
    return VisualObservation(torch.randn(batch, 2, 256, 384), torch.zeros(batch, 2, 6),
        torch.full((batch, 2), t, dtype=torch.float64), torch.tensor([valid]).expand(batch, -1).clone())


def test_memory_is_actual_immutable_deduplicated_and_per_view_fifo():
    memory = VisualMemory.empty(1); first = observation()
    result, inserted = memory.write(first, torch.tensor([10.], dtype=torch.float64))
    assert inserted.all() and not memory.valid.any()
    duplicate, inserted = result.write(first, torch.tensor([11.], dtype=torch.float64))
    assert duplicate is result and not inserted.any()
    first.features.zero_()
    assert result.features.abs().sum() > 0
    with pytest.raises(ValueError, match="changed features"):
        result.write(first, torch.tensor([11.], dtype=torch.float64))
    for t in (11., 12., 13.):
        result, _ = result.write(observation(t, valid=(True, False)), torch.tensor([t], dtype=torch.float64))
    assert result.image_times[0, 0].tolist() == [11., 12., 13.]
    assert result.image_times[0, 1, -1].item() == 10.
    assert result.valid[0, 1].sum() == 1


@pytest.mark.parametrize("change", ["future", "forecast", "wrong-clock"])
def test_memory_rejects_bad_evidence(change):
    value = observation()
    if change == "future": value = replace(value, image_times=value.image_times+1)
    if change == "forecast": value = replace(value, source="predicted")
    if change == "wrong-clock": value = replace(value, image_times=value.image_times.float())
    with pytest.raises(ValueError):
        VisualMemory.empty(1).write(value, torch.tensor([10.], dtype=torch.float64))


def test_invalid_camera_values_do_not_poison_memory():
    value = observation(valid=(False, False))
    value.features.fill_(float("nan")); value.q_at_image.fill_(float("nan")); value.image_times.fill_(float("nan"))
    memory = VisualMemory.empty(1)
    same, added = memory.write(value, torch.tensor([10.], dtype=torch.float64))
    assert same is memory and not added.any()


def test_readback_addresses_change_read_not_memory_and_temporal_control_has_same_size():
    memory, _ = VisualMemory.empty(1).write(observation(), torch.tensor([10.], dtype=torch.float64))
    reader = ReadbackMemoryReader().eval()
    baseline = ReadbackMemoryReader(mode="temporal").eval(); baseline.load_state_dict(reader.state_dict())
    q = torch.zeros(1, 6); t = torch.tensor([11.], dtype=torch.float64); history = empty_history(1)
    before = memory.features.clone()
    with torch.no_grad():
        one = reader(memory, q, t, history)
        moved = reader(memory, q+20, t, history)
        assert (one-moved).abs().max() > 1e-7
        torch.testing.assert_close(baseline(memory, q, t, history), baseline(memory, q+20, t, history), atol=0, rtol=0)
    assert torch.equal(memory.features, before)
    assert sum(p.numel() for p in reader.parameters()) == sum(p.numel() for p in baseline.parameters())


def test_projection_cache_is_eval_only_and_invalidated_by_weight_changes():
    memory, _ = VisualMemory.empty(1).write(observation(), torch.tensor([10.], dtype=torch.float64))
    reader = ReadbackMemoryReader().eval(); q = torch.zeros(1, 6); t = torch.tensor([10.], dtype=torch.float64)
    with torch.inference_mode():
        reader(memory, q, t, empty_history(1)); cached = reader._projection_cache
        reader(memory, q, t, empty_history(1)); assert reader._projection_cache is cached
    with torch.no_grad(): reader.feature_projection.weight[0, 0].add_(.1)
    with torch.inference_mode(): reader(memory, q, t, empty_history(1))
    assert reader._projection_cache is not cached
    reader.train(); assert reader._projection_cache is None
    reader(memory, q, t, empty_history(1)); assert reader._projection_cache is None


def make_trace(batch=1, count=15):
    return CommandTrace(torch.randn(batch, count, 6),
        (torch.arange(count, dtype=torch.float64)[None]/30).expand(batch, -1),
        torch.ones(batch, count, dtype=torch.bool), "recorded", True)


def test_arm_dense_command_causality_and_q0():
    model = V21LeWM().eval(); q = torch.zeros(1, 6); trace = make_trace()
    qt = torch.arange(1, 6, dtype=torch.float64)[None]*.1
    changed = trace.positions.clone(); changed[:, 9:] += 20
    with torch.no_grad():
        a = model(q, trace, empty_history(1), qt)
        b = model(q, replace(trace, positions=changed), empty_history(1), qt)
    torch.testing.assert_close(a.predicted_q_path[:, :4], b.predicted_q_path[:, :4], atol=1e-6, rtol=1e-6)
    assert (a.predicted_q_path[:, 4:]-b.predicted_q_path[:, 4:]).abs().max() > 1.
    assert torch.equal(a.predicted_q_path[:, 0], q)
    assert (a.q_uncertainty > 0).all() and a.hidden.shape == (1, 5, 192)


def test_command_boundary_and_padding_are_not_future_actions():
    q = torch.zeros(1, 6); trace = make_trace(count=5)
    trace = replace(trace, offsets=torch.arange(5, dtype=torch.float64)[None]*.1)
    qt = torch.arange(1, 6, dtype=torch.float64)[None]*.1
    torch.testing.assert_close(trace.active_at(qt, q), trace.positions)
    padded = CommandTrace(torch.cat((trace.positions, torch.full((1, 2, 6), float("nan"))), 1),
        torch.cat((trace.offsets, torch.full((1, 2), float("nan"), dtype=torch.float64)), 1),
        torch.tensor([[True]*5+[False]*2]), "recorded", True)
    model = V21LeWM().eval()
    with torch.no_grad():
        a = model(q, trace, empty_history(1), qt)
        b = model(q, padded, empty_history(1), qt)
    torch.testing.assert_close(a.predicted_q_path, b.predicted_q_path, atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="complete"):
        replace(trace, complete=False).validate(1)


def test_actor_keeps_available_memory_when_current_column_is_missing():
    model = V21Actor(8).eval(); g = torch.randn(1, 2, 8, 192); e = torch.zeros_like(g)
    q = torch.zeros(1, 6); text = torch.tensor([[2, 4, 3]])
    kwargs = dict(memory_available=torch.tensor([[True, True]]), measurement_pending=torch.tensor([[True, False]]),
        command_offsets=torch.arange(5, dtype=torch.float64)[None]*.1,
        motion_context=torch.zeros(1, 2, 3, 28), motion_available=torch.zeros(1, 2, 3, dtype=torch.bool),
        visual_feedback_available=torch.zeros(1, 2, dtype=torch.bool), q_error=torch.zeros(1, 5, 6),
        visual_feedback_age=torch.zeros(1, 2), feedback_offsets=torch.zeros(1, 5, dtype=torch.float64),
        feedback_available=torch.zeros(1, dtype=torch.bool), feedback_age=torch.zeros(1))
    changed = g.clone(); changed[:, 1] = torch.randn_like(changed[:, 1])*5
    with torch.no_grad():
        a = model(g, e, q, text, **kwargs).commands_position
        b = model(changed, e, q, text, **kwargs).commands_position
        assert (a-b).abs().max() > 1e-6
        kwargs["memory_available"][:, 1] = False
        a = model(g, e, q, text, **kwargs).commands_position
        b = model(changed, e, q, text, **kwargs).commands_position
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_core_h0_forecast_complete_and_readback_only_history():
    vocabulary = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocabulary.words))).eval()
    batch = synthetic_chain(vocabulary, batch=2)
    with torch.no_grad():
        initial = VisualObservation(batch.features[:, 0], batch.q[:, 0, None].expand(-1, 2, -1),
            batch.endpoint_times[:, :1].expand(-1, 2), batch.view_valid[:, 0])
        state = core.observe(None, initial, batch.q[:, 0], batch.endpoint_times[:, 0])
        assert not state.feedback_available.any() and not state.visual_feedback_available.any()
        memory = state.memory
        predicted = core.predict(state, batch.traces[0], batch.readback_offsets[:, 0, 1:])
        assert state.memory is memory and not state.history.valid.any()
        times = state.time[:, None]+batch.readback_offsets[:, 0]
        actual = core.complete(state, predicted, batch.traces[0], batch.paths[:, 0], times)
        assert actual.history.valid.sum() == 2 and actual.feedback_available.all()
        assert actual.memory is not memory and actual.memory.available.all()
        for field in ('features', 'q_at_image', 'image_times', 'valid', 'readback_indices'):
            # Actual completion advances only progress; it does not insert or
            # relabel a camera exposure when no new image arrived.
            assert getattr(actual.memory, field) is getattr(memory, field)
            torch.testing.assert_close(getattr(actual.memory, field), getattr(memory, field), atol=0, rtol=0)
        assert memory.current_readback_index.tolist() == [0, 0]
        assert actual.memory.current_readback_index.tolist() == [5, 5]
        assert not actual.measurement_available.any()
        refreshed = core.observe(actual, None, actual.q+.1, actual.time+.2)
        assert refreshed.memory is actual.memory and refreshed.history.valid.sum() == 2
        assert refreshed.memory.current_readback_index.tolist() == [5, 5]
        torch.testing.assert_close(refreshed.q_error, actual.q_error)
        with pytest.raises(ValueError, match="recorded"):
            core.complete(state, predicted, replace(batch.traces[0], source="planned"), batch.paths[:, 0], times)


def test_frozen_encoder_does_not_reencode_duplicate_or_missing_views():
    class Encoder(torch.nn.Module):
        def __init__(self): super().__init__(); self.count = 0
        def forward_features(self, x):
            self.count += len(x)
            return {"x_norm_patchtokens": torch.ones(len(x), 256, 384, device=x.device)}
    backbone = Encoder(); encoder = FrozenDINOFrames(backbone)
    rgb = torch.zeros(1, 2, 3, 224, 224, dtype=torch.uint8); q = torch.zeros(1, 2, 6)
    t = torch.full((1, 2), 10., dtype=torch.float64); now = t[:, 0]
    available = torch.ones(1, 2, dtype=torch.bool)
    first = encoder.encode(rgb, q, t, available, None, now)
    memory, _ = VisualMemory.empty(1).write(first, now)
    repeated = encoder.encode(rgb, q, t, available, memory, now+1)
    assert backbone.count == 2 and not repeated.measurement_available.any()
    latest = encoder.encode(rgb, q, t+1, torch.tensor([[True, False]]), memory, now+1)
    assert backbone.count == 3 and latest.measurement_available.tolist() == [[True, False]]


def test_runtime_stop_and_physical_rejection_before_arm(monkeypatch):
    vocabulary = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocabulary.words)))
    called = []
    def physical(q, a): called.append("physical"); raise RuntimeError("mechanical rejection")
    runtime = V21ResearchRuntime(core, vocabulary.encode(["pick red block"]), physical_check=physical,
                                lease_check=lambda: None, clock_fn=lambda: 10.)
    runtime.observe(EncodedEndpoint(torch.zeros(1, 6), torch.tensor([10.], dtype=torch.float64), observation()))
    assert runtime.stopped
    monkeypatch.setattr(core, "predict", lambda *a, **k: called.append("lewm"))
    with pytest.raises(RuntimeError, match="mechanical rejection"): runtime.plan()
    assert called == ["physical"] and runtime.stopped and runtime.pending is None


def test_native_vocabulary_is_compositional_and_strict():
    vocabulary = TaskVocabulary.build(["pick red block", "push red block", "夹起鱼放进盒子"])
    a = vocabulary.encode(["pick red block"]); b = vocabulary.encode(["push red block"])
    assert a[0, 1] != b[0, 1] and torch.equal(a[:, 2:], b[:, 2:])
    assert 1 in vocabulary.encode(["unseen noun"])
    with pytest.raises(ValueError, match="too long"): vocabulary.encode(["word "*70])


def test_stage_gradients_and_effective_batch_sigreg(tmp_path):
    vocabulary = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocabulary.words)))
    trainer = V21Trainer(core, vocabulary, tmp_path, microbatch=1, smoke=True)
    chain = synthetic_chain(vocabulary, batch=4)
    opt = trainer.configure("lewm")
    report = trainer.step([replace(chain, features=None)], opt, "lewm")
    assert report["independent_chain_rows"] == 4
    assert report["source_coverage"]["eligible_rows"] == 4
    assert all(p.grad is None for p in core.reader.parameters())
    assert all(p.grad is None for p in core.actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.projector.parameters())
    opt = trainer.configure("actor"); trainer.step([chain], opt, "actor")
    assert all(p.grad is None for p in core.lewm.parameters())
    assert all(p.grad is None for p in trainer.projector.parameters())
    assert core.reader.address[0].weight.grad.abs().sum() > 0


def test_checkpoint_contract_smoke_and_cpu_rng(tmp_path, monkeypatch):
    vocabulary = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocabulary.words))); schedule = V21Schedule()
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("CPU checks must not touch GPUs"))
    assert rng_state(torch.device("cpu"))["cuda"] is None
    path = tmp_path/"final.pt"
    save_checkpoint(path, core, vocabulary, schedule, stage="actor", step=1, complete=True, smoke=True)
    with pytest.raises(ValueError, match="smoke"): load_checkpoint(path, core, vocabulary)
    payload = load_checkpoint(path, core, vocabulary, allow_smoke=True)
    assert payload["contract"] == CONTRACT
    payload["contract"] = "v21-image-first-column-runtime-v1"; torch.save(payload, path)
    with pytest.raises(ValueError, match="contract"): load_checkpoint(path, core, vocabulary, allow_smoke=True)


def test_checkpoint_version_alias_preserves_weights_and_rejects_unknown(tmp_path):
    vocabulary = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocabulary.words)))
    path = tmp_path / "final.pt"
    save_checkpoint(path, core, vocabulary, V21Schedule(), stage="actor", step=1,
                    complete=True, smoke=False)
    payload = torch.load(path, weights_only=False)
    assert payload["version"] == "21.0.2"
    payload["version"] = "21.0.1"
    torch.save(payload, path)
    original = path.read_bytes()
    target = V21ResearchCore(core.config)
    loaded = load_checkpoint(path, target, vocabulary)
    assert loaded["version"] == "21.0.1"
    assert path.read_bytes() == original
    for key, value in core.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])
    payload["version"] = "21.0.3"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(path, target, vocabulary)
