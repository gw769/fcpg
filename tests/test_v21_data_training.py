from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch

from fcpg.v21_data import V21Sources, TaskVocabulary
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_memory import VisualMemory, VisualObservation
from fcpg.v21_training import V21Trainer, synthetic_chain, event_weights
from fcpg.v21_runtime import V21ResearchRuntime, EncodedEndpoint
from fcpg.prod_v20_1_sources import V201FishTrainingSource


@pytest.fixture(autouse=True)
def seeded():
    torch.set_num_threads(2); torch.manual_seed(21)


def test_fish_loader_recovers_all_recorded_commands_without_episode_2():
    fish = V201FishTrainingSource.__new__(V201FishTrainingSource)
    fish.episodes = np.repeat(np.arange(4), 61)
    fish.frame_indices = np.tile(np.arange(61), 4)
    fish.q = np.repeat(np.arange(244, dtype=np.float32)[:, None]/10, 6, axis=1)
    fish.actions = fish.q + .1
    fish.q[fish.episodes == 2] = np.nan; fish.actions[fish.episodes == 2] = np.nan
    fish.actions[1, 5] = 55.  # an intermediate command absent from 5-step BC targets
    fish.starts = np.asarray([0, 61, 183])
    source = V21Sources.__new__(V21Sources)
    source.fish = fish; source.fish_source = 0; source.fish_task = "pick red block"
    source.device = torch.device("cpu"); source.counts = dict(fish=3)
    source.vocabulary = TaskVocabulary.build([source.fish_task]); source.fish_dino = object()
    accessed = []
    def features(_cache, ids):
        fish.validate_frames(ids); accessed.extend(ids.tolist())
        return np.ones((len(ids), 2, 256, 384), np.float16), np.ones((len(ids), 2), bool)
    source._feature_values = features
    batch = source.chain([0, 1, 2], "fish")
    assert batch.traces[0].positions.shape == (3, 15, 6)
    assert batch.traces[0].positions[0, 1, 5] == 55.
    assert not (batch.commands[0, 0, :, 5] == 55).any()
    assert batch.features.shape == (3, 5, 2, 256, 384)
    assert set(fish.episodes[accessed]) == {0, 1, 3}
    assert torch.isfinite(batch.q).all()
    assert batch.readback_offsets.dtype == torch.float64


def test_future_inputs_cannot_change_earlier_actor_output():
    vocabulary = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocabulary.words))).eval()
    batch = synthetic_chain(vocabulary, batch=1)
    def first(x):
        obs = VisualObservation(x.features[:, 0], x.q[:, 0, None].expand(-1, 2, -1),
                                x.endpoint_times[:, :1].expand(-1, 2), x.view_valid[:, 0])
        state = core.observe(None, obs, x.q[:, 0], x.endpoint_times[:, 0])
        return core.act(state, x.text_ids, x.readback_offsets[:, 0, :5]).commands_position
    changed = batch.features.clone(); changed[:, 1:] *= 1000
    with torch.no_grad():
        torch.testing.assert_close(first(batch), first(replace(batch, features=changed)), atol=0, rtol=0)


def test_forecast_owns_its_trace_and_rejects_changed_commands():
    vocabulary = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocabulary.words))).eval()
    batch = synthetic_chain(vocabulary, batch=1, images=False)
    state = core.initial_state(batch.q[:, 0], batch.endpoint_times[:, 0])
    with torch.no_grad():
        forecast = core.predict(state, batch.traces[0], batch.readback_offsets[:, 0, 1:])
        batch.traces[0].positions.add_(1.)
        assert not torch.equal(forecast.trace.positions, batch.traces[0].positions)
        with pytest.raises(ValueError, match="command values"):
            core.complete(state, forecast, batch.traces[0], batch.paths[:, 0], batch.readback_offsets[:, 0])


def test_actor_reduction_is_independent_of_microbatch(tmp_path):
    vocabulary = TaskVocabulary.build(["pick red block"])
    first = V21ResearchCore(V21Config(len(vocabulary.words)))
    second = V21ResearchCore(V21Config(len(vocabulary.words))); second.load_state_dict(first.state_dict())
    batch = synthetic_chain(vocabulary, batch=4)
    a = V21Trainer(first, vocabulary, tmp_path/"a", microbatch=1, smoke=True)
    b = V21Trainer(second, vocabulary, tmp_path/"b", microbatch=4, smoke=True)
    ra = a.step([batch], a.configure("actor"), "actor")
    rb = b.step([batch], b.configure("actor"), "actor")
    assert ra["loss"] == pytest.approx(rb["loss"], rel=2e-5, abs=1e-7)
    torch.testing.assert_close(first.actor.output.weight.grad, second.actor.output.weight.grad, rtol=2e-4, atol=2e-6)


def test_oom_retry_preserves_the_same_batch_and_restores_gradients(tmp_path, monkeypatch):
    import fcpg.v21_training as training
    vocabulary = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocabulary.words)))
    batch = synthetic_chain(vocabulary, batch=4)
    cleared = []; calls = []
    trainer = V21Trainer(core, vocabulary, tmp_path, microbatch=2, smoke=True, clear_cache=lambda: cleared.append(True))
    original = training.chain_forward
    def sometimes(*args, **kwargs):
        calls.append(len(args[1].q))
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("synthetic retry test")
        return original(*args, **kwargs)
    monkeypatch.setattr(training, "chain_forward", sometimes)
    result = trainer.step([batch], trainer.configure("actor"), "actor")
    assert cleared == [True] and result["microbatch"] == 1
    assert result["chains"] == 4 and result["pair_exposures"] == 16
    assert calls == [2, 2, 1, 1, 1, 1]


def test_runtime_completion_uses_later_actual_image_clock_and_preserves_stop():
    vocabulary = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocabulary.words)))
    now = [10.]
    runtime = V21ResearchRuntime(core, vocabulary.encode(["pick red block"]),
        physical_check=lambda q, u: None, lease_check=lambda: None, clock_fn=lambda: now[0])
    q = torch.zeros(1, 6)
    observation = VisualObservation(torch.randn(1, 2, 256, 384), q[:, None].expand(-1, 2, -1),
        torch.full((1, 2), 10., dtype=torch.float64), torch.ones(1, 2, dtype=torch.bool))
    runtime.observe(EncodedEndpoint(q, torch.tensor([10.], dtype=torch.float64), observation))
    assert runtime.stopped
    runtime.plan(); now[0] = 10.01
    commands = runtime.begin_execution(q, torch.tensor([10.01], dtype=torch.float64))
    actual = replace(runtime.pending.trace, source="recorded")
    times = torch.tensor([[10.01, 10.11, 10.21, 10.31, 10.41, 10.51]], dtype=torch.float64)
    path = torch.cat((q[:, None], commands), 1)
    q_image = commands[:, -1]+.1; now[0] = 10.53
    new_obs = replace(observation, q_at_image=q_image[:, None].expand(-1, 2, -1),
                      image_times=torch.full((1, 2), 10.53, dtype=torch.float64))
    runtime.complete(actual, path, times, EncodedEndpoint(q_image, torch.tensor([10.53], dtype=torch.float64), new_obs))
    assert runtime.state.feedback_time.item() == pytest.approx(10.51)
    assert runtime.state.time.item() == pytest.approx(10.53)
    assert runtime.state.history.valid.sum() == 1 and runtime.pending is None
    runtime.stop()
    now[0] = 10.54
    runtime.observe(EncodedEndpoint(q_image, torch.tensor([10.54], dtype=torch.float64), None))
    assert runtime.stopped and runtime.state.memory.available.all()


def test_implementation_cli_does_not_start_formal_training_without_opt_in(tmp_path):
    from scripts.train_so101_v21_research import main, configuration
    with pytest.raises(ValueError, match="--run-formal"):
        main(["all", "--output", str(tmp_path)])
    (tmp_path/"user-file.txt").write_text("preserve")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        configuration(tmp_path, {"contract": "test"})
    assert (tmp_path/"user-file.txt").read_text() == "preserve"
