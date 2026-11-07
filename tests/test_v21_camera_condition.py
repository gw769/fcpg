"""Offline preparation contracts; CPU synthetic tensors, no checkpoint or IO."""
from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch

from fcpg.v21_camera_condition import CONDITIONS, V21CameraCondition
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_memory import VisualObservation
from fcpg.v21_runtime import EncodedEndpoint, V21ResearchRuntime


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        yield
    torch.set_num_threads(threads)


def observation(stamp=10., *, valid=(True, True), batch=1, marker=1.):
    q = torch.tensor([20., 10., 0., 0., 0., 40.]).repeat(batch, 1)
    return VisualObservation(
        torch.full((batch, 2, 256, 384), marker), q[:, None].repeat(1, 2, 1),
        torch.tensor([[stamp - .02, stamp]], dtype=torch.float64).repeat(batch, 1),
        torch.tensor(valid, dtype=torch.bool).reshape(-1, 2).expand(batch, 2).clone(),
        object_relative=torch.tensor([[1., .8, .3, -.2, .4, .15, -.8, .6]]).repeat(batch, 1))


EXPECTED = {
    "both_fresh": ((True, True), (True, True)),
    "wrist_h0": ((True, True), (False, True)),
    "wrist_cold": ((False, True), (False, True)),
    "external_h0": ((True, True), (True, False)),
    "external_cold": ((True, False), (True, False)),
}


@pytest.mark.parametrize("name", CONDITIONS)
def test_h0_h1_masks_owned_values_and_immutable_preview(name):
    source = observation()
    copies = {f.name: getattr(source, f.name).clone() for f in fields(source)
              if isinstance(getattr(source, f.name), torch.Tensor)}
    condition = V21CameraCondition(name)
    preview, candidate = condition.filter(source)
    seed, committed = condition.filter(source)
    assert not condition.h0_committed and candidate.h0_committed and committed.h0_committed
    assert candidate == committed and torch.equal(preview.features, seed.features)
    with pytest.raises(FrozenInstanceError):
        condition.name = "both_fresh"
    h1, again = committed.filter(observation(11., marker=2.))
    assert again is committed
    for value, expected in ((seed, EXPECTED[name][0]), (h1, EXPECTED[name][1])):
        assert value.measurement_available.tolist() == [list(expected)]
        missing = ~value.measurement_available
        for key in ("features", "q_at_image", "image_times"):
            assert not getattr(value, key)[missing].count_nonzero()
        if not expected[0]:
            assert not value.object_relative.count_nonzero()
    for key, before in copies.items():
        assert torch.equal(getattr(source, key), before)
        assert getattr(seed, key).data_ptr() != getattr(source, key).data_ptr()
    available = seed.measurement_available
    assert torch.equal(seed.image_times[available], source.image_times[available])
    assert torch.equal(seed.q_at_image[available], source.q_at_image[available])


@pytest.mark.parametrize("name", CONDITIONS)
@pytest.mark.parametrize("valid", [(False, True), (True, False), (False, False)])
def test_every_formal_seed_requires_both_actual_deliveries(name, valid):
    condition = V21CameraCondition(name)
    with pytest.raises(ValueError, match="H0 seed"):
        condition.filter(observation(valid=valid))
    assert not condition.h0_committed


@pytest.mark.parametrize("name", CONDITIONS)
def test_later_natural_missing_views_never_become_fabricated_frames(name):
    _, committed = V21CameraCondition(name).filter(observation())
    raw = observation(11., valid=((True, False), (False, True)), batch=2)
    raw.features[~raw.measurement_available] = float("nan")
    raw.q_at_image[~raw.measurement_available] = float("nan")
    raw.image_times[~raw.measurement_available] = float("nan")
    result, _ = committed.filter(raw)
    assert not (result.measurement_available & ~raw.measurement_available).any()
    for value in (result.features, result.q_at_image, result.image_times, result.object_relative):
        assert torch.isfinite(value).all()
    assert not result.object_relative[~result.measurement_available[:, 0]].count_nonzero()


@pytest.mark.parametrize("replacement", [
    {"source": "predicted"}, {"view_order": ("column", "wrist")},
    {"measurement_available": torch.ones(1, 2)},
    {"image_times": torch.ones(1, 2)},
    {"q_at_image": torch.zeros(1, 2, 5)},
    {"object_relative": torch.full((1, 8), float("nan"))},
])
def test_invalid_source_order_shapes_clocks_and_cues_are_rejected(replacement):
    with pytest.raises(ValueError):
        V21CameraCondition().filter(replace(observation(), **replacement))


def test_absent_optional_cue_remains_absent():
    value, _ = V21CameraCondition("wrist_cold").filter(replace(observation(), object_relative=None))
    assert value.object_relative is None


def test_configure_only_new_eval_core_without_weight_changes_or_hooks():
    core = V21ResearchCore(V21Config(4))
    condition = V21CameraCondition("wrist_h0", "zero")
    with pytest.raises(ValueError, match="eval"):
        condition.configure_reader(core)
    core.eval()
    parameters = {key: value.clone() for key, value in core.state_dict().items()}
    condition.configure_reader(core)
    assert core.reader.mode == "temporal" and core.config.memory_mode == "readback"
    assert all(torch.equal(value, core.state_dict()[key]) for key, value in parameters.items())
    assert not core.reader._forward_pre_hooks and not core.reader.address._forward_pre_hooks
    state = core.initial_state(torch.zeros(1, 6), torch.tensor([10.], dtype=torch.float64))
    with pytest.raises(ValueError, match="actual state"):
        condition.configure_reader(core, state=state)
    _, committed = condition.filter(observation())
    with pytest.raises(ValueError, match="H0 commit"):
        committed.configure_reader(core)
    with torch.inference_mode():
        core.observe(None, observation(), state.q, state.time)
    with pytest.raises(ValueError, match="actual state"):
        condition.configure_reader(core)


@pytest.mark.parametrize("name", CONDITIONS)
@pytest.mark.parametrize("address", ["factual", "zero"])
def test_actual_completion_refresh_dispatch_and_all_reader_inputs(name, address):
    condition = V21CameraCondition(name, address)
    core = V21ResearchCore(V21Config(4, object_relative=True)).eval().requires_grad_(False)
    condition.configure_reader(core)
    now = [10.]
    runtime = V21ResearchRuntime(core, torch.tensor([[2, 3]]),
        physical_check=lambda q, commands: None, lease_check=lambda: None,
        clock_fn=lambda: now[0], max_plan_age_s=2.)
    runtime.configure_source_update_tolerance(.25)
    phase = ["initial"]
    reader_calls, metadata, actor_contexts = {}, [], []
    # These hooks observe test evidence; the adapter installs no hooks.
    reader_hook = core.reader.register_forward_pre_hook(
        lambda module, args: reader_calls.__setitem__(phase[0], reader_calls.get(phase[0], 0) + 1))
    address_hook = core.reader.address.register_forward_pre_hook(
        lambda module, args: metadata.append(args[0].clone()))
    actor_hook = core.actor.register_forward_pre_hook(
        lambda module, args, kwargs: actor_contexts.append(kwargs["motion_context"].clone()), with_kwargs=True)

    def endpoint(stamp, q):
        raw = observation(stamp, marker=stamp)
        raw = replace(raw, q_at_image=q[:, None].repeat(1, 2, 1))
        filtered, following = condition.filter(raw)
        return EncodedEndpoint(q.clone(), torch.tensor([stamp], dtype=torch.float64), filtered), following

    try:
        with torch.inference_mode():
            q = torch.tensor([[20., 10., 0., 0., 0., 40.]])
            seed, following = endpoint(10., q)
            runtime.observe(seed)
            condition = following  # H0 accepted; later refreshes cannot replace it.
            original = runtime.state
            now[0] = 10.01
            refresh, _ = endpoint(now[0], q)
            phase[0] = "preview"
            runtime.preview(refresh)
            assert runtime.state is original and runtime.pending is None
            phase[0] = "plan"
            runtime.plan(endpoint=refresh)
            candidate = runtime._pending_state
            expected_context = core.reader.frame_metadata(candidate.memory, candidate.q, candidate.time, candidate.history)
            assert torch.equal(actor_contexts[-1], expected_context)
            assert actor_contexts[-1][..., :24].count_nonzero()
            phase[0] = "dispatch_drift"
            now[0] = 10.02
            runtime.begin_execution(q + .1, torch.tensor([now[0]], dtype=torch.float64))
            assert runtime.last_dispatch_update["commands_recomputed"]
            # Synthetic recorded readbacks, independent of the proposed actions.
            path = (q + .1)[:, None].repeat(1, 6, 1)
            times = torch.tensor([[10.02, 10.12, 10.22, 10.32, 10.42, 10.52]], dtype=torch.float64)
            trace = replace(runtime.pending.trace, source="recorded")
            phase[0] = "completion"
            now[0] = 10.7
            completed, _ = endpoint(now[0], q + .1)
            runtime.complete(trace, path, times, completed)
            after = runtime.state
            assert int(after.history.valid.sum()) == 1
            assert torch.equal(after.history.readbacks_position[:, -1], path)
            assert reader_calls == {"initial": 2, "preview": 1, "plan": 1, "dispatch_drift": 1, "completion": 3}
            assert len(metadata) == 7  # Empty initial memory returns before address MLP.
            if address == "zero":
                assert all(not value[..., :24].count_nonzero() for value in metadata)
            else:
                assert any(value[..., :24].count_nonzero() for value in metadata)
            if name != "both_fresh":
                missing = 0 if name.startswith("wrist_") else 1
                assert not after.measurement_available[0, missing]
                for key in ("features", "q_at_image", "image_times", "valid"):
                    assert torch.equal(getattr(after.memory, key)[:, missing], getattr(original.memory, key)[:, missing])
                assert int(after.memory.valid[:, missing].sum()) == int(name.endswith("_h0"))
            if name.startswith("wrist_"):
                assert not after.object_relative.count_nonzero()
                assert not candidate.object_relative.count_nonzero()
    finally:
        reader_hook.remove()
        address_hook.remove()
        actor_hook.remove()
