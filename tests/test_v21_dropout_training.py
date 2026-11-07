from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_memory import VisualObservation
from fcpg.v21_training import V21Trainer, chain_forward, synthetic_chain
from scripts.train_so101_v21_research import configuration, dropout_contract, measurement_dropout, parser


@pytest.fixture
def batch():
    torch.set_num_threads(2)
    return synthetic_chain(TaskVocabulary.build(["pick red block"]), batch=10)


def test_legacy_is_exactly_the_previous_mask_and_remains_default(batch):
    expected = batch.view_valid.clone()
    rows = np.arange(10)
    for h in (1, 2, 3):
        expected[rows % 4 == 1, h, h % 2] = False
        expected[rows % 4 == 2, h, 1-h % 2] = False
        if h == 2:
            expected[rows % 4 == 3, h] = False
    actual = measurement_dropout(batch, rows, variant=31)
    assert torch.equal(actual.view_valid, expected)
    assert parser().parse_args(["smoke", "--output", "/tmp/unused"]).measurement_dropout == "legacy"
    assert dropout_contract("legacy") == {}


def test_burst_and_cold_schedules_only_remove_actual_measurements(batch):
    original = batch.view_valid.clone()
    out = measurement_dropout(batch, np.arange(10), protocol="burst-cold-v1")
    assert out.view_valid[4, 0].all() and not out.view_valid[4, 1:, 1].any()
    assert out.view_valid[5, 0].all() and not out.view_valid[5, 1:, 0].any()
    assert not out.view_valid[6, :, 1].any() and out.view_valid[6, :, 0].all()
    assert not out.view_valid[7, :, 0].any() and out.view_valid[7, :, 1].all()
    assert not out.view_valid[8, 1:3].any() and out.view_valid[8, 3:].all()
    assert torch.equal(batch.view_valid, original)
    for field in fields(batch):
        if field.name != "view_valid":
            assert getattr(out, field.name) is getattr(batch, field.name)
    invalid = replace(batch, view_valid=torch.zeros_like(batch.view_valid))
    assert not measurement_dropout(invalid, np.arange(10), protocol="burst-cold-v1").view_valid.any()


def test_schedule_is_independent_of_batch_order_and_does_not_use_rng(batch):
    rng = torch.get_rng_state()
    order = np.arange(10)[::-1].copy()
    first = measurement_dropout(batch, np.arange(10), protocol="burst-cold-v1", variant=3)
    shuffled = measurement_dropout(batch, order, protocol="burst-cold-v1", variant=3)
    assert torch.equal(shuffled.view_valid, first.view_valid[order])
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.equal(first.view_valid, measurement_dropout(batch, np.arange(10), protocol="burst-cold-v1", variant=4).view_valid)


def test_warm_outage_carries_original_frame_but_cold_start_does_not_invent_it(batch):
    core = V21ResearchCore(V21Config(len(TaskVocabulary.build(["pick red block"]).words))).eval()
    out = measurement_dropout(batch, np.arange(10), protocol="burst-cold-v1")
    state = None
    with torch.inference_mode():
        for h in range(4):
            obs = VisualObservation(out.features[:, h], out.q[:, h, None].expand(-1,2,-1),
                                    out.endpoint_times[:, h, None].expand(-1,2), out.view_valid[:, h])
            state = core.observe(state, obs, out.q[:, h], out.endpoint_times[:, h])
        assert state.memory.available[4,1]
        assert state.memory.image_times[4,1,-1] == out.endpoint_times[4,0]
        assert not state.memory.available[6,1]
        assert state.memory.image_times[8,1,-1] == out.endpoint_times[8,3]


def test_new_protocol_cannot_overwrite_a_legacy_run(tmp_path):
    old = dict(contract="fixture")
    configuration(tmp_path, old)
    with pytest.raises(ValueError, match="immutable training contract"):
        configuration(tmp_path, dict(old, **dropout_contract("burst-cold-v1")))
    configuration(tmp_path, old)


def test_actor_step_with_missing_views_does_not_train_lewm(tmp_path, batch):
    vocab = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocab.words)))
    trainer = V21Trainer(core, vocab, tmp_path, microbatch=2, smoke=True)
    optimizer = trainer.configure("actor")
    arm = {name: value.detach().clone() for name, value in core.lewm.state_dict().items()}
    selected = measurement_dropout(batch, np.arange(10), protocol="burst-cold-v1").slice(4,9)
    metrics = trainer.step([selected], optimizer, "actor")
    assert np.isfinite(metrics["loss"])
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in core.reader.parameters())
    assert all(p.grad is None for p in core.lewm.parameters())
    for name, value in core.lewm.state_dict().items():
        assert torch.equal(value, arm[name])


def test_bad_rows_or_unknown_protocol_fail_before_training(batch):
    with pytest.raises(ValueError, match="aligned"):
        measurement_dropout(batch, np.arange(9), protocol="burst-cold-v1")
    with pytest.raises(ValueError, match="unknown"):
        measurement_dropout(batch, np.arange(10), protocol="mystery")


def test_hidden_camera_features_cannot_leak_into_actor_or_feedback(batch):
    vocab = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocab.words))).eval()
    selected = measurement_dropout(batch, np.arange(10), protocol="burst-cold-v1").slice(4,9)
    poisoned = selected.features.clone()
    poisoned[~selected.view_valid] = float('nan')
    outputs = []
    hook = core.actor.register_forward_hook(lambda _m, _i, result: outputs.append(result.commands_position.clone()))
    try:
        with torch.inference_mode():
            chain_forward(core, selected, "actor")
            reference = list(outputs); outputs.clear()
            chain_forward(core, replace(selected, features=poisoned), "actor")
        assert len(outputs) == len(reference) == 4
        for actual, expected in zip(outputs, reference):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        hook.remove()
