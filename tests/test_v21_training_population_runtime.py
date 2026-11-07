"""Load real checkpoint fixtures under legacy and original-split populations.

The fixtures exercise model loading, not training or physical execution.
"""
from dataclasses import asdict
import copy
import json

import pytest
import torch

from fcpg.v21_core import CONTRACT, V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_runtime import load_runtime
from fcpg.v21_training import V21Schedule, save_checkpoint


ORIGINAL_SPLIT = dict(
    schema="v21-original-source-split-v1",
    archive_train_rows=272000, archive_development_rows=48000,
    dense_train_rows=17000, dense_development_rows=3000,
    fish_train_episodes=[0, 1], fish_development_episodes=[3],
    fish_test_episode_used=False,
)


@pytest.fixture(scope="module")
def checkpoint_fixture(tmp_path_factory):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    vocabulary = TaskVocabulary.build(["pick red block"])
    core = V21ResearchCore(V21Config(len(vocabulary.words), object_relative=True))
    schedule = V21Schedule()
    path = tmp_path_factory.mktemp("training-population-checkpoint")/"final.pt"
    save_checkpoint(path, core, vocabulary, schedule, stage="fish",
                    step=schedule.fish_updates, complete=True, smoke=False)
    config = dict(contract=CONTRACT, smoke=False, model=asdict(core.config),
                  vocabulary=list(vocabulary.words), schedule=asdict(schedule))
    yield path, config, {key: value.clone() for key, value in core.state_dict().items()}
    torch.set_num_threads(previous_threads)


def make_run(tmp_path, checkpoint_fixture, *, original_split):
    path, config, _ = checkpoint_fixture
    config = copy.deepcopy(config)
    if original_split:
        config["training_data_contract"] = copy.deepcopy(ORIGINAL_SPLIT)
    # Independent expected totals for the unchanged original four-stage schedule.
    totals = ([4_352_000, 4_352_000, 68_000, 256_000] if original_split else
              [5_120_000, 5_120_000, 80_000, 256_000])
    metrics = {name: dict(complete=True, pair_exposures=count)
               for name, count in zip(("lewm", "actor", "dense", "fish"), totals)}
    (tmp_path/"fish").mkdir()
    (tmp_path/"fish/final.pt").symlink_to(path)
    write_run(tmp_path, config, metrics)
    return config, metrics


def write_run(path, config, metrics):
    (path/"config.json").write_text(json.dumps(config))
    (path/"metrics.json").write_text(json.dumps(metrics))


def load_model_only(path):
    def no_hardware(*args, **kwargs):
        pytest.fail("loading must not call hardware/lease callbacks")
    return load_runtime(path, "pick red block", device="cpu",
                        physical_check=no_hardware, lease_check=no_hardware)


@pytest.mark.parametrize("original_split", [False, True])
def test_complete_declared_populations_load_identical_stopped_model(
        tmp_path, checkpoint_fixture, original_split):
    make_run(tmp_path, checkpoint_fixture, original_split=original_split)
    runtime = load_model_only(tmp_path)
    assert runtime.stopped and runtime.pending is None and not runtime.executing
    for key, expected in checkpoint_fixture[2].items():
        assert torch.equal(runtime.core.state_dict()[key], expected)


@pytest.mark.parametrize("stage", ["lewm", "actor", "dense", "fish"])
def test_original_split_rejects_incomplete_stage(tmp_path, checkpoint_fixture, stage):
    config, metrics = make_run(tmp_path, checkpoint_fixture, original_split=True)
    metrics[stage]["pair_exposures"] -= 4
    write_run(tmp_path, config, metrics)
    with pytest.raises(ValueError, match=f"formal {stage} exposure/completion contract is incomplete"):
        load_model_only(tmp_path)


@pytest.mark.parametrize("original_split", [False, True])
def test_declared_population_cannot_borrow_other_recipes_exposures(
        tmp_path, checkpoint_fixture, original_split):
    config, metrics = make_run(tmp_path, checkpoint_fixture, original_split=original_split)
    metrics["lewm"]["pair_exposures"] = 5_120_000 if original_split else 4_352_000
    write_run(tmp_path, config, metrics)
    with pytest.raises(ValueError, match="formal lewm exposure/completion contract is incomplete"):
        load_model_only(tmp_path)


@pytest.mark.parametrize("change", [
    {"schema": "unknown-split"},
    {"archive_train_rows": 64},
    {"archive_train_rows": 272000.0},
    {"dense_development_rows": 0},
    {"fish_train_episodes": [False, True]},
    {"fish_development_episodes": [2]},
    {"fish_test_episode_used": 0},
    {"fish_test_episode_used": True},
])
def test_malformed_original_population_is_rejected(tmp_path, checkpoint_fixture, change):
    config, metrics = make_run(tmp_path, checkpoint_fixture, original_split=True)
    config["training_data_contract"].update(change)
    write_run(tmp_path, config, metrics)
    with pytest.raises(ValueError, match="unsupported formal training data population contract"):
        load_model_only(tmp_path)


@pytest.mark.parametrize("profile", [None, {}, [], "legacy"])
def test_explicit_invalid_profile_is_not_legacy(tmp_path, checkpoint_fixture, profile):
    config, metrics = make_run(tmp_path, checkpoint_fixture, original_split=False)
    config["training_data_contract"] = profile
    write_run(tmp_path, config, metrics)
    with pytest.raises(ValueError, match="unsupported formal training data population contract"):
        load_model_only(tmp_path)


def test_canary_config_cannot_claim_formal_population(tmp_path, checkpoint_fixture):
    config, metrics = make_run(tmp_path, checkpoint_fixture, original_split=True)
    config["smoke"] = True
    write_run(tmp_path, config, metrics)
    with pytest.raises(ValueError, match="only formal V21 research runs may load"):
        load_model_only(tmp_path)
