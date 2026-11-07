from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fcpg.v21_response_sampling import response_features, response_sampling, SAMPLING_MODES


def row(run="run-a", *, q3=50., command=1., actual=.5):
    q0 = [0., 0., q3, 0., 0., 0.]
    commands = [q0.copy() for _ in range(5)]
    measured = [q0.copy() for _ in range(6)]
    actual = [actual]*5 if np.isscalar(actual) else actual
    for i in range(5):
        commands[i][2] += command
        measured[i+1][2] += actual[i]
    return dict(path=f"/records/{run}/execution.json", source_q_position=q0,
                commands_position=commands, measured_q_path=measured)


@pytest.mark.parametrize("seed", [21, 42, 2026])
def test_equal_run_is_bitwise_identical_to_old_probability_and_draws(seed):
    rows = [row(run) for run in ["a"]*3+["b"]*7+["c"]*11+["held-out"]*4]
    indices = np.array([7, 2, 12, 0, 18, 4, 6, 5, 3, 9, 8, 1, 20, 10, 11, 13, 15, 16, 14, 17, 19])
    groups = np.array([Path(r["path"]).parent.name for r in rows])
    old_weights = np.array([1. / sum(groups[indices] == groups[index]) for index in indices])
    old = torch.tensor(old_weights / old_weights.sum(), dtype=torch.float64)
    probabilities, metadata = response_sampling(rows, indices, "equal_run")
    assert probabilities.dtype == torch.float64 and probabilities.device.type == "cpu"
    assert torch.equal(probabilities, old)
    assert torch.equal(torch.multinomial(old, 500, replacement=True,
                                       generator=torch.Generator().manual_seed(seed)),
                       torch.multinomial(probabilities, 500, replacement=True,
                                         generator=torch.Generator().manual_seed(seed)))
    assert metadata["train_indices"] == indices.tolist()
    assert set(metadata["runs"]) == {"a", "b", "c"}
    assert all(v["probability_mass"] == pytest.approx(1/3) for v in metadata["runs"].values())


@pytest.mark.parametrize("mode", SAMPLING_MODES)
def test_held_out_labels_and_even_missing_held_out_rows_do_not_affect_sampling(mode):
    rows = [row(actual=0.), row("never-read", actual=0.), row("b", actual=.6)]
    before = deepcopy(rows)
    probabilities, metadata = response_sampling(rows, [2, 0], mode)
    assert rows == before
    rows[1] = {"malformed": float("nan")}
    rows.append(None)
    changed_probabilities, changed_metadata = response_sampling(rows, [2, 0], mode)
    assert torch.equal(probabilities, changed_probabilities)
    assert metadata == changed_metadata
    json.dumps(metadata, allow_nan=False)


def test_balancing_assigns_half_mass_to_rare_weak_response_class():
    rows = [row(actual=.8) for _ in range(9)] + [row("rare", actual=.1)]
    p, metadata = response_sampling(rows, np.arange(10), "weak_response_balanced")
    assert p[-1].item() == pytest.approx(.5)
    assert p[:-1].tolist() == pytest.approx([.5/9]*9)
    assert metadata["strata"]["weak_response"] == dict(count=1, probability_mass=.5)
    assert metadata["strata"]["non_weak_response"]["count"] == 9
    assert metadata["balance_fallback"] is None


@pytest.mark.parametrize("weak", [True, False])
def test_one_available_stratum_is_uniform_and_explicit(weak):
    rows = [row(actual=.1 if weak else .8) for _ in range(7)]
    p, metadata = response_sampling(rows, np.arange(7), "weak_response_balanced")
    assert p.tolist() == pytest.approx([1/7]*7)
    missing = "non_weak_response" if weak else "weak_response"
    assert metadata["missing_strata"] == [missing]
    assert metadata["balance_fallback"] == "uniform_over_only_observed_stratum"
    assert metadata["strata"][missing] == dict(count=0, probability_mass=0.)


def test_whole_trajectory_criterion_strict_boundary_and_no_q0_filter():
    rows = [
        row(q3=-50., command=-1., actual=[0., .25, -.1, .2, 0.]),
        row(q3=99., command=-1., actual=[0., .5, .2, .1, 0.]),
        row(command=.25, actual=0.),
        row(command=.251, actual=0.),
        row(command=1., actual=.250001),
    ]
    _, metadata = response_sampling(rows, np.arange(5), "weak_response_balanced")
    assert [s["weak_response"] for s in metadata["samples"]] == [True, False, False, True, False]
    assert metadata["weak_response_definition"]["whole_trajectory"] is True
    assert metadata["weak_response_definition"]["q0_range_filter"] is False


def test_equal_sample_retains_input_order_and_uniform_probability():
    rows = [row("a"), row("a"), row("b", actual=0.)]
    p, metadata = response_sampling(rows, torch.tensor([2, 0, 1]), "equal_sample")
    assert torch.equal(p, torch.full((3,), 1/3, dtype=torch.float64))
    assert [s["train_index"] for s in metadata["samples"]] == [2, 0, 1]
    assert metadata["runs"]["a"]["probability_mass"] == pytest.approx(2/3)
    assert metadata["runs"]["b"]["probability_mass"] == pytest.approx(1/3)


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [1], [.0], [True], [[0]], [2**64-1]])
def test_invalid_training_indices_fail_closed(indices):
    with pytest.raises(ValueError, match="training"):
        response_sampling([row()], indices, "equal_run")


def test_unknown_mode_is_rejected_before_rows_are_read():
    with pytest.raises(ValueError, match="unknown response sampling mode"):
        response_sampling([None], [0], "auto")


@pytest.mark.parametrize("field,value", [
    ("source_q_position", [0.]*5),
    ("commands_position", [[0.]*6]*4),
    ("measured_q_path", [[0.]*6]*5),
    ("source_q_position", [0., 0., float("nan"), 0., 0., 0.]),
])
def test_invalid_selected_labels_are_rejected(field, value):
    example = row()
    example[field] = value
    with pytest.raises(ValueError, match="training"):
        response_sampling([example], [0], "equal_sample")


def test_inconsistent_source_pose_is_rejected():
    example = row()
    example["measured_q_path"][0][2] += 1.
    with pytest.raises(ValueError, match="source differs"):
        response_sampling([example], [0], "equal_run")


def test_ratio_stratum_catches_large_command_with_motion_just_above_absolute_threshold():
    rows = [row(command=-3., actual=-.26619720458984375), row(command=-3., actual=-2.9)]
    _, old = response_sampling(rows, [0, 1], 'weak_response_balanced')
    p, new = response_sampling(rows, [0, 1], 'response_ratio_balanced')
    assert old['samples'][0]['weak_response'] is False
    assert new['samples'][0]['weak_response'] is True
    assert new['samples'][0]['directional_response_ratio'] == pytest.approx(.26619720458984375/3.)
    assert new['samples'][1]['weak_response'] is False
    assert p.tolist() == pytest.approx([.5, .5])


def test_ratio_has_correct_direction_zero_motion_and_whole_trajectory_semantics():
    wrong = response_features(row(command=-3., actual=1.))
    assert wrong['directional_response_ratio'] == 0. and wrong['low_ratio_response']
    zero = response_features(row(command=0., actual=0.))
    assert zero['directional_response_ratio'] is None and not zero['ratio_or_absolute_weak_response']
    returned = response_features(row(command=3., actual=[0., 2., 0., 0., 0.]))
    assert not returned['ratio_or_absolute_weak_response']
    boundary = response_features(row(command=-3., actual=-.75))
    assert boundary['low_ratio_response']


def test_ratio_sampling_preserves_absolute_low_response_membership():
    rows = [row(command=.26, actual=.24), row(command=3., actual=.7), row(command=3., actual=2.9)]
    _, new = response_sampling(rows, [0, 1, 2], 'response_ratio_balanced')
    assert [s['weak_response'] for s in new['samples']] == [True, True, False]
    assert [s['legacy_weak_response'] for s in new['samples']] == [True, False, False]
