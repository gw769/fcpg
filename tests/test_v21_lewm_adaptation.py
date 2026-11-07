from copy import deepcopy
import hashlib

import pytest
import torch

from fcpg.prod_v20_policy import empty_history
from fcpg.v21_models import CommandTrace, V21LeWM
from fcpg.v21_lewm_adaptation import (
    LeWMAdaptationError, MEAN_PARAMETER_NAMES, PARENT_CONTRACTS, VISION_PARAMETER_NAMES, load_lewm_shadow_candidate,
)


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def bundle(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        parent = V21LeWM()
    state = {name: value.detach().clone() for name, value in parent.state_dict().items()}
    checkpoint = tmp_path / "parent.pt"
    torch.save({"model": {"lewm."+name: value for name, value in state.items()}}, checkpoint)
    parent_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    parameters = {name: state[name].clone() for name in MEAN_PARAMETER_NAMES}
    parameters["response.bias"] += .1
    payload = dict(parent_sha256=parent_sha, method="lewm_mean", seed=21,
                   held_out_run="reviewed_recent_four", requires_independent_validation=True,
                   parameters=parameters)
    candidate = tmp_path / "candidate.pt"
    torch.save(payload, candidate)
    return dict(parent=parent, state=state, checkpoint=checkpoint, parent_sha=parent_sha,
                candidate=candidate, payload=payload)


def load(bundle):
    return load_lewm_shadow_candidate(bundle["parent"], parent_checkpoint=bundle["checkpoint"],
        candidate_path=bundle["candidate"], expected_parent_sha256=bundle["parent_sha"])


def assert_parent_unchanged(bundle):
    assert all(torch.equal(value, bundle["parent"].state_dict()[name])
               for name, value in bundle["state"].items())
    assert hashlib.sha256(bundle["checkpoint"].read_bytes()).hexdigest() == bundle["parent_sha"]


def test_loads_exact_candidate_only_into_independent_eval_model(bundle):
    parent = bundle["parent"]
    rng_before = torch.random.get_rng_state().clone()
    result = load(bundle)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert result.model is not parent and result.model.training is False
    assert parent.training is True and all(p.requires_grad for p in parent.parameters())
    assert not any(p.requires_grad for p in result.model.parameters())
    assert result.requires_independent_validation is True
    assert result.seed == 21 and result.method == "lewm_mean"
    assert len(result.parameter_names) == 43
    assert sum(dict(result.model.named_parameters())[n].numel() for n in result.parameter_names) == 992646
    assert result.parent_sha256 == bundle["parent_sha"]
    assert result.candidate_sha256 == hashlib.sha256(bundle["candidate"].read_bytes()).hexdigest()
    for name, value in result.model.state_dict().items():
        expected = bundle["payload"]["parameters"].get(name, bundle["state"][name])
        assert torch.equal(value, expected)
        assert value.device == parent.state_dict()[name].device
        assert value.dtype == parent.state_dict()[name].dtype
        assert value.data_ptr() != parent.state_dict()[name].data_ptr()
    with torch.no_grad():
        result.model.response.bias.add_(1)
    assert_parent_unchanged(bundle)


def test_shadow_inference_preserves_source_and_command_causality(bundle):
    shadow = load(bundle).model
    q0 = torch.tensor([[0., -80., 90., 50., -40., 20.]])
    commands = q0[:, None].expand(-1, 5, -1).clone()
    commands[:, :, 2] -= torch.arange(1, 6)
    trace = CommandTrace.policy(commands, torch.full((1, 5), .1))
    query = torch.arange(1, 6, dtype=torch.float64)[None]*.1
    with torch.inference_mode():
        before = bundle["parent"].eval()(q0, trace, empty_history(1), query)
        output = shadow(q0, trace, empty_history(1), query)
        changed = commands.clone()
        changed[:, -1] += 10
        later = shadow(q0, CommandTrace(changed, trace.offsets, trace.valid, "planned", True),
                       empty_history(1), query)
    assert torch.equal(output.predicted_q_path[:, 0], q0)
    assert torch.isfinite(output.predicted_q_path).all()
    assert torch.isfinite(output.q_uncertainty).all() and (output.q_uncertainty > 0).all()
    assert not torch.equal(output.predicted_q_path[:, 1:], before.predicted_q_path[:, 1:])
    torch.testing.assert_close(output.predicted_q_path[:, :5], later.predicted_q_path[:, :5], rtol=0, atol=0)
    assert_parent_unchanged(bundle)


@pytest.mark.parametrize("damage,match", [
    ("missing", "parameter names"), ("extra", "parameter names"),
    ("scale", "parameter names"), ("buffer", "parameter names"),
    ("shape", "shape mismatch"), ("dtype", "dtype mismatch"),
    ("nan", "nonfinite"), ("inf", "nonfinite"), ("not_tensor", "dense tensor"),
    ("method", "lewm_mean"), ("seed", "seed"), ("held_out", "held_out_run"),
    ("validation", "independent-validation"), ("parent_sha", "candidate parent"),
    ("payload_key", "payload keys"),
])
def test_invalid_candidate_is_rejected_without_parent_changes(bundle, damage, match):
    payload = deepcopy(bundle["payload"])
    parameters = payload["parameters"]
    if damage == "missing":
        del parameters["response.bias"]
    elif damage in ("extra", "scale", "buffer"):
        name = {"extra": "actor.output.bias", "scale": "scale.bias", "buffer": "increment_scale"}[damage]
        parameters[name] = torch.zeros(6)
    elif damage == "shape":
        parameters["response.bias"] = torch.zeros(7)
    elif damage == "dtype":
        parameters["response.bias"] = parameters["response.bias"].double()
    elif damage in ("nan", "inf"):
        parameters["response.bias"][0] = float(damage)
    elif damage == "not_tensor":
        parameters["response.bias"] = [0.]*6
    elif damage == "method":
        payload["method"] = "head_only"
    elif damage == "seed":
        payload["seed"] = True
    elif damage == "held_out":
        payload["held_out_run"] = ""
    elif damage == "validation":
        payload["requires_independent_validation"] = False
    elif damage == "parent_sha":
        payload["parent_sha256"] = "0"*64
    elif damage == "payload_key":
        payload["extra"] = 1
    torch.save(payload, bundle["candidate"])
    with pytest.raises(LeWMAdaptationError, match=match):
        load(bundle)
    assert_parent_unchanged(bundle)


def test_parent_digest_is_supplied_independently(bundle):
    bundle["parent_sha"] = "0"*64
    with pytest.raises(LeWMAdaptationError, match="parent checkpoint SHA-256"):
        load(bundle)


@pytest.mark.parametrize("name", ["response.bias", "scale.bias", "increment_scale"])
def test_already_changed_parent_is_not_silently_rebased(bundle, name):
    with torch.no_grad():
        bundle["parent"].state_dict()[name].add_(.25)
    before = {k: v.clone() for k, v in bundle["parent"].state_dict().items()}
    with pytest.raises(LeWMAdaptationError, match="in-memory parent differs"):
        load(bundle)
    assert all(torch.equal(v, bundle["parent"].state_dict()[k]) for k, v in before.items())


def test_parent_state_schema_must_match_the_complete_checkpoint(bundle):
    torch.save({"model": {"lewm.response.bias": torch.zeros(6)}}, bundle["checkpoint"])
    bundle["parent_sha"] = hashlib.sha256(bundle["checkpoint"].read_bytes()).hexdigest()
    bundle["payload"]["parent_sha256"] = bundle["parent_sha"]
    torch.save(bundle["payload"], bundle["candidate"])
    with pytest.raises(LeWMAdaptationError, match="parent checkpoint LeWM state names"):
        load(bundle)


def test_corrupt_candidate_does_not_fallback_to_parent(bundle):
    bundle["candidate"].write_bytes(b"not a checkpoint")
    with pytest.raises(LeWMAdaptationError, match="cannot decode candidate"):
        load(bundle)
    assert_parent_unchanged(bundle)


def test_parent_contract_is_exact_and_recorded(bundle):
    assert load(bundle).parent_contract == "v21-lewm-vision-v2"
    assert len(VISION_PARAMETER_NAMES) == 60 and len(PARENT_CONTRACTS["v21-lewm-vision-v2"]) == 105
    assert PARENT_CONTRACTS["v21-lewm-vision-v2"] == set(dict(bundle["parent"].named_parameters()))


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_parent_with_another_parameter_set_is_refused(bundle, change):
    if change == "extra":
        bundle["parent"].register_parameter("extra", torch.nn.Parameter(torch.zeros(1)))
    else:
        bundle["parent"].vision_read[1].register_parameter("q_proj_weight", None)
    with pytest.raises(LeWMAdaptationError, match="parameter schema differs"):
        load(bundle)


def test_visual_parameters_must_equal_the_verified_checkpoint(bundle):
    with torch.no_grad():
        bundle["parent"].vision_layers[0].norm1.weight.add_(.25)
    with pytest.raises(LeWMAdaptationError, match="in-memory parent differs from checkpoint: vision_layers.0.norm1"):
        load(bundle)


def test_a_candidate_cannot_carry_visual_parameters(bundle):
    name = "vision_layers.0.norm1.weight"
    bundle["payload"]["parameters"][name] = bundle["state"][name].clone()
    torch.save(bundle["payload"], bundle["candidate"])
    with pytest.raises(LeWMAdaptationError, match="candidate mean parameter names"):
        load(bundle)

