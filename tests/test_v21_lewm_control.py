from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest
import torch

from fcpg.prod_v20_policy import empty_history
from fcpg.v21_core import V21Config
from fcpg.v21_models import V21LeWM, CommandTrace
from fcpg.v21_lewm_adaptation import MEAN_PARAMETER_NAMES
from fcpg.v21_lewm_shadow import LeWMShadowEvaluator
from fcpg.v21_lewm_control import (
    LeWMControllerError, prepare_lewm_controller, install_prepared_lewm_controller,
    lewm_state_sha256,
)


@pytest.fixture(autouse=True, scope='module')
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def bundle(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(53)
        parent = V21LeWM().eval().requires_grad_(False)
    root = tmp_path/'parent'
    (root/'fish').mkdir(parents=True)
    parent_path = root/'fish/final.pt'
    torch.save({'model': {'lewm.'+name: value.clone() for name, value in parent.state_dict().items()},
                'config': asdict(V21Config(vocabulary_size=401))}, parent_path)
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    parameters = {name: parent.state_dict()[name].clone() for name in MEAN_PARAMETER_NAMES}
    parameters['response.bias'] += .2
    payload = dict(parent_sha256=parent_sha, parameters=parameters, method='lewm_mean', seed=21,
                   held_out_run='reviewed_recent_four', requires_independent_validation=True)
    path = tmp_path/'candidate.pt'
    torch.save(payload, path)
    return dict(parent=parent, root=root, parent_path=parent_path, candidate_path=path,
                parent_sha=parent_sha, candidate_sha=hashlib.sha256(path.read_bytes()).hexdigest(),
                payload=payload)


def prepare(bundle):
    return prepare_lewm_controller(bundle['root'], bundle['candidate_path'],
                                    bundle['parent_sha'], bundle['candidate_sha'])


class SmallCore(torch.nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.lewm = deepcopy(parent)
        self.actor = torch.nn.Linear(3, 2)
        self.reader = torch.nn.Linear(4, 3)
        self.register_buffer('outside_scale', torch.arange(6.))


def non_lewm_state(core):
    return {k: v.clone() for k, v in core.state_dict().items() if not k.startswith('lewm.')}


def test_preflight_identity_cpu_eval_rng_and_file_integrity(bundle):
    rng = torch.random.get_rng_state().clone()
    cuda_initialized = torch.cuda.is_initialized()
    prepared = prepare(bundle)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.cuda.is_initialized() == cuda_initialized
    for model in (prepared.parent, prepared.candidate.model):
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        assert {v.device.type for v in model.state_dict().values()} == {'cpu'}
    identity = prepared.identity
    assert identity['source'] == 'adapted_lewm_mean' and identity['control_influence'] is True
    assert identity['experimental'] is True and identity['uncertainty_calibrated'] is False
    assert identity['requires_independent_validation'] is True
    assert identity['base_checkpoint'] == str(bundle['parent_path'].resolve())
    assert identity['candidate_checkpoint'] == str(bundle['candidate_path'].resolve())
    assert identity['parent_sha256'] == bundle['parent_sha']
    assert identity['candidate_sha256'] == bundle['candidate_sha']
    assert identity['controller_state_sha256'] == LeWMShadowEvaluator._state_sha(prepared.candidate.model)
    assert lewm_state_sha256(prepared.parent) == LeWMShadowEvaluator._state_sha(prepared.parent)
    json.dumps(identity, allow_nan=False)
    assert hashlib.sha256(bundle['parent_path'].read_bytes()).hexdigest() == bundle['parent_sha']
    assert hashlib.sha256(bundle['candidate_path'].read_bytes()).hexdigest() == bundle['candidate_sha']


@pytest.mark.parametrize('field', ['parent_sha', 'candidate_sha'])
def test_both_file_hashes_are_checked_before_any_torch_load(bundle, monkeypatch, field):
    bundle[field] = '0'*64
    def forbidden(*_args, **_kwargs): raise AssertionError('torch.load ran before both hashes passed')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(LeWMControllerError, match='SHA-256 mismatch'):
        prepare(bundle)


@pytest.mark.parametrize('damage,match', [
    ('shape', 'shape mismatch'), ('dtype', 'dtype mismatch'), ('missing', 'parameter names'),
    ('sigma', 'parameter names'), ('buffer', 'parameter names'), ('nan', 'nonfinite'),
])
def test_invalid_candidate_contract_fails_preflight_and_preserves_rng(bundle, damage, match):
    payload = deepcopy(bundle['payload'])
    if damage == 'shape': payload['parameters']['response.bias'] = torch.zeros(7)
    elif damage == 'dtype': payload['parameters']['response.bias'] = torch.zeros(6, dtype=torch.float64)
    elif damage == 'missing': del payload['parameters']['response.bias']
    elif damage == 'sigma': payload['parameters']['scale.bias'] = torch.zeros(6)
    elif damage == 'buffer': payload['parameters']['increment_scale'] = torch.ones(6)
    elif damage == 'nan': payload['parameters']['response.bias'][2] = float('nan')
    torch.save(payload, bundle['candidate_path'])
    bundle['candidate_sha'] = hashlib.sha256(bundle['candidate_path'].read_bytes()).hexdigest()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(LeWMControllerError, match=match): prepare(bundle)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_install_replaces_only_lewm_and_prediction_matches_verified_candidate(bundle):
    prepared = prepare(bundle)
    core = SmallCore(bundle['parent']).eval()
    original = core.lewm
    outside = non_lewm_state(core)
    source_hash = lewm_state_sha256(original)
    returned = install_prepared_lewm_controller(core, prepared)
    assert returned is original and core.lewm is not original
    assert core.lewm is not prepared.candidate.model
    assert all(torch.equal(v, core.state_dict()[k]) for k, v in outside.items())
    assert lewm_state_sha256(original) == source_hash
    assert lewm_state_sha256(core.lewm) == prepared.identity['controller_state_sha256']
    assert not core.lewm.training and not any(p.requires_grad for p in core.lewm.parameters())
    q = torch.tensor([[0., -70., 80., 50., -30., 25.]])
    commands = q[:, None].expand(-1, 5, -1).clone()
    commands[:, :, 2] -= torch.arange(1, 6)
    dt = torch.full((1, 5), .1)
    trace, queries = CommandTrace.policy(commands, dt), dt.double().cumsum(1)
    with torch.inference_mode():
        expected = prepared.candidate.model(q, trace, empty_history(1), queries)
        actual = core.lewm(q, trace, empty_history(1), queries)
    assert torch.equal(actual.predicted_q_path, expected.predicted_q_path)
    assert torch.equal(actual.q_uncertainty, expected.q_uncertainty)
    assert torch.equal(actual.predicted_q_path[:, 0], q)
    with torch.no_grad(): core.lewm.response.bias.add_(1.)
    assert lewm_state_sha256(prepared.candidate.model) == prepared.identity['controller_state_sha256']
    assert lewm_state_sha256(original) == source_hash
    assert hashlib.sha256(bundle['parent_path'].read_bytes()).hexdigest() == bundle['parent_sha']
    assert hashlib.sha256(bundle['candidate_path'].read_bytes()).hexdigest() == bundle['candidate_sha']


@pytest.mark.parametrize('name', ['response.bias', 'scale.bias', 'increment_scale'])
def test_in_memory_parent_mismatch_fails_before_swap(bundle, name):
    prepared = prepare(bundle)
    core = SmallCore(bundle['parent'])
    original = core.lewm
    with torch.no_grad(): core.lewm.state_dict()[name].add_(.1)
    before = {k: v.clone() for k, v in core.state_dict().items()}
    with pytest.raises(LeWMControllerError, match='in-memory parent state differs'):
        install_prepared_lewm_controller(core, prepared)
    assert core.lewm is original
    assert all(torch.equal(v, core.state_dict()[k]) for k, v in before.items())


def test_install_does_not_silently_cast_parent_dtype(bundle):
    prepared = prepare(bundle)
    core = SmallCore(bundle['parent'])
    core.lewm.double()
    original = core.lewm
    with pytest.raises(LeWMControllerError, match='dtype differs'):
        install_prepared_lewm_controller(core, prepared)
    assert core.lewm is original and core.lewm.response.weight.dtype == torch.float64


@pytest.mark.parametrize('part,name', [('parent', 'response.bias'), ('candidate', 'scale.bias'),
                                     ('candidate', 'increment_scale')])
def test_mutation_after_preflight_is_rejected(bundle, part, name):
    prepared = prepare(bundle)
    core = SmallCore(bundle['parent'])
    original = core.lewm
    model = prepared.parent if part == 'parent' else prepared.candidate.model
    with torch.no_grad(): model.state_dict()[name].add_(.1)
    with pytest.raises(LeWMControllerError, match=f'prepared {part} state changed'):
        install_prepared_lewm_controller(core, prepared)
    assert core.lewm is original


@pytest.mark.parametrize('field,value', [('experimental', False), ('base_checkpoint', '/changed/base.pt'),
                                       ('candidate_checkpoint', '/changed/candidate.pt')])
def test_identity_cannot_be_changed_after_preflight(bundle, field, value):
    prepared = prepare(bundle)
    prepared.identity[field] = value
    core = SmallCore(bundle['parent'])
    original = core.lewm
    with pytest.raises(LeWMControllerError, match='identity changed'):
        install_prepared_lewm_controller(core, prepared)
    assert core.lewm is original


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_is_built_from_its_own_checkpoint_config(bundle):
    prepared = prepare(bundle)
    assert prepared.parent.horizon == 5 and prepared.identity['parent_contract'] == 'v21-lewm-vision-v2'
    payload = torch.load(bundle['parent_path'], map_location='cpu', weights_only=False)
    for horizon, match in ((None, 'no model config'), (40, 'controller preflight failed')):
        damaged = dict(payload)
        if horizon is None:
            del damaged['config']
        else:                                   # a config that does not fit the saved tensors
            damaged['config'] = dict(payload['config'], prediction_horizon=horizon)
        torch.save(damaged, bundle['parent_path'])
        torch.save(dict(bundle['payload'], parent_sha256=sha(bundle['parent_path'])), bundle['candidate_path'])
        with pytest.raises(LeWMControllerError, match=match):
            prepare_lewm_controller(bundle['root'], bundle['candidate_path'], sha(bundle['parent_path']),
                                    sha(bundle['candidate_path']))


def test_a_horizon_40_parent_is_built_with_horizon_40(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(54)
        parent = V21LeWM(horizon=40).eval().requires_grad_(False)
    root = tmp_path/'parent'
    (root/'fish').mkdir(parents=True)
    parent_path = root/'fish/final.pt'
    torch.save({'model': {'lewm.'+name: value.clone() for name, value in parent.state_dict().items()},
                'config': asdict(V21Config(vocabulary_size=401, prediction_horizon=40))}, parent_path)
    parameters = {name: parent.state_dict()[name].clone() for name in MEAN_PARAMETER_NAMES}
    path = tmp_path/'candidate.pt'
    torch.save(dict(parent_sha256=sha(parent_path), parameters=parameters, method='lewm_mean', seed=21,
                    held_out_run='reviewed_recent_four', requires_independent_validation=True), path)
    prepared = prepare_lewm_controller(root, path, sha(parent_path), sha(path))
    assert prepared.parent.horizon == 40 and prepared.candidate.model.horizon == 40

