"""Experimental controller admission and evidence identity using CPU/fake IO."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from fcpg.v21_live_service import V21LiveService
from scripts import serve_so101_v21_research as cli
from test_v21_live_service import service  # noqa: F401 -- fake IO fixture
from test_v21_partial_failure_audit import install_bridge


PARENT_SHA = 'a' * 64
IDENTITY = dict(schema='v21-lewm-control-identity-v1', experimental=True, control_influence=True,
                parent_sha256=PARENT_SHA, controller_state_sha256='c' * 64,
                uncertainty_calibrated=False, requires_independent_validation=True)
CONTROL_KEYS = ('lewm_control_checkpoint', 'lewm_control_parent_sha256', 'lewm_control_candidate_sha256')


@pytest.fixture
def control_config(tmp_path):
    path = tmp_path / 'candidate.pt'
    path.write_bytes(b'fake candidate for service boundary tests')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(lewm_control_checkpoint=path, lewm_control_parent_sha256=PARENT_SHA,
                lewm_control_candidate_sha256=digest, lewm_shadow_checkpoints=[path],
                lewm_shadow_parent_sha256=PARENT_SHA, lewm_cem=True,
                lewm_cem_objective='actor_response_tracking', trial_max_blocks=2)


@pytest.mark.parametrize('present', list(itertools.product((False, True), repeat=3))[1:-1])
def test_partial_control_options_rejected_before_owner(tmp_path, present):
    kwargs = {key: value for key, value, included in zip(CONTROL_KEYS, ('candidate.pt', PARENT_SHA, 'b' * 64), present) if included}
    with pytest.raises(ValueError, match='required together'):
        V21LiveService(tmp_path / 'model', tmp_path / 'out',
                       owner_factory=lambda *a, **kw: pytest.fail('owner constructed'), **kwargs)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('override', [
    {'lewm_cem': False}, {'lewm_cem_objective': 'actor_command_tracking'},
    {'lewm_cem_objective': 'column_visual'}, {'raw_actor': True},
    {'grounding_correction': True}, {'head_grasp': True}, {'alignment_motion_enabled': True},
    {'column_dino_fg_calibration': True}, {'gripper_check_enabled': True},
    {'resume_auxiliary_state': 'old.pt'}, {'column_memory': 'old.pt'}, {'trial_max_blocks': 3},
    {'lewm_shadow_checkpoints': []}, {'lewm_shadow_parent_sha256': 'd' * 64}])
def test_invalid_control_mode_rejected_before_owner(tmp_path, control_config, override):
    with pytest.raises(ValueError, match='LeWM control'):
        V21LiveService(tmp_path / 'model', tmp_path / 'out',
                       owner_factory=lambda *a, **kw: pytest.fail('owner constructed'),
                       **dict(control_config, **override))
    assert not (tmp_path / 'out').exists()


def install_preflight_mock(monkeypatch, prepare, install=None):
    module = ModuleType('fcpg.v21_lewm_control')
    module.prepare_lewm_controller = prepare
    module.install_prepared_lewm_controller = install
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_cpu_preflight_failure_precedes_owner_and_output(tmp_path, monkeypatch, control_config):
    def reject(*args):
        assert args == (tmp_path / 'model', control_config['lewm_control_checkpoint'],
                        PARENT_SHA, control_config['lewm_control_candidate_sha256'])
        raise ValueError('candidate digest mismatch')

    install_preflight_mock(monkeypatch, reject)
    with pytest.raises(ValueError, match='digest mismatch'):
        V21LiveService(tmp_path / 'model', tmp_path / 'out',
                       owner_factory=lambda *a, **kw: pytest.fail('owner constructed'), **control_config)
    assert not (tmp_path / 'out').exists()


def test_control_candidate_must_be_present_in_shadows_before_owner(tmp_path, monkeypatch, control_config):
    different = tmp_path / 'different.pt'
    different.write_bytes(b'another candidate')
    install_preflight_mock(monkeypatch, lambda *a: SimpleNamespace(identity=IDENTITY))
    with pytest.raises(ValueError, match='included among shadow'):
        V21LiveService(tmp_path / 'model', tmp_path / 'out',
                       owner_factory=lambda *a, **kw: pytest.fail('owner constructed'),
                       **dict(control_config, lewm_shadow_checkpoints=[different]))
    assert not (tmp_path / 'out').exists()


def test_install_preserves_original_parent_for_all_shadow_calls(service, tmp_path, monkeypatch, control_config):
    events = []
    installed = {}
    candidate = deepcopy(service.owner.runtime.core.lewm).eval().requires_grad_(False)
    identity = dict(IDENTITY, candidate_sha256=control_config['lewm_control_candidate_sha256'])
    prepared = SimpleNamespace(parent=deepcopy(candidate), candidate=SimpleNamespace(model=candidate), identity=identity)

    def prepare(*args):
        events.append('preflight')
        assert not (tmp_path / 'controlled').exists()
        return prepared

    def owner(*args, **kwargs):
        events.append('owner')
        actual = type(service.owner)(*args, **kwargs)
        installed['parent'] = actual.runtime.core.lewm
        return actual

    def install(core, value):
        events.append('install')
        assert value is prepared
        original = core.lewm
        core.lewm = value.candidate.model
        return original

    def load_shadow(parent, **kwargs):
        events.append('shadow_load')
        assert parent is installed['parent'] and parent is not candidate
        assert kwargs['expected_parent_sha256'] == PARENT_SHA
        return SimpleNamespace(candidate_sha256=control_config['lewm_control_candidate_sha256'])

    def evaluator(parent, candidates, **kwargs):
        events.append('evaluator')
        assert parent is installed['parent'] and len(candidates) == 1
        assert kwargs == dict(controller_candidate_sha256=control_config['lewm_control_candidate_sha256'])
        return SimpleNamespace()

    install_preflight_mock(monkeypatch, prepare, install)
    monkeypatch.setattr('fcpg.v21_lewm_adaptation.load_lewm_shadow_candidate', load_shadow)
    monkeypatch.setattr('fcpg.v21_lewm_shadow.LeWMShadowEvaluator', evaluator)
    configured = V21LiveService(tmp_path / 'model', tmp_path / 'controlled', device='cpu',
        owner_factory=owner, clock=service.clock, target_diagnostics=False, **control_config)
    try:
        assert events == ['preflight', 'owner', 'install', 'shadow_load', 'evaluator']
        assert configured.owner.runtime.core.lewm is candidate
        assert configured.lewm_parent_baseline is installed['parent']
        status = configured.status()
        assert status['model']['lewm_controller'] == identity
        assert status['model']['checkpoint'] == str(tmp_path / 'model/fish/final.pt')
        status['model']['lewm_controller']['experimental'] = False
        assert configured.lewm_controller['experimental'] is True
        assert configured.owner.writes == 0 and configured.owner.runtime.stopped
    finally:
        configured.close()


@pytest.mark.parametrize('rejected', [False, True])
def test_controller_identity_in_plan_execution_snapshot_and_history(service, monkeypatch, rejected):
    bridge, writes, _ = install_bridge(service, monkeypatch, stall_j3=rejected)
    service.record_policy_inputs = True
    service.lewm_controller = dict(IDENTITY, candidate_sha256='b' * 64)
    expected = deepcopy(service.lewm_controller)
    service.handle('plan')
    assert service.plan_record['lewm_controller'] == expected
    if rejected:
        with pytest.raises(ValueError, match='actual.*response|respond'):
            service.handle('execute')
    else:
        service.handle('execute')
    files = list(service.output.glob('execution-*.json'))
    assert len(files) == 1 and len(writes) == 5
    record = json.loads(files[0].read_text())
    payload = torch.load(service.output / record['policy_input_snapshot'], weights_only=False)
    assert payload['schema'] == 2 and payload['lewm_controller'] == expected
    assert record['lewm_controller'] == service.execution_history[-1]['lewm_controller'] == expected
    assert record['response_verified'] is (not rejected)
    assert bridge.last_attempt['lewm_controller'] == expected
    assert service.owner.runtime.stopped


def test_default_does_not_preflight_install_or_change_original_model(service, monkeypatch):
    assert service.lewm_controller is None and service.lewm_parent_baseline is None
    assert service.status()['model']['lewm_controller'] is None
    assert service.owner.writes == 0


def cli_control_args():
    return ['--lewm-control-checkpoint', 'candidate.pt', '--lewm-control-parent-sha256', PARENT_SHA,
            '--lewm-control-candidate-sha256', 'b' * 64, '--lewm-shadow-checkpoint', 'candidate.pt',
            '--lewm-shadow-parent-sha256', PARENT_SHA, '--lewm-cem-objective', 'actor_response_tracking',
            '--trial-max-blocks', '2']


@pytest.mark.parametrize('args', [
    ['--lewm-control-checkpoint', 'candidate.pt'],
    cli_control_args() + ['--no-lewm-cem'], cli_control_args() + ['--raw-actor'],
    cli_control_args() + ['--grounding-correction'],
    cli_control_args() + ['--resume-auxiliary-state', 'resume.pt'],
    cli_control_args() + ['--trial-max-blocks', '3'],
    cli_control_args() + ['--lewm-cem-objective', 'actor_command_tracking']])
def test_cli_invalid_control_never_constructs_server_or_owner(monkeypatch, args):
    def forbidden(*a, **kw):
        pytest.fail('invalid config constructed server/owner')

    monkeypatch.setattr(cli, 'make_server', forbidden)
    monkeypatch.setattr(cli, 'V21LiveService', forbidden)
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize('enabled', [False, True])
def test_cli_only_forwards_control_kwargs_when_enabled(monkeypatch, enabled):
    calls = []
    server = SimpleNamespace(serve_forever=lambda **kw: None, shutdown=lambda: None, server_close=lambda: None)
    monkeypatch.setattr(cli, 'make_server', lambda *a: server)
    monkeypatch.setattr(cli.signal, 'signal', lambda *a: None)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')

    def create(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(cli, 'V21LiveService', create)
    cli.main(['--enable-hardware', *(cli_control_args() if enabled else [])])
    assert len(calls) == 1
    assert all((key in calls[0]) is enabled for key in CONTROL_KEYS)
    if enabled:
        assert calls[0]['lewm_control_checkpoint'] == Path('candidate.pt')
        assert calls[0]['lewm_control_parent_sha256'] == PARENT_SHA
        assert calls[0]['lewm_control_candidate_sha256'] == 'b' * 64
