"""Training-only reproducibility; never changes model or robot inference."""
from copy import deepcopy
import os

import pytest
import torch

from fcpg.v21_training_determinism import (
    configure_training_determinism, reproducibility_record, validate_determinism_resume)
from scripts.train_v21_horizon40 import canonical_protocol, parser, validate_resource_continuation


@pytest.fixture(autouse=True)
def restore_process_flags():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn)
    torch.backends.cudnn.benchmark = benchmark


def test_legacy_default_is_explicitly_unchanged(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    state = torch.get_rng_state().clone()
    before = torch.are_deterministic_algorithms_enabled()
    configure_training_determinism(False)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.are_deterministic_algorithms_enabled() == before
    assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
    assert reproducibility_record(False) is None
    assert not parser().parse_args(['--parent', 'p', '--source-config', 'c', '--output', 'o']).deterministic_training


def test_strict_flags_set_without_consuming_rng(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    state = torch.get_rng_state().clone()
    configure_training_determinism(True)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.backends.cudnn.benchmark
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert torch.equal(state, torch.get_rng_state())
    record = reproducibility_record(True)
    assert record['torch'] == torch.__version__
    assert record['cublas_workspace_config'] == ':4096:8'
    assert record['sdpa_backends']


def test_conflicting_workspace_is_not_overwritten(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    with pytest.raises(ValueError, match='workspace'):
        configure_training_determinism(True)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':16:8'


def test_cannot_set_workspace_after_cuda_started(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(ValueError, match='before CUDA'):
        configure_training_determinism(True)
    assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ


def test_legacy_protocol_and_disabled_new_protocol_compare_equal():
    old = dict(schedule={}, device={})
    new = deepcopy(old)
    new['schedule']['deterministic_training'] = False
    new['reproducibility'] = None
    assert canonical_protocol(old) == canonical_protocol(new)
    validate_determinism_resume(old, False)
    validate_resource_continuation(old, new)


def test_resume_cannot_silently_change_policy_or_runtime():
    old = dict(schedule={}, device={})
    strict = dict(schedule=dict(deterministic_training=True), reproducibility=dict(torch='2.8', cuda='12.8'), device={})
    with pytest.raises(ValueError, match='determinism'):
        validate_determinism_resume(old, True)
    with pytest.raises(ValueError, match='determinism'):
        validate_determinism_resume(strict, False)
    validate_determinism_resume(strict, True, current_runtime=strict['reproducibility'])
    with pytest.raises(ValueError, match='runtime'):
        validate_determinism_resume(strict, True, current_runtime=dict(torch='other', cuda='12.8'))
    with pytest.raises(ValueError, match='cannot change'):
        validate_resource_continuation(old, strict)


def test_checkpoint_checks_before_loading_any_weights():
    from scripts.train_v21_horizon40 import restore_training_checkpoint
    # Real V21ResearchCore models expose .lewm; earlier contract checks read it.
    core = torch.nn.Module()
    core.lewm = torch.nn.Linear(2, 2)
    before = {k: v.clone() for k, v in core.state_dict().items()}
    saved = dict(native40_protocol=dict(schedule={}), model={})
    with pytest.raises(ValueError, match='determinism'):
        restore_training_checkpoint(core, saved, deterministic_training=True)
    assert all(torch.equal(before[k], v) for k, v in core.state_dict().items())
