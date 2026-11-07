"""Hash-bound preflight and explicit installation of experimental LeWM control.

Preparation is CPU-only and constructs no runtime or hardware owner. Installation
only swaps core.lewm; the caller owns operation/STOP and trial-scope decisions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import re

import torch

from .v21_core import V21Config
from .v21_lewm_adaptation import LeWMShadowCandidate, load_lewm_shadow_candidate
from .v21_models import V21LeWM


class LeWMControllerError(ValueError):
    """The requested controller does not match its verified preparation."""


def _mapping_sha256(state):
    # Identical name/dtype/shape/raw-byte convention to the shadow evaluator.
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def lewm_state_sha256(model):
    """Use the exact state digest used by LeWMShadowEvaluator._state_sha."""
    return _mapping_sha256(model.state_dict())


def _non_lewm_sha256(core):
    return _mapping_sha256({name: value for name, value in core.state_dict().items()
                            if not name.startswith('lewm.')})


def _verified_bytes(path, expected, description):
    if type(expected) is not str or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
        raise LeWMControllerError(f'{description} requires an explicit lowercase SHA-256')
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise LeWMControllerError(f'cannot read {description}: {path}') from error
    if hashlib.sha256(raw).hexdigest() != expected:
        raise LeWMControllerError(f'{description} SHA-256 mismatch')
    return raw


@dataclass(frozen=True)
class PreparedLeWMController:
    parent: V21LeWM
    candidate: LeWMShadowCandidate
    identity: dict
    _parent_state_sha256: str = field(repr=False)
    _candidate_state_sha256: str = field(repr=False)
    _identity_sha256: str = field(repr=False)


def _identity(parent_path, candidate_path, candidate, candidate_state_sha):
    return dict(
        schema='v21-lewm-control-identity-v1', kind='experimental-adapted-lewm-mean',
        base_checkpoint=str(parent_path), candidate_checkpoint=str(candidate_path),
        parent_sha256=candidate.parent_sha256, candidate_sha256=candidate.candidate_sha256,
        controller_state_sha256=candidate_state_sha, source='adapted_lewm_mean',
        control_influence=True, experimental=True, uncertainty_calibrated=False,
        requires_independent_validation=True, method=candidate.method, seed=candidate.seed,
        held_out_run=candidate.held_out_run, parent_contract=candidate.parent_contract,
    )


def prepare_lewm_controller(root, candidate_path, expected_parent_sha256,
                            expected_candidate_sha256):
    """Verify both files before decoding, then prepare frozen CPU predictors.

    The parent pickle contains established Python metadata and is decoded only
    from bytes already matched to its independently supplied digest. The parent
    LeWM is built from that checkpoint's own model config (its prediction
    horizon), never from defaults, and loaded strictly. Candidate
    decoding/schema validation is delegated to the strict shadow loader, which
    uses weights_only=True and accepts exactly the 43 mean parameter tensors on
    a parent of a known parameter contract.
    """
    parent_path = (Path(root).expanduser()/'fish'/'final.pt').resolve()
    candidate_path = Path(candidate_path).expanduser().resolve()
    parent_bytes = _verified_bytes(parent_path, expected_parent_sha256, 'parent checkpoint')
    _verified_bytes(candidate_path, expected_candidate_sha256, 'candidate checkpoint')
    # Neither constructor initialization nor a validation failure may perturb
    # the controller's CPU random stream. Explicit device guards default-device
    # settings from turning preflight into a CUDA allocation.
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        try:
            payload = torch.load(io.BytesIO(parent_bytes), map_location='cpu', weights_only=False)
            if type(payload) is not dict or not isinstance(payload.get('model'), dict):
                raise LeWMControllerError('parent checkpoint has no core model state')
            if not isinstance(payload.get('config'), dict):
                raise LeWMControllerError('parent checkpoint has no model config')
            horizon = V21Config(**payload['config']).prediction_horizon
            state = {name.removeprefix('lewm.'): value for name, value in payload['model'].items()
                     if isinstance(name, str) and name.startswith('lewm.')}
            parent = V21LeWM(horizon=horizon).eval().requires_grad_(False)
            parent.load_state_dict(state, strict=True)
            candidate = load_lewm_shadow_candidate(parent,
                parent_checkpoint=parent_path, candidate_path=candidate_path,
                expected_parent_sha256=expected_parent_sha256)
        except LeWMControllerError:
            raise
        except Exception as error:
            raise LeWMControllerError(f'controller preflight failed: {error}') from error
    if candidate.candidate_sha256 != expected_candidate_sha256:
        raise LeWMControllerError('candidate changed between digest verification and loading')
    # Recheck file identities after the existing strict loader has read them.
    _verified_bytes(parent_path, expected_parent_sha256, 'parent checkpoint')
    _verified_bytes(candidate_path, expected_candidate_sha256, 'candidate checkpoint')
    parent_sha = lewm_state_sha256(parent)
    candidate_sha = lewm_state_sha256(candidate.model)
    identity = _identity(parent_path, candidate_path, candidate, candidate_sha)
    return PreparedLeWMController(
        parent=parent, candidate=candidate,
        identity=identity,
        _parent_state_sha256=parent_sha, _candidate_state_sha256=candidate_sha,
        _identity_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
    )


def install_prepared_lewm_controller(core, prepared):
    """Swap only core.lewm after all checks; return its untouched original object.

    Device moves are allowed; dtype conversion is not. The prepared CPU models
    remain independent and the original parent stays available as a shadow
    baseline. This function does not observe, plan, execute, or change STOP.
    """
    if type(prepared) is not PreparedLeWMController:
        raise LeWMControllerError('verified PreparedLeWMController required')
    original = getattr(core, 'lewm', None)
    if type(original) is not V21LeWM:
        raise LeWMControllerError('core.lewm must be an unwrapped original V21LeWM')
    if lewm_state_sha256(prepared.parent) != prepared._parent_state_sha256:
        raise LeWMControllerError('prepared parent state changed')
    if lewm_state_sha256(prepared.candidate.model) != prepared._candidate_state_sha256:
        raise LeWMControllerError('prepared candidate state changed')
    identity = prepared.identity
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != prepared._identity_sha256:
        raise LeWMControllerError('prepared controller identity changed')
    expected_identity = _identity(identity.get('base_checkpoint'), identity.get('candidate_checkpoint'),
                                  prepared.candidate, prepared._candidate_state_sha256)
    if identity != expected_identity:
        raise LeWMControllerError('prepared controller identity changed')
    if (prepared.candidate.method != 'lewm_mean'
            or prepared.candidate.requires_independent_validation is not True):
        raise LeWMControllerError('prepared candidate contract differs')
    actual, expected = original.state_dict(), prepared.parent.state_dict()
    if set(actual) != set(expected):
        raise LeWMControllerError('in-memory parent state names differ')
    for name, value in actual.items():
        if value.dtype != expected[name].dtype:
            raise LeWMControllerError(f'in-memory parent dtype differs: {name}')
        if value.shape != expected[name].shape or not torch.equal(value.detach().cpu(), expected[name]):
            raise LeWMControllerError(f'in-memory parent state differs: {name}')
    devices = {value.device for value in actual.values()}
    if len(devices) != 1:
        raise LeWMControllerError('in-memory parent must occupy one device')
    replacement = deepcopy(prepared.candidate.model).to(device=next(iter(devices)))
    replacement.eval().requires_grad_(False)
    if lewm_state_sha256(replacement) != prepared._candidate_state_sha256:
        raise LeWMControllerError('candidate changed during device transfer')
    non_lewm_before = _non_lewm_sha256(core)
    try:
        core.lewm = replacement
        if _non_lewm_sha256(core) != non_lewm_before:
            raise LeWMControllerError('non-LeWM core state changed during installation')
    except Exception:
        core.lewm = original
        raise
    return original
