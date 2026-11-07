"""Strict, model-only loading of mean-response candidates for shadow inference.

The caller supplies the trusted parent digest independently of the candidate.
The returned LeWM is a deepcopy; this module never installs it on a core or
runtime. Its uncertainty head stays at the parent weights, but changed hidden
states can change uncertainty outputs. Independent validation remains required.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re

import torch

from .v21_models import V21LeWM


MEAN_PARAMETER_NAMES = frozenset({
    "joint_ids", "history_ids", "query", "joint.weight", "joint.bias",
    "history_projection.0.weight", "history_projection.0.bias",
    "history_projection.2.weight", "history_projection.2.bias",
    "command_projection.0.weight", "command_projection.0.bias",
    "command_projection.2.weight", "command_projection.2.bias",
    "time_projection.weight", "time_projection.bias", "norm.weight", "norm.bias",
    "response.weight", "response.bias",
} | {
    f"layers.{layer}.{name}"
    for layer in range(2)
    for name in (
        "norm1.weight", "norm1.bias", "attn.in_proj_weight", "attn.in_proj_bias",
        "attn.out_proj.weight", "attn.out_proj.bias", "norm2.weight", "norm2.bias",
        "ffn.0.weight", "ffn.0.bias", "ffn.2.weight", "ffn.2.bias",
    )
})
SCALE_PARAMETER_NAMES = frozenset({"scale.weight", "scale.bias"})
# The visual layers V21LeWM gained on 2026-09-23 (0a6424fd), named one by one so the contract below is
# fixed here rather than read off whatever model is supplied.
VISION_PARAMETER_NAMES = frozenset({
    f"vision_layers.{layer}.{name}"
    for layer in range(4)
    for name in (
        "norm1.weight", "norm1.bias", "attn.in_proj_weight", "attn.in_proj_bias",
        "attn.out_proj.weight", "attn.out_proj.bias", "norm2.weight", "norm2.bias",
        "ffn.0.weight", "ffn.0.bias", "ffn.2.weight", "ffn.2.bias",
    )
} | {
    f"vision_read.{reader}.{name}"
    for reader in range(2)
    for name in ("q_proj_weight", "k_proj_weight", "v_proj_weight", "in_proj_bias",
                 "out_proj.weight", "out_proj.bias")
})
# A parent must have exactly one of these parameter sets. A candidate only ever replaces the 43 mean
# parameters; every other parent tensor must equal the verified parent checkpoint bit for bit. Joint-only
# parents from before 2026-09-23 (mean + scale, 45 tensors) do not load into the current V21LeWM, which always
# has the visual layers, nor into the runtime (strict loading); supporting them would need a legacy
# architecture factory, which does not exist.
PARENT_CONTRACTS = {
    "v21-lewm-vision-v2": MEAN_PARAMETER_NAMES | SCALE_PARAMETER_NAMES | VISION_PARAMETER_NAMES,
}
_PAYLOAD_KEYS = {"parent_sha256", "parameters", "method", "seed",
                 "held_out_run", "requires_independent_validation"}


class LeWMAdaptationError(ValueError):
    """A candidate or its claimed parent failed the shadow-loading contract."""


@dataclass(frozen=True)
class LeWMShadowCandidate:
    model: V21LeWM
    parent_sha256: str
    candidate_sha256: str
    method: str
    seed: int
    held_out_run: str
    parameter_names: tuple[str, ...]
    requires_independent_validation: bool = True
    parent_contract: str | None = None


def _read_bytes(path, description):
    try:
        return Path(path).read_bytes()
    except (OSError, TypeError) as error:
        raise LeWMAdaptationError(f"cannot read {description}") from error


def _load(payload_bytes, *, weights_only, description):
    try:
        return torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=weights_only)
    except Exception as error:
        raise LeWMAdaptationError(f"cannot decode {description}") from error


def _validate_tensor(name, value, expected):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise LeWMAdaptationError(f"{name}: a dense tensor is required")
    if value.shape != expected.shape:
        raise LeWMAdaptationError(f"{name}: shape mismatch")
    if value.dtype != expected.dtype:
        raise LeWMAdaptationError(f"{name}: dtype mismatch")
    if not torch.isfinite(value).all():
        raise LeWMAdaptationError(f"{name}: nonfinite tensor")


def load_lewm_shadow_candidate(
    parent_lewm: V21LeWM,
    *,
    parent_checkpoint: str | Path,
    candidate_path: str | Path,
    expected_parent_sha256: str,
) -> LeWMShadowCandidate:
    """Return an independent, frozen eval LeWM on the parent's device/dtype.

    Both the parent file and its in-memory LeWM must match. Candidate names are
    the exact 43 mean parameters: the uncertainty head and position buffers may
    not be replaced. All shapes, dtypes and values are checked before a strict
    whole-state load; failures raise without changing the supplied model.

    The established parent checkpoint contains Python metadata, so it is read
    with ``weights_only=False`` only AFTER its bytes match the separately trusted
    digest. Candidate artifacts always require ``weights_only=True``; no unsafe
    candidate fallback is attempted. Digests and decoding use the same bytes.
    """
    if type(parent_lewm) is not V21LeWM:
        raise LeWMAdaptationError("parent_lewm must be an unwrapped V21LeWM")
    if (type(expected_parent_sha256) is not str or
            re.fullmatch(r"[0-9a-f]{64}", expected_parent_sha256) is None):
        raise LeWMAdaptationError("a trusted lowercase parent SHA-256 is required")

    parent_bytes = _read_bytes(parent_checkpoint, "parent checkpoint")
    parent_digest = hashlib.sha256(parent_bytes).hexdigest()
    if parent_digest != expected_parent_sha256:
        raise LeWMAdaptationError("parent checkpoint SHA-256 mismatch")
    candidate_bytes = _read_bytes(candidate_path, "candidate")
    payload = _load(candidate_bytes, weights_only=True, description="candidate")
    if type(payload) is not dict or set(payload) != _PAYLOAD_KEYS:
        raise LeWMAdaptationError("candidate payload keys differ from the supported contract")
    if payload["parent_sha256"] != parent_digest:
        raise LeWMAdaptationError("candidate parent SHA-256 mismatch")
    if payload["method"] != "lewm_mean":
        raise LeWMAdaptationError("only lewm_mean candidates are supported")
    if type(payload["seed"]) is not int or payload["seed"] < 0:
        raise LeWMAdaptationError("candidate seed must be a nonnegative Python integer")
    if type(payload["held_out_run"]) is not str or not payload["held_out_run"].strip():
        raise LeWMAdaptationError("candidate held_out_run must be a nonempty Python string")
    if payload["requires_independent_validation"] is not True:
        raise LeWMAdaptationError("candidate must retain its independent-validation requirement")
    parameters = payload["parameters"]
    if type(parameters) is not dict or set(parameters) != MEAN_PARAMETER_NAMES:
        raise LeWMAdaptationError("candidate mean parameter names must match exactly")

    shadow = deepcopy(parent_lewm)
    parent_state = shadow.state_dict()
    names = set(dict(shadow.named_parameters()))
    contract = next((name for name, expected in PARENT_CONTRACTS.items() if names == expected), None)
    if contract is None:
        raise LeWMAdaptationError("parent LeWM parameter schema differs")
    if len({value.device for value in parent_state.values()}) != 1:
        raise LeWMAdaptationError("parent LeWM state must be on one device")
    checkpoint = _load(parent_bytes, weights_only=False, description="verified parent checkpoint")
    if type(checkpoint) is not dict or not isinstance(checkpoint.get("model"), dict):
        raise LeWMAdaptationError("parent checkpoint must contain a core model state")
    recorded = {name.removeprefix("lewm."): value for name, value in checkpoint["model"].items()
                if isinstance(name, str) and name.startswith("lewm.")}
    if set(recorded) != set(parent_state):
        raise LeWMAdaptationError("parent checkpoint LeWM state names differ")
    for name, expected in parent_state.items():
        _validate_tensor(f"parent.{name}", recorded[name], expected)
        if not torch.equal(recorded[name].cpu(), expected.detach().cpu()):
            raise LeWMAdaptationError(f"in-memory parent differs from checkpoint: {name}")
    for name in sorted(MEAN_PARAMETER_NAMES):
        _validate_tensor(f"candidate.{name}", parameters[name], parent_state[name])

    complete_state = {
        name: (parameters[name].to(device=value.device).detach().clone()
               if name in MEAN_PARAMETER_NAMES else value.detach().clone())
        for name, value in parent_state.items()
    }
    shadow.load_state_dict(complete_state, strict=True)
    shadow.eval().requires_grad_(False)
    return LeWMShadowCandidate(
        model=shadow,
        parent_sha256=parent_digest,
        candidate_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
        method=payload["method"], seed=payload["seed"], held_out_run=payload["held_out_run"],
        parameter_names=tuple(sorted(MEAN_PARAMETER_NAMES)), parent_contract=contract,
    )
