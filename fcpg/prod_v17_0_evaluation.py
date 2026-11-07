"""Checkpoint-bound offline evaluation for the Prod-V17.0 model stack.

The evaluators in this module are deliberately read-only.  Every entry point
constructs the fixed V17 runtime module, strictly loads one final checkpoint,
freezes it, and performs inference only on an explicitly declared validation
or DEV population.  No model selection, online update, robot I/O, or task TEST
access is part of this surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fcpg.prod_v16_5 import StateFKLeWM
from fcpg.prod_v16_8 import FullHistoryFiveCommandActor
from fcpg.prod_v17_0 import (
    IMAGE_SIZE,
    MODEL_DIMENSION,
    OCCLUDED_INDEX,
    TIPS,
    VIEWS,
    V170RecurrentQueryFCPG,
)


CHECKPOINT_FORMAT = "so101-prod-v17.0-stack-v1"
DEV_SPLIT = 1
TEST_SPLIT = 2
OBSERVATIONS = 5
TRANSITIONS = 4
COMMAND_STEPS = 5
STATE_DIM = 6
PATCHES = 256
TOKEN_DIM = 192


class V170EvaluationError(RuntimeError):
    """Raised when a frozen-evaluation contract is not satisfied."""


@dataclass(frozen=True)
class V170FinalModels:
    fcpg: V170RecurrentQueryFCPG
    lewm: StateFKLeWM
    actor: FullHistoryFiveCommandActor


def _action_contract() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    step = torch.tensor([0.04, 0.04, 0.04, 0.04, 0.04, 0.08])
    lower = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, 0.0])
    upper = torch.ones(STATE_DIM)
    return step, lower, upper


def _new_module(stage: Literal["fcpg", "lewm", "actor"]) -> nn.Module:
    if stage == "fcpg":
        return V170RecurrentQueryFCPG()
    step, lower, upper = _action_contract()
    if stage == "lewm":
        return StateFKLeWM(
            step_cap=step,
            action_lower=lower,
            action_upper=upper,
            residual_cap=0.05,
            dimension=MODEL_DIMENSION,
            layers=2,
        )
    if stage == "actor":
        return FullHistoryFiveCommandActor(
            step_cap=step,
            action_lower=lower,
            action_upper=upper,
            dimension=MODEL_DIMENSION,
            fusion_mode="query_residual",
            history_mode="full",
            residual_gate=True,
        )
    raise AssertionError(stage)


def _checkpoint_payload(path: str | Path, *, stage: str) -> Mapping[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise V170EvaluationError(f"final {stage} checkpoint is absent: {resolved}")
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exact torch error is version-specific
        raise V170EvaluationError(f"final {stage} checkpoint cannot be read") from exc
    if not isinstance(payload, Mapping):
        raise V170EvaluationError("final checkpoint must contain a mapping")
    expected = {"format", "stage", "global_step", "config", "model"}
    if set(payload) != expected:
        raise V170EvaluationError(
            f"final checkpoint fields differ: missing={sorted(expected - set(payload))} "
            f"unexpected={sorted(set(payload) - expected)}"
        )
    if payload["format"] != CHECKPOINT_FORMAT or payload["stage"] != stage:
        raise V170EvaluationError("final checkpoint format or stage differs")
    step = payload["global_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise V170EvaluationError("final checkpoint global_step must be non-negative")
    if not isinstance(payload["config"], Mapping):
        raise V170EvaluationError("final checkpoint config must be a mapping")
    if not isinstance(payload["model"], Mapping):
        raise V170EvaluationError("final checkpoint model must be a state mapping")
    return payload


def load_v170_final_model(
    path: str | Path,
    *,
    stage: Literal["fcpg", "lewm", "actor"],
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Instantiate the fixed module recipe and strictly load a final state."""

    payload = _checkpoint_payload(path, stage=stage)
    model = _new_module(stage)
    expected = model.state_dict()
    received = payload["model"]
    if set(received) != set(expected):
        raise V170EvaluationError(
            f"{stage} state fields differ: missing={sorted(set(expected) - set(received))} "
            f"unexpected={sorted(set(received) - set(expected))}"
        )
    checked: dict[str, torch.Tensor] = {}
    for name, reference in expected.items():
        value = received[name]
        if not isinstance(value, torch.Tensor):
            raise V170EvaluationError(f"{stage} state[{name}] is not a tensor")
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise V170EvaluationError(f"{stage} state[{name}] shape or dtype differs")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise V170EvaluationError(f"{stage} state[{name}] is non-finite")
        checked[name] = value.detach().clone()
    model.load_state_dict(checked, strict=True)
    model.to(torch.device(device)).eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise V170EvaluationError("final model did not enter frozen evaluation mode")
    return model


def load_v170_final_models(
    *,
    fcpg_checkpoint: str | Path,
    lewm_checkpoint: str | Path,
    actor_checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> V170FinalModels:
    return V170FinalModels(
        fcpg=load_v170_final_model(
            fcpg_checkpoint, stage="fcpg", device=device
        ),  # type: ignore[arg-type]
        lewm=load_v170_final_model(
            lewm_checkpoint, stage="lewm", device=device
        ),  # type: ignore[arg-type]
        actor=load_v170_final_model(
            actor_checkpoint, stage="actor", device=device
        ),  # type: ignore[arg-type]
    )


def _population_rows(adapter: Any, name: str) -> np.ndarray:
    if not hasattr(adapter, name):
        raise V170EvaluationError(f"adapter must declare {name}")
    raw = getattr(adapter, name)
    raw = raw() if callable(raw) else raw
    if isinstance(raw, bool):
        raise V170EvaluationError(f"adapter {name} is invalid")
    if isinstance(raw, (int, np.integer)):
        if int(raw) <= 0:
            raise V170EvaluationError(f"adapter {name} must be positive")
        return np.arange(int(raw), dtype=np.int64)
    rows = np.asarray(raw)
    if rows.ndim != 1 or len(rows) == 0 or not np.issubdtype(rows.dtype, np.integer):
        raise V170EvaluationError(f"adapter {name} must be a row count or integer rows")
    rows = rows.astype(np.int64, copy=False)
    if bool((rows < 0).any()) or len(np.unique(rows)) != len(rows):
        raise V170EvaluationError(f"adapter {name} rows must be unique and non-negative")
    return rows


def _batch(
    adapter: Any,
    rows: np.ndarray,
    *,
    validation: bool = False,
) -> Mapping[str, Any]:
    method = getattr(
        adapter,
        "validation_batch" if validation and hasattr(adapter, "validation_batch") else "batch",
        None,
    )
    if not callable(method):
        raise V170EvaluationError("adapter must provide batch(rows)")
    value = method(np.asarray(rows, dtype=np.int64))
    if not isinstance(value, Mapping):
        raise V170EvaluationError("adapter batch must return a mapping")
    return value


def _split(batch: Mapping[str, Any], rows: int) -> None:
    if "split" not in batch:
        raise V170EvaluationError("evaluation batch must carry split")
    value = np.asarray(batch["split"])
    if value.shape != (rows,) or not np.issubdtype(value.dtype, np.integer):
        raise V170EvaluationError("evaluation split must be integer [B]")
    if bool((value == TEST_SPLIT).any()):
        raise V170EvaluationError("task TEST is sealed")
    if not bool((value == DEV_SPLIT).all()):
        raise V170EvaluationError("offline evaluator accepts only validation/DEV rows")


def _numpy(batch: Mapping[str, Any], name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in batch:
        raise V170EvaluationError(f"evaluation batch omits {name}")
    value = np.asarray(batch[name])
    if value.shape != shape:
        raise V170EvaluationError(
            f"evaluation {name} shape differs: observed={value.shape} expected={shape}"
        )
    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
        raise V170EvaluationError(f"evaluation {name} is non-finite")
    return value


def _tensor(
    batch: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(
        _numpy(batch, name, shape), device=device, dtype=dtype
    )


def _rgb(
    batch: Mapping[str, Any], name: str, shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    value = _tensor(batch, name, shape, device=device, dtype=torch.float32)
    if float(value.min()) < 0.0:
        raise V170EvaluationError(f"evaluation {name} has negative pixels")
    if float(value.max()) > 1.0:
        if float(value.max()) > 255.0:
            raise V170EvaluationError(f"evaluation {name} pixels exceed uint8 range")
        value = value / 255.0
    return value


def _bool(
    batch: Mapping[str, Any], name: str, shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    value = _numpy(batch, name, shape)
    if value.dtype != np.bool_:
        raise V170EvaluationError(f"evaluation {name} must be boolean")
    return torch.as_tensor(value, device=device, dtype=torch.bool)


def _frozen_encoder(encoder: Any) -> None:
    if isinstance(encoder, nn.Module):
        if encoder.training:
            raise V170EvaluationError("frozen V12 encoder must already be in eval mode")
        if any(parameter.requires_grad for parameter in encoder.parameters()):
            raise V170EvaluationError("frozen V12 encoder has trainable parameters")
    if not callable(encoder):
        raise V170EvaluationError("frozen V12 encoder must be callable")


def _v12_tokens(encoder: Any, rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 5 or rgb.shape[1:] != (VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise V170EvaluationError("V12 input must be [B,2,3,224,224]")
    batch = len(rgb)
    encoded = encoder(rgb.reshape(batch * VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE))
    if hasattr(encoded, "last_hidden_state"):
        encoded = encoded.last_hidden_state
    if isinstance(encoded, (tuple, list)):
        encoded = encoded[0]
    if not isinstance(encoded, torch.Tensor):
        raise V170EvaluationError("frozen V12 encoder did not return a tensor")
    if encoded.shape == (batch * VIEWS, PATCHES + 1, TOKEN_DIM):
        encoded = encoded[:, 1:]
    if encoded.shape != (batch * VIEWS, PATCHES, TOKEN_DIM):
        raise V170EvaluationError("frozen V12 tokens must be [B*2,256,192]")
    if not bool(torch.isfinite(encoded).all()):
        raise V170EvaluationError("frozen V12 tokens are non-finite")
    return encoded.detach().reshape(batch, VIEWS, PATCHES, TOKEN_DIM).float()


def _chunks(rows: np.ndarray, batch_size: int) -> Sequence[np.ndarray]:
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise V170EvaluationError("batch_size must be positive")
    return tuple(rows[start : start + int(batch_size)] for start in range(0, len(rows), int(batch_size)))


def _donor_map(rows: np.ndarray) -> dict[int, int]:
    donors = np.roll(rows, 1) if len(rows) > 1 else rows.copy()
    return {int(row): int(donor) for row, donor in zip(rows, donors, strict=True)}


def _stats(values: np.ndarray) -> dict[str, float | int]:
    clean = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(clean) == 0 or not np.isfinite(clean).all():
        raise V170EvaluationError("evaluation statistic has no finite observations")
    return {
        "count": int(len(clean)),
        "mean": float(clean.mean()),
        "median": float(np.median(clean)),
        "p90": float(np.quantile(clean, 0.9)),
    }


def _tip_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    visible: np.ndarray,
    view_valid: np.ndarray,
    predicted_occluded: np.ndarray,
) -> dict[str, Any]:
    if predicted.shape != target.shape or predicted.shape[-3:] != (VIEWS, TIPS, 2):
        raise V170EvaluationError("F/M prediction and target shapes differ")
    active = np.asarray(view_valid, dtype=bool)[..., None] & np.asarray(visible, dtype=bool)
    epe = np.linalg.norm(predicted - target, axis=-1)
    if not bool(active.any()):
        raise V170EvaluationError("tip panel has no visible labels")
    occ_active = np.asarray(view_valid, dtype=bool)[..., None].repeat(TIPS, axis=-1)
    occurrence = np.asarray(predicted_occluded, dtype=bool)
    truth_occ = ~np.asarray(visible, dtype=bool)
    per_view: dict[str, Any] = {}
    for index, name in enumerate(("wrist", "column")):
        mask = active[..., index, :]
        per_view[name] = _stats(epe[..., index, :][mask]) if bool(mask.any()) else {"status": "INADEQUATE"}
    return {
        "ordered_fm_epe_px": _stats(epe[active]),
        "per_view": per_view,
        "occlusion_accuracy": float((occurrence == truth_occ)[occ_active].mean()),
        "occlusion_count": int(occ_active.sum()),
    }


def _static_pass(
    model: V170RecurrentQueryFCPG,
    encoder: Any,
    adapter: Any,
    rows: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    donors: Mapping[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    visible_labels: list[np.ndarray] = []
    views_all: list[np.ndarray] = []
    occurrence: list[np.ndarray] = []
    for indices in _chunks(rows, batch_size):
        batch = _batch(adapter, indices, validation=True)
        size = len(indices)
        _split(batch, size)
        source = batch
        if donors is not None:
            source = _batch(
                adapter,
                np.asarray([donors[int(index)] for index in indices], dtype=np.int64),
                validation=True,
            )
            _split(source, size)
        rgb = _rgb(source, "rgb", (size, VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE), device)
        nominal = _tensor(
            batch,
            "nominal_fm_px",
            (size, VIEWS, TIPS, 2),
            device=device,
            dtype=torch.float32,
        )
        target = _numpy(batch, "fm_px", (size, VIEWS, TIPS, 2)).astype(np.float32)
        visible = _numpy(batch, "tip_valid", (size, VIEWS, TIPS))
        if visible.dtype != np.bool_:
            raise V170EvaluationError("tip_valid must be boolean")
        view_valid_np = _numpy(batch, "view_valid", (size, VIEWS))
        if view_valid_np.dtype != np.bool_:
            raise V170EvaluationError("view_valid must be boolean")
        view_valid = torch.as_tensor(view_valid_np, device=device, dtype=torch.bool)
        tokens = _v12_tokens(encoder, rgb)
        query = model.initial_query(size, device=device, dtype=torch.float32)
        output = model(
            rgb,
            tokens,
            nominal,
            torch.zeros(size, COMMAND_STEPS, STATE_DIM, device=device),
            torch.zeros(size, COMMAND_STEPS + 1, STATE_DIM, device=device),
            query,
            torch.zeros(size, dtype=torch.bool, device=device),
            view_valid,
        )
        predictions.append(output.fm_px.detach().cpu().numpy())
        targets.append(target)
        visible_labels.append(visible.astype(bool))
        views_all.append(view_valid_np.astype(bool))
        occurrence.append(
            (output.categorical_logits.argmax(dim=-1) == OCCLUDED_INDEX)
            .detach()
            .cpu()
            .numpy()
        )
    return tuple(np.concatenate(values, axis=0) for values in (predictions, targets, visible_labels, views_all, occurrence))  # type: ignore[return-value]


@torch.inference_mode()
def evaluate_static_cad_validation(
    checkpoint: str | Path,
    adapter: Any,
    frozen_v12_encoder: Any,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run final static-CAD F/M inference plus a deterministic donor-image test."""

    target_device = torch.device(device)
    model = load_v170_final_model(checkpoint, stage="fcpg", device=target_device)
    assert isinstance(model, V170RecurrentQueryFCPG)
    _frozen_encoder(frozen_v12_encoder)
    rows = _population_rows(adapter, "validation_rows")
    donors = _donor_map(rows)
    factual = _static_pass(
        model, frozen_v12_encoder, adapter, rows, batch_size=batch_size, device=target_device, donors=None
    )
    shuffled = _static_pass(
        model, frozen_v12_encoder, adapter, rows, batch_size=batch_size, device=target_device, donors=donors
    )
    result = {
        "population": "CAD_VALIDATION",
        "rows": int(len(rows)),
        "factual": _tip_metrics(*factual),
        "image_shuffled": _tip_metrics(*shuffled),
        "image_shuffle_changed_fraction": float(
            np.mean(np.asarray([donors[int(row)] != int(row) for row in rows], dtype=np.float64))
        ),
    }
    result["image_shuffle_mean_epe_delta_px"] = float(
        result["image_shuffled"]["ordered_fm_epe_px"]["mean"]
        - result["factual"]["ordered_fm_epe_px"]["mean"]
    )
    return result


def _causal_condition(
    model: V170RecurrentQueryFCPG,
    rgb: torch.Tensor,
    tokens: torch.Tensor,
    nominal: torch.Tensor,
    commands: torch.Tensor,
    paths: torch.Tensor,
    view_valid: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    batch = len(rgb)
    query = model.initial_query(batch, device=rgb.device, dtype=torch.float32)
    predictions: list[np.ndarray] = []
    occurrence: list[np.ndarray] = []
    for horizon in range(OBSERVATIONS):
        if horizon == 0:
            transition_commands = torch.zeros(
                batch, COMMAND_STEPS, STATE_DIM, device=rgb.device
            )
            transition_path = torch.zeros(
                batch, COMMAND_STEPS + 1, STATE_DIM, device=rgb.device
            )
            transition_valid = torch.zeros(batch, dtype=torch.bool, device=rgb.device)
        else:
            transition_commands = commands[:, horizon - 1]
            transition_path = paths[:, horizon - 1]
            transition_valid = torch.ones(batch, dtype=torch.bool, device=rgb.device)
        output = model(
            rgb[:, horizon],
            tokens[:, horizon],
            nominal[:, horizon],
            transition_commands,
            transition_path,
            query,
            transition_valid,
            view_valid[:, horizon],
        )
        predictions.append(output.fm_px.detach().cpu().numpy())
        occurrence.append(
            (output.categorical_logits.argmax(dim=-1) == OCCLUDED_INDEX)
            .detach()
            .cpu()
            .numpy()
        )
        query = output.next_query
    return np.stack(predictions, axis=1), np.stack(occurrence, axis=1)


@torch.inference_mode()
def evaluate_causal_cad_validation(
    checkpoint: str | Path,
    adapter: Any,
    frozen_v12_encoder: Any,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 16,
) -> dict[str, Any]:
    """Evaluate actual H0--H4 recurrence under factual/zero/shuffled actions."""

    target_device = torch.device(device)
    model = load_v170_final_model(checkpoint, stage="fcpg", device=target_device)
    assert isinstance(model, V170RecurrentQueryFCPG)
    _frozen_encoder(frozen_v12_encoder)
    rows = _population_rows(adapter, "validation_rows")
    donors = _donor_map(rows)
    accumulated: dict[str, list[np.ndarray]] = {
        name: [] for name in ("target", "visible", "views", "factual", "factual_occ", "zero", "zero_occ", "shuffled", "shuffled_occ")
    }
    for indices in _chunks(rows, batch_size):
        batch = _batch(adapter, indices)
        size = len(indices)
        _split(batch, size)
        donor = _batch(
            adapter,
            np.asarray([donors[int(index)] for index in indices], dtype=np.int64),
        )
        _split(donor, size)
        rgb = _rgb(
            batch,
            "rgb",
            (size, OBSERVATIONS, VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE),
            target_device,
        )
        flat = rgb.reshape(size * OBSERVATIONS, VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE)
        tokens = _v12_tokens(frozen_v12_encoder, flat).reshape(
            size, OBSERVATIONS, VIEWS, PATCHES, TOKEN_DIM
        )
        nominal = _tensor(
            batch,
            "nominal_fm_px",
            (size, OBSERVATIONS, VIEWS, TIPS, 2),
            device=target_device,
            dtype=torch.float32,
        )
        commands = _tensor(
            batch,
            "completed_commands",
            (size, TRANSITIONS, COMMAND_STEPS, STATE_DIM),
            device=target_device,
            dtype=torch.float32,
        )
        paths = _tensor(
            batch,
            "measured_joint_path",
            (size, TRANSITIONS, COMMAND_STEPS + 1, STATE_DIM),
            device=target_device,
            dtype=torch.float32,
        )
        donor_commands = _tensor(
            donor,
            "completed_commands",
            (size, TRANSITIONS, COMMAND_STEPS, STATE_DIM),
            device=target_device,
            dtype=torch.float32,
        )
        donor_paths = _tensor(
            donor,
            "measured_joint_path",
            (size, TRANSITIONS, COMMAND_STEPS + 1, STATE_DIM),
            device=target_device,
            dtype=torch.float32,
        )
        views = _bool(
            batch,
            "view_valid",
            (size, OBSERVATIONS, VIEWS),
            target_device,
        )
        target = _numpy(
            batch, "fm_px", (size, OBSERVATIONS, VIEWS, TIPS, 2)
        ).astype(np.float32)
        visible = _numpy(
            batch, "tip_valid", (size, OBSERVATIONS, VIEWS, TIPS)
        )
        if visible.dtype != np.bool_:
            raise V170EvaluationError("tip_valid must be boolean")
        factual, factual_occ = _causal_condition(
            model, rgb, tokens, nominal, commands, paths, views
        )
        zero, zero_occ = _causal_condition(
            model, rgb, tokens, nominal, torch.zeros_like(commands), torch.zeros_like(paths), views
        )
        shuffled, shuffled_occ = _causal_condition(
            model, rgb, tokens, nominal, donor_commands, donor_paths, views
        )
        accumulated["target"].append(target)
        accumulated["visible"].append(visible.astype(bool))
        accumulated["views"].append(views.cpu().numpy())
        accumulated["factual"].append(factual)
        accumulated["factual_occ"].append(factual_occ)
        accumulated["zero"].append(zero)
        accumulated["zero_occ"].append(zero_occ)
        accumulated["shuffled"].append(shuffled)
        accumulated["shuffled_occ"].append(shuffled_occ)
    values = {name: np.concatenate(parts, axis=0) for name, parts in accumulated.items()}
    result: dict[str, Any] = {
        "population": "CAUSAL_CAD_VALIDATION",
        "rows": int(len(rows)),
        "action_shuffle_changed_fraction": float(
            np.mean(np.asarray([donors[int(row)] != int(row) for row in rows], dtype=np.float64))
        ),
    }
    for condition in ("factual", "zero", "shuffled"):
        metric = _tip_metrics(
            values[condition], values["target"], values["visible"], values["views"], values[f"{condition}_occ"]
        )
        per_horizon: dict[str, Any] = {}
        for horizon in range(OBSERVATIONS):
            per_horizon[f"H{horizon}"] = _tip_metrics(
                values[condition][:, horizon],
                values["target"][:, horizon],
                values["visible"][:, horizon],
                values["views"][:, horizon],
                values[f"{condition}_occ"][:, horizon],
            )
        metric["per_horizon"] = per_horizon
        result[condition] = metric
    result["zero_action_mean_epe_delta_px"] = float(
        result["zero"]["ordered_fm_epe_px"]["mean"]
        - result["factual"]["ordered_fm_epe_px"]["mean"]
    )
    result["shuffled_action_mean_epe_delta_px"] = float(
        result["shuffled"]["ordered_fm_epe_px"]["mean"]
        - result["factual"]["ordered_fm_epe_px"]["mean"]
    )
    return result


def _control_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "smooth_l1": float(np.where(np.abs(error) < 1.0, 0.5 * error**2, np.abs(error) - 0.5).mean()),
        "mae": float(np.abs(error).mean()),
        "terminal_l2": float(np.linalg.norm(error[:, -1], axis=-1).mean()),
    }


@torch.inference_mode()
def evaluate_lewm_dev(
    checkpoint: str | Path,
    adapter: Any,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Run descriptive LeWM DEV errors with factual/zero/shuffled history."""

    target_device = torch.device(device)
    model = load_v170_final_model(checkpoint, stage="lewm", device=target_device)
    assert isinstance(model, StateFKLeWM)
    rows = _population_rows(adapter, "dev_rows")
    donors = _donor_map(rows)
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in ("target", "factual", "zero", "shuffled")}
    for indices in _chunks(rows, batch_size):
        batch = _batch(adapter, indices)
        size = len(indices)
        _split(batch, size)
        donor = _batch(adapter, np.asarray([donors[int(index)] for index in indices], dtype=np.int64))
        _split(donor, size)
        current = _tensor(batch, "current_state", (size, STATE_DIM), device=target_device, dtype=torch.float32)
        past = _tensor(batch, "past_completed_commands", (size, 2, STATE_DIM), device=target_device, dtype=torch.float32)
        candidate = _tensor(batch, "candidate_commands", (size, COMMAND_STEPS, STATE_DIM), device=target_device, dtype=torch.float32)
        target = _tensor(batch, "measured_state_path", (size, COMMAND_STEPS + 1, STATE_DIM), device=target_device, dtype=torch.float32)
        donor_past = _tensor(donor, "past_completed_commands", (size, 2, STATE_DIM), device=target_device, dtype=torch.float32)
        if not torch.allclose(target[:, 0], current, rtol=0.0, atol=1.0e-6):
            raise V170EvaluationError("LeWM DEV q0 differs from current_state")
        conditions = {"factual": past, "zero": torch.zeros_like(past), "shuffled": donor_past}
        outputs["target"].append(target.cpu().numpy())
        for name, history in conditions.items():
            prediction = model(
                current_state=current,
                past_completed_commands=history,
                candidate_commands=candidate,
            ).predicted_state_path
            outputs[name].append(prediction.cpu().numpy())
    values = {name: np.concatenate(parts, axis=0) for name, parts in outputs.items()}
    result: dict[str, Any] = {"population": "ARCHIVE_DEV", "rows": int(len(rows))}
    for name in ("factual", "zero", "shuffled"):
        result[name] = _control_metrics(values[name][:, 1:], values["target"][:, 1:])
    result["zero_history_mae_delta"] = float(result["zero"]["mae"] - result["factual"]["mae"])
    result["shuffled_history_mae_delta"] = float(result["shuffled"]["mae"] - result["factual"]["mae"])
    return result


def _actor_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    repeat: np.ndarray,
) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    previous_prediction = np.concatenate((current[:, None], prediction[:, :-1]), axis=1)
    previous_target = np.concatenate((current[:, None], target[:, :-1]), axis=1)
    repeat_mae = np.abs(np.asarray(repeat, dtype=np.float64) - target).mean()
    mae = np.abs(error).mean()
    return {
        "action_l1": float(mae),
        "velocity_l1": float(np.abs((prediction - previous_prediction) - (target - previous_target)).mean()),
        "repeat_action_l1": float(repeat_mae),
        "improvement_over_repeat_l1": float(repeat_mae - mae),
    }


@torch.inference_mode()
def evaluate_actor_dev(
    checkpoint: str | Path,
    adapter: Any,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Run descriptive Actor DEV errors with factual/zero/shuffled histories."""

    target_device = torch.device(device)
    model = load_v170_final_model(checkpoint, stage="actor", device=target_device)
    assert isinstance(model, FullHistoryFiveCommandActor)
    rows = _population_rows(adapter, "dev_rows")
    donors = _donor_map(rows)
    outputs: dict[str, list[np.ndarray]] = {
        name: [] for name in ("target", "current", "repeat", "factual", "zero", "shuffled")
    }
    for indices in _chunks(rows, batch_size):
        batch = _batch(adapter, indices)
        size = len(indices)
        _split(batch, size)
        donor = _batch(adapter, np.asarray([donors[int(index)] for index in indices], dtype=np.int64))
        _split(donor, size)
        scene = _tensor(batch, "current_full256", (size, VIEWS, PATCHES, TOKEN_DIM), device=target_device, dtype=torch.float32)
        view_valid = _bool(batch, "view_valid", (size, VIEWS), target_device)
        current = _tensor(batch, "current_state", (size, STATE_DIM), device=target_device, dtype=torch.float32)
        past = _tensor(batch, "past_completed_commands", (size, 2, STATE_DIM), device=target_device, dtype=torch.float32)
        history = _tensor(batch, "completed_command_history", (size, 3, COMMAND_STEPS, STATE_DIM), device=target_device, dtype=torch.float32)
        history_valid = _bool(batch, "history_valid", (size, 3), target_device)
        transition = _tensor(batch, "transition", (size,), device=target_device, dtype=torch.int64)
        goal = _tensor(batch, "desired_joint_goal", (size, STATE_DIM), device=target_device, dtype=torch.float32)
        target = _tensor(batch, "completed_commands", (size, COMMAND_STEPS, STATE_DIM), device=target_device, dtype=torch.float32)
        donor_history = _tensor(donor, "completed_command_history", (size, 3, COMMAND_STEPS, STATE_DIM), device=target_device, dtype=torch.float32)
        donor_history = donor_history * history_valid[:, :, None, None].to(torch.float32)
        conditions = {
            "factual": history,
            "zero": torch.zeros_like(history),
            "shuffled": donor_history,
        }
        outputs["target"].append(target.cpu().numpy())
        outputs["current"].append(current.cpu().numpy())
        repeat: np.ndarray | None = None
        for name, condition in conditions.items():
            result = model(
                current_full256=scene,
                view_valid=view_valid,
                current_state=current,
                past_completed_commands=past,
                completed_command_history=condition,
                history_valid=history_valid,
                transition=transition,
                desired_joint_goal=goal,
                baseline_kind="repeat",
                baseline_joint_goal=None,
                park_mask=None,
            )
            outputs[name].append(result.action.cpu().numpy())
            if repeat is None:
                repeat = result.baseline.cpu().numpy()
        assert repeat is not None
        outputs["repeat"].append(repeat)
    values = {name: np.concatenate(parts, axis=0) for name, parts in outputs.items()}
    result: dict[str, Any] = {"population": "ARCHIVE_DEV", "rows": int(len(rows))}
    for name in ("factual", "zero", "shuffled"):
        result[name] = _actor_metrics(
            values[name], values["target"], values["current"], values["repeat"]
        )
    result["zero_history_action_l1_delta"] = float(
        result["zero"]["action_l1"] - result["factual"]["action_l1"]
    )
    result["shuffled_history_action_l1_delta"] = float(
        result["shuffled"]["action_l1"] - result["factual"]["action_l1"]
    )
    return result


__all__ = [
    "V170EvaluationError",
    "V170FinalModels",
    "evaluate_actor_dev",
    "evaluate_causal_cad_validation",
    "evaluate_lewm_dev",
    "evaluate_static_cad_validation",
    "load_v170_final_model",
    "load_v170_final_models",
]
