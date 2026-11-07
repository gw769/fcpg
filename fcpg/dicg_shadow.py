"""Production-shadow-only DICG hypothesis scorer sidecar.

This runner never returns a planner-consumable score, never mutates the
caller's observation, and never raises out of ``submit``.  Adjacent live
observations are paired on a single-slot queue; a newer observation may
drop an unprocessed predecessor.  All records are uncalibrated logit-space
audit lines.  Fail-closed excitation gating matches the training contract:
arm ``Δq`` of the first five joints must be at least ``0.2``, and at least
two valid CAD hypotheses must have dual-tip displacement
``excitation_px >= 1.0`` before the scorer runs.  A gated pair writes
``record_kind=skip`` and never increments ``pairs_ok`` / ``pairs_failed`` /
``over_budget``.

``control_influence`` and ``planner_dataflow_influence`` are ``none``: the
public API is ``submit`` / ``status`` / ``close``, and ``submit`` returns
only a bool.  That is not a hard-realtime isolation claim.  The worker is
an in-process daemon thread (``process_isolated=false``), so
``resource_contention_possible`` and ``timing_influence_possible`` stay true.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fcpg.action_conditioned_hypothesis_scorer import (
    CHECKPOINT_FORMAT,
    DEFAULT_MIN_EXCITATION_PX,
    MIN_RANKABLE,
    TRAINING_TARGET,
    select_amp_dtype,
    sha256_file,
    validate_checkpoint_metadata,
)


SCHEMA = "fcpg.dicg_production_shadow.v1"
METRIC_SPACE = "uncalibrated_hypothesis_logit"
CONTROL_INFLUENCE = "none"
PLANNER_DATAFLOW_INFLUENCE = "none"
COLUMN_VIEW = "column"
IMAGE_HW = 224
JOINT_DIM = 6
ARM_JOINT_COUNT = 5
QUEUE_MAXSIZE = 1
DEFAULT_TOP_K = 5
DEFAULT_LATENCY_BUDGET_S = 0.10
DEFAULT_MIN_ARM_DQ = 0.2
SKIP_INSUFFICIENT_ARM_DQ = "insufficient_arm_dq"
SKIP_INSUFFICIENT_EXCITATION = "insufficient_excitation"
SCORE_RECORD_KEYS = (
    "logits",
    "hypothesis_logits",
    "uncalibrated_scale",
    "motion_innovation_px",
    "top1_index",
    "top1_logit",
    "topk_indices",
    "topk_logits",
    "tip_observability_sigmoid",
)
_WORKER_POLL_S = 0.05
_CLOSE_JOIN_S = 5.0

_METADATA_MUST_BE_FALSE = (
    "live_ready",
    "uncertainty_calibrated",
    "has_physical_labels",
    "absolute_tip_accuracy_claim",
)

__all__ = [
    "ARM_JOINT_COUNT",
    "CONTROL_INFLUENCE",
    "DEFAULT_MIN_ARM_DQ",
    "DEFAULT_MIN_EXCITATION_PX",
    "DICGShadowEngine",
    "DICGShadowError",
    "DICGShadowRunner",
    "METRIC_SPACE",
    "MIN_RANKABLE",
    "PLANNER_DATAFLOW_INFLUENCE",
    "SCHEMA",
    "SCORE_RECORD_KEYS",
    "SKIP_INSUFFICIENT_ARM_DQ",
    "SKIP_INSUFFICIENT_EXCITATION",
    "arm_dq_norm",
    "cad_excitation_px",
]


class DICGShadowError(RuntimeError):
    """Raised only while constructing the sidecar.  Never from ``submit``."""


class _CopiedObservation:
    __slots__ = ("timestamp_s", "state", "column")

    def __init__(
        self, timestamp_s: float, state: np.ndarray, column: np.ndarray
    ) -> None:
        self.timestamp_s = float(timestamp_s)
        self.state = state
        self.column = column


def _honesty_fields() -> dict[str, Any]:
    """Fixed sidecar contract.  Honesty always overwrites caller keys.

    ``control_influence=none`` means no score/command enters the planner.
    ``process_isolated=false`` means the worker still shares the host
    process, so contention and timing jitter remain possible.
    """
    return {
        "shadow_only": True,
        "control_influence": CONTROL_INFLUENCE,
        "planner_dataflow_influence": PLANNER_DATAFLOW_INFLUENCE,
        "process_isolated": False,
        "resource_contention_possible": True,
        "timing_influence_possible": True,
        "heldout_gate_passed": False,
        "live_ready": False,
        "interpretation_licensed": False,
        "physical_tip_accuracy_claim": False,
        "uncertainty_calibrated": False,
        "metric_space": METRIC_SPACE,
    }


def _stamp_honesty(record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(record) if record is not None else {}
    payload.update(_honesty_fields())
    return payload


def arm_dq_norm(delta_q: Any) -> float:
    """L2 norm of the first five arm joints.  Matches ``build_dicg_motion_dataset``."""

    joints = np.asarray(delta_q, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(joints[:ARM_JOINT_COUNT]))


def cad_excitation_px(candidate_t: Any, candidate_tp1: Any) -> np.ndarray:
    """Per-hypothesis CAD dual-tip displacement.  Same as training ``excitation_px``.

    ``predicted = candidate_tp1 - candidate_t`` with shape ``[N, 2, 2]``;
    excitation is the L2 norm of each flattened 4-vector.
    """

    source = np.asarray(candidate_t, dtype=np.float64)
    target = np.asarray(candidate_tp1, dtype=np.float64)
    if source.shape != target.shape:
        raise DICGShadowError(
            f"CAD candidate shapes must match, got t={source.shape} tp1={target.shape}"
        )
    if source.ndim != 3 or source.shape[-2:] != (2, 2):
        raise DICGShadowError(
            f"CAD candidates must be [N,2,2], got {source.shape}"
        )
    predicted = target - source
    n_hyp = int(predicted.shape[0])
    return np.linalg.norm(predicted.reshape(n_hyp, -1), axis=-1).astype(np.float32)


def rankable_excitation_count(
    valid: Any,
    excitation_px: Any,
    *,
    min_excitation_px: float = DEFAULT_MIN_EXCITATION_PX,
) -> int:
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    excitation = np.asarray(excitation_px, dtype=np.float32).reshape(-1)
    if mask.shape != excitation.shape:
        raise DICGShadowError(
            f"valid/excitation length mismatch: {mask.shape} vs {excitation.shape}"
        )
    return int(np.count_nonzero(mask & (excitation >= float(min_excitation_px))))


def _resolve_device(name: str) -> torch.device:
    label = str(name).strip()
    if label not in {"cpu", "cuda:0"}:
        raise DICGShadowError(f"device must be 'cpu' or 'cuda:0', got {name!r}")
    device = torch.device(label)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise DICGShadowError("cuda:0 requested but CUDA is unavailable")
        if int(torch.cuda.device_count()) < 1:
            raise DICGShadowError("cuda:0 requested but no CUDA device is visible")
    return device


def _require_false_metadata(metadata: Mapping[str, Any]) -> None:
    for key in _METADATA_MUST_BE_FALSE:
        if key not in metadata:
            raise DICGShadowError(f"refusing to start: checkpoint metadata missing {key}")
        if metadata[key] is not False:
            raise DICGShadowError(
                f"refusing to start: checkpoint metadata {key}={metadata[key]!r} "
                "must be False"
            )
    if metadata.get("training_target") != TRAINING_TARGET:
        raise DICGShadowError(
            "refusing to start: training_target="
            f"{metadata.get('training_target')!r} must be {TRAINING_TARGET!r}"
        )
    fmt = metadata.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise DICGShadowError(
            f"refusing to start: checkpoint format {fmt!r} must be "
            f"{CHECKPOINT_FORMAT!r}"
        )


def _validate_payload_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise DICGShadowError("refusing to start: checkpoint missing metadata mapping")
    _require_false_metadata(metadata)
    try:
        validate_checkpoint_metadata(metadata)
    except Exception as exc:
        raise DICGShadowError(
            f"refusing to start: checkpoint metadata rejected ({type(exc).__name__}: {exc})"
        ) from exc
    top_target = payload.get("training_target")
    if top_target not in {None, TRAINING_TARGET}:
        raise DICGShadowError(
            f"refusing to start: payload training_target={top_target!r} "
            f"must be {TRAINING_TARGET!r}"
        )
    return dict(metadata)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _as_array(name: str, value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = np.array(value, copy=True)
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    if array.dtype.kind in {"f", "c"} and not np.all(np.isfinite(array)):
        raise DICGShadowError(f"{name} contains non-finite values")
    return array


def _copy_observation(observation: Any) -> _CopiedObservation:
    if isinstance(observation, Mapping):
        timestamp = observation.get("timestamp_s")
        state = observation.get("state")
        frames = observation.get("frames")
    else:
        timestamp = getattr(observation, "timestamp_s", None)
        state = getattr(observation, "state", None)
        frames = getattr(observation, "frames", None)
    if timestamp is None or not np.isfinite(float(timestamp)):
        raise DICGShadowError("observation timestamp_s must be finite")
    joints = _as_array("state", state, dtype=np.float32)
    if joints.shape != (JOINT_DIM,):
        raise DICGShadowError(f"q readback must be shape {(JOINT_DIM,)}, got {joints.shape}")
    if not isinstance(frames, Mapping) or COLUMN_VIEW not in frames:
        raise DICGShadowError("observation must provide frames['column']")
    column = np.array(frames[COLUMN_VIEW], copy=True)
    if column.shape != (IMAGE_HW, IMAGE_HW, 3) or column.dtype != np.uint8:
        raise DICGShadowError("column frame must be uint8 [224,224,3]")
    return _CopiedObservation(float(timestamp), joints, column)


def _column_nchw(column: np.ndarray) -> torch.Tensor:
    rgb = np.ascontiguousarray(column.astype(np.float32) / 255.0)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def _geometry_field(result: Any, *names: str) -> Any:
    if isinstance(result, Mapping):
        for name in names:
            if name in result:
                return result[name]
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
    raise DICGShadowError(f"geometry.project_pair result missing {names[0]}")


def _unpack_geometry(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    hypotheses = _as_array(
        "hypotheses",
        _geometry_field(result, "hypotheses", "hypotheses_tool_offsets_m"),
        dtype=np.float32,
    )
    candidate_t = _as_array(
        "candidate_t_px", _geometry_field(result, "candidate_t_px"), dtype=np.float32
    )
    candidate_tp1 = _as_array(
        "candidate_tp1_px", _geometry_field(result, "candidate_tp1_px"), dtype=np.float32
    )
    if hypotheses.ndim != 3 or hypotheses.shape[-2:] != (2, 3):
        raise DICGShadowError(f"hypotheses must be [N,2,3], got {hypotheses.shape}")
    n_hyp = int(hypotheses.shape[0])
    if candidate_t.shape != (n_hyp, 2, 2) or candidate_tp1.shape != (n_hyp, 2, 2):
        raise DICGShadowError(
            "candidate pixels must be [N,2,2] matching hypotheses, "
            f"got t={candidate_t.shape} tp1={candidate_tp1.shape} N={n_hyp}"
        )
    valid: np.ndarray
    try:
        valid_raw = _geometry_field(result, "hypothesis_valid", "valid_mask", "valid")
        valid = np.asarray(valid_raw).reshape(n_hyp).astype(bool, copy=False)
    except DICGShadowError:
        tip_valid = np.asarray(
            _geometry_field(result, "tip_valid"), dtype=bool
        )
        if tip_valid.shape != (n_hyp, 2):
            raise DICGShadowError(
                f"tip_valid must be [N,2], got {tip_valid.shape}"
            ) from None
        valid = np.asarray(tip_valid.all(axis=-1), dtype=bool)
    metadata_raw = {}
    try:
        metadata_raw = _geometry_field(result, "metadata")
    except DICGShadowError:
        metadata_raw = {}
    metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
    return hypotheses, candidate_t, candidate_tp1, valid, metadata


def _tensor_field(output: Any, name: str) -> torch.Tensor:
    value = getattr(output, name, None)
    if value is None and isinstance(output, Mapping):
        value = output.get(name)
    if not torch.is_tensor(value):
        raise DICGShadowError(f"scorer output missing tensor {name}")
    return value.detach()


def _squeeze_batch(name: str, tensor: torch.Tensor, tail: tuple[int, ...]) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == len(tail) + 1 and int(array.shape[0]) == 1:
        array = array[0]
    if array.shape[-len(tail) :] != tail and tail:
        raise DICGShadowError(f"{name} trailing shape {array.shape} does not end with {tail}")
    if not np.all(np.isfinite(array)):
        raise DICGShadowError(f"{name} contains non-finite values")
    return np.asarray(array, dtype=np.float32)


def _select_topk(
    logits: np.ndarray,
    valid: np.ndarray,
    *,
    top_k: int,
) -> tuple[int, float, list[int], list[float]]:
    if logits.shape != valid.shape:
        raise DICGShadowError(
            f"hypothesis logits shape {logits.shape} != valid {valid.shape}"
        )
    eligible = np.where(valid)[0]
    if eligible.size < 1:
        raise DICGShadowError("no valid CAD candidates")
    order = eligible[np.argsort(-logits[eligible], kind="stable")]
    k = max(1, min(int(top_k), int(order.size)))
    chosen = [int(index) for index in order[:k]]
    scores = [float(logits[index]) for index in chosen]
    return chosen[0], scores[0], chosen, scores


def _project_pair(geometry: Any, q_t: np.ndarray, q_tp1: np.ndarray) -> Any:
    if geometry is None:
        try:
            from fcpg.dicg_shadow_geometry import DICGShadowGeometry as _Geometry
        except Exception as exc:
            raise DICGShadowError(
                "DICGShadowGeometry import failed "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        raise DICGShadowError(
            "geometry instance is required "
            f"(pass {_Geometry.__name__}.from_files(...) or a project_pair mock)"
        )
    if not callable(getattr(geometry, "project_pair", None)):
        raise DICGShadowError("geometry must provide project_pair(q_t, q_tp1)")
    try:
        return geometry.project_pair(q_t, q_tp1)
    except DICGShadowError:
        raise
    except Exception as exc:
        raise DICGShadowError(
            f"project_pair failed ({type(exc).__name__}: {exc})"
        ) from exc


def _load_core_and_encoder(
    *,
    payload: Mapping[str, Any],
    vit_checkpoint: Path,
    device: torch.device,
) -> Any:
    from fcpg.action_conditioned_hypothesis_scorer import (
        ActionConditionedHypothesisScorer,
        ActionConditionedHypothesisScorerCore,
    )

    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise DICGShadowError("refusing to start: checkpoint model state_dict is empty")
    leaked = [
        str(key)
        for key in state
        if str(key).startswith(("encoder.", "vit.", "core."))
    ]
    if leaked:
        raise DICGShadowError(
            "refusing to start: checkpoint must be "
            "ActionConditionedHypothesisScorerCore.state_dict, "
            f"not wrapper keys such as {leaked[:4]}"
        )
    core = ActionConditionedHypothesisScorerCore()
    try:
        core.load_state_dict(state, strict=True)
    except Exception as exc:
        raise DICGShadowError(
            f"refusing to start: core state_dict rejected ({type(exc).__name__}: {exc})"
        ) from exc
    core.to(device=device, dtype=torch.float32)
    try:
        model = ActionConditionedHypothesisScorer(vit_checkpoint, core=core)
    except Exception as exc:
        raise DICGShadowError(
            "refusing to start: frozen ViT checkpoint failed "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    to_fn = getattr(model, "to", None)
    if callable(to_fn):
        model = to_fn(device) or model
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        eval_fn()
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        encoder_eval = getattr(encoder, "eval", None)
        if callable(encoder_eval):
            encoder_eval()
    for module in (model, getattr(model, "core", None), encoder):
        if module is None or not hasattr(module, "parameters"):
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return model


def _coerce_observation(item: Any) -> _CopiedObservation:
    if isinstance(item, _CopiedObservation):
        return item
    if isinstance(item, Mapping) and "column" in item and "state" in item:
        timestamp = item.get("timestamp_s")
        if timestamp is None or not np.isfinite(float(timestamp)):
            raise DICGShadowError("observation timestamp_s must be finite")
        joints = _as_array("state", item.get("state"), dtype=np.float32)
        if joints.shape != (JOINT_DIM,):
            raise DICGShadowError(
                f"q readback must be shape {(JOINT_DIM,)}, got {joints.shape}"
            )
        column = np.array(item["column"], copy=True)
        if column.shape != (IMAGE_HW, IMAGE_HW, 3) or column.dtype != np.uint8:
            raise DICGShadowError("column frame must be uint8 [224,224,3]")
        return _CopiedObservation(float(timestamp), joints, column)
    return _copy_observation(item)


class DICGShadowEngine:
    """Threadless checkpoint/ViT/geometry load, pair infer, and JSONL write.

    No queue and no worker thread.  ``DICGShadowRunner`` wraps this engine
    in-process; the spawn sidecar child calls it on its inbound mailbox.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        vit_checkpoint_path: str | Path,
        log_path: str | Path,
        device: str = "cpu",
        geometry: Any | None = None,
        model: Any | None = None,
        top_k: int = DEFAULT_TOP_K,
        latency_budget_s: float = DEFAULT_LATENCY_BUDGET_S,
        honesty_fields: Any | None = None,
        min_arm_dq: float = DEFAULT_MIN_ARM_DQ,
        min_excitation_px: float = DEFAULT_MIN_EXCITATION_PX,
    ) -> None:
        self._honesty_fields = honesty_fields if callable(honesty_fields) else _honesty_fields
        self._device = _resolve_device(device)
        self._top_k = max(1, int(top_k))
        budget = float(latency_budget_s)
        if not np.isfinite(budget) or budget <= 0.0:
            raise DICGShadowError("latency_budget_s must be finite and positive")
        self._latency_budget_s = budget
        arm_gate = float(min_arm_dq)
        excitation_gate = float(min_excitation_px)
        if not np.isfinite(arm_gate) or arm_gate < 0.0:
            raise DICGShadowError("min_arm_dq must be finite and non-negative")
        if not np.isfinite(excitation_gate) or excitation_gate < 0.0:
            raise DICGShadowError("min_excitation_px must be finite and non-negative")
        self._min_arm_dq = arm_gate
        self._min_excitation_px = excitation_gate
        self._amp_dtype = select_amp_dtype(self._device)

        ckpt = Path(checkpoint_path).expanduser().resolve()
        vit = Path(vit_checkpoint_path).expanduser().resolve()
        if not ckpt.is_file():
            raise DICGShadowError(f"refusing to start: checkpoint is not a file: {ckpt}")
        if not vit.is_file():
            raise DICGShadowError(
                f"refusing to start: frozen ViT checkpoint is not a file: {vit}"
            )
        try:
            payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        except Exception as exc:
            raise DICGShadowError(
                f"refusing to start: failed to load checkpoint {ckpt} "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        if not isinstance(payload, Mapping):
            raise DICGShadowError("refusing to start: checkpoint payload is not a mapping")
        metadata = _validate_payload_metadata(payload)
        self._metadata = metadata
        self._checkpoint_path = str(ckpt)
        self._vit_path = str(vit)
        self._checkpoint_sha256 = sha256_file(ckpt)
        self._vit_sha256 = sha256_file(vit)

        if geometry is None:
            try:
                from fcpg.dicg_shadow_geometry import DICGShadowGeometry as _Geometry
            except Exception as exc:
                raise DICGShadowError(
                    "refusing to start: DICGShadowGeometry import failed "
                    f"({type(exc).__name__}: {exc})"
                ) from exc
            raise DICGShadowError(
                "refusing to start: geometry instance is required "
                f"(pass {_Geometry.__name__}.from_files(...) or a project_pair mock)"
            )
        if not callable(getattr(geometry, "project_pair", None)):
            raise DICGShadowError(
                "refusing to start: geometry must provide project_pair(q_t, q_tp1)"
            )
        self._geometry = geometry

        if model is None:
            self._model = _load_core_and_encoder(
                payload=payload, vit_checkpoint=vit, device=self._device
            )
        else:
            self._model = model
            eval_fn = getattr(self._model, "eval", None)
            if callable(eval_fn):
                eval_fn()
            for module in (
                self._model,
                getattr(self._model, "core", None),
                getattr(self._model, "encoder", None),
            ):
                if module is None or not hasattr(module, "parameters"):
                    continue
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

        self._log_path = Path(log_path)
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self._lock = threading.Lock()
        self._pairs_ok = 0
        self._pairs_failed = 0
        self._pairs_skipped = 0
        self._last_skip_reason: str | None = None
        self._skip_reasons = {
            SKIP_INSUFFICIENT_ARM_DQ: 0,
            SKIP_INSUFFICIENT_EXCITATION: 0,
        }
        self._log_errors = 0
        self._over_budget = 0

    @classmethod
    def from_paths(
        cls,
        *,
        checkpoint_path: str | Path,
        vit_checkpoint_path: str | Path,
        log_path: str | Path,
        dataset_contract: str | Path,
        observer_bundle: str | Path,
        mujoco_model: str | Path,
        extrinsic_index: int | str | None = None,
        extrinsic_group_key: str | None = None,
        device: str = "cpu",
        top_k: int = DEFAULT_TOP_K,
        latency_budget_s: float = DEFAULT_LATENCY_BUDGET_S,
        honesty_fields: Any | None = None,
        min_arm_dq: float = DEFAULT_MIN_ARM_DQ,
        min_excitation_px: float = DEFAULT_MIN_EXCITATION_PX,
    ) -> DICGShadowEngine:
        from fcpg.dicg_shadow_geometry import DICGShadowGeometry

        geometry_kwargs: dict[str, Any] = {
            "dataset_contract": dataset_contract,
            "observer_bundle": observer_bundle,
            "mujoco_model": mujoco_model,
        }
        if isinstance(extrinsic_index, str) and extrinsic_group_key is None:
            geometry_kwargs["extrinsic_group_key"] = extrinsic_index
        elif extrinsic_index is not None:
            geometry_kwargs["extrinsic_index"] = int(extrinsic_index)
        if extrinsic_group_key is not None:
            geometry_kwargs["extrinsic_group_key"] = extrinsic_group_key
        geometry = DICGShadowGeometry.from_files(**geometry_kwargs)
        return cls(
            checkpoint_path=checkpoint_path,
            vit_checkpoint_path=vit_checkpoint_path,
            log_path=log_path,
            device=device,
            geometry=geometry,
            top_k=top_k,
            latency_budget_s=latency_budget_s,
            honesty_fields=honesty_fields,
            min_arm_dq=min_arm_dq,
            min_excitation_px=min_excitation_px,
        )

    def _stamp(self, record: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(record) if record is not None else {}
        payload.update(self._honesty_fields())
        return payload

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pairs_ok": int(self._pairs_ok),
                "pairs_failed": int(self._pairs_failed),
                "pairs_skipped": int(self._pairs_skipped),
                "last_skip_reason": self._last_skip_reason,
                "skip_reasons": {
                    str(key): int(value) for key, value in self._skip_reasons.items()
                },
                "log_errors": int(self._log_errors),
                "over_budget": int(self._over_budget),
                "device": str(self._device),
                "checkpoint_path": self._checkpoint_path,
                "vit_path": self._vit_path,
                "checkpoint_sha256": self._checkpoint_sha256,
                "vit_sha256": self._vit_sha256,
                "checkpoint_format": CHECKPOINT_FORMAT,
                "training_target": TRAINING_TARGET,
            }

    def _note_skip(self, reason: str) -> None:
        key = str(reason)
        with self._lock:
            self._pairs_skipped += 1
            self._last_skip_reason = key
            self._skip_reasons[key] = int(self._skip_reasons.get(key, 0)) + 1

    def _skip_record(
        self,
        *,
        obs_t: Any,
        obs_tp1: Any,
        delta_q: np.ndarray,
        reason: str,
        arm_norm: float,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ok": True,
            "record_kind": "skip",
            "skip_reason": str(reason),
            "schema": SCHEMA,
            "timestamp_t_s": float(obs_t.timestamp_s),
            "timestamp_tp1_s": float(obs_tp1.timestamp_s),
            "delta_q": [float(value) for value in np.asarray(delta_q, dtype=np.float64).reshape(-1).tolist()],
            "arm_dq_norm": float(arm_norm),
            "min_arm_dq": float(self._min_arm_dq),
        }
        if extra is not None:
            record.update(dict(extra))
        for key in SCORE_RECORD_KEYS:
            record.pop(key, None)
        return self._stamp(record)

    def infer_pair(self, obs_t: Any, obs_tp1: Any) -> dict[str, Any]:
        obs_t = _coerce_observation(obs_t)
        obs_tp1 = _coerce_observation(obs_tp1)
        q_t = np.array(obs_t.state, dtype=np.float32, copy=True)
        q_tp1 = np.array(obs_tp1.state, dtype=np.float32, copy=True)
        delta_q = q_tp1 - q_t
        arm_norm = arm_dq_norm(delta_q)
        if arm_norm < self._min_arm_dq:
            return self._skip_record(
                obs_t=obs_t,
                obs_tp1=obs_tp1,
                delta_q=delta_q,
                reason=SKIP_INSUFFICIENT_ARM_DQ,
                arm_norm=arm_norm,
            )
        projected = _project_pair(self._geometry, q_t, q_tp1)
        hypotheses, candidate_t, candidate_tp1, valid, geometry_metadata = _unpack_geometry(
            projected
        )
        valid_count = int(np.count_nonzero(valid))
        excitation = cad_excitation_px(candidate_t, candidate_tp1)
        excited_valid = rankable_excitation_count(
            valid,
            excitation,
            min_excitation_px=self._min_excitation_px,
        )
        if excited_valid < int(MIN_RANKABLE):
            return self._skip_record(
                obs_t=obs_t,
                obs_tp1=obs_tp1,
                delta_q=delta_q,
                reason=SKIP_INSUFFICIENT_EXCITATION,
                arm_norm=arm_norm,
                extra={
                    "excited_valid_count": int(excited_valid),
                    "min_excitation_px": float(self._min_excitation_px),
                    "n_hypotheses": int(hypotheses.shape[0]),
                    "valid_candidate_count": valid_count,
                },
            )
        image_t = _column_nchw(obs_t.column).to(self._device)
        image_tp1 = _column_nchw(obs_tp1.column).to(self._device)
        hyp_t = torch.from_numpy(np.ascontiguousarray(hypotheses)).to(
            device=self._device, dtype=torch.float32
        )
        cand_t = torch.from_numpy(np.ascontiguousarray(candidate_t)).unsqueeze(0).to(
            device=self._device, dtype=torch.float32
        )
        cand_tp1 = torch.from_numpy(np.ascontiguousarray(candidate_tp1)).unsqueeze(0).to(
            device=self._device, dtype=torch.float32
        )
        delta_t = torch.from_numpy(np.ascontiguousarray(delta_q)).unsqueeze(0).to(
            device=self._device, dtype=torch.float32
        )
        model = self._model
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()
        use_amp = self._device.type == "cuda" and self._amp_dtype is torch.float16
        autocast = torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        )
        with torch.no_grad():
            with autocast:
                feature_t, feature_tp1 = self._encode_pair(image_t, image_tp1)
            if not torch.isfinite(feature_t).all() or not torch.isfinite(feature_tp1).all():
                raise DICGShadowError("frozen ViT produced non-finite features")
            core = getattr(model, "core", model)
            output = core(
                feature_t.float(),
                feature_tp1.float(),
                cand_t,
                cand_tp1,
                delta_t,
                hyp_t,
            )
        logits = _squeeze_batch(
            "hypothesis_logits", _tensor_field(output, "hypothesis_logits"), (hypotheses.shape[0],)
        )
        if logits.ndim != 1:
            logits = np.asarray(logits, dtype=np.float32).reshape(hypotheses.shape[0])
        tip_logits = _squeeze_batch(
            "tip_validity_logits",
            _tensor_field(output, "tip_validity_logits"),
            (hypotheses.shape[0], 2),
        )
        motion = _squeeze_batch(
            "motion_innovation_px",
            _tensor_field(output, "motion_innovation_px"),
            (hypotheses.shape[0], 2, 2),
        )
        scale = _squeeze_batch(
            "uncalibrated_scale",
            _tensor_field(output, "uncalibrated_scale"),
            (hypotheses.shape[0],),
        )
        if scale.ndim != 1:
            scale = np.asarray(scale, dtype=np.float32).reshape(hypotheses.shape[0])
        top1_index, top1_logit, topk_indices, topk_logits = _select_topk(
            logits, valid, top_k=self._top_k
        )
        tip_prob = 1.0 / (1.0 + np.exp(-np.clip(tip_logits, -40.0, 40.0)))
        record = {
            "ok": True,
            "record_kind": "success",
            "schema": SCHEMA,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_path": self._checkpoint_path,
            "vit_path": self._vit_path,
            "checkpoint_sha256": self._checkpoint_sha256,
            "vit_sha256": self._vit_sha256,
            "timestamp_t_s": float(obs_t.timestamp_s),
            "timestamp_tp1_s": float(obs_tp1.timestamp_s),
            "delta_q": [float(value) for value in delta_q.tolist()],
            "top1_index": int(top1_index),
            "top1_logit": float(top1_logit),
            "topk_indices": [int(index) for index in topk_indices],
            "topk_logits": [float(value) for value in topk_logits],
            "tip_observability_sigmoid": _jsonable(tip_prob),
            "motion_innovation_px": _jsonable(motion),
            "uncalibrated_scale": _jsonable(scale),
            "valid_candidate_count": valid_count,
            "n_hypotheses": int(hypotheses.shape[0]),
            "geometry_metadata": _jsonable(geometry_metadata),
        }
        return self._stamp(record)

    def safe_process_pair(self, obs_t: Any, obs_tp1: Any) -> None:
        obs_t = _coerce_observation(obs_t)
        obs_tp1 = _coerce_observation(obs_tp1)
        started = time.perf_counter()
        try:
            record = self.infer_pair(obs_t, obs_tp1)
            latency_s = float(time.perf_counter() - started)
            record["latency_s"] = latency_s
            if record.get("record_kind") == "skip":
                record["over_budget"] = False
                self.append_jsonl(self._stamp(record))
                self._note_skip(str(record.get("skip_reason") or "skip"))
                return
            over_budget = bool(latency_s > self._latency_budget_s)
            record["over_budget"] = over_budget
            if over_budget:
                with self._lock:
                    self._over_budget += 1
            self.append_jsonl(self._stamp(record))
            with self._lock:
                self._pairs_ok += 1
        except Exception as exc:
            latency_s = float(time.perf_counter() - started)
            over_budget = bool(latency_s > self._latency_budget_s)
            with self._lock:
                self._pairs_failed += 1
                if over_budget:
                    self._over_budget += 1
            self.append_jsonl(
                self._stamp(
                    {
                        "ok": False,
                        "record_kind": "failure",
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                        "timestamp_t_s": float(obs_t.timestamp_s),
                        "timestamp_tp1_s": float(obs_tp1.timestamp_s),
                        "latency_s": latency_s,
                        "over_budget": over_budget,
                    }
                )
            )

    def _encode_pair(
        self, image_t: torch.Tensor, image_tp1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self._model
        encode_pair = getattr(model, "encode_pair", None)
        if callable(encode_pair):
            pair = encode_pair(image_t, image_tp1)
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise DICGShadowError("encode_pair must return (feature_t, feature_tp1)")
            return pair[0], pair[1]
        encoder = getattr(model, "encoder", None)
        if encoder is None or not callable(encoder):
            raise DICGShadowError("model must provide encode_pair or encoder")
        encoder_eval = getattr(encoder, "eval", None)
        if callable(encoder_eval):
            encoder_eval()
        return encoder(image_t), encoder(image_tp1)

    def append_jsonl(self, record: Mapping[str, Any]) -> None:
        payload = self._stamp({"schema": SCHEMA, **dict(record)})
        payload["schema"] = SCHEMA
        try:
            line = json.dumps(_jsonable(payload), ensure_ascii=True, allow_nan=False)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
        except Exception:
            with self._lock:
                self._log_errors += 1


class DICGShadowRunner:
    """Async in-process DICG production shadow.  Public API: submit, status, close."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        vit_checkpoint_path: str | Path,
        log_path: str | Path,
        device: str = "cpu",
        geometry: Any | None = None,
        model: Any | None = None,
        top_k: int = DEFAULT_TOP_K,
        latency_budget_s: float = DEFAULT_LATENCY_BUDGET_S,
    ) -> None:
        self._engine = DICGShadowEngine(
            checkpoint_path=checkpoint_path,
            vit_checkpoint_path=vit_checkpoint_path,
            log_path=log_path,
            device=device,
            geometry=geometry,
            model=model,
            top_k=top_k,
            latency_budget_s=latency_budget_s,
        )
        self._device = self._engine._device
        self._top_k = self._engine._top_k
        self._latency_budget_s = self._engine._latency_budget_s
        self._amp_dtype = self._engine._amp_dtype
        self._metadata = self._engine._metadata
        self._checkpoint_path = self._engine._checkpoint_path
        self._vit_path = self._engine._vit_path
        self._checkpoint_sha256 = self._engine._checkpoint_sha256
        self._vit_sha256 = self._engine._vit_sha256
        self._geometry = self._engine._geometry
        self._model = self._engine._model
        self._log_path = self._engine._log_path
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._submitted = 0
        self._dropped = 0
        self._submit_errors = 0
        self._thread = threading.Thread(
            target=self._worker,
            name="dicg-shadow-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, observation: Any) -> bool:
        """Enqueue a copied observation.  Never returns a score; never raises."""
        try:
            with self._lock:
                if self._closed:
                    return False
            snapshot = _copy_observation(observation)
            try:
                self._queue.put_nowait(snapshot)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    with self._lock:
                        self._dropped += 1
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(snapshot)
                except queue.Full:
                    with self._lock:
                        self._dropped += 1
                    return False
            with self._lock:
                self._submitted += 1
            return True
        except Exception:
            with self._lock:
                self._submit_errors += 1
            return False

    def status(self) -> dict[str, Any]:
        snapshot = self._engine.snapshot()
        with self._lock:
            payload = {
                "closed": bool(self._closed),
                "device": str(self._device),
                "queue_maxsize": QUEUE_MAXSIZE,
                "submitted": int(self._submitted),
                "dropped": int(self._dropped),
                "pairs_ok": int(snapshot["pairs_ok"]),
                "pairs_failed": int(snapshot["pairs_failed"]),
                "pairs_skipped": int(snapshot["pairs_skipped"]),
                "last_skip_reason": snapshot.get("last_skip_reason"),
                "skip_reasons": dict(snapshot.get("skip_reasons") or {}),
                "log_errors": int(snapshot["log_errors"]),
                "submit_errors": int(self._submit_errors),
                "over_budget": int(snapshot["over_budget"]),
                "worker_alive": bool(self._thread is not None and self._thread.is_alive()),
                "checkpoint_path": self._checkpoint_path,
                "vit_path": self._vit_path,
                "checkpoint_sha256": self._checkpoint_sha256,
                "vit_sha256": self._vit_sha256,
                "checkpoint_format": CHECKPOINT_FORMAT,
                "training_target": TRAINING_TARGET,
            }
        return _stamp_honesty(payload)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_CLOSE_JOIN_S)

    def _worker(self) -> None:
        previous: _CopiedObservation | None = None
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=_WORKER_POLL_S)
            except queue.Empty:
                continue
            if not isinstance(item, _CopiedObservation):
                continue
            if previous is None:
                previous = item
                continue
            try:
                self._safe_process_pair(previous, item)
            except Exception:
                # Isolation: a worker-frame failure must not kill the thread
                # or surface into submit / planner dataflow.
                pass
            previous = item

    def _safe_process_pair(
        self, obs_t: _CopiedObservation, obs_tp1: _CopiedObservation
    ) -> None:
        self._engine.safe_process_pair(obs_t, obs_tp1)

    def _infer_pair(
        self, obs_t: _CopiedObservation, obs_tp1: _CopiedObservation
    ) -> dict[str, Any]:
        return self._engine.infer_pair(obs_t, obs_tp1)

    def _encode_pair(
        self, image_t: torch.Tensor, image_tp1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._engine._encode_pair(image_t, image_tp1)

    def _append_jsonl(self, record: Mapping[str, Any]) -> None:
        self._engine.append_jsonl(_stamp_honesty(record))
