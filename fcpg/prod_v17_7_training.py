"""Fixed data, cache, optimization, and evaluation contracts for Prod-V17.7.

The 20K/320K visual archives provide stopped observer states, not human pixel
labels.  Exact spatial supervision comes from CAD and the existing labelled
real fitting panel.  The three optimizers are deliberately separated:
measurement first, belief dynamics second, and the Actor last.

Differences from the V17.6.2 contract this file was derived from:

* the 48K archive DEV rows (``split == 1``) are excluded from every optimizer
  and used only as a previously-exposed archive development probe;
* the GDM receives a per-row camera-rig identity (archive dataset id) with
  source dropout, and, on the calibrated fish rig, the analytic FK-projected
  anchor displacement as its transport mean;
* the GDM has a third, fixed-length fish phase so the analytic-residual path
  is trained before the Actor consumes it;
* the measurement optimizer adds the bounded hypothesis-separation penalty;
* the Actor is trained in three matched modes (``full``, ``memoryless``,
  ``raw_history``) that share one schedule;
* evaluation reports credible coverage, hypothesis overlap, source-stratified
  archive dynamics on DEV rows, the analytic-only transport control on fish,
  and every available Actor mode on the same contexts.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fcpg.prod_v17_5 import restore_companion_joint_units
from fcpg.prod_v17_5_training import (
    V175AllRealRGB,
    V175CADArchive,
    _real_label_arrays,
    balanced_real_anchor_batch,
    extract_dino_tensor,
    load_frozen_dinov2,
)
from fcpg.prod_v17_7 import (
    ACTOR_MODES,
    HISTORY_STEPS,
    SOURCE_UNKNOWN,
    V177Actor,
    V177AssignedMeasurement,
    V177Core,
    V177FilterUpdate,
    V177GroundedState,
    V177Innovation,
    V177Measurement,
    V177RawHistory,
    act_with,
    action_chunk_l1,
    analytic_shift_from_pixels,
    append_raw_history,
    compute_innovation,
    credible_coverage,
    empty_raw_history,
    exchangeable_pair_nll,
    fuse_measurement_prior,
    hypothesis_overlap,
    hypothesis_separation_penalty,
    identity_or_track_assignment,
    normalize_actor_scene_tokens,
    pool_scene_tokens,
    propagate_belief_v177,
    state_path_smooth_l1,
    uniform_spatial_prior,
)
from fcpg.prod_v18_1_data import (
    TRANSITIONS,
    V181FeatureCache,
    V181FishArchive,
    V181TemporalArchive,
)
from fcpg.prod_v18_1_training import V181DeviceFeatureBank
from fcpg.robot_kinematic_graph import DifferentiableRobotKinematics, so101_robot_spec
from fcpg.so101_camera_calibration import ProjectionFit, predict_projection


VERSION = "17.7.0"
SEED = 17_700
VIEWS = 2
HYPOTHESES = 2
GRID_SIZE = 56
SPATIAL_CELLS = GRID_SIZE * GRID_SIZE
BELIEF_CELLS = SPATIAL_CELLS + 1
ABSENT_INDEX = SPATIAL_CELLS
FEATURE_DIM = 384
VIT_PATCHES = 256
STATE_DIM = 6
COMMAND_STEPS = 5
PATH_STEPS = 6
VIEW_ORDER = ("wrist", "column")
CHECKPOINT_EVERY = 500
GRADIENT_CLIP = 1.0
FCPG_APPEARANCE_LR = 1.0e-4
FCPG_CAD_LR = 1.0e-4
FCPG_REAL_LR = 3.0e-5
LEWM_LR = 1.0e-4
ACTOR_ARCHIVE_LR = 1.0e-4
ACTOR_DENSE_LR = 5.0e-5
ACTOR_FISH_LR = 3.0e-5
CAD_PASSES = 4
REAL_REFIT_UPDATES = 600
ACTOR_ARCHIVE_PASSES = 4
ACTOR_DENSE_PASSES = 4
ACTOR_FISH_UPDATES = 4_000
LEWM_FISH_UPDATES = 500
LEWM_FISH_LR = 3.0e-5
# Fraction of fish GDM updates that hide the analytic mean so the learned-only
# path stays usable when no calibration is available.
LEWM_FISH_ANALYTIC_DROPOUT = 0.5
ARCHITECTURE_CONTRACT = "camera-aware-belief-fcpg-v1"
READBACK_ERROR_MEAN_DEG = (
    -0.094596,
    1.031626,
    1.422024,
    0.267222,
    -0.058734,
    2.245575,
)
READBACK_ERROR_STD_DEG = (
    0.838503,
    1.786527,
    1.967177,
    1.386477,
    0.714411,
    4.626208,
)


class V177TrainingError(RuntimeError):
    """Raised before a malformed V17.7 batch changes model state."""


def seed_v177(seed: int = SEED) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_cuda(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def require_physical_gpu1(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if str(device) != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise V177TrainingError(
            "launch with CUDA_VISIBLE_DEVICES=1 and use process-local cuda:0"
        )


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _append_exposed_development_access(
    evaluation: Path,
    *,
    event: str,
    smoke: bool,
) -> None:
    """Durably append one direct-API fish episode-2 access event."""

    if event not in {"evaluation_started", "evaluation_completed"}:
        raise V177TrainingError("unknown exposed-development access event")
    evaluation.mkdir(parents=True, exist_ok=True)
    record = {
        "unix_time_ns": time.time_ns(),
        "event": event,
        "version": VERSION,
        "dataset": "fish_episode_2",
        "data_role": "exposed_development_previously_opened",
        "strict_held_evidence": False,
        "smoke": bool(smoke),
        "process_id": os.getpid(),
    }
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    path = evaluation / "exposed-development-access.jsonl"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise V177TrainingError(
                    "failed to append exposed-development access record"
                )
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _model_config(module: nn.Module) -> dict[str, Any]:
    value = getattr(module, "config", None)
    if value is None:
        return {}
    try:
        return asdict(value)
    except TypeError:
        return dict(value) if isinstance(value, Mapping) else {}


def save_component(
    path: str | Path,
    module: nn.Module,
    *,
    component: str,
    step: int,
    metrics: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    cursor: Mapping[str, Any] | None = None,
) -> Path:
    """Save an ordinary resumable checkpoint without repository binding."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": VERSION,
        "architecture": ARCHITECTURE_CONTRACT,
        "component": str(component),
        "seed": SEED,
        "step": int(step),
        "config": _model_config(module),
        "model": module.state_dict(),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if cursor is not None:
        payload["cursor"] = dict(cursor)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_component(
    path: str | Path,
    module: nn.Module,
    *,
    component: str,
    device: torch.device,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise V177TrainingError(f"V17.7 component is absent: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise V177TrainingError("V17.7 checkpoint must be a mapping")
    if payload.get("version") != VERSION or payload.get("architecture") != ARCHITECTURE_CONTRACT:
        raise V177TrainingError("V17.7 checkpoint contract differs")
    if payload.get("component") != str(component) or not isinstance(payload.get("model"), dict):
        raise V177TrainingError("V17.7 checkpoint component differs")
    module.load_state_dict(payload["model"], strict=True)
    module.to(device)
    return payload


def import_measurement_weights(
    core: V177Core,
    path: str | Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Copy only ``measurement.*`` weights from a completed V17.6.2 checkpoint.

    The measurement architecture is unchanged between V17.6.2 and V17.7, so an
    exact key match is required.  Nothing else (GDM, innovation, Actor) is
    imported; those contracts changed.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise V177TrainingError(f"measurement init checkpoint is absent: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise V177TrainingError("measurement init checkpoint must be a mapping")
    if payload.get("component") != "measurement":
        raise V177TrainingError("measurement init must come from a measurement checkpoint")
    prefix = "measurement."
    imported = {
        key[len(prefix):]: value
        for key, value in payload["model"].items()
        if key.startswith(prefix)
    }
    expected = set(core.measurement.state_dict().keys())
    if set(imported.keys()) != expected:
        raise V177TrainingError("measurement init keys differ from the V17.7 measurement")
    core.measurement.load_state_dict(imported, strict=True)
    core.to(device)
    metrics = dict(payload.get("metrics", {}))
    metrics["imported_from"] = str(source)
    metrics["imported_version"] = str(payload.get("version"))
    metrics["imported_architecture"] = str(payload.get("architecture"))
    metrics["separation_penalty_applied"] = False
    return metrics


def _progress(stage: str, step: int, total: int, loss: float, started: float) -> None:
    if step != 1 and step != total and step % 50:
        return
    elapsed = max(time.monotonic() - started, 1.0e-6)
    rate = step / elapsed
    payload = {
        "stage": stage,
        "step": int(step),
        "total": int(total),
        "loss": float(loss),
        "elapsed_s": round(elapsed, 1),
        "eta_s": round((total - step) / max(rate, 1.0e-12), 1),
        "gpu_allocated_mib": (
            int(torch.cuda.memory_allocated() / 1024**2)
            if torch.cuda.is_available()
            else 0
        ),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def _batches(values: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(values), int(batch_size)):
        yield values[start : start + int(batch_size)]


def _localized_archive_rows(
    archive: V181TemporalArchive,
    rows: np.ndarray,
    *,
    horizon: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Shuffle endpoint-local blocks so the fixed GPU page cache stays hot."""

    selected = np.asarray(rows, dtype=np.int64)
    endpoint = np.asarray(
        archive.row_endpoint_index[selected, int(horizon)], dtype=np.int64
    )
    position = np.argsort(endpoint, kind="stable")
    block_rows = max(int(batch_size), 512)
    blocks = [
        position[start : start + block_rows].copy()
        for start in range(0, len(position), block_rows)
    ]
    generator = np.random.default_rng(int(seed))
    for block in blocks:
        generator.shuffle(block)
    generator.shuffle(blocks)
    return selected[np.concatenate(blocks)]


def cosine_schedule(step: int, *, total: int, warmup: int) -> float:
    if step < warmup:
        return float(step + 1) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


class V177BeliefCache:
    """Stopped FCPG measurements addressed by ordinary endpoint identity."""

    def __init__(self, root: str | Path, *, expected_kind: str | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise V177TrainingError(f"belief cache is absent: {self.root}")
        try:
            config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise V177TrainingError("belief cache config is malformed") from error
        if config.get("version") != VERSION or config.get("contract") != ARCHITECTURE_CONTRACT:
            raise V177TrainingError("belief cache contract differs")
        if tuple(config.get("view_order", ())) != VIEW_ORDER or not bool(config.get("complete")):
            raise V177TrainingError("belief cache is incomplete or has wrong view order")
        self.kind = str(config.get("kind", ""))
        if expected_kind is not None and self.kind != expected_kind:
            raise V177TrainingError(
                f"belief cache kind is {self.kind}, expected {expected_kind}"
            )
        self.endpoint_index = np.load(self.root / "endpoint_index.npy", mmap_mode="r", allow_pickle=False)
        self.anchor_hypothesis_log_likelihood = np.load(self.root / "anchor_hypothesis_log_likelihood.npy", mmap_mode="r", allow_pickle=False)
        self.view_valid = np.load(self.root / "view_valid.npy", mmap_mode="r", allow_pickle=False)
        rows = len(self.endpoint_index)
        expected = {
            "endpoint_index": ((rows,), np.dtype(np.int64)),
            "anchor_hypothesis_log_likelihood": ((rows, VIEWS, HYPOTHESES, BELIEF_CELLS), np.dtype(np.float16)),
            "view_valid": ((rows, VIEWS), np.dtype(np.bool_)),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise V177TrainingError(f"belief cache {name} shape/dtype drifted")
        if rows == 0 or bool((np.diff(self.endpoint_index) <= 0).any()):
            raise V177TrainingError("belief endpoint identities must be sorted unique")
        sample = np.unique(np.linspace(0, rows - 1, min(rows, 128), dtype=np.int64))
        logp = np.asarray(self.anchor_hypothesis_log_likelihood[sample], dtype=np.float32)
        if not bool(np.isfinite(logp).all()):
            raise V177TrainingError("belief cache contains non-finite values")
        normalizer = np.logaddexp.reduce(logp, axis=-1)
        if float(np.max(np.abs(normalizer))) > 3.0e-2:
            raise V177TrainingError("cached beliefs are not normalized")

    def rows_for(self, endpoint_indices: Any) -> np.ndarray:
        requested = np.asarray(endpoint_indices, dtype=np.int64)
        flat = requested.reshape(-1)
        positions = np.searchsorted(self.endpoint_index, flat)
        safe = np.minimum(positions, len(self.endpoint_index) - 1)
        if bool((positions >= len(self.endpoint_index)).any()) or not bool(
            np.array_equal(self.endpoint_index[safe], flat)
        ):
            raise V177TrainingError("endpoint is absent from belief cache")
        return positions.reshape(requested.shape)

    def get_numpy(
        self, endpoint_indices: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(endpoint_indices, dtype=np.int64)
        rows = self.rows_for(requested)
        shape = requested.shape
        return (
            np.asarray(self.anchor_hypothesis_log_likelihood[rows], dtype=np.float32).reshape(
                *shape, VIEWS, HYPOTHESES, BELIEF_CELLS
            ),
            np.asarray(self.view_valid[rows], dtype=np.bool_).reshape(*shape, VIEWS),
        )


class V177DeviceBeliefBank:
    """Load a compact belief cache onto one accelerator exactly once."""

    def __init__(self, cache: V177BeliefCache, *, device: torch.device) -> None:
        if device.type != "cuda":
            raise V177TrainingError("full belief bank requires CUDA")
        self.cache = cache
        self.device = device
        self.endpoint = torch.from_numpy(
            np.array(cache.endpoint_index, dtype=np.int64, copy=True)
        ).to(device)
        self.anchor_hypothesis_log_likelihood = torch.from_numpy(
            np.array(
                cache.anchor_hypothesis_log_likelihood,
                dtype=np.float16,
                copy=True,
            )
        ).to(device)
        self.view_valid = torch.from_numpy(
            np.array(cache.view_valid, dtype=np.bool_, copy=True)
        ).to(device)

    def get(
        self, endpoint_indices: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        requested = np.asarray(endpoint_indices, dtype=np.int64)
        rows = self.cache.rows_for(requested).reshape(-1)
        index = torch.from_numpy(rows).to(self.device)
        shape = requested.shape
        return (
            self.anchor_hypothesis_log_likelihood[index].reshape(*shape, VIEWS, HYPOTHESES, BELIEF_CELLS).float(),
            self.view_valid[index].reshape(*shape, VIEWS),
        )

    def stats(self) -> dict[str, int]:
        tensors = (
            self.endpoint,
            self.anchor_hypothesis_log_likelihood,
            self.view_valid,
        )
        return {"allocated_bytes": int(sum(x.numel() * x.element_size() for x in tensors))}


def assignment_matrix_from_direct(probability: torch.Tensor) -> torch.Tensor:
    direct = torch.as_tensor(probability).float().clamp(0.0, 1.0)
    return torch.stack(
        (
            torch.stack((direct, 1.0 - direct), dim=-1),
            torch.stack((1.0 - direct, direct), dim=-1),
        ),
        dim=-2,
    )


def _merge_metrics(path: Path, stage: str, values: Mapping[str, Any]) -> None:
    payload: dict[str, Any] = {"version": VERSION, "seed": SEED, "stages": {}}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    stages = payload.setdefault("stages", {})
    if not isinstance(stages, dict):
        stages = {}
        payload["stages"] = stages
    stages[str(stage)] = dict(values)
    _write_json(path, payload)


def _tensor(value: Any, device: torch.device, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(value), device=device, dtype=dtype)


def _set_trainable(module: nn.Module, trainable: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    selected = {id(value) for value in trainable}
    result: list[nn.Parameter] = []
    for value in module.parameters():
        active = id(value) in selected
        value.requires_grad_(active)
        if active:
            result.append(value)
    if not result:
        raise V177TrainingError("optimizer has no trainable parameters")
    return result


def _measurement_spatial_parameters(
    core: V177Core,
    *,
    update_rgb: bool,
) -> list[nn.Parameter]:
    """Select spatial paths without letting CAD overwrite real RGB adapters."""

    frozen = {
        id(value) for value in core.measurement.dino_projection.parameters()
    }
    if not update_rgb:
        frozen.update(
            id(value) for value in core.measurement.rgb_stem.parameters()
        )
        frozen.update(
            id(value)
            for value in core.measurement.rgb_dino_fusion.parameters()
        )
    return [
        value
        for value in core.measurement.parameters()
        if id(value) not in frozen
    ]


def _exchangeable_pair_sum_count(
    predicted_log: torch.Tensor,
    target_probability: torch.Tensor,
    view_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the pair-CE numerator and valid-view denominator."""

    valid_count = view_valid.bool().sum(dtype=torch.float32)
    mean = exchangeable_pair_nll(
        predicted_log,
        target_probability,
        view_valid,
    )
    return mean * valid_count, valid_count


def _restore_latest(
    path: Path,
    core: V177Core,
    *,
    component: str,
    device: torch.device,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return load_component(path, core, component=component, device=device)


def _restore_optimizer(
    optimizer: torch.optim.Optimizer,
    payload: Mapping[str, Any] | None,
    *,
    phase: str,
) -> int:
    if payload is None:
        return 0
    cursor = payload.get("cursor")
    if not isinstance(cursor, Mapping) or cursor.get("phase") != phase:
        return 0
    state = payload.get("optimizer")
    if isinstance(state, Mapping):
        optimizer.load_state_dict(state)
    return int(cursor.get("phase_step", 0))


def _bilinear_point_belief(
    points_px: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Encode point labels as four-neighbour mass plus an absent class."""

    points = torch.as_tensor(points_px).float()
    active = torch.as_tensor(valid, device=points.device, dtype=torch.bool)
    if points.ndim != 4 or tuple(points.shape[1:]) != (VIEWS, HYPOTHESES, 2):
        raise V177TrainingError("point labels must be [B,2,2,2]")
    if tuple(active.shape) != tuple(points.shape[:-1]):
        raise V177TrainingError("point validity must be [B,2,2]")
    finite = torch.isfinite(points).all(dim=-1)
    inside = (
        (points[..., 0] >= 0.0)
        & (points[..., 0] < 224.0)
        & (points[..., 1] >= 0.0)
        & (points[..., 1] < 224.0)
    )
    active = active & finite & inside
    safe = torch.where(active[..., None], points, torch.zeros_like(points))
    grid = ((safe + 0.5) / 4.0 - 0.5).clamp(0.0, GRID_SIZE - 1.0)
    x0 = grid[..., 0].floor().long()
    y0 = grid[..., 1].floor().long()
    x1 = (x0 + 1).clamp_max(GRID_SIZE - 1)
    y1 = (y0 + 1).clamp_max(GRID_SIZE - 1)
    wx = grid[..., 0] - x0.float()
    wy = grid[..., 1] - y0.float()
    result = points.new_zeros(*points.shape[:-1], BELIEF_CELLS)
    for x, y, weight in (
        (x0, y0, (1.0 - wx) * (1.0 - wy)),
        (x1, y0, wx * (1.0 - wy)),
        (x0, y1, (1.0 - wx) * wy),
        (x1, y1, wx * wy),
    ):
        result.scatter_add_(-1, (y * GRID_SIZE + x)[..., None], weight[..., None] * active[..., None])
    result[..., ABSENT_INDEX] = (~active).to(result)
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _measurement_feature_distillation(
    core: V177Core,
    rgb: torch.Tensor,
    dino_tokens: torch.Tensor,
    view_valid: torch.Tensor,
) -> torch.Tensor:
    """Train only the high-resolution RGB fusion against stopped DINO features."""

    measurement = core.measurement
    batch = len(rgb)
    image = rgb.float().div(255.0)
    mean = image.mean(dim=(-1, -2), keepdim=True)
    std = image.std(dim=(-1, -2), keepdim=True).clamp_min(1.0e-4)
    rgb_map = measurement.rgb_stem(
        ((image - mean) / std).reshape(batch * VIEWS, 3, 224, 224)
    )
    dino = F.layer_norm(dino_tokens.float(), (FEATURE_DIM,)).reshape(
        batch * VIEWS, 16, 16, FEATURE_DIM
    ).permute(0, 3, 1, 2)
    target = F.interpolate(dino, (GRID_SIZE, GRID_SIZE), mode="bilinear", align_corners=False)
    target = measurement.dino_projection(target).detach()
    fused = target + measurement.rgb_dino_fusion(torch.cat((target, rgb_map), dim=1))
    rgb_loss = 1.0 - F.cosine_similarity(rgb_map, target, dim=1)
    fused_loss = 1.0 - F.cosine_similarity(fused, target, dim=1)
    mask = view_valid.reshape(batch * VIEWS, 1, 1).expand_as(rgb_loss).to(rgb_loss)
    return ((0.5 * rgb_loss + 0.5 * fused_loss) * mask).sum() / mask.sum().clamp_min(1)


class V177GeometryIdentityProvider:
    """Broad SO-101 image prior used only to score hypothesis permutations."""

    def __init__(self, calibration_path: str | Path, *, device: torch.device) -> None:
        path = Path(calibration_path).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise V177TrainingError(f"camera calibration is malformed: {path}") from error
        fits: dict[str, ProjectionFit] = {}
        for name in VIEW_ORDER:
            row = payload.get("fits", {}).get(name)
            if not isinstance(row, dict):
                raise V177TrainingError(f"camera calibration omits {name}")
            fits[name] = ProjectionFit(
                view=str(row["view"]),
                feature_names=tuple(map(str, row["feature_names"])),
                coefficients=np.asarray(row["coefficients"], dtype=np.float64),
                ridge=float(row["ridge"]),
            )
        self.fits = fits
        self.kinematics = DifferentiableRobotKinematics(so101_robot_spec()).to(device)
        self.device = device

    @torch.inference_mode()
    def pixels(self, current_q_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = torch.as_tensor(current_q_deg, device=self.device, dtype=torch.float32)
        if q.ndim != 2 or q.shape[1] != STATE_DIM or not bool(torch.isfinite(q).all()):
            raise V177TrainingError("identity q must be finite [B,6]")
        base, _, _ = self.kinematics(q)
        reference_points = self.kinematics.anchors_in_view_references(q)
        # ``pixels`` can be called under the surrounding bf16 Actor autocast;
        # NumPy has no bfloat16 dtype, and calibration is defined in float32.
        base_np = base[:, 4:6].float().cpu().numpy()
        wrist_np = reference_points[:, 0, 4:6].float().cpu().numpy()
        opening = q[:, -1].float().cpu().numpy()
        projected = np.stack(
            (
                predict_projection(self.fits["wrist"], base_np, wrist_np, opening),
                predict_projection(self.fits["column"], base_np, wrist_np, opening),
            ),
            axis=1,
        ).astype(np.float32)
        value = torch.from_numpy(projected).to(self.device)
        valid = torch.isfinite(value).all(-1) & (value[..., 0] >= 0) & (value[..., 0] < 224) & (value[..., 1] >= 0) & (value[..., 1] < 224)
        return value, valid

    @torch.inference_mode()
    def broad_log_prior(
        self,
        current_q_deg: torch.Tensor,
        view_valid: torch.Tensor,
        *,
        sigma_px: float = 36.0,
    ) -> torch.Tensor:
        points, point_valid = self.pixels(current_q_deg)
        valid = torch.as_tensor(view_valid, device=self.device, dtype=torch.bool)
        centers = (points + 0.5) / 4.0 - 0.5
        axis = torch.arange(GRID_SIZE, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        xy = torch.stack((xx, yy), dim=-1).reshape(1, 1, 1, SPATIAL_CELLS, 2)
        distance = (xy - centers[..., None, :]).square().sum(-1)
        sigma_grid = float(sigma_px) / 4.0
        visible_logits = -0.5 * distance / (sigma_grid * sigma_grid)
        absent_logit = visible_logits.new_full((*visible_logits.shape[:-1], 1), math.log(0.10))
        logits = torch.cat((visible_logits, absent_logit), dim=-1)
        usable = point_valid & valid[:, :, None]
        uniform = uniform_spatial_prior(len(points), device=self.device)
        return torch.where(usable[..., None], F.log_softmax(logits, -1), uniform)

    def analytic_transport(
        self,
        current_q_deg: torch.Tensor,
        commands_deg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytic anchor displacement over one command chunk.

        The chunk's final projected command is the intended end state; the
        GDM learns the residual (servo lag, backlash, calibration error)
        around this mean.  Returns ``shift [B,2,2,2]`` and ``valid [B,2]``.
        """

        q = torch.as_tensor(current_q_deg, dtype=torch.float32)
        commands = torch.as_tensor(commands_deg, dtype=torch.float32)
        if commands.ndim != 3 or commands.shape[1:] != (COMMAND_STEPS, STATE_DIM):
            raise V177TrainingError("analytic transport commands must be [B,5,6]")
        start, start_valid = self.pixels(q)
        end, _ = self.pixels(commands[:, -1])
        return analytic_shift_from_pixels(start, start_valid, end)


class V177SourceVocabulary:
    """Ordinary camera-rig identities for the GDM, derived from manifest names.

    Index ``0`` is the unknown rig.  Archive datasets receive stable indices in
    sorted order of their dataset id; the fish rig receives the next index
    unless its dataset id already appears in the archive manifest.  This is a
    plain lookup table, not content matching or checkpoint binding.
    """

    def __init__(self, names: Sequence[str], *, fish_name: str = "") -> None:
        ordered = sorted({str(name).strip() for name in names if str(name).strip()})
        if not ordered:
            raise V177TrainingError("source vocabulary needs at least one dataset id")
        self.names: tuple[str, ...] = tuple(ordered)
        self.index: dict[str, int] = {
            name: position + 1 for position, name in enumerate(self.names)
        }
        fish = str(fish_name).strip()
        if fish and fish in self.index:
            self.fish_index = int(self.index[fish])
        else:
            self.fish_index = len(self.names) + 1
        self.fish_name = fish

    @property
    def size(self) -> int:
        return int(len(self.names) + 2)

    @staticmethod
    def episode_source_name(episode: Mapping[str, Any]) -> str:
        row = episode.get("manifest_row")
        if not isinstance(row, Mapping):
            raise V177TrainingError("temporal episode omits its manifest row")
        name = str(row.get("dataset_id", "")).strip()
        if not name:
            name = Path(str(row.get("dataset_root", ""))).name.strip()
        if not name:
            name = str(row.get("dataset_provenance", "")).strip()
        return name or "unnamed-source"

    @classmethod
    def from_archive(
        cls,
        archive: V181TemporalArchive,
        fish: V181FishArchive | None = None,
    ) -> "V177SourceVocabulary":
        names = [cls.episode_source_name(episode) for episode in archive.episodes]
        fish_name = ""
        if fish is not None:
            fish_name = str(getattr(fish, "dataset_id", "") or getattr(fish, "source_name", ""))
        return cls(names, fish_name=fish_name)

    def episode_index(self, archive: V181TemporalArchive) -> np.ndarray:
        return np.asarray(
            [self.index[self.episode_source_name(episode)] for episode in archive.episodes],
            dtype=np.int64,
        )

    def row_index(self, archive: V181TemporalArchive) -> np.ndarray:
        """Per-row rig index via the H0 endpoint's episode."""

        per_episode = self.episode_index(archive)
        endpoint = np.asarray(archive.row_endpoint_index[:, 0], dtype=np.int64)
        episode = np.asarray(archive.endpoint_episode_index[endpoint], dtype=np.int64)
        return per_episode[episode]

    def seen_in_rows(self, archive: V181TemporalArchive, rows: np.ndarray) -> np.ndarray:
        """Boolean per vocabulary index: does any of ``rows`` use that rig?"""

        seen = np.zeros(self.size, dtype=np.bool_)
        seen[np.unique(self.row_index(archive)[np.asarray(rows, dtype=np.int64)])] = True
        return seen

    def to_json(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "unknown_index": SOURCE_UNKNOWN,
            "names": list(self.names),
            "fish_name": self.fish_name,
            "fish_index": int(self.fish_index),
            "size": int(self.size),
        }


def _source_tensor(
    row_source: np.ndarray,
    rows: np.ndarray,
    *,
    device: torch.device,
    dropout: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Per-row rig indices with deterministic dropout to the unknown rig."""

    selected = np.asarray(row_source, dtype=np.int64)[np.asarray(rows, dtype=np.int64)]
    if float(dropout) > 0.0:
        if seed is None:
            raise V177TrainingError("source dropout requires a deterministic seed")
        generator = np.random.default_rng(int(seed))
        drop = generator.random(len(selected)) < float(dropout)
        selected = np.where(drop, SOURCE_UNKNOWN, selected)
    return torch.as_tensor(selected, device=device, dtype=torch.long)


def _check_source_vocabulary(core: V177Core, vocabulary: V177SourceVocabulary) -> None:
    if vocabulary.size > int(core.config.source_vocabulary):  # type: ignore[attr-defined]
        raise V177TrainingError(
            f"source vocabulary needs {vocabulary.size} slots but the model has "
            f"{int(core.config.source_vocabulary)}"  # type: ignore[attr-defined]
        )


def actor_stage_name(actor_mode: str) -> str:
    """Stage directory per Actor mode; ``full`` keeps the V17.6.2 name."""

    if actor_mode not in ACTOR_MODES:
        raise V177TrainingError(f"actor_mode must be one of {ACTOR_MODES}")
    return "actor" if actor_mode == "full" else f"actor-{actor_mode}"


def _optimizer_rows(archive: V181TemporalArchive, rows: np.ndarray) -> np.ndarray:
    """Fail closed if any optimizer row is outside the TRAIN partition."""

    selected = np.asarray(rows, dtype=np.int64)
    if bool((np.asarray(archive.split)[selected] != 0).any()):
        raise V177TrainingError("archive DEV rows may not enter a V17.7 optimizer")
    return selected


class V177CombinedFeatureSource:
    def __init__(self, first: Any, second: Any | None = None) -> None:
        self.first = first
        self.second = second
        self.first_endpoint = np.asarray(first.cache.endpoint_index if hasattr(first, "cache") else first.endpoint_index)

    def get(self, endpoints: Any) -> tuple[Any, Any]:
        requested = np.asarray(endpoints, dtype=np.int64)
        flat = requested.reshape(-1)
        position = np.searchsorted(self.first_endpoint, flat)
        safe = np.minimum(position, len(self.first_endpoint) - 1)
        in_first = (position < len(self.first_endpoint)) & (self.first_endpoint[safe] == flat)
        if bool(in_first.all()):
            return self.first.get(requested)
        if self.second is None or bool(in_first.any()):
            if self.second is None:
                raise V177TrainingError("feature endpoint is absent")
            a, av = self.first.get(flat[in_first])
            b, bv = self.second.get(flat[~in_first])
            torch_output = torch.is_tensor(a)
            if torch_output:
                output = a.new_empty(len(flat), VIEWS, VIT_PATCHES, FEATURE_DIM)
                valid = av.new_empty(len(flat), VIEWS)
                output[torch.as_tensor(np.flatnonzero(in_first), device=a.device)] = a
                output[torch.as_tensor(np.flatnonzero(~in_first), device=a.device)] = b
                valid[torch.as_tensor(np.flatnonzero(in_first), device=a.device)] = av
                valid[torch.as_tensor(np.flatnonzero(~in_first), device=a.device)] = bv
            else:
                output = np.empty((len(flat), VIEWS, VIT_PATCHES, FEATURE_DIM), np.float32)
                valid = np.empty((len(flat), VIEWS), np.bool_)
                output[in_first], valid[in_first] = a, av
                output[~in_first], valid[~in_first] = b, bv
            return output.reshape(*requested.shape, VIEWS, VIT_PATCHES, FEATURE_DIM), valid.reshape(*requested.shape, VIEWS)
        return self.second.get(requested)

    def cache_page_locations(
        self, endpoints: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve endpoints to backing-bank and cache-page identities."""

        requested = np.asarray(endpoints, dtype=np.int64)
        flat = requested.reshape(-1)
        position = np.searchsorted(self.first_endpoint, flat)
        safe = np.minimum(position, len(self.first_endpoint) - 1)
        in_first = (position < len(self.first_endpoint)) & (
            self.first_endpoint[safe] == flat
        )
        if self.second is None and not bool(in_first.all()):
            raise V177TrainingError("feature endpoint is absent")

        bank = np.where(in_first, 0, 1).astype(np.int8, copy=False)
        page = np.empty(len(flat), dtype=np.int64)
        for selected, source in (
            (in_first, self.first),
            (~in_first, self.second),
        ):
            if not bool(selected.any()):
                continue
            if source is None:
                raise V177TrainingError("feature endpoint is absent")
            cache = source.cache if hasattr(source, "cache") else source
            cache_rows = cache.rows_for(flat[selected])
            page_rows = int(getattr(source, "page_rows", 512))
            if page_rows <= 0:
                raise V177TrainingError("feature cache page size must be positive")
            page[selected] = cache_rows // page_rows
        return bank.reshape(requested.shape), page.reshape(requested.shape)


def _feature_cache_page_locations(
    source: Any, endpoints: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable ``(bank, page)`` identities without loading token data."""

    if hasattr(source, "cache_page_locations"):
        return source.cache_page_locations(endpoints)
    requested = np.asarray(endpoints, dtype=np.int64)
    cache = source.cache if hasattr(source, "cache") else source
    if not hasattr(cache, "rows_for"):
        raise V177TrainingError("feature source cannot expose cache locality")
    page_rows = int(getattr(source, "page_rows", 512))
    if page_rows <= 0:
        raise V177TrainingError("feature cache page size must be positive")
    cache_rows = cache.rows_for(requested.reshape(-1)).reshape(requested.shape)
    return np.zeros(requested.shape, dtype=np.int8), cache_rows // page_rows


def _localized_feature_rows(
    rows: np.ndarray,
    endpoints: np.ndarray,
    source: Any,
    *,
    seed: int,
) -> np.ndarray:
    """Shuffle cache pages and rows within a page without changing exposure."""

    selected = np.asarray(rows, dtype=np.int64)
    requested = np.asarray(endpoints, dtype=np.int64)
    if (
        selected.ndim != 1
        or requested.ndim != 1
        or len(selected) == 0
        or len(selected) != len(requested)
    ):
        raise V177TrainingError("localized feature schedule inputs are invalid")
    bank, page = _feature_cache_page_locations(source, requested)
    bank = np.asarray(bank, dtype=np.int64).reshape(-1)
    page = np.asarray(page, dtype=np.int64).reshape(-1)
    position = np.lexsort((page, bank))
    sorted_bank = bank[position]
    sorted_page = page[position]
    boundary = np.flatnonzero(
        (sorted_bank[1:] != sorted_bank[:-1])
        | (sorted_page[1:] != sorted_page[:-1])
    ) + 1
    starts = np.concatenate((np.asarray([0]), boundary))
    stops = np.concatenate((boundary, np.asarray([len(position)])))
    generator = np.random.default_rng(int(seed))
    blocks: list[np.ndarray] = []
    for start, stop in zip(starts, stops, strict=True):
        block = position[int(start) : int(stop)].copy()
        generator.shuffle(block)
        blocks.append(block)
    block_order = generator.permutation(len(blocks))
    schedule = np.concatenate([blocks[int(index)] for index in block_order])
    return selected[schedule]


class V177CombinedBeliefSource:
    def __init__(self, first: Any, second: Any | None = None) -> None:
        self.first = first
        self.second = second
        self.first_endpoint = np.asarray(first.cache.endpoint_index if hasattr(first, "cache") else first.endpoint_index)

    def get(self, endpoints: Any) -> tuple[Any, Any]:
        requested = np.asarray(endpoints, dtype=np.int64)
        flat = requested.reshape(-1)
        position = np.searchsorted(self.first_endpoint, flat)
        safe = np.minimum(position, len(self.first_endpoint) - 1)
        in_first = (position < len(self.first_endpoint)) & (self.first_endpoint[safe] == flat)
        if bool(in_first.all()):
            return self.first.get(requested) if hasattr(self.first, "get") else self.first.get_numpy(requested)
        if self.second is None:
            raise V177TrainingError("belief endpoint is absent")
        getter_a = self.first.get if hasattr(self.first, "get") else self.first.get_numpy
        getter_b = self.second.get if hasattr(self.second, "get") else self.second.get_numpy
        a = getter_a(flat[in_first])
        b = getter_b(flat[~in_first])
        use_torch = torch.is_tensor(a[0])
        shapes = (
            (VIEWS, HYPOTHESES, BELIEF_CELLS),
            (VIEWS,),
        )
        result: list[Any] = []
        for left, right, tail in zip(a, b, shapes):
            if use_torch:
                value = left.new_empty((len(flat), *tail))
                ia = torch.as_tensor(np.flatnonzero(in_first), device=left.device)
                ib = torch.as_tensor(np.flatnonzero(~in_first), device=left.device)
                value[ia], value[ib] = left, right
            else:
                value = np.empty((len(flat), *tail), dtype=left.dtype)
                value[in_first], value[~in_first] = left, right
            result.append(value.reshape(*requested.shape, *tail))
        return tuple(result)  # type: ignore[return-value]


class V177CanonicalRealRGB:
    """Bind the legacy 20K RGB payload to canonical wrist/column endpoints.

    The old RGB cache predates an explicit ``view_order`` field.  Its manifest
    does, however, name the V12 full-grid cache from which the two-view payload
    was built.  We compare that cache with the current canonical endpoint bank
    on ordinary sidecar/chain identities.  This rejects a silent crossed-view
    join without inventing a new manifest or trusting the stale sidecar mask.
    """

    view_order = VIEW_ORDER

    def __init__(
        self,
        rgb_root: str | Path,
        sidecar_root: str | Path,
        archive: V181TemporalArchive,
    ) -> None:
        legacy = V175AllRealRGB(rgb_root, sidecar_root)
        self.rgb_root = legacy.rgb_root
        self.sidecar_root = legacy.sidecar_root
        self.rgb = legacy.rgb
        self.sidecar_row = legacy.sidecar_row
        self.endpoint_rows = legacy.endpoint_rows

        order = np.argsort(np.asarray(archive.sidecar_row, dtype=np.int64))
        sorted_sidecar = np.asarray(archive.sidecar_row[order], dtype=np.int64)
        selected_sidecar = np.asarray(self.sidecar_row, dtype=np.int64)
        location = np.searchsorted(sorted_sidecar, selected_sidecar)
        safe = np.minimum(location, len(sorted_sidecar) - 1)
        if bool((location >= len(sorted_sidecar)).any()) or not bool(
            np.array_equal(sorted_sidecar[safe], selected_sidecar)
        ):
            raise V177TrainingError("20K RGB rows do not map to the canonical archive")
        self.archive_rows = order[location]
        self.canonical_endpoint = np.asarray(
            archive.row_endpoint_index[self.archive_rows], dtype=np.int64
        )
        self.canonical_view_valid = np.asarray(
            archive.endpoint_view_valid[self.canonical_endpoint], dtype=np.bool_
        )

        rgb_chain = np.load(
            self.rgb_root / "chain_id.npy", mmap_mode="r", allow_pickle=False
        )
        if rgb_chain.shape != (20_000,) or rgb_chain.dtype != np.int64:
            raise V177TrainingError("20K RGB chain identity drifted")
        if not bool(
            np.array_equal(
                np.asarray(rgb_chain, dtype=np.int64),
                np.asarray(archive.chain_id[self.archive_rows], dtype=np.int64),
            )
        ):
            raise V177TrainingError("20K RGB and canonical chain clocks disagree")

        self.view_order_evidence = self._validate_legacy_view_order(archive, rgb_chain)
        stale = np.asarray(
            legacy.view_valid[selected_sidecar], dtype=np.bool_
        )
        self.stale_view_valid_disagreements = int(
            np.count_nonzero(stale != self.canonical_view_valid)
        )

    def _validate_legacy_view_order(
        self,
        archive: V181TemporalArchive,
        rgb_chain: np.ndarray,
    ) -> str:
        try:
            manifest = json.loads(
                (self.rgb_root / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise V177TrainingError("20K RGB manifest is malformed") from error
        declared = manifest.get("view_order")
        if declared is not None:
            if tuple(map(str, declared)) != VIEW_ORDER:
                raise V177TrainingError("20K RGB declared view order differs")
            return "declared_manifest"
        if manifest.get("format") != "so101-prod-v16-3-rgb-cache-v1":
            raise V177TrainingError("20K RGB legacy format declaration differs")
        source_value = manifest.get("full_grid_cache")
        if not isinstance(source_value, str) or not source_value.strip():
            raise V177TrainingError("20K RGB omits its legacy view-order evidence")
        source_root = Path(source_value).expanduser().resolve()
        try:
            source_tokens = np.load(
                source_root / "tokens.npy", mmap_mode="r", allow_pickle=False
            )
            source_sidecar = np.load(
                source_root / "sidecar_row.npy", mmap_mode="r", allow_pickle=False
            )
            source_chain = np.load(
                source_root / "chain_id.npy", mmap_mode="r", allow_pickle=False
            )
            canonical_tokens = np.load(
                archive.endpoint_root / "tokens.npy", mmap_mode="r", allow_pickle=False
            )
        except (OSError, ValueError) as error:
            raise V177TrainingError("legacy view-order evidence is unreadable") from error
        if (
            source_tokens.dtype != np.float16
            or source_tokens.ndim != 5
            or tuple(source_tokens.shape[1:3]) != (5, VIEWS)
            or source_sidecar.dtype != np.int32
            or source_sidecar.shape != (len(source_tokens),)
            or source_chain.dtype != np.int64
            or source_chain.shape != (len(source_tokens),)
            or canonical_tokens.dtype != np.float16
            or canonical_tokens.ndim != 4
            or tuple(canonical_tokens.shape[1:3]) != (VIEWS, VIT_PATCHES)
            or len(canonical_tokens) != archive.endpoint_count
        ):
            raise V177TrainingError("legacy/canonical view-order evidence shape drifted")
        source_order = np.argsort(np.asarray(source_sidecar, dtype=np.int64))
        source_sorted = np.asarray(source_sidecar[source_order], dtype=np.int64)
        location = np.searchsorted(source_sorted, np.asarray(self.sidecar_row, dtype=np.int64))
        safe = np.minimum(location, len(source_sorted) - 1)
        if bool((location >= len(source_sorted)).any()) or not bool(
            np.array_equal(source_sorted[safe], np.asarray(self.sidecar_row, dtype=np.int64))
        ):
            raise V177TrainingError("20K RGB rows are absent from legacy view evidence")
        source_rows = source_order[location]
        if not bool(
            np.array_equal(
                np.asarray(source_chain[source_rows], dtype=np.int64),
                np.asarray(rgb_chain, dtype=np.int64),
            )
        ):
            raise V177TrainingError("20K RGB and legacy view-evidence clocks disagree")

        sample = np.unique(
            np.linspace(0, len(self.sidecar_row) - 1, min(32, len(self.sidecar_row)), dtype=np.int64)
        )
        canonical = np.asarray(
            canonical_tokens[self.canonical_endpoint[sample]], dtype=np.float32
        )
        legacy = np.asarray(source_tokens[source_rows[sample]], dtype=np.float32)
        valid = self.canonical_view_valid[sample]
        for view in range(VIEWS):
            active = valid[:, :, view]
            if not bool(active.any()):
                raise V177TrainingError("20K alignment panel has no valid view evidence")

            def cosine(right: np.ndarray) -> np.ndarray:
                numerator = np.sum(canonical[:, :, view] * right, axis=-1)
                denominator = np.linalg.norm(canonical[:, :, view], axis=-1) * np.linalg.norm(
                    right, axis=-1
                )
                return numerator / np.maximum(denominator, 1.0e-8)

            direct = cosine(legacy[:, :, view])[active]
            crossed = cosine(legacy[:, :, 1 - view])[active]
            if (
                float(direct.min()) < 0.999
                or float(direct.mean() - crossed.mean()) < 0.10
            ):
                raise V177TrainingError(
                    "20K RGB view identity cannot be aligned to canonical wrist/column"
                )
        return "legacy_v12_token_alignment"

    def batch(self, endpoint_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = np.asarray(endpoint_rows, dtype=np.int64)
        if selected.ndim != 1 or len(selected) == 0 or bool(
            ((selected < 0) | (selected >= 100_000)).any()
        ):
            raise V177TrainingError("20K endpoint batch leaves its population")
        row = selected // 5
        endpoint = selected % 5
        rgb = np.ascontiguousarray(self.rgb[row, endpoint])
        valid = np.ascontiguousarray(self.canonical_view_valid[row, endpoint])
        return rgb, valid


def dense_endpoint_map(
    real_rgb: V175AllRealRGB | V177CanonicalRealRGB,
    archive: V181TemporalArchive,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted canonical endpoints and their 20K RGB endpoint rows."""

    order = np.argsort(np.asarray(archive.sidecar_row, dtype=np.int64))
    sorted_sidecar = np.asarray(archive.sidecar_row[order], dtype=np.int64)
    location = np.searchsorted(sorted_sidecar, np.asarray(real_rgb.sidecar_row, dtype=np.int64))
    if bool((location >= len(sorted_sidecar)).any()) or not bool(
        np.array_equal(sorted_sidecar[location], np.asarray(real_rgb.sidecar_row, dtype=np.int64))
    ):
        raise V177TrainingError("20K RGB rows do not map to the 320K archive")
    archive_rows = order[location]
    endpoints = np.asarray(archive.row_endpoint_index[archive_rows], dtype=np.int64).reshape(-1)
    local = np.arange(100_000, dtype=np.int64)
    unique, first = np.unique(endpoints, return_index=True)
    return unique, local[first]


def _create_belief_arrays(
    output: Path,
    endpoint_index: np.ndarray,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    output.mkdir(parents=True, exist_ok=True)
    endpoint = np.asarray(endpoint_index, dtype=np.int64)
    if endpoint.ndim != 1 or len(endpoint) == 0 or bool((np.diff(endpoint) <= 0).any()):
        raise V177TrainingError("belief cache endpoints must be sorted unique")
    identity = output / "endpoint_index.npy"
    if identity.is_file():
        if not bool(np.array_equal(np.load(identity, allow_pickle=False), endpoint)):
            raise V177TrainingError("resumed belief cache endpoints differ")
    else:
        np.save(identity, endpoint, allow_pickle=False)
    specs = (
        ("anchor_hypothesis_log_likelihood", np.float16, (len(endpoint), VIEWS, HYPOTHESES, BELIEF_CELLS)),
        ("view_valid", np.bool_, (len(endpoint), VIEWS)),
        ("done", np.bool_, (len(endpoint),)),
    )
    arrays: list[np.memmap] = []
    for name, dtype, shape in specs:
        path = output / f"{name}.npy"
        if path.is_file():
            value = np.lib.format.open_memmap(path, mode="r+")
            if value.shape != shape or value.dtype != np.dtype(dtype):
                raise V177TrainingError(f"resumed belief cache {name} differs")
        else:
            value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            if name == "done":
                value[:] = False
                value.flush()
        arrays.append(value)
    return tuple(arrays)  # type: ignore[return-value]


@torch.inference_mode()
def prepare_belief_cache(
    core: V177Core,
    feature_source: Any,
    endpoint_index: np.ndarray,
    output_root: str | Path,
    *,
    device: torch.device,
    kind: str,
    batch_size: int = 64,
    rgb_source: tuple[V175AllRealRGB, np.ndarray] | V181FishArchive | None = None,
) -> Path:
    """Cache stopped visual measurements; predictions are never persisted."""

    output = Path(output_root).expanduser().resolve()
    endpoint = np.asarray(endpoint_index, dtype=np.int64)
    config_path = output / "config.json"
    expected_config = {
        "version": VERSION,
        "contract": ARCHITECTURE_CONTRACT,
        "kind": str(kind),
        "view_order": list(VIEW_ORDER),
        "rows": int(len(endpoint)),
        "contents": "stopped current-image measurement only",
        "identity_mode": "exchangeable_raw_measurement",
    }
    if config_path.is_file():
        try:
            existing_config = json.loads(
                config_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise V177TrainingError(
                "belief cache config is malformed"
            ) from error
        for key, expected in expected_config.items():
            if existing_config.get(key) != expected:
                raise V177TrainingError(
                    f"resumed belief cache {key} differs"
                )
    else:
        orphaned = any(
            (output / name).is_file()
            for name in (
                "endpoint_index.npy",
                "anchor_hypothesis_log_likelihood.npy",
                "view_valid.npy",
                "done.npy",
            )
        )
        if orphaned:
            raise V177TrainingError(
                "belief cache arrays exist without a contract config"
            )
        _write_json(config_path, {**expected_config, "complete": False})
    anchor_hypothesis_log, valid_out, done = _create_belief_arrays(output, endpoint)
    pending = np.flatnonzero(~done)
    total = math.ceil(len(pending) / int(batch_size))
    started = time.monotonic()
    core.eval()
    awaiting_durable: list[np.ndarray] = []
    for step, cache_rows in enumerate(_batches(pending, batch_size), start=1):
        selected_endpoint = endpoint[cache_rows]
        dino, cache_valid = feature_source.get(selected_endpoint)
        dino_tensor = torch.as_tensor(dino, device=device, dtype=torch.float32)
        valid = torch.as_tensor(cache_valid, device=device, dtype=torch.bool)
        rgb_tensor: torch.Tensor | None = None
        if isinstance(rgb_source, tuple):
            archive_rgb, local_by_cache_row = rgb_source
            rgb_np, rgb_valid = archive_rgb.batch(local_by_cache_row[cache_rows])
            valid = valid & torch.as_tensor(rgb_valid, device=device, dtype=torch.bool)
            rgb_tensor = torch.as_tensor(rgb_np, device=device, dtype=torch.uint8)
        elif isinstance(rgb_source, V181FishArchive):
            rgb_np = rgb_source.rgb(selected_endpoint)
            rgb_tensor = torch.as_tensor(rgb_np, device=device, dtype=torch.uint8).permute(0, 1, 4, 2, 3)
        with _autocast(device):
            measurement = core.measure(dino_tensor, valid, rgb_tensor)
        anchor_hypothesis_log[cache_rows] = (
            measurement.anchor_hypothesis_log_likelihood.float()
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        valid_out[cache_rows] = valid.cpu().numpy()
        awaiting_durable.append(np.asarray(cache_rows, dtype=np.int64))
        if step % 32 == 0 or step == total:
            anchor_hypothesis_log.flush()
            valid_out.flush()
            durable_rows = np.concatenate(awaiting_durable)
            done[durable_rows] = True
            done.flush()
            awaiting_durable.clear()
        _progress(f"belief-cache-{kind}", step, total, 0.0, started)
    complete = bool(done.all())
    _write_json(
        config_path,
        {**expected_config, "complete": complete},
    )
    if not complete:
        raise V177TrainingError("belief cache ended incomplete")
    return output


def _grounded_from_raw(
    core: V177Core,
    anchor_hypothesis_log: torch.Tensor,
    view_valid: torch.Tensor,
    *,
    ordered_prior: torch.Tensor | None = None,
    track_direct: torch.Tensor | None = None,
) -> V177GroundedState:
    if ordered_prior is None:
        direct = (
            anchor_hypothesis_log.new_ones(anchor_hypothesis_log.shape[:2])
            if track_direct is None
            else track_direct.to(anchor_hypothesis_log)
        )
        matrix = assignment_matrix_from_direct(direct)
        ordered = torch.logsumexp(
            matrix.clamp_min(1.0e-12).log()[..., :, :, None]
            + anchor_hypothesis_log.float()[..., None, :, :],
            dim=-2,
        )
        ordered = torch.log_softmax(ordered, dim=-1)
    else:
        ordered, assignment = identity_or_track_assignment(
            anchor_hypothesis_log, ordered_prior, view_valid,
        )
    return _grounded_from_ordered(core, ordered, view_valid)


def _grounded_from_ordered(
    core: V177Core,
    ordered_log_belief: torch.Tensor,
    view_valid: torch.Tensor,
    ordered_tokens: torch.Tensor | None = None,
) -> V177GroundedState:
    """Encode an ordered belief without smuggling appearance into dynamics."""

    has_appearance = ordered_tokens is not None
    features = ordered_tokens
    if features is None:
        features = ordered_log_belief.new_zeros(
            len(ordered_log_belief),
            VIEWS,
            HYPOTHESES,
            core.config.feature_dim,
        )
    appearance_available = (
        view_valid.bool()
        if has_appearance
        else torch.zeros_like(view_valid, dtype=torch.bool)
    )
    dynamics_grounding, statistics = core.grounding_encoder(
        ordered_log_belief, view_valid
    )
    return V177GroundedState(
        log_posterior=ordered_log_belief,
        ordered_tokens=features,
        dynamics_grounding=dynamics_grounding,
        belief_statistics=statistics,
        view_valid=view_valid,
        appearance_available=appearance_available,
    )


def _cache_tensors(source: Any, endpoint: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    values = source.get(endpoint) if hasattr(source, "get") else source.get_numpy(endpoint)
    return (
        torch.as_tensor(values[0], device=device, dtype=torch.float32),
        torch.as_tensor(values[1], device=device, dtype=torch.bool),
    )


def _completed_readback_feedback(
    commands: torch.Tensor,
    measured_path: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal measured motion and servo error for a completed chunk."""

    return (
        measured_path[:, 1:] - measured_path[:, :-1],
        measured_path[:, 1:] - commands,
    )


def _predict(
    core: V177Core,
    grounded: V177GroundedState,
    current_q: torch.Tensor,
    commands: torch.Tensor,
    recent_readback_deltas: torch.Tensor | None,
    recent_readback_errors: torch.Tensor | None,
    *,
    source_index: torch.Tensor | None = None,
    analytic: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    return core.predict(
        grounded,
        current_q,
        commands,
        recent_readback_deltas=recent_readback_deltas,
        recent_readback_errors=recent_readback_errors,
        source_index=source_index,
        analytic_shift_px=None if analytic is None else analytic[0],
        analytic_valid=None if analytic is None else analytic[1],
    )


def _fish_source_tensor(
    vocabulary: V177SourceVocabulary | None, batch: int, *, device: torch.device
) -> torch.Tensor | None:
    if vocabulary is None:
        return None
    return torch.full((int(batch),), int(vocabulary.fish_index), dtype=torch.long, device=device)


def _fish_analytic(
    core: V177Core,
    identity: V177GeometryIdentityProvider | None,
    current_q: torch.Tensor,
    commands: torch.Tensor,
    *,
    enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if identity is None or not enabled or not bool(core.config.analytic_transport):  # type: ignore[attr-defined]
        return None
    return identity.analytic_transport(current_q, commands)


def _archive_transition(
    archive: V181TemporalArchive,
    rows: np.ndarray,
    transition: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    selected = np.asarray(rows, dtype=np.int64)
    source = np.asarray(archive.row_endpoint_index[selected, transition], dtype=np.int64)
    target = np.asarray(archive.row_endpoint_index[selected, transition + 1], dtype=np.int64)
    commands = restore_companion_joint_units(
        np.asarray(archive.completed_commands[selected, transition], dtype=np.float32)
    )
    path = restore_companion_joint_units(
        np.asarray(archive.measured_state_path[selected, transition], dtype=np.float32)
    )
    if not bool(np.isfinite(path).all() and np.isfinite(commands).all()):
        raise V177TrainingError("archive transition contains non-finite joints")
    if int(transition) < TRANSITIONS - 1:
        next_q = restore_companion_joint_units(
            np.asarray(archive.measured_state_path[selected, transition + 1, 0], dtype=np.float32)
        )
        mismatch = float(np.max(np.abs(path[:, -1] - next_q)))
        if mismatch > 2.0e-4:
            raise V177TrainingError(f"archive endpoint/q clock shifted by {mismatch:.6g}")
    recent_deltas: np.ndarray | None = None
    recent_errors: np.ndarray | None = None
    if int(transition) > 0:
        previous_commands = restore_companion_joint_units(
            np.asarray(archive.completed_commands[selected, transition - 1], dtype=np.float32)
        )
        previous_path = restore_companion_joint_units(
            np.asarray(archive.measured_state_path[selected, transition - 1], dtype=np.float32)
        )
        recent_deltas = previous_path[:, 1:] - previous_path[:, :-1]
        recent_errors = previous_path[:, 1:] - previous_commands
    return (
        source,
        target,
        path[:, 0],
        commands,
        path,
        recent_deltas,
        recent_errors,
    )


def _archive_chain_arrays(
    archive: V181TemporalArchive,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one complete H0--H4 chain and validate its causal q clock."""

    selected = np.asarray(rows, dtype=np.int64)
    endpoint = np.asarray(archive.row_endpoint_index[selected], dtype=np.int64)
    commands = restore_companion_joint_units(
        np.asarray(archive.completed_commands[selected], dtype=np.float32)
    )
    paths = restore_companion_joint_units(
        np.asarray(archive.measured_state_path[selected], dtype=np.float32)
    )
    expected_endpoint_shape = (len(selected), TRANSITIONS + 1)
    expected_command_shape = (len(selected), TRANSITIONS, COMMAND_STEPS, STATE_DIM)
    expected_path_shape = (len(selected), TRANSITIONS, PATH_STEPS, STATE_DIM)
    if endpoint.shape != expected_endpoint_shape:
        raise V177TrainingError(
            f"archive endpoint chain must be {expected_endpoint_shape}, got {endpoint.shape}"
        )
    if commands.shape != expected_command_shape or paths.shape != expected_path_shape:
        raise V177TrainingError("archive command/readback chain shape differs")
    if not bool(np.isfinite(commands).all() and np.isfinite(paths).all()):
        raise V177TrainingError("archive chain contains non-finite joints")
    if TRANSITIONS > 1:
        mismatch = float(np.max(np.abs(paths[:, :-1, -1] - paths[:, 1:, 0])))
        if mismatch > 2.0e-4:
            raise V177TrainingError(
                f"archive recurrent q/readback clock shifted by {mismatch:.6g}"
            )
    return endpoint, commands, paths


def _anonymous_track_direct(
    archive: V181TemporalArchive,
    rows: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Choose anonymous H0 tracks without inventing physical identity.

    The query order has no cross-camera meaning, so the parity is independent
    for wrist and column while remaining fixed for every endpoint in a chain.
    """

    chain = np.asarray(archive.chain_id[np.asarray(rows, dtype=np.int64)], dtype=np.int64)
    view = np.arange(VIEWS, dtype=np.int64)[None, :]
    direct = ((chain[:, None] + SEED + view) % 2 == 0).astype(np.float32)
    return torch.from_numpy(direct).to(device)


def _archive_actor_batch(
    archive: V181TemporalArchive,
    ordinals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordinal = np.asarray(ordinals, dtype=np.int64)
    row = ordinal // TRANSITIONS
    transition = ordinal % TRANSITIONS
    batch = np.arange(len(row), dtype=np.int64)
    path = np.asarray(archive.measured_state_path[row], dtype=np.float32)
    command = np.asarray(archive.completed_commands[row], dtype=np.float32)
    current = path[batch, transition, 0]
    target = archive._project_targets(current, command[batch, transition])
    endpoint = np.asarray(archive.row_endpoint_index[row, transition], dtype=np.int64)
    return (
        row,
        transition,
        endpoint,
        restore_companion_joint_units(current),
        restore_companion_joint_units(target),
    )


def train_measurement(
    core: V177Core,
    real_rgb: V175AllRealRGB,
    cad: V175CADArchive,
    temporal_features: V177CombinedFeatureSource,
    cad_features: Any,
    archive: V181TemporalArchive,
    backbone: nn.Module,
    real_label_cache: str | Path,
    output_root: str | Path,
    *,
    device: torch.device,
    appearance_batch: int = 64,
    cad_batch: int = 64,
    real_batch: int = 32,
    smoke: bool = False,
    measurement_init: str | Path | None = None,
) -> dict[str, Any]:
    """Fit appearance, exchangeable CAD measurement, then the fixed real panel.

    ``measurement_init`` copies the measurement weights of a completed
    V17.6.2 run (identical measurement architecture) and skips this stage.
    That path cannot apply the separation penalty; it exists so the GDM and
    Actor changes can be evaluated without repeating the measurement fit.
    """

    output = Path(output_root).expanduser().resolve()
    stage_dir = output / "measurement"
    latest_path = stage_dir / "latest.pt"
    final_path = stage_dir / "final.pt"
    if final_path.is_file():
        finished = load_component(
            final_path, core, component="measurement", device=device
        )
        return dict(finished.get("metrics", {}))
    separation_weight = float(core.config.separation_weight)  # type: ignore[attr-defined]
    last_separation = float("nan")
    if measurement_init is not None:
        metrics = import_measurement_weights(core, measurement_init, device=device)
        core.requires_grad_(False)
        save_component(final_path, core, component="measurement", step=0, metrics=metrics)
        _merge_metrics(output / "train-metrics.json", "measurement", metrics)
        return metrics
    payload = _restore_latest(latest_path, core, component="measurement", device=device)
    resume_phase = str(payload.get("cursor", {}).get("phase", "appearance")) if payload else "appearance"
    phases = ("appearance", "cad", "real")
    phase_order = phases.index(resume_phase) if resume_phase in phases else 0
    losses: dict[str, Any] = (
        dict(payload.get("metrics", {})) if payload is not None else {}
    )
    losses["feature_schedule"] = "deterministic_cache_page_local"
    core.to(device)

    sorted_rows = np.argsort(np.asarray(archive.sidecar_row, dtype=np.int64))
    sorted_sidecar = np.asarray(archive.sidecar_row[sorted_rows], dtype=np.int64)
    locations = np.searchsorted(sorted_sidecar, np.asarray(real_rgb.sidecar_row, dtype=np.int64))
    archive_rows_for_rgb = sorted_rows[locations]
    canonical_for_local = np.asarray(
        archive.row_endpoint_index[archive_rows_for_rgb], dtype=np.int64
    ).reshape(-1)

    for phase_index, phase in enumerate(phases):
        if phase_index < phase_order:
            continue
        if phase == "appearance":
            selected_parameters = list(core.measurement.rgb_stem.parameters()) + list(
                core.measurement.rgb_dino_fusion.parameters()
            )
            parameters = _set_trainable(core, selected_parameters)
            population = np.arange(100_000, dtype=np.int64)
            population = _localized_feature_rows(
                population,
                canonical_for_local,
                temporal_features,
                seed=SEED + 1,
            )
            total = math.ceil(len(population) / appearance_batch)
            if smoke:
                total = min(total, 2)
            optimizer = torch.optim.AdamW(parameters, lr=FCPG_APPEARANCE_LR)
            start = _restore_optimizer(optimizer, payload, phase=phase)
            started = time.monotonic()
            last = float("nan")
            for step in range(start, total):
                rows = population[step * appearance_batch : (step + 1) * appearance_batch]
                rgb_np, valid_np = real_rgb.batch(rows)
                endpoint = canonical_for_local[rows]
                dino, cache_valid = temporal_features.get(endpoint)
                rgb = torch.as_tensor(rgb_np, device=device, dtype=torch.uint8)
                valid = torch.as_tensor(valid_np, device=device, dtype=torch.bool) & torch.as_tensor(cache_valid, device=device, dtype=torch.bool)
                dino_tensor = torch.as_tensor(dino, device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device):
                    loss = _measurement_feature_distillation(core, rgb, dino_tensor, valid)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
                optimizer.step()
                last = float(loss.detach())
                ordinal = step + 1
                _progress("measurement-appearance", ordinal, total, last, started)
                if ordinal % CHECKPOINT_EVERY == 0 or ordinal == total:
                    save_component(
                        latest_path, core, component="measurement", step=ordinal,
                        metrics={**losses, "appearance_loss": last}, optimizer=optimizer,
                        cursor={"phase": phase, "phase_step": ordinal},
                    )
            if not math.isfinite(last):
                last = float(losses.get("appearance_loss", last))
            losses["appearance_loss"] = last

        elif phase == "cad":
            parameters = _set_trainable(
                core,
                _measurement_spatial_parameters(core, update_rgb=False),
            )
            orders: list[np.ndarray] = []
            for pass_index in range(CAD_PASSES):
                values = np.asarray(cad.train_rows, dtype=np.int64).copy()
                orders.append(
                    _localized_feature_rows(
                        values,
                        values,
                        cad_features,
                        seed=SEED + 100 + pass_index,
                    )
                )
            population = np.concatenate(orders)
            total = math.ceil(len(population) / cad_batch)
            if smoke:
                total = min(total, 2)
            optimizer = torch.optim.AdamW(parameters, lr=FCPG_CAD_LR)
            start = _restore_optimizer(optimizer, payload, phase=phase)
            started = time.monotonic()
            last = float("nan")
            for step in range(start, total):
                rows = population[step * cad_batch : (step + 1) * cad_batch]
                raw_rgb = np.asarray(cad.rgb[rows], dtype=np.uint8)
                rgb = torch.from_numpy(np.ascontiguousarray(raw_rgb[:, ::-1])).to(device).permute(0, 1, 4, 2, 3)
                dino, feature_valid = cad_features.get(rows)
                dino_tensor = torch.as_tensor(dino, device=device, dtype=torch.float32)
                valid = torch.as_tensor(feature_valid, device=device, dtype=torch.bool)
                target_px = torch.from_numpy(
                    np.ascontiguousarray(np.asarray(cad.keypoints_px[rows], dtype=np.float32)[:, ::-1, 4:6])
                ).to(device)
                target_valid = torch.from_numpy(
                    np.ascontiguousarray(np.asarray(cad.valid[rows], dtype=np.bool_)[:, ::-1, 4:6])
                ).to(device)
                target = _bilinear_point_belief(target_px, target_valid)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device):
                    # Half of supervised CAD batches exercise the exact
                    # DINO-only path used by the large archive cache; the
                    # other half retains the real-RGB fusion path.
                    measured = core.measure(
                        dino_tensor,
                        valid,
                        rgb if step % 2 == 0 else None,
                    )
                    pair_loss = exchangeable_pair_nll(
                        measured.anchor_hypothesis_log_likelihood, target, valid
                    )
                    separation = hypothesis_separation_penalty(
                        measured.anchor_hypothesis_log_likelihood, target, valid
                    )
                    loss = pair_loss + separation_weight * separation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
                optimizer.step()
                last = float(pair_loss.detach())
                last_separation = float(separation.detach())
                ordinal = step + 1
                _progress("measurement-cad", ordinal, total, last, started)
                if ordinal % CHECKPOINT_EVERY == 0 or ordinal == total:
                    save_component(
                        latest_path, core, component="measurement", step=ordinal,
                        metrics={
                            **losses,
                            "cad_exchangeable_soft_target_cross_entropy": last,
                            "cad_hypothesis_overlap": last_separation,
                        },
                        optimizer=optimizer,
                        cursor={"phase": phase, "phase_step": ordinal},
                    )
            if not math.isfinite(last):
                last = float(
                    losses.get("cad_exchangeable_soft_target_cross_entropy", last)
                )
            losses["cad_exchangeable_soft_target_cross_entropy"] = last
            losses["cad_hypothesis_overlap"] = last_separation

        else:
            labels = _real_label_arrays(real_label_cache, backbone, device=device)
            valid_numpy = labels["valid"].detach().cpu().numpy()
            parameters = _set_trainable(
                core,
                _measurement_spatial_parameters(core, update_rgb=True),
            )
            total = 2 if smoke else REAL_REFIT_UPDATES
            optimizer = torch.optim.AdamW(parameters, lr=FCPG_REAL_LR)
            start = _restore_optimizer(optimizer, payload, phase=phase)
            sampler = np.random.default_rng(SEED + 300)
            # Replay deterministic draws up to the resume cursor.
            for _ in range(start):
                balanced_real_anchor_batch(valid_numpy, real_batch, sampler)
            started = time.monotonic()
            last = float("nan")
            for step in range(start, total):
                selected, supervised = balanced_real_anchor_batch(valid_numpy, real_batch, sampler)
                index = torch.as_tensor(selected, device=device, dtype=torch.long)
                valid = labels["valid"][index] & torch.as_tensor(supervised, device=device, dtype=torch.bool)
                target_valid = valid[:, :, None].expand(-1, -1, HYPOTHESES)
                target = _bilinear_point_belief(labels["target"][index], target_valid)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device):
                    measured = core.measure(
                        labels["dino"][index],
                        valid,
                        labels["rgb"][index] if step % 2 == 0 else None,
                    )
                    pair_loss = exchangeable_pair_nll(
                        measured.anchor_hypothesis_log_likelihood, target, valid
                    )
                    separation = hypothesis_separation_penalty(
                        measured.anchor_hypothesis_log_likelihood, target, valid
                    )
                    loss = pair_loss + separation_weight * separation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
                optimizer.step()
                last = float(pair_loss.detach())
                last_separation = float(separation.detach())
                ordinal = step + 1
                _progress("measurement-real-fit", ordinal, total, last, started)
                if ordinal % CHECKPOINT_EVERY == 0 or ordinal == total:
                    save_component(
                        latest_path, core, component="measurement", step=ordinal,
                        metrics={
                            **losses,
                            "real_training_fit_exchangeable_soft_target_cross_entropy": last,
                        },
                        optimizer=optimizer,
                        cursor={"phase": phase, "phase_step": ordinal},
                    )
            if not math.isfinite(last):
                last = float(
                    losses.get(
                        "real_training_fit_exchangeable_soft_target_cross_entropy",
                        last,
                    )
                )
            losses[
                "real_training_fit_exchangeable_soft_target_cross_entropy"
            ] = last
            losses["real_training_fit_hypothesis_overlap"] = last_separation
        payload = None

    core.requires_grad_(False)
    losses["separation_weight"] = float(separation_weight)
    losses["cad_dino_only_batch_fraction"] = 0.5
    losses["real_refit_dino_only_batch_fraction"] = 0.5
    losses["cad_fused_batches_update_rgb_stem_and_fusion"] = False
    losses["cad_fused_batches_train_readout_over_fixed_rgb_features"] = True
    losses["real_refit_updates_rgb_stem_and_fusion"] = True
    losses["dino_projection_frozen_all_measurement_stages"] = True
    completed_refit_updates = 2 if smoke else REAL_REFIT_UPDATES
    losses["completed_real_refit_updates"] = int(completed_refit_updates)
    save_component(
        final_path,
        core,
        component="measurement",
        step=completed_refit_updates,
        metrics=losses,
    )
    _merge_metrics(output / "train-metrics.json", "measurement", losses)
    return losses


def train_lewm(
    core: V177Core,
    archive: V181TemporalArchive,
    temporal_beliefs: V177CombinedBeliefSource,
    dense_beliefs: V177BeliefCache,
    dense_rows: np.ndarray,
    output_root: str | Path,
    *,
    device: torch.device,
    vocabulary: V177SourceVocabulary,
    fish: V181FishArchive | None = None,
    fish_source: "V177FishMeasurementSource | None" = None,
    identity: V177GeometryIdentityProvider | None = None,
    effective_batch: int = 256,
    microbatch: int = 64,
    smoke: bool = False,
) -> dict[str, Any]:
    """Fit the GDM with causal H0--H3 unrolls and stopped raw measurements.

    ``effective_batch`` and ``microbatch`` count transition examples.  Each
    chain contributes all four transitions, preserving the original exposure
    budget while making the process state genuinely recurrent.

    V17.7: only TRAIN rows enter the optimizer; every archive chain carries
    its camera-rig index (with source dropout); a third fixed fish phase trains
    the analytic-residual transport on the calibrated rig.
    """

    _check_source_vocabulary(core, vocabulary)
    row_source = vocabulary.row_index(archive)
    if effective_batch % TRANSITIONS or microbatch % TRANSITIONS:
        raise V177TrainingError(
            "LeWM transition batches must be divisible by the four transitions"
        )
    chain_effective_batch = effective_batch // TRANSITIONS
    chain_microbatch = microbatch // TRANSITIONS
    if chain_effective_batch % chain_microbatch:
        raise V177TrainingError("LeWM chain batch must divide into microbatches")
    configured_scale = np.asarray(
        core.config.servo_error_scale_deg, dtype=np.float32
    )
    if not np.allclose(
        configured_scale,
        np.asarray(READBACK_ERROR_STD_DEG, dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise V177TrainingError(
            "LeWM servo-error scale differs from the fixed full-archive statistic"
        )
    output = Path(output_root).expanduser().resolve()
    stage_dir = output / "lewm"
    latest_path = stage_dir / "latest.pt"
    final_path = stage_dir / "final.pt"
    if final_path.is_file():
        finished = load_component(final_path, core, component="lewm", device=device)
        return dict(finished.get("metrics", {}))
    payload = _restore_latest(latest_path, core, component="lewm", device=device)
    parameters = _set_trainable(
        core,
        list(core.grounding_encoder.parameters()) + list(core.lewm.parameters()),
    )
    optimizer = torch.optim.AdamW(parameters, lr=LEWM_LR)
    if payload is not None and isinstance(payload.get("optimizer"), Mapping):
        optimizer.load_state_dict(payload["optimizer"])
    all_rows = _optimizer_rows(archive, archive.rows_for_partition("train"))
    dense_population = _optimizer_rows(
        archive,
        np.intersect1d(np.asarray(dense_rows, dtype=np.int64), all_rows, assume_unique=False),
    )
    if len(dense_population) == 0:
        raise V177TrainingError("dense 20K subset has no TRAIN rows")
    phases: list[tuple[str, np.ndarray, Any]] = [
        ("archive", all_rows, temporal_beliefs),
        ("dense", dense_population, dense_beliefs),
    ]
    fish_updates = 0
    if fish is not None:
        if fish_source is None or identity is None:
            raise V177TrainingError("fish GDM phase needs a fish measurement source and calibration")
        fish_updates = 2 if smoke else LEWM_FISH_UPDATES
    if smoke:
        # A smoke run must consume a small, predictable prefix so its belief
        # cache can be bounded as well.  The full run keeps the localized,
        # deterministically shuffled schedule below.
        schedules = [
            (name, rows[:chain_effective_batch], source)
            for name, rows, source in phases[:2]
        ]
        total = len(schedules) + fish_updates
    else:
        schedules = []
        for phase_index, (name, rows, beliefs) in enumerate(phases):
            order = _localized_archive_rows(
                archive,
                rows,
                horizon=0,
                batch_size=chain_effective_batch,
                seed=SEED + 500 + phase_index,
            )
            schedules.append((name, order, beliefs))
        total = sum(
            math.ceil(len(rows) / chain_effective_batch)
            for _, rows, _ in schedules
        ) + fish_updates
    start = int(payload.get("step", 0)) if payload is not None else 0
    global_step = 0
    started = time.monotonic()
    previous_metrics = payload.get("metrics", {}) if payload is not None else {}
    last_world = float(previous_metrics.get("world_loss", float("nan")))
    last_belief = float(
        previous_metrics.get(
            "exchangeable_soft_target_cross_entropy", float("nan")
        )
    )
    last_state = float(previous_metrics.get("q_path_smooth_l1", float("nan")))
    restored_accumulator = previous_metrics.get("metric_accumulator", {})
    horizon_belief_sum = np.asarray(
        restored_accumulator.get("belief_sum", [0.0] * TRANSITIONS),
        dtype=np.float64,
    )
    horizon_belief_count = np.asarray(
        restored_accumulator.get("belief_count", [0] * TRANSITIONS),
        dtype=np.int64,
    )
    horizon_state_sum = np.asarray(
        restored_accumulator.get("state_sum", [0.0] * TRANSITIONS),
        dtype=np.float64,
    )
    horizon_state_count = np.asarray(
        restored_accumulator.get("state_count", [0] * TRANSITIONS),
        dtype=np.int64,
    )
    phase_chain_exposures: dict[str, int] = {
        name: int(
            restored_accumulator.get("phase_chain_exposures", {}).get(name, 0)
        )
        for name, _, _ in schedules
    }
    core.train()
    core.measurement.eval()
    for phase, rows, belief_source in schedules:
        for selected in _batches(rows, chain_effective_batch):
            if global_step < start:
                global_step += 1
                continue
            target_endpoints = np.asarray(
                archive.row_endpoint_index[selected, 1 : TRANSITIONS + 1],
                dtype=np.int64,
            )
            expected_valid_by_horizon = np.asarray(
                archive.endpoint_view_valid[target_endpoints],
                dtype=np.bool_,
            ).sum(axis=(0, 2), dtype=np.int64)
            total_valid_views = float(
                max(1, int(expected_valid_by_horizon.sum()))
            )
            optimizer.zero_grad(set_to_none=True)
            belief_sum = torch.zeros(TRANSITIONS, device=device)
            belief_count = torch.zeros(TRANSITIONS, device=device)
            state_sum = torch.zeros(TRANSITIONS, device=device)
            chain_count = 0
            transition_denominator = float(len(selected) * TRANSITIONS)
            chunks = list(_batches(selected, chain_microbatch))
            for micro in chunks:
                endpoint_np, commands_np, paths_np = _archive_chain_arrays(
                    archive, micro
                )
                source_values = _cache_tensors(
                    belief_source, endpoint_np[:, 0], device
                )
                grounded = _grounded_from_raw(
                    core,
                    *source_values,
                    track_direct=_anonymous_track_direct(archive, micro, device=device),
                )
                commands_all = _tensor(commands_np, device)
                paths_all = _tensor(paths_np, device)
                current_q = paths_all[:, 0, 0]
                recent_deltas: torch.Tensor | None = None
                recent_errors: torch.Tensor | None = None
                state_weight = float(len(micro)) / transition_denominator
                # The rig identity is fixed for the whole chain; dropout is
                # decided once per chain so the unknown-rig path stays trained.
                source_index = _source_tensor(
                    row_source,
                    micro,
                    device=device,
                    dropout=float(core.config.source_dropout),  # type: ignore[attr-defined]
                    seed=SEED + 900 + global_step * 7919 + int(micro[0]),
                )
                for transition in range(TRANSITIONS):
                    commands = commands_all[:, transition]
                    path = paths_all[:, transition]
                    if not torch.equal(current_q, path[:, 0]):
                        mismatch = float((current_q - path[:, 0]).abs().max().cpu())
                        raise V177TrainingError(
                            f"recursive LeWM q clock shifted by {mismatch:.6g}"
                        )
                    target_values = _cache_tensors(
                        belief_source, endpoint_np[:, transition + 1], device
                    )
                    with _autocast(device):
                        prediction = _predict(
                            core,
                            grounded,
                            current_q,
                            commands,
                            recent_deltas,
                            recent_errors,
                            source_index=source_index,
                        )
                        # Supervise with the stopped, raw next-image measurement.
                        belief_loss_sum, valid_count = (
                            _exchangeable_pair_sum_count(
                                prediction.prior_log_belief,
                                target_values[0].detach().exp(),
                                target_values[1],
                            )
                        )
                        state_loss = state_path_smooth_l1(
                            prediction.predicted_q_path, path
                        )
                        loss = (
                            0.5 * belief_loss_sum / total_valid_views
                            + 0.5 * state_loss * state_weight
                        )
                    loss.backward()
                    belief_sum[transition] += belief_loss_sum.detach()
                    belief_count[transition] += valid_count.detach()
                    state_sum[transition] += state_loss.detach() * len(micro)

                    # Association and factorized PoE form the next stopped state.
                    with torch.no_grad():
                        ordered_target, _ = identity_or_track_assignment(
                            target_values[0],
                            prediction.prior_log_belief.detach(),
                            target_values[1],
                        )
                        next_posterior = fuse_measurement_prior(
                            prediction.prior_log_belief.detach(),
                            ordered_target.detach(),
                            target_values[1],
                        ).detach()
                    if transition + 1 < TRANSITIONS:
                        grounded = _grounded_from_ordered(
                            core,
                            next_posterior,
                            target_values[1],
                        )
                    recent_deltas, recent_errors = _completed_readback_feedback(
                        commands, path
                    )
                    current_q = path[:, -1]
                chain_count += len(micro)
            observed_valid = belief_count.detach().cpu().numpy().astype(np.int64)
            if not bool(np.array_equal(observed_valid, expected_valid_by_horizon)):
                raise V177TrainingError(
                    "belief-cache validity differs from canonical endpoint validity"
                )
            torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
            optimizer.step()
            global_step += 1
            if chain_count != len(selected):
                raise V177TrainingError("LeWM chain microbatch accounting differs")
            belief_by_horizon = (
                belief_sum / belief_count.clamp_min(1)
            ).detach().cpu().numpy()
            state_by_horizon = (
                state_sum / max(1, chain_count)
            ).detach().cpu().numpy()
            last_belief = float(
                (belief_sum.sum() / belief_count.sum().clamp_min(1)).detach().cpu()
            )
            last_state = float(state_by_horizon.mean())
            last_world = 0.5 * last_belief + 0.5 * last_state
            horizon_belief_sum += belief_sum.detach().cpu().numpy()
            horizon_belief_count += observed_valid
            horizon_state_sum += state_sum.detach().cpu().numpy()
            horizon_state_count += chain_count
            phase_chain_exposures[phase] += int(chain_count)
            _progress(f"lewm-{phase}", global_step, total, last_world, started)
            if global_step % CHECKPOINT_EVERY == 0 or global_step == total:
                save_component(
                    latest_path, core, component="lewm", step=global_step,
                    metrics={
                        "world_loss": last_world,
                        "exchangeable_soft_target_cross_entropy": last_belief,
                        "q_path_smooth_l1": last_state,
                        "metric_accumulator": {
                            "belief_sum": horizon_belief_sum.tolist(),
                            "belief_count": horizon_belief_count.tolist(),
                            "state_sum": horizon_state_sum.tolist(),
                            "state_count": horizon_state_count.tolist(),
                            "phase_chain_exposures": dict(
                                phase_chain_exposures
                            ),
                        },
                    },
                    optimizer=optimizer,
                    cursor={"phase": phase},
                )

    fish_metrics: dict[str, Any] = {}
    if fish_updates > 0:
        assert fish is not None and fish_source is not None and identity is not None
        fish_optimizer = torch.optim.AdamW(parameters, lr=LEWM_FISH_LR)
        if (
            payload is not None
            and payload.get("cursor", {}).get("phase") == "fish"
            and isinstance(payload.get("optimizer"), Mapping)
        ):
            fish_optimizer.load_state_dict(payload["optimizer"])
        train_sequence_count = len(fish.sequences_for_partition("train"))
        if train_sequence_count <= 0:
            raise V177TrainingError("fish training partition has no causal sequence")
        fish_batch = max(1, chain_microbatch)
        fish_belief = {"analytic": [0.0, 0], "learned_only": [0.0, 0]}
        fish_state = {"analytic": [0.0, 0], "learned_only": [0.0, 0]}
        for fish_step in range(fish_updates):
            if global_step < start:
                global_step += 1
                continue
            generator = np.random.default_rng(SEED + 1200 + fish_step)
            sequences = generator.integers(0, train_sequence_count, size=fish_batch, dtype=np.int64)
            analytic_enabled = (fish_step % 2 == 0) or LEWM_FISH_ANALYTIC_DROPOUT <= 0.0
            fish_optimizer.zero_grad(set_to_none=True)
            belief_sum, valid_count, state_mean = _fish_lewm_chain_loss(
                core,
                fish,
                fish_source,
                identity,
                vocabulary,
                sequences,
                device=device,
                analytic_enabled=analytic_enabled,
            )
            loss = 0.5 * belief_sum / valid_count.clamp_min(1.0) + 0.5 * state_mean
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
            fish_optimizer.step()
            global_step += 1
            key = "analytic" if analytic_enabled else "learned_only"
            fish_belief[key][0] += float(belief_sum.detach().cpu())
            fish_belief[key][1] += int(valid_count.detach().cpu())
            fish_state[key][0] += float(state_mean.detach().cpu())
            fish_state[key][1] += 1
            last_world = float(loss.detach().cpu())
            _progress("lewm-fish", global_step, total, last_world, started)
            if global_step % CHECKPOINT_EVERY == 0 or global_step == total:
                save_component(
                    latest_path, core, component="lewm", step=global_step,
                    metrics={
                        "world_loss": last_world,
                        "exchangeable_soft_target_cross_entropy": last_belief,
                        "q_path_smooth_l1": last_state,
                        "metric_accumulator": {
                            "belief_sum": horizon_belief_sum.tolist(),
                            "belief_count": horizon_belief_count.tolist(),
                            "state_sum": horizon_state_sum.tolist(),
                            "state_count": horizon_state_count.tolist(),
                            "phase_chain_exposures": dict(phase_chain_exposures),
                        },
                    },
                    optimizer=fish_optimizer,
                    cursor={"phase": "fish", "fish_step": fish_step + 1},
                )
        fish_metrics = {
            "fish_updates": int(fish_updates),
            "fish_sequence_batch": int(fish_batch),
            "fish_train_episodes": [0, 1, 3],
            "fish_analytic_dropout": float(LEWM_FISH_ANALYTIC_DROPOUT),
            "fish_geometry_prior_at_h0": bool(core.config.geometry_prior_fusion),  # type: ignore[attr-defined]
            "fish_exchangeable_soft_target_cross_entropy": {
                key: (value[0] / value[1] if value[1] else None)
                for key, value in fish_belief.items()
            },
            "fish_q_path_smooth_l1": {
                key: (value[0] / value[1] if value[1] else None)
                for key, value in fish_state.items()
            },
        }

    metrics = {
        "steps": int(global_step),
        "world_loss": last_world,
        "exchangeable_soft_target_cross_entropy": last_belief,
        "q_path_smooth_l1": last_state,
        "effective_batch": int(effective_batch),
        "microbatch": int(microbatch),
        "chain_effective_batch": int(chain_effective_batch),
        "chain_microbatch": int(chain_microbatch),
        "causal_recursive_unroll": True,
        "stop_gradient_between_transitions": True,
        "belief_target": "stopped_raw_next_measurement",
        "belief_scored_when": "next_measurement_view_valid",
        "loss_reduction": "pair_ce_valid_view_sum_count_q_future_sample_mean",
        "q_path_supervised_steps": [1, 2, 3, 4, 5],
        "phase_order": ["archive", "dense"] + (["fish"] if fish_updates > 0 else []),
        "archive_partition": "train_only",
        "archive_train_rows": int(len(all_rows)),
        "archive_dev_rows_excluded": int((np.asarray(archive.split) == 1).sum()),
        "source_conditioning": {
            "vocabulary_size": int(vocabulary.size),
            "unknown_index": SOURCE_UNKNOWN,
            "dropout": float(core.config.source_dropout),  # type: ignore[attr-defined]
            "fish_index": int(vocabulary.fish_index),
        },
        "analytic_transport": bool(core.config.analytic_transport),  # type: ignore[attr-defined]
        "mass_preserving_blur": bool(core.config.mass_preserving_blur),  # type: ignore[attr-defined]
        **fish_metrics,
        "phase_chain_exposures": {
            key: int(value) for key, value in phase_chain_exposures.items()
        },
        "transition_exposures": int(
            sum(phase_chain_exposures.values()) * TRANSITIONS
        ),
        "per_horizon_exchangeable_soft_target_cross_entropy": {
            f"H{index}_to_H{index + 1}": float(
                horizon_belief_sum[index] / max(1, horizon_belief_count[index])
            )
            for index in range(TRANSITIONS)
        },
        "per_horizon_q_path_smooth_l1": {
            f"H{index}_to_H{index + 1}": float(
                horizon_state_sum[index] / max(1, horizon_state_count[index])
            )
            for index in range(TRANSITIONS)
        },
        "uses_current_transition_readback_as_input": False,
        "completed_readback_inputs": ["motion_delta", "servo_error"],
        "readback_error_statistics": {
            "source": "all_320000_archive_rows_all_four_completed_transitions",
            "mean_deg": list(READBACK_ERROR_MEAN_DEG),
            "std_deg": list(READBACK_ERROR_STD_DEG),
            "normalization": "per_joint_std_deg",
        },
    }
    core.requires_grad_(False)
    save_component(final_path, core, component="lewm", step=global_step, metrics=metrics)
    _merge_metrics(output / "train-metrics.json", "lewm", metrics)
    return metrics


def _fish_h0_update(
    core: V177Core,
    identity: V177GeometryIdentityProvider | None,
    measurement: V177Measurement,
    current_q: torch.Tensor,
) -> V177FilterUpdate:
    """H0 filter update on the calibrated rig: geometry orders and (V17.7) primes."""

    if identity is None:
        return core.update(core.assign_identity(measurement))
    # ``broad_log_prior`` runs under inference mode; clone so autograd never
    # tries to save an inference tensor.
    geometry = identity.broad_log_prior(
        current_q,
        measurement.view_valid,
        sigma_px=float(core.config.geometry_prior_sigma_px),  # type: ignore[attr-defined]
    ).clone()
    assigned = core.assign_identity(measurement, geometry)
    return core.update(assigned, geometry_log_prior=geometry)


def _fish_lewm_chain_loss(
    core: V177Core,
    fish: V181FishArchive,
    fish_source: "V177FishMeasurementSource",
    identity: V177GeometryIdentityProvider,
    vocabulary: V177SourceVocabulary,
    sequences: np.ndarray,
    *,
    device: torch.device,
    analytic_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal H0--H4 GDM loss on fish TRAIN sequences with stopped targets.

    Mirrors the archive chain loop: the prior is scored against the raw next
    measurement, the PoE posterior is detached before it becomes the next
    state, and only completed readback enters the next prediction.
    """

    base_ordinal = np.asarray(sequences, dtype=np.int64) * TRANSITIONS
    first = fish.pair_batch(base_ordinal, partition="train", allow_held=False)
    measurement = fish_source.measure(core, first.source_frame, device)
    q = _tensor(first.current_q * 100.0, device)
    with _autocast(device):
        grounded = _fish_h0_update(core, identity, measurement, q).grounded
    source = _fish_source_tensor(vocabulary, len(q), device=device)
    recent_deltas: torch.Tensor | None = None
    recent_errors: torch.Tensor | None = None
    belief_sum = torch.zeros((), device=device)
    valid_count = torch.zeros((), device=device)
    state_losses: list[torch.Tensor] = []
    for step in range(TRANSITIONS):
        pair = fish.pair_batch(base_ordinal + step, partition="train", allow_held=False)
        commands = _tensor(_fish_commands_degrees(pair), device)
        path = _tensor(pair.measured_q_path * 100.0, device)
        if not torch.allclose(q, path[:, 0], atol=2.0e-3):
            raise V177TrainingError("fish GDM chain q clock is shifted")
        analytic = _fish_analytic(core, identity, q, commands, enabled=analytic_enabled)
        next_measurement = fish_source.measure(core, pair.target_frame, device)
        with _autocast(device):
            prediction = _predict(
                core,
                grounded,
                q,
                commands,
                recent_deltas,
                recent_errors,
                source_index=source,
                analytic=analytic,
            )
            target = next_measurement.anchor_hypothesis_log_likelihood.detach().exp()
            loss_sum, count = _exchangeable_pair_sum_count(
                prediction.prior_log_belief, target, next_measurement.view_valid
            )
            state_loss = state_path_smooth_l1(prediction.predicted_q_path, path)
        belief_sum = belief_sum + loss_sum
        valid_count = valid_count + count
        state_losses.append(state_loss)
        with torch.no_grad():
            ordered_target, _ = identity_or_track_assignment(
                next_measurement.anchor_hypothesis_log_likelihood,
                prediction.prior_log_belief.detach(),
                next_measurement.view_valid,
            )
            posterior = fuse_measurement_prior(
                prediction.prior_log_belief.detach(),
                ordered_target,
                next_measurement.view_valid,
            ).detach()
        if step + 1 < TRANSITIONS:
            with _autocast(device):
                grounded = _grounded_from_ordered(core, posterior, next_measurement.view_valid)
        recent_deltas, recent_errors = _completed_readback_feedback(commands, path)
        q = path[:, -1]
    return belief_sum, valid_count, torch.stack(state_losses).mean()


def _null_cached_update(
    core: V177Core,
    values: tuple[torch.Tensor, torch.Tensor],
    *,
    track_direct: torch.Tensor,
) -> V177FilterUpdate:
    grounded = _grounded_from_raw(core, *values, track_direct=track_direct)
    batch = len(grounded.dynamics_grounding)
    compact = core.innovation_encoder.null(
        batch,
        grounded.view_valid,
        device=grounded.dynamics_grounding.device,
        dtype=grounded.dynamics_grounding.dtype,
    )
    innovation = V177Innovation(
        innovation=compact,
        belief_surprise=grounded.dynamics_grounding.new_zeros(
            batch, VIEWS, HYPOTHESES
        ),
        state_residual=grounded.dynamics_grounding.new_zeros(
            batch, PATH_STEPS, STATE_DIM
        ),
        view_valid=grounded.view_valid,
        has_prediction=torch.zeros(
            batch, device=grounded.dynamics_grounding.device, dtype=torch.bool
        ),
    )
    return V177FilterUpdate(grounded=grounded, innovation=innovation)


def _predicted_cached_update(
    core: V177Core,
    values: tuple[torch.Tensor, torch.Tensor],
    prediction: Any,
    measured_path: torch.Tensor,
) -> V177FilterUpdate:
    raw_log, view_valid = values
    ordered_measurement, _ = identity_or_track_assignment(
        raw_log,
        prediction.prior_log_belief,
        view_valid,
    )
    posterior = fuse_measurement_prior(
        prediction.prior_log_belief, ordered_measurement, view_valid
    )
    # Archive G uses only the belief map/statistics.  Current appearance remains
    # available through Actor scene tokens; exact posterior feature pooling is
    # recomputed in the 20K RGB and fish stages.
    grounded = _grounded_from_ordered(
        core,
        posterior,
        view_valid,
    )
    signals = compute_innovation(
        posterior,
        prediction.prior_log_belief,
        view_valid,
        measured_path,
        prediction.predicted_q_path,
    )
    innovation = V177Innovation(
        innovation=core.innovation_encoder(signals),
        belief_surprise=signals.surprise,
        state_residual=signals.state_residual,
        view_valid=view_valid,
        has_prediction=torch.ones(len(posterior), device=posterior.device, dtype=torch.bool),
    )
    return V177FilterUpdate(grounded=grounded, innovation=innovation)


def _pooled_scene_from_features(
    features: Any, endpoint: np.ndarray, *, device: torch.device
) -> torch.Tensor:
    """Mean-pooled normalized DINO tokens for one endpoint set."""

    scene, scene_valid = features.get(np.asarray(endpoint, dtype=np.int64))
    valid = torch.as_tensor(scene_valid, device=device, dtype=torch.bool)
    tokens = normalize_actor_scene_tokens(
        torch.as_tensor(scene, device=device, dtype=torch.float32), valid
    )
    return pool_scene_tokens(tokens, valid)


def _archive_cached_actor_context(
    core: V177Core,
    archive: V181TemporalArchive,
    beliefs: V177CombinedBeliefSource,
    features: V177CombinedFeatureSource,
    rows: np.ndarray,
    horizon: int,
    *,
    device: torch.device,
    row_source: np.ndarray | None = None,
    collect_history: bool = False,
    source_override: int | None = None,
) -> tuple[torch.Tensor, V177FilterUpdate, torch.Tensor, V177RawHistory]:
    """Recursive causal Actor context from cached archive beliefs.

    ``row_source`` supplies the rig index per archive row; ``source_override``
    replaces it (``SOURCE_UNKNOWN`` reproduces a rig-blind GDM for probes).
    ``collect_history`` also assembles the raw-history baseline inputs.
    """

    selected = np.asarray(rows, dtype=np.int64)
    endpoint0 = np.asarray(archive.row_endpoint_index[selected, 0], dtype=np.int64)
    update = _null_cached_update(
        core,
        _cache_tensors(beliefs, endpoint0, device),
        track_direct=_anonymous_track_direct(archive, selected, device=device),
    )
    raw_paths = np.asarray(archive.measured_state_path[selected], dtype=np.float32)
    raw_commands = np.asarray(archive.completed_commands[selected], dtype=np.float32)
    if int(horizon) > 0:
        left = raw_paths[:, : int(horizon), -1]
        right = raw_paths[:, 1 : int(horizon) + 1, 0]
        if float(np.max(np.abs(left - right))) > 2.0e-6:
            raise V177TrainingError("archive recurrent q/readback clock shifted")
    q = _tensor(restore_companion_joint_units(raw_paths[:, 0, 0]), device)
    if source_override is not None:
        source_index = torch.full((len(selected),), int(source_override), dtype=torch.long, device=device)
    elif row_source is not None:
        source_index = _source_tensor(row_source, selected, device=device)
    else:
        source_index = None
    history = empty_raw_history(len(selected), device=device)
    recent_deltas: torch.Tensor | None = None
    recent_errors: torch.Tensor | None = None
    for step in range(int(horizon)):
        commands = _tensor(restore_companion_joint_units(raw_commands[:, step]), device)
        path = _tensor(restore_companion_joint_units(raw_paths[:, step]), device)
        prediction = _predict(
            core,
            update.grounded,
            q,
            commands,
            recent_deltas,
            recent_errors,
            source_index=source_index,
        )
        endpoint = np.asarray(archive.row_endpoint_index[selected, step + 1], dtype=np.int64)
        update = _predicted_cached_update(
            core, _cache_tensors(beliefs, endpoint, device), prediction, path
        )
        if collect_history:
            past_endpoint = np.asarray(archive.row_endpoint_index[selected, step], dtype=np.int64)
            history = append_raw_history(
                history,
                commands,
                path,
                _pooled_scene_from_features(features, past_endpoint, device=device),
            )
        recent_deltas, recent_errors = _completed_readback_feedback(
            commands, path
        )
        q = path[:, -1]
    current_endpoint = np.asarray(
        archive.row_endpoint_index[selected, int(horizon)], dtype=np.int64
    )
    scene, scene_valid = features.get(current_endpoint)
    valid = torch.as_tensor(scene_valid, device=device, dtype=torch.bool)
    scene_tensor = normalize_actor_scene_tokens(
        torch.as_tensor(scene, device=device, dtype=torch.float32),
        valid,
    )
    return scene_tensor, update, q, history


class V177DenseMeasurementSource:
    """Recompute full RGB measurements for the existing 20K endpoint subset."""

    def __init__(
        self,
        real_rgb: V175AllRealRGB,
        archive: V181TemporalArchive,
        features: V177CombinedFeatureSource,
    ) -> None:
        endpoint, local = dense_endpoint_map(real_rgb, archive)
        self.endpoint = endpoint
        self.local = local
        self.real_rgb = real_rgb
        self.features = features

    def archive_rows(self, archive: V181TemporalArchive) -> np.ndarray:
        return np.flatnonzero(
            np.isin(np.asarray(archive.sidecar_row, dtype=np.int64), np.asarray(self.real_rgb.sidecar_row, dtype=np.int64))
        ).astype(np.int64)

    def measure(self, core: V177Core, endpoint: np.ndarray, device: torch.device) -> V177Measurement:
        requested = np.asarray(endpoint, dtype=np.int64)
        position = np.searchsorted(self.endpoint, requested)
        safe = np.minimum(position, len(self.endpoint) - 1)
        if bool((position >= len(self.endpoint)).any()) or not bool(np.array_equal(self.endpoint[safe], requested)):
            raise V177TrainingError("endpoint is outside the 20K RGB subset")
        rgb_np, rgb_valid = self.real_rgb.batch(self.local[position])
        dino, dino_valid = self.features.get(requested)
        rgb = torch.as_tensor(rgb_np, device=device, dtype=torch.uint8)
        tokens = torch.as_tensor(dino, device=device, dtype=torch.float32)
        valid = torch.as_tensor(rgb_valid, device=device, dtype=torch.bool) & torch.as_tensor(dino_valid, device=device, dtype=torch.bool)
        with torch.no_grad():
            return core.measure(tokens, valid, rgb)


class V177FishMeasurementSource:
    def __init__(self, fish: V181FishArchive, features: Any) -> None:
        self.fish = fish
        self.features = features

    def measure(self, core: V177Core, endpoint: np.ndarray, device: torch.device) -> V177Measurement:
        requested = np.asarray(endpoint, dtype=np.int64)
        rgb_np = self.fish.rgb(requested)
        dino, dino_valid = self.features.get(requested)
        rgb = torch.as_tensor(rgb_np, device=device, dtype=torch.uint8).permute(0, 1, 4, 2, 3)
        tokens = torch.as_tensor(dino, device=device, dtype=torch.float32)
        valid = torch.as_tensor(dino_valid, device=device, dtype=torch.bool)
        with torch.no_grad():
            return core.measure(tokens, valid, rgb)


def _assigned_from_measurement(
    core: V177Core,
    measurement: V177Measurement,
    prior: torch.Tensor,
) -> V177AssignedMeasurement:
    return core.assign_identity(measurement, prior)


def _actor_update_from_measurements(
    core: V177Core,
    identity: V177GeometryIdentityProvider | None,
    current_measurement: V177Measurement,
    current_q: torch.Tensor,
    *,
    previous_measurement: V177Measurement | None = None,
    previous_q: torch.Tensor | None = None,
    previous_commands: torch.Tensor | None = None,
    previous_path: torch.Tensor | None = None,
    previous_recent_readback_deltas: torch.Tensor | None = None,
    previous_recent_readback_errors: torch.Tensor | None = None,
    source_index: torch.Tensor | None = None,
):
    if previous_measurement is None:
        return _fish_h0_update(core, identity, current_measurement, current_q)
    if previous_q is None or previous_commands is None or previous_path is None:
        raise V177TrainingError("previous transition inputs are incomplete")
    previous_update = _fish_h0_update(core, identity, previous_measurement, previous_q)
    prediction = _predict(
        core,
        previous_update.grounded,
        previous_q,
        previous_commands,
        previous_recent_readback_deltas,
        previous_recent_readback_errors,
        source_index=source_index,
        analytic=_fish_analytic(core, identity, previous_q, previous_commands),
    )
    current_assigned = _assigned_from_measurement(
        core, current_measurement, prediction.prior_log_belief
    )
    return core.update(current_assigned, prediction, previous_path)


def _archive_actor_context(
    core: V177Core,
    archive: V181TemporalArchive,
    source: V177DenseMeasurementSource,
    rows: np.ndarray,
    transition: int,
    *,
    device: torch.device,
    row_source: np.ndarray | None = None,
    collect_history: bool = False,
):
    selected = np.asarray(rows, dtype=np.int64)
    raw_paths = np.asarray(archive.measured_state_path[selected], dtype=np.float32)
    raw_commands = np.asarray(archive.completed_commands[selected], dtype=np.float32)
    if int(transition) > 0:
        left = raw_paths[:, : int(transition), -1]
        right = raw_paths[:, 1 : int(transition) + 1, 0]
        if float(np.max(np.abs(left - right))) > 2.0e-6:
            raise V177TrainingError("dense recurrent q/readback clock shifted")
    endpoint = np.asarray(archive.row_endpoint_index[selected, 0], dtype=np.int64)
    measurement = source.measure(core, endpoint, device)
    q = _tensor(restore_companion_joint_units(raw_paths[:, 0, 0]), device)
    # The archive has no per-source camera calibration.  Keep both visual
    # hypotheses as deterministic anonymous tracks instead of merging an
    # ambiguous direct/swap pair into identical marginal maps.
    direct = _anonymous_track_direct(archive, selected, device=device)
    update = core.update(
        core.assign_identity(measurement, direct_assignment=direct)
    )
    source_index = (
        _source_tensor(row_source, selected, device=device)
        if row_source is not None
        else None
    )
    history = empty_raw_history(len(selected), device=device)
    recent_deltas: torch.Tensor | None = None
    recent_errors: torch.Tensor | None = None
    for step in range(int(transition)):
        commands = _tensor(restore_companion_joint_units(raw_commands[:, step]), device)
        path = _tensor(restore_companion_joint_units(raw_paths[:, step]), device)
        prediction = _predict(
            core,
            update.grounded,
            q,
            commands,
            recent_deltas,
            recent_errors,
            source_index=source_index,
        )
        if collect_history:
            history = append_raw_history(
                history,
                commands,
                path,
                pool_scene_tokens(measurement.scene_tokens, measurement.view_valid),
            )
        endpoint = np.asarray(archive.row_endpoint_index[selected, step + 1], dtype=np.int64)
        measurement = source.measure(core, endpoint, device)
        update = core.update(
            core.assign_identity(measurement, prediction.prior_log_belief),
            prediction,
            path,
        )
        recent_deltas, recent_errors = _completed_readback_feedback(
            commands, path
        )
        q = path[:, -1]
    return measurement.scene_tokens, update, q, history


def _fish_commands_degrees(pair: Any) -> np.ndarray:
    """Return the same projected command chunk for GDM history and BC target."""

    value = np.asarray(pair.target_commands, dtype=np.float32) * 100.0
    if value.ndim != 3 or value.shape[1:] != (COMMAND_STEPS, STATE_DIM):
        raise V177TrainingError("fish projected commands must be [B,5,6]")
    if not bool(np.isfinite(value).all()):
        raise V177TrainingError("fish projected commands must be finite")
    return value


def _fish_actor_context(
    core: V177Core,
    identity: V177GeometryIdentityProvider,
    fish: V181FishArchive,
    source: V177FishMeasurementSource,
    ordinals: np.ndarray,
    *,
    device: torch.device,
    partition: str = "train",
    return_diagnostics: bool = False,
    vocabulary: V177SourceVocabulary | None = None,
    analytic_enabled: bool = True,
    analytic_only: bool = False,
    collect_history: bool = False,
):
    """Recursive causal Actor context on the calibrated fish rig.

    ``analytic_only`` replaces the GDM prior with the analytic FK displacement
    at a fixed process scale; it is an evaluation control, never a training
    path.  The returned tuple always ends with the raw history and, when
    requested, a diagnostics mapping.
    """

    allow_held = partition == "held"
    pair = fish.pair_batch(ordinals, partition=partition, allow_held=allow_held)
    if not bool((pair.transition == pair.transition[0]).all()):
        raise V177TrainingError("fish Actor batches must be transition-homogeneous")
    horizon = int(pair.transition[0])
    base_ordinal = np.asarray(ordinals, dtype=np.int64) - horizon
    first = fish.pair_batch(
        base_ordinal, partition=partition, allow_held=allow_held
    )
    measurement = source.measure(core, first.source_frame, device)
    q = _tensor(first.current_q * 100.0, device)
    update = _fish_h0_update(core, identity, measurement, q)
    source_index = _fish_source_tensor(vocabulary, len(q), device=device)
    history = empty_raw_history(len(q), device=device)
    recent_deltas: torch.Tensor | None = None
    recent_errors: torch.Tensor | None = None
    if horizon == 0:
        result = (pair, measurement.scene_tokens, update, q, history)
        return (*result, {"prediction": None, "measured_path": None}) if return_diagnostics else result

    prediction = None
    path = None
    for step in range(horizon):
        transition_pair = fish.pair_batch(
            base_ordinal + step,
            partition=partition,
            allow_held=allow_held,
        )
        expected_q = np.asarray(
            fish.states_degrees[transition_pair.target_frame], dtype=np.float32
        )
        realized_q = np.asarray(
            transition_pair.measured_q_path[:, -1] * 100.0, dtype=np.float32
        )
        mismatch = float(np.max(np.abs(realized_q - expected_q)))
        if mismatch > 2.0e-4:
            raise V177TrainingError(f"fish endpoint/q clock shifted by {mismatch:.6g}")
        commands = _tensor(_fish_commands_degrees(transition_pair), device)
        path = _tensor(transition_pair.measured_q_path * 100.0, device)
        analytic = _fish_analytic(core, identity, q, commands, enabled=analytic_enabled or analytic_only)
        prediction = _predict(
            core,
            update.grounded,
            q,
            commands,
            recent_deltas,
            recent_errors,
            source_index=source_index,
            analytic=analytic,
        )
        if analytic_only:
            prediction = _analytic_only_prediction(core, update.grounded, prediction, analytic)
        if collect_history:
            history = append_raw_history(
                history,
                commands,
                path,
                pool_scene_tokens(measurement.scene_tokens, measurement.view_valid),
            )
        measurement = source.measure(core, transition_pair.target_frame, device)
        assigned = core.assign_identity(measurement, prediction.prior_log_belief)
        update = core.update(assigned, prediction, path)
        recent_deltas, recent_errors = _completed_readback_feedback(
            commands, path
        )
        q = path[:, -1]
    # The target endpoint reached by the recurrent unroll is the source of the
    # requested Actor pair.  This assertion prevents a silent horizon shift.
    if not bool(np.array_equal(transition_pair.target_frame, pair.source_frame)):
        raise V177TrainingError("fish recurrent belief clock is shifted")
    result = (pair, measurement.scene_tokens, update, q, history)
    diagnostics = {"prediction": prediction, "measured_path": path}
    return (*result, diagnostics) if return_diagnostics else result


ANALYTIC_ONLY_SIGMA_PX = 4.0


def _analytic_only_prediction(
    core: V177Core,
    grounded: V177GroundedState,
    prediction: Any,
    analytic: tuple[torch.Tensor, torch.Tensor] | None,
):
    """Evaluation control: transport by the analytic displacement alone.

    Views without an analytic displacement keep the learned prior so the
    control degrades gracefully instead of producing a fake stationary prior.
    """

    if analytic is None:
        return prediction
    shift, valid = analytic
    learned_shift = prediction.transport_shift_px
    valid_tracks = valid[:, :, None, None].to(torch.bool)
    control_shift = torch.where(valid_tracks, shift.to(learned_shift), learned_shift)
    sigma = torch.where(
        valid_tracks,
        learned_shift.new_full((), float(ANALYTIC_ONLY_SIGMA_PX)),
        prediction.transport_sigma_px,
    )
    prior = propagate_belief_v177(
        grounded.log_posterior,
        control_shift,
        sigma,
        uniform_mix=core.config.transport_uniform_mix,
        blur_radius=core.config.blur_radius,
        mass_preserving=bool(core.config.mass_preserving_blur),  # type: ignore[attr-defined]
    )
    return type(prediction)(
        prior_log_belief=prior,
        predicted_q_path=prediction.predicted_q_path,
        transport_shift_px=control_shift,
        transport_sigma_px=sigma,
        tokens_by_layer=prediction.tokens_by_layer,
    )


def train_actor(
    core: V177Core,
    archive: V181TemporalArchive,
    fish: V181FishArchive,
    archive_beliefs: V177CombinedBeliefSource,
    archive_features: V177CombinedFeatureSource,
    dense_source: V177DenseMeasurementSource,
    fish_source: V177FishMeasurementSource,
    identity: V177GeometryIdentityProvider,
    output_root: str | Path,
    *,
    device: torch.device,
    vocabulary: V177SourceVocabulary,
    actor_mode: str = "full",
    archive_effective_batch: int = 256,
    full_effective_batch: int = 64,
    microbatch: int = 64,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train generic visuomotor feedback first, then specialize on fish.

    Every TRAIN H0--H3 pair receives the full current scene, recursively
    accumulated G/E state and q.  Commands/readbacks never bypass G/E into the
    ``full`` Actor API.  ``actor_mode`` selects the matched baseline trained
    on the identical schedule; each mode has its own stage directory.
    """

    if archive_effective_batch % microbatch or full_effective_batch % microbatch:
        raise V177TrainingError("Actor effective batches must divide into microbatches")
    if actor_mode not in ACTOR_MODES:
        raise V177TrainingError(f"actor_mode must be one of {ACTOR_MODES}")
    _check_source_vocabulary(core, vocabulary)
    row_source = vocabulary.row_index(archive)
    collect_history = actor_mode == "raw_history"
    output = Path(output_root).expanduser().resolve()
    stage_dir = output / actor_stage_name(actor_mode)
    latest_path = stage_dir / "latest.pt"
    final_path = stage_dir / "final.pt"
    core.actor.set_mode(actor_mode)
    if final_path.is_file():
        finished = load_component(final_path, core, component="actor", device=device)
        core.actor.set_mode(actor_mode)
        return dict(finished.get("metrics", {}))
    payload = _restore_latest(latest_path, core, component="actor", device=device)
    resume_step = int(payload.get("step", 0)) if payload is not None else 0
    core.to(device)
    core.measurement.eval()
    core.grounding_encoder.eval()
    core.lewm.eval()

    all_rows = _optimizer_rows(archive, archive.rows_for_partition("train"))
    dense_rows = _optimizer_rows(
        archive,
        np.intersect1d(dense_source.archive_rows(archive), all_rows, assume_unique=False),
    )
    archive_schedule: list[tuple[int, np.ndarray]] = []
    for pass_index in range(ACTOR_ARCHIVE_PASSES):
        for horizon in range(TRANSITIONS):
            if smoke:
                rows = _localized_archive_rows(
                    archive,
                    all_rows,
                    horizon=horizon,
                    batch_size=archive_effective_batch,
                    seed=SEED + 700 + pass_index * 10 + horizon,
                )
            else:
                current_endpoint = np.asarray(
                    archive.row_endpoint_index[all_rows, horizon], dtype=np.int64
                )
                rows = _localized_feature_rows(
                    all_rows,
                    current_endpoint,
                    archive_features,
                    seed=SEED + 700 + pass_index * 10 + horizon,
                )
            archive_schedule.append((horizon, rows))
    dense_schedule: list[tuple[int, np.ndarray]] = []
    for pass_index in range(ACTOR_DENSE_PASSES):
        for horizon in range(TRANSITIONS):
            if smoke:
                rows = _localized_archive_rows(
                    archive,
                    dense_rows,
                    horizon=horizon,
                    batch_size=full_effective_batch,
                    seed=SEED + 800 + pass_index * 10 + horizon,
                )
            else:
                current_endpoint = np.asarray(
                    archive.row_endpoint_index[dense_rows, horizon], dtype=np.int64
                )
                rows = _localized_feature_rows(
                    dense_rows,
                    current_endpoint,
                    dense_source.features,
                    seed=SEED + 800 + pass_index * 10 + horizon,
                )
            dense_schedule.append((horizon, rows))
    fish_steps = ACTOR_FISH_UPDATES
    if smoke:
        archive_schedule = [(0, all_rows[:archive_effective_batch])]
        dense_schedule = [(0, dense_rows[:full_effective_batch])]
        fish_steps = 2
    archive_steps = sum(math.ceil(len(rows) / archive_effective_batch) for _, rows in archive_schedule)
    dense_steps = sum(math.ceil(len(rows) / full_effective_batch) for _, rows in dense_schedule)
    total = archive_steps + dense_steps + fish_steps
    global_step = 0
    started = time.monotonic()
    last = float("nan")
    phase_losses: dict[str, Any] = (
        dict(payload.get("metrics", {})) if payload is not None else {}
    )
    phase_losses["feature_schedule"] = (
        "smoke_endpoint_block"
        if smoke
        else "deterministic_cache_page_local"
    )
    phase_losses["microbatch"] = int(microbatch)
    phase_losses["archive_effective_batch"] = int(archive_effective_batch)
    phase_losses["dense_fish_effective_batch"] = int(full_effective_batch)
    phase_losses["actor_mode"] = str(actor_mode)

    def optimize_microbatches(
        optimizer: torch.optim.Optimizer,
        batches: Sequence[
            tuple[
                torch.Tensor,
                V177GroundedState,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                V177RawHistory,
            ]
        ],
    ) -> float:
        optimizer.zero_grad(set_to_none=True)
        value = torch.zeros((), device=device)
        examples = sum(int(len(scene)) for scene, *_ in batches)
        if examples <= 0:
            raise V177TrainingError("Actor optimizer received an empty batch")
        for scene, grounded, innovation, q, target, history in batches:
            with _autocast(device):
                output_value = core.act(scene, grounded, innovation, q, history=history)
                loss = action_chunk_l1(output_value.commands, target, core.actor.step_cap)
            weight = float(len(scene)) / float(examples)
            (loss * weight).backward()
            value = value + loss.detach() * weight
        return float(value.cpu())

    # The innovation encoder only matters for the ``full`` Actor; the baselines
    # keep it frozen so their parameter budget is the shared decoder alone.
    trainable = list(core.actor.parameters())
    if actor_mode == "full":
        trainable += list(core.innovation_encoder.parameters())
    full_parameters = _set_trainable(core, trainable)
    optimizer = torch.optim.AdamW(full_parameters, lr=ACTOR_ARCHIVE_LR)
    if payload is not None and payload.get("cursor", {}).get("phase") == "archive" and isinstance(payload.get("optimizer"), Mapping):
        optimizer.load_state_dict(payload["optimizer"])
    for horizon, rows in archive_schedule:
        for selected in _batches(rows, archive_effective_batch):
            if global_step < resume_step:
                global_step += 1
                continue
            batches = []
            for micro_rows in _batches(selected, microbatch):
                ordinal = micro_rows * TRANSITIONS + int(horizon)
                _, _, _, _, target_np = _archive_actor_batch(archive, ordinal)
                with _autocast(device):
                    scene, update, q, history = _archive_cached_actor_context(
                        core,
                        archive,
                        archive_beliefs,
                        archive_features,
                        micro_rows,
                        horizon,
                        device=device,
                        row_source=row_source,
                        collect_history=collect_history,
                    )
                batches.append(
                    (
                        scene,
                        update.grounded,
                        update.innovation.innovation,
                        q,
                        _tensor(target_np, device),
                        history,
                    )
                )
            last = optimize_microbatches(optimizer, batches)
            torch.nn.utils.clip_grad_norm_(full_parameters, GRADIENT_CLIP)
            optimizer.step()
            global_step += 1
            phase_losses["archive_full_action_l1"] = last
            _progress("actor-archive-full", global_step, total, last, started)
            if global_step % CHECKPOINT_EVERY == 0 or global_step == archive_steps:
                save_component(
                    latest_path,
                    core,
                    component="actor",
                    step=global_step,
                    metrics=phase_losses,
                    optimizer=optimizer,
                    cursor={"phase": "archive", "horizon": horizon},
                )

    optimizer = torch.optim.AdamW(full_parameters, lr=ACTOR_DENSE_LR)
    if payload is not None and payload.get("cursor", {}).get("phase") == "dense" and isinstance(payload.get("optimizer"), Mapping):
        optimizer.load_state_dict(payload["optimizer"])
    for horizon, rows in dense_schedule:
        for selected in _batches(rows, full_effective_batch):
            if global_step < resume_step:
                global_step += 1
                continue
            batches = []
            for micro_rows in _batches(selected, microbatch):
                ordinal = micro_rows * TRANSITIONS + int(horizon)
                _, _, _, _, target_np = _archive_actor_batch(archive, ordinal)
                with _autocast(device):
                    scene, update, q, history = _archive_actor_context(
                        core,
                        archive,
                        dense_source,
                        micro_rows,
                        horizon,
                        device=device,
                        row_source=row_source,
                        collect_history=collect_history,
                    )
                batches.append(
                    (
                        scene,
                        update.grounded,
                        update.innovation.innovation,
                        q,
                        _tensor(target_np, device),
                        history,
                    )
                )
            last = optimize_microbatches(optimizer, batches)
            torch.nn.utils.clip_grad_norm_(full_parameters, GRADIENT_CLIP)
            optimizer.step()
            global_step += 1
            phase_losses["dense_real_rgb_action_l1"] = last
            _progress("actor-dense-full", global_step, total, last, started)
            if global_step % CHECKPOINT_EVERY == 0 or global_step == archive_steps + dense_steps:
                save_component(
                    latest_path,
                    core,
                    component="actor",
                    step=global_step,
                    metrics=phase_losses,
                    optimizer=optimizer,
                    cursor={"phase": "dense", "horizon": horizon},
                )

    optimizer = torch.optim.AdamW(full_parameters, lr=ACTOR_FISH_LR)
    if payload is not None and payload.get("cursor", {}).get("phase") == "fish" and isinstance(payload.get("optimizer"), Mapping):
        optimizer.load_state_dict(payload["optimizer"])
    train_sequence_count = len(fish.sequences_for_partition("train"))
    if train_sequence_count <= 0:
        raise V177TrainingError("fish training partition has no causal sequence")
    fish_schedule: list[np.ndarray] = []
    for fish_step in range(fish_steps):
        transition = fish_step % TRANSITIONS
        generator = np.random.default_rng(SEED + 1000 + fish_step)
        # ``pair_batch`` consumes partition-local ordinals, not the archive's
        # global sequence indices returned by ``sequences_for_partition``.
        sequences = generator.integers(
            0,
            train_sequence_count,
            size=full_effective_batch,
            dtype=np.int64,
        )
        ordinals = sequences * TRANSITIONS + transition
        fish_schedule.append(ordinals.copy())
        if global_step < resume_step:
            global_step += 1
            continue
        micro_values = []
        for micro in _batches(ordinals, microbatch):
            with _autocast(device):
                pair, scene, update, q, history = _fish_actor_context(
                    core,
                    identity,
                    fish,
                    fish_source,
                    micro,
                    device=device,
                    vocabulary=vocabulary,
                    collect_history=collect_history,
                )
            micro_values.append(
                (
                    scene,
                    update.grounded,
                    update.innovation.innovation,
                    q,
                    _tensor(_fish_commands_degrees(pair), device),
                    history,
                )
            )
        last = optimize_microbatches(optimizer, micro_values)
        torch.nn.utils.clip_grad_norm_(full_parameters, GRADIENT_CLIP)
        optimizer.step()
        global_step += 1
        phase_losses["fish_action_l1"] = last
        _progress("actor-fish", global_step, total, last, started)
        if global_step % CHECKPOINT_EVERY == 0 or fish_step + 1 == fish_steps:
            save_component(
                latest_path, core, component="actor", step=global_step,
                metrics=phase_losses, optimizer=optimizer,
                cursor={"phase": "fish", "fish_step": fish_step + 1},
            )
    horizon_exposures = {
        horizon: sum(
            len(rows) for current, rows in archive_schedule if current == horizon
        )
        for horizon in range(TRANSITIONS)
    }
    horizon_unique = {
        horizon: len(
            np.unique(
                np.concatenate(
                    [
                        rows
                        for current, rows in archive_schedule
                        if current == horizon
                    ]
                )
            )
        )
        if any(current == horizon for current, _ in archive_schedule)
        else 0
        for horizon in range(TRANSITIONS)
    }
    covered_row_ids = (
        np.unique(np.concatenate([rows for _, rows in archive_schedule]))
        if archive_schedule
        else np.empty(0, dtype=np.int64)
    )
    covered_rows = len(covered_row_ids)
    archive_current_endpoints = (
        np.concatenate(
            [
                np.asarray(
                    archive.row_endpoint_index[rows, horizon],
                    dtype=np.int64,
                )
                for horizon, rows in archive_schedule
            ]
        )
        if archive_schedule
        else np.empty(0, dtype=np.int64)
    )
    dense_horizon_exposures = {
        horizon: sum(
            len(rows) for current, rows in dense_schedule if current == horizon
        )
        for horizon in range(TRANSITIONS)
    }
    dense_horizon_unique = {
        horizon: len(
            np.unique(
                np.concatenate(
                    [
                        rows
                        for current, rows in dense_schedule
                        if current == horizon
                    ]
                )
            )
        )
        if any(current == horizon for current, _ in dense_schedule)
        else 0
        for horizon in range(TRANSITIONS)
    }
    dense_current_endpoints = (
        np.concatenate(
            [
                np.asarray(
                    archive.row_endpoint_index[rows, horizon],
                    dtype=np.int64,
                )
                for horizon, rows in dense_schedule
            ]
        )
        if dense_schedule
        else np.empty(0, dtype=np.int64)
    )
    fish_flat = (
        np.concatenate(fish_schedule)
        if fish_schedule
        else np.empty(0, dtype=np.int64)
    )
    train_rows = int(len(all_rows))
    complete_archive = (
        not smoke
        and all(
            horizon_unique[horizon] == train_rows
            for horizon in range(TRANSITIONS)
        )
        and all(
            horizon_exposures[horizon] == train_rows * ACTOR_ARCHIVE_PASSES
            for horizon in range(TRANSITIONS)
        )
    )
    actor_context = {
        "full": [
            "current_scene",
            "dynamics_grounding",
            "posterior_pooled_appearance",
            "innovation",
            "current_q",
        ],
        "memoryless": ["current_scene", "current_q"],
        "raw_history": [
            "current_scene",
            "current_q",
            "pooled_past_scenes",
            "past_commands",
            "past_readback_paths",
        ],
    }[actor_mode]
    metrics = {
        **phase_losses,
        "steps": int(global_step),
        "actor_mode": str(actor_mode),
        "innovation_encoder_trained": actor_mode == "full",
        "fish_updates": int(fish_steps),
        "fish_train_episodes": [0, 1, 3],
        "archive_rows": int(archive.rows),
        "archive_train_rows": train_rows,
        "archive_dev_rows_excluded": int((np.asarray(archive.split) == 1).sum()),
        "archive_partition": "train_only",
        "source_conditioning": {
            "vocabulary_size": int(vocabulary.size),
            "fish_index": int(vocabulary.fish_index),
            "dropout_in_actor_stage": 0.0,
        },
        "run_kind": "smoke" if smoke else "full",
        "archive_contract_complete": bool(complete_archive),
        "archive_row_coverage_fraction": float(covered_rows / max(1, train_rows)),
        "archive_endpoint_coverage": int(
            len(np.unique(archive_current_endpoints))
        ),
        "archive_source_episode_coverage": int(
            len(
                np.unique(
                    np.asarray(archive.endpoint_episode_index)[
                        archive_current_endpoints
                    ]
                )
            )
        ),
        "archive_unique_pairs": int(sum(horizon_unique.values())),
        "archive_complete_passes": int(ACTOR_ARCHIVE_PASSES if complete_archive else 0),
        "archive_pair_exposures": int(sum(horizon_exposures.values())),
        "archive_horizon_exposures": {
            f"H{horizon}": int(horizon_exposures[horizon])
            for horizon in range(TRANSITIONS)
        },
        "dense_rows": int(len(dense_rows)),
        "dense_unique_pairs": int(sum(dense_horizon_unique.values())),
        "dense_pair_exposures": int(sum(dense_horizon_exposures.values())),
        "dense_endpoint_coverage": int(
            len(np.unique(dense_current_endpoints))
        ),
        "dense_horizon_exposures": {
            f"H{horizon}": int(dense_horizon_exposures[horizon])
            for horizon in range(TRANSITIONS)
        },
        "fish_pair_population": int(train_sequence_count * TRANSITIONS),
        "fish_pair_exposures": int(len(fish_flat)),
        "fish_unique_pairs": int(len(np.unique(fish_flat))),
        "fish_pair_coverage_fraction": float(
            len(np.unique(fish_flat))
            / max(1, train_sequence_count * TRANSITIONS)
        ),
        "fish_unique_sequences": int(
            len(np.unique(fish_flat // TRANSITIONS))
        ),
        "fish_sequence_coverage_fraction": float(
            len(np.unique(fish_flat // TRANSITIONS))
            / max(1, train_sequence_count)
        ),
        "fish_horizon_exposures": {
            f"H{horizon}": int(
                np.count_nonzero(fish_flat % TRANSITIONS == horizon)
            )
            for horizon in range(TRANSITIONS)
        },
        "raw_actor_history_input": actor_mode == "raw_history",
        "actor_context": actor_context,
        "lewm_recursive_actor_horizons": [1, 2, 3],
        "archive_task_conditioning": "current_dual_view_scene_only",
        "feature_schedule": phase_losses["feature_schedule"],
    }
    core.requires_grad_(False)
    save_component(final_path, core, component="actor", step=global_step, metrics=metrics)
    # Preserve every matched Actor run in the aggregate metrics.  The full
    # model retains the historical ``actor`` key; baseline modes use their
    # stage-directory names instead of overwriting the preceding result.
    _merge_metrics(output / "train-metrics.json", actor_stage_name(actor_mode), metrics)
    return metrics


def _belief_mean_px(log_belief: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(log_belief.float(), dim=-1)[..., :SPATIAL_CELLS]
    mass = probability.sum(-1, keepdim=True).clamp_min(1.0e-12)
    axis = torch.arange(GRID_SIZE, device=probability.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    center = torch.stack((xx, yy), dim=-1).reshape(SPATIAL_CELLS, 2) * 4.0 + 1.5
    return torch.einsum("...c,cd->...d", probability / mass, center)


def _exchangeable_soft_target_ce_per_view(
    predicted_log: torch.Tensor,
    target_probability: torch.Tensor,
    view_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact per-view quantity reduced by ``exchangeable_pair_nll``."""

    predicted = torch.log_softmax(predicted_log.float(), dim=-1)
    target = target_probability.float().to(predicted)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    cost = -torch.einsum("bvac,bvsc->bvas", target, predicted)
    same = cost[:, :, 0, 0] + cost[:, :, 1, 1]
    swapped = cost[:, :, 0, 1] + cost[:, :, 1, 0]
    per_view = -torch.logsumexp(torch.stack((-same, -swapped), dim=-1), dim=-1)
    duplicate = (target[:, :, 0] - target[:, :, 1]).abs().amax(-1) <= 1.0e-7
    per_view = per_view + duplicate.to(per_view) * math.log(2.0)
    return per_view, view_valid.bool().to(per_view.device)


def _unordered_epe_per_view(
    predicted_log: torch.Tensor,
    target_px: torch.Tensor,
    target_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted = _belief_mean_px(predicted_log)
    active = target_valid.all(-1)
    direct = 0.5 * (
        torch.linalg.vector_norm(predicted[:, :, 0] - target_px[:, :, 0], dim=-1)
        + torch.linalg.vector_norm(predicted[:, :, 1] - target_px[:, :, 1], dim=-1)
    )
    swapped = 0.5 * (
        torch.linalg.vector_norm(predicted[:, :, 0] - target_px[:, :, 1], dim=-1)
        + torch.linalg.vector_norm(predicted[:, :, 1] - target_px[:, :, 0], dim=-1)
    )
    span = torch.linalg.vector_norm(
        predicted[:, :, 0] - predicted[:, :, 1], dim=-1
    )
    return torch.minimum(direct, swapped), span, active


def _state_path_loss_per_example(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: float = 4.0,
) -> torch.Tensor:
    residual = (
        predicted[:, 1:].float() - target[:, 1:].float()
    ) / float(scale)
    return F.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        reduction="none",
    ).mean(dim=(1, 2))


def _servo_lag_path(
    current_q: torch.Tensor, commands: torch.Tensor
) -> torch.Tensor:
    """One-command-clock lag baseline with the current state held at q1."""

    return torch.cat(
        (current_q[:, None], current_q[:, None], commands[:, :-1]), dim=1
    )


def _action_loss_per_example(
    commands: torch.Tensor,
    target: torch.Tensor,
    step_cap: torch.Tensor,
) -> torch.Tensor:
    return (
        (commands.float() - target.float()).abs()
        / step_cap.float()[None, None]
    ).mean(dim=(1, 2))


def _accumulate_metric(
    store: dict[str, dict[str, float | int]],
    name: str,
    values: torch.Tensor,
    active: torch.Tensor | None = None,
) -> None:
    flattened = values.detach().float().reshape(-1)
    if active is not None:
        flattened = flattened[active.detach().bool().reshape(-1)]
    entry = store.setdefault(name, {"sum": 0.0, "count": 0})
    entry["sum"] = float(entry["sum"]) + float(flattened.sum().cpu())
    entry["count"] = int(entry["count"]) + int(flattened.numel())


def _metric_summary(
    store: Mapping[str, Mapping[str, float | int]],
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for name, entry in store.items():
        total = float(entry["sum"])
        count = int(entry["count"])
        result[name] = {
            "mean": total / count if count else None,
            "sum": total,
            "count": count,
        }
    return result


def _command_deltas(
    current_q: torch.Tensor, commands: torch.Tensor
) -> torch.Tensor:
    previous = torch.cat((current_q[:, None], commands[:, :-1]), dim=1)
    return commands - previous


def _reanchor_command_deltas(
    current_q: torch.Tensor, deltas: torch.Tensor
) -> torch.Tensor:
    return current_q[:, None] + torch.cumsum(deltas, dim=1)


def _shuffled_reanchored_commands(
    current_q: torch.Tensor,
    commands: torch.Tensor,
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> torch.Tensor:
    deltas = _command_deltas(current_q, commands)
    shuffled = (
        torch.roll(deltas, shifts=1, dims=0)
        if len(deltas) > 1
        else torch.flip(deltas, dims=(1,))
    )
    desired = _reanchor_command_deltas(current_q, shuffled)
    cap = step_cap.to(desired)
    lower = action_lower.to(desired)
    upper = action_upper.to(desired)
    projected: list[torch.Tensor] = []
    previous = current_q
    for step in range(COMMAND_STEPS):
        increment = (desired[:, step] - previous).clamp(-cap, cap)
        previous = (previous + increment).clamp(lower, upper)
        projected.append(previous)
    return torch.stack(projected, dim=1)


def _evaluation_batches(values: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Partition without a singleton tail so within-horizon shuffles are defined."""

    selected = np.asarray(values, dtype=np.int64)
    size = int(batch_size)
    if len(selected) > size and len(selected) % size == 1:
        boundary = len(selected) - (size + 1)
        yield from _batches(selected[:boundary], size)
        yield selected[boundary : boundary + size - 1]
        yield selected[boundary + size - 1 :]
        return
    yield from _batches(selected, size)


def _same_horizon_innovation_shuffle(innovation: torch.Tensor) -> torch.Tensor:
    """Deterministically donor-shuffle a transition-homogeneous batch."""

    return (
        torch.roll(innovation, shifts=1, dims=0)
        if len(innovation) > 1
        else torch.flip(innovation, dims=(2,))
    )


def _innovation_interventions(
    core: V177Core,
    update: V177FilterUpdate,
    prediction: Any | None,
    measured_path: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Decompose belief and q residuals while keeping current G/scene/q fixed."""

    null = core.innovation_encoder.null(
        len(update.grounded.log_posterior),
        update.grounded.view_valid,
        device=update.grounded.dynamics_grounding.device,
        dtype=update.grounded.dynamics_grounding.dtype,
    )
    if prediction is None:
        # H0 has no completed prediction.  Every feedback intervention
        # therefore collapses to the learned no-prediction token.
        return {
            "full": update.innovation.innovation,
            "belief_only": null,
            "q_only": null,
            "neither": null,
            "no_prediction_null": null,
        }
    if measured_path is None:
        raise V177TrainingError("predicted innovation requires a measured q path")
    posterior = update.grounded.log_posterior
    prior = prediction.prior_log_belief
    valid = update.grounded.view_valid
    predicted_q = prediction.predicted_q_path

    def encode(
        observed_belief: torch.Tensor,
        observed_q: torch.Tensor,
    ) -> torch.Tensor:
        return core.innovation_encoder(
            compute_innovation(
                observed_belief,
                prior,
                valid,
                observed_q,
                predicted_q,
            )
        )

    return {
        "full": update.innovation.innovation,
        "belief_only": encode(posterior, predicted_q),
        "q_only": encode(prior, measured_path),
        # This is prediction-present zero residual, not the learned H0/null
        # embedding.  It controls for the encoder's present marker.
        "neither": encode(prior, predicted_q),
        "no_prediction_null": null,
    }


def load_actor_modes(
    core: V177Core,
    output_root: str | Path,
    *,
    device: torch.device,
) -> dict[str, V177Actor]:
    """Instantiate one Actor module per completed mode checkpoint.

    Actor checkpoints hold the whole core state; only ``actor.*`` keys are
    copied so every mode is evaluated against the same frozen measurement,
    GDM and innovation encoder currently loaded in ``core``.
    """

    output = Path(output_root).expanduser().resolve()
    actors: dict[str, V177Actor] = {}
    for mode in ACTOR_MODES:
        path = output / actor_stage_name(mode) / "final.pt"
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("component") != "actor":
            raise V177TrainingError(f"{path} is not an Actor checkpoint")
        if payload.get("version") != VERSION or payload.get("architecture") != ARCHITECTURE_CONTRACT:
            raise V177TrainingError(f"{path} has a different V17.7 contract")
        recorded_mode = str(payload.get("metrics", {}).get("actor_mode", mode))
        if recorded_mode != mode:
            raise V177TrainingError(
                f"{path} records actor_mode={recorded_mode!r} but lives in the {mode!r} stage"
            )
        actor = V177Actor(core.config).to(device)  # type: ignore[arg-type]
        prefix = "actor."
        weights = {
            key[len(prefix):]: value
            for key, value in payload["model"].items()
            if key.startswith(prefix)
        }
        actor.load_state_dict(weights, strict=True)
        actor.set_mode(mode)
        actor.eval()
        actors[mode] = actor
    if not actors:
        raise V177TrainingError("no completed Actor mode checkpoint was found")
    return actors


def _blank_history(history: V177RawHistory) -> V177RawHistory:
    return V177RawHistory(
        commands=history.commands,
        paths=history.paths,
        scene_pooled=history.scene_pooled,
        valid=torch.zeros_like(history.valid),
    )


def _fish_dynamics_probe(
    core: V177Core,
    fish: V181FishArchive,
    fish_source: V177FishMeasurementSource,
    identity: V177GeometryIdentityProvider,
    vocabulary: V177SourceVocabulary,
    base_ordinals: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Matched GDM transport controls on exposed fish sequences.

    Each variant runs its own self-consistent filter from the same H0 so the
    comparison is between complete filters, not between one-step priors that
    share a factual context.
    """

    variants = ("factual", "learned_only", "analytic_only", "persistence")
    store: dict[str, dict[str, dict[str, float | int]]] = {name: {} for name in variants}
    by_horizon: dict[int, dict[str, dict[str, dict[str, float | int]]]] = {
        horizon: {name: {} for name in variants} for horizon in range(TRANSITIONS)
    }
    for base in _batches(np.asarray(base_ordinals, dtype=np.int64), batch_size):
        first = fish.pair_batch(base, partition="held", allow_held=True)
        measurement = fish_source.measure(core, first.source_frame, device)
        q0 = _tensor(first.current_q * 100.0, device)
        h0 = _fish_h0_update(core, identity, measurement, q0)
        states = {name: h0.grounded for name in variants}
        source = _fish_source_tensor(vocabulary, len(q0), device=device)
        q = q0
        recent_deltas: torch.Tensor | None = None
        recent_errors: torch.Tensor | None = None
        for step in range(TRANSITIONS):
            pair = fish.pair_batch(base + step, partition="held", allow_held=True)
            commands = _tensor(_fish_commands_degrees(pair), device)
            path = _tensor(pair.measured_q_path * 100.0, device)
            next_measurement = fish_source.measure(core, pair.target_frame, device)
            target = next_measurement.anchor_hypothesis_log_likelihood.exp()
            analytic = _fish_analytic(core, identity, q, commands, enabled=True)
            priors: dict[str, torch.Tensor] = {}
            for name in variants:
                grounded = states[name]
                if name == "persistence":
                    priors[name] = grounded.log_posterior
                    continue
                prediction = _predict(
                    core,
                    grounded,
                    q,
                    commands,
                    recent_deltas,
                    recent_errors,
                    source_index=source,
                    analytic=None if name == "learned_only" else analytic,
                )
                if name == "analytic_only":
                    prediction = _analytic_only_prediction(core, grounded, prediction, analytic)
                priors[name] = prediction.prior_log_belief
            for name, prior in priors.items():
                value, active = _exchangeable_soft_target_ce_per_view(
                    prior, target, next_measurement.view_valid
                )
                _accumulate_metric(store[name], "exchangeable_soft_target_cross_entropy", value, active)
                _accumulate_metric(
                    by_horizon[step][name], "exchangeable_soft_target_cross_entropy", value, active
                )
                ordered_target, _ = identity_or_track_assignment(
                    next_measurement.anchor_hypothesis_log_likelihood,
                    prior,
                    next_measurement.view_valid,
                )
                posterior = fuse_measurement_prior(prior, ordered_target, next_measurement.view_valid)
                states[name] = _grounded_from_ordered(core, posterior, next_measurement.view_valid)
            recent_deltas, recent_errors = _completed_readback_feedback(commands, path)
            q = path[:, -1]
    return {
        "sequences": int(len(base_ordinals)),
        "analytic_only_sigma_px": float(ANALYTIC_ONLY_SIGMA_PX),
        "self_consistent_filters": True,
        "metrics": {name: _metric_summary(values) for name, values in store.items()},
        "by_horizon": {
            f"H{horizon}_to_H{horizon + 1}": {
                name: _metric_summary(values) for name, values in per_variant.items()
            }
            for horizon, per_variant in by_horizon.items()
        },
    }


@torch.inference_mode()
def evaluate_v177(
    core: V177Core,
    cad: V175CADArchive,
    cad_features: Any,
    archive: V181TemporalArchive,
    temporal_beliefs: V177CombinedBeliefSource,
    fish: V181FishArchive,
    fish_source: V177FishMeasurementSource,
    identity: V177GeometryIdentityProvider,
    output_root: str | Path,
    *,
    device: torch.device,
    vocabulary: V177SourceVocabulary,
    archive_features: V177CombinedFeatureSource | None = None,
    batch_size: int = 32,
    probe_rows: int = 2048,
    smoke: bool = False,
    authorize_exposed_development: bool = False,
) -> dict[str, Any]:
    """Run fixed descriptive CAD, archive-DEV dynamics, fish and Actor-mode evaluation.

    Nothing here selects a checkpoint.  Fish episode 2 is exposed development
    data and the 48K archive DEV rows were opened by earlier versions; both are
    reported as development probes, never as strict held evidence.
    """

    if authorize_exposed_development is not True:
        raise V177TrainingError(
            "fish episode 2 is exposed development data; evaluation "
            "requires explicit internal authorization"
        )
    _check_source_vocabulary(core, vocabulary)
    core.eval().to(device)
    output = Path(output_root).expanduser().resolve()
    evaluation = output / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    actors = load_actor_modes(core, output, device=device)
    if "full" in actors:
        # Keep the core's own Actor identical to the ``full`` checkpoint so
        # ``core.act`` and ``act_with(actors["full"], ...)`` agree.
        core.actor.load_state_dict(actors["full"].state_dict(), strict=True)
        core.actor.set_mode("full")

    cad_rows = np.asarray(cad.validation_rows, dtype=np.int64)
    if smoke:
        cad_rows = cad_rows[:batch_size]
    cad_metrics: dict[str, dict[str, float | int]] = {}
    for rows in _batches(cad_rows, batch_size):
        raw_rgb = np.asarray(cad.rgb[rows], dtype=np.uint8)
        rgb = torch.from_numpy(np.ascontiguousarray(raw_rgb[:, ::-1])).to(device).permute(0, 1, 4, 2, 3)
        dino, feature_valid = cad_features.get(rows)
        valid = torch.as_tensor(feature_valid, device=device, dtype=torch.bool)
        dino_tensor = torch.as_tensor(dino, device=device).float()
        measurement = core.measure(dino_tensor, valid, rgb)
        dino_only = core.measure(dino_tensor, valid, None)
        target_px = torch.from_numpy(
            np.ascontiguousarray(np.asarray(cad.keypoints_px[rows], dtype=np.float32)[:, ::-1, 4:6])
        ).to(device)
        target_valid = torch.from_numpy(
            np.ascontiguousarray(np.asarray(cad.valid[rows], dtype=np.bool_)[:, ::-1, 4:6])
        ).to(device)
        target = _bilinear_point_belief(target_px, target_valid)
        for prefix, measured in (("fused", measurement), ("dino_only", dino_only)):
            log_likelihood = measured.anchor_hypothesis_log_likelihood
            ce, ce_active = _exchangeable_soft_target_ce_per_view(log_likelihood, target, valid)
            epe, span, epe_active = _unordered_epe_per_view(log_likelihood, target_px, target_valid)
            _accumulate_metric(
                cad_metrics, f"{prefix}_exchangeable_soft_target_cross_entropy", ce, ce_active
            )
            _accumulate_metric(cad_metrics, f"{prefix}_unordered_epe_px", epe, epe_active)
            if prefix == "fused":
                _accumulate_metric(cad_metrics, "predicted_span_px", span, epe_active)
            coverage = credible_coverage(log_likelihood, target_px, target_valid)
            coverage_active = coverage["active"] & valid[:, :, None]
            for key in ("coverage_at_50", "coverage_at_90", "needed_mass", "absent_probability"):
                _accumulate_metric(cad_metrics, f"{prefix}_{key}", coverage[key], coverage_active)
            overlap = hypothesis_overlap(log_likelihood)
            _accumulate_metric(cad_metrics, f"{prefix}_hypothesis_overlap", overlap, valid)

    # Archive DEV dynamics probe, stratified by whether the rig was seen in TRAIN.
    train_rows = archive.rows_for_partition("train")
    dev_rows = archive.rows_for_partition("dev")
    seen = vocabulary.seen_in_rows(archive, train_rows)
    row_source = vocabulary.row_index(archive)
    row_source_eval = np.where(seen[row_source], row_source, SOURCE_UNKNOWN).astype(np.int64)
    sample_count = min(int(probe_rows), len(dev_rows))
    sampled_rows = dev_rows[np.linspace(0, len(dev_rows) - 1, sample_count, dtype=np.int64)]
    if smoke:
        sampled_rows = sampled_rows[:batch_size]
    strata = ("all", "source_seen", "source_unseen")
    dynamics: dict[str, dict[str, dict[str, float | int]]] = {name: {} for name in strata}
    dynamics_by_horizon: dict[int, dict[str, dict[str, float | int]]] = {
        horizon: {} for horizon in range(TRANSITIONS)
    }
    conditions_order = (
        "factual",
        "factual_unknown_source",
        "persistence",
        "hold_current_q",
        "shuffled_action_deltas",
    )
    for rows in _batches(sampled_rows, batch_size):
        endpoint_np, commands_np, paths_np = _archive_chain_arrays(archive, rows)
        source_values = _cache_tensors(temporal_beliefs, endpoint_np[:, 0], device)
        grounded = _grounded_from_raw(
            core,
            *source_values,
            track_direct=_anonymous_track_direct(archive, rows, device=device),
        )
        row_seen = torch.as_tensor(seen[row_source[rows]], device=device, dtype=torch.bool)
        source_index = _source_tensor(row_source_eval, rows, device=device)
        unknown_index = torch.full_like(source_index, SOURCE_UNKNOWN)
        commands_all = _tensor(commands_np, device)
        paths_all = _tensor(paths_np, device)
        current = paths_all[:, 0, 0]
        recent_deltas: torch.Tensor | None = None
        recent_errors: torch.Tensor | None = None
        for transition in range(TRANSITIONS):
            commands = commands_all[:, transition]
            path = paths_all[:, transition]
            target_values = _cache_tensors(temporal_beliefs, endpoint_np[:, transition + 1], device)
            factual = _predict(
                core, grounded, current, commands, recent_deltas, recent_errors,
                source_index=source_index,
            )
            unknown = _predict(
                core, grounded, current, commands, recent_deltas, recent_errors,
                source_index=unknown_index,
            )
            hold_commands = current[:, None].expand(-1, COMMAND_STEPS, -1)
            hold = _predict(
                core, grounded, current, hold_commands, recent_deltas, recent_errors,
                source_index=source_index,
            )
            shuffled_commands = _shuffled_reanchored_commands(
                current, commands, core.actor.step_cap, core.actor.action_lower, core.actor.action_upper,
            )
            shuffled = _predict(
                core, grounded, current, shuffled_commands, recent_deltas, recent_errors,
                source_index=source_index,
            )
            target_probability = target_values[0].exp()
            conditions = {
                "factual": factual.prior_log_belief,
                "factual_unknown_source": unknown.prior_log_belief,
                "persistence": grounded.log_posterior,
                "hold_current_q": hold.prior_log_belief,
                "shuffled_action_deltas": shuffled.prior_log_belief,
            }
            for name in conditions_order:
                value, active = _exchangeable_soft_target_ce_per_view(
                    conditions[name], target_probability, target_values[1]
                )
                _accumulate_metric(dynamics["all"], name, value, active)
                _accumulate_metric(dynamics["source_seen"], name, value, active & row_seen[:, None])
                _accumulate_metric(dynamics["source_unseen"], name, value, active & ~row_seen[:, None])
                _accumulate_metric(dynamics_by_horizon[transition], name, value, active)
            q_conditions = {
                "q_factual": factual.predicted_q_path,
                "q_direct_command": torch.cat((current[:, None], commands), dim=1),
                "q_servo_lag": _servo_lag_path(current, commands),
                "q_hold_current": current[:, None].expand(-1, PATH_STEPS, -1),
            }
            for name, predicted_q in q_conditions.items():
                value = _state_path_loss_per_example(predicted_q, path)
                _accumulate_metric(dynamics["all"], name, value)
                _accumulate_metric(dynamics["source_seen"], name, value, row_seen)
                _accumulate_metric(dynamics["source_unseen"], name, value, ~row_seen)
                _accumulate_metric(dynamics_by_horizon[transition], name, value)
            ordered_target, _ = identity_or_track_assignment(
                target_values[0], factual.prior_log_belief, target_values[1]
            )
            posterior = fuse_measurement_prior(
                factual.prior_log_belief, ordered_target, target_values[1]
            )
            grounded = _grounded_from_ordered(core, posterior, target_values[1])
            recent_deltas, recent_errors = _completed_readback_feedback(commands, path)
            current = path[:, -1]

    # Archive DEV Actor probe: every mode on identical contexts.
    archive_actor: dict[str, dict[str, dict[str, float | int]]] = {name: {} for name in strata}
    archive_actor_by_horizon: dict[int, dict[str, dict[str, float | int]]] = {
        horizon: {} for horizon in range(TRANSITIONS)
    }
    if archive_features is not None:
        actor_rows = sampled_rows[: max(batch_size, len(sampled_rows) // 4)]
        for horizon in range(TRANSITIONS):
            for rows in _batches(actor_rows, batch_size):
                ordinal = rows * TRANSITIONS + int(horizon)
                _, _, _, _, target_np = _archive_actor_batch(archive, ordinal)
                target = _tensor(target_np, device)
                scene, update, q, history = _archive_cached_actor_context(
                    core, archive, temporal_beliefs, archive_features, rows, horizon,
                    device=device, row_source=row_source_eval, collect_history=True,
                )
                row_seen = torch.as_tensor(seen[row_source[rows]], device=device, dtype=torch.bool)
                for mode, actor in actors.items():
                    output_value = act_with(
                        actor, scene, update.grounded, update.innovation.innovation, q, history=history
                    )
                    value = _action_loss_per_example(output_value.commands, target, actor.step_cap)
                    _accumulate_metric(archive_actor["all"], mode, value)
                    _accumulate_metric(archive_actor["source_seen"], mode, value, row_seen)
                    _accumulate_metric(archive_actor["source_unseen"], mode, value, ~row_seen)
                    _accumulate_metric(archive_actor_by_horizon[horizon], mode, value)

    _append_exposed_development_access(evaluation, event="evaluation_started", smoke=smoke)
    held_ordinals = fish.pair_ordinals("held")
    if smoke:
        held_ordinals = held_ordinals[:batch_size]
    fish_dynamics = _fish_dynamics_probe(
        core, fish, fish_source, identity, vocabulary,
        held_ordinals[held_ordinals % TRANSITIONS == 0],
        device=device, batch_size=batch_size,
    )
    actor_loss: dict[str, dict[str, float | int]] = {}
    actor_feedback_loss: dict[str, dict[str, float | int]] = {}
    actor_by_horizon: dict[int, dict[str, dict[str, float | int]]] = {
        horizon: {} for horizon in range(TRANSITIONS)
    }
    for transition in range(TRANSITIONS):
        selected = held_ordinals[held_ordinals % TRANSITIONS == transition]
        for ordinal in _evaluation_batches(selected, batch_size):
            pair, scene, update, q, history, diagnostics = _fish_actor_context(
                core, identity, fish, fish_source, ordinal,
                device=device, partition="held", return_diagnostics=True,
                vocabulary=vocabulary, collect_history=True,
            )
            target = _tensor(_fish_commands_degrees(pair), device)
            outputs: dict[str, torch.Tensor] = {}
            if "full" in actors:
                full_actor = actors["full"]
                innovations = _innovation_interventions(
                    core, update, diagnostics["prediction"], diagnostics["measured_path"]
                )
                innovations["innovation_shuffled_same_horizon"] = _same_horizon_innovation_shuffle(
                    innovations["full"]
                )
                for name, innovation in innovations.items():
                    outputs[f"full/{name}"] = act_with(
                        full_actor, scene, update.grounded, innovation, q, history=history
                    ).commands
                measurement = fish_source.measure(core, pair.source_frame, device)
                current = _actor_update_from_measurements(core, identity, measurement, q)
                outputs["full/current_only"] = act_with(
                    full_actor, scene, current.grounded, current.innovation.innovation, q,
                    history=empty_raw_history(len(q), device=device),
                ).commands
            if "memoryless" in actors:
                outputs["memoryless/full"] = act_with(
                    actors["memoryless"], scene, update.grounded, update.innovation.innovation, q,
                    history=history,
                ).commands
            if "raw_history" in actors:
                raw_actor = actors["raw_history"]
                outputs["raw_history/full"] = act_with(
                    raw_actor, scene, update.grounded, update.innovation.innovation, q, history=history
                ).commands
                outputs["raw_history/history_blank"] = act_with(
                    raw_actor, scene, update.grounded, update.innovation.innovation, q,
                    history=_blank_history(history),
                ).commands
            for name, commands in outputs.items():
                value = _action_loss_per_example(commands, target, core.actor.step_cap)
                _accumulate_metric(actor_loss, name, value)
                _accumulate_metric(actor_by_horizon[transition], name, value)
                if transition > 0:
                    _accumulate_metric(actor_feedback_loss, name, value)

    result = {
        "version": VERSION,
        "actor_modes_evaluated": sorted(actors.keys()),
        "cad_validation": {
            "rows": int(len(cad_rows)),
            "coverage_definition": "hpd_mass_on_visible_conditional_map",
            "metrics": _metric_summary(cad_metrics),
        },
        "archive_dev_dynamics_probe": {
            "role": "previously_exposed_archive_development_rows",
            "rows": int(len(sampled_rows)),
            "dev_rows_total": int(len(dev_rows)),
            "recursive_factual_context": True,
            "target": "stopped_raw_next_measurement",
            "unseen_source_index_policy": "mapped_to_unknown",
            "metrics": {name: _metric_summary(values) for name, values in dynamics.items()},
            "by_horizon": {
                f"H{horizon}_to_H{horizon + 1}": _metric_summary(values)
                for horizon, values in dynamics_by_horizon.items()
            },
        },
        "archive_dev_actor_probe": {
            "role": "previously_exposed_archive_development_rows",
            "evaluated": archive_features is not None,
            "metrics": {name: _metric_summary(values) for name, values in archive_actor.items()},
            "by_horizon": {
                f"H{horizon}": _metric_summary(values)
                for horizon, values in archive_actor_by_horizon.items()
            },
        },
        "fish_episode2_dynamics_probe": fish_dynamics,
        "fish_episode2_exposed_dev": {
            "role": "descriptive_only_previously_exposed",
            "same_horizon_matched": True,
            "innovation_shuffle": "within_horizon_batch_roll",
            "feedback_conditions_apply_from_horizon": 1,
            "neither_definition": "prediction_present_belief_and_q_residuals_zero",
            "h0_feedback_conditions_collapse_to_no_prediction_null": True,
            "condition_naming": "<actor_mode>/<intervention>",
            "aggregate": _metric_summary(actor_loss),
            "feedback_horizons_H1_H3": _metric_summary(actor_feedback_loss),
            "by_horizon": {
                f"H{horizon}": _metric_summary(values)
                for horizon, values in actor_by_horizon.items()
            },
        },
        "selection_role": "descriptive_only_no_checkpoint_selection",
        "strict_held_evidence": False,
        "formal_real_robot_evidence": "requires_new_session_or_robot_panel",
    }
    _write_json(evaluation / "metrics.json", result)
    _append_exposed_development_access(evaluation, event="evaluation_completed", smoke=smoke)
    _merge_metrics(output / "train-metrics.json", "evaluation", result)
    return result


__all__ = [
    "ABSENT_INDEX",
    "ACTOR_ARCHIVE_PASSES",
    "ACTOR_DENSE_PASSES",
    "ACTOR_FISH_UPDATES",
    "ARCHITECTURE_CONTRACT",
    "BELIEF_CELLS",
    "CAD_PASSES",
    "CHECKPOINT_EVERY",
    "GRID_SIZE",
    "LEWM_FISH_ANALYTIC_DROPOUT",
    "LEWM_FISH_UPDATES",
    "REAL_REFIT_UPDATES",
    "SEED",
    "VERSION",
    "V177BeliefCache",
    "V177CombinedBeliefSource",
    "V177CombinedFeatureSource",
    "V177CanonicalRealRGB",
    "V177DenseMeasurementSource",
    "V177DeviceBeliefBank",
    "V177FishMeasurementSource",
    "V177GeometryIdentityProvider",
    "V177SourceVocabulary",
    "V177TrainingError",
    "actor_stage_name",
    "assignment_matrix_from_direct",
    "configure_cuda",
    "dense_endpoint_map",
    "evaluate_v177",
    "import_measurement_weights",
    "load_actor_modes",
    "load_component",
    "require_physical_gpu1",
    "save_component",
    "seed_v177",
    "prepare_belief_cache",
    "train_actor",
    "train_lewm",
    "train_measurement",
]
