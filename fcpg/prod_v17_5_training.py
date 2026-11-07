"""Data, objectives, caches, and fixed training loops for structured V17.5.

CAD is the only large absolute-pixel supervision source and therefore opens
state, camera geometry and all six ordered anchors.  The 20K cache provides
real appearance only.  The 320K archive provides action/readback dynamics
only; its compact ``/100`` storage convention is restored exactly once at the
loader boundary.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import mmap
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fcpg.prod_v17_5 import (
    ANCHOR_DIM,
    ANCHORS,
    COMMAND_STEPS,
    COMPANION_STORAGE_SCALE,
    HISTORY_CHUNKS,
    PATH_STEPS,
    STATE_DIM,
    VIEW_ORDER,
    VIEWS,
    VIT_DIM,
    VIT_PATCHES,
    V175Config,
    V175Controller,
    V175FCPGMemory,
    V175GroundedAnchorFCPG,
    V175GroundedLeWM,
    V175GroundedMeasurement,
    V175Innovation,
    V175Memory,
    compute_v175_innovation,
    ordered_anchor_smooth_l1,
    restore_companion_joint_units,
    unordered_tip_reprojection_loss,
)
from fcpg.prod_v18_1_data import (
    TRANSITIONS,
    V181FeatureCache,
    V181FishArchive,
    V181TemporalActorBatch,
    V181TemporalArchive,
)
from fcpg.prod_v18_1_training import (
    V181DeviceFeatureBank,
    extract_dinov2_tokens,
    load_frozen_dinov2,
    prepare_fish_dino_cache,
    prepare_temporal_dino_cache,
)
from fcpg.so101_camera_calibration import (
    ProjectionFit,
    predict_projection,
    valid_projection_mask,
)


SEED = 17_500
FCPG_DISTILL_LR = 1.0e-4
FCPG_CAD_LR = 1.0e-4
FCPG_REAL_LR = 3.0e-5
LEWM_LR = 1.0e-4
ACTOR_ARCHIVE_LR = 1.0e-4
ACTOR_FISH_LR = 3.0e-5
FCPG_CAD_PASSES = 4
FCPG_REAL_UPDATES = 600
ACTOR_ARCHIVE_PASSES = 4
ACTOR_FISH_UPDATES = 4_000
CHECKPOINT_EVERY = 500
GRADIENT_CLIP = 1.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CAD_TIP_INDICES = (4, 5)
ARCHITECTURE_CONTRACT = "morphology-conditioned-kinematic-v2"


class V175TrainingError(RuntimeError):
    """Raised before an invalid V17.5 experiment mutates an artifact."""


def _progress(
    stage: str,
    step: int,
    total: int,
    loss: float,
    started: float,
) -> None:
    if step != 1 and step % 50 and step != total:
        return
    elapsed = max(1.0e-6, time.monotonic() - started)
    rate = step / elapsed
    eta = (total - step) / rate if rate > 0 else float("inf")
    gpu_memory = (
        int(torch.cuda.memory_allocated() / 1024**2)
        if torch.cuda.is_available()
        else 0
    )
    print(
        json.dumps(
            {
                "stage": stage,
                "step": int(step),
                "total": int(total),
                "loss": float(loss),
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(eta, 1),
                "gpu_allocated_mib": gpu_memory,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def seed_v175(seed: int = SEED) -> None:
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
        raise V175TrainingError("unknown exposed-development access event")
    evaluation.mkdir(parents=True, exist_ok=True)
    record = {
        "unix_time_ns": time.time_ns(),
        "event": event,
        "version": "17.5",
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
                raise V175TrainingError(
                    "failed to append exposed-development access record"
                )
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalize_episode2_role(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Remove stale machine-readable claims that episode 2 remained held."""

    result = dict(metrics)
    result.pop("held_episode", None)
    result.update(
        {
            "exposed_development_episode": 2,
            "episode_2_role": "exposed_development_previously_opened",
            "strict_held_evidence": False,
        }
    )
    return result


def _ordinary_payload(
    module: nn.Module,
    *,
    component: str,
    step: int,
    metrics: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    cursor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "version": "17.5",
        "architecture": ARCHITECTURE_CONTRACT,
        "component": str(component),
        "seed": SEED,
        "step": int(step),
        "config": asdict(V175Config()),
        "model": module.state_dict(),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        result["optimizer"] = optimizer.state_dict()
        result["rng"] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
    if scheduler is not None:
        result["scheduler"] = scheduler.state_dict()
    if cursor is not None:
        result["cursor"] = dict(cursor)
    return result


def _restore_latest(
    path: Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    component: str,
) -> tuple[int, dict[str, Any]] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != "17.5"
        or payload.get("architecture") != ARCHITECTURE_CONTRACT
    ):
        raise V175TrainingError("latest checkpoint is not V17.5")
    if payload.get("component") != component:
        return None
    if not all(isinstance(payload.get(name), dict) for name in ("model", "optimizer", "scheduler", "cursor", "rng")):
        raise V175TrainingError("latest checkpoint lacks resumable state")
    module.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    rng = payload["rng"]
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    if torch.cuda.is_available() and rng.get("cuda"):
        torch.cuda.set_rng_state_all(rng["cuda"])
    return int(payload.get("step", 0)), dict(payload["cursor"])


def _restore_latest_without_scheduler(
    path: Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    component: str,
) -> tuple[int, dict[str, Any], dict[str, Any]] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != "17.5"
        or payload.get("architecture") != ARCHITECTURE_CONTRACT
    ):
        raise V175TrainingError("latest checkpoint is not V17.5")
    if payload.get("component") != component:
        return None
    if not all(isinstance(payload.get(name), dict) for name in ("model", "optimizer", "cursor", "rng", "metrics")):
        raise V175TrainingError("latest checkpoint lacks resumable state")
    module.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    rng = payload["rng"]
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    if torch.cuda.is_available() and rng.get("cuda"):
        torch.cuda.set_rng_state_all(rng["cuda"])
    return (
        int(payload.get("step", 0)),
        dict(payload["cursor"]),
        dict(payload["metrics"]),
    )


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
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        _ordinary_payload(
            module,
            component=component,
            step=step,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            cursor=cursor,
        ),
        temporary,
    )
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
        raise V175TrainingError(f"V17.5 component is absent: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != "17.5"
        or payload.get("architecture") != ARCHITECTURE_CONTRACT
    ):
        raise V175TrainingError("artifact is not a V17.5 checkpoint")
    if payload.get("component") != component or not isinstance(payload.get("model"), dict):
        raise V175TrainingError("V17.5 checkpoint component differs")
    module.load_state_dict(payload["model"], strict=True)
    module.to(device)
    return payload


def cosine_schedule(step: int, *, total: int, warmup: int) -> float:
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def batches(values: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(values), int(batch_size)):
        value = values[start : start + int(batch_size)]
        if len(value):
            yield value


def _flush_memmaps(*arrays: np.memmap) -> None:
    """Persist cache progress without retaining the entire file in host RSS."""

    for array in arrays:
        array.flush()
    advice = getattr(mmap, "MADV_DONTNEED", None)
    if advice is None:
        return
    for array in arrays:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None and hasattr(mapping, "madvise"):
            mapping.madvise(advice)


def _contiguous_selection(indices: np.ndarray) -> slice | np.ndarray:
    """Use a basic slice for a contiguous cache chunk, preserving hole resumes."""

    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or not len(selected):
        return selected
    start = int(selected[0])
    if np.array_equal(selected, np.arange(start, start + len(selected))):
        return slice(start, start + len(selected))
    return selected


class V175AllRealRGB:
    """All 20K cached rows and five endpoints, without opening SAM files."""

    def __init__(self, rgb_root: str | Path, sidecar_root: str | Path) -> None:
        self.rgb_root = Path(rgb_root).expanduser().resolve()
        self.sidecar_root = Path(sidecar_root).expanduser().resolve()
        if not self.rgb_root.is_dir() or not self.sidecar_root.is_dir():
            raise V175TrainingError("20K RGB or sidecar root is absent")
        self.rgb = np.load(self.rgb_root / "rgb.npy", mmap_mode="r", allow_pickle=False)
        self.sidecar_row = np.load(
            self.rgb_root / "sidecar_row.npy", mmap_mode="r", allow_pickle=False
        )
        if self.rgb.shape != (20_000, 5, VIEWS, 3, 224, 224) or self.rgb.dtype != np.uint8:
            raise V175TrainingError("20K RGB shape/dtype drifted")
        if self.sidecar_row.shape != (20_000,) or self.sidecar_row.dtype != np.int32:
            raise V175TrainingError("20K sidecar-row identity drifted")
        sidecar_chain = np.load(
            self.sidecar_root / "chain_id.npy", mmap_mode="r", allow_pickle=False
        )
        mapped = np.asarray(self.sidecar_row, dtype=np.int64)
        if bool(((mapped < 0) | (mapped >= len(sidecar_chain))).any()):
            raise V175TrainingError("20K sidecar mapping is out of range")
        view_valid = np.load(
            self.sidecar_root / "view_valid.npy", mmap_mode="r", allow_pickle=False
        )
        if view_valid.shape != (len(sidecar_chain), 5, VIEWS) or view_valid.dtype != np.bool_:
            raise V175TrainingError("20K view-valid shape/dtype drifted")
        self.view_valid = view_valid
        self.endpoint_rows = np.arange(100_000, dtype=np.int64)

    def batch(self, endpoint_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = np.asarray(endpoint_rows, dtype=np.int64)
        if selected.ndim != 1 or len(selected) == 0 or bool(
            ((selected < 0) | (selected >= 100_000)).any()
        ):
            raise V175TrainingError("20K endpoint batch leaves its population")
        row = selected // 5
        endpoint = selected % 5
        rgb = np.ascontiguousarray(self.rgb[row, endpoint])
        valid = np.ascontiguousarray(
            self.view_valid[np.asarray(self.sidecar_row[row], dtype=np.int64), endpoint]
        )
        return rgb, valid


class V175CADArchive:
    """CAD reader that refuses to discard state, camera or structural labels."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise V175TrainingError(f"CAD root is absent: {self.root}")
        self.rgb = np.load(self.root / "rgb.npy", mmap_mode="r", allow_pickle=False)
        self.keypoints_px = np.load(
            self.root / "keypoints_px.npy", mmap_mode="r", allow_pickle=False
        )
        self.state = np.load(
            self.root / "state.npy", mmap_mode="r", allow_pickle=False
        )
        self.keypoints_base_xyz = np.load(
            self.root / "keypoints_base_xyz.npy", mmap_mode="r", allow_pickle=False
        )
        self.camera_from_reference = np.load(
            self.root / "camera_from_reference.npy", mmap_mode="r", allow_pickle=False
        )
        self.intrinsic = np.load(
            self.root / "intrinsic.npy", mmap_mode="r", allow_pickle=False
        )
        self.valid = np.load(self.root / "valid.npy", mmap_mode="r", allow_pickle=False)
        self.split = np.load(self.root / "split.npy", mmap_mode="r", allow_pickle=False)
        rows = len(self.rgb)
        if rows != 200_000 or self.rgb.shape != (rows, VIEWS, 224, 224, 3):
            raise V175TrainingError("CAD RGB population/shape drifted")
        if self.rgb.dtype != np.uint8:
            raise V175TrainingError("CAD RGB must be uint8")
        if self.keypoints_px.shape != (rows, VIEWS, 6, 2) or self.keypoints_px.dtype != np.float32:
            raise V175TrainingError("CAD keypoint shape/dtype drifted")
        if self.state.shape != (rows, STATE_DIM) or self.state.dtype != np.float32:
            raise V175TrainingError("CAD state shape/dtype drifted")
        if self.keypoints_base_xyz.shape != (rows, ANCHORS, 3) or self.keypoints_base_xyz.dtype != np.float32:
            raise V175TrainingError("CAD base-keypoint shape/dtype drifted")
        if self.camera_from_reference.shape != (rows, VIEWS, 4, 4) or self.camera_from_reference.dtype != np.float32:
            raise V175TrainingError("CAD camera transform shape/dtype drifted")
        if self.intrinsic.shape != (rows, VIEWS, 3, 3) or self.intrinsic.dtype != np.float32:
            raise V175TrainingError("CAD intrinsic shape/dtype drifted")
        if self.valid.shape != (rows, VIEWS, 6) or self.valid.dtype != np.bool_:
            raise V175TrainingError("CAD keypoint-valid shape/dtype drifted")
        if self.split.shape != (rows,) or not set(np.unique(self.split)).issubset({0, 1}):
            raise V175TrainingError("CAD split contract drifted")
        self.train_rows = np.flatnonzero(self.split == 0).astype(np.int64)
        self.validation_rows = np.flatnonzero(self.split == 1).astype(np.int64)
        if len(self.train_rows) != 190_000 or len(self.validation_rows) != 10_000:
            raise V175TrainingError("CAD 190K/10K population drifted")


@torch.inference_mode()
def prepare_cad_dino_cache(
    cad: V175CADArchive,
    output_root: str | Path,
    backbone: nn.Module,
    *,
    device: torch.device,
    row_batch: int = 512,
    dino_batch: int = 512,
) -> Path:
    """Extract frozen CAD DINO tokens once for all four spatial passes."""

    if int(row_batch) <= 0 or int(dino_batch) <= 0:
        raise V175TrainingError("CAD DINO cache batches must be positive")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = len(cad.rgb)
    endpoint = np.arange(rows, dtype=np.int64)
    valid = np.ones((rows, VIEWS), dtype=np.bool_)
    identity_path = output / "endpoint_index.npy"
    valid_path = output / "view_valid.npy"
    token_path = output / "tokens.npy"
    done_path = output / "done.npy"
    if identity_path.is_file() or valid_path.is_file():
        if not identity_path.is_file() or not valid_path.is_file():
            raise V175TrainingError("resumed CAD feature identities are incomplete")
        if not np.array_equal(
            np.load(identity_path, mmap_mode="r", allow_pickle=False), endpoint
        ):
            raise V175TrainingError("resumed CAD feature identities differ")
        if not np.array_equal(
            np.load(valid_path, mmap_mode="r", allow_pickle=False), valid
        ):
            raise V175TrainingError("resumed CAD feature validity differs")
    else:
        np.save(identity_path, endpoint, allow_pickle=False)
        np.save(valid_path, valid, allow_pickle=False)
    required_bytes = rows * VIEWS * VIT_PATCHES * VIT_DIM * 2
    if not token_path.is_file():
        available = shutil.disk_usage(output).free
        if available < required_bytes + 16 * 1024**3:
            raise V175TrainingError(
                "CAD DINO cache lacks disk workspace: "
                f"required={required_bytes} available={available}"
            )
    if token_path.is_file() or done_path.is_file():
        if not token_path.is_file() or not done_path.is_file():
            raise V175TrainingError("resumed CAD feature arrays are incomplete")
        tokens = np.lib.format.open_memmap(token_path, mode="r+")
        done = np.lib.format.open_memmap(done_path, mode="r+")
        if tokens.shape != (rows, VIEWS, VIT_PATCHES, VIT_DIM) or tokens.dtype != np.float16:
            raise V175TrainingError("resumed CAD token shape/dtype differs")
        if done.shape != (rows,) or done.dtype != np.bool_:
            raise V175TrainingError("resumed CAD completion shape/dtype differs")
    else:
        tokens = np.lib.format.open_memmap(
            token_path,
            mode="w+",
            dtype=np.float16,
            shape=(rows, VIEWS, VIT_PATCHES, VIT_DIM),
        )
        done = np.lib.format.open_memmap(
            done_path, mode="w+", dtype=np.bool_, shape=(rows,)
        )
        done[:] = False
        _flush_memmaps(tokens, done)

    pending = np.flatnonzero(~done)
    total = math.ceil(len(pending) / int(row_batch))
    started = time.monotonic()
    for step, selected in enumerate(batches(pending, row_batch), start=1):
        # Raw CAD is [column,wrist]; every maintained V17.5 cache is [wrist,column].
        selection = _contiguous_selection(selected)
        raw_rgb = np.asarray(cad.rgb[selection], dtype=np.uint8)
        rgb = np.ascontiguousarray(raw_rgb[:, ::-1])
        encoded = extract_dinov2_tokens(
            backbone,
            rgb.reshape(-1, 224, 224, 3),
            device=device,
            batch_size=int(dino_batch),
        ).reshape(len(selected), VIEWS, VIT_PATCHES, VIT_DIM)
        tokens[selection] = encoded
        # A row is durable only after its token payload.  Flush tokens before
        # publishing the corresponding completion bits so a resumed process
        # never trusts a partially persisted chunk.
        _flush_memmaps(tokens)
        done[selection] = True
        _flush_memmaps(done)
        _write_json(
            output / "config.json",
            {
                "version": "17.5",
                "kind": "cad",
                "backbone": "dinov2_vits14",
                "view_order": list(VIEW_ORDER),
                "patch_grid": [16, 16],
                "token_dim": VIT_DIM,
                "dtype": "float16",
                "rows": rows,
                "complete": bool(done.all()),
            },
        )
        _progress("cad-dino-cache", step, total, 0.0, started)
    if not bool(done.all()):
        raise V175TrainingError("CAD DINO cache finished incomplete")
    # Repair a stale progress file after a crash between the final array flush
    # and config replacement.
    _write_json(
        output / "config.json",
        {
            "version": "17.5",
            "kind": "cad",
            "backbone": "dinov2_vits14",
            "view_order": list(VIEW_ORDER),
            "patch_grid": [16, 16],
            "token_dim": VIT_DIM,
            "dtype": "float16",
            "rows": rows,
            "complete": True,
        },
    )
    return output


class V175TemporalEndpointStateProvider:
    """Resolve each archived endpoint to its strictly causal measured state.

    The provider restores the companion builder's ``/100`` storage transform
    exactly once.  It never opens legacy pixel measurements or pseudo labels.
    """

    def __init__(self, archive: V181TemporalArchive) -> None:
        self.archive = archive
        flat = np.asarray(archive.row_endpoint_index, dtype=np.int64).reshape(-1)
        first = np.unique(flat, return_index=True)[1]
        unique = flat[first]
        if len(unique) != archive.endpoint_count or not bool(
            np.array_equal(np.sort(unique), np.arange(archive.endpoint_count))
        ):
            raise V175TrainingError("320K endpoint clock does not cover its endpoint bank")
        self.endpoint_row = np.full(archive.endpoint_count, -1, dtype=np.int64)
        self.endpoint_horizon = np.full(archive.endpoint_count, -1, dtype=np.int8)
        self.endpoint_row[unique] = first // 5
        self.endpoint_horizon[unique] = first % 5

    def __call__(self, endpoint_indices: Any) -> dict[str, np.ndarray]:
        endpoint = np.asarray(endpoint_indices, dtype=np.int64)
        if endpoint.ndim != 1 or len(endpoint) == 0:
            raise V175TrainingError("temporal state provider expects nonempty [N] identities")
        if bool(((endpoint < 0) | (endpoint >= self.archive.endpoint_count)).any()):
            raise V175TrainingError("temporal endpoint identity leaves archive")
        row = self.endpoint_row[endpoint]
        horizon = self.endpoint_horizon[endpoint].astype(np.int64)
        if bool((row < 0).any()):
            raise V175TrainingError("temporal endpoint has no causal source row")
        paths = np.asarray(self.archive.measured_state_path[row], dtype=np.float32)
        commands = np.asarray(self.archive.completed_commands[row], dtype=np.float32)
        current = np.empty((len(endpoint), STATE_DIM), dtype=np.float32)
        history_commands = np.zeros(
            (len(endpoint), HISTORY_CHUNKS, COMMAND_STEPS, STATE_DIM), dtype=np.float32
        )
        history_paths = np.zeros(
            (len(endpoint), HISTORY_CHUNKS, PATH_STEPS, STATE_DIM), dtype=np.float32
        )
        history_valid = np.zeros((len(endpoint), HISTORY_CHUNKS), dtype=np.bool_)
        for value in range(5):
            selected = horizon == value
            if not bool(selected.any()):
                continue
            if value == 0:
                current[selected] = paths[selected, 0, 0]
            else:
                current[selected] = paths[selected, value - 1, -1]
                history_commands[selected, :value] = commands[selected, :value]
                history_paths[selected, :value] = paths[selected, :value]
                history_valid[selected, :value] = True
                # The endpoint following a completed transition must agree
                # with the last measured readback; otherwise H0--H4 is shifted.
                if value < 4:
                    mismatch = np.max(
                        np.abs(current[selected] - paths[selected, value, 0])
                    )
                    if float(mismatch) > 2.0e-4:
                        raise V175TrainingError(
                            f"320K endpoint/readback clock mismatch: {mismatch:.6g}"
                        )
        return {
            "current_q": restore_companion_joint_units(current),
            "completed_commands": restore_companion_joint_units(history_commands),
            "measured_paths": restore_companion_joint_units(history_paths),
            "history_valid": history_valid,
        }

    def current(self, endpoint_indices: Any) -> np.ndarray:
        """Fast current-q lookup for endpoint-indexed measurement caches."""

        endpoint = np.asarray(endpoint_indices, dtype=np.int64)
        if endpoint.ndim != 1 or len(endpoint) == 0 or bool(
            ((endpoint < 0) | (endpoint >= self.archive.endpoint_count)).any()
        ):
            raise V175TrainingError("temporal current-q lookup identities are invalid")
        row = self.endpoint_row[endpoint]
        horizon = self.endpoint_horizon[endpoint].astype(np.int64)
        raw = np.asarray(self.archive.measured_state_path[row], dtype=np.float32)
        current = np.empty((len(endpoint), STATE_DIM), dtype=np.float32)
        for value in range(5):
            selected = horizon == value
            if bool(selected.any()):
                current[selected] = (
                    raw[selected, 0, 0]
                    if value == 0
                    else raw[selected, value - 1, -1]
                )
        return restore_companion_joint_units(current)


class V175FishEndpointStateProvider:
    """Current measured fish state; chunk history remains an Actor/runtime state."""

    def __init__(self, archive: V181FishArchive) -> None:
        self.archive = archive

    def __call__(self, endpoint_indices: Any) -> dict[str, np.ndarray]:
        endpoint = np.asarray(endpoint_indices, dtype=np.int64)
        if endpoint.ndim != 1 or len(endpoint) == 0:
            raise V175TrainingError("fish state provider expects nonempty [N] identities")
        if bool(((endpoint < 0) | (endpoint >= self.archive.frames)).any()):
            raise V175TrainingError("fish endpoint identity leaves frame bank")
        batch = len(endpoint)
        return {
            "current_q": np.asarray(
                self.archive.states_degrees[endpoint], dtype=np.float32
            ),
            "completed_commands": np.zeros(
                (batch, HISTORY_CHUNKS, COMMAND_STEPS, STATE_DIM), dtype=np.float32
            ),
            "measured_paths": np.zeros(
                (batch, HISTORY_CHUNKS, PATH_STEPS, STATE_DIM), dtype=np.float32
            ),
            "history_valid": np.zeros((batch, HISTORY_CHUNKS), dtype=np.bool_),
        }

    def current(self, endpoint_indices: Any) -> np.ndarray:
        endpoint = np.asarray(endpoint_indices, dtype=np.int64)
        if endpoint.ndim != 1 or len(endpoint) == 0 or bool(
            ((endpoint < 0) | (endpoint >= self.archive.frames)).any()
        ):
            raise V175TrainingError("fish current-q lookup identities are invalid")
        return np.asarray(self.archive.states_degrees[endpoint], dtype=np.float32)


class V175AffineProjectionProvider:
    """Audited real-camera affine proposal plus a finite-difference q Jacobian."""

    def __init__(
        self,
        calibration_path: str | Path,
        fcpg: V175GroundedAnchorFCPG,
    ) -> None:
        path = Path(calibration_path).expanduser().resolve()
        if not path.is_file():
            raise V175TrainingError(f"real camera calibration is absent: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("geometry_protocol") != "letterbox_v1":
            raise V175TrainingError("real camera calibration must use letterbox_v1")
        fits: dict[str, ProjectionFit] = {}
        for name in VIEW_ORDER:
            row = payload.get("fits", {}).get(name)
            if not isinstance(row, dict):
                raise V175TrainingError(f"real calibration omits {name}")
            fits[name] = ProjectionFit(
                view=str(row["view"]),
                feature_names=tuple(map(str, row["feature_names"])),
                coefficients=np.asarray(row["coefficients"], dtype=np.float64),
                ridge=float(row["ridge"]),
            )
        self.fits = fits
        self.fcpg = fcpg
        self.path = path

    def _pixels(self, q: torch.Tensor) -> np.ndarray:
        reference = self.fcpg.kinematics.anchors_in_view_references(q)
        coordinates = reference.detach().float().cpu().numpy()
        state = q.detach().float().cpu().numpy()
        # Canonical reference order is wrist-link then base.
        wrist_local = coordinates[:, 0]
        base = coordinates[:, 1]
        projected = [
            predict_projection(self.fits[name], base, wrist_local, state[:, -1])
            for name in VIEW_ORDER
        ]
        return np.stack(projected, axis=1).astype(np.float32)

    def __call__(
        self,
        current_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.as_tensor(
            current_q,
            device=self.fcpg.kinematics.position_lower.device,
            dtype=torch.float32,
        )
        if q.ndim != 2 or q.shape[1] != STATE_DIM or not bool(torch.isfinite(q).all()):
            raise V175TrainingError("projection provider q must be finite [B,6]")
        nominal_np = self._pixels(q)
        lower = self.fcpg.kinematics.position_lower.to(q)
        upper = self.fcpg.kinematics.position_upper.to(q)
        epsilon = q.new_full((STATE_DIM,), 0.25)
        derivatives: list[np.ndarray] = []
        for joint in range(STATE_DIM):
            plus = q.clone()
            minus = q.clone()
            plus[:, joint] = torch.minimum(q[:, joint] + epsilon[joint], upper[joint])
            minus[:, joint] = torch.maximum(q[:, joint] - epsilon[joint], lower[joint])
            denominator = (plus[:, joint] - minus[:, joint]).clamp_min(1.0e-6)
            derivative = (
                self._pixels(plus) - self._pixels(minus)
            ) / denominator.detach().cpu().numpy()[:, None, None, None]
            derivatives.append(derivative)
        jacobian_np = np.stack(derivatives, axis=-1).astype(np.float32)
        valid_np = np.stack(
            [
                valid_projection_mask(nominal_np[:, view], image_size=(224, 224))
                for view in range(VIEWS)
            ],
            axis=1,
        )
        return (
            torch.from_numpy(nominal_np).to(q.device),
            torch.from_numpy(valid_np).to(q.device),
            torch.from_numpy(jacobian_np).to(q.device),
        )


class V175GroundedCache:
    """Compact posterior-pooled anchor states addressed by endpoint identity."""

    def __init__(self, root: str | Path, *, expected_kind: str | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        config_path = self.root / "config.json"
        if not config_path.is_file():
            raise V175TrainingError(f"grounded cache is absent: {self.root}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            config.get("version") != "17.5"
            or config.get("contract") != "structured-kinematic-v2"
            or tuple(config.get("view_order", ())) != VIEW_ORDER
        ):
            raise V175TrainingError("grounded cache contract differs")
        if expected_kind is not None and config.get("kind") != expected_kind:
            raise V175TrainingError("grounded cache kind differs")
        if not bool(config.get("complete")):
            raise V175TrainingError("grounded cache is incomplete")
        self.kind = str(config.get("kind"))
        self.spatial_mode = str(config.get("spatial_mode", ""))
        self.endpoint_index = np.load(
            self.root / "endpoint_index.npy", mmap_mode="r", allow_pickle=False
        )
        self.anchor_tokens = np.load(
            self.root / "anchor_tokens.npy", mmap_mode="r", allow_pickle=False
        )
        self.anchor_valid = np.load(
            self.root / "anchor_valid.npy", mmap_mode="r", allow_pickle=False
        )
        self.positions_px = np.load(
            self.root / "positions_px.npy", mmap_mode="r", allow_pickle=False
        )
        self.position_valid = np.load(
            self.root / "position_valid.npy", mmap_mode="r", allow_pickle=False
        )
        self.entropy = np.load(
            self.root / "entropy.npy", mmap_mode="r", allow_pickle=False
        )
        rows = len(self.endpoint_index)
        if self.endpoint_index.shape != (rows,) or self.endpoint_index.dtype != np.int64:
            raise V175TrainingError("grounded endpoint identity drifted")
        if rows == 0 or bool((np.diff(self.endpoint_index) <= 0).any()):
            raise V175TrainingError("grounded endpoints must be sorted unique")
        if self.anchor_tokens.shape != (rows, VIEWS, ANCHORS, ANCHOR_DIM):
            raise V175TrainingError("grounded anchor-token shape drifted")
        if self.anchor_tokens.dtype != np.float16:
            raise V175TrainingError("grounded anchor tokens must be float16")
        if self.anchor_valid.shape != (rows, VIEWS, ANCHORS) or self.anchor_valid.dtype != np.bool_:
            raise V175TrainingError("grounded anchor-valid shape/dtype drifted")
        if self.positions_px.shape != (rows, VIEWS, ANCHORS, 2):
            raise V175TrainingError("grounded position probe shape drifted")
        if (
            self.position_valid.shape != (rows, VIEWS, ANCHORS)
            or self.position_valid.dtype != np.bool_
        ):
            raise V175TrainingError("grounded position-valid shape/dtype drifted")
        if self.entropy.shape != (rows, VIEWS, ANCHORS):
            raise V175TrainingError("grounded entropy probe shape drifted")
        audit_rows = np.linspace(0, rows - 1, min(rows, 1_024)).round().astype(np.int64)
        if not bool(np.isfinite(np.asarray(self.anchor_tokens[audit_rows])).all()):
            raise V175TrainingError("grounded anchor tokens failed finite audit")
        if not bool(np.isfinite(np.asarray(self.positions_px[audit_rows])).all()):
            raise V175TrainingError("grounded position probes failed finite audit")
        if not bool(np.isfinite(np.asarray(self.entropy[audit_rows])).all()):
            raise V175TrainingError("grounded entropy probes failed finite audit")

    def rows_for(self, endpoint: Any) -> np.ndarray:
        requested = np.asarray(endpoint, dtype=np.int64)
        flat = requested.reshape(-1)
        positions = np.searchsorted(self.endpoint_index, flat)
        safe = np.minimum(positions, len(self.endpoint_index) - 1)
        if bool((positions >= len(self.endpoint_index)).any()) or not bool(
            np.array_equal(self.endpoint_index[safe], flat)
        ):
            raise V175TrainingError("endpoint is absent from grounded cache")
        return positions.reshape(requested.shape)

    def get_numpy(self, endpoint: Any) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(endpoint, dtype=np.int64)
        rows = self.rows_for(requested)
        return (
            np.asarray(self.anchor_tokens[rows], dtype=np.float32),
            np.asarray(self.anchor_valid[rows], dtype=np.bool_),
        )


class V175DeviceGroundedBank:
    """The complete compact grounded state fits on the 80 GiB training GPU."""

    def __init__(self, cache: V175GroundedCache, *, device: torch.device) -> None:
        if device.type != "cuda":
            raise V175TrainingError("full grounded bank requires CUDA")
        self.cache = cache
        self.device = device
        self.anchor_tokens = torch.from_numpy(
            np.array(cache.anchor_tokens, copy=True)
        ).to(device=device, non_blocking=False)
        self.anchor_valid = torch.from_numpy(
            np.array(cache.anchor_valid, copy=True)
        ).to(device=device, non_blocking=False)

    def get(self, endpoint: Any) -> tuple[torch.Tensor, torch.Tensor]:
        requested = np.asarray(endpoint, dtype=np.int64)
        rows = self.cache.rows_for(requested).reshape(-1)
        index = torch.from_numpy(rows).to(self.device)
        shape = requested.shape
        return (
            self.anchor_tokens[index].reshape(*shape, VIEWS, ANCHORS, ANCHOR_DIM),
            self.anchor_valid[index].reshape(*shape, VIEWS, ANCHORS),
        )

    def stats(self) -> dict[str, Any]:
        return {
            "kind": self.cache.kind,
            "spatial_mode": self.cache.spatial_mode,
            "rows": len(self.cache.endpoint_index),
            "allocated_bytes": int(
                self.anchor_tokens.numel() * self.anchor_tokens.element_size()
                + self.anchor_valid.numel() * self.anchor_valid.element_size()
            ),
        }


class V175CombinedGroundedBank:
    def __init__(
        self,
        train: V175DeviceGroundedBank,
        development: V175DeviceGroundedBank,
    ) -> None:
        self.train = train
        self.development = development

    def get(self, endpoint: Any) -> tuple[torch.Tensor, torch.Tensor]:
        requested = np.asarray(endpoint, dtype=np.int64)
        flat = requested.reshape(-1)
        result = torch.empty(
            len(flat), VIEWS, ANCHORS, ANCHOR_DIM,
            dtype=self.train.anchor_tokens.dtype, device=self.train.device,
        )
        valid = torch.empty(
            len(flat), VIEWS, ANCHORS,
            dtype=torch.bool, device=self.train.device,
        )
        train_position = np.searchsorted(self.train.cache.endpoint_index, flat)
        in_train = train_position < len(self.train.cache.endpoint_index)
        train_safe = np.minimum(train_position, len(self.train.cache.endpoint_index) - 1)
        in_train &= self.train.cache.endpoint_index[train_safe] == flat
        if bool(in_train.any()):
            token, mask = self.train.get(flat[in_train])
            destination = torch.from_numpy(np.flatnonzero(in_train)).to(self.train.device)
            result[destination] = token
            valid[destination] = mask
        if bool((~in_train).any()):
            token, mask = self.development.get(flat[~in_train])
            destination = torch.from_numpy(np.flatnonzero(~in_train)).to(self.train.device)
            result[destination] = token
            valid[destination] = mask
        return (
            result.reshape(*requested.shape, VIEWS, ANCHORS, ANCHOR_DIM),
            valid.reshape(*requested.shape, VIEWS, ANCHORS),
        )


class V175CombinedFeatureBank:
    """One 48 GiB page budget split over the two ordinary 320K partitions."""

    def __init__(
        self,
        train_cache: V181FeatureCache,
        development_cache: V181FeatureCache,
        *,
        device: torch.device,
        total_bytes: int = 48 * 1024**3,
    ) -> None:
        if int(total_bytes) < 2 * 1024**3:
            raise V175TrainingError("combined DINO page budget is too small")
        development_bytes = max(2 * 1024**3, int(total_bytes) // 6)
        self.train = V181DeviceFeatureBank(
            train_cache,
            device=device,
            max_bytes=int(total_bytes) - development_bytes,
        )
        self.development = V181DeviceFeatureBank(
            development_cache,
            device=device,
            max_bytes=development_bytes,
        )
        self.train_endpoint = np.asarray(train_cache.endpoint_index)
        self.device = device

    def get(self, endpoint: Any) -> tuple[torch.Tensor, torch.Tensor]:
        requested = np.asarray(endpoint, dtype=np.int64)
        flat = requested.reshape(-1)
        positions = np.searchsorted(self.train_endpoint, flat)
        safe = np.minimum(positions, len(self.train_endpoint) - 1)
        in_train = (positions < len(self.train_endpoint)) & (
            self.train_endpoint[safe] == flat
        )
        tokens = torch.empty(
            len(flat), VIEWS, VIT_PATCHES, VIT_DIM,
            dtype=torch.float16, device=self.device,
        )
        valid = torch.empty(
            len(flat), VIEWS, dtype=torch.bool, device=self.device
        )
        if bool(in_train.any()):
            source, mask = self.train.get(flat[in_train])
            destination = torch.from_numpy(np.flatnonzero(in_train)).to(self.device)
            tokens[destination] = source
            valid[destination] = mask
        if bool((~in_train).any()):
            source, mask = self.development.get(flat[~in_train])
            destination = torch.from_numpy(np.flatnonzero(~in_train)).to(self.device)
            tokens[destination] = source
            valid[destination] = mask
        return (
            tokens.reshape(*requested.shape, VIEWS, VIT_PATCHES, VIT_DIM),
            valid.reshape(*requested.shape, VIEWS),
        )

    def stats(self) -> dict[str, Any]:
        return {"train": self.train.stats(), "development": self.development.stats()}


@torch.inference_mode()
def extract_dino_tensor(
    backbone: nn.Module,
    rgb: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Extract full DINOv2 patch tokens without a CPU round-trip."""

    value = torch.as_tensor(rgb, device=device)
    if value.ndim != 5 or tuple(value.shape[1:3]) != (VIEWS, 3) or tuple(value.shape[-2:]) != (224, 224):
        raise V175TrainingError("DINO RGB must be [B,2,3,224,224]")
    batch = len(value)
    if value.dtype == torch.uint8:
        value = value.float().div_(255.0)
    else:
        value = value.float()
    flat = value.reshape(batch * VIEWS, 3, 224, 224)
    mean = flat.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = flat.new_tensor(IMAGENET_STD)[None, :, None, None]
    with _autocast(device):
        output = backbone.forward_features((flat - mean) / std)
    if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
        raise V175TrainingError("DINOv2 did not expose patch tokens")
    tokens = output["x_norm_patchtokens"]
    if tuple(tokens.shape) != (batch * VIEWS, VIT_PATCHES, VIT_DIM):
        raise V175TrainingError("DINOv2 patch-token shape drifted")
    return tokens.reshape(batch, VIEWS, VIT_PATCHES, VIT_DIM)


def feature_distillation_objective(
    fcpg: V175GroundedAnchorFCPG,
    rgb: torch.Tensor,
    dino_tokens: torch.Tensor,
    view_valid: torch.Tensor,
) -> torch.Tensor:
    fused, rgb_feature, target = fcpg.fused_features(rgb, dino_tokens)
    target = target.detach().to(rgb_feature)
    rgb_distance = 1.0 - F.cosine_similarity(rgb_feature, target, dim=2)
    fused_distance = 1.0 - F.cosine_similarity(fused, target, dim=2)
    per_cell = 0.5 * rgb_distance + 0.5 * fused_distance
    mask = view_valid[:, :, None, None].expand_as(per_cell)
    return (per_cell * mask).sum() / mask.sum().clamp_min(1)


def _matched_anchor_losses(
    predicted: torch.Tensor,
    target: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ordered part-token loss, independently balanced per view."""

    predicted = F.layer_norm(predicted.float(), (ANCHOR_DIM,))
    target = F.layer_norm(target.float(), (ANCHOR_DIM,))
    active = source_valid & target_valid
    magnitude = F.smooth_l1_loss(predicted, target, reduction="none").mean(-1)
    cosine = 1.0 - F.cosine_similarity(predicted, target, dim=-1)
    per_anchor = 0.5 * magnitude + 0.5 * cosine
    direct = (per_anchor * active).sum(-1) / active.sum(-1).clamp_min(1)
    per_view = direct
    view_active = source_valid.any(-1) & target_valid.any(-1)
    token_loss = (per_view * view_active).sum() / view_active.sum().clamp_min(1)
    return token_loss, direct.detach(), direct.detach()


@torch.no_grad()
def align_target_anchor_set(
    predicted: torch.Tensor,
    target: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ordered anchors need no self-referential prediction-based slot matching."""

    if predicted.shape != target.shape or source_valid.shape != target_valid.shape:
        raise V175TrainingError("ordered anchor alignment shapes differ")
    return (
        target,
        target_valid,
        torch.zeros(target.shape[:2], dtype=torch.bool, device=target.device),
    )


def lewm_objective(
    prediction: Any,
    target_anchor_tokens: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
    measured_q_path: torch.Tensor,
    *,
    step_cap: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    visual, direct, swapped = _matched_anchor_losses(
        prediction.predicted_anchor_tokens,
        target_anchor_tokens,
        source_valid,
        target_valid,
    )
    # ``step_cap`` belongs to the Actor module, which may still be resident on
    # CPU while the independently trained LeWM already lives on the requested
    # accelerator.  Treat it as immutable normalization data and place it next
    # to the prediction at the loss boundary.
    cap = step_cap.to(
        device=prediction.predicted_q_path.device,
        dtype=torch.float32,
    )
    normalized = (
        prediction.predicted_q_path.float()
        - measured_q_path.to(
            device=prediction.predicted_q_path.device,
            dtype=torch.float32,
        )
    ) / cap[None, None]
    state = F.smooth_l1_loss(normalized, torch.zeros_like(normalized))
    total = 0.5 * visual + 0.5 * state
    return total, {
        "total": total.detach(),
        "visual": visual.detach(),
        "state": state.detach(),
        "direct_assignment": direct.mean(),
        "swapped_assignment": swapped.mean(),
    }


def actor_objective(
    commands: torch.Tensor,
    target: torch.Tensor,
    *,
    step_cap: torch.Tensor,
) -> torch.Tensor:
    return (
        (commands.float() - target.float()).abs()
        / step_cap.float()[None, None]
    ).mean()


@torch.no_grad()
def build_actor_conditioning(
    lewm: V175GroundedLeWM,
    bank: V175CombinedGroundedBank | V175DeviceGroundedBank,
    pair: V181TemporalActorBatch,
    *,
    device: torch.device,
) -> tuple[V175Memory, V175Innovation]:
    batch = len(pair.ordinal)
    actual = torch.zeros(
        batch, HISTORY_CHUNKS, VIEWS, ANCHORS, ANCHOR_DIM,
        device=device,
    )
    actual_valid = torch.zeros(
        batch, HISTORY_CHUNKS, VIEWS, ANCHORS,
        dtype=torch.bool, device=device,
    )
    visual = torch.zeros_like(actual)
    state = torch.zeros(
        batch, HISTORY_CHUNKS, PATH_STEPS, STATE_DIM, device=device
    )
    innovation_valid = torch.zeros_like(actual_valid)
    path_valid = torch.zeros(
        batch, HISTORY_CHUNKS, PATH_STEPS, dtype=torch.bool, device=device
    )
    history_valid = torch.from_numpy(pair.history_valid).to(device)
    if history_valid.shape != (batch, HISTORY_CHUNKS):
        raise V175TrainingError("Actor history must be padded to four causal chunks")
    commands = torch.from_numpy(
        restore_companion_joint_units(pair.history_commands)
    ).to(device)
    paths = torch.from_numpy(
        restore_companion_joint_units(pair.history_paths)
    ).to(device)
    current = V175Innovation(
        visual=torch.zeros(batch, VIEWS, ANCHORS, ANCHOR_DIM, device=device),
        state=torch.zeros(batch, PATH_STEPS, STATE_DIM, device=device),
        anchor_valid=torch.zeros(batch, VIEWS, ANCHORS, dtype=torch.bool, device=device),
        path_valid=torch.zeros(batch, PATH_STEPS, dtype=torch.bool, device=device),
    )
    for slot in range(HISTORY_CHUNKS):
        active = history_valid[:, slot]
        source, source_valid = bank.get(pair.history_source_endpoint_index[:, slot])
        target, target_valid = bank.get(pair.history_target_endpoint_index[:, slot])
        source_valid = source_valid & active[:, None, None]
        target_valid = target_valid & active[:, None, None]
        with _autocast(device):
            prediction = lewm(
                source,
                source_valid,
                paths[:, slot, 0],
                commands[:, slot],
            )
        target, target_valid, _ = align_target_anchor_set(
            prediction.predicted_anchor_tokens,
            target,
            source_valid,
            target_valid,
        )
        innovation = compute_v175_innovation(
            target,
            target_valid,
            paths[:, slot],
            prediction,
            path_valid=active[:, None].expand(-1, PATH_STEPS),
        )
        actual[:, slot] = target.float()
        actual_valid[:, slot] = target_valid
        visual[:, slot] = innovation.visual.float()
        state[:, slot] = innovation.state.float()
        innovation_valid[:, slot] = innovation.anchor_valid
        path_valid[:, slot] = innovation.path_valid
        last = torch.from_numpy(pair.transition == slot + 1).to(device)
        current = V175Innovation(
            visual=torch.where(last[:, None, None, None], innovation.visual, current.visual),
            state=torch.where(last[:, None, None], innovation.state, current.state),
            anchor_valid=torch.where(last[:, None, None], innovation.anchor_valid, current.anchor_valid),
            path_valid=torch.where(last[:, None], innovation.path_valid, current.path_valid),
        )
    memory = V175Memory(
        actual_anchor_tokens=actual,
        actual_anchor_valid=actual_valid,
        completed_commands=commands * history_valid[..., None, None],
        measured_paths=paths * history_valid[..., None, None],
        visual_innovations=visual,
        state_innovations=state,
        innovation_valid=innovation_valid,
        path_valid=path_valid,
        history_valid=history_valid,
    )
    memory.validate()
    return memory, current


def _open_grounded_arrays(
    output: Path,
    endpoint_index: np.ndarray,
) -> tuple[
    np.memmap,
    np.memmap,
    np.memmap,
    np.memmap,
    np.memmap,
    np.memmap,
]:
    output.mkdir(parents=True, exist_ok=True)
    endpoints = np.asarray(endpoint_index, dtype=np.int64)
    if endpoints.ndim != 1 or len(endpoints) == 0 or bool((np.diff(endpoints) <= 0).any()):
        raise V175TrainingError("grounded-cache endpoints must be sorted unique")
    identity_path = output / "endpoint_index.npy"
    if identity_path.is_file():
        previous = np.load(identity_path, mmap_mode="r", allow_pickle=False)
        if not bool(np.array_equal(previous, endpoints)):
            raise V175TrainingError("resumed grounded-cache identities differ")
    else:
        np.save(identity_path, endpoints, allow_pickle=False)

    specifications = {
        "anchor_tokens": (np.float16, (len(endpoints), VIEWS, ANCHORS, ANCHOR_DIM)),
        "anchor_valid": (np.bool_, (len(endpoints), VIEWS, ANCHORS)),
        "positions_px": (np.float16, (len(endpoints), VIEWS, ANCHORS, 2)),
        "position_valid": (np.bool_, (len(endpoints), VIEWS, ANCHORS)),
        "entropy": (np.float16, (len(endpoints), VIEWS, ANCHORS)),
        "done": (np.bool_, (len(endpoints),)),
    }
    values: list[np.memmap] = []
    for name, (dtype, shape) in specifications.items():
        path = output / f"{name}.npy"
        if path.is_file():
            array = np.lib.format.open_memmap(path, mode="r+")
            if array.shape != shape or array.dtype != np.dtype(dtype):
                raise V175TrainingError(f"resumed grounded {name} shape/dtype differs")
        else:
            array = np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=shape
            )
            if name == "done":
                array[:] = False
                array.flush()
        values.append(array)
    return tuple(values)  # type: ignore[return-value]


@torch.inference_mode()
def prepare_grounded_cache(
    feature_cache: V181FeatureCache,
    fcpg: V175GroundedAnchorFCPG,
    output_root: str | Path,
    *,
    device: torch.device,
    kind: str,
    state_provider: Any,
    rgb_provider: Any | None = None,
    projection_provider: V175AffineProjectionProvider | None = None,
    batch_size: int = 128,
) -> Path:
    """Materialize measured image states or explicitly kinematic archive states."""

    output = Path(output_root).expanduser().resolve()
    anchor, valid_out, positions, position_valid, entropy, done = _open_grounded_arrays(
        output, np.asarray(feature_cache.endpoint_index, dtype=np.int64)
    )
    fcpg.to(device).eval()
    pending = np.flatnonzero(~done)
    total_batches = math.ceil(len(pending) / int(batch_size))
    started = time.monotonic()
    spatial_mode = (
        "calibrated-current-image-measurement"
        if projection_provider is not None
        else "kinematic-only-no-camera"
    )
    if (projection_provider is None) != (rgb_provider is None):
        raise V175TrainingError(
            "RGB and calibrated projection providers must be supplied together"
        )
    for batch_index, selected in enumerate(batches(pending, batch_size), start=1):
        endpoint = np.asarray(feature_cache.endpoint_index[selected], dtype=np.int64)
        if hasattr(state_provider, "current"):
            current_np = state_provider.current(endpoint)
        else:
            state_np = state_provider(endpoint)
            if not isinstance(state_np, dict) or "current_q" not in state_np:
                raise V175TrainingError("grounded state provider omits current_q")
            current_np = state_np["current_q"]
        current_q = torch.from_numpy(
            np.ascontiguousarray(current_np)
        ).to(device)
        view_valid = torch.from_numpy(
            np.ascontiguousarray(feature_cache.view_valid[selected])
        ).to(device)
        if projection_provider is None:
            with _autocast(device):
                tokens, measured_valid, _ = fcpg.kinematic_part_tokens(
                    current_q,
                    view_valid,
                )
            measured_position = torch.zeros(
                len(selected), VIEWS, ANCHORS, 2, device=device
            )
            pixel_valid = torch.zeros_like(measured_valid)
            measured_entropy = torch.zeros(
                len(selected), VIEWS, ANCHORS, device=device
            )
        else:
            supplied = np.asarray(
                rgb_provider(endpoint)
            )
            if supplied.shape == (len(selected), VIEWS, 224, 224, 3):
                supplied = np.transpose(supplied, (0, 1, 4, 2, 3))
            if supplied.shape != (len(selected), VIEWS, 3, 224, 224):
                raise V175TrainingError("grounded RGB provider shape drifted")
            rgb = torch.from_numpy(np.ascontiguousarray(supplied)).to(device)
            cached = np.asarray(feature_cache.tokens[selected])
            dino = torch.from_numpy(np.ascontiguousarray(cached)).to(device)
            nominal, nominal_valid, jacobian = projection_provider(current_q)
            with _autocast(device):
                measurement = fcpg(
                    rgb,
                    dino,
                    view_valid,
                    current_q,
                    None,
                    None,
                    cached_nominal_keypoints_px=nominal,
                    cached_nominal_valid=nominal_valid,
                    cached_nominal_jacobian_px=jacobian,
                )
            tokens = measurement.grounded_part_tokens
            measured_valid = measurement.anchor_valid
            measured_position = measurement.corrected_keypoints_px
            pixel_valid = measurement.anchor_valid
            probability = measurement.correspondence_maps["local"].float()
            measured_entropy = -(
                probability.clamp_min(1.0e-8)
                * probability.clamp_min(1.0e-8).log()
            ).sum(dim=-1) / math.log(probability.shape[-1])
        outputs = (tokens, measured_position, measured_entropy)
        if not all(bool(torch.isfinite(value).all()) for value in outputs):
            raise V175TrainingError("grounded cache batch is non-finite")
        anchor[selected] = tokens.float().cpu().numpy().astype(np.float16)
        valid_out[selected] = measured_valid.cpu().numpy()
        positions[selected] = measured_position.float().cpu().numpy().astype(np.float16)
        position_valid[selected] = pixel_valid.cpu().numpy()
        entropy[selected] = measured_entropy.float().cpu().numpy().astype(np.float16)
        done[selected] = True
        for array in (anchor, valid_out, positions, position_valid, entropy, done):
            array.flush()
        _write_json(
            output / "config.json",
            {
                "version": "17.5",
                "contract": "structured-kinematic-v2",
                "kind": str(kind),
                "spatial_mode": spatial_mode,
                "view_order": list(VIEW_ORDER),
                "rows": int(len(feature_cache.endpoint_index)),
                "complete": bool(done.all()),
                "identity_mode": "ordered-robot-structure",
                "absolute_pixels_available": bool(projection_provider is not None),
                "history_conditioned": False,
                "history_reason": "endpoint-indexed cache cannot encode pair-specific history",
                "predicted_states_stored": False,
            },
        )
        _progress(
            f"grounded-cache-{kind}", batch_index, total_batches,
            float(measured_entropy.mean()), started,
        )
    if len(pending) == 0:
        _write_json(
            output / "config.json",
            {
                "version": "17.5",
                "contract": "structured-kinematic-v2",
                "kind": str(kind),
                "spatial_mode": spatial_mode,
                "view_order": list(VIEW_ORDER),
                "rows": int(len(feature_cache.endpoint_index)),
                "complete": bool(done.all()),
                "identity_mode": "ordered-robot-structure",
                "absolute_pixels_available": bool(projection_provider is not None),
                "history_conditioned": False,
                "history_reason": "endpoint-indexed cache cannot encode pair-specific history",
                "predicted_states_stored": False,
            },
        )
    return output


def _cad_arrays(
    cad: V175CADArchive,
    rows: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    selected = np.asarray(rows, dtype=np.int64)
    # CAD cache is [column,wrist]; V17.5 is [wrist,column].
    view_order = np.asarray((1, 0), dtype=np.int64)
    rgb = np.asarray(cad.rgb[selected], dtype=np.uint8)[:, view_order]
    rgb = np.transpose(rgb, (0, 1, 4, 2, 3)).copy()
    target = np.asarray(cad.keypoints_px[selected], dtype=np.float32)[:, view_order]
    valid = np.asarray(cad.valid[selected], dtype=np.bool_)[:, view_order]
    state = np.asarray(cad.state[selected], dtype=np.float32)
    keypoints_base_xyz = np.asarray(
        cad.keypoints_base_xyz[selected], dtype=np.float32
    )
    camera_from_reference = np.asarray(
        cad.camera_from_reference[selected], dtype=np.float32
    )[:, view_order]
    intrinsic = np.asarray(cad.intrinsic[selected], dtype=np.float32)[:, view_order]
    return (
        rgb,
        state,
        keypoints_base_xyz,
        camera_from_reference,
        intrinsic,
        target,
        valid,
    )


def _real_label_arrays(
    cache_path: str | Path,
    backbone: nn.Module,
    *,
    device: torch.device,
    projection_provider: V175AffineProjectionProvider | None = None,
) -> dict[str, torch.Tensor]:
    payload = torch.load(
        Path(cache_path).expanduser().resolve(), map_location="cpu", weights_only=False
    )
    required = {
        "rgb",
        "nominal_fm_px",
        "target_fm_px",
        "view_valid",
        "ordered",
        "current_q",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise V175TrainingError("real-anchor cache fields are incomplete")
    # This cache was produced in V17.4 [column,wrist] order.
    order = torch.tensor((1, 0), dtype=torch.long)
    rgb = torch.as_tensor(payload["rgb"], dtype=torch.uint8)[:, order]
    nominal_fm = torch.as_tensor(
        payload["nominal_fm_px"], dtype=torch.float32
    )[:, order]
    target = torch.as_tensor(payload["target_fm_px"], dtype=torch.float32)[:, order]
    view_valid = torch.as_tensor(payload["view_valid"], dtype=torch.bool)[:, order]
    ordered = torch.as_tensor(payload["ordered"], dtype=torch.bool)
    current_q = torch.as_tensor(payload["current_q"], dtype=torch.float32)
    if rgb.ndim != 5 or tuple(rgb.shape[1:]) != (VIEWS, 3, 224, 224):
        raise V175TrainingError("real-anchor RGB shape drifted")
    if current_q.shape != (len(rgb), STATE_DIM) or ordered.shape != (len(rgb),):
        raise V175TrainingError("real-anchor q/identity shape drifted")
    # The legacy cache has a recoverable settled q only on its 24 ordered
    # rows.  All-zero q on the other rows is a missing-value sentinel, not a
    # physical pose, and must never be treated as geometry supervision.
    q_valid = ordered & torch.isfinite(current_q).all(dim=-1)
    nominal_all = torch.zeros(len(rgb), VIEWS, ANCHORS, 2, dtype=torch.float32)
    nominal_all[:, :, list(CAD_TIP_INDICES)] = nominal_fm
    nominal_valid = torch.zeros(len(rgb), VIEWS, ANCHORS, dtype=torch.bool)
    nominal_valid[:, :, list(CAD_TIP_INDICES)] = view_valid[:, :, None]
    nominal_jacobian = torch.zeros(
        len(rgb), VIEWS, ANCHORS, 2, STATE_DIM, dtype=torch.float32
    )
    if projection_provider is not None and bool(q_valid.any()):
        selected = torch.nonzero(q_valid, as_tuple=False).flatten()
        projected, projected_valid, projected_jacobian = projection_provider(
            current_q[selected].to(device)
        )
        nominal_all[selected] = projected.cpu()
        nominal_valid[selected] = projected_valid.cpu()
        nominal_jacobian[selected] = projected_jacobian.cpu()
    rgb = rgb.to(device)
    dino = extract_dino_tensor(backbone, rgb, device=device).detach()
    return {
        "rgb": rgb,
        "dino": dino,
        "target": target.to(device),
        "valid": view_valid.to(device),
        "ordered": ordered.to(device),
        "current_q": current_q.to(device),
        "q_valid": q_valid.to(device),
        "nominal_all": nominal_all.to(device),
        "nominal_valid": nominal_valid.to(device),
        "nominal_jacobian": nominal_jacobian.to(device),
    }


def balanced_real_anchor_batch(
    view_valid: np.ndarray,
    batch_size: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample equal column/wrist supervision without inventing point labels."""

    valid = np.asarray(view_valid, dtype=np.bool_)
    if valid.ndim != 2 or valid.shape[1] != VIEWS:
        raise V175TrainingError("real-anchor validity must be [N,2]")
    if int(batch_size) < VIEWS:
        raise V175TrainingError("real-anchor batch must cover both views")
    supervised_view = np.arange(int(batch_size), dtype=np.int64) % VIEWS
    selected = np.empty(int(batch_size), dtype=np.int64)
    for view in range(VIEWS):
        candidates = np.flatnonzero(valid[:, view])
        if len(candidates) == 0:
            raise V175TrainingError("real-anchor cache has an unsupervised view")
        locations = np.flatnonzero(supervised_view == view)
        selected[locations] = generator.choice(
            candidates, size=len(locations), replace=True
        )
    order = generator.permutation(int(batch_size))
    selected = selected[order]
    supervised_view = supervised_view[order]
    supervision = np.zeros((int(batch_size), VIEWS), dtype=np.bool_)
    supervision[np.arange(int(batch_size)), supervised_view] = True
    if not bool(valid[selected][supervision].all()):
        raise V175TrainingError("balanced real-anchor sampler selected an invalid label")
    return selected, supervision


def localized_cad_order(
    rows: np.ndarray,
    *,
    seed: int,
    page_rows: int = 512,
) -> np.ndarray:
    """Shuffle CAD pages and rows locally so the fixed GPU page bank is useful."""

    population = np.asarray(rows, dtype=np.int64)
    if population.ndim != 1 or len(population) == 0 or int(page_rows) <= 0:
        raise V175TrainingError("localized CAD population/page size is invalid")
    generator = np.random.default_rng(int(seed))
    pages = np.unique(population // int(page_rows))
    generator.shuffle(pages)
    blocks: list[np.ndarray] = []
    for page in pages:
        block = population[population // int(page_rows) == page].copy()
        generator.shuffle(block)
        blocks.append(block)
    return np.concatenate(blocks)


def train_v175_fcpg(
    fcpg: V175GroundedAnchorFCPG,
    real_rgb: V175AllRealRGB,
    cad: V175CADArchive,
    real_label_cache: str | Path,
    backbone: nn.Module,
    output_root: str | Path,
    *,
    device: torch.device,
    real_projection: V175AffineProjectionProvider | None = None,
    temporal_archive: V181TemporalArchive | None = None,
    cad_feature_bank: V181DeviceFeatureBank | None = None,
    distill_batch: int = 64,
    cad_microbatch: int = 64,
    cad_accumulation: int = 1,
    real_batch: int = 32,
    limits: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Fixed appearance init, four-pass structured CAD, and real training fit."""

    limits = dict(limits or {})
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    final = output / "fcpg-final.pt"
    if final.is_file() and not limits:
        payload = load_component(final, fcpg, component="fcpg", device=device)
        return dict(payload.get("metrics", {}))
    seed_v175()
    fcpg.to(device).train()
    metrics: dict[str, Any] = {"distill": [], "cad": [], "real": []}
    global_step = 0
    latest_path = output / "latest.pt"
    latest_payload: dict[str, Any] | None = None
    latest_phase = ""
    if latest_path.is_file() and not limits:
        candidate = torch.load(latest_path, map_location="cpu", weights_only=False)
        if (
            isinstance(candidate, dict)
            and candidate.get("version") == "17.5"
            and candidate.get("architecture") == ARCHITECTURE_CONTRACT
            and candidate.get("component") == "fcpg"
        ):
            latest_payload = candidate
            latest_phase = str(candidate.get("metrics", {}).get("phase", ""))
            if not isinstance(candidate.get("model"), dict):
                raise V175TrainingError("FCPG latest checkpoint lacks model state")
            fcpg.load_state_dict(candidate["model"], strict=True)
            global_step = int(candidate.get("step", 0))
            completed = candidate.get("cursor", {}).get("completed_metrics")
            if isinstance(completed, dict):
                metrics.update(completed)

    # Stage 1: all 100K endpoints from the 20K cache; no masks are opened.
    distill_population = real_rgb.endpoint_rows.copy()
    np.random.default_rng(SEED).shuffle(distill_population)
    if limits.get("distill", 0) > 0:
        distill_population = distill_population[: limits["distill"]]
    appearance_parameters = list(fcpg.rgb_stem.parameters()) + list(
        fcpg.rgb_coarse.parameters()
    ) + list(fcpg.fusion.parameters())
    optimizer = torch.optim.AdamW(appearance_parameters, lr=FCPG_DISTILL_LR)
    distill_start = 0
    distill_sum = 0.0
    distill_count = 0
    if latest_phase == "distill" and latest_payload is not None:
        restored = _restore_latest_without_scheduler(
            latest_path, fcpg, optimizer, component="fcpg"
        )
        assert restored is not None
        global_step, cursor, previous = restored
        distill_start = int(cursor.get("examples", 0))
        distill_count = distill_start
        distill_sum = float(previous.get("loss", 0.0)) * distill_count
    elif latest_phase in {"cad", "real"}:
        distill_start = len(distill_population)
    stage_started = time.monotonic()
    distill_steps = math.ceil(len(distill_population) / int(distill_batch))
    for distill_step, selected in enumerate(
        batches(distill_population[distill_start:], distill_batch),
        start=distill_start // int(distill_batch) + 1,
    ):
        rgb_np, valid_np = real_rgb.batch(selected)
        rgb = torch.from_numpy(rgb_np).to(device)
        valid = torch.from_numpy(valid_np).to(device)
        dino = extract_dino_tensor(backbone, rgb, device=device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            loss = feature_distillation_objective(fcpg, rgb, dino, valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(appearance_parameters, GRADIENT_CLIP)
        optimizer.step()
        global_step += 1
        distill_sum += float(loss.detach()) * len(selected)
        distill_count += len(selected)
        _progress(
            "fcpg-distill", distill_step, distill_steps,
            distill_sum / distill_count, stage_started,
        )
        if global_step % CHECKPOINT_EVERY == 0:
            save_component(
                output / "latest.pt", fcpg, component="fcpg", step=global_step,
                metrics={"phase": "distill", "loss": distill_sum / distill_count},
                optimizer=optimizer,
                cursor={"examples": distill_count, "completed_metrics": metrics},
            )
    if distill_count:
        metrics["distill"] = {
            "endpoints": distill_count,
            "loss": distill_sum / max(1, distill_count),
        }

    # Stage 2: exact CAD pixels.  DINO is extracted online to avoid consuming
    # the final ~129 GiB of /data with a redundant ~79 GiB cache.
    parameters = list(fcpg.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=FCPG_CAD_LR)
    cad_rows = np.asarray(cad.train_rows, dtype=np.int64)
    if limits.get("cad", 0) > 0:
        cad_rows = cad_rows[: limits["cad"]]
    cad_start_pass = 0
    cad_start_examples = 0
    cad_optimizer_step = 0
    if latest_phase == "cad" and latest_payload is not None:
        restored = _restore_latest_without_scheduler(
            latest_path, fcpg, optimizer, component="fcpg"
        )
        assert restored is not None
        global_step, cursor, _ = restored
        cad_start_pass = int(cursor.get("pass", 0))
        cad_start_examples = int(cursor.get("examples", 0))
        cad_optimizer_step = cad_start_pass * math.ceil(
            len(cad_rows) / max(1, int(cad_microbatch) * int(cad_accumulation))
        ) + math.ceil(cad_start_examples / max(1, int(cad_microbatch) * int(cad_accumulation)))
    elif latest_phase in {"history", "real"}:
        cad_start_pass = FCPG_CAD_PASSES
    stage_started = time.monotonic()
    cad_total_steps = FCPG_CAD_PASSES * math.ceil(
        len(cad_rows) / max(1, int(cad_microbatch) * int(cad_accumulation))
    )
    for pass_index in range(cad_start_pass, FCPG_CAD_PASSES):
        order = localized_cad_order(
            cad_rows,
            seed=SEED + 100 + pass_index,
            page_rows=(cad_feature_bank.page_rows if cad_feature_bank is not None else 512),
        )
        loss_sum = 0.0
        examples = 0
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        pass_offset = cad_start_examples if pass_index == cad_start_pass else 0
        for selected in batches(order[pass_offset:], cad_microbatch):
            (
                rgb_np,
                state_np,
                keypoints_base_np,
                camera_from_reference_np,
                intrinsic_np,
                target_np,
                target_valid_np,
            ) = _cad_arrays(cad, selected)
            rgb = torch.from_numpy(rgb_np).to(device)
            current_q = torch.from_numpy(state_np).to(device)
            camera_from_reference = torch.from_numpy(
                camera_from_reference_np
            ).to(device)
            intrinsic = torch.from_numpy(intrinsic_np).to(device)
            target = torch.from_numpy(target_np).to(device)
            target_valid = torch.from_numpy(target_valid_np).to(device)
            if cad_feature_bank is None:
                dino = extract_dino_tensor(backbone, rgb, device=device)
            else:
                dino, cached_view_valid = cad_feature_bank.get(selected)
                if not bool(cached_view_valid.all()):
                    raise V175TrainingError("CAD DINO cache unexpectedly masks a view")
            view_valid = torch.ones(
                len(selected), VIEWS, dtype=torch.bool, device=device
            )
            # Perturb the *prior correction state*, not the measured q.  Half
            # the rows get a small disturbance and half a large one.  This
            # trains current-image re-anchoring while preserving the endpoint
            # clock and exact CAD state.
            row_id = torch.as_tensor(selected, device=device, dtype=torch.float32)
            phase = row_id[:, None] * current_q.new_tensor(
                (0.017, 0.029, 0.043, 0.061, 0.079, 0.097)
            )[None]
            large = ((torch.as_tensor(selected, device=device) + pass_index) % 2).float()[:, None]
            amplitude = 0.01 + 0.03 * large
            joint_prior = torch.sin(phase + float(pass_index)) * amplitude
            view_phase = row_id[:, None, None] * current_q.new_tensor(
                (0.013, 0.021, 0.034, 0.055)
            )[None, None]
            view_sign = current_q.new_tensor(((1.0,), (-1.0,)))[None]
            raw_registration = torch.sin(view_phase + view_sign)
            registration_scale = current_q.new_tensor((24.0, 24.0, 0.12, 0.12))
            registration = raw_registration * registration_scale * (0.25 + 0.75 * large[:, None])
            prior_memory = V175FCPGMemory.empty(
                len(selected), STATE_DIM, device=device, dtype=current_q.dtype
            )
            prior_memory = V175FCPGMemory(
                joint_correction_fraction=joint_prior,
                view_registration=registration,
                previous_q=current_q,
                readback_summary=prior_memory.readback_summary,
                history_valid=prior_memory.history_valid,
            )
            with _autocast(device):
                measurement = fcpg(
                    rgb,
                    dino,
                    view_valid,
                    current_q,
                    intrinsic,
                    camera_from_reference,
                    previous_memory=prior_memory,
                )
                loss, _ = ordered_anchor_smooth_l1(
                    measurement, target, target_valid
                )
                scaled = loss / int(cad_accumulation)
            # Loader/geometry contract audit: the differentiable graph must
            # reproduce the CAD's stored all-six base points.
            if pass_index == 0 and examples == 0:
                with torch.no_grad():
                    observed_base = torch.from_numpy(keypoints_base_np).to(device)
                    predicted_base = fcpg.kinematics(current_q)[0]
                    maximum = float((predicted_base - observed_base).abs().max())
                    if maximum > 2.0e-5:
                        raise V175TrainingError(
                            f"CAD state/all-six FK contract differs: max_abs_m={maximum:.6g}"
                        )
            scaled.backward()
            accumulated += 1
            examples += len(selected)
            loss_sum += float(loss.detach()) * len(selected)
            if accumulated == int(cad_accumulation):
                torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                global_step += 1
                cad_optimizer_step += 1
                _progress(
                    "fcpg-cad", cad_optimizer_step, cad_total_steps,
                    loss_sum / examples, stage_started,
                )
                if global_step % CHECKPOINT_EVERY == 0:
                    save_component(
                        output / "latest.pt", fcpg, component="fcpg", step=global_step,
                        metrics={"phase": "cad", "pass": pass_index, "loss": loss_sum / examples},
                        optimizer=optimizer,
                        cursor={
                            "pass": pass_index,
                            "examples": pass_offset + examples,
                            "completed_metrics": metrics,
                        },
                    )
        if accumulated:
            torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
            optimizer.step()
            global_step += 1
            cad_optimizer_step += 1
            _progress(
                "fcpg-cad", cad_optimizer_step, cad_total_steps,
                loss_sum / examples, stage_started,
            )
        metrics["cad"].append(
            {"pass": pass_index, "examples": examples, "smooth_l1": loss_sum / max(1, examples)}
        )
        cad_start_examples = 0

    # Stage 3: learn how actual command/readback history indexes the same
    # ordered morphology state.  No pixel target is invented: the target is
    # the current-q structural token, and the source must reconstruct it from
    # the completed path with current q masked.
    if temporal_archive is not None:
        history_parameters = (
            list(fcpg.history_projection.parameters())
            + [fcpg.history_embedding]
            + list(fcpg.motion_layers.parameters())
        )
        fcpg.requires_grad_(False)
        for parameter in history_parameters:
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(history_parameters, lr=FCPG_DISTILL_LR)
        history_start_horizon = 1
        history_start_offset = 0
        history_metrics: list[dict[str, Any]] = []
        if latest_phase == "history" and latest_payload is not None:
            restored = _restore_latest_without_scheduler(
                latest_path, fcpg, optimizer, component="fcpg"
            )
            assert restored is not None
            global_step, cursor, _ = restored
            history_start_horizon = int(cursor.get("horizon", 1))
            history_start_offset = int(cursor.get("offset", 0))
            history_metrics = [dict(value) for value in cursor.get("trace", [])]
        elif latest_phase == "real":
            history_start_horizon = 5
        history_rows = np.arange(temporal_archive.rows, dtype=np.int64)
        if limits.get("history", 0) > 0:
            history_rows = history_rows[: int(limits["history"])]
        history_total_steps = 4 * math.ceil(len(history_rows) / 256)
        history_progress = sum(
            math.ceil(len(history_rows) / 256) for _ in range(1, history_start_horizon)
        )
        stage_started = time.monotonic()
        for horizon in range(history_start_horizon, 5):
            order = history_rows.copy()
            np.random.default_rng(SEED + 700 + horizon).shuffle(order)
            offset = history_start_offset if horizon == history_start_horizon else 0
            loss_sum = 0.0
            examples = 0
            for begin in range(offset, len(order), 256):
                selected = order[begin : begin + 256]
                raw_paths = np.asarray(
                    temporal_archive.measured_state_path[selected], dtype=np.float32
                )
                raw_commands = np.asarray(
                    temporal_archive.completed_commands[selected], dtype=np.float32
                )
                current_q = torch.from_numpy(
                    restore_companion_joint_units(raw_paths[:, horizon - 1, -1])
                ).to(device)
                commands = np.zeros(
                    (len(selected), HISTORY_CHUNKS, COMMAND_STEPS, STATE_DIM),
                    dtype=np.float32,
                )
                paths = np.zeros(
                    (len(selected), HISTORY_CHUNKS, PATH_STEPS, STATE_DIM),
                    dtype=np.float32,
                )
                commands[:, :horizon] = raw_commands[:, :horizon]
                paths[:, :horizon] = raw_paths[:, :horizon]
                causal = torch.zeros(
                    len(selected), HISTORY_CHUNKS, dtype=torch.bool, device=device
                )
                causal[:, :horizon] = True
                endpoint = np.asarray(
                    temporal_archive.row_endpoint_index[selected, horizon],
                    dtype=np.int64,
                )
                view_valid = torch.from_numpy(
                    np.asarray(
                        temporal_archive.endpoint_view_valid[endpoint], dtype=np.bool_
                    )
                ).to(device)
                commands_t = torch.from_numpy(
                    restore_companion_joint_units(commands)
                ).to(device)
                paths_t = torch.from_numpy(
                    restore_companion_joint_units(paths)
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    target_tokens, target_valid, _ = fcpg.kinematic_part_tokens(
                        current_q, view_valid
                    )
                with _autocast(device):
                    predicted_tokens, predicted_valid, _ = fcpg.kinematic_part_tokens(
                        current_q,
                        view_valid,
                        completed_commands_deg=commands_t,
                        measured_q_paths_deg=paths_t,
                        history_valid=causal,
                        joint_state_valid=torch.zeros(
                            len(selected), dtype=torch.bool, device=device
                        ),
                    )
                    active = target_valid & predicted_valid
                    element = F.smooth_l1_loss(
                        predicted_tokens.float(), target_tokens.float(), reduction="none"
                    ).mean(dim=-1)
                    loss = (element * active).sum() / active.sum().clamp_min(1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(history_parameters, GRADIENT_CLIP)
                optimizer.step()
                global_step += 1
                history_progress += 1
                examples += len(selected)
                loss_sum += float(loss.detach()) * len(selected)
                _progress(
                    "fcpg-history",
                    history_progress,
                    history_total_steps,
                    loss_sum / max(1, examples),
                    stage_started,
                )
                if global_step % CHECKPOINT_EVERY == 0:
                    save_component(
                        latest_path,
                        fcpg,
                        component="fcpg",
                        step=global_step,
                        metrics={
                            "phase": "history",
                            "horizon": horizon,
                            "loss": loss_sum / max(1, examples),
                        },
                        optimizer=optimizer,
                        cursor={
                            "horizon": horizon,
                            "offset": begin + len(selected),
                            "trace": history_metrics,
                            "completed_metrics": metrics,
                        },
                    )
            history_metrics.append(
                {
                    "horizon": horizon,
                    "examples": examples,
                    "smooth_l1": loss_sum / max(1, examples),
                }
            )
            history_start_offset = 0
        metrics["history"] = {
            "archive_rows": int(len(history_rows)),
            "endpoint_examples": int(len(history_rows) * 4),
            "current_q_masked": True,
            "target": "current-q ordered morphology tokens",
            "trace": history_metrics,
        }
        fcpg.requires_grad_(True)

    # Stage 4: the 182 valid real images are explicitly training-fit only.
    labels = _real_label_arrays(
        real_label_cache,
        backbone,
        device=device,
        projection_provider=real_projection,
    )
    optimizer = torch.optim.AdamW(parameters, lr=FCPG_REAL_LR)
    real_updates = int(limits.get("real_updates", FCPG_REAL_UPDATES))
    real_sum = 0.0
    real_start = 0
    if latest_phase == "real" and latest_payload is not None:
        restored = _restore_latest_without_scheduler(
            latest_path, fcpg, optimizer, component="fcpg"
        )
        assert restored is not None
        global_step, cursor, previous = restored
        real_start = int(cursor.get("updates", 0))
        real_sum = float(previous.get("loss", 0.0)) * real_start
    generator = np.random.default_rng(SEED + 500)
    stage_started = time.monotonic()
    label_valid_np = labels["valid"].cpu().numpy()
    # Advancing the deterministic generator preserves sampled rows on resume.
    if real_start:
        for _ in range(real_start):
            balanced_real_anchor_batch(label_valid_np, real_batch, generator)
    for update in range(real_start, real_updates):
        selected, supervised_np = balanced_real_anchor_batch(
            label_valid_np, real_batch, generator
        )
        index = torch.from_numpy(selected.astype(np.int64, copy=False)).to(device)
        supervised = torch.from_numpy(supervised_np).to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            measurement = fcpg(
                labels["rgb"][index],
                labels["dino"][index],
                supervised,
                labels["current_q"][index],
                None,
                None,
                cached_nominal_keypoints_px=labels["nominal_all"][index],
                cached_nominal_valid=(
                    labels["nominal_valid"][index]
                    & supervised[:, :, None]
                ),
                cached_nominal_jacobian_px=labels["nominal_jacobian"][index],
                joint_state_valid=labels["q_valid"][index],
            )
            ordered_mask = labels["ordered"][index]
            direct = torch.nn.functional.smooth_l1_loss(
                measurement.fm_px,
                labels["target"][index],
                reduction="none",
                beta=2.0,
            ).mean(dim=(-1, -2))
            swapped = torch.nn.functional.smooth_l1_loss(
                measurement.fm_px,
                labels["target"][index][:, :, (1, 0)],
                reduction="none",
                beta=2.0,
            ).mean(dim=(-1, -2))
            per_view = torch.where(
                ordered_mask[:, None], direct, torch.minimum(direct, swapped)
            )
            loss = (per_view * supervised).sum() / supervised.sum().clamp_min(1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
        optimizer.step()
        global_step += 1
        real_sum += float(loss.detach())
        _progress(
            "fcpg-real-refit", update + 1, real_updates,
            real_sum / (update + 1), stage_started,
        )
        if global_step % CHECKPOINT_EVERY == 0:
            save_component(
                output / "latest.pt", fcpg, component="fcpg", step=global_step,
                metrics={"phase": "real", "loss": real_sum / (update + 1)},
                optimizer=optimizer,
                cursor={
                    "updates": update + 1,
                    "completed_metrics": metrics,
                },
            )
    metrics["real"] = {
        "updates": real_updates,
        "examples_in_cache": len(labels["rgb"]),
        "valid_views_in_cache": {
            VIEW_ORDER[view]: int(label_valid_np[:, view].sum())
            for view in range(VIEWS)
        },
        "supervised_views_per_update": {
            VIEW_ORDER[view]: int(real_batch // VIEWS)
            for view in range(VIEWS)
        },
        "ordered_rows": int(labels["ordered"].sum().item()),
        "q_recoverable_rows": int(labels["q_valid"].sum().item()),
        "training_fit_smooth_l1": real_sum / max(1, real_updates),
    }
    metrics["steps"] = global_step
    metrics["parameters"] = fcpg.parameter_count
    metrics["cad_feature_bank"] = (
        None if cad_feature_bank is None else cad_feature_bank.stats()
    )
    save_component(
        final, fcpg, component="fcpg", step=global_step, metrics=metrics
    )
    _write_json(output / "fcpg-metrics.json", metrics)
    return metrics


def _archive_partition_rows(
    archive: V181TemporalArchive, partition: str
) -> np.ndarray:
    if partition not in {"train", "dev"}:
        raise V175TrainingError("archive partition must be train or dev")
    return archive.rows_for_partition(partition)  # ordinary identity is retained


def train_v175_lewm(
    lewm: V175GroundedLeWM,
    actor_step_cap: torch.Tensor,
    archive: V181TemporalArchive,
    bank: V175CombinedGroundedBank,
    output_root: str | Path,
    *,
    device: torch.device,
    batch_size: int = 256,
    max_rows_per_partition: int | None = None,
) -> dict[str, Any]:
    """Train one complete pass for each H0-H1 through H3-H4 transition."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    final = output / "lewm-final.pt"
    if final.is_file() and max_rows_per_partition is None:
        payload = load_component(final, lewm, component="lewm", device=device)
        return dict(payload.get("metrics", {}))
    seed_v175()
    lewm.to(device).train().requires_grad_(True)
    parameters = list(lewm.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LEWM_LR)
    partition_rows = {
        name: _archive_partition_rows(archive, name)
        for name in ("train", "dev")
    }
    if max_rows_per_partition is not None:
        partition_rows = {
            name: rows[: int(max_rows_per_partition)]
            for name, rows in partition_rows.items()
        }
    total_steps = TRANSITIONS * sum(
        math.ceil(len(rows) / int(batch_size)) for rows in partition_rows.values()
    )
    warmup = max(1, int(round(total_steps * 0.05)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(step, total=total_steps, warmup=warmup),
    )
    step = 0
    start_transition = 0
    start_partition = 0
    start_offset = 0
    trace: list[dict[str, Any]] = []
    restored = _restore_latest(
        output / "latest.pt", lewm, optimizer, scheduler, component="lewm"
    )
    if restored is not None and max_rows_per_partition is None:
        step, cursor = restored
        start_transition = int(cursor.get("transition", 0))
        start_partition = int(cursor.get("partition", 0))
        start_offset = int(cursor.get("offset", 0))
        trace = [dict(value) for value in cursor.get("trace", [])]
    stage_started = time.monotonic()

    for transition in range(start_transition, TRANSITIONS):
        transition_sum = {"total": 0.0, "visual": 0.0, "state": 0.0}
        transition_examples = 0
        for partition_index, partition in enumerate(("train", "dev")):
            if transition == start_transition and partition_index < start_partition:
                continue
            rows = partition_rows[partition].copy()
            np.random.default_rng(SEED + transition * 10 + partition_index).shuffle(rows)
            offset = (
                start_offset
                if transition == start_transition and partition_index == start_partition
                else 0
            )
            if offset > len(rows):
                raise V175TrainingError("LeWM resume cursor leaves its population")
            for begin in range(offset, len(rows), int(batch_size)):
                selected = rows[begin : begin + int(batch_size)]
                batch = archive.transition_batch(
                    selected,
                    transition,
                    partition=partition,
                    allow_dev=partition == "dev",
                )
                source, source_valid = bank.get(batch.source_endpoint_index)
                target, target_valid = bank.get(batch.target_endpoint_index)
                source_valid &= torch.from_numpy(batch.source_view_valid).to(device)[:, :, None]
                target_valid &= torch.from_numpy(batch.target_view_valid).to(device)[:, :, None]
                current_q = torch.from_numpy(
                    restore_companion_joint_units(batch.current_q)
                ).to(device)
                commands = torch.from_numpy(
                    restore_companion_joint_units(batch.commands)
                ).to(device)
                measured_path = torch.from_numpy(
                    restore_companion_joint_units(batch.measured_q_path)
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device):
                    prediction = lewm(source, source_valid, current_q, commands)
                    loss, pieces = lewm_objective(
                        prediction,
                        target,
                        source_valid,
                        target_valid,
                        measured_path,
                        step_cap=actor_step_cap,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
                optimizer.step()
                scheduler.step()
                step += 1
                transition_examples += len(selected)
                for name in transition_sum:
                    transition_sum[name] += float(pieces[name]) * len(selected)
                _progress(
                    "lewm", step, total_steps, float(loss.detach()), stage_started
                )
                if step % CHECKPOINT_EVERY == 0:
                    save_component(
                        output / "latest.pt", lewm, component="lewm", step=step,
                        metrics={
                            "transition": transition,
                            **{
                                name: value / max(1, transition_examples)
                                for name, value in transition_sum.items()
                            },
                        },
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cursor={
                            "transition": transition,
                            "partition": partition_index,
                            "offset": begin + len(selected),
                            "trace": trace,
                        },
                    )
            start_offset = 0
        trace.append(
            {
                "transition": transition,
                "examples": transition_examples,
                **{
                    name: value / max(1, transition_examples)
                    for name, value in transition_sum.items()
                },
            }
        )
        start_partition = 0
    metrics = {
        "steps": step,
        "archive_rows": int(sum(len(value) for value in partition_rows.values())),
        "transitions": trace,
        "grounded_bank": {
            "train": bank.train.stats(),
            "development": bank.development.stats(),
        },
    }
    save_component(final, lewm, component="lewm", step=step, metrics=metrics)
    _write_json(output / "lewm-metrics.json", metrics)
    return metrics


def _archive_pair(
    archive: V181TemporalArchive,
    ordinal: np.ndarray,
    partition: str,
) -> V181TemporalActorBatch:
    return _pad_actor_history(archive.actor_pair_batch(
        ordinal,
        partition=partition,
        allow_dev=partition == "dev",
    ))


def _pad_actor_history(pair: V181TemporalActorBatch) -> V181TemporalActorBatch:
    """Extend the legacy three-slot batch to V17.5's H0--H4 four-slot clock."""

    batch = len(pair.ordinal)
    old = np.asarray(pair.history_valid, dtype=np.bool_)
    if old.shape == (batch, HISTORY_CHUNKS):
        return pair
    if old.shape != (batch, 3):
        raise V175TrainingError("legacy Actor history must contain three slots")
    source = np.broadcast_to(
        np.asarray(pair.source_endpoint_index, dtype=np.int64)[:, None],
        (batch, HISTORY_CHUNKS),
    ).copy()
    target = source.copy()
    source[:, :3] = np.asarray(pair.history_source_endpoint_index, dtype=np.int64)
    target[:, :3] = np.asarray(pair.history_target_endpoint_index, dtype=np.int64)
    valid = np.zeros((batch, HISTORY_CHUNKS), dtype=np.bool_)
    valid[:, :3] = old
    commands = np.zeros(
        (batch, HISTORY_CHUNKS, COMMAND_STEPS, STATE_DIM), dtype=np.float32
    )
    paths = np.zeros(
        (batch, HISTORY_CHUNKS, PATH_STEPS, STATE_DIM), dtype=np.float32
    )
    commands[:, :3] = np.asarray(pair.history_commands, dtype=np.float32)
    paths[:, :3] = np.asarray(pair.history_paths, dtype=np.float32)
    return V181TemporalActorBatch(
        ordinal=np.asarray(pair.ordinal, dtype=np.int64),
        row=np.asarray(pair.row, dtype=np.int64),
        transition=np.asarray(pair.transition, dtype=np.int64),
        source_endpoint_index=np.asarray(pair.source_endpoint_index, dtype=np.int64),
        source_view_valid=np.asarray(pair.source_view_valid, dtype=np.bool_),
        history_source_endpoint_index=source,
        history_target_endpoint_index=target,
        history_valid=valid,
        current_q=np.asarray(pair.current_q, dtype=np.float32),
        target_commands=np.asarray(pair.target_commands, dtype=np.float32),
        history_commands=commands,
        history_paths=paths,
    )


def localized_actor_order(
    archive: V181TemporalArchive,
    population: np.ndarray,
    *,
    partition: str,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Shuffle endpoint-local blocks so the 48 GiB page bank is effective."""

    rows = archive.rows_for_partition(partition)
    ordinal = np.asarray(population, dtype=np.int64)
    local_row = ordinal // TRANSITIONS
    transition = ordinal % TRANSITIONS
    if bool((local_row >= len(rows)).any()):
        raise V175TrainingError("Actor ordinal leaves its partition")
    endpoint = np.asarray(
        archive.row_endpoint_index[rows[local_row], transition], dtype=np.int64
    )
    sorted_position = np.argsort(endpoint, kind="stable")
    block_rows = max(int(batch_size), int(batch_size) * 16)
    blocks = [
        sorted_position[start : start + block_rows]
        for start in range(0, len(sorted_position), block_rows)
    ]
    np.random.default_rng(int(seed)).shuffle(blocks)
    return ordinal[np.concatenate(blocks)]


def _fish_as_temporal_pair(pair: Any) -> V181TemporalActorBatch:
    history_source = np.asarray(pair.history_source_frames, dtype=np.int64).copy()
    inactive = ~np.asarray(pair.history_valid, dtype=np.bool_)
    history_source[inactive] = np.broadcast_to(
        np.asarray(pair.source_frame)[:, None], history_source.shape
    )[inactive]
    history_target = history_source.copy()
    history_target[~inactive] += 15
    return _pad_actor_history(V181TemporalActorBatch(
        ordinal=np.asarray(pair.ordinal, dtype=np.int64),
        row=np.zeros(len(pair.ordinal), dtype=np.int64),
        transition=np.asarray(pair.transition, dtype=np.int64),
        source_endpoint_index=np.asarray(pair.source_frame, dtype=np.int64),
        source_view_valid=np.ones((len(pair.ordinal), VIEWS), dtype=np.bool_),
        history_source_endpoint_index=history_source,
        history_target_endpoint_index=history_target,
        history_valid=np.asarray(pair.history_valid, dtype=np.bool_),
        current_q=np.asarray(pair.current_q, dtype=np.float32),
        target_commands=np.asarray(pair.target_commands, dtype=np.float32),
        history_commands=np.asarray(pair.history_commands, dtype=np.float32),
        history_paths=np.asarray(pair.history_paths, dtype=np.float32),
    ))


def _actor_step(
    controller: V175Controller,
    pair: V181TemporalActorBatch,
    grounded_bank: V175CombinedGroundedBank | V175DeviceGroundedBank,
    feature_bank: V175CombinedFeatureBank | V181DeviceFeatureBank,
    *,
    device: torch.device,
    current_view_mask: torch.Tensor | None = None,
    innovation_zero: bool = False,
    history_prefix: int = HISTORY_CHUNKS,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    dino, feature_valid = feature_bank.get(pair.source_endpoint_index)
    anchors, anchor_valid = grounded_bank.get(pair.source_endpoint_index)
    archive_valid = torch.from_numpy(pair.source_view_valid).to(device)
    view_valid = feature_valid & archive_valid
    if current_view_mask is not None:
        view_valid &= current_view_mask
    anchor_valid &= view_valid[:, :, None]
    scene = controller.fcpg.scene_tokens_from_dino(dino, view_valid)
    memory, innovation = build_actor_conditioning(
        controller.lewm, grounded_bank, pair, device=device
    )
    if innovation_zero:
        innovation = V175Innovation(
            visual=torch.zeros_like(innovation.visual),
            state=torch.zeros_like(innovation.state),
            anchor_valid=torch.zeros_like(innovation.anchor_valid),
            path_valid=torch.zeros_like(innovation.path_valid),
        )
    else:
        innovation = V175Innovation(
            visual=innovation.visual * anchor_valid[..., None],
            state=innovation.state,
            anchor_valid=innovation.anchor_valid & anchor_valid,
            path_valid=innovation.path_valid,
        )
    current_q = torch.from_numpy(
        restore_companion_joint_units(pair.current_q)
    ).to(device)
    with _autocast(device):
        output = controller.actor(
            scene,
            anchors,
            anchor_valid,
            innovation.visual,
            innovation.state,
            innovation.anchor_valid,
            innovation.path_valid,
            current_q,
            memory,
            history_prefix=history_prefix,
        )
    target = torch.from_numpy(
        restore_companion_joint_units(pair.target_commands)
    ).to(device)
    return output.commands, target, output


def train_v175_archive_actor(
    controller: V175Controller,
    archive: V181TemporalArchive,
    grounded_bank: V175CombinedGroundedBank,
    feature_bank: V175CombinedFeatureBank,
    output_root: str | Path,
    *,
    device: torch.device,
    batch_size: int = 256,
    passes: int = ACTOR_ARCHIVE_PASSES,
    max_pairs_per_partition: int | None = None,
) -> dict[str, Any]:
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # If fish fine-tuning has already started, latest.pt contains the exact
    # archive-initialized Actor plus fish optimizer state.  Preserve it and let
    # train_v175_fish_actor resume instead of accidentally rebuilding archive
    # training from a newly initialized Actor.
    latest_path = output / "latest.pt"
    archive_metrics_path = output / "actor-archive-metrics.json"
    if latest_path.is_file() and max_pairs_per_partition is None:
        candidate = torch.load(latest_path, map_location="cpu", weights_only=False)
        if isinstance(candidate, dict) and candidate.get("component") == "actor-fish":
            model_state = candidate.get("model")
            if not isinstance(model_state, dict) or not archive_metrics_path.is_file():
                raise V175TrainingError(
                    "fish resume exists without its completed archive Actor state"
                )
            controller.actor.load_state_dict(model_state, strict=True)
            controller.actor.to(device)
            return json.loads(archive_metrics_path.read_text(encoding="utf-8"))
    seed_v175()
    controller.to(device)
    controller.fcpg.requires_grad_(False).eval()
    controller.lewm.requires_grad_(False).eval()
    controller.actor.requires_grad_(True).train()
    parameters = list(controller.actor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=ACTOR_ARCHIVE_LR)
    populations = {
        partition: archive.actor_pair_ordinals(
            partition=partition, allow_dev=partition == "dev"
        )
        for partition in ("train", "dev")
    }
    if max_pairs_per_partition is not None:
        populations = {
            key: value[: int(max_pairs_per_partition)]
            for key, value in populations.items()
        }
    total_steps = int(passes) * sum(
        math.ceil(len(value) / int(batch_size)) for value in populations.values()
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(
            step, total=total_steps, warmup=max(1, int(total_steps * 0.05))
        ),
    )
    step = 0
    start_pass = 0
    start_partition = 0
    start_offset = 0
    sums = np.zeros(TRANSITIONS, dtype=np.float64)
    counts = np.zeros(TRANSITIONS, dtype=np.int64)
    restored = _restore_latest(
        output / "latest.pt", controller.actor, optimizer, scheduler, component="actor-archive"
    )
    if restored is not None and max_pairs_per_partition is None:
        step, cursor = restored
        start_pass = int(cursor.get("pass", 0))
        start_partition = int(cursor.get("partition", 0))
        start_offset = int(cursor.get("offset", 0))
        sums = np.asarray(cursor.get("sums", sums), dtype=np.float64)
        counts = np.asarray(cursor.get("counts", counts), dtype=np.int64)
    stage_started = time.monotonic()

    for pass_index in range(start_pass, int(passes)):
        for partition_index, partition in enumerate(("train", "dev")):
            if pass_index == start_pass and partition_index < start_partition:
                continue
            order = localized_actor_order(
                archive,
                populations[partition],
                partition=partition,
                batch_size=batch_size,
                seed=SEED + 1_000 + pass_index * 10 + partition_index,
            )
            offset = (
                start_offset
                if pass_index == start_pass and partition_index == start_partition
                else 0
            )
            for begin in range(offset, len(order), int(batch_size)):
                ordinal = order[begin : begin + int(batch_size)]
                pair = _archive_pair(archive, ordinal, partition)
                optimizer.zero_grad(set_to_none=True)
                commands, target, _ = _actor_step(
                    controller,
                    pair,
                    grounded_bank,
                    feature_bank,
                    device=device,
                )
                loss = actor_objective(
                    commands, target, step_cap=controller.actor.step_cap
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
                optimizer.step()
                scheduler.step()
                step += 1
                per_pair = (
                    (commands.detach().float() - target.float()).abs()
                    / controller.actor.step_cap.float()[None, None]
                ).mean((1, 2)).cpu().numpy()
                for horizon in range(TRANSITIONS):
                    mask = pair.transition == horizon
                    sums[horizon] += float(per_pair[mask].sum())
                    counts[horizon] += int(mask.sum())
                _progress(
                    "actor-archive", step, total_steps,
                    float(loss.detach()), stage_started,
                )
                if step % CHECKPOINT_EVERY == 0:
                    save_component(
                        output / "latest.pt", controller.actor,
                        component="actor-archive", step=step,
                        metrics={"loss": float(sums.sum() / max(1, counts.sum()))},
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cursor={
                            "pass": pass_index,
                            "partition": partition_index,
                            "offset": begin + len(ordinal),
                            "sums": sums.tolist(),
                            "counts": counts.tolist(),
                        },
                    )
            start_offset = 0
        start_partition = 0
    metrics = {
        "phase": "archive",
        "steps": step,
        "passes": int(passes),
        "pairs_per_pass": int(sum(len(value) for value in populations.values())),
        "loss": float(sums.sum() / max(1, counts.sum())),
        "horizon": {
            str(index): float(sums[index] / max(1, counts[index]))
            for index in range(TRANSITIONS)
        },
        "feature_bank": feature_bank.stats(),
    }
    save_component(
        output / "latest.pt", controller.actor,
        component="actor-archive", step=step, metrics=metrics,
        optimizer=optimizer, scheduler=scheduler,
        cursor={
            "pass": int(passes),
            "partition": 0,
            "offset": 0,
            "sums": sums.tolist(),
            "counts": counts.tolist(),
        },
    )
    _write_json(output / "actor-archive-metrics.json", metrics)
    return metrics


def train_v175_fish_actor(
    controller: V175Controller,
    archive: V181FishArchive,
    grounded_bank: V175DeviceGroundedBank,
    feature_bank: V181DeviceFeatureBank,
    output_root: str | Path,
    *,
    device: torch.device,
    batch_size: int = 64,
    updates: int = ACTOR_FISH_UPDATES,
) -> dict[str, Any]:
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    final = output / "actor-final.pt"
    if final.is_file() and updates == ACTOR_FISH_UPDATES:
        payload = load_component(final, controller.actor, component="actor", device=device)
        metrics = _normalize_episode2_role(payload.get("metrics", {}))
        _write_json(output / "actor-metrics.json", metrics)
        return metrics
    controller.to(device)
    controller.fcpg.requires_grad_(False).eval()
    controller.lewm.requires_grad_(False).eval()
    controller.actor.requires_grad_(True).train()
    parameters = list(controller.actor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=ACTOR_FISH_LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(step, total=updates, warmup=min(200, max(1, updates // 10))),
    )
    population = archive.pair_ordinals("train")
    generator = np.random.default_rng(SEED + 2_000)
    loss_sum = 0.0
    start_update = 0
    restored = _restore_latest(
        output / "latest.pt", controller.actor, optimizer, scheduler, component="actor-fish"
    )
    if restored is not None and updates == ACTOR_FISH_UPDATES:
        start_update, cursor = restored
        loss_sum = float(cursor.get("loss_sum", 0.0))
        raw_state = cursor.get("generator_state")
        if isinstance(raw_state, dict):
            generator.bit_generator.state = raw_state
    stage_started = time.monotonic()

    for update in range(start_update, int(updates)):
        ordinal = generator.choice(population, size=batch_size, replace=True)
        fish_pair = archive.pair_batch(ordinal, partition="train")
        pair = _fish_as_temporal_pair(fish_pair)
        bucket = (np.asarray(pair.ordinal) * 37 + SEED * 17 + update * 11) % 100
        view_mask = np.ones((len(pair.ordinal), VIEWS), dtype=np.bool_)
        view_mask[(bucket >= 70) & (bucket < 85), 1] = False
        view_mask[bucket >= 85, 0] = False
        optimizer.zero_grad(set_to_none=True)
        commands, target, _ = _actor_step(
            controller,
            pair,
            grounded_bank,
            feature_bank,
            device=device,
            current_view_mask=torch.from_numpy(view_mask).to(device),
        )
        loss = actor_objective(commands, target, step_cap=controller.actor.step_cap)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()
        loss_sum += float(loss.detach())
        completed = update + 1
        _progress(
            "actor-fish", completed, int(updates),
            loss_sum / completed, stage_started,
        )
        if completed % CHECKPOINT_EVERY == 0:
            save_component(
                output / "latest.pt", controller.actor,
                component="actor-fish", step=completed,
                metrics={"loss": loss_sum / completed},
                optimizer=optimizer,
                scheduler=scheduler,
                cursor={
                    "loss_sum": loss_sum,
                    "generator_state": generator.bit_generator.state,
                },
            )
    metrics = _normalize_episode2_role(
        {
            "phase": "fish",
            "updates": int(updates),
            "train_episodes": [0, 1, 3],
            "loss": loss_sum / max(1, int(updates)),
        }
    )
    save_component(
        final, controller.actor, component="actor", step=int(updates), metrics=metrics
    )
    _write_json(output / "actor-metrics.json", metrics)
    return metrics


def _summary(values: list[np.ndarray]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    merged = np.concatenate(values).astype(np.float64, copy=False)
    return {
        "count": int(len(merged)),
        "mean": float(np.mean(merged)),
        "median": float(np.median(merged)),
        "p90": float(np.percentile(merged, 90)),
    }


def _unordered_two_anchor_epe(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-view EPE and whether the swapped assignment was selected."""

    if predicted.shape != target.shape or predicted.shape[-2:] != (2, 2):
        raise V175TrainingError("two-tip EPE shapes differ")
    if valid.shape != predicted.shape[:-2]:
        raise V175TrainingError("two-tip view validity shape differs")
    direct = (predicted - target).norm(dim=-1).mean(dim=-1)
    swapped = (predicted - target[:, :, (1, 0)]).norm(dim=-1).mean(dim=-1)
    choose = swapped < direct
    epe = torch.minimum(direct, swapped)
    return epe[valid].float().cpu().numpy(), choose[valid].cpu().numpy()


def _representation_statistics(value: torch.Tensor) -> dict[str, float]:
    tensor = value.detach().float().reshape(-1, value.shape[-1])
    if len(tensor) > 2_048:
        index = torch.linspace(0, len(tensor) - 1, 2_048, device=tensor.device).round().long()
        tensor = tensor[index]
    tensor = tensor - tensor.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(tensor)
    probability = singular / singular.sum().clamp_min(1.0e-12)
    effective_rank = torch.exp(
        -(probability * probability.clamp_min(1.0e-12).log()).sum()
    )
    square = singular.square()
    participation = square.sum().square() / square.square().sum().clamp_min(1.0e-12)
    return {
        "effective_rank": float(effective_rank),
        "participation_ratio": float(participation),
        "variance": float(tensor.var(0, unbiased=False).mean()),
        "norm_mean": float(value.detach().float().norm(dim=-1).mean()),
    }


@torch.inference_mode()
def evaluate_v175(
    controller: V175Controller,
    cad: V175CADArchive,
    real_label_cache: str | Path,
    backbone: nn.Module,
    temporal: V181TemporalArchive,
    grounded_bank: V175CombinedGroundedBank,
    fish: V181FishArchive,
    fish_grounded_bank: V175DeviceGroundedBank,
    fish_feature_bank: V181DeviceFeatureBank,
    output_root: str | Path,
    *,
    device: torch.device,
    real_projection: V175AffineProjectionProvider | None = None,
    cad_batch: int = 32,
    temporal_batch: int = 256,
    fish_batch: int = 64,
    limits: Mapping[str, int] | None = None,
    authorize_exposed_development: bool = False,
) -> dict[str, Any]:
    """Frozen panel plus previously exposed development fish episode 2."""

    if authorize_exposed_development is not True:
        raise V175TrainingError(
            "fish episode 2 is exposed development data; evaluation "
            "requires explicit internal authorization"
        )
    limits = dict(limits or {})
    destination = Path(output_root).expanduser().resolve() / "evaluation"
    controller.to(device).eval().requires_grad_(False)
    backbone.eval().requires_grad_(False)
    report: dict[str, Any] = {
        "version": "17.5",
        "fresh_physical_panel": {
            "status": "PENDING_NEW_COLLECTION",
            "optimizer_reuse": False,
        },
    }

    cad_rows = np.asarray(cad.validation_rows, dtype=np.int64)
    if limits.get("cad", 0):
        cad_rows = cad_rows[: limits["cad"]]
    cad_all_error: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_ORDER}
    cad_tip_error: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_ORDER}
    cad_nominal_tip_error: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_ORDER}
    cad_midpoint_error: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_ORDER}
    cad_span_error: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_ORDER}
    cad_axis_error: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_ORDER}
    cad_worse: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_ORDER}
    cad_collapse: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_ORDER}
    cad_entropy: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_ORDER}
    fcpg_probe: dict[str, Any] = {}
    for selected in batches(cad_rows, cad_batch):
        (
            rgb_np,
            state_np,
            _keypoints_base_np,
            camera_from_reference_np,
            intrinsic_np,
            target_np,
            valid_np,
        ) = _cad_arrays(cad, selected)
        rgb = torch.from_numpy(rgb_np).to(device)
        state = torch.from_numpy(state_np).to(device)
        camera_from_reference = torch.from_numpy(camera_from_reference_np).to(device)
        intrinsic = torch.from_numpy(intrinsic_np).to(device)
        target = torch.from_numpy(target_np).to(device)
        valid = torch.from_numpy(valid_np).to(device)
        dino = extract_dino_tensor(backbone, rgb, device=device)
        with _autocast(device):
            measurement = controller.fcpg(
                rgb,
                dino,
                torch.ones(len(selected), VIEWS, dtype=torch.bool, device=device),
                state,
                intrinsic,
                camera_from_reference,
            )
        position = measurement.corrected_keypoints_px
        prior = measurement.prior_keypoints_px
        tip = position[:, :, list(CAD_TIP_INDICES)]
        target_tip = target[:, :, list(CAD_TIP_INDICES)]
        tip_valid = valid[:, :, list(CAD_TIP_INDICES)].all(dim=-1)
        for view, name in enumerate(VIEW_ORDER):
            active_anchor = valid[:, view]
            active_view = active_anchor.any(dim=-1)
            if bool(active_view.any()):
                per_anchor = (position[:, view] - target[:, view]).norm(dim=-1)
                all_error = (per_anchor * active_anchor).sum(dim=-1) / active_anchor.sum(
                    dim=-1
                ).clamp_min(1)
                cad_all_error[name].append(all_error[active_view].cpu().numpy())
                probability = measurement.correspondence_maps["local"][:, view].float()
                entropy = -(
                    probability.clamp_min(1.0e-8)
                    * probability.clamp_min(1.0e-8).log()
                ).sum(-1) / math.log(probability.shape[-1])
                mean_entropy = (entropy * active_anchor).sum(-1) / active_anchor.sum(
                    dim=-1
                ).clamp_min(1)
                cad_entropy[name].append(mean_entropy[active_view].cpu().numpy())
            active_tip = tip_valid[:, view]
            if bool(active_tip.any()):
                tip_epe = (tip[:, view] - target_tip[:, view]).norm(dim=-1).mean(-1)
                nominal_epe = (
                    prior[:, view, list(CAD_TIP_INDICES)] - target_tip[:, view]
                ).norm(dim=-1).mean(-1)
                midpoint = tip[:, view].mean(dim=1)
                target_midpoint = target_tip[:, view].mean(dim=1)
                span = (tip[:, view, 1] - tip[:, view, 0]).norm(dim=-1)
                target_span = (
                    target_tip[:, view, 1] - target_tip[:, view, 0]
                ).norm(dim=-1)
                predicted_axis = tip[:, view, 1] - tip[:, view, 0]
                target_axis = target_tip[:, view, 1] - target_tip[:, view, 0]
                axis_cosine = (
                    (predicted_axis * target_axis).sum(dim=-1)
                    / (predicted_axis.norm(dim=-1) * target_axis.norm(dim=-1)).clamp_min(1.0e-6)
                ).clamp(-1.0, 1.0)
                axis_degrees = torch.rad2deg(torch.acos(axis_cosine))
                cad_tip_error[name].append(tip_epe[active_tip].cpu().numpy())
                cad_nominal_tip_error[name].append(
                    nominal_epe[active_tip].cpu().numpy()
                )
                cad_midpoint_error[name].append(
                    (midpoint - target_midpoint).norm(dim=-1)[active_tip].cpu().numpy()
                )
                cad_span_error[name].append(
                    (span - target_span).abs()[active_tip].cpu().numpy()
                )
                cad_axis_error[name].append(axis_degrees[active_tip].cpu().numpy())
                cad_worse[name].append((tip_epe > nominal_epe)[active_tip].cpu().numpy())
                cad_collapse[name].append((span < 2.0)[active_tip].cpu().numpy())
        if not fcpg_probe:
            fcpg_probe = {
                str(layer): _representation_statistics(
                    measurement.query_tokens_by_layer[:, layer]
                )
                for layer in range(measurement.query_tokens_by_layer.shape[1])
            }
            fcpg_probe["prior_to_corrected_rms_px"] = float(
                (measurement.corrected_keypoints_px - measurement.prior_keypoints_px)
                .float().square().mean().sqrt()
            )
            fcpg_probe["joint_correction_rms_fraction"] = float(
                measurement.next_memory.joint_correction_fraction.float().square().mean().sqrt()
            )
            blank = controller.fcpg(
                torch.zeros_like(rgb),
                torch.zeros_like(dino),
                torch.ones(len(selected), VIEWS, dtype=torch.bool, device=device),
                state,
                intrinsic,
                camera_from_reference,
            )
            fcpg_probe["blank_image_position_response_rms_px"] = float(
                (measurement.corrected_keypoints_px - blank.corrected_keypoints_px)
                .float().square().mean().sqrt()
            )
            if len(selected) > 1:
                donor = controller.fcpg(
                    torch.roll(rgb, 1, 0),
                    torch.roll(dino, 1, 0),
                    torch.ones(len(selected), VIEWS, dtype=torch.bool, device=device),
                    state,
                    intrinsic,
                    camera_from_reference,
                )
                shuffled_q = controller.fcpg(
                    rgb,
                    dino,
                    torch.ones(len(selected), VIEWS, dtype=torch.bool, device=device),
                    torch.roll(state, 1, 0),
                    intrinsic,
                    camera_from_reference,
                )
                fcpg_probe["image_donor_position_response_rms_px"] = float(
                    (measurement.corrected_keypoints_px - donor.corrected_keypoints_px)
                    .float().square().mean().sqrt()
                )
                fcpg_probe["q_shuffle_position_response_rms_px"] = float(
                    (measurement.corrected_keypoints_px - shuffled_q.corrected_keypoints_px)
                    .float().square().mean().sqrt()
                )
            probe_rows = min(4, len(selected))
            jacobian_cosines: list[float] = []
            for joint in range(STATE_DIM):
                plus_q = state[:probe_rows].clone()
                minus_q = state[:probe_rows].clone()
                plus_q[:, joint] += 0.25
                minus_q[:, joint] -= 0.25
                plus = controller.fcpg(
                    rgb[:probe_rows], dino[:probe_rows],
                    torch.ones(probe_rows, VIEWS, dtype=torch.bool, device=device),
                    plus_q, intrinsic[:probe_rows], camera_from_reference[:probe_rows],
                )
                minus = controller.fcpg(
                    rgb[:probe_rows], dino[:probe_rows],
                    torch.ones(probe_rows, VIEWS, dtype=torch.bool, device=device),
                    minus_q, intrinsic[:probe_rows], camera_from_reference[:probe_rows],
                )
                learned = plus.corrected_keypoints_px - minus.corrected_keypoints_px
                analytic = plus.prior_keypoints_px - minus.prior_keypoints_px
                cosine = F.cosine_similarity(learned.float(), analytic.float(), dim=-1)
                active = plus.anchor_valid & minus.anchor_valid & (analytic.norm(dim=-1) > 1.0e-4)
                if bool(active.any()):
                    jacobian_cosines.append(float(cosine[active].mean()))
            fcpg_probe["learned_vs_analytic_pixel_jacobian_cosine"] = (
                float(np.mean(jacobian_cosines)) if jacobian_cosines else None
            )
    report["fcpg_cad_validation"] = {
        "rows": int(len(cad_rows)),
        "all_anchor_epe_px": {
            name: _summary(values) for name, values in cad_all_error.items()
        },
        "fm_epe_px": {name: _summary(values) for name, values in cad_tip_error.items()},
        "nominal_fm_epe_px": {
            name: _summary(values) for name, values in cad_nominal_tip_error.items()
        },
        "midpoint_epe_px": {
            name: _summary(values) for name, values in cad_midpoint_error.items()
        },
        "span_absolute_error_px": {
            name: _summary(values) for name, values in cad_span_error.items()
        },
        "axis_error_degrees": {
            name: _summary(values) for name, values in cad_axis_error.items()
        },
        "measurement_entropy": {
            name: _summary(values) for name, values in cad_entropy.items()
        },
        "worse_than_nominal_fraction": {
            name: float(np.concatenate(values).mean()) if values else None
            for name, values in cad_worse.items()
        },
        "collapse_fraction_below_2px": {
            name: float(np.concatenate(values).mean()) if values else None
            for name, values in cad_collapse.items()
        },
        "ordered_identity": True,
    }

    labels = _real_label_arrays(
        real_label_cache,
        backbone,
        device=device,
        projection_provider=real_projection,
    )
    with _autocast(device):
        real_measurement = controller.fcpg(
            labels["rgb"],
            labels["dino"],
            labels["valid"],
            labels["current_q"],
            None,
            None,
            cached_nominal_keypoints_px=labels["nominal_all"],
            cached_nominal_valid=labels["nominal_valid"],
            cached_nominal_jacobian_px=labels["nominal_jacobian"],
            joint_state_valid=labels["q_valid"],
        )
    real_position = real_measurement.fm_px
    real_errors: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_ORDER}
    real_swaps: list[np.ndarray] = []
    for view, name in enumerate(VIEW_ORDER):
        active = labels["valid"][:, view]
        direct = (real_position[:, view] - labels["target"][:, view]).norm(dim=-1).mean(-1)
        swapped = (
            real_position[:, view] - labels["target"][:, view, (1, 0)]
        ).norm(dim=-1).mean(-1)
        selected_error = torch.where(
            labels["ordered"], direct, torch.minimum(direct, swapped)
        )
        real_errors[name].append(selected_error[active].cpu().numpy())
        real_swaps.append((swapped < direct)[active].cpu().numpy())
    report["fcpg_real_training_fit"] = {
        "cache_rows": int(len(labels["rgb"])),
        "epe_px": {name: _summary(values) for name, values in real_errors.items()},
        "set_assignment_swap_fraction": float(np.concatenate(real_swaps).mean()),
        "is_unseen_evidence": False,
    }

    lewm_sums = {
        "factual_visual": 0.0,
        "shuffled_visual": 0.0,
        "persistence_visual": 0.0,
        "factual_state": 0.0,
        "direct_command_state": 0.0,
        "servo_lag_state": 0.0,
    }
    lewm_count = 0
    lewm_probe: dict[str, Any] = {}
    maximum = limits.get("temporal", 0)
    for transition in range(TRANSITIONS):
        for partition in ("train", "dev"):
            rows = temporal.rows_for_partition(partition)
            if maximum:
                rows = rows[:maximum]
            for selected in batches(rows, temporal_batch):
                batch = temporal.transition_batch(
                    selected, transition, partition=partition, allow_dev=partition == "dev"
                )
                source, source_valid = grounded_bank.get(batch.source_endpoint_index)
                target, target_valid = grounded_bank.get(batch.target_endpoint_index)
                current_q = torch.from_numpy(
                    restore_companion_joint_units(batch.current_q)
                ).to(device)
                commands = torch.from_numpy(
                    restore_companion_joint_units(batch.commands)
                ).to(device)
                measured = torch.from_numpy(
                    restore_companion_joint_units(batch.measured_q_path)
                ).to(device)
                with _autocast(device):
                    factual = controller.lewm(source, source_valid, current_q, commands)
                    shuffled = controller.lewm(
                        source, source_valid, current_q, torch.roll(commands, 1, 0)
                    )
                _, factual_parts = lewm_objective(
                    factual, target, source_valid, target_valid, measured,
                    step_cap=controller.actor.step_cap,
                )
                _, shuffled_parts = lewm_objective(
                    shuffled, target, source_valid, target_valid, measured,
                    step_cap=controller.actor.step_cap,
                )
                persistence_visual, _, _ = _matched_anchor_losses(
                    source, target, source_valid, target_valid
                )
                direct_path = torch.cat((current_q[:, None], commands), dim=1)
                lag = current_q
                lag_rows = [lag]
                for command_step in range(COMMAND_STEPS):
                    lag = lag + 0.65 * (commands[:, command_step] - lag)
                    lag_rows.append(lag)
                lag_path = torch.stack(lag_rows, dim=1)
                cap = controller.actor.step_cap.float()[None, None]
                direct_state = F.smooth_l1_loss((direct_path - measured) / cap, torch.zeros_like(measured))
                lag_state = F.smooth_l1_loss((lag_path - measured) / cap, torch.zeros_like(measured))
                count = len(selected)
                lewm_sums["factual_visual"] += float(factual_parts["visual"]) * count
                lewm_sums["shuffled_visual"] += float(shuffled_parts["visual"]) * count
                lewm_sums["persistence_visual"] += float(persistence_visual) * count
                lewm_sums["factual_state"] += float(factual_parts["state"]) * count
                lewm_sums["direct_command_state"] += float(direct_state) * count
                lewm_sums["servo_lag_state"] += float(lag_state) * count
                lewm_count += count
                if not lewm_probe:
                    lewm_probe = {
                        str(layer): _representation_statistics(factual.tokens_by_layer[:, layer])
                        for layer in range(factual.tokens_by_layer.shape[1])
                    }
                    lewm_probe["shuffle_anchor_response_rms"] = float(
                        (factual.predicted_anchor_tokens - shuffled.predicted_anchor_tokens)
                        .float().square().mean().sqrt()
                    )
                    lewm_probe["shuffle_q_response_rms"] = float(
                        (factual.predicted_q_path - shuffled.predicted_q_path)
                        .float().square().mean().sqrt()
                    )
    report["lewm_archive_training_fit"] = {
        "examples": lewm_count,
        **{name: value / max(1, lewm_count) for name, value in lewm_sums.items()},
        "is_held_evidence": False,
    }

    _append_exposed_development_access(
        destination,
        event="evaluation_started",
        smoke=bool(limits),
    )
    held = fish.pair_ordinals("held")
    if limits.get("fish", 0):
        held = held[: limits["fish"]]
    errors: dict[str, list[np.ndarray]] = {
        "factual": [], "innovation_zero": [], "history_zero": [],
        "prefix_0": [], "prefix_1": [], "prefix_2": [], "prefix_3": [],
        "prefix_4": [],
        "wrist_only": [], "column_only": [],
    }
    actor_probe: dict[str, Any] = {}
    for ordinal in batches(held, fish_batch):
        pair = _fish_as_temporal_pair(
            fish.pair_batch(ordinal, partition="held", allow_held=True)
        )
        outputs: dict[str, torch.Tensor] = {}
        factual, target, trace = _actor_step(
            controller, pair, fish_grounded_bank, fish_feature_bank, device=device
        )
        outputs["factual"] = factual
        outputs["innovation_zero"] = _actor_step(
            controller, pair, fish_grounded_bank, fish_feature_bank,
            device=device, innovation_zero=True,
        )[0]
        for prefix in range(HISTORY_CHUNKS + 1):
            outputs[f"prefix_{prefix}"] = _actor_step(
                controller, pair, fish_grounded_bank, fish_feature_bank,
                device=device, history_prefix=prefix,
            )[0]
        outputs["history_zero"] = outputs["prefix_0"]
        wrist = torch.tensor((True, False), device=device)[None].expand(len(ordinal), -1)
        column = torch.tensor((False, True), device=device)[None].expand(len(ordinal), -1)
        outputs["wrist_only"] = _actor_step(
            controller, pair, fish_grounded_bank, fish_feature_bank,
            device=device, current_view_mask=wrist,
        )[0]
        outputs["column_only"] = _actor_step(
            controller, pair, fish_grounded_bank, fish_feature_bank,
            device=device, current_view_mask=column,
        )[0]
        for name, commands in outputs.items():
            per_pair = (
                (commands.float() - target.float()).abs()
                / controller.actor.step_cap.float()[None, None]
            ).mean((1, 2)).cpu().numpy()
            errors[name].append(per_pair)
        if not actor_probe:
            actor_probe = {
                str(layer): _representation_statistics(trace.action_queries_by_layer[:, layer])
                for layer in range(trace.action_queries_by_layer.shape[1])
            }
            actor_probe["innovation_action_response_rms"] = float(
                (outputs["factual"] - outputs["innovation_zero"]).float().square().mean().sqrt()
            )
            actor_probe["history_action_response_rms"] = float(
                (outputs["factual"] - outputs["history_zero"]).float().square().mean().sqrt()
            )
            actor_probe["wrist_drop_response_rms"] = float(
                (outputs["factual"] - outputs["column_only"]).float().square().mean().sqrt()
            )
            actor_probe["column_drop_response_rms"] = float(
                (outputs["factual"] - outputs["wrist_only"]).float().square().mean().sqrt()
            )
    report["actor_fish_episode_2_exposed_development"] = {
        "pairs": int(len(held)),
        "normalized_action_error": {
            name: _summary(value) for name, value in errors.items()
        },
        "data_role": "descriptive_only_previously_exposed",
        "optimizer_excluded": True,
        "checkpoint_selection": False,
        "strict_held_evidence": False,
    }
    report["layer_probes"] = {
        "fcpg": fcpg_probe,
        "lewm": lewm_probe,
        "actor": actor_probe,
        "attention_is_causal_evidence": False,
    }
    _write_json(destination / "metrics.json", report)
    _append_exposed_development_access(
        destination,
        event="evaluation_completed",
        smoke=bool(limits),
    )
    return report


__all__ = [
    "ACTOR_ARCHIVE_PASSES",
    "ACTOR_FISH_UPDATES",
    "FCPG_CAD_PASSES",
    "FCPG_REAL_UPDATES",
    "SEED",
    "V175AllRealRGB",
    "V175AffineProjectionProvider",
    "V175CADArchive",
    "V175CombinedFeatureBank",
    "V175CombinedGroundedBank",
    "V175DeviceGroundedBank",
    "V175GroundedCache",
    "V175FishEndpointStateProvider",
    "V175TemporalEndpointStateProvider",
    "V175TrainingError",
    "actor_objective",
    "batches",
    "build_actor_conditioning",
    "configure_cuda",
    "cosine_schedule",
    "extract_dino_tensor",
    "feature_distillation_objective",
    "evaluate_v175",
    "lewm_objective",
    "load_component",
    "load_frozen_dinov2",
    "prepare_fish_dino_cache",
    "prepare_grounded_cache",
    "prepare_temporal_dino_cache",
    "save_component",
    "seed_v175",
    "train_v175_archive_actor",
    "train_v175_fcpg",
    "train_v175_fish_actor",
    "train_v175_lewm",
]
