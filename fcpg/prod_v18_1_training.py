"""Caching and fixed training loops for Prod-V18.1.

The frozen DINOv2 backbone is used only while extracting ordinary fp16 patch
tokens.  Training operates on those tokens with a fixed-size GPU page bank, so
eviction reuses already allocated storage instead of growing the CUDA allocator.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from itertools import islice
import math
import mmap
from pathlib import Path
import random
import shutil
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fcpg.isolated_video_decode import IsolatedVideoDecodePool
from fcpg.prod_v18_1 import (
    COMMAND_STEPS,
    HISTORY_CHUNKS,
    MODEL_DIM,
    OBSERVER_DIM,
    OBSERVER_SLOTS,
    PATH_STEPS,
    STATE_DIM,
    VIEWS,
    VIT_DIM,
    VIT_PATCHES,
    V181Config,
    V181Innovation,
    V181LeWMPrediction,
    V181Memory,
    V181Measurement,
    V181TokenFCPG,
    V181TipFreeController,
    compute_v181_innovation,
    empty_v181_innovation,
    normalize_observer_slots,
    predicted_observer_slots,
    project_v181_action_trajectory,
)
from fcpg.prod_v18_1_data import (
    TRANSITIONS,
    V181DataContractError,
    V181FeatureCache,
    V181FishArchive,
    V181FishPair,
    V181TemporalArchive,
    V181TemporalActorBatch,
    deterministic_fish_view_valid,
)


SEEDS = (18_000, 18_001, 18_002)
FISH_VARIANTS = (
    "q_only",
    "dense_only",
    "observer_innovation_only",
    "current_only",
    "visual_history",
    "proprio_history",
    "current_innovation",
    "actual_history",
    "temporal_delta",
    "lewm_transition",
    "servo_only",
    "servo_lag",
    "visual_only",
    "state_only",
    "full",
)
PRIMARY_MATCHED_VARIANTS = (
    "q_only",
    "current_only",
    "visual_history",
    "proprio_history",
    "current_innovation",
    "actual_history",
    "state_only",
    "full",
)
WORLD_PASSES = TRANSITIONS
WORLD_EFFECTIVE_BATCH = 256
WORLD_LR = 1.0e-4
WORLD_WARMUP_FRACTION = 0.05
ARCHIVE_ACTOR_PASSES = 4
ARCHIVE_ACTOR_BATCH = 256
ARCHIVE_ACTOR_LR = 1.0e-4
DENSE_ADAPTATION_PASSES = 8
DENSE_ADAPTATION_BATCH = 128
DENSE_ACTOR_LR = 5.0e-5
FISH_UPDATES = 4_000
FISH_EFFECTIVE_BATCH = 64
FISH_WARMUP_UPDATES = 200
FCPG_LR = 3.0e-5
ACTOR_LR = 1.0e-4
LEWM_LR = 5.0e-5
GRADIENT_CLIP = 1.0
CHECKPOINT_EVERY = 500
GPU_PAGE_BANK_BYTES = 48 * 1024**3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class V181TrainingError(RuntimeError):
    """Raised before a partial or contract-violating V18.1 run proceeds."""


def _flush_and_release_pages(*arrays: np.memmap) -> None:
    """Persist cache writes, then release clean mapped pages from process RSS."""

    for array in arrays:
        array.flush()
    advice = getattr(mmap, "MADV_DONTNEED", None)
    if advice is None:
        return
    for array in arrays:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None and hasattr(mapping, "madvise"):
            mapping.madvise(advice)


@dataclass(frozen=True)
class V181FishConditioning:
    memory: V181Memory
    innovation: V181Innovation
    predicted_missing_slots: torch.Tensor
    prediction_valid: torch.Tensor
    predicted_transition: torch.Tensor
    servo_error: torch.Tensor
    servo_lag_error: torch.Tensor


def fixed_servo_lag_path(
    initial_q: torch.Tensor,
    commands: torch.Tensor,
    *,
    gain: float = 0.65,
) -> torch.Tensor:
    """A parameter-free first-order tracking baseline for attribution."""

    if commands.shape[:-2] != initial_q.shape[:-1] or commands.shape[-2:] != (
        COMMAND_STEPS,
        STATE_DIM,
    ):
        raise V181TrainingError("servo-lag inputs must align [...,6] and [...,5,6]")
    if not 0.0 < float(gain) <= 1.0:
        raise V181TrainingError("servo-lag gain must be in (0,1]")
    state = initial_q
    path = [state]
    for step in range(COMMAND_STEPS):
        state = state + float(gain) * (commands[..., step, :] - state)
        path.append(state)
    return torch.stack(path, dim=-2)


def seed_v181(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint_payload(
    model: V181TipFreeController,
    *,
    phase: str,
    seed: int,
    step: int,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": "18.1",
        "phase": str(phase),
        "seed": int(seed),
        "step": int(step),
        "config": asdict(model.config),
        "model": model.state_dict(),
        "metrics": dict(metrics),
    }
    if training_state is not None:
        payload["training_state"] = dict(training_state)
    return payload


def save_v181_checkpoint(
    path: str | Path,
    model: V181TipFreeController,
    *,
    phase: str,
    seed: int,
    step: int,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        _checkpoint_payload(
            model,
            phase=phase,
            seed=seed,
            step=step,
            metrics=metrics,
            training_state=training_state,
        ),
        temporary,
    )
    temporary.replace(destination)
    return destination


def load_v181_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[V181TipFreeController, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise V181TrainingError(f"V18.1 checkpoint is absent: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("version") != "18.1":
        raise V181TrainingError("checkpoint is not an independent V18.1 artifact")
    config_raw = payload.get("config")
    if not isinstance(config_raw, dict) or not isinstance(payload.get("model"), dict):
        raise V181TrainingError("V18.1 checkpoint fields are incomplete")
    config = V181Config(**config_raw)
    config.validate()
    model = V181TipFreeController(config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    return model, payload


def _resume_payload(
    path: Path,
    model: V181TipFreeController,
    *,
    phase: str,
    seed: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("version") != "18.1":
        raise V181TrainingError(f"resume artifact is not V18.1: {path}")
    if payload.get("phase") != phase or int(payload.get("seed", -1)) != int(seed):
        raise V181TrainingError(f"resume phase/seed differs: {path}")
    state = payload.get("model")
    training_state = payload.get("training_state")
    if not isinstance(state, dict) or not isinstance(training_state, dict):
        raise V181TrainingError(f"resume artifact lacks exact loop state: {path}")
    model.load_state_dict(state, strict=True)
    return payload


def load_frozen_dinov2(
    repository: str | Path,
    weights: str | Path,
    *,
    device: torch.device | str,
) -> nn.Module:
    """Load the official local DINOv2-S/14 implementation without networking."""

    repo = Path(repository).expanduser().resolve()
    checkpoint = Path(weights).expanduser().resolve()
    if not (repo / "hubconf.py").is_file():
        raise V181TrainingError(f"official DINOv2 repository is absent: {repo}")
    if not checkpoint.is_file():
        raise V181TrainingError(f"DINOv2-S/14 weights are absent: {checkpoint}")
    model = torch.hub.load(
        str(repo), "dinov2_vits14", source="local", pretrained=False
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise V181TrainingError("DINOv2-S/14 weight file is malformed")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    return model


def _rgb_tensor(rgb: np.ndarray, *, device: torch.device) -> torch.Tensor:
    frames = np.asarray(rgb)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise V181TrainingError("DINO input must be uint8 RGB [N,H,W,3]")
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).to(
        device=device, non_blocking=True
    )
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    mean = tensor.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[None, :, None, None]
    return (tensor - mean) / std


@torch.inference_mode()
def extract_dinov2_tokens(
    backbone: nn.Module,
    rgb: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    if batch_size <= 0:
        raise V181TrainingError("DINO extraction batch must be positive")
    outputs: list[np.ndarray] = []
    for start in range(0, len(rgb), int(batch_size)):
        pixels = _rgb_tensor(rgb[start : start + int(batch_size)], device=device)
        context = nullcontext()
        if device.type == "cuda":
            # V18.1 was trained on a bf16-capable RTX 6000D, while the local
            # robot host uses V100s.  DINO inference is identical in shape and
            # normalization there, but V100 has no native bf16 execution.
            autocast_dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability(device)[0] >= 8
                else torch.float16
            )
            context = torch.autocast(
                device_type="cuda", dtype=autocast_dtype
            )
        with context:
            features = backbone.forward_features(pixels)
        if not isinstance(features, dict) or "x_norm_patchtokens" not in features:
            raise V181TrainingError("DINOv2 did not expose normalized patch tokens")
        tokens = features["x_norm_patchtokens"]
        if tokens.shape != (len(pixels), VIT_PATCHES, VIT_DIM):
            raise V181TrainingError(
                f"DINOv2 patch shape drifted: observed={list(tokens.shape)}"
            )
        if not bool(torch.isfinite(tokens).all()):
            raise V181TrainingError("DINOv2 produced non-finite tokens")
        outputs.append(tokens.float().cpu().numpy().astype(np.float16))
    return np.concatenate(outputs, axis=0)


def _create_feature_arrays(
    output: Path,
    endpoint_index: np.ndarray,
    view_valid: np.ndarray,
    *,
    kind: str,
) -> tuple[np.memmap, np.memmap]:
    output.mkdir(parents=True, exist_ok=True)
    endpoint = np.asarray(endpoint_index, dtype=np.int64)
    valid = np.asarray(view_valid, dtype=np.bool_)
    if endpoint.ndim != 1 or len(endpoint) == 0 or bool(
        (np.diff(endpoint) <= 0).any()
    ):
        raise V181TrainingError("cache endpoint identities must be sorted unique")
    if valid.shape != (len(endpoint), VIEWS):
        raise V181TrainingError("cache view-valid shape drifted")
    required = int(len(endpoint) * VIEWS * VIT_PATCHES * VIT_DIM * 2)
    endpoint_path = output / "endpoint_index.npy"
    valid_path = output / "view_valid.npy"
    if endpoint_path.is_file() or valid_path.is_file():
        if not endpoint_path.is_file() or not valid_path.is_file():
            raise V181TrainingError("resumed cache identity files are incomplete")
        previous_endpoint = np.load(endpoint_path, mmap_mode="r", allow_pickle=False)
        previous_valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
        if not bool(np.array_equal(previous_endpoint, endpoint)):
            raise V181TrainingError("resumed cache endpoint identities differ")
        if not bool(np.array_equal(previous_valid, valid)):
            raise V181TrainingError("resumed cache view validity differs")
    else:
        np.save(endpoint_path, endpoint, allow_pickle=False)
        np.save(valid_path, valid, allow_pickle=False)
    token_path = output / "tokens.npy"
    done_path = output / "done.npy"
    if not token_path.is_file():
        available = shutil.disk_usage(output).free
        if available < required + 8 * 1024**3:
            raise V181TrainingError(
                f"feature cache needs {required} bytes plus workspace; available={available}"
            )
    if token_path.is_file() and done_path.is_file():
        tokens = np.lib.format.open_memmap(token_path, mode="r+")
        done = np.lib.format.open_memmap(done_path, mode="r+")
        if tokens.shape != (len(endpoint), VIEWS, VIT_PATCHES, VIT_DIM):
            raise V181TrainingError("resumed token cache shape drifted")
        if done.shape != (len(endpoint), VIEWS):
            raise V181TrainingError("resumed cache completion shape drifted")
    else:
        tokens = np.lib.format.open_memmap(
            token_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(endpoint), VIEWS, VIT_PATCHES, VIT_DIM),
        )
        # open_memmap extends a new file with zero-filled pages.  Do not touch
        # all ~194 GiB here: only decoded, valid rows are written below.
        done = np.lib.format.open_memmap(
            done_path, mode="w+", dtype=np.bool_, shape=(len(endpoint), VIEWS)
        )
        done[:] = ~valid
        _flush_and_release_pages(tokens, done)
    _json_write(
        output / "config.json",
        {
            "version": "18.1",
            "kind": str(kind),
            "backbone": "dinov2_vits14",
            "view_order": list(("wrist", "column")),
            "patch_grid": [16, 16],
            "token_dim": VIT_DIM,
            "dtype": "float16",
            "rows": int(len(endpoint)),
            "complete": bool(done.all()),
        },
    )
    return tokens, done


@dataclass(frozen=True)
class _DecodeJob:
    payload: tuple[str, tuple[int, ...], int, int]
    locations: tuple[tuple[tuple[int, int], ...], ...]


def _decode_jobs(
    pool: IsolatedVideoDecodePool,
    jobs: Sequence[_DecodeJob],
    *,
    timeout: float,
) -> list[tuple[_DecodeJob, np.ndarray]]:
    decoded = pool.map_ordered((job.payload for job in jobs), timeout=timeout)
    result: list[tuple[_DecodeJob, np.ndarray]] = []
    for job, value in zip(jobs, decoded, strict=True):
        path_text, frames, rgb = value
        if str(path_text) != job.payload[0] or not bool(
            np.array_equal(frames, np.asarray(job.payload[1], dtype=np.int64))
        ):
            raise V181TrainingError("video decoder returned a misordered job")
        result.append((job, np.asarray(rgb, dtype=np.uint8)))
    return result


def prepare_temporal_dino_cache(
    archive: V181TemporalArchive,
    output_root: str | Path,
    backbone: nn.Module,
    *,
    device: torch.device,
    partition: str = "train",
    decode_workers: int = 4,
    decode_frames_per_job: int = 512,
    decoder_threads: int = 2,
    dino_batch_size: int = 512,
    timeout_seconds: float = 3_600.0,
) -> Path:
    """Decode each unique TRAIN endpoint/view and cache full patch tokens."""

    if partition not in {"train", "dev"}:
        raise V181TrainingError("temporal feature partition must be train or dev")
    rows = archive.rows_for_partition(partition)  # DEV is cacheable only by explicit call.
    endpoints = np.unique(archive.row_endpoint_index[rows].reshape(-1)).astype(np.int64)
    valid = np.asarray(archive.endpoint_view_valid[endpoints], dtype=np.bool_)
    output = Path(output_root).expanduser().resolve()
    tokens, done = _create_feature_arrays(
        output, endpoints, valid, kind=f"temporal-{partition}"
    )
    endpoint_row = {int(endpoint): row for row, endpoint in enumerate(endpoints.tolist())}
    grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    block = 8_192
    for start in range(0, len(endpoints), block):
        for endpoint, view, path, frame, is_valid in archive.media_requests(
            endpoints[start : start + block]
        ):
            row = endpoint_row[int(endpoint)]
            if is_valid and not bool(done[row, view]):
                grouped[str(path)].append((int(frame), row, int(view)))

    def jobs() -> Iterator[_DecodeJob]:
        for path_text in sorted(grouped):
            records = sorted(grouped[path_text])
            by_frame: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for frame, row, view in records:
                by_frame[frame].append((row, view))
            frames = sorted(by_frame)
            for start in range(0, len(frames), int(decode_frames_per_job)):
                selected = tuple(frames[start : start + int(decode_frames_per_job)])
                yield _DecodeJob(
                    payload=(path_text, selected, 224, int(decoder_threads)),
                    locations=tuple(tuple(by_frame[frame]) for frame in selected),
                )

    jobs_total = sum(
        math.ceil(len({frame for frame, _, _ in records}) / int(decode_frames_per_job))
        for records in grouped.values()
    )
    bundles_total = math.ceil(jobs_total / int(decode_workers)) if jobs_total else 0

    def bundles() -> Iterator[list[_DecodeJob]]:
        stream = iter(jobs())
        while True:
            bundle = list(islice(stream, int(decode_workers)))
            if not bundle:
                return
            yield bundle

    with IsolatedVideoDecodePool(workers=int(decode_workers)) as pool:
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            bundle_stream = iter(bundles())
            bundle = next(bundle_stream, None)
            future = (
                prefetch.submit(_decode_jobs, pool, bundle, timeout=timeout_seconds)
                if bundle is not None
                else None
            )
            index = 0
            while future is not None:
                decoded = future.result()
                next_bundle = next(bundle_stream, None)
                next_future = (
                    prefetch.submit(
                        _decode_jobs, pool, next_bundle, timeout=timeout_seconds
                    )
                    if next_bundle is not None
                    else None
                )
                # Encoding this bundle overlaps the next bundle's CPU decode.
                for job, rgb in decoded:
                    encoded = extract_dinov2_tokens(
                        backbone, rgb, device=device, batch_size=dino_batch_size
                    )
                    rows_to_write: list[int] = []
                    views_to_write: list[int] = []
                    features_to_write: list[int] = []
                    for offset, locations in enumerate(job.locations):
                        for row, view in locations:
                            rows_to_write.append(row)
                            views_to_write.append(view)
                            features_to_write.append(offset)
                    if rows_to_write:
                        cache_rows = np.asarray(rows_to_write, dtype=np.int64)
                        cache_views = np.asarray(views_to_write, dtype=np.int64)
                        feature_rows = np.asarray(features_to_write, dtype=np.int64)
                        order = np.lexsort((cache_views, cache_rows))
                        tokens[cache_rows[order], cache_views[order]] = encoded[
                            feature_rows[order]
                        ]
                        done[cache_rows, cache_views] = True
                _flush_and_release_pages(tokens, done)
                _json_write(
                    output / "progress.json",
                    {
                        "version": "18.1",
                        "partition": partition,
                        "bundles_completed": index + 1,
                        "bundles_total": bundles_total,
                        "view_slots_completed": int(done.sum()),
                        "view_slots_total": int(done.size),
                    },
                )
                future = next_future
                index += 1
    if not bool(done.all()):
        raise V181TrainingError("temporal DINO cache finished with missing views")
    config = dict(
        json.loads((output / "config.json").read_text(encoding="utf-8"))
    )
    config["complete"] = True
    _json_write(output / "config.json", config)
    return output


def prepare_fish_dino_cache(
    archive: V181FishArchive,
    output_root: str | Path,
    backbone: nn.Module,
    *,
    device: torch.device,
    dino_batch_size: int = 512,
) -> Path:
    endpoints = np.arange(archive.frames, dtype=np.int64)
    valid = np.ones((archive.frames, VIEWS), dtype=np.bool_)
    output = Path(output_root).expanduser().resolve()
    tokens, done = _create_feature_arrays(output, endpoints, valid, kind="fish")
    pending = np.flatnonzero(~done.all(axis=1))
    for start in range(0, len(pending), int(dino_batch_size)):
        selected = pending[start : start + int(dino_batch_size)]
        rgb = archive.rgb(selected).reshape(-1, 224, 224, 3)
        encoded = extract_dinov2_tokens(
            backbone, rgb, device=device, batch_size=dino_batch_size
        ).reshape(len(selected), VIEWS, VIT_PATCHES, VIT_DIM)
        tokens[selected] = encoded
        done[selected] = True
        _flush_and_release_pages(tokens, done)
    config = dict(
        json.loads((output / "config.json").read_text(encoding="utf-8"))
    )
    config["complete"] = bool(done.all())
    _json_write(output / "config.json", config)
    return output


class V181DeviceFeatureBank:
    """Fixed-allocation GPU page cache indexed by unique endpoint identity."""

    def __init__(
        self,
        cache: V181FeatureCache,
        *,
        device: torch.device,
        max_bytes: int = GPU_PAGE_BANK_BYTES,
        page_rows: int = 512,
    ) -> None:
        if device.type != "cuda":
            raise V181TrainingError("the fixed feature bank requires a CUDA device")
        if max_bytes <= 0 or page_rows <= 0:
            raise V181TrainingError("feature bank budget and page size must be positive")
        self.cache = cache
        self.device = device
        self.page_rows = int(page_rows)
        bytes_per_page = (
            self.page_rows * VIEWS * VIT_PATCHES * VIT_DIM * 2
            + self.page_rows * VIEWS
        )
        required_pages = math.ceil(len(cache.endpoint_index) / self.page_rows)
        self.slots = min(
            required_pages,
            max(1, int(max_bytes) // bytes_per_page),
        )
        self.token_bank = torch.empty(
            self.slots,
            self.page_rows,
            VIEWS,
            VIT_PATCHES,
            VIT_DIM,
            dtype=torch.float16,
            device=device,
        )
        self.valid_bank = torch.empty(
            self.slots,
            self.page_rows,
            VIEWS,
            dtype=torch.bool,
            device=device,
        )
        # One reusable pinned page keeps host memory bounded. Creating a fresh
        # pinned tensor for every cache miss lets the CUDA host allocator retain
        # many 200 MiB pages while a scattered batch is being loaded.
        self.host_token_page = torch.empty(
            self.page_rows,
            VIEWS,
            VIT_PATCHES,
            VIT_DIM,
            dtype=torch.float16,
            pin_memory=True,
        )
        self.host_valid_page = torch.empty(
            self.page_rows,
            VIEWS,
            dtype=torch.bool,
            pin_memory=True,
        )
        self.resident: OrderedDict[int, int] = OrderedDict()
        self.free_slots = list(range(self.slots - 1, -1, -1))
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.page_loads = 0

    @property
    def allocated_bytes(self) -> int:
        return int(
            self.token_bank.numel() * self.token_bank.element_size()
            + self.valid_bank.numel() * self.valid_bank.element_size()
        )

    def _load_page(self, page: int) -> int:
        if page in self.resident:
            slot = self.resident.pop(page)
            self.resident[page] = slot
            self.hits += 1
            return slot
        self.misses += 1
        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            _, slot = self.resident.popitem(last=False)
            self.evictions += 1
        start = page * self.page_rows
        stop = min(start + self.page_rows, len(self.cache.endpoint_index))
        count = stop - start
        np.copyto(
            self.host_token_page[:count].numpy(),
            np.asarray(self.cache.tokens[start:stop]),
            casting="no",
        )
        np.copyto(
            self.host_valid_page[:count].numpy(),
            np.asarray(self.cache.view_valid[start:stop]),
            casting="no",
        )
        # Synchronous reuse is intentional: the next miss may overwrite the
        # sole staging page, so completion is bounded before that happens.
        self.token_bank[slot, :count].copy_(
            self.host_token_page[:count], non_blocking=False
        )
        self.valid_bank[slot, :count].copy_(
            self.host_valid_page[:count], non_blocking=False
        )
        if count < self.page_rows:
            self.token_bank[slot, count:].zero_()
            self.valid_bank[slot, count:].zero_()
        self.page_loads += 1
        if self.page_loads % 8 == 0:
            _flush_and_release_pages(self.cache.tokens, self.cache.view_valid)
        self.resident[page] = slot
        return slot

    def get(self, endpoint_indices: Any) -> tuple[torch.Tensor, torch.Tensor]:
        requested = np.asarray(endpoint_indices, dtype=np.int64)
        shape = requested.shape
        cache_rows = self.cache.rows_for(requested.reshape(-1))
        pages = cache_rows // self.page_rows
        offsets = cache_rows % self.page_rows
        slots = np.empty_like(pages)
        for page in np.unique(pages):
            slot = self._load_page(int(page))
            slots[pages == page] = slot
        slot_tensor = torch.from_numpy(slots).to(self.device)
        offset_tensor = torch.from_numpy(offsets).to(self.device)
        tokens = self.token_bank[slot_tensor, offset_tensor]
        valid = self.valid_bank[slot_tensor, offset_tensor]
        return (
            tokens.reshape(*shape, VIEWS, VIT_PATCHES, VIT_DIM),
            valid.reshape(*shape, VIEWS),
        )

    def stats(self) -> dict[str, int]:
        return {
            "allocated_bytes": self.allocated_bytes,
            "host_staging_bytes": int(
                self.host_token_page.numel() * self.host_token_page.element_size()
                + self.host_valid_page.numel() * self.host_valid_page.element_size()
            ),
            "resident_pages": len(self.resident),
            "page_slots": self.slots,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


class V181DeviceObserverBank:
    """One-time full-GPU cache of fixed spatial observer tokens for world training."""

    def __init__(
        self,
        cache: V181FeatureCache,
        observer: V181TokenFCPG,
        *,
        device: torch.device,
        chunk_rows: int = 512,
        progress_path: str | Path | None = None,
    ) -> None:
        if device.type != "cuda" or int(chunk_rows) <= 0:
            raise V181TrainingError("observer bank requires CUDA and positive chunks")
        self.cache = cache
        self.device = device
        self.chunk_rows = int(chunk_rows)
        rows = len(cache.endpoint_index)
        self.slot_bank = torch.empty(
            rows,
            VIEWS,
            OBSERVER_SLOTS,
            OBSERVER_DIM,
            dtype=torch.float16,
            device=device,
        )
        self.valid_bank = torch.empty(
            rows, VIEWS, dtype=torch.bool, device=device
        )
        self.host_tokens = torch.empty(
            self.chunk_rows,
            VIEWS,
            VIT_PATCHES,
            VIT_DIM,
            dtype=torch.float16,
            pin_memory=True,
        )
        self.host_valid = torch.empty(
            self.chunk_rows, VIEWS, dtype=torch.bool, pin_memory=True
        )
        self.device_tokens = torch.empty(
            self.chunk_rows,
            VIEWS,
            VIT_PATCHES,
            VIT_DIM,
            dtype=torch.float16,
            device=device,
        )
        destination = (
            Path(progress_path).expanduser().resolve()
            if progress_path is not None
            else None
        )
        chunks = math.ceil(rows / self.chunk_rows)
        observer.eval()
        with torch.inference_mode():
            for chunk, start in enumerate(range(0, rows, self.chunk_rows), start=1):
                stop = min(start + self.chunk_rows, rows)
                count = stop - start
                np.copyto(
                    self.host_tokens[:count].numpy(),
                    np.asarray(cache.tokens[start:stop]),
                    casting="no",
                )
                np.copyto(
                    self.host_valid[:count].numpy(),
                    np.asarray(cache.view_valid[start:stop]),
                    casting="no",
                )
                self.device_tokens[:count].copy_(
                    self.host_tokens[:count], non_blocking=False
                )
                valid = self.host_valid[:count].to(device, non_blocking=False)
                slots = observer.fixed_local_slots(self.device_tokens[:count])
                slots = slots * valid[:, :, None, None]
                self.slot_bank[start:stop].copy_(slots, non_blocking=False)
                self.valid_bank[start:stop].copy_(valid, non_blocking=False)
                if chunk % 8 == 0:
                    _flush_and_release_pages(cache.tokens, cache.view_valid)
                if destination is not None and (
                    chunk == 1 or chunk % 32 == 0 or chunk == chunks
                ):
                    _json_write(
                        destination,
                        {
                            "version": "18.1",
                            "rows_loaded": int(stop),
                            "rows_total": int(rows),
                            "chunks_completed": int(chunk),
                            "chunks_total": int(chunks),
                            "complete": bool(stop == rows),
                        },
                    )
        _flush_and_release_pages(cache.tokens, cache.view_valid)

    @property
    def allocated_bytes(self) -> int:
        tensors = (self.slot_bank, self.valid_bank, self.device_tokens)
        return int(sum(value.numel() * value.element_size() for value in tensors))

    def get(self, endpoint_indices: Any) -> tuple[torch.Tensor, torch.Tensor]:
        requested = np.asarray(endpoint_indices, dtype=np.int64)
        shape = requested.shape
        rows = self.cache.rows_for(requested.reshape(-1))
        index = torch.from_numpy(rows).to(self.device)
        slots = self.slot_bank[index]
        valid = self.valid_bank[index]
        return (
            slots.reshape(*shape, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM),
            valid.reshape(*shape, VIEWS),
        )

    def stats(self) -> dict[str, int | str]:
        return {
            "kind": "full_gpu_fixed_observer_slots",
            "allocated_bytes": self.allocated_bytes,
            "rows": len(self.cache.endpoint_index),
            "host_staging_bytes": int(
                self.host_tokens.numel() * self.host_tokens.element_size()
                + self.host_valid.numel() * self.host_valid.element_size()
            ),
        }


class V181DeviceSubsetFeatureBank:
    """One-time GPU cache for an explicit subset of full DINO patch tokens."""

    def __init__(
        self,
        cache: V181FeatureCache,
        endpoint_indices: Any,
        *,
        device: torch.device,
        chunk_rows: int = 256,
        progress_path: str | Path | None = None,
    ) -> None:
        endpoints = np.unique(np.asarray(endpoint_indices, dtype=np.int64))
        if device.type != "cuda" or int(chunk_rows) <= 0 or len(endpoints) == 0:
            raise V181TrainingError(
                "subset feature bank requires CUDA, endpoints, and positive chunks"
            )
        cache_rows = cache.rows_for(endpoints)
        self.cache = cache
        self.endpoint_index = endpoints
        self.device = device
        self.tokens = torch.empty(
            len(endpoints),
            VIEWS,
            VIT_PATCHES,
            VIT_DIM,
            dtype=torch.float16,
            device=device,
        )
        self.view_valid = torch.empty(
            len(endpoints), VIEWS, dtype=torch.bool, device=device
        )
        chunk_rows = int(chunk_rows)
        host_tokens = torch.empty(
            chunk_rows,
            VIEWS,
            VIT_PATCHES,
            VIT_DIM,
            dtype=torch.float16,
            pin_memory=True,
        )
        host_valid = torch.empty(
            chunk_rows, VIEWS, dtype=torch.bool, pin_memory=True
        )
        destination = (
            Path(progress_path).expanduser().resolve()
            if progress_path is not None
            else None
        )
        chunks = math.ceil(len(endpoints) / chunk_rows)
        for chunk, start in enumerate(range(0, len(endpoints), chunk_rows), start=1):
            stop = min(start + chunk_rows, len(endpoints))
            count = stop - start
            selected = cache_rows[start:stop]
            np.copyto(
                host_tokens[:count].numpy(),
                np.asarray(cache.tokens[selected]),
                casting="no",
            )
            np.copyto(
                host_valid[:count].numpy(),
                np.asarray(cache.view_valid[selected]),
                casting="no",
            )
            self.tokens[start:stop].copy_(host_tokens[:count], non_blocking=False)
            self.view_valid[start:stop].copy_(host_valid[:count], non_blocking=False)
            if chunk % 8 == 0:
                _flush_and_release_pages(cache.tokens, cache.view_valid)
            if destination is not None and (
                chunk == 1 or chunk % 32 == 0 or chunk == chunks
            ):
                _json_write(
                    destination,
                    {
                        "version": "18.1",
                        "endpoints_loaded": int(stop),
                        "endpoints_total": int(len(endpoints)),
                        "chunks_completed": int(chunk),
                        "chunks_total": int(chunks),
                        "complete": bool(stop == len(endpoints)),
                    },
                )
        _flush_and_release_pages(cache.tokens, cache.view_valid)

    @property
    def allocated_bytes(self) -> int:
        return int(
            self.tokens.numel() * self.tokens.element_size()
            + self.view_valid.numel() * self.view_valid.element_size()
        )

    def get(self, endpoint_indices: Any) -> tuple[torch.Tensor, torch.Tensor]:
        requested = np.asarray(endpoint_indices, dtype=np.int64)
        shape = requested.shape
        positions = np.searchsorted(self.endpoint_index, requested.reshape(-1))
        safe = np.minimum(positions, len(self.endpoint_index) - 1)
        if bool((positions >= len(self.endpoint_index)).any()) or not bool(
            np.array_equal(self.endpoint_index[safe], requested.reshape(-1))
        ):
            raise V181TrainingError("endpoint is absent from the subset feature bank")
        index = torch.from_numpy(positions.astype(np.int64, copy=False)).to(self.device)
        return (
            self.tokens[index].reshape(*shape, VIEWS, VIT_PATCHES, VIT_DIM),
            self.view_valid[index].reshape(*shape, VIEWS),
        )

    def stats(self) -> dict[str, int | str]:
        return {
            "kind": "full_gpu_subset_patch_tokens",
            "allocated_bytes": self.allocated_bytes,
            "endpoints": len(self.endpoint_index),
        }


def v181_world_objective(
    prediction: V181LeWMPrediction,
    source_slots: torch.Tensor,
    target_slots: torch.Tensor,
    source_view_valid: torch.Tensor,
    target_view_valid: torch.Tensor,
    measured_q_path: torch.Tensor,
    *,
    step_cap: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = source_view_valid & target_view_valid
    actual_delta = normalize_observer_slots(target_slots.float()) - normalize_observer_slots(
        source_slots.float()
    )
    predicted_delta = prediction.predicted_slot_delta.float()
    mask = valid[:, :, None].expand(actual_delta.shape[:-1])
    magnitude = F.smooth_l1_loss(
        predicted_delta, actual_delta, reduction="none"
    ).mean(dim=-1)
    magnitude = (magnitude * mask).sum() / mask.sum().clamp_min(1)
    directional_mask = mask & (actual_delta.norm(dim=-1) > 1.0e-3)
    cosine = 1.0 - F.cosine_similarity(predicted_delta, actual_delta, dim=-1)
    direction = (cosine * directional_mask).sum() / directional_mask.sum().clamp_min(1)
    visual = 0.5 * magnitude + 0.5 * direction
    normalized_difference = (
        prediction.predicted_q_path.float() - measured_q_path.float()
    ) / step_cap.float()[None, None]
    state = F.smooth_l1_loss(
        normalized_difference, torch.zeros_like(normalized_difference)
    )
    total = 0.5 * visual + 0.5 * state
    return total, {
        "world": total,
        "visual": visual,
        "visual_magnitude": magnitude,
        "visual_direction": direction,
        "state": state,
    }


def v181_actor_objective(
    commands: torch.Tensor,
    target_commands: torch.Tensor,
    *,
    step_cap: torch.Tensor,
) -> torch.Tensor:
    return (
        (commands.float() - target_commands.float()).abs()
        / step_cap.float()[None, None]
    ).mean()


@torch.no_grad()
def v181_training_representation_statistics(
    value: torch.Tensor,
    *,
    max_rows: int = 4_096,
) -> dict[str, float]:
    """Small deterministic rank/collapse audit suitable for checkpoint traces."""

    tensor = torch.as_tensor(value).detach().float()
    if tensor.ndim < 2 or tensor.shape[-1] < 2:
        raise V181TrainingError("representation audit needs [...,D] with D >= 2")
    matrix = tensor.reshape(-1, tensor.shape[-1])
    if len(matrix) > int(max_rows):
        indices = torch.linspace(
            0, len(matrix) - 1, int(max_rows), device=matrix.device
        ).round().long()
        matrix = matrix[indices]
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    eigenvalues = singular.square()
    if float(singular.sum()) <= 1.0e-12:
        effective_rank = singular.new_zeros(())
        participation = singular.new_zeros(())
    else:
        probability = singular / singular.sum()
        effective_rank = torch.exp(
            -(probability * probability.clamp_min(1.0e-12).log()).sum()
        )
        participation = (
            eigenvalues.sum().square()
            / eigenvalues.square().sum().clamp_min(1.0e-12)
        )
    result = {
        "effective_rank": float(effective_rank),
        "participation_ratio": float(participation),
        "feature_variance": float(centered.var(dim=0, unbiased=False).mean()),
        "token_norm_mean": float(matrix.norm(dim=-1).mean()),
    }
    if tensor.ndim >= 4 and tensor.shape[-2] > 1:
        slots = F.normalize(tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1]), dim=-1)
        gram = torch.matmul(slots, slots.transpose(-1, -2))
        count = gram.shape[-1]
        off_diagonal = ~torch.eye(count, dtype=torch.bool, device=gram.device)
        values = gram[:, off_diagonal]
        result["cross_slot_cosine_mean"] = float(values.mean())
        result["cross_slot_cosine_abs_mean"] = float(values.abs().mean())
    return result


def _module_gradient_norm(module: nn.Module) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().double().square().sum().cpu()
    return float(squared.sqrt())


@torch.no_grad()
def v181_world_training_probe(
    model: V181TipFreeController,
    source_slots: torch.Tensor,
    target_slots: torch.Tensor,
    source_view_valid: torch.Tensor,
    target_view_valid: torch.Tensor,
    current_q: torch.Tensor,
    commands: torch.Tensor,
    measured_q_path: torch.Tensor,
    *,
    observer_reference: torch.Tensor,
    seed: int = 18_100,
) -> dict[str, Any]:
    """Factual/shuffled/random-action and frozen-coordinate sanity panel."""

    if observer_reference.shape != source_slots.shape:
        raise V181TrainingError("observer drift reference shape differs")
    if len(commands) > 1:
        shuffled = torch.roll(commands, shifts=1, dims=0)
    else:
        shuffled = torch.flip(commands, dims=(1,))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.rand(commands.shape, generator=generator, dtype=torch.float32)
    noise = noise.to(device=commands.device, dtype=commands.dtype).mul_(2).sub_(1)
    desired = current_q[:, None] + noise * model.actor.step_cap[None, None]
    random_commands = project_v181_action_trajectory(
        current_q,
        desired,
        step_cap=model.actor.step_cap,
        action_lower=model.actor.action_lower,
        action_upper=model.actor.action_upper,
    )

    def evaluate(action: torch.Tensor) -> tuple[V181LeWMPrediction, dict[str, float]]:
        with _autocast(source_slots.device):
            prediction = model.lewm(
                source_slots,
                source_view_valid,
                current_q,
                action,
            )
            _, pieces = v181_world_objective(
                prediction,
                source_slots,
                target_slots,
                source_view_valid,
                target_view_valid,
                measured_q_path,
                step_cap=model.actor.step_cap,
            )
        return prediction, {name: float(value) for name, value in pieces.items()}

    factual, factual_loss = evaluate(commands)
    shuffled_prediction, shuffled_loss = evaluate(shuffled)
    random_prediction, random_loss = evaluate(random_commands)
    reference = observer_reference.to(source_slots)
    drift = source_slots.float() - reference.float()
    return {
        "observer": {
            **v181_training_representation_statistics(source_slots),
            "frozen_drift_rms": float(drift.square().mean().sqrt()),
            "frozen_drift_max": float(drift.abs().max()),
        },
        "predicted_transition": v181_training_representation_statistics(
            factual.predicted_slot_delta
        ),
        "loss": {
            "factual": factual_loss,
            "shuffled_action": shuffled_loss,
            "random_action": random_loss,
        },
        "action_response": {
            "shuffle_visual_rms": float(
                (factual.predicted_slot_delta - shuffled_prediction.predicted_slot_delta)
                .float()
                .square()
                .mean()
                .sqrt()
            ),
            "random_visual_rms": float(
                (factual.predicted_slot_delta - random_prediction.predicted_slot_delta)
                .float()
                .square()
                .mean()
                .sqrt()
            ),
            "shuffle_q_path_rms": float(
                (factual.predicted_q_path - shuffled_prediction.predicted_q_path)
                .float()
                .square()
                .mean()
                .sqrt()
            ),
            "random_q_path_rms": float(
                (factual.predicted_q_path - random_prediction.predicted_q_path)
                .float()
                .square()
                .mean()
                .sqrt()
            ),
        },
    }


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _cosine_lambda(step: int, *, total: int, warmup: int) -> float:
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _batch_chunks(values: np.ndarray, batch: int) -> Iterator[np.ndarray]:
    for start in range(0, len(values), int(batch)):
        selected = values[start : start + int(batch)]
        if len(selected):
            yield selected


def train_v181_world(
    model: V181TipFreeController,
    archive: V181TemporalArchive,
    observer_bank: V181DeviceObserverBank,
    output_root: str | Path,
    *,
    device: torch.device,
    seed: int = 18_000,
    batch_size: int = WORLD_EFFECTIVE_BATCH,
    resume: bool = True,
) -> dict[str, Any]:
    """Four fixed passes: H0-H1, H1-H2, H2-H3, and H3-H4."""

    if int(batch_size) <= 0:
        raise V181TrainingError("world batch size must be positive")
    seed_v181(seed)
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "world-final.pt"
    latest_path = output / "world-latest.pt"
    if resume and final_path.is_file():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != "18.1"
            or payload.get("phase") != "world"
            or int(payload.get("seed", -1)) != int(seed)
            or not isinstance(payload.get("model"), dict)
            or not isinstance(payload.get("metrics"), dict)
        ):
            raise V181TrainingError("completed world artifact is incompatible")
        model.load_state_dict(payload["model"], strict=True)
        model.to(device)
        return dict(payload["metrics"])
    model.to(device).train()
    # The world coordinate system is parameter-free frozen-DINO spatial
    # pooling.  Only LeWM is optimized on 320K; the policy adapter first sees
    # gradients from the fish Actor objective.
    parameters = list(model.lewm.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=WORLD_LR)
    train_rows = archive.rows_for_partition("train")
    total_steps = math.ceil(len(train_rows) / int(batch_size)) * TRANSITIONS
    warmup = int(round(total_steps * WORLD_WARMUP_FRACTION))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_lambda(step, total=total_steps, warmup=warmup),
    )
    step = 0
    pass_metrics: list[dict[str, float]] = []
    probe_trace: list[dict[str, Any]] = []
    observer_reference: torch.Tensor | None = None
    next_transition = 0
    next_offset = 0
    resumed_sums: dict[str, float] | None = None
    resumed_samples = 0
    if resume:
        payload = _resume_payload(
            latest_path, model, phase="world", seed=seed
        )
        if payload is not None:
            training_state = payload["training_state"]
            optimizer_state = training_state.get("optimizer")
            scheduler_state = training_state.get("scheduler")
            if not isinstance(optimizer_state, dict) or not isinstance(
                scheduler_state, dict
            ):
                raise V181TrainingError("world resume omits optimizer state")
            if int(training_state.get("batch_size", -1)) != int(batch_size):
                raise V181TrainingError("world resume batch size differs")
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
            step = int(payload["step"])
            next_transition = int(training_state.get("next_transition", -1))
            next_offset = int(training_state.get("next_offset", -1))
            raw_metrics = training_state.get("pass_metrics", [])
            raw_sums = training_state.get("current_sums", {})
            if (
                not 0 <= next_transition <= TRANSITIONS
                or next_offset < 0
                or not isinstance(raw_metrics, list)
                or not isinstance(raw_sums, dict)
            ):
                raise V181TrainingError("world resume cursor is malformed")
            pass_metrics = [dict(item) for item in raw_metrics]
            raw_probe_trace = training_state.get("probe_trace", [])
            raw_reference = training_state.get("observer_reference")
            if not isinstance(raw_probe_trace, list) or not isinstance(
                raw_reference, torch.Tensor
            ):
                raise V181TrainingError("world resume omits frozen-observer audit state")
            probe_trace = [dict(item) for item in raw_probe_trace]
            observer_reference = raw_reference.float().cpu()
            resumed_sums = {str(name): float(value) for name, value in raw_sums.items()}
            resumed_samples = int(training_state.get("current_samples", 0))

    metric_names = (
        "world",
        "visual",
        "visual_magnitude",
        "visual_direction",
        "state",
    )

    # Keep the pre-registered 32-row panel stable across exact resume.  The
    # fixed observer coordinates have already been compacted into one GPU bank.
    audit_rows = train_rows[: min(32, len(train_rows))]
    audit_batch = archive.transition_batch(audit_rows, 0, partition="train")

    @torch.no_grad()
    def collect_probe(probe_step: int) -> dict[str, Any]:
        nonlocal observer_reference
        source_slots, source_valid = observer_bank.get(
            audit_batch.source_endpoint_index
        )
        target_slots, target_valid = observer_bank.get(
            audit_batch.target_endpoint_index
        )
        source_valid = source_valid & torch.from_numpy(
            audit_batch.source_view_valid
        ).to(device)
        target_valid = target_valid & torch.from_numpy(
            audit_batch.target_view_valid
        ).to(device)
        source_slots = source_slots * source_valid[:, :, None, None]
        target_slots = target_slots * target_valid[:, :, None, None]
        if observer_reference is None:
            observer_reference = source_slots.detach().float().cpu().clone()
        report = v181_world_training_probe(
            model,
            source_slots,
            target_slots,
            source_valid,
            target_valid,
            torch.from_numpy(audit_batch.current_q).to(device),
            torch.from_numpy(audit_batch.commands).to(device),
            torch.from_numpy(audit_batch.measured_q_path).to(device),
            observer_reference=observer_reference,
            seed=seed + int(probe_step),
        )
        return {"step": int(probe_step), **report}

    if not probe_trace:
        probe_trace.append(collect_probe(step))

    def save_latest(
        *,
        transition_cursor: int,
        offset_cursor: int,
        sums: Mapping[str, float],
        samples: int,
    ) -> None:
        running = {
            name: (float(sums[name]) / samples if samples else 0.0)
            for name in metric_names
        }
        save_v181_checkpoint(
            latest_path,
            model,
            phase="world",
            seed=seed,
            step=step,
            metrics=running,
            training_state={
                "next_transition": int(transition_cursor),
                "next_offset": int(offset_cursor),
                "batch_size": int(batch_size),
                "pass_metrics": pass_metrics,
                "current_sums": dict(sums),
                "current_samples": int(samples),
                "probe_trace": probe_trace,
                "observer_reference": observer_reference,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
        )

    for transition in range(next_transition, TRANSITIONS):
        # Endpoint-local ordering is retained for exact resume compatibility;
        # all fixed observer slots are already resident on this GPU.
        source_endpoint = archive.row_endpoint_index[train_rows, transition]
        order = np.argsort(source_endpoint, kind="stable")
        group = max(1, int(batch_size) * 32)
        blocks = [order[start : start + group] for start in range(0, len(order), group)]
        generator = np.random.default_rng(seed + transition)
        generator.shuffle(blocks)
        ordered_rows = train_rows[np.concatenate(blocks)]
        offset = next_offset if transition == next_transition else 0
        if offset > len(ordered_rows):
            raise V181TrainingError("world resume offset leaves transition rows")
        sums = (
            {name: float((resumed_sums or {}).get(name, 0.0)) for name in metric_names}
            if transition == next_transition and offset
            else {name: 0.0 for name in metric_names}
        )
        samples = resumed_samples if transition == next_transition and offset else 0
        for start in range(offset, len(ordered_rows), int(batch_size)):
            rows = ordered_rows[start : start + int(batch_size)]
            batch = archive.transition_batch(rows, transition, partition="train")
            source_slots, source_valid = observer_bank.get(
                batch.source_endpoint_index
            )
            target_slots, target_valid = observer_bank.get(
                batch.target_endpoint_index
            )
            current_q = torch.from_numpy(batch.current_q).to(device)
            commands = torch.from_numpy(batch.commands).to(device)
            measured_path = torch.from_numpy(batch.measured_q_path).to(device)
            source_valid = source_valid & torch.from_numpy(
                batch.source_view_valid
            ).to(device)
            target_valid = target_valid & torch.from_numpy(
                batch.target_view_valid
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                source_slots = source_slots * source_valid[:, :, None, None]
                target_slots = target_slots * target_valid[:, :, None, None]
                prediction = model.lewm(
                    source_slots,
                    source_valid,
                    current_q,
                    commands,
                )
                loss, pieces = v181_world_objective(
                    prediction,
                    source_slots,
                    target_slots,
                    source_valid,
                    target_valid,
                    measured_path,
                    step_cap=model.actor.step_cap,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            step += 1
            samples += len(rows)
            for name in sums:
                sums[name] += float(pieces[name].detach()) * len(rows)
            if step % CHECKPOINT_EVERY == 0:
                probe = collect_probe(step)
                probe["gradient_norm"] = {"lewm": _module_gradient_norm(model.lewm)}
                probe_trace.append(probe)
                save_latest(
                    transition_cursor=transition,
                    offset_cursor=start + len(rows),
                    sums=sums,
                    samples=samples,
                )
        pass_metrics.append(
            {
                "transition": transition,
                "rows": samples,
                **{name: value / samples for name, value in sums.items()},
            }
        )
        save_latest(
            transition_cursor=transition + 1,
            offset_cursor=0,
            sums={name: 0.0 for name in metric_names},
            samples=0,
        )
        next_offset = 0
        resumed_sums = None
        resumed_samples = 0
    if not probe_trace or int(probe_trace[-1]["step"]) != step:
        probe_trace.append(collect_probe(step))
    metrics: dict[str, Any] = {
        "version": "18.1",
        "phase": "world",
        "seed": seed,
        "steps": step,
        "passes": pass_metrics,
        "training_probe_trace": probe_trace,
        "observer_bank": observer_bank.stats(),
    }
    save_v181_checkpoint(
        final_path,
        model,
        phase="world",
        seed=seed,
        step=step,
        metrics=metrics,
    )
    _json_write(output / "world-metrics.json", metrics)
    return metrics


@torch.no_grad()
def _actor_bank_slots(
    model: V181TipFreeController,
    bank: V181DeviceObserverBank | V181DeviceSubsetFeatureBank,
    endpoint_indices: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    values, valid = bank.get(endpoint_indices)
    if values.shape[-2:] == (VIT_PATCHES, VIT_DIM):
        slots = model.observer.fixed_local_slots(values)
    elif values.shape[-2:] == (OBSERVER_SLOTS, OBSERVER_DIM):
        slots = values
    else:
        raise V181TrainingError("Actor bank returned an unknown visual shape")
    return slots * valid[..., None, None], valid


@torch.no_grad()
def _temporal_actor_conditioning(
    model: V181TipFreeController,
    bank: V181DeviceObserverBank | V181DeviceSubsetFeatureBank,
    pair: V181TemporalActorBatch,
    *,
    device: torch.device,
    dropout_seed: int | None = None,
    dropout_clock: int = 0,
) -> V181FishConditioning:
    """Reconstruct only completed H0--H2 transitions for the next Actor pair."""

    batch = len(pair.ordinal)
    actual = torch.zeros(
        batch, HISTORY_CHUNKS, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM, device=device
    )
    actual_valid = torch.zeros(
        batch, HISTORY_CHUNKS, VIEWS, dtype=torch.bool, device=device
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
    commands = torch.from_numpy(pair.history_commands).to(device)
    paths = torch.from_numpy(pair.history_paths).to(device)
    current = empty_v181_innovation(batch, device=device)
    predicted_missing = torch.zeros(
        batch, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM, device=device
    )
    prediction_valid = torch.zeros(batch, VIEWS, dtype=torch.bool, device=device)
    predicted_transition = torch.zeros_like(predicted_missing)
    servo_error = torch.zeros(batch, PATH_STEPS, STATE_DIM, device=device)
    servo_lag_error = torch.zeros_like(servo_error)

    def use_views(endpoint: np.ndarray, valid: torch.Tensor) -> torch.Tensor:
        if dropout_seed is None:
            return valid
        sampled = deterministic_fish_view_valid(
            np.asarray(endpoint, dtype=np.int64) + int(dropout_clock) * 13,
            seed=int(dropout_seed),
        )
        return valid & torch.from_numpy(sampled).to(device)

    for slot in range(HISTORY_CHUNKS):
        active = history_valid[:, slot]
        source_index = pair.history_source_endpoint_index[:, slot]
        target_index = pair.history_target_endpoint_index[:, slot]
        source_slots, source_views = _actor_bank_slots(model, bank, source_index)
        target_slots, target_views = _actor_bank_slots(model, bank, target_index)
        source_views = use_views(source_index, source_views) & active[:, None]
        target_views = use_views(target_index, target_views) & active[:, None]
        source_slots = source_slots * source_views[:, :, None, None]
        target_slots = target_slots * target_views[:, :, None, None]
        with _autocast(device):
            prediction = model.lewm(
                source_slots,
                source_views,
                paths[:, slot, 0],
                commands[:, slot],
            )
        innovation = compute_v181_innovation(
            source_slots,
            source_views,
            target_slots,
            target_views,
            paths[:, slot],
            prediction,
            path_valid=active[:, None].expand(-1, PATH_STEPS),
        )
        actual[:, slot] = source_slots
        actual_valid[:, slot] = source_views
        visual[:, slot] = innovation.visual
        state[:, slot] = innovation.state
        innovation_valid[:, slot] = innovation.innovation_valid
        path_valid[:, slot] = innovation.path_valid
        last = torch.from_numpy(pair.transition == slot + 1).to(device)
        current = V181Innovation(
            visual=torch.where(
                last[:, None, None, None], innovation.visual, current.visual
            ),
            state=torch.where(last[:, None, None], innovation.state, current.state),
            actual_slot_delta=torch.where(
                last[:, None, None, None],
                innovation.actual_slot_delta,
                current.actual_slot_delta,
            ),
            innovation_valid=torch.where(
                last[:, None], innovation.innovation_valid, current.innovation_valid
            ),
            path_valid=torch.where(last[:, None], innovation.path_valid, current.path_valid),
        )
        predicted_missing = torch.where(
            last[:, None, None, None],
            predicted_observer_slots(source_slots, prediction),
            predicted_missing,
        )
        prediction_valid = torch.where(
            last[:, None], source_views & ~target_views, prediction_valid
        )
        predicted_transition = torch.where(
            last[:, None, None, None],
            prediction.predicted_slot_delta,
            predicted_transition,
        )
        commanded = torch.cat((paths[:, slot, :1], commands[:, slot]), dim=1)
        slot_servo = (paths[:, slot] - commanded) * innovation.path_valid[..., None]
        servo_error = torch.where(last[:, None, None], slot_servo, servo_error)
        lag = fixed_servo_lag_path(paths[:, slot, 0], commands[:, slot])
        slot_lag = (paths[:, slot] - lag) * innovation.path_valid[..., None]
        servo_lag_error = torch.where(last[:, None, None], slot_lag, servo_lag_error)

    memory = V181Memory(
        actual_observer_slots=actual,
        actual_view_valid=actual_valid,
        completed_commands=commands * history_valid[..., None, None],
        measured_paths=paths * history_valid[..., None, None],
        visual_innovations=visual,
        state_innovations=state,
        innovation_valid=innovation_valid,
        path_valid=path_valid,
        delta_t_s=history_valid.float() * 0.5,
        history_valid=history_valid,
    )
    memory.validate()
    return V181FishConditioning(
        memory=memory,
        innovation=current,
        predicted_missing_slots=predicted_missing,
        prediction_valid=prediction_valid,
        predicted_transition=predicted_transition,
        servo_error=servo_error,
        servo_lag_error=servo_lag_error,
    )


def train_v181_archive_actor(
    model: V181TipFreeController,
    archive: V181TemporalArchive,
    observer_bank: V181DeviceObserverBank,
    output_root: str | Path,
    *,
    device: torch.device,
    seed: int,
    variant: str = "full",
    passes: int = ARCHIVE_ACTOR_PASSES,
    batch_size: int = ARCHIVE_ACTOR_BATCH,
    resume: bool = True,
) -> dict[str, Any]:
    """Causal 320K behavior-cloning warm start using no task or H4 goal input."""

    if variant not in FISH_VARIANTS:
        raise V181TrainingError(f"archive Actor variant must be one of {FISH_VARIANTS}")
    if int(seed) not in SEEDS or int(passes) <= 0 or int(batch_size) <= 0:
        raise V181TrainingError("archive Actor seed/passes/batch are invalid")
    seed_v181(seed)
    output = Path(output_root).expanduser().resolve() / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / "latest.pt"
    final_path = output / "actor-final.pt"
    if resume and final_path.is_file():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        observed_variant = payload.get("metrics", {}).get("variant", "full")
        if (
            payload.get("phase") != "archive-actor"
            or int(payload.get("seed", -1)) != seed
            or observed_variant != variant
        ):
            raise V181TrainingError("archive Actor final artifact is incompatible")
        model.load_state_dict(payload["model"], strict=True)
        model.to(device)
        return dict(payload["metrics"])
    model.to(device).train()
    model.observer.requires_grad_(False).eval()
    model.lewm.requires_grad_(False).eval()
    model.actor.requires_grad_(True).train()
    optimizer = torch.optim.AdamW(model.actor.parameters(), lr=ARCHIVE_ACTOR_LR)
    population = archive.actor_pair_ordinals(partition="train")
    steps_per_pass = math.ceil(len(population) / int(batch_size))
    total_steps = int(passes) * steps_per_pass
    warmup = int(round(total_steps * WORLD_WARMUP_FRACTION))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda value: _cosine_lambda(value, total=total_steps, warmup=warmup),
    )
    step = 0
    next_pass = 0
    next_offset = 0
    sums = np.zeros(TRANSITIONS, dtype=np.float64)
    counts = np.zeros(TRANSITIONS, dtype=np.int64)
    if resume:
        payload = _resume_payload(latest_path, model, phase="archive-actor", seed=seed)
        if payload is not None:
            state = payload["training_state"]
            if int(state.get("passes", -1)) != int(passes) or int(
                state.get("batch_size", -1)
            ) != int(batch_size) or state.get("variant", "full") != variant:
                raise V181TrainingError("archive Actor resume contract differs")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step = int(payload["step"])
            next_pass = int(state["next_pass"])
            next_offset = int(state["next_offset"])
            sums = np.asarray(state["horizon_sums"], dtype=np.float64)
            counts = np.asarray(state["horizon_counts"], dtype=np.int64)

    def save_latest(pass_cursor: int, offset_cursor: int) -> None:
        metrics = {
            "actor": float(sums.sum() / max(1, counts.sum())),
            "variant": variant,
            "horizon": {
                str(index): float(sums[index] / max(1, counts[index]))
                for index in range(TRANSITIONS)
            },
        }
        save_v181_checkpoint(
            latest_path,
            model,
            phase="archive-actor",
            seed=seed,
            step=step,
            metrics=metrics,
            training_state={
                "passes": int(passes),
                "batch_size": int(batch_size),
                "variant": variant,
                "next_pass": int(pass_cursor),
                "next_offset": int(offset_cursor),
                "horizon_sums": sums.tolist(),
                "horizon_counts": counts.tolist(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
        )

    for pass_index in range(next_pass, int(passes)):
        order = population.copy()
        np.random.default_rng(seed + pass_index).shuffle(order)
        offset = next_offset if pass_index == next_pass else 0
        for start in range(offset, len(order), int(batch_size)):
            ordinal = order[start : start + int(batch_size)]
            pair = archive.actor_pair_batch(ordinal, partition="train")
            slots, current_valid = observer_bank.get(pair.source_endpoint_index)
            current_valid &= torch.from_numpy(pair.source_view_valid).to(device)
            slots = slots * current_valid[:, :, None, None]
            conditioning = _temporal_actor_conditioning(
                model, observer_bank, pair, device=device
            )
            q = torch.from_numpy(pair.current_q).to(device)
            target = torch.from_numpy(pair.target_commands).to(device)
            dense = torch.zeros(
                len(ordinal), VIEWS, VIT_PATCHES, MODEL_DIM,
                device=device, dtype=slots.dtype,
            )
            measurement = V181Measurement(
                policy_dense_tokens=dense,
                local_observer_slots=slots,
                policy_by_layer=torch.empty(0, device=device, dtype=slots.dtype),
            )
            (
                used_dense,
                used_slots,
                used_innovation,
                used_predicted,
                used_prediction_valid,
                used_memory,
            ) = fish_variant_inputs(
                variant, measurement, current_valid, conditioning
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                output_commands = model.actor(
                    used_dense,
                    used_slots,
                    current_valid,
                    used_innovation.visual,
                    used_innovation.state,
                    used_innovation.innovation_valid,
                    used_innovation.path_valid,
                    used_predicted,
                    used_prediction_valid,
                    q,
                    used_memory,
                ).commands
                loss = v181_actor_objective(
                    output_commands, target, step_cap=model.actor.step_cap
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            step += 1
            per_pair = (
                ((output_commands.detach() - target).abs() / model.actor.step_cap)
                .mean(dim=(1, 2))
                .float()
                .cpu()
                .numpy()
            )
            for horizon in range(TRANSITIONS):
                mask = pair.transition == horizon
                sums[horizon] += float(per_pair[mask].sum())
                counts[horizon] += int(mask.sum())
            if step % CHECKPOINT_EVERY == 0:
                save_latest(pass_index, start + len(ordinal))
        save_latest(pass_index + 1, 0)
        next_offset = 0
    metrics = {
        "version": "18.1",
        "phase": "archive-actor",
        "seed": seed,
        "variant": variant,
        "steps": step,
        "passes": int(passes),
        "pairs_per_pass": int(len(population)),
        "actor": float(sums.sum() / max(1, counts.sum())),
        "horizon": {
            str(index): float(sums[index] / max(1, counts[index]))
            for index in range(TRANSITIONS)
        },
        "observer_bank": observer_bank.stats(),
    }
    save_v181_checkpoint(
        final_path,
        model,
        phase="archive-actor",
        seed=seed,
        step=step,
        metrics=metrics,
    )
    _json_write(output / "train-metrics.json", metrics)
    return metrics


def train_v181_dense_actor_adaptation(
    model: V181TipFreeController,
    archive: V181TemporalArchive,
    feature_bank: V181DeviceSubsetFeatureBank,
    train_rows: np.ndarray,
    output_root: str | Path,
    *,
    device: torch.device,
    seed: int,
    variant: str = "full",
    passes: int = DENSE_ADAPTATION_PASSES,
    batch_size: int = DENSE_ADAPTATION_BATCH,
    resume: bool = True,
) -> dict[str, Any]:
    """Train the dense policy branch on the existing 17K TRAIN RGB subset."""

    if variant not in FISH_VARIANTS:
        raise V181TrainingError(f"dense Actor variant must be one of {FISH_VARIANTS}")
    if int(seed) not in SEEDS or int(passes) <= 0 or int(batch_size) <= 0:
        raise V181TrainingError("dense Actor seed/passes/batch are invalid")
    seed_v181(seed)
    output = Path(output_root).expanduser().resolve() / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / "latest.pt"
    final_path = output / "dense-final.pt"
    if resume and final_path.is_file():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        observed_variant = payload.get("metrics", {}).get("variant", "full")
        if (
            payload.get("phase") != "dense20k"
            or int(payload.get("seed", -1)) != seed
            or observed_variant != variant
        ):
            raise V181TrainingError("dense Actor final artifact is incompatible")
        model.load_state_dict(payload["model"], strict=True)
        model.to(device)
        return dict(payload["metrics"])
    model.to(device).train()
    model.lewm.requires_grad_(False).eval()
    model.observer.requires_grad_(True).train()
    model.actor.requires_grad_(True).train()
    optimizer = torch.optim.AdamW(
        (
            {"params": model.observer.parameters(), "lr": FCPG_LR},
            {"params": model.actor.parameters(), "lr": DENSE_ACTOR_LR},
        )
    )
    population = archive.actor_pair_ordinals(partition="train", rows=train_rows)
    steps_per_pass = math.ceil(len(population) / int(batch_size))
    total_steps = int(passes) * steps_per_pass
    warmup = int(round(total_steps * WORLD_WARMUP_FRACTION))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda value: _cosine_lambda(value, total=total_steps, warmup=warmup),
    )
    step = 0
    next_pass = 0
    next_offset = 0
    loss_sum = 0.0
    samples = 0
    if resume:
        payload = _resume_payload(latest_path, model, phase="dense20k", seed=seed)
        if payload is not None:
            state = payload["training_state"]
            if int(state.get("passes", -1)) != int(passes) or int(
                state.get("batch_size", -1)
            ) != int(batch_size) or state.get("variant", "full") != variant:
                raise V181TrainingError("dense Actor resume contract differs")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step = int(payload["step"])
            next_pass = int(state["next_pass"])
            next_offset = int(state["next_offset"])
            loss_sum = float(state["loss_sum"])
            samples = int(state["samples"])

    def save_latest(pass_cursor: int, offset_cursor: int) -> None:
        save_v181_checkpoint(
            latest_path,
            model,
            phase="dense20k",
            seed=seed,
            step=step,
            metrics={
                "actor": loss_sum / max(1, samples),
                "variant": variant,
            },
            training_state={
                "passes": int(passes),
                "batch_size": int(batch_size),
                "variant": variant,
                "next_pass": int(pass_cursor),
                "next_offset": int(offset_cursor),
                "loss_sum": float(loss_sum),
                "samples": int(samples),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
        )

    for pass_index in range(next_pass, int(passes)):
        order = population.copy()
        np.random.default_rng(seed + 10_000 + pass_index).shuffle(order)
        offset = next_offset if pass_index == next_pass else 0
        for start in range(offset, len(order), int(batch_size)):
            ordinal = order[start : start + int(batch_size)]
            pair = archive.actor_pair_batch(
                ordinal, partition="train", rows=train_rows
            )
            tokens, cached_valid = feature_bank.get(pair.source_endpoint_index)
            sampled_valid = torch.from_numpy(
                deterministic_fish_view_valid(
                    pair.source_endpoint_index + step * 13, seed=seed
                )
            ).to(device)
            current_valid = (
                cached_valid
                & torch.from_numpy(pair.source_view_valid).to(device)
                & sampled_valid
            )
            conditioning = _temporal_actor_conditioning(
                model,
                feature_bank,
                pair,
                device=device,
                dropout_seed=seed,
                dropout_clock=step,
            )
            q = torch.from_numpy(pair.current_q).to(device)
            target = torch.from_numpy(pair.target_commands).to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                measurement = model.observer(tokens, current_valid)
                (
                    used_dense,
                    used_slots,
                    used_innovation,
                    used_predicted,
                    used_prediction_valid,
                    used_memory,
                ) = fish_variant_inputs(
                    variant, measurement, current_valid, conditioning
                )
                commands = model.actor(
                    used_dense,
                    used_slots,
                    current_valid,
                    used_innovation.visual,
                    used_innovation.state,
                    used_innovation.innovation_valid,
                    used_innovation.path_valid,
                    used_predicted,
                    used_prediction_valid,
                    q,
                    used_memory,
                ).commands
                loss = v181_actor_objective(
                    commands, target, step_cap=model.actor.step_cap
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                tuple(model.observer.parameters()) + tuple(model.actor.parameters()),
                GRADIENT_CLIP,
            )
            optimizer.step()
            scheduler.step()
            step += 1
            loss_sum += float(loss.detach()) * len(ordinal)
            samples += len(ordinal)
            if step % CHECKPOINT_EVERY == 0:
                save_latest(pass_index, start + len(ordinal))
        save_latest(pass_index + 1, 0)
        next_offset = 0
    metrics = {
        "version": "18.1",
        "phase": "dense20k",
        "seed": seed,
        "variant": variant,
        "steps": step,
        "passes": int(passes),
        "train_rows": int(len(train_rows)),
        "pairs_per_pass": int(len(population)),
        "actor": loss_sum / max(1, samples),
        "feature_bank": feature_bank.stats(),
    }
    save_v181_checkpoint(
        final_path,
        model,
        phase="dense20k",
        seed=seed,
        step=step,
        metrics=metrics,
    )
    _json_write(output / "train-metrics.json", metrics)
    return metrics


def _measure_frames(
    model: V181TipFreeController,
    feature_bank: V181DeviceFeatureBank,
    frames: torch.Tensor | np.ndarray,
    view_valid: torch.Tensor,
) -> Any:
    tokens, cached_valid = feature_bank.get(
        frames.detach().cpu().numpy() if isinstance(frames, torch.Tensor) else frames
    )
    valid = cached_valid & view_valid
    with _autocast(feature_bank.device):
        return model.observer(tokens, valid), valid


@torch.no_grad()
def _fixed_slots_frames(
    model: V181TipFreeController,
    feature_bank: V181DeviceFeatureBank,
    frames: np.ndarray,
    view_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, cached_valid = feature_bank.get(frames)
    valid = cached_valid & view_valid
    slots = model.observer.fixed_local_slots(tokens)
    return slots * valid[:, :, None, None], valid


@torch.no_grad()
def _fish_memory(
    model: V181TipFreeController,
    feature_bank: V181DeviceFeatureBank,
    pair: V181FishPair,
    *,
    device: torch.device,
    seed: int,
    update: int,
) -> V181FishConditioning:
    batch = len(pair.ordinal)
    actual = torch.zeros(
        batch, HISTORY_CHUNKS, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM, device=device
    )
    actual_valid = torch.zeros(
        batch, HISTORY_CHUNKS, VIEWS, dtype=torch.bool, device=device
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
    commands = torch.from_numpy(pair.history_commands).to(device)
    paths = torch.from_numpy(pair.history_paths).to(device)
    current = empty_v181_innovation(batch, device=device)
    predicted_missing = torch.zeros(
        batch, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM, device=device
    )
    prediction_valid = torch.zeros(
        batch, VIEWS, dtype=torch.bool, device=device
    )
    predicted_transition = torch.zeros_like(predicted_missing)
    servo_error = torch.zeros(
        batch, PATH_STEPS, STATE_DIM, device=device
    )
    servo_lag_error = torch.zeros_like(servo_error)
    for slot in range(HISTORY_CHUNKS):
        active = history_valid[:, slot]
        safe_source = np.maximum(pair.history_source_frames[:, slot], 0)
        target = safe_source + 15
        source_views = torch.from_numpy(
            deterministic_fish_view_valid(
                safe_source + update * 13, seed=seed
            )
        ).to(device)
        target_views = torch.from_numpy(
            deterministic_fish_view_valid(
                target + update * 13, seed=seed
            )
        ).to(device)
        source_views &= active[:, None]
        target_views &= active[:, None]
        source_slots, source_views = _fixed_slots_frames(
            model, feature_bank, safe_source, source_views
        )
        target_slots, target_views = _fixed_slots_frames(
            model, feature_bank, target, target_views
        )
        with _autocast(device):
            prediction = model.lewm(
                source_slots,
                source_views,
                paths[:, slot, 0],
                commands[:, slot],
            )
        innovation = compute_v181_innovation(
            source_slots,
            source_views,
            target_slots,
            target_views,
            paths[:, slot],
            prediction,
            path_valid=active[:, None].expand(-1, PATH_STEPS),
        )
        actual[:, slot] = source_slots
        actual_valid[:, slot] = source_views
        visual[:, slot] = innovation.visual
        state[:, slot] = innovation.state
        innovation_valid[:, slot] = innovation.innovation_valid
        path_valid[:, slot] = innovation.path_valid
        last = torch.from_numpy(pair.transition == slot + 1).to(device)
        current = V181Innovation(
            visual=torch.where(
                last[:, None, None, None], innovation.visual, current.visual
            ),
            state=torch.where(last[:, None, None], innovation.state, current.state),
            actual_slot_delta=torch.where(
                last[:, None, None, None],
                innovation.actual_slot_delta,
                current.actual_slot_delta,
            ),
            innovation_valid=torch.where(
                last[:, None], innovation.innovation_valid, current.innovation_valid
            ),
            path_valid=torch.where(last[:, None], innovation.path_valid, current.path_valid),
        )
        predicted_missing = torch.where(
            last[:, None, None, None],
            predicted_observer_slots(source_slots, prediction),
            predicted_missing,
        )
        missing = source_views & ~target_views
        prediction_valid = torch.where(last[:, None], missing, prediction_valid)
        predicted_transition = torch.where(
            last[:, None, None, None],
            prediction.predicted_slot_delta,
            predicted_transition,
        )
        commanded_path = torch.cat(
            (paths[:, slot, :1], commands[:, slot]), dim=1
        )
        slot_servo = (paths[:, slot] - commanded_path) * innovation.path_valid[..., None]
        servo_error = torch.where(last[:, None, None], slot_servo, servo_error)
        lag_path = fixed_servo_lag_path(paths[:, slot, 0], commands[:, slot])
        slot_lag = (paths[:, slot] - lag_path) * innovation.path_valid[..., None]
        servo_lag_error = torch.where(last[:, None, None], slot_lag, servo_lag_error)
    memory = V181Memory(
        actual_observer_slots=actual,
        actual_view_valid=actual_valid,
        completed_commands=commands * history_valid[..., None, None],
        measured_paths=paths * history_valid[..., None, None],
        visual_innovations=visual,
        state_innovations=state,
        innovation_valid=innovation_valid,
        path_valid=path_valid,
        delta_t_s=history_valid.to(torch.float32) * 0.5,
        history_valid=history_valid,
    )
    memory.validate()
    return V181FishConditioning(
        memory=memory,
        innovation=current,
        predicted_missing_slots=predicted_missing,
        prediction_valid=prediction_valid,
        predicted_transition=predicted_transition,
        servo_error=servo_error,
        servo_lag_error=servo_lag_error,
    )


def _null_memory(memory: V181Memory, *, keep_actual: bool) -> V181Memory:
    if not keep_actual:
        return V181Memory.empty(
            memory.batch_size,
            device=memory.actual_observer_slots.device,
            dtype=memory.actual_observer_slots.dtype,
        )
    return V181Memory(
        actual_observer_slots=memory.actual_observer_slots,
        actual_view_valid=memory.actual_view_valid,
        completed_commands=memory.completed_commands,
        measured_paths=memory.measured_paths,
        visual_innovations=torch.zeros_like(memory.visual_innovations),
        state_innovations=torch.zeros_like(memory.state_innovations),
        innovation_valid=memory.innovation_valid,
        path_valid=memory.path_valid,
        delta_t_s=memory.delta_t_s,
        history_valid=memory.history_valid,
    )


def _visual_history_memory(memory: V181Memory) -> V181Memory:
    """Keep only actual endpoint tokens and their causal clock."""

    return V181Memory(
        actual_observer_slots=memory.actual_observer_slots,
        actual_view_valid=memory.actual_view_valid,
        completed_commands=torch.zeros_like(memory.completed_commands),
        measured_paths=torch.zeros_like(memory.measured_paths),
        visual_innovations=torch.zeros_like(memory.visual_innovations),
        state_innovations=torch.zeros_like(memory.state_innovations),
        innovation_valid=torch.zeros_like(memory.innovation_valid),
        path_valid=torch.zeros_like(memory.path_valid),
        delta_t_s=memory.delta_t_s,
        history_valid=memory.history_valid,
    )


def _proprio_history_memory(memory: V181Memory) -> V181Memory:
    """Keep only completed commands/readbacks and their causal clock."""

    return V181Memory(
        actual_observer_slots=torch.zeros_like(memory.actual_observer_slots),
        actual_view_valid=torch.zeros_like(memory.actual_view_valid),
        completed_commands=memory.completed_commands,
        measured_paths=memory.measured_paths,
        visual_innovations=torch.zeros_like(memory.visual_innovations),
        state_innovations=torch.zeros_like(memory.state_innovations),
        innovation_valid=torch.zeros_like(memory.innovation_valid),
        path_valid=memory.path_valid,
        delta_t_s=memory.delta_t_s,
        history_valid=memory.history_valid,
    )


def _servo_memory(memory: V181Memory) -> V181Memory:
    commanded = torch.cat(
        (memory.measured_paths[:, :, :1], memory.completed_commands), dim=2
    )
    servo = (memory.measured_paths - commanded) * memory.path_valid[..., None]
    return V181Memory(
        actual_observer_slots=memory.actual_observer_slots,
        actual_view_valid=memory.actual_view_valid,
        completed_commands=memory.completed_commands,
        measured_paths=memory.measured_paths,
        visual_innovations=torch.zeros_like(memory.visual_innovations),
        state_innovations=servo,
        innovation_valid=memory.innovation_valid,
        path_valid=memory.path_valid,
        delta_t_s=memory.delta_t_s,
        history_valid=memory.history_valid,
    )


def _servo_lag_memory(memory: V181Memory) -> V181Memory:
    predicted = fixed_servo_lag_path(
        memory.measured_paths[:, :, 0], memory.completed_commands
    )
    error = (memory.measured_paths - predicted) * memory.path_valid[..., None]
    return V181Memory(
        actual_observer_slots=memory.actual_observer_slots,
        actual_view_valid=memory.actual_view_valid,
        completed_commands=memory.completed_commands,
        measured_paths=memory.measured_paths,
        visual_innovations=torch.zeros_like(memory.visual_innovations),
        state_innovations=error,
        innovation_valid=memory.innovation_valid,
        path_valid=memory.path_valid,
        delta_t_s=memory.delta_t_s,
        history_valid=memory.history_valid,
    )


def _innovation_memory(
    memory: V181Memory,
    *,
    keep_visual: bool,
    keep_state: bool,
) -> V181Memory:
    """Remove one innovation modality from both current and historical inputs."""

    return V181Memory(
        actual_observer_slots=memory.actual_observer_slots,
        actual_view_valid=memory.actual_view_valid,
        completed_commands=memory.completed_commands,
        measured_paths=memory.measured_paths,
        visual_innovations=(
            memory.visual_innovations
            if keep_visual
            else torch.zeros_like(memory.visual_innovations)
        ),
        state_innovations=(
            memory.state_innovations
            if keep_state
            else torch.zeros_like(memory.state_innovations)
        ),
        innovation_valid=memory.innovation_valid,
        path_valid=memory.path_valid,
        delta_t_s=memory.delta_t_s,
        history_valid=memory.history_valid,
    )


def fish_variant_inputs(
    variant: str,
    measurement: Any,
    current_view_valid: torch.Tensor,
    conditioning: V181FishConditioning,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    V181Innovation,
    torch.Tensor,
    torch.Tensor,
    V181Memory,
]:
    """Capacity-matched null-token substitutions for trained Actor variants."""

    if variant not in FISH_VARIANTS:
        raise V181TrainingError(f"unknown fish conditioning variant: {variant}")
    if current_view_valid.shape != (
        len(measurement.local_observer_slots),
        VIEWS,
    ) or current_view_valid.dtype != torch.bool:
        raise V181TrainingError("current view mask must be bool [B,2]")
    dense = measurement.policy_dense_tokens
    actual = measurement.local_observer_slots
    innovation = conditioning.innovation
    predicted = conditioning.predicted_missing_slots
    prediction_valid = conditioning.prediction_valid
    memory = conditioning.memory

    zero_visual = torch.zeros_like(innovation.visual)
    zero_state = torch.zeros_like(innovation.state)
    null_error = V181Innovation(
        visual=zero_visual,
        state=zero_state,
        actual_slot_delta=torch.zeros_like(innovation.actual_slot_delta),
        innovation_valid=torch.zeros_like(innovation.innovation_valid),
        path_valid=torch.zeros_like(innovation.path_valid),
    )
    zero_predicted = torch.zeros_like(predicted)
    zero_prediction_valid = torch.zeros_like(prediction_valid)

    if variant == "q_only":
        return (
            torch.zeros_like(dense),
            torch.zeros_like(actual),
            null_error,
            zero_predicted,
            zero_prediction_valid,
            _null_memory(memory, keep_actual=False),
        )
    if variant == "dense_only":
        return (
            dense,
            torch.zeros_like(actual),
            null_error,
            zero_predicted,
            zero_prediction_valid,
            _null_memory(memory, keep_actual=False),
        )
    if variant == "observer_innovation_only":
        return (
            torch.zeros_like(dense),
            actual,
            innovation,
            predicted,
            prediction_valid,
            memory,
        )
    if variant == "current_only":
        return (
            dense,
            actual,
            null_error,
            zero_predicted,
            zero_prediction_valid,
            _null_memory(memory, keep_actual=False),
        )
    if variant == "visual_history":
        return (
            dense,
            actual,
            null_error,
            zero_predicted,
            zero_prediction_valid,
            _visual_history_memory(memory),
        )
    if variant == "proprio_history":
        return (
            dense,
            actual,
            null_error,
            zero_predicted,
            zero_prediction_valid,
            _proprio_history_memory(memory),
        )
    if variant == "current_innovation":
        return (
            dense,
            actual,
            innovation,
            zero_predicted,
            zero_prediction_valid,
            _null_memory(memory, keep_actual=False),
        )
    if variant == "actual_history":
        return (
            dense,
            actual,
            null_error,
            zero_predicted,
            zero_prediction_valid,
            _null_memory(memory, keep_actual=True),
        )
    if variant == "temporal_delta":
        temporal = V181Innovation(
            visual=innovation.actual_slot_delta,
            state=zero_state,
            actual_slot_delta=innovation.actual_slot_delta,
            innovation_valid=innovation.innovation_valid,
            path_valid=innovation.path_valid,
        )
        return (
            dense,
            actual,
            temporal,
            zero_predicted,
            zero_prediction_valid,
            _null_memory(memory, keep_actual=True),
        )
    if variant == "lewm_transition":
        predicted_context = V181Innovation(
            visual=conditioning.predicted_transition,
            state=zero_state,
            actual_slot_delta=innovation.actual_slot_delta,
            innovation_valid=innovation.innovation_valid,
            path_valid=innovation.path_valid,
        )
        return (
            dense,
            actual,
            predicted_context,
            zero_predicted,
            zero_prediction_valid,
            _null_memory(memory, keep_actual=True),
        )
    if variant == "servo_only":
        servo = V181Innovation(
            visual=zero_visual,
            state=conditioning.servo_error,
            actual_slot_delta=innovation.actual_slot_delta,
            innovation_valid=innovation.innovation_valid,
            path_valid=innovation.path_valid,
        )
        return (
            dense,
            actual,
            servo,
            zero_predicted,
            zero_prediction_valid,
            _servo_memory(memory),
        )
    if variant == "servo_lag":
        lag = V181Innovation(
            visual=zero_visual,
            state=conditioning.servo_lag_error,
            actual_slot_delta=innovation.actual_slot_delta,
            innovation_valid=innovation.innovation_valid,
            path_valid=innovation.path_valid,
        )
        return (
            dense,
            actual,
            lag,
            zero_predicted,
            zero_prediction_valid,
            _servo_lag_memory(memory),
        )
    if variant == "visual_only":
        visual_only = V181Innovation(
            visual=innovation.visual,
            state=zero_state,
            actual_slot_delta=innovation.actual_slot_delta,
            innovation_valid=innovation.innovation_valid,
            path_valid=innovation.path_valid,
        )
        return (
            dense,
            actual,
            visual_only,
            predicted,
            prediction_valid,
            _innovation_memory(memory, keep_visual=True, keep_state=False),
        )
    if variant == "state_only":
        state_only = V181Innovation(
            visual=zero_visual,
            state=innovation.state,
            actual_slot_delta=innovation.actual_slot_delta,
            innovation_valid=innovation.innovation_valid,
            path_valid=innovation.path_valid,
        )
        return (
            dense,
            actual,
            state_only,
            zero_predicted,
            zero_prediction_valid,
            _innovation_memory(memory, keep_visual=False, keep_state=True),
        )
    return dense, actual, innovation, predicted, prediction_valid, memory


def train_v181_fish(
    model: V181TipFreeController,
    archive: V181FishArchive,
    feature_bank: V181DeviceFeatureBank,
    output_root: str | Path,
    *,
    device: torch.device,
    seed: int,
    variant: str = "full",
    updates: int = FISH_UPDATES,
    batch_size: int = FISH_EFFECTIVE_BATCH,
    resume: bool = True,
) -> dict[str, Any]:
    if int(seed) not in SEEDS:
        raise V181TrainingError(f"fish seed must be one of {SEEDS}")
    if variant not in FISH_VARIANTS:
        raise V181TrainingError(f"fish variant must be one of {FISH_VARIANTS}")
    if int(updates) <= 0 or int(batch_size) <= 0:
        raise V181TrainingError("fish updates and batch size must be positive")
    seed_v181(seed)
    output = (
        Path(output_root).expanduser().resolve()
        / f"variant-{variant}"
        / f"seed-{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "final.pt"
    latest_path = output / "latest.pt"
    if resume and final_path.is_file():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != "18.1"
            or payload.get("phase") != "fish"
            or int(payload.get("seed", -1)) != int(seed)
            or not isinstance(payload.get("model"), dict)
            or not isinstance(payload.get("metrics"), dict)
            or payload["metrics"].get("variant") != variant
        ):
            raise V181TrainingError("completed fish artifact is incompatible")
        model.load_state_dict(payload["model"], strict=True)
        model.to(device)
        return dict(payload["metrics"])
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        (
            {"params": model.observer.parameters(), "lr": FCPG_LR},
            {"params": model.actor.parameters(), "lr": ACTOR_LR},
            {"params": model.lewm.parameters(), "lr": LEWM_LR},
        )
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_lambda(
            step, total=int(updates), warmup=FISH_WARMUP_UPDATES
        ),
    )
    population = archive.pair_ordinals("train")
    generator = np.random.default_rng(seed)
    sums = {
        "actor": 0.0,
        "world": 0.0,
        "visual": 0.0,
        "visual_magnitude": 0.0,
        "visual_direction": 0.0,
        "state": 0.0,
    }
    probe_trace: list[dict[str, Any]] = []
    observer_reference: torch.Tensor | None = None
    next_update = 1
    if resume:
        payload = _resume_payload(latest_path, model, phase="fish", seed=seed)
        if payload is not None:
            training_state = payload["training_state"]
            if (
                training_state.get("variant") != variant
                or int(training_state.get("updates", -1)) != int(updates)
                or int(training_state.get("batch_size", -1)) != int(batch_size)
            ):
                raise V181TrainingError("fish resume training contract differs")
            optimizer_state = training_state.get("optimizer")
            scheduler_state = training_state.get("scheduler")
            generator_state = training_state.get("generator")
            raw_sums = training_state.get("sums")
            if (
                not isinstance(optimizer_state, dict)
                or not isinstance(scheduler_state, dict)
                or not isinstance(generator_state, dict)
                or not isinstance(raw_sums, dict)
            ):
                raise V181TrainingError("fish resume omits exact loop state")
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
            generator.bit_generator.state = generator_state
            sums = {name: float(raw_sums[name]) for name in sums}
            raw_probe_trace = training_state.get("probe_trace", [])
            raw_reference = training_state.get("observer_reference")
            if not isinstance(raw_probe_trace, list) or not isinstance(
                raw_reference, torch.Tensor
            ):
                raise V181TrainingError("fish resume omits training probe state")
            probe_trace = [dict(item) for item in raw_probe_trace]
            observer_reference = raw_reference.float().cpu()
            next_update = int(training_state.get("next_update", -1))
            if not 1 <= next_update <= int(updates) + 1:
                raise V181TrainingError("fish resume cursor is malformed")

    audit_ordinals = population[: min(16, len(population))]
    audit_pair = archive.pair_batch(audit_ordinals, partition="train")

    @torch.no_grad()
    def collect_probe(probe_update: int) -> dict[str, Any]:
        nonlocal observer_reference
        current_views = torch.from_numpy(
            deterministic_fish_view_valid(audit_pair.source_frame, seed=seed)
        ).to(device)
        target_views = torch.from_numpy(
            deterministic_fish_view_valid(audit_pair.target_frame, seed=seed)
        ).to(device)
        measurement, current_views = _measure_frames(
            model, feature_bank, audit_pair.source_frame, current_views
        )
        target_slots, target_views = _fixed_slots_frames(
            model, feature_bank, audit_pair.target_frame, target_views
        )
        conditioning = _fish_memory(
            model,
            feature_bank,
            audit_pair,
            device=device,
            seed=seed,
            update=0,
        )
        current_q = torch.from_numpy(audit_pair.current_q).to(device)
        target_commands = torch.from_numpy(audit_pair.target_commands).to(device)
        measured_path = torch.from_numpy(audit_pair.measured_q_path).to(device)
        full_inputs = fish_variant_inputs(
            "full", measurement, current_views, conditioning
        )

        def actor_commands(
            dense: torch.Tensor,
            slots: torch.Tensor,
            error: V181Innovation,
            predicted: torch.Tensor,
            prediction_valid: torch.Tensor,
            memory: V181Memory,
        ) -> torch.Tensor:
            with _autocast(device):
                return model.actor(
                    dense,
                    slots,
                    current_views,
                    error.visual,
                    error.state,
                    error.innovation_valid,
                    error.path_valid,
                    predicted,
                    prediction_valid,
                    current_q,
                    memory,
                ).commands

        factual = actor_commands(*full_inputs)
        empty_error = empty_v181_innovation(
            len(current_q), device=device, dtype=measurement.policy_dense_tokens.dtype
        )
        zero_innovation = actor_commands(
            full_inputs[0],
            full_inputs[1],
            empty_error,
            full_inputs[3],
            full_inputs[4],
            _innovation_memory(full_inputs[5], keep_visual=False, keep_state=False),
        )
        q_only = actor_commands(
            *fish_variant_inputs("q_only", measurement, current_views, conditioning)
        )

        def action_magnitude(commands: torch.Tensor) -> dict[str, float]:
            previous = torch.cat((current_q[:, None], commands[:, :-1]), dim=1)
            increments = commands - previous
            return {
                "increment_abs_mean": float(increments.float().abs().mean()),
                "increment_abs_max": float(increments.float().abs().max()),
                "normalized_increment_abs_mean": float(
                    (increments.float().abs() / model.actor.step_cap[None, None]).mean()
                ),
            }

        if observer_reference is None:
            observer_reference = (
                measurement.local_observer_slots.detach().float().cpu().clone()
            )
        world = v181_world_training_probe(
            model,
            measurement.local_observer_slots,
            target_slots,
            current_views,
            target_views,
            current_q,
            target_commands,
            measured_path,
            observer_reference=observer_reference,
            seed=seed + int(probe_update),
        )
        return {
            "update": int(probe_update),
            "world": world,
            "actor": {
                "factual": action_magnitude(factual),
                "innovation_zero": action_magnitude(zero_innovation),
                "q_only": action_magnitude(q_only),
                "innovation_zero_response_rms": float(
                    (factual - zero_innovation).float().square().mean().sqrt()
                ),
                "q_only_response_rms": float(
                    (factual - q_only).float().square().mean().sqrt()
                ),
            },
        }

    # A fish-only run enters here with the deliberately zero-initialized Actor,
    # whereas ``fish-pretrained`` enters with the Actor already learned by the
    # archive and dense20k stages.  The former must still prove the exact hold
    # fixed point; requiring the latter to be zero would reject the very weights
    # this continuation is meant to train.
    actor_head_started_zero = not bool(
        model.actor.action_head.weight.detach().count_nonzero()
        or model.actor.action_head.bias.detach().count_nonzero()
    )
    if not probe_trace:
        initial_probe = collect_probe(0)
        if (
            actor_head_started_zero
            and initial_probe["actor"]["factual"]["increment_abs_max"] > 1.0e-7
        ):
            raise V181TrainingError("zero-initialized Actor does not hold current q")
        probe_trace.append(initial_probe)

    def save_latest(update: int) -> None:
        running = {
            name: value / max(1, update)
            for name, value in sums.items()
        }
        save_v181_checkpoint(
            latest_path,
            model,
            phase="fish",
            seed=seed,
            step=update,
            metrics={"variant": variant, **running},
            training_state={
                "variant": variant,
                "updates": int(updates),
                "batch_size": int(batch_size),
                "next_update": int(update) + 1,
                "sums": dict(sums),
                "probe_trace": probe_trace,
                "observer_reference": observer_reference,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "generator": generator.bit_generator.state,
            },
        )

    for update in range(next_update, int(updates) + 1):
        ordinals = generator.choice(
            population, size=int(batch_size), replace=len(population) < int(batch_size)
        )
        pair = archive.pair_batch(ordinals, partition="train")
        current_views = torch.from_numpy(
            deterministic_fish_view_valid(
                pair.source_frame + update * 13, seed=seed
            )
        ).to(device)
        target_views = torch.from_numpy(
            deterministic_fish_view_valid(
                pair.target_frame + update * 13, seed=seed
            )
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            current_measurement, current_views = _measure_frames(
                model, feature_bank, pair.source_frame, current_views
            )
            target_slots, target_views = _fixed_slots_frames(
                model, feature_bank, pair.target_frame, target_views
            )
            conditioning = _fish_memory(
                model,
                feature_bank,
                pair,
                device=device,
                seed=seed,
                update=update,
            )
            current_q = torch.from_numpy(pair.current_q).to(device)
            target_commands = torch.from_numpy(pair.target_commands).to(device)
            measured_path = torch.from_numpy(pair.measured_q_path).to(device)
            (
                actor_dense,
                actor_slots,
                actor_innovation,
                predicted_missing,
                prediction_valid,
                actor_memory,
            ) = fish_variant_inputs(
                variant, current_measurement, current_views, conditioning
            )
            actor_output = model.actor(
                actor_dense,
                actor_slots,
                current_views,
                actor_innovation.visual,
                actor_innovation.state,
                actor_innovation.innovation_valid,
                actor_innovation.path_valid,
                predicted_missing,
                prediction_valid,
                current_q,
                actor_memory,
            )
            actor_loss = v181_actor_objective(
                actor_output.commands,
                target_commands,
                step_cap=model.actor.step_cap,
            )
            prediction = model.lewm(
                current_measurement.local_observer_slots.detach(),
                current_views,
                current_q,
                target_commands,
            )
            world_loss, world_pieces = v181_world_objective(
                prediction,
                current_measurement.local_observer_slots.detach(),
                target_slots,
                current_views,
                target_views,
                measured_path,
                step_cap=model.actor.step_cap,
            )
            total = actor_loss + world_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()
        values = {
            "actor": actor_loss,
            "world": world_pieces["world"],
            "visual": world_pieces["visual"],
            "visual_magnitude": world_pieces["visual_magnitude"],
            "visual_direction": world_pieces["visual_direction"],
            "state": world_pieces["state"],
        }
        for name, value in values.items():
            sums[name] += float(value.detach())
        running = {name: value / update for name, value in sums.items()}
        if update % CHECKPOINT_EVERY == 0:
            probe = collect_probe(update)
            probe["gradient_norm"] = {
                "observer_policy_adapter": _module_gradient_norm(model.observer),
                "actor": _module_gradient_norm(model.actor),
                "lewm": _module_gradient_norm(model.lewm),
            }
            probe_trace.append(probe)
            save_latest(update)
            _json_write(
                output / "train-metrics.json",
                {
                    "version": "18.1",
                    "seed": seed,
                    "variant": variant,
                    "updates": update,
                    **running,
                    "training_probe_trace": probe_trace,
                    "feature_bank": feature_bank.stats(),
                },
            )
    metrics: dict[str, Any] = {
        "version": "18.1",
        "phase": "fish",
        "seed": seed,
        "variant": variant,
        "updates": int(updates),
        **{name: value / int(updates) for name, value in sums.items()},
        "training_probe_trace": probe_trace,
        "feature_bank": feature_bank.stats(),
    }
    save_v181_checkpoint(
        final_path,
        model,
        phase="fish",
        seed=seed,
        step=int(updates),
        metrics=metrics,
    )
    _json_write(output / "train-metrics.json", metrics)
    return metrics


__all__ = [
    "ACTOR_LR",
    "ARCHIVE_ACTOR_BATCH",
    "ARCHIVE_ACTOR_PASSES",
    "ARCHIVE_ACTOR_LR",
    "DENSE_ADAPTATION_BATCH",
    "DENSE_ADAPTATION_PASSES",
    "FCPG_LR",
    "FISH_VARIANTS",
    "PRIMARY_MATCHED_VARIANTS",
    "FISH_UPDATES",
    "GPU_PAGE_BANK_BYTES",
    "LEWM_LR",
    "SEEDS",
    "V181DeviceFeatureBank",
    "V181DeviceObserverBank",
    "V181DeviceSubsetFeatureBank",
    "V181FishConditioning",
    "V181TrainingError",
    "extract_dinov2_tokens",
    "fixed_servo_lag_path",
    "fish_variant_inputs",
    "load_frozen_dinov2",
    "load_v181_checkpoint",
    "prepare_fish_dino_cache",
    "prepare_temporal_dino_cache",
    "save_v181_checkpoint",
    "seed_v181",
    "train_v181_fish",
    "train_v181_archive_actor",
    "train_v181_dense_actor_adaptation",
    "train_v181_world",
    "v181_actor_objective",
    "v181_training_representation_statistics",
    "v181_world_training_probe",
    "v181_world_objective",
]
