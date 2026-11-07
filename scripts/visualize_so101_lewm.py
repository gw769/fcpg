#!/usr/bin/env python3
"""Create an honest visualization of a trained SO-101 LeWM.

The model predicts image latents, not pixels.  The RGB shown on the
right side of the report is therefore an explicitly labelled nearest neighbor
from a training-split episode.  Held-out target RGB is not read for prediction
or neighbor selection; it is encoded only afterward for evaluation and shown
only as a reference image.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from fcpg.so101_kinematics import SO101MuJoCoKinematics
from fcpg.so101_multiscale_memory import MultiScaleHistoryLeWM
from fcpg.so101_v9 import MEMORY_PROTOCOLS
from fcpg.so101_state_residual_lewm import (
    StateResidualLeWM,
    TemporalStateResidualLeWM,
)
from scripts.train_so101_lewm import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    Cache as TrainingCache,
    build_model,
    episode_splits,
    parse_exclusions,
)


REPORT_FORMAT = "so101-lewm-visualization-v1"
CANVAS_SIZE = (1280, 820)
ACTUAL_COLOR = "#087e8b"
PREDICTED_COLOR = "#d1495b"
DEFAULT_SO101_MODEL = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "so_arm100"
    / "Simulation"
    / "SO101"
    / "so101_new_calib.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a held-out SO-101 episode without presenting retrieved "
            "training RGB as model-generated RGB."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache",
        action="append",
        type=Path,
        help=(
            "Override the checkpoint cache list for a local replay. Repeat for "
            "multiple caches; model weights and model contract stay unchanged."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device. Use cuda:0 after training; CUDA1 is forbidden.",
    )
    parser.add_argument(
        "--episode",
        metavar="SOURCE:ID",
        help="Held-out episode to visualize. Defaults to the first eligible episode.",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Generate one held-out episode report for every semantic task.",
    )
    parser.add_argument(
        "--source",
        help=(
            "Limit episode selection to one checkpoint cache source, for example "
            "so101_fruit_front. The retrieval/ridge training bank remains unchanged."
        ),
    )
    parser.add_argument(
        "--cache-base",
        type=Path,
        help=(
            "Base directory for resolving relative cache paths embedded in the "
            "checkpoint. This never changes cache identity or order."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("teacher-forced", "open-loop", "prediction-cycle"),
        default="teacher-forced",
        help=(
            "teacher-forced predicts every frameskip-horizon target from prior "
            "observations; open-loop recursively rolls one frameskip phase; "
            "prediction-cycle pauses the current observation and recursively "
            "plays a fresh multi-step forecast before advancing it."
        ),
    )
    parser.add_argument(
        "--cycle-horizon",
        type=int,
        default=8,
        help="Recursive frameskip horizons per prediction-cycle observation.",
    )
    parser.add_argument(
        "--media",
        choices=("mp4", "gif", "both"),
        default="mp4",
        help="Animation format. MP4 is the compact default.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--so101-model",
        type=Path,
        default=DEFAULT_SO101_MODEL if DEFAULT_SO101_MODEL.is_file() else None,
        help=(
            "Official SO-101 MuJoCo XML used to convert six-joint predictions "
            "into base-frame gripper XYZ. Omit only when the model is unavailable."
        ),
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-3,
        help="L2 penalty for the standardized train-only ridge decoder.",
    )
    parser.add_argument(
        "--bank-max-per-group",
        type=int,
        default=0,
        help=(
            "Maximum retrieval frames per source+task group; 0 keeps every "
            "training-split frame. Ridge fitting always uses every train frame."
        ),
    )
    parser.add_argument(
        "--retrieval-max-command-joint-rmse",
        type=float,
        default=8.0,
        help=(
            "Hard six-joint pose gate for visualization retrieval, in the dataset's "
            "joint units. Candidate training frames farther from the final commanded "
            "joint target for a forecast horizon are rejected; unmatched predictions "
            "are shown explicitly instead of borrowing a geometrically wrong frame."
        ),
    )
    parser.add_argument(
        "--retrieval-max-arm-joint-error",
        type=float,
        default=12.0,
        help=(
            "Hard maximum absolute error for each of the five arm joints. This "
            "prevents one large kinematic mismatch from being hidden by the RMS."
        ),
    )
    parser.add_argument(
        "--retrieval-max-gripper-error",
        type=float,
        default=10.0,
        help="Hard absolute gripper-state error for visualization retrieval.",
    )
    parser.add_argument(
        "--render-stride",
        type=int,
        default=1,
        help="Render every Nth prediction while retaining all predictions in metrics.json.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        help="Animation FPS. Defaults to source-time playback after render subsampling.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if not np.isfinite(args.ridge_alpha) or args.ridge_alpha < 0:
        parser.error("--ridge-alpha must be non-negative")
    if args.bank_max_per_group < 0:
        parser.error("--bank-max-per-group must be non-negative")
    if (
        not np.isfinite(args.retrieval_max_command_joint_rmse)
        or args.retrieval_max_command_joint_rmse <= 0
    ):
        parser.error("--retrieval-max-command-joint-rmse must be positive")
    if (
        not np.isfinite(args.retrieval_max_arm_joint_error)
        or args.retrieval_max_arm_joint_error <= 0
    ):
        parser.error("--retrieval-max-arm-joint-error must be positive")
    if (
        not np.isfinite(args.retrieval_max_gripper_error)
        or args.retrieval_max_gripper_error <= 0
    ):
        parser.error("--retrieval-max-gripper-error must be positive")
    if args.render_stride <= 0:
        parser.error("--render-stride must be positive")
    if args.cycle_horizon <= 0:
        parser.error("--cycle-horizon must be positive")
    if args.output_fps is not None and (
        not np.isfinite(args.output_fps) or args.output_fps <= 0
    ):
        parser.error("--output-fps must be positive")
    if args.device == "cuda:1" or args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    if args.episode and args.all_tasks:
        parser.error("--episode and --all-tasks are mutually exclusive")
    return args


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_write_text(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
    atomic_write_text(path, text + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheView:
    cache_id: int
    root: Path
    data: TrainingCache
    source: str
    joint_names: tuple[str, ...]

    @property
    def manifest(self) -> dict[str, Any]:
        return self.data.manifest

    @property
    def dataset_id(self) -> str:
        return self.data.dataset_id


@dataclass(frozen=True)
class EpisodeChoice:
    cache: CacheView
    episode: int
    rows: np.ndarray


@dataclass
class PredictionResult:
    mode: str
    destination_local: np.ndarray
    display_input_local: np.ndarray
    baseline_local: np.ndarray
    tasks: np.ndarray
    context_sizes: np.ndarray
    predicted_latents: np.ndarray
    predicted_states: np.ndarray | None = None


@dataclass(frozen=True)
class BankGroup:
    cache_id: int
    source: str
    task: int
    rows: np.ndarray
    latents: np.ndarray
    available_train_frames: int
    sampling: str


SplitAssignments = dict[tuple[str, int], str]


@dataclass(frozen=True)
class RetrievalResult:
    rows: np.ndarray
    latent_mse: np.ndarray
    command_joint_rmse: np.ndarray
    command_arm_max_error: np.ndarray
    command_gripper_error: np.ndarray
    valid: np.ndarray


class RidgeAccumulator:
    """Accumulate train-only sufficient statistics without storing all latents."""

    def __init__(self) -> None:
        self.count = 0
        self.sum_x: np.ndarray | None = None
        self.sum_y: np.ndarray | None = None
        self.xtx: np.ndarray | None = None
        self.xty: np.ndarray | None = None

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        if x64.ndim != 2 or y64.ndim != 2 or len(x64) != len(y64):
            raise ValueError(f"invalid ridge batch shapes: {x64.shape}, {y64.shape}")
        if y64.shape[1] != 6:
            raise ValueError(f"ridge target must have six joints, got {y64.shape}")
        if self.sum_x is None:
            self.sum_x = np.zeros(x64.shape[1], dtype=np.float64)
            self.sum_y = np.zeros(y64.shape[1], dtype=np.float64)
            self.xtx = np.zeros((x64.shape[1], x64.shape[1]), dtype=np.float64)
            self.xty = np.zeros((x64.shape[1], y64.shape[1]), dtype=np.float64)
        assert self.sum_y is not None and self.xtx is not None and self.xty is not None
        self.count += len(x64)
        self.sum_x += x64.sum(axis=0)
        self.sum_y += y64.sum(axis=0)
        self.xtx += x64.T @ x64
        self.xty += x64.T @ y64

    def fit(self, alpha: float) -> "RidgeDecoder":
        if self.count == 0 or self.sum_x is None or self.sum_y is None:
            raise ValueError("cannot fit ridge decoder without training rows")
        assert self.xtx is not None and self.xty is not None
        x_mean = self.sum_x / self.count
        y_mean = self.sum_y / self.count
        centered_xx = self.xtx - self.count * np.outer(x_mean, x_mean)
        centered_xy = self.xty - self.count * np.outer(x_mean, y_mean)
        variance = np.maximum(np.diag(centered_xx) / self.count, 1e-12)
        x_scale = np.sqrt(variance)
        inv_scale = 1.0 / x_scale
        covariance = centered_xx / self.count
        cross_covariance = centered_xy / self.count
        standardized_covariance = covariance * np.outer(inv_scale, inv_scale)
        standardized_cross = cross_covariance * inv_scale[:, None]
        system = standardized_covariance + alpha * np.eye(len(x_mean), dtype=np.float64)
        try:
            weights = np.linalg.solve(system, standardized_cross)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(system, standardized_cross, rcond=None)[0]
        if not np.all(np.isfinite(weights)):
            raise FloatingPointError("ridge fit produced non-finite weights")
        return RidgeDecoder(
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            weights=weights,
            alpha=alpha,
            training_rows=self.count,
        )


@dataclass(frozen=True)
class RidgeDecoder:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    weights: np.ndarray
    alpha: float
    training_rows: int

    def predict(self, latent: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(latent, dtype=np.float64) - self.x_mean) / self.x_scale
        result = self.y_mean + standardized @ self.weights
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("ridge decoder produced non-finite joint states")
        return result


def resolve_embedded_cache(
    raw_path: str | Path,
    checkpoint: Path,
    cache_base: Path | None,
) -> Path:
    embedded = Path(raw_path).expanduser()
    if embedded.is_absolute():
        candidates = [embedded]
        if cache_base is not None:
            candidates.append(cache_base.expanduser().resolve() / embedded.name)
    else:
        candidates = []
        if cache_base is not None:
            resolved_base = cache_base.expanduser().resolve()
            candidates.extend([resolved_base / embedded, resolved_base / embedded.name])
        candidates.extend(
            [
                Path.cwd() / embedded,
                Path(__file__).resolve().parent / embedded,
                checkpoint.parent / embedded,
            ]
        )
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique_candidates:
            unique_candidates.append(resolved)
    for candidate in unique_candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    checked = ", ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(
        f"embedded cache path {str(embedded)!r} could not be resolved; checked: {checked}"
    )


#: Runtime protocol for the frozen B2-head dynamics checkpoint.  The B2 head
#: genuinely differs from V9 (41 retrained dynamics tensors), so it gets its own
#: string rather than being relabelled to slip past a loader whitelist.  It is
#: deliberately NOT a member of ``MEMORY_PROTOCOLS``: membership there routes to
#: the 370-tensor ``MultiScaleHistoryLeWM`` builder, and ``strict=True`` would
#: fail against the checkpoint's 344 bare keys.
B2HEAD_RUNTIME_PROTOCOL = "so101-state-residual-lewm-b2head-runtime-v1"

#: Protocols that build a bare ``TemporalStateResidualLeWM`` (base + head, no
#: memory adapter).
TEMPORAL_STATE_RESIDUAL_PROTOCOLS = frozenset(
    {
        "so101-state-residual-lewm-v5-temporal",
        B2HEAD_RUNTIME_PROTOCOL,
    }
)

DEPLOYMENT_CONTRACT_FILENAME = "deployment_contract.json"
DEPLOYMENT_CONTRACT_FORMAT = "so101-b2head-deployment-contract-v1"


def load_deployment_contract(
    checkpoint_path: Path, checkpoint_sha256: str
) -> dict[str, Any] | None:
    """Read a sidecar runtime contract for a checkpoint that carries none.

    Returns ``None`` when no sidecar is present, so checkpoints that already
    embed ``run_config.contract`` are entirely unaffected.

    The sidecar is bound to its checkpoint by SHA-256: a contract whose recorded
    hash does not match the file just loaded is rejected rather than applied.
    That makes it impossible to graft one model's geometry onto another's
    weights, or to keep using a stale contract after a checkpoint is replaced.
    Weights are never sourced from the sidecar -- only metadata.
    """

    sidecar_path = checkpoint_path.parent / DEPLOYMENT_CONTRACT_FILENAME
    if not sidecar_path.is_file():
        return None
    with sidecar_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"deployment contract must be a JSON object: {sidecar_path}")
    declared_format = payload.get("format")
    if declared_format != DEPLOYMENT_CONTRACT_FORMAT:
        raise ValueError(
            f"unsupported deployment contract format {declared_format!r} "
            f"(expected {DEPLOYMENT_CONTRACT_FORMAT!r}): {sidecar_path}"
        )
    declared_sha = payload.get("source_checkpoint_sha256")
    if not isinstance(declared_sha, str) or not declared_sha:
        raise ValueError(
            f"deployment contract must record source_checkpoint_sha256: {sidecar_path}"
        )
    if declared_sha != checkpoint_sha256:
        raise ValueError(
            "deployment contract is bound to a different checkpoint: sidecar "
            f"records {declared_sha}, loaded checkpoint hashes to "
            f"{checkpoint_sha256} ({sidecar_path})"
        )
    contract = payload.get("contract")
    if not isinstance(contract, dict) or not contract:
        raise ValueError(
            f"deployment contract must carry a non-empty contract block: {sidecar_path}"
        )
    return contract


def checkpoint_contract(
    checkpoint_path: Path,
    cache_base: Path | None,
    cache_override: list[Path] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[Path], int, int, str]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    initial_stat = checkpoint_path.stat()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loaded_stat = checkpoint_path.stat()
    initial_identity = (initial_stat.st_ino, initial_stat.st_size, initial_stat.st_mtime_ns)
    loaded_identity = (loaded_stat.st_ino, loaded_stat.st_size, loaded_stat.st_mtime_ns)
    if initial_identity != loaded_identity:
        raise RuntimeError(f"checkpoint changed while it was being loaded: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    hashed_stat = checkpoint_path.stat()
    hashed_identity = (hashed_stat.st_ino, hashed_stat.st_size, hashed_stat.st_mtime_ns)
    if loaded_identity != hashed_identity:
        raise RuntimeError(f"checkpoint changed while it was being hashed: {checkpoint_path}")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "args" not in checkpoint:
        raise ValueError(
            "checkpoint must be a train_so101_lewm.py checkpoint containing model and args"
        )
    embedded_args = checkpoint["args"]
    if not isinstance(embedded_args, dict):
        raise ValueError("checkpoint args must be a dictionary")
    # A frozen scientific checkpoint may carry no run_config.contract because it
    # was written by a trainer that recorded its fields at run_config top level.
    # Merge a hash-bound sidecar into the in-memory contract so every downstream
    # reader (this function, build_checkpoint_model, checkpoint_action_normalizer,
    # and the realtime planner) sees one consistent block.  The .pt file on disk
    # is never modified.
    deployment_contract = load_deployment_contract(checkpoint_path, checkpoint_sha256)
    if deployment_contract is not None:
        run_config = checkpoint.setdefault("run_config", {})
        if not isinstance(run_config, dict):
            raise ValueError("checkpoint run_config must be a dictionary")
        existing = run_config.get("contract")
        if isinstance(existing, dict) and existing:
            # Refuse to arbitrate between two contracts for the same file.
            if existing != deployment_contract:
                raise ValueError(
                    "checkpoint already carries run_config.contract that differs "
                    f"from its deployment contract sidecar: {checkpoint_path}"
                )
        else:
            run_config["contract"] = dict(deployment_contract)
    contract = checkpoint.get("run_config", {}).get("contract", {})
    raw_caches = cache_override or embedded_args.get("cache") or contract.get("caches")
    if isinstance(raw_caches, (str, Path)):
        raw_caches = [raw_caches]
    if not isinstance(raw_caches, (list, tuple)) or not raw_caches:
        raise ValueError("checkpoint args do not contain a non-empty cache list")
    frameskip = int(embedded_args.get("frameskip", contract.get("frameskip", 0)))
    history = int(embedded_args.get("history", contract.get("history", 0)))
    if frameskip <= 0 or history <= 0:
        raise ValueError(
            f"invalid embedded sequence contract: frameskip={frameskip}, history={history}"
        )
    cache_paths = [
        resolve_embedded_cache(path, checkpoint_path, cache_base) for path in raw_caches
    ]
    return checkpoint, embedded_args, cache_paths, frameskip, history, checkpoint_sha256


def checkpoint_action_normalizer(
    checkpoint: dict[str, Any], embedded_args: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, str]:
    normalization = checkpoint.get("run_config", {}).get("action_normalization")
    if isinstance(normalization, dict):
        center = np.asarray(normalization.get("center"), dtype=np.float32)
        scale = np.asarray(normalization.get("scale"), dtype=np.float32)
        description = str(normalization.get("kind", "checkpoint action normalization"))
    elif checkpoint.get("run_config", {}).get("contract", {}).get("protocol") in {
        "so101-state-residual-lewm-v4",
        *TEMPORAL_STATE_RESIDUAL_PROTOCOLS,
        *MEMORY_PROTOCOLS,
    }:
        contract = checkpoint["run_config"]["contract"]
        center = np.asarray(contract.get("normalization_center"), dtype=np.float32)
        scale = np.asarray(contract.get("normalization_scale"), dtype=np.float32)
        description = str(contract.get("normalization"))
    else:
        joint_scale = float(embedded_args["joint_scale"])
        center = np.zeros(6, dtype=np.float32)
        scale = np.full(6, joint_scale, dtype=np.float32)
        description = "absolute joint targets divided by checkpoint joint_scale"
    if center.shape != (6,) or scale.shape != (6,):
        raise ValueError("checkpoint action normalization vectors must have shape (6,)")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ValueError("checkpoint action normalization vectors must be finite")
    if np.any(scale <= 0):
        raise ValueError("checkpoint action normalization scale must be positive")
    return center, scale, description


def build_checkpoint_model(
    checkpoint: dict[str, Any],
    frameskip: int,
    history: int,
    device: torch.device,
) -> torch.nn.Module:
    """Build either the original visual LeWM or its six-state residual wrapper."""
    base = build_model(action_dim=frameskip * 6, history=history)
    contract = checkpoint.get("run_config", {}).get("contract", {})
    protocol = contract.get("protocol")
    if protocol in MEMORY_PROTOCOLS:
        if int(contract.get("state_dim", 0)) != 6:
            raise ValueError("memory checkpoint must use six state dimensions")
        backbone = TemporalStateResidualLeWM(
            base,
            state_dim=6,
            latent_dim=192,
            model_dim=int(contract["backbone_temporal_model_dim"]),
            layers=int(contract["backbone_temporal_layers"]),
            heads=int(contract["backbone_temporal_heads"]),
            ffn_dim=int(contract["backbone_temporal_ffn_dim"]),
            max_steps=int(contract["backbone_temporal_max_steps"]),
            latent_residual_cap=float(contract["backbone_latent_residual_cap"]),
            state_residual_cap=float(contract["backbone_state_residual_cap"]),
        )
        model = MultiScaleHistoryLeWM(
            backbone,
            latent_dim=192,
            model_dim=int(contract["memory_model_dim"]),
            heads=int(contract["memory_heads"]),
            latent_residual_cap=float(contract["memory_latent_cap"]),
            state_residual_cap=float(contract["memory_state_cap"]),
            recent_history=int(contract.get("recent_history", 3)),
            use_recent_reliability=(
                bool(contract.get("use_recent_reliability", True))
                if protocol == "so101-state-residual-lewm-v7-reliability-memory"
                else False
            ),
            use_fourier_age_encoding=(
                bool(contract.get("use_fourier_age_encoding", False))
                if protocol == "so101-state-residual-lewm-v7-reliability-memory"
                else False
            ),
            freeze_backbone=True,
        )
    elif protocol in {
        "so101-state-residual-lewm-v4",
        *TEMPORAL_STATE_RESIDUAL_PROTOCOLS,
    }:
        if int(contract.get("state_dim", 0)) != 6:
            raise ValueError("state-residual checkpoint must use six state dimensions")
        if protocol in TEMPORAL_STATE_RESIDUAL_PROTOCOLS:
            model = TemporalStateResidualLeWM(
                base,
                state_dim=6,
                latent_dim=192,
                model_dim=int(contract["temporal_model_dim"]),
                layers=int(contract["temporal_layers"]),
                heads=int(contract["temporal_heads"]),
                ffn_dim=int(contract["temporal_ffn_dim"]),
                max_steps=int(contract["temporal_max_steps"]),
                latent_residual_cap=float(contract["latent_residual_cap"]),
                state_residual_cap=float(contract["state_residual_cap"]),
            )
        else:
            model = StateResidualLeWM(
                base,
                state_dim=6,
                latent_dim=192,
                latent_residual_cap=float(contract["latent_residual_cap"]),
                state_residual_cap=float(contract["state_residual_cap"]),
            )
    else:
        model = base
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def load_caches(cache_paths: list[Path]) -> list[CacheView]:
    caches: list[CacheView] = []
    sources: set[str] = set()
    expected_joints: tuple[str, ...] | None = None
    for cache_id, root in enumerate(cache_paths):
        data = TrainingCache(root)
        source = str(data.manifest.get("name", ""))
        if not source:
            raise ValueError(f"cache has no non-empty manifest name: {root}")
        if source in sources:
            raise ValueError(f"checkpoint contains duplicate cache source name {source!r}")
        sources.add(source)
        joint_names = tuple(str(name) for name in data.manifest.get("joint_names", ()))
        if len(joint_names) != 6:
            raise ValueError(f"cache {source!r} must define six joint_names")
        if expected_joints is None:
            expected_joints = joint_names
        elif joint_names != expected_joints:
            raise ValueError(
                f"joint order differs between caches: {expected_joints} versus {joint_names}"
            )
        arrays = {
            "frames": data.frames,
            "actions": data.actions,
            "states": data.states,
            "episodes": data.episodes,
            "tasks": data.tasks,
            "frame_indices": data.frame_indices,
        }
        expected = int(data.manifest["frames"])
        for name, array in arrays.items():
            if len(array) != expected:
                raise ValueError(
                    f"cache {source!r} {name}.npy has {len(array)} rows, expected {expected}"
                )
        if data.frames.ndim != 4 or data.frames.shape[-1] != 3:
            raise ValueError(f"cache {source!r} has invalid frame shape {data.frames.shape}")
        if data.frames.dtype != np.uint8:
            raise ValueError(f"cache {source!r} frames must be uint8, got {data.frames.dtype}")
        if data.actions.shape != (expected, 6) or data.states.shape != (expected, 6):
            raise ValueError(
                f"cache {source!r} requires actions/states shaped ({expected}, 6)"
            )
        if data.action_mean.shape != (6,) or data.action_std.shape != (6,):
            raise ValueError(f"cache {source!r} has invalid action normalization vectors")
        if not np.all(np.isfinite(data.action_std)) or np.any(data.action_std <= 0):
            raise ValueError(f"cache {source!r} has invalid action_std")
        fps = float(data.manifest.get("fps") or 0.0)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"cache {source!r} has invalid fps {data.manifest.get('fps')!r}")
        caches.append(CacheView(cache_id, root, data, source, joint_names))
    return caches


def episode_rows(cache: CacheView, episode: int) -> np.ndarray:
    rows = np.flatnonzero(np.asarray(cache.data.episodes) == episode).astype(np.int64)
    if len(rows) == 0:
        raise ValueError(f"episode does not exist: {cache.source}:{episode}")
    if len(rows) > 1 and not np.all(np.diff(rows) == 1):
        raise ValueError(f"non-contiguous cache rows for {cache.source}:{episode}")
    local_frames = np.asarray(cache.data.frame_indices[rows], dtype=np.int64)
    if len(local_frames) > 1 and not np.all(np.diff(local_frames) == 1):
        raise ValueError(f"non-contiguous frame indices for {cache.source}:{episode}")
    return rows


def heldout_candidates(
    caches: list[CacheView],
    frameskip: int,
    history: int,
    assignments: SplitAssignments,
) -> list[EpisodeChoice]:
    choices: list[EpisodeChoice] = []
    span = frameskip * history
    for cache in caches:
        for raw_episode in np.unique(np.asarray(cache.data.episodes)):
            episode = int(raw_episode)
            if assignments.get((cache.dataset_id, episode)) != "val":
                continue
            rows = episode_rows(cache, episode)
            if len(rows) > span:
                choices.append(EpisodeChoice(cache, episode, rows))
    return sorted(choices, key=lambda item: (item.cache.source, item.episode))


def choose_episode(
    caches: list[CacheView],
    frameskip: int,
    history: int,
    requested: str | None,
    assignments: SplitAssignments,
) -> EpisodeChoice:
    choices = heldout_candidates(caches, frameskip, history, assignments)
    if not choices:
        raise ValueError("checkpoint caches contain no eligible held-out episodes")
    if requested is None:
        return choices[0]
    if ":" not in requested:
        raise ValueError("--episode must use SOURCE:ID syntax")
    source, raw_episode = requested.rsplit(":", 1)
    try:
        episode = int(raw_episode)
    except ValueError as error:
        raise ValueError(f"invalid episode ID in {requested!r}") from error
    matches = [
        choice
        for choice in choices
        if choice.cache.source == source and choice.episode == episode
    ]
    if matches:
        return matches[0]
    known_source = next((cache for cache in caches if cache.source == source), None)
    if known_source is None:
        raise ValueError(
            f"unknown source {source!r}; available sources: "
            + ", ".join(cache.source for cache in caches)
        )
    existing = np.any(np.asarray(known_source.data.episodes) == episode)
    if existing and assignments.get((known_source.dataset_id, episode)) == "train":
        raise ValueError(f"{requested} is a training-split episode, not a held-out episode")
    raise ValueError(
        f"{requested} is not an eligible held-out episode with more than "
        f"{frameskip * history} contiguous frames"
    )


def episode_task_key(choice: EpisodeChoice) -> str:
    raw_tasks = np.unique(np.asarray(choice.cache.data.tasks[choice.rows], dtype=np.int64))
    if len(raw_tasks) != 1:
        raise ValueError(
            f"episode spans multiple tasks: {choice.cache.source}:{choice.episode}"
        )
    return choice.cache.data.task_key(int(raw_tasks[0]))


def choose_one_episode_per_task(
    caches: list[CacheView],
    frameskip: int,
    history: int,
    assignments: SplitAssignments,
) -> list[EpisodeChoice]:
    selected: dict[tuple[str, str], EpisodeChoice] = {}
    for choice in heldout_candidates(caches, frameskip, history, assignments):
        key = (choice.cache.dataset_id, episode_task_key(choice))
        selected.setdefault(key, choice)
    return [selected[key] for key in sorted(selected)]


def report_slug(choice: EpisodeChoice) -> str:
    raw = f"{choice.cache.dataset_id}-{episode_task_key(choice)}-ep{choice.episode}"
    slug = "".join(character if character.isalnum() else "-" for character in raw.lower())
    return "-".join(part for part in slug.split("-") if part)


def training_rows(cache: CacheView, assignments: SplitAssignments) -> np.ndarray:
    episodes = np.asarray(cache.data.episodes)
    train_episodes = [
        int(episode)
        for episode in np.unique(episodes)
        if assignments.get((cache.dataset_id, int(episode))) == "train"
    ]
    if not train_episodes:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.isin(episodes, train_episodes)).astype(np.int64)


@torch.inference_mode()
def encode_rows(
    model: torch.nn.Module,
    cache: CacheView,
    rows: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("rows must be one-dimensional")
    chunks: list[np.ndarray] = []
    device = next(model.parameters()).device
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        frame_array = np.asarray(cache.data.frames[batch_rows]).copy()
        pixels = torch.from_numpy(frame_array).permute(0, 3, 1, 2).float().div_(255.0)
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        pixels = pixels.to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            encoded = model.encode({"pixels": pixels.unsqueeze(1)})["emb"][:, 0]
        latent = encoded.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.all(np.isfinite(latent)):
            raise FloatingPointError(f"non-finite encoded latent in cache {cache.source!r}")
        chunks.append(latent)
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def evenly_sample(rows: np.ndarray, limit: int) -> tuple[np.ndarray, str]:
    if limit == 0 or len(rows) <= limit:
        return rows, "all training-split frames"
    indices = np.linspace(0, len(rows) - 1, num=limit, dtype=np.int64)
    return rows[indices], f"deterministic even subsample of {limit}/{len(rows)} frames"


def build_training_artifacts(
    model: torch.nn.Module,
    caches: list[CacheView],
    assignments: SplitAssignments,
    batch_size: int,
    ridge_alpha: float,
    bank_max_per_group: int,
) -> tuple[RidgeDecoder, dict[tuple[int, int], BankGroup], list[dict[str, Any]]]:
    ridge = RidgeAccumulator()
    latent_parts: dict[tuple[int, int], list[np.ndarray]] = {}
    row_parts: dict[tuple[int, int], list[np.ndarray]] = {}
    group_contract: dict[tuple[int, int], tuple[int, str]] = {}
    bank_summaries: list[dict[str, Any]] = []

    for cache in caches:
        train = training_rows(cache, assignments)
        if len(train) == 0:
            continue
        selected_mask = np.zeros(len(cache.data.frames), dtype=bool)
        train_tasks = np.asarray(cache.data.tasks[train], dtype=np.int64)
        for raw_task in np.unique(train_tasks):
            task = int(raw_task)
            available = train[train_tasks == task]
            selected, sampling = evenly_sample(available, bank_max_per_group)
            selected_mask[selected] = True
            key = (cache.cache_id, task)
            group_contract[key] = (len(available), sampling)
            bank_summaries.append(
                {
                    "source": cache.source,
                    "task": task,
                    "available_train_episode_frames": len(available),
                    "retrieval_frames": len(selected),
                    "sampling": sampling,
                }
            )

        for start in range(0, len(train), batch_size):
            rows = train[start : start + batch_size]
            latents = encode_rows(model, cache, rows, batch_size)
            states = np.asarray(cache.data.states[rows], dtype=np.float32)
            if not np.all(np.isfinite(states)):
                raise FloatingPointError(f"non-finite joint state in cache {cache.source!r}")
            ridge.update(latents, states)
            retained = selected_mask[rows]
            if not np.any(retained):
                continue
            retained_rows = rows[retained]
            retained_latents = latents[retained]
            retained_tasks = np.asarray(cache.data.tasks[retained_rows], dtype=np.int64)
            for raw_task in np.unique(retained_tasks):
                task = int(raw_task)
                task_mask = retained_tasks == task
                key = (cache.cache_id, task)
                latent_parts.setdefault(key, []).append(retained_latents[task_mask])
                row_parts.setdefault(key, []).append(retained_rows[task_mask])

    decoder = ridge.fit(ridge_alpha)
    groups: dict[tuple[int, int], BankGroup] = {}
    for cache in caches:
        for key in sorted(key for key in latent_parts if key[0] == cache.cache_id):
            task = key[1]
            rows = np.concatenate(row_parts[key]).astype(np.int64, copy=False)
            latents = np.concatenate(latent_parts[key]).astype(np.float32, copy=False)
            available, sampling = group_contract[key]
            episodes = np.unique(np.asarray(cache.data.episodes[rows], dtype=np.int64))
            if any(
                assignments.get((cache.dataset_id, int(episode))) != "train"
                for episode in episodes
            ):
                raise AssertionError("held-out episode entered retrieval bank")
            if not np.all(np.asarray(cache.data.tasks[rows], dtype=np.int64) == task):
                raise AssertionError("retrieval bank task mismatch")
            groups[key] = BankGroup(
                cache_id=cache.cache_id,
                source=cache.source,
                task=task,
                rows=rows,
                latents=latents,
                available_train_frames=available,
                sampling=sampling,
            )
    return decoder, groups, bank_summaries


def normalized_action_chunk(
    cache: CacheView,
    global_start: int,
    frameskip: int,
    action_center: np.ndarray,
    action_scale: np.ndarray,
) -> np.ndarray:
    stop = global_start + frameskip
    raw = np.asarray(cache.data.actions[global_start:stop], dtype=np.float32)
    if raw.shape != (frameskip, 6):
        raise ValueError(f"incomplete action chunk at cache row {global_start}")
    center = np.asarray(action_center, dtype=np.float32)
    scale = np.asarray(action_scale, dtype=np.float32)
    if center.shape != (6,) or scale.shape != (6,):
        raise ValueError("action normalization vectors must have shape (6,)")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ValueError("action normalization vectors must be finite")
    if np.any(scale <= 0):
        raise ValueError("action normalization scale must be positive")
    normalized = (raw - center) / scale
    return normalized.reshape(frameskip * 6).astype(np.float32, copy=False)


@torch.inference_mode()
def predict_batch(
    model: torch.nn.Module,
    context_latents: np.ndarray,
    action_chunks: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    latent_tensor = torch.from_numpy(
        np.asarray(context_latents, dtype=np.float32)
    ).to(device)
    action_tensor = torch.from_numpy(
        np.asarray(action_chunks, dtype=np.float32)
    ).to(device)
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
    ):
        action_embedding = model.action_encoder(action_tensor)
        predicted = model.predict(latent_tensor, action_embedding)[:, -1]
    result = predicted.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("model produced a non-finite predicted latent")
    return result


def normalized_state_rows(
    cache: CacheView,
    rows: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    states = np.asarray(cache.data.states[np.asarray(rows, dtype=np.int64)], dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError(f"state rows must have shape [N, 6], got {states.shape}")
    if center.shape != (6,) or scale.shape != (6,) or np.any(scale <= 0):
        raise ValueError("state normalization requires positive six-dimensional vectors")
    normalized = (states - center) / scale
    if not np.all(np.isfinite(normalized)):
        raise FloatingPointError("normalized state contains non-finite values")
    return normalized.astype(np.float32, copy=False)


@torch.inference_mode()
def predict_state_residual_batch(
    model: StateResidualLeWM,
    context_latents: np.ndarray,
    action_chunks: np.ndarray,
    context_states: np.ndarray,
    *,
    residual_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    latent_tensor = torch.from_numpy(
        np.asarray(context_latents, dtype=np.float32)
    ).to(device)
    action_tensor = torch.from_numpy(
        np.asarray(action_chunks, dtype=np.float32)
    ).to(device)
    state_tensor = torch.from_numpy(
        np.asarray(context_states, dtype=np.float32)
    ).to(device)
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
    ):
        _, latent_prediction, _, state_prediction = model.predict_context(
            latent_tensor,
            action_tensor,
            state_tensor,
            residual_scale=residual_scale,
        )
    latent = latent_prediction[:, -1].detach().cpu().numpy().astype(np.float32)
    state = state_prediction[:, -1].detach().cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(latent)) or not np.all(np.isfinite(state)):
        raise FloatingPointError("state-residual model produced a non-finite prediction")
    return latent, state


@torch.inference_mode()
def predict_multiscale_batch(
    model: MultiScaleHistoryLeWM,
    context_latents: np.ndarray,
    action_chunks: np.ndarray,
    context_states: np.ndarray,
    memory_latents: np.ndarray,
    memory_actions: np.ndarray,
    memory_states: np.ndarray,
    memory_age: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the complete V9 memory path for a batch of forecast origins."""
    values = {
        "context_latents": np.asarray(context_latents, dtype=np.float32),
        "action_chunks": np.asarray(action_chunks, dtype=np.float32),
        "context_states": np.asarray(context_states, dtype=np.float32),
        "memory_latents": np.asarray(memory_latents, dtype=np.float32),
        "memory_actions": np.asarray(memory_actions, dtype=np.float32),
        "memory_states": np.asarray(memory_states, dtype=np.float32),
        "memory_age": np.asarray(memory_age, dtype=np.float32),
    }
    batch = values["context_latents"].shape[0]
    if values["context_latents"].ndim != 3:
        raise ValueError("V9 context_latents must have shape [B,T,D]")
    if values["action_chunks"].shape[:2] != values["context_latents"].shape[:2]:
        raise ValueError("V9 action/context histories must align")
    if values["context_states"].shape[:2] != values["context_latents"].shape[:2]:
        raise ValueError("V9 state/context histories must align")
    if values["memory_latents"].ndim != 3 or values["memory_latents"].shape[0] != batch:
        raise ValueError("V9 memory_latents must have shape [B,M,D]")
    if values["memory_actions"].shape[:2] != values["memory_latents"].shape[:2]:
        raise ValueError("V9 memory actions/latents must align")
    if values["memory_states"].shape[:2] != values["memory_latents"].shape[:2]:
        raise ValueError("V9 memory states/latents must align")
    if values["memory_age"].shape != values["memory_latents"].shape[:2]:
        raise ValueError("V9 memory ages must have shape [B,M]")
    if not all(np.all(np.isfinite(value)) for value in values.values()):
        raise FloatingPointError("V9 visualization inputs contain non-finite values")

    device = next(model.parameters()).device
    tensors = {
        name: torch.from_numpy(value).to(device) for name, value in values.items()
    }
    memory_valid = torch.ones(
        tensors["memory_age"].shape,
        dtype=torch.bool,
        device=device,
    )
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
    ):
        _, prediction, _, _, state_prediction = model.predict_context(
            tensors["context_latents"],
            tensors["action_chunks"],
            tensors["context_states"],
            memory_embedding=tensors["memory_latents"],
            memory_actions=tensors["memory_actions"],
            memory_state=tensors["memory_states"],
            memory_age=tensors["memory_age"],
            memory_valid=memory_valid,
        )
    latent = prediction[:, -1].float().cpu().numpy().astype(np.float32, copy=False)
    state = state_prediction[:, -1].float().cpu().numpy().astype(np.float32, copy=False)
    if not np.all(np.isfinite(latent)) or not np.all(np.isfinite(state)):
        raise FloatingPointError("V9 memory model produced a non-finite prediction")
    return latent, state


def teacher_forced_predictions(
    model: torch.nn.Module,
    choice: EpisodeChoice,
    frameskip: int,
    history: int,
    action_center: np.ndarray,
    action_scale: np.ndarray,
    batch_size: int,
    state_center: np.ndarray | None = None,
    state_scale: np.ndarray | None = None,
) -> PredictionResult:
    rows = choice.rows
    frame_count = len(rows)
    destinations = np.arange(frameskip * history, frame_count, dtype=np.int64)
    # These are the only held-out RGB rows encoded before prediction. Every
    # prediction below indexes contexts strictly earlier than its destination.
    observed_latents = encode_rows(model, choice.cache, rows[:-frameskip], batch_size)
    latent_dim = observed_latents.shape[1]
    predicted = np.empty((len(destinations), latent_dim), dtype=np.float32)
    predicted_states = (
        np.empty((len(destinations), 6), dtype=np.float32)
        if isinstance(model, StateResidualLeWM)
        else None
    )
    context_sizes = np.empty(len(destinations), dtype=np.int16)
    if isinstance(model, StateResidualLeWM) and (
        state_center is None or state_scale is None
    ):
        raise ValueError("state-residual visualization requires state normalization")

    grouped: dict[int, list[tuple[int, int, list[int]]]] = {}
    for output_index, destination in enumerate(destinations):
        current = int(destination - frameskip)
        context_size = history
        first = current - (history - 1) * frameskip
        contexts = list(range(first, current + 1, frameskip))
        if len(contexts) != context_size or any(index >= destination for index in contexts):
            raise AssertionError("teacher-forced context includes its target or has wrong length")
        grouped.setdefault(context_size, []).append((output_index, int(destination), contexts))
        context_sizes[output_index] = context_size

    for context_size, records in sorted(grouped.items()):
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            latent_batch = np.stack(
                [observed_latents[contexts] for _, _, contexts in batch_records]
            )
            action_batch = np.stack(
                [
                    np.stack(
                        [
                            normalized_action_chunk(
                                choice.cache,
                                int(rows[context]),
                                frameskip,
                                action_center,
                                action_scale,
                            )
                            for context in contexts
                        ]
                    )
                    for _, _, contexts in batch_records
                ]
            )
            if latent_batch.shape[1] != context_size:
                raise AssertionError("teacher-forced context batch shape mismatch")
            if isinstance(model, StateResidualLeWM):
                state_batch = np.stack(
                    [
                        normalized_state_rows(
                            choice.cache,
                            rows[np.asarray(contexts, dtype=np.int64)],
                            state_center,
                            state_scale,
                        )
                        for _, _, contexts in batch_records
                    ]
                )
                values, normalized_predictions = predict_state_residual_batch(
                    model, latent_batch, action_batch, state_batch
                )
                state_values = normalized_predictions * state_scale + state_center
            else:
                values = predict_batch(model, latent_batch, action_batch)
                state_values = None
            for batch_index, (record, value) in enumerate(
                zip(batch_records, values, strict=True)
            ):
                predicted[record[0]] = value
                if predicted_states is not None and state_values is not None:
                    predicted_states[record[0]] = state_values[batch_index]

    baseline_local = destinations - frameskip
    tasks = np.asarray(choice.cache.data.tasks[rows[destinations]], dtype=np.int64)
    return PredictionResult(
        mode="teacher-forced",
        destination_local=destinations,
        display_input_local=baseline_local.copy(),
        baseline_local=baseline_local,
        tasks=tasks,
        context_sizes=context_sizes,
        predicted_latents=predicted,
        predicted_states=predicted_states,
    )


def open_loop_predictions(
    model: torch.nn.Module,
    choice: EpisodeChoice,
    frameskip: int,
    history: int,
    action_center: np.ndarray,
    action_scale: np.ndarray,
    batch_size: int,
    state_center: np.ndarray | None = None,
    state_scale: np.ndarray | None = None,
) -> PredictionResult:
    del batch_size  # Open-loop has one recursive trajectory and cannot be frame-batched.
    rows = choice.rows
    destinations = np.arange(frameskip * history, len(rows), frameskip, dtype=np.int64)
    seed_local = np.arange(0, frameskip * history, frameskip, dtype=np.int64)
    trajectory = list(encode_rows(model, choice.cache, rows[seed_local], history))
    state_trajectory = None
    if isinstance(model, StateResidualLeWM):
        if state_center is None or state_scale is None:
            raise ValueError("state-residual visualization requires state normalization")
        state_trajectory = list(
            normalized_state_rows(
                choice.cache, rows[seed_local], state_center, state_scale
            )
        )
    action_history = [
        normalized_action_chunk(
            choice.cache,
            int(rows[local]),
            frameskip,
            action_center,
            action_scale,
        )
        for local in seed_local[:-1]
    ]
    predicted: list[np.ndarray] = []
    predicted_states: list[np.ndarray] = []
    context_sizes: list[int] = []

    for destination in destinations:
        action_start = int(destination - frameskip)
        action_history.append(
            normalized_action_chunk(
                choice.cache,
                int(rows[action_start]),
                frameskip,
                action_center,
                action_scale,
            )
        )
        context_size = min(history, len(trajectory))
        latent_batch = np.stack(trajectory[-context_size:])[None, ...]
        action_batch = np.stack(action_history[-context_size:])[None, ...]
        if isinstance(model, StateResidualLeWM):
            if state_trajectory is None:
                raise AssertionError("missing state trajectory")
            state_batch = np.stack(state_trajectory[-context_size:])[None, ...]
            latent_values, state_values = predict_state_residual_batch(
                model, latent_batch, action_batch, state_batch
            )
            next_latent = latent_values[0]
            next_state = state_values[0]
            state_trajectory.append(next_state)
            predicted_states.append(next_state * state_scale + state_center)
        else:
            next_latent = predict_batch(model, latent_batch, action_batch)[0]
        trajectory.append(next_latent)
        predicted.append(next_latent)
        context_sizes.append(context_size)

    baseline_local = destinations - frameskip
    tasks = np.asarray(choice.cache.data.tasks[rows[destinations]], dtype=np.int64)
    return PredictionResult(
        mode="open-loop",
        destination_local=destinations,
        display_input_local=np.full(
            len(destinations), frameskip * (history - 1), dtype=np.int64
        ),
        baseline_local=baseline_local,
        tasks=tasks,
        context_sizes=np.asarray(context_sizes, dtype=np.int16),
        predicted_latents=np.stack(predicted).astype(np.float32, copy=False),
        predicted_states=(
            np.stack(predicted_states).astype(np.float32, copy=False)
            if predicted_states
            else None
        ),
    )


def prediction_cycle_origins(
    frame_count: int,
    frameskip: int,
    history: int,
    horizon: int,
) -> np.ndarray:
    if min(frame_count, frameskip, history, horizon) <= 0:
        raise ValueError("prediction-cycle dimensions must be positive")
    first = frameskip * (history - 1)
    stop = frame_count - frameskip * horizon
    if stop <= first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, stop, frameskip, dtype=np.int64)


def prediction_cycle_predictions(
    model: torch.nn.Module,
    choice: EpisodeChoice,
    frameskip: int,
    history: int,
    horizon: int,
    action_center: np.ndarray,
    action_scale: np.ndarray,
    batch_size: int,
    state_center: np.ndarray | None = None,
    state_scale: np.ndarray | None = None,
    memory_offsets: tuple[int, ...] | None = None,
) -> PredictionResult:
    rows = choice.rows
    origins = prediction_cycle_origins(len(rows), frameskip, history, horizon)
    is_multiscale = isinstance(model, MultiScaleHistoryLeWM)
    if is_multiscale:
        offsets = tuple(int(value) for value in (memory_offsets or ()))
        if not offsets or any(value <= 0 for value in offsets):
            raise ValueError("V9 visualization requires positive memory offsets")
        if tuple(sorted(offsets)) != offsets or len(set(offsets)) != len(offsets):
            raise ValueError("V9 memory offsets must be sorted and unique")
        if any(value % frameskip for value in offsets):
            raise ValueError("V9 memory offsets must align to frameskip")
        oldest_recent = frameskip * (history - 1)
        origins = origins[origins >= oldest_recent + max(offsets)]
    else:
        offsets = ()
    if len(origins) == 0:
        raise ValueError(
            f"episode {choice.cache.source}:{choice.episode} is too short for "
            f"history={history}, frameskip={frameskip}, horizon={horizon}"
        )

    # Encode only rows at or before the latest forecast origin. Every rollout
    # indexes the checkpoint history ending at its own origin.
    observed_latents = encode_rows(
        model,
        choice.cache,
        rows[: int(origins[-1]) + 1],
        batch_size,
    )
    trajectories: list[list[np.ndarray]] = []
    stateful = isinstance(model, StateResidualLeWM) or is_multiscale
    state_trajectories: list[list[np.ndarray]] | None = [] if stateful else None
    if stateful and (
        state_center is None or state_scale is None
    ):
        raise ValueError("state-residual visualization requires state normalization")

    memory_latents: np.ndarray | None = None
    memory_actions: np.ndarray | None = None
    memory_states: np.ndarray | None = None
    memory_ages: np.ndarray | None = None
    if is_multiscale:
        oldest_recent = frameskip * (history - 1)
        memory_local_rows = [
            [origin - oldest_recent - offset for offset in offsets]
            for origin in origins
        ]
        if any(any(local < 0 for local in values) for values in memory_local_rows):
            raise ValueError("V9 memory history reaches before the episode start")
        memory_latents = np.stack(
            [observed_latents[values] for values in memory_local_rows]
        )
        memory_states = np.stack(
            [
                normalized_state_rows(
                    choice.cache,
                    rows[np.asarray(values, dtype=np.int64)],
                    state_center,
                    state_scale,
                )
                for values in memory_local_rows
            ]
        )
        memory_actions = np.stack(
            [
                np.stack(
                    [
                        normalized_action_chunk(
                            choice.cache,
                            int(rows[local]),
                            frameskip,
                            action_center,
                            action_scale,
                        )
                        for local in values
                    ]
                )
                for values in memory_local_rows
            ]
        )
        memory_ages = np.broadcast_to(
            np.asarray(offsets, dtype=np.float32) / float(frameskip),
            (len(origins), len(offsets)),
        ).copy()
    action_histories: list[list[np.ndarray]] = []
    for raw_origin in origins:
        origin = int(raw_origin)
        context_locals = list(
            range(origin - (history - 1) * frameskip, origin + 1, frameskip)
        )
        if len(context_locals) != history or context_locals[-1] != origin:
            raise AssertionError("invalid prediction-cycle history")
        trajectories.append([observed_latents[local] for local in context_locals])
        if state_trajectories is not None:
            state_trajectories.append(
                list(
                    normalized_state_rows(
                        choice.cache,
                        rows[np.asarray(context_locals, dtype=np.int64)],
                        state_center,
                        state_scale,
                    )
                )
            )
        action_histories.append(
            [
                normalized_action_chunk(
                    choice.cache,
                    int(rows[local]),
                    frameskip,
                    action_center,
                    action_scale,
                )
                for local in context_locals[:-1]
            ]
        )

    predicted_by_origin: list[list[np.ndarray]] = [[] for _ in origins]
    predicted_states_by_origin: list[list[np.ndarray]] | None = (
        [[] for _ in origins] if state_trajectories is not None else None
    )
    for step in range(1, horizon + 1):
        for origin_index, raw_origin in enumerate(origins):
            origin = int(raw_origin)
            action_local = origin + (step - 1) * frameskip
            destination = origin + step * frameskip
            if destination >= len(rows):
                raise AssertionError("prediction-cycle target exceeds episode")
            action_histories[origin_index].append(
                normalized_action_chunk(
                    choice.cache,
                    int(rows[action_local]),
                    frameskip,
                    action_center,
                    action_scale,
                )
            )
        context_size = history
        latent_batch = np.stack(
            [trajectory[-context_size:] for trajectory in trajectories]
        )
        action_batch = np.stack(
            [actions[-context_size:] for actions in action_histories]
        )
        if isinstance(model, StateResidualLeWM):
            if state_trajectories is None or predicted_states_by_origin is None:
                raise AssertionError("missing state-residual rollout buffers")
            state_batch = np.stack(
                [trajectory[-context_size:] for trajectory in state_trajectories]
            )
            next_latents, next_states = predict_state_residual_batch(
                model, latent_batch, action_batch, state_batch
            )
        elif is_multiscale:
            if (
                state_trajectories is None
                or predicted_states_by_origin is None
                or memory_latents is None
                or memory_actions is None
                or memory_states is None
                or memory_ages is None
            ):
                raise AssertionError("missing V9 memory rollout buffers")
            state_batch = np.stack(
                [trajectory[-context_size:] for trajectory in state_trajectories]
            )
            next_latent_parts: list[np.ndarray] = []
            next_state_parts: list[np.ndarray] = []
            for start in range(0, len(origins), batch_size):
                stop = min(start + batch_size, len(origins))
                next_latent, next_state = predict_multiscale_batch(
                    model,
                    latent_batch[start:stop],
                    action_batch[start:stop],
                    state_batch[start:stop],
                    memory_latents[start:stop],
                    memory_actions[start:stop],
                    memory_states[start:stop],
                    memory_ages[start:stop] + float(step - 1),
                )
                next_latent_parts.append(next_latent)
                next_state_parts.append(next_state)
            next_latents = np.concatenate(next_latent_parts, axis=0)
            next_states = np.concatenate(next_state_parts, axis=0)
        else:
            next_latents = predict_batch(model, latent_batch, action_batch)
            next_states = None
        for origin_index, next_latent in enumerate(next_latents):
            trajectories[origin_index].append(next_latent)
            predicted_by_origin[origin_index].append(next_latent)
            if next_states is not None:
                state_trajectories[origin_index].append(next_states[origin_index])
                predicted_states_by_origin[origin_index].append(
                    next_states[origin_index] * state_scale + state_center
                )

    destinations: list[int] = []
    display_inputs: list[int] = []
    baselines: list[int] = []
    tasks: list[int] = []
    context_sizes: list[int] = []
    predicted: list[np.ndarray] = []
    predicted_states: list[np.ndarray] = []
    for origin_index, raw_origin in enumerate(origins):
        origin = int(raw_origin)
        for step, next_latent in enumerate(predicted_by_origin[origin_index], start=1):
            destination = origin + step * frameskip
            predicted.append(next_latent)
            if predicted_states_by_origin is not None:
                predicted_states.append(
                    predicted_states_by_origin[origin_index][step - 1]
                )
            destinations.append(destination)
            display_inputs.append(origin)
            baselines.append(origin)
            tasks.append(int(choice.cache.data.tasks[rows[destination]]))
            context_sizes.append(history)

    return PredictionResult(
        mode="prediction-cycle",
        destination_local=np.asarray(destinations, dtype=np.int64),
        display_input_local=np.asarray(display_inputs, dtype=np.int64),
        baseline_local=np.asarray(baselines, dtype=np.int64),
        tasks=np.asarray(tasks, dtype=np.int64),
        context_sizes=np.asarray(context_sizes, dtype=np.int16),
        predicted_latents=np.stack(predicted).astype(np.float32, copy=False),
        predicted_states=(
            np.stack(predicted_states).astype(np.float32, copy=False)
            if predicted_states
            else None
        ),
    )


def retrieve_neighbors(
    predictions: PredictionResult,
    cache: CacheView,
    groups: dict[tuple[int, int], BankGroup],
    assignments: SplitAssignments,
    expected_joint_states: np.ndarray,
    max_command_joint_rmse: float,
    max_arm_joint_error: float,
    max_gripper_error: float,
    query_chunk_size: int = 64,
    bank_chunk_size: int = 8192,
) -> RetrievalResult:
    expected = np.asarray(expected_joint_states, dtype=np.float32)
    if expected.shape != (len(predictions.predicted_latents), 6):
        raise ValueError(
            "expected_joint_states must have shape "
            f"{(len(predictions.predicted_latents), 6)}, got {expected.shape}"
        )
    if not np.all(np.isfinite(expected)):
        raise ValueError("expected_joint_states contains non-finite values")
    if not np.isfinite(max_command_joint_rmse) or max_command_joint_rmse <= 0:
        raise ValueError("max_command_joint_rmse must be positive")
    if not np.isfinite(max_arm_joint_error) or max_arm_joint_error <= 0:
        raise ValueError("max_arm_joint_error must be positive")
    if not np.isfinite(max_gripper_error) or max_gripper_error <= 0:
        raise ValueError("max_gripper_error must be positive")

    result_rows = np.full(len(predictions.predicted_latents), -1, dtype=np.int64)
    result_distance = np.full(len(predictions.predicted_latents), np.nan, dtype=np.float64)
    result_joint_rmse = np.full(
        len(predictions.predicted_latents), np.inf, dtype=np.float64
    )
    result_arm_max_error = np.full(
        len(predictions.predicted_latents), np.inf, dtype=np.float64
    )
    result_gripper_error = np.full(
        len(predictions.predicted_latents), np.inf, dtype=np.float64
    )
    latent_dim = predictions.predicted_latents.shape[1]
    maximum_squared_joint_error = float(max_command_joint_rmse) ** 2

    for raw_task in np.unique(predictions.tasks):
        task = int(raw_task)
        key = (cache.cache_id, task)
        if key not in groups:
            raise ValueError(
                f"no training-episode retrieval frames for source={cache.source!r}, task={task}"
            )
        group = groups[key]
        if group.source != cache.source or group.task != task:
            raise AssertionError("retrieval group source/task contract mismatch")
        query_indices = np.flatnonzero(predictions.tasks == task)
        for query_start in range(0, len(query_indices), query_chunk_size):
            indices = query_indices[query_start : query_start + query_chunk_size]
            query = predictions.predicted_latents[indices]
            query_norm = np.sum(query * query, axis=1, keepdims=True)
            best_squared = np.full(len(query), np.inf, dtype=np.float32)
            best_group_index = np.full(len(query), -1, dtype=np.int64)
            nearest_joint_squared = np.full(len(query), np.inf, dtype=np.float32)
            for bank_start in range(0, len(group.latents), bank_chunk_size):
                bank = group.latents[bank_start : bank_start + bank_chunk_size]
                bank_rows = group.rows[bank_start : bank_start + bank_chunk_size]
                bank_states = np.asarray(cache.data.states[bank_rows], dtype=np.float32)
                joint_squared = np.mean(
                    (
                        bank_states[None, :, :]
                        - expected[indices, None, :]
                    )
                    ** 2,
                    axis=2,
                )
                absolute_joint_error = np.abs(
                    bank_states[None, :, :] - expected[indices, None, :]
                )
                arm_max_error = np.max(absolute_joint_error[:, :, :5], axis=2)
                gripper_error = absolute_joint_error[:, :, 5]
                nearest_joint_squared = np.minimum(
                    nearest_joint_squared, np.min(joint_squared, axis=1)
                )
                bank_norm = np.sum(bank * bank, axis=1)[None, :]
                squared = query_norm + bank_norm - 2.0 * (query @ bank.T)
                unsupported = (
                    (joint_squared > maximum_squared_joint_error)
                    | (arm_max_error > max_arm_joint_error)
                    | (gripper_error > max_gripper_error)
                )
                squared[unsupported] = np.inf
                local_index = np.argmin(squared, axis=1)
                local_value = squared[np.arange(len(query)), local_index]
                improved = local_value < best_squared
                best_squared[improved] = local_value[improved]
                best_group_index[improved] = bank_start + local_index[improved]
            matched = best_group_index >= 0
            matched_indices = indices[matched]
            if len(matched_indices):
                selected_group_indices = best_group_index[matched]
                selected_rows = group.rows[selected_group_indices]
                selected_states = np.asarray(
                    cache.data.states[selected_rows], dtype=np.float32
                )
                result_rows[matched_indices] = selected_rows
                result_distance[matched_indices] = (
                    np.maximum(best_squared[matched], 0.0) / latent_dim
                )
                result_joint_rmse[matched_indices] = np.sqrt(
                    np.mean(
                        (selected_states - expected[matched_indices]) ** 2,
                        axis=1,
                    )
                )
                selected_absolute_error = np.abs(
                    selected_states - expected[matched_indices]
                )
                result_arm_max_error[matched_indices] = np.max(
                    selected_absolute_error[:, :5], axis=1
                )
                result_gripper_error[matched_indices] = selected_absolute_error[:, 5]
            unmatched_indices = indices[~matched]
            result_joint_rmse[unmatched_indices] = np.sqrt(
                nearest_joint_squared[~matched]
            )

    valid = result_rows >= 0
    if not np.any(valid):
        raise ValueError(
            "the command-pose gate rejected every retrieval candidate; "
            "the training bank has no geometric support for this episode"
        )
    if not np.all(np.isfinite(result_distance[valid])):
        raise AssertionError("non-finite matched nearest-neighbor distance")
    if not np.all(np.isfinite(result_joint_rmse)):
        raise AssertionError("non-finite command-joint retrieval distance")
    retrieved_episodes = np.asarray(
        cache.data.episodes[result_rows[valid]], dtype=np.int64
    )
    if any(
        assignments.get((cache.dataset_id, int(episode))) != "train"
        for episode in retrieved_episodes
    ):
        raise AssertionError("retrieval selected a held-out episode")
    retrieved_tasks = np.asarray(cache.data.tasks[result_rows[valid]], dtype=np.int64)
    if not np.array_equal(retrieved_tasks, predictions.tasks[valid]):
        raise AssertionError("retrieval selected a different task")
    if np.any(result_joint_rmse[valid] > max_command_joint_rmse + 1e-6):
        raise AssertionError("retrieval violated the command-joint hard gate")
    if np.any(result_arm_max_error[valid] > max_arm_joint_error + 1e-6):
        raise AssertionError("retrieval violated the arm-joint hard gate")
    if np.any(result_gripper_error[valid] > max_gripper_error + 1e-6):
        raise AssertionError("retrieval violated the gripper hard gate")
    return RetrievalResult(
        rows=result_rows,
        latent_mse=result_distance,
        command_joint_rmse=result_joint_rmse,
        command_arm_max_error=result_arm_max_error,
        command_gripper_error=result_gripper_error,
        valid=valid,
    )


def retrieval_expected_joint_states(
    choice: EpisodeChoice,
    predictions: PredictionResult,
) -> tuple[np.ndarray, str]:
    if predictions.predicted_states is not None:
        expected = np.asarray(predictions.predicted_states, dtype=np.float32)
        source = "direct six-state residual-head prediction"
    else:
        command_rows = choice.rows[predictions.destination_local - 1]
        expected = np.asarray(choice.cache.data.actions[command_rows], dtype=np.float32)
        source = "final commanded joint target"
    if expected.shape != (len(predictions.predicted_latents), 6):
        raise ValueError(f"retrieval pose source has invalid shape {expected.shape}")
    if not np.all(np.isfinite(expected)):
        raise ValueError("retrieval pose source contains non-finite values")
    return expected, source


def finite_r2(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    if denominator <= 0:
        return None
    return 1.0 - float(np.sum((actual - predicted) ** 2)) / denominator


def build_metrics(
    choice: EpisodeChoice,
    predictions: PredictionResult,
    retrieval: RetrievalResult,
    target_latents: np.ndarray,
    baseline_latents: np.ndarray,
    decoder: RidgeDecoder,
    retrieval_pose_source: str,
    kinematics: SO101MuJoCoKinematics | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    destination_rows = choice.rows[predictions.destination_local]
    actual_states = np.asarray(choice.cache.data.states[destination_rows], dtype=np.float64)
    direct_state_prediction = predictions.predicted_states is not None
    predicted_states = (
        np.asarray(predictions.predicted_states, dtype=np.float64)
        if direct_state_prediction
        else decoder.predict(predictions.predicted_latents)
    )
    latent_error = np.mean(
        (predictions.predicted_latents - target_latents) ** 2, axis=1
    ).astype(np.float64)
    copy_error = np.mean((baseline_latents - target_latents) ** 2, axis=1).astype(np.float64)
    state_error = predicted_states - actual_states
    matched_retrieval = retrieval.valid
    per_joint: dict[str, Any] = {}
    for joint_index, joint_name in enumerate(choice.cache.joint_names):
        actual = actual_states[:, joint_index]
        predicted = predicted_states[:, joint_index]
        per_joint[joint_name] = {
            "mae": float(np.mean(np.abs(predicted - actual))),
            "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "r2": finite_r2(actual, predicted),
        }
    frame_indices = np.asarray(
        choice.cache.data.frame_indices[destination_rows], dtype=np.int64
    )
    source_fps = float(choice.cache.manifest.get("fps") or 30.0)
    metrics = {
        "format": REPORT_FORMAT,
        "source": choice.cache.source,
        "episode": choice.episode,
        "mode": predictions.mode,
        "prediction_count": len(predictions.predicted_latents),
        "latent": {
            "distance": "mean squared error in LeWM prediction-latent coordinates",
            "prediction_mse": float(np.mean(latent_error)),
            "copy_baseline_mse": float(np.mean(copy_error)),
            "improvement_over_copy": float(np.mean(copy_error) - np.mean(latent_error)),
            "per_prediction_mse": latent_error.tolist(),
        },
        "retrieval": {
            "distance": "mean squared error in LeWM prediction-latent coordinates",
            "command_pose_gate": (
                "six-joint RMS plus per-arm-joint and gripper hard limits to the "
                f"{retrieval_pose_source}"
            ),
            "pose_source": retrieval_pose_source,
            "matched_fraction": float(np.mean(matched_retrieval)),
            "matched_predictions": int(np.sum(matched_retrieval)),
            "unmatched_predictions": int(np.sum(~matched_retrieval)),
            "predicted_to_neighbor_mse": float(
                np.mean(retrieval.latent_mse[matched_retrieval])
            ),
            "selected_command_joint_rmse": float(
                np.mean(retrieval.command_joint_rmse[matched_retrieval])
            ),
            "selected_command_arm_max_error": float(
                np.mean(retrieval.command_arm_max_error[matched_retrieval])
            ),
            "selected_command_gripper_error": float(
                np.mean(retrieval.command_gripper_error[matched_retrieval])
            ),
            "per_prediction_mse": [
                float(value) if np.isfinite(value) else None
                for value in retrieval.latent_mse
            ],
            "per_prediction_command_joint_rmse": (
                retrieval.command_joint_rmse.tolist()
            ),
            "per_prediction_command_arm_max_error": [
                float(value) if np.isfinite(value) else None
                for value in retrieval.command_arm_max_error
            ],
            "per_prediction_command_gripper_error": [
                float(value) if np.isfinite(value) else None
                for value in retrieval.command_gripper_error
            ],
        },
        "joint_decoder": {
            "kind": (
                "checkpoint direct six-state residual head"
                if direct_state_prediction
                else "ridge with intercept and train-only latent standardization"
            ),
            "training_scope": (
                "joint-state supervision used by the checkpoint training run"
                if direct_state_prediction
                else "all frames from training-split episodes in checkpoint caches"
            ),
            "training_rows": None if direct_state_prediction else decoder.training_rows,
            "alpha": None if direct_state_prediction else decoder.alpha,
            "objective_scaling": (
                "shared checkpoint z-score coordinates, reported here in raw joint units"
                if direct_state_prediction
                else "mean squared error plus alpha times squared weights"
            ),
            "overall_mae": float(np.mean(np.abs(state_error))),
            "overall_rmse": float(np.sqrt(np.mean(state_error**2))),
            "per_joint": per_joint,
            "traces": {
                "joint_names": list(choice.cache.joint_names),
                "frame_index": frame_indices.tolist(),
                "seconds": (frame_indices / source_fps).tolist(),
                "actual": actual_states.tolist(),
                "predicted": predicted_states.tolist(),
            },
        },
    }
    if kinematics is not None:
        actual_xyz = kinematics.forward(actual_states)
        predicted_xyz = kinematics.forward(predicted_states)
        xyz_error = predicted_xyz - actual_xyz
        euclidean_error = np.linalg.norm(xyz_error, axis=1)
        metrics["end_effector"] = {
            "kind": "official SO-101 MuJoCo gripperframe forward kinematics",
            "coordinate_frame": "SO-101 base frame",
            "units": "meters",
            "model_path": str(kinematics.model_path),
            "joint_convention": (
                "LeRobot post-PR-777 normalized arm positions [-100,100] mapped "
                "linearly to the official new-calibration joint ranges"
            ),
            "gripper_note": (
                "gripper command is excluded from gripperframe XYZ because the "
                "official MuJoCo model does not yet implement LeRobot's linear "
                "[0,100] gripper convention"
            ),
            "rmse_m": float(np.sqrt(np.mean(euclidean_error**2))),
            "mae_m": float(np.mean(euclidean_error)),
            "p95_m": float(np.quantile(euclidean_error, 0.95)),
            "max_m": float(np.max(euclidean_error)),
            "per_axis_rmse_m": {
                axis: float(np.sqrt(np.mean(xyz_error[:, index] ** 2)))
                for index, axis in enumerate(("x", "y", "z"))
            },
            "actual_state_clipping_fraction": kinematics.clipping_fraction(actual_states),
            "predicted_state_clipping_fraction": kinematics.clipping_fraction(
                predicted_states
            ),
            "traces": {
                "actual_xyz_m": actual_xyz.tolist(),
                "predicted_xyz_m": predicted_xyz.tolist(),
                "euclidean_error_m": euclidean_error.tolist(),
            },
        }
    return metrics, actual_states, predicted_states


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    preferred_size: int,
    bold: bool = False,
    minimum_size: int = 10,
) -> ImageFont.ImageFont:
    for size in range(preferred_size, minimum_size - 1, -1):
        font = load_font(size, bold)
        if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
            return font
    return load_font(minimum_size, bold)


def draw_scaled_rgb(canvas: Image.Image, frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    image = image.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    canvas.paste(image, (left, top))


def draw_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    title_lines: list[str],
    frame: np.ndarray,
    details: list[str],
    accent: str,
) -> None:
    panel_width = 390
    image_size = 350
    draw.rectangle((x, 82, x + panel_width, 532), fill="#ffffff", outline="#d7d9dc", width=2)
    draw.rectangle((x, 82, x + 6, 532), fill=accent)
    title_y = 91
    for line in title_lines:
        font = fit_font(draw, line, panel_width - 24, 15, bold=True)
        draw.text((x + 14, title_y), line, fill="#202124", font=font)
        title_y += 19
    image_top = 130
    image_left = x + (panel_width - image_size) // 2
    draw_scaled_rgb(
        canvas,
        frame,
        (image_left, image_top, image_left + image_size, image_top + image_size),
    )
    draw.rectangle(
        (image_left, image_top, image_left + image_size, image_top + image_size),
        outline="#202124",
        width=1,
    )
    detail_font = load_font(13)
    for line_number, line in enumerate(details[:2]):
        font = fit_font(draw, line, panel_width - 28, 13)
        draw.text(
            (x + 14, 488 + line_number * 18),
            line,
            fill="#4b4f52",
            font=font if font else detail_font,
        )


def trace_limits(actual: np.ndarray, predicted: np.ndarray) -> list[tuple[float, float]]:
    limits: list[tuple[float, float]] = []
    for joint in range(actual.shape[1]):
        low = float(min(actual[:, joint].min(), predicted[:, joint].min()))
        high = float(max(actual[:, joint].max(), predicted[:, joint].max()))
        span = high - low
        padding = max(span * 0.08, 1e-3)
        limits.append((low - padding, high + padding))
    return limits


def draw_trace_panels(
    draw: ImageDraw.ImageDraw,
    actual: np.ndarray,
    predicted: np.ndarray,
    joint_names: tuple[str, ...],
    limits: list[tuple[float, float]],
    current: int,
) -> None:
    chart_top = 570
    chart_width = 390
    chart_height = 105
    gap_x = 20
    gap_y = 18
    left_margin = 30
    small_font = load_font(12)
    label_font = load_font(12, bold=True)
    total = len(actual)
    for joint in range(6):
        column = joint % 3
        row = joint // 3
        left = left_margin + column * (chart_width + gap_x)
        top = chart_top + row * (chart_height + gap_y)
        right = left + chart_width
        bottom = top + chart_height
        draw.rectangle((left, top, right, bottom), fill="#ffffff", outline="#d7d9dc")
        name = joint_names[joint].removesuffix(".pos")
        draw.text((left + 7, top + 5), name, fill="#202124", font=label_font)
        low, high = limits[joint]
        draw.text((right - 68, top + 5), f"{high:.2f}", fill="#6b7075", font=small_font)
        draw.text((right - 68, bottom - 18), f"{low:.2f}", fill="#6b7075", font=small_font)
        plot_left, plot_right = left + 7, right - 76
        plot_top, plot_bottom = top + 25, bottom - 10
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#d7d9dc")

        def points(values: np.ndarray) -> list[tuple[float, float]]:
            output = []
            for index, value in enumerate(values[: current + 1]):
                x = plot_left + (plot_right - plot_left) * index / max(1, total - 1)
                y = plot_bottom - (plot_bottom - plot_top) * (float(value) - low) / (high - low)
                output.append((x, y))
            return output

        actual_points = points(actual[:, joint])
        predicted_points = points(predicted[:, joint])
        if len(actual_points) > 1:
            draw.line(actual_points, fill=ACTUAL_COLOR, width=2)
            draw.line(predicted_points, fill=PREDICTED_COLOR, width=2)
        else:
            x_a, y_a = actual_points[0]
            x_p, y_p = predicted_points[0]
            draw.ellipse((x_a - 2, y_a - 2, x_a + 2, y_a + 2), fill=ACTUAL_COLOR)
            draw.ellipse((x_p - 2, y_p - 2, x_p + 2, y_p + 2), fill=PREDICTED_COLOR)


def render_frame(
    choice: EpisodeChoice,
    predictions: PredictionResult,
    retrieval: RetrievalResult,
    actual_states: np.ndarray,
    predicted_states: np.ndarray,
    limits: list[tuple[float, float]],
    index: int,
    frameskip: int,
) -> np.ndarray:
    canvas = Image.new("RGB", CANVAS_SIZE, "#f4f5f6")
    draw = ImageDraw.Draw(canvas)
    destination_local = int(predictions.destination_local[index])
    input_local = int(predictions.display_input_local[index])
    destination_row = int(choice.rows[destination_local])
    input_row = int(choice.rows[input_local])
    neighbor_row = int(retrieval.rows[index])
    task = int(predictions.tasks[index])
    source_fps = float(choice.cache.manifest.get("fps") or 30.0)
    destination_frame = int(choice.cache.data.frame_indices[destination_row])
    retrieval_valid = bool(retrieval.valid[index])
    direct_state_head = predictions.predicted_states is not None
    pose_gate_title = (
        "STATE-PREDICTION-GATED TRAIN NEIGHBOR"
        if direct_state_head
        else "COMMAND-POSE-GATED TRAIN NEIGHBOR"
    )
    pose_gate_name = "predicted-state" if direct_state_head else "command-pose"
    if retrieval_valid:
        neighbor_episode = int(choice.cache.data.episodes[neighbor_row])
        neighbor_frame = int(choice.cache.data.frame_indices[neighbor_row])
        neighbor_rgb = np.asarray(choice.cache.data.frames[neighbor_row])
        neighbor_details = [
            f"train episode={neighbor_episode} frame={neighbor_frame}",
            (
                f"latent mse={retrieval.latent_mse[index]:.5f} "
                f"joint rms/max/grip={retrieval.command_joint_rmse[index]:.1f}/"
                f"{retrieval.command_arm_max_error[index]:.1f}/"
                f"{retrieval.command_gripper_error[index]:.1f}"
            ),
        ]
    else:
        neighbor_rgb = np.full_like(
            np.asarray(choice.cache.data.frames[destination_row]), 232
        )
        neighbor_details = [
            f"no training frame passed the {pose_gate_name} hard gate",
            f"nearest pose joint rmse={retrieval.command_joint_rmse[index]:.2f}",
        ]

    header_font = fit_font(
        draw,
        f"SO-101 LeWM | {predictions.mode.upper()} | {choice.cache.source}:{choice.episode}",
        1220,
        25,
        bold=True,
    )
    draw.text(
        (30, 20),
        f"SO-101 LeWM | {predictions.mode.upper()} | {choice.cache.source}:{choice.episode}",
        fill="#202124",
        font=header_font,
    )
    sub_font = load_font(14)
    horizon_steps = max(1, (destination_local - input_local) // frameskip)
    draw.text(
        (30, 54),
        f"target frame {destination_frame} ({destination_frame / source_fps:.2f}s) | "
        f"forecast H{horizon_steps} = {horizon_steps * frameskip} raw frames | "
        f"context {int(predictions.context_sizes[index])} | "
        "actual joints: teal | predicted joints: red",
        fill="#5f6368",
        font=sub_font,
    )

    if predictions.mode == "teacher-forced":
        left_title = ["OBSERVED RGB CONTEXT", "MODEL INPUT"]
    elif predictions.mode == "open-loop":
        left_title = ["INITIAL HELD-OUT RGB SEED", "ONLY RGB INPUT TO OPEN LOOP"]
    else:
        left_title = ["CURRENT HELD-OUT RGB", "PAUSED WHILE H1-H8 FORECAST PLAYS"]
    draw_panel(
        canvas,
        draw,
        30,
        left_title,
        np.asarray(choice.cache.data.frames[input_row]),
        [
            f"source={choice.cache.source} episode={choice.episode}",
            f"frame={int(choice.cache.data.frame_indices[input_row])} task={task}",
        ],
        "#f4b942",
    )
    draw_panel(
        canvas,
        draw,
        445,
        ["HELD-OUT FUTURE RGB", "DISPLAY / EVALUATION ONLY"],
        np.asarray(choice.cache.data.frames[destination_row]),
        [
            f"source={choice.cache.source} episode={choice.episode}",
            f"frame={destination_frame} never used for prediction/retrieval",
        ],
        ACTUAL_COLOR,
    )
    draw_panel(
        canvas,
        draw,
        860,
        [
            pose_gate_title,
            "PREDICTED-LATENT RGB - NOT GENERATED",
        ],
        neighbor_rgb,
        neighbor_details,
        PREDICTED_COLOR,
    )
    if predictions.mode == "prediction-cycle":
        cycle_indices = np.flatnonzero(predictions.display_input_local == input_local)
        cycle_actual = actual_states[cycle_indices]
        cycle_predicted = predicted_states[cycle_indices]
        cycle_index = int(np.flatnonzero(cycle_indices == index)[0])
        draw_trace_panels(
            draw,
            cycle_actual,
            cycle_predicted,
            choice.cache.joint_names,
            trace_limits(cycle_actual, cycle_predicted),
            cycle_index,
        )
    else:
        draw_trace_panels(
            draw,
            actual_states,
            predicted_states,
            choice.cache.joint_names,
            limits,
            index,
        )
    return np.asarray(canvas, dtype=np.uint8)


class RawFFmpegWriter:
    def __init__(
        self,
        output: Path,
        fps: float,
        media_type: str,
    ) -> None:
        self.output = output
        self.temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
        self.closed = False
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        width, height = CANVAS_SIZE
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "pipe:0",
            "-an",
        ]
        if media_type == "mp4":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-f",
                    "mp4",
                ]
            )
        elif media_type == "gif":
            command.extend(
                [
                    "-vf",
                    "scale=960:-2:flags=lanczos",
                    "-loop",
                    "0",
                    "-f",
                    "gif",
                ]
            )
        else:
            raise ValueError(media_type)
        command.append(str(self.temporary))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (CANVAS_SIZE[1], CANVAS_SIZE[0], 3) or frame.dtype != np.uint8:
            raise ValueError(f"invalid rendered frame shape/dtype: {frame.shape}, {frame.dtype}")
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.closed:
            return
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        stderr = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr is not None
            else ""
        )
        if return_code != 0:
            self.temporary.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed for {self.output}: {stderr[-3000:]}")
        self.temporary.replace(self.output)
        self.closed = True

    def abort(self) -> None:
        if self.closed:
            return
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
        self.temporary.unlink(missing_ok=True)
        self.closed = True


def render_media(
    output: Path,
    media: str,
    fps: float,
    render_stride: int,
    choice: EpisodeChoice,
    predictions: PredictionResult,
    retrieval: RetrievalResult,
    actual_states: np.ndarray,
    predicted_states: np.ndarray,
    frameskip: int,
) -> tuple[list[str], int]:
    stems = {
        "teacher-forced": "teacher_forced_h1",
        "open-loop": "open_loop",
        "prediction-cycle": "prediction_cycle",
    }
    stem = stems[predictions.mode]
    media_types = ["mp4", "gif"] if media == "both" else [media]
    writers: list[RawFFmpegWriter] = []
    try:
        for kind in media_types:
            writers.append(RawFFmpegWriter(output / f"{stem}.{kind}", fps, kind))
    except Exception:
        for writer in writers:
            writer.abort()
        raise
    indices = list(range(0, len(predictions.predicted_latents), render_stride))
    if indices[-1] != len(predictions.predicted_latents) - 1:
        indices.append(len(predictions.predicted_latents) - 1)
    limits = trace_limits(actual_states, predicted_states)
    try:
        for index in indices:
            frame = render_frame(
                choice,
                predictions,
                retrieval,
                actual_states,
                predicted_states,
                limits,
                index,
                frameskip,
            )
            for writer in writers:
                writer.write(frame)
        for writer in writers:
            writer.close()
    except Exception:
        for writer in writers:
            writer.abort()
        raise
    return [writer.output.name for writer in writers], len(indices)


def svg_polyline(
    values: np.ndarray,
    low: float,
    high: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    if len(values) == 1:
        x_values = [left]
    else:
        x_values = [left + width * index / (len(values) - 1) for index in range(len(values))]
    points = [
        f"{x:.2f},{top + height - height * (float(value) - low) / (high - low):.2f}"
        for x, value in zip(x_values, values, strict=True)
    ]
    return " ".join(points)


def trace_svg(
    actual: np.ndarray,
    predicted: np.ndarray,
    joint_names: tuple[str, ...],
) -> str:
    width, height = 1200, 560
    plot_width, plot_height = 350, 205
    gap_x, gap_y = 50, 45
    origin_x, origin_y = 20, 38
    limits = trace_limits(actual, predicted)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Actual and ridge-decoded predicted joint traces">',
        '<rect width="1200" height="560" fill="#ffffff"/>',
    ]
    for joint in range(6):
        column, row = joint % 3, joint // 3
        x = origin_x + column * (plot_width + gap_x)
        y = origin_y + row * (plot_height + gap_y)
        low, high = limits[joint]
        actual_points = svg_polyline(actual[:, joint], low, high, x, y, plot_width, plot_height)
        predicted_points = svg_polyline(
            predicted[:, joint], low, high, x, y, plot_width, plot_height
        )
        name = html.escape(joint_names[joint].removesuffix(".pos"))
        parts.extend(
            [
                f'<text x="{x}" y="{y - 12}" font-size="15" font-weight="700" '
                f'fill="#202124">{name}</text>',
                f'<text x="{x + plot_width}" y="{y - 12}" text-anchor="end" '
                f'font-size="12" fill="#6b7075">{low:.3f} to {high:.3f}</text>',
                f'<rect x="{x}" y="{y}" width="{plot_width}" height="{plot_height}" '
                'fill="#fafafa" stroke="#d7d9dc"/>',
                f'<polyline points="{actual_points}" fill="none" stroke="{ACTUAL_COLOR}" '
                'stroke-width="2" vector-effect="non-scaling-stroke"/>',
                f'<polyline points="{predicted_points}" fill="none" '
                f'stroke="{PREDICTED_COLOR}" stroke-width="2" '
                'vector-effect="non-scaling-stroke"/>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def end_effector_trace_svg(metrics: dict[str, Any]) -> str:
    record = metrics.get("end_effector")
    if not isinstance(record, dict):
        return ""
    traces = record["traces"]
    actual = np.asarray(traces["actual_xyz_m"], dtype=np.float64) * 1000.0
    predicted = np.asarray(traces["predicted_xyz_m"], dtype=np.float64) * 1000.0
    if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != 3:
        raise ValueError("end-effector traces must both have shape (N, 3)")

    width, height = 1200, 550
    plot_width, plot_height = 540, 190
    origin_x, origin_y = 35, 50
    gap_x, gap_y = 70, 70
    limits = trace_limits(actual, predicted)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Actual and predicted SO-101 base-frame end-effector trajectories">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    for axis in range(3):
        column, row = axis % 2, axis // 2
        x = origin_x + column * (plot_width + gap_x)
        y = origin_y + row * (plot_height + gap_y)
        low, high = limits[axis]
        actual_points = svg_polyline(
            actual[:, axis], low, high, x, y, plot_width, plot_height
        )
        predicted_points = svg_polyline(
            predicted[:, axis], low, high, x, y, plot_width, plot_height
        )
        axis_name = ("X", "Y", "Z")[axis]
        parts.extend(
            [
                f'<text x="{x}" y="{y - 14}" font-size="15" font-weight="700" '
                f'fill="#202124">Base-frame {axis_name} (mm)</text>',
                f'<text x="{x + plot_width}" y="{y - 14}" text-anchor="end" '
                f'font-size="12" fill="#6b7075">{low:.1f} to {high:.1f}</text>',
                f'<rect x="{x}" y="{y}" width="{plot_width}" height="{plot_height}" '
                'fill="#fafafa" stroke="#d7d9dc"/>',
                f'<polyline points="{actual_points}" fill="none" stroke="{ACTUAL_COLOR}" '
                'stroke-width="2" vector-effect="non-scaling-stroke"/>',
                f'<polyline points="{predicted_points}" fill="none" '
                f'stroke="{PREDICTED_COLOR}" stroke-width="2" '
                'vector-effect="non-scaling-stroke"/>',
            ]
        )

    x = origin_x + plot_width + gap_x
    y = origin_y + plot_height + gap_y
    xy_actual = actual[:, :2]
    xy_predicted = predicted[:, :2]
    x_low, x_high = limits[0]
    y_low, y_high = limits[1]

    def xy_points(values: np.ndarray) -> str:
        return " ".join(
            f"{x + plot_width * (float(px) - x_low) / (x_high - x_low):.2f},"
            f"{y + plot_height - plot_height * (float(py) - y_low) / (y_high - y_low):.2f}"
            for px, py in values
        )

    parts.extend(
        [
            f'<text x="{x}" y="{y - 14}" font-size="15" font-weight="700" '
            'fill="#202124">Base-frame XY path (mm)</text>',
            f'<rect x="{x}" y="{y}" width="{plot_width}" height="{plot_height}" '
            'fill="#fafafa" stroke="#d7d9dc"/>',
            f'<polyline points="{xy_points(xy_actual)}" fill="none" '
            f'stroke="{ACTUAL_COLOR}" stroke-width="2" vector-effect="non-scaling-stroke"/>',
            f'<polyline points="{xy_points(xy_predicted)}" fill="none" '
            f'stroke="{PREDICTED_COLOR}" stroke-width="2" '
            'vector-effect="non-scaling-stroke"/>',
        ]
    )
    parts.append("</svg>")
    return "".join(parts)


def build_html(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    actual_states: np.ndarray,
    predicted_states: np.ndarray,
    joint_names: tuple[str, ...],
) -> str:
    artifacts = manifest["artifacts"]["media"]
    mp4 = next((name for name in artifacts if name.endswith(".mp4")), None)
    gif = next((name for name in artifacts if name.endswith(".gif")), None)
    if mp4:
        media_html = (
            f'<video controls preload="metadata"><source src="{html.escape(mp4)}" '
            'type="video/mp4"></video>'
        )
        if gif:
            media_html += f'<p><a href="{html.escape(gif)}">Open GIF version</a></p>'
    elif gif:
        media_html = f'<img class="animation" src="{html.escape(gif)}" alt="LeWM episode report">'
    else:
        raise ValueError("manifest has no media artifact")

    joint_rows = []
    per_joint = metrics["joint_decoder"]["per_joint"]
    direct_state_head = (
        metrics["joint_decoder"]["kind"]
        == "checkpoint direct six-state residual head"
    )
    state_label = "Direct state-head" if direct_state_head else "Joint decoder"
    predicted_label = (
        "Predicted by checkpoint state head"
        if direct_state_head
        else "Predicted from latent"
    )
    for name in joint_names:
        record = per_joint[name]
        r2 = "n/a" if record["r2"] is None else f'{record["r2"]:.4f}'
        joint_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f'<td>{record["mae"]:.6f}</td>'
            f'<td>{record["rmse"]:.6f}</td>'
            f"<td>{r2}</td>"
            "</tr>"
        )
    mode_label = html.escape(str(manifest["evaluation"]["mode"]))
    source = html.escape(str(manifest["evaluation"]["source"]))
    episode = manifest["evaluation"]["episode"]
    generated = html.escape(str(manifest["generated_at_utc"]))
    if manifest["evaluation"]["mode"] == "prediction-cycle":
        horizon = max(
            int(record["forecast_horizon_steps"])
            for record in manifest["prediction_records"]
        )
        trace_markup = trace_svg(
            actual_states[:horizon], predicted_states[:horizon], joint_names
        )
        trace_title = f"First H1-H8 forecast: actual and {predicted_label.lower()} traces"
    else:
        trace_markup = trace_svg(actual_states, predicted_states, joint_names)
        trace_title = f"Actual and {predicted_label.lower()} joint traces"
    end_effector = metrics.get("end_effector")
    if isinstance(end_effector, dict):
        eef_fact_markup = (
            '<div class="fact"><span>End-effector XYZ RMSE</span>'
            f'<strong>{1000.0 * float(end_effector["rmse_m"]):.2f} mm</strong></div>'
            '<div class="fact"><span>End-effector XYZ p95</span>'
            f'<strong>{1000.0 * float(end_effector["p95_m"]):.2f} mm</strong></div>'
        )
        eef_trace_markup = end_effector_trace_svg(metrics)
        eef_section = f"""
    <section>
      <h2>SO-ARM101 base-frame end-effector trajectory</h2>
      <p>This is official SO101 MuJoCo forward kinematics from the six predicted joint states. It is not an image-plane projection; front-camera calibration is required for a pixel overlay.</p>
      <div class="legend"><span><i class="key" style="background:var(--actual)"></i>Actual FK</span><span><i class="key" style="background:var(--pred)"></i>Predicted FK</span></div>
      <div class="trace">{eef_trace_markup}</div>
    </section>"""
    else:
        eef_fact_markup = ""
        eef_section = ""
    pose_source = html.escape(str(metrics["retrieval"]["pose_source"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SO-101 LeWM held-out episode report</title>
  <style>
    :root {{ --ink:#202124; --muted:#5f6368; --line:#d7d9dc; --actual:{ACTUAL_COLOR}; --pred:{PREDICTED_COLOR}; --warn:#f4b942; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:#fff; font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }}
    header {{ background:#202124; color:#fff; padding:24px max(24px,calc((100% - 1200px)/2)); border-bottom:6px solid var(--warn); }}
    h1 {{ margin:0 0 5px; font-size:28px; letter-spacing:0; }}
    h2 {{ margin:0 0 14px; font-size:20px; letter-spacing:0; }}
    header p {{ margin:0; color:#d7d9dc; }}
    main {{ max-width:1200px; margin:0 auto; padding:24px; }}
    section {{ padding:22px 0; border-bottom:1px solid var(--line); }}
    .notice {{ border-left:6px solid var(--pred); padding:12px 16px; background:#f7f7f7; font-weight:650; }}
    .facts {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); }}
    .fact {{ background:#fff; padding:12px; min-width:0; overflow-wrap:anywhere; }}
    .fact span {{ display:block; color:var(--muted); font-size:12px; }}
    .fact strong {{ display:block; margin-top:3px; font-size:17px; }}
    video,.animation {{ display:block; width:100%; height:auto; background:#202124; }}
    .legend {{ display:flex; gap:22px; align-items:center; color:var(--muted); margin:8px 0 14px; }}
    .key {{ display:inline-block; width:24px; height:3px; vertical-align:middle; margin-right:7px; }}
    .trace {{ width:100%; overflow-x:auto; border:1px solid var(--line); }}
    .trace svg {{ display:block; min-width:780px; width:100%; height:auto; }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
    th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); }}
    th {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
    a {{ color:#075e68; }}
    footer {{ color:var(--muted); padding:18px 0; font-size:13px; }}
    @media (max-width:760px) {{ .facts {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} main {{ padding:16px; }} h1 {{ font-size:23px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>SO-101 LeWM held-out episode report</h1>
    <p>{source}:{episode} | {mode_label} | generated {generated}</p>
  </header>
  <main>
    <section>
      <div class="notice">The right panel is train-episode-only RGB selected after a hard pose gate based on {pose_source}, then ranked by predicted-latent distance. It is not generated RGB. No observed future state or RGB enters retrieval.</div>
    </section>
    <section class="facts" aria-label="Summary metrics">
      <div class="fact"><span>Latent prediction MSE</span><strong>{metrics["latent"]["prediction_mse"]:.6f}</strong></div>
      <div class="fact"><span>Copy baseline MSE</span><strong>{metrics["latent"]["copy_baseline_mse"]:.6f}</strong></div>
      <div class="fact"><span>{state_label} RMSE</span><strong>{metrics["joint_decoder"]["overall_rmse"]:.6f}</strong></div>
      <div class="fact"><span>Predicted-to-neighbor MSE</span><strong>{metrics["retrieval"]["predicted_to_neighbor_mse"]:.6f}</strong></div>
      <div class="fact"><span>Geometry-supported retrievals</span><strong>{metrics["retrieval"]["matched_fraction"]:.1%}</strong></div>
      <div class="fact"><span>Selected command-joint RMSE</span><strong>{metrics["retrieval"]["selected_command_joint_rmse"]:.3f}</strong></div>
      <div class="fact"><span>Selected arm max error</span><strong>{metrics["retrieval"]["selected_command_arm_max_error"]:.3f}</strong></div>
      <div class="fact"><span>Selected gripper error</span><strong>{metrics["retrieval"]["selected_command_gripper_error"]:.3f}</strong></div>
      {eef_fact_markup}
    </section>
    <section>
      <h2>Episode animation</h2>
      {media_html}
    </section>
    <section>
      <h2>{trace_title}</h2>
      <div class="legend"><span><i class="key" style="background:var(--actual)"></i>Actual</span><span><i class="key" style="background:var(--pred)"></i>{predicted_label}</span></div>
      <div class="trace">{trace_markup}</div>
    </section>
    {eef_section}
    <section>
      <h2>Six-joint prediction metrics</h2>
      <table>
        <thead><tr><th>Joint</th><th>MAE</th><th>RMSE</th><th>R2</th></tr></thead>
        <tbody>{''.join(joint_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>Audit files</h2>
      <p><a href="manifest.json">manifest.json</a> records the checkpoint contract, split, bank membership rules, retrieval rows, and leakage controls. <a href="metrics.json">metrics.json</a> contains aggregate metrics and complete six-joint traces.</p>
    </section>
    <footer>Held-out target RGB and observed future state are accessed only after prediction and retrieval selection, for evaluation and display.</footer>
  </main>
</body>
</html>
"""


def prepare_output(output: Path, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    known = [
        "index.html",
        "manifest.json",
        "metrics.json",
        "teacher_forced_h1.mp4",
        "teacher_forced_h1.gif",
        "open_loop.mp4",
        "open_loop.gif",
        "prediction_cycle.mp4",
        "prediction_cycle.gif",
        # Older action-policy visualizers used these names.  Remove them when
        # refreshing the canonical latest report so stale media cannot be
        # mistaken for the current checkpoint.
        "forecast.mp4",
        "forecast.gif",
    ]
    existing = [output / name for name in known if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"report outputs already exist below {output}; pass --overwrite to replace them"
        )
    if overwrite:
        for path in existing:
            path.unlink()


def build_manifest(
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint: dict[str, Any],
    embedded_args: dict[str, Any],
    caches: list[CacheView],
    choice: EpisodeChoice,
    frameskip: int,
    history: int,
    predictions: PredictionResult,
    retrieval: RetrievalResult,
    bank_summaries: list[dict[str, Any]],
    decoder: RidgeDecoder,
    media_files: list[str],
    rendered_frames: int,
    media_fps: float,
    device: torch.device,
    action_normalization: str,
) -> dict[str, Any]:
    direct_state_prediction = predictions.predicted_states is not None
    protocol = checkpoint.get("run_config", {}).get("contract", {}).get("protocol")
    is_memory_model = protocol in MEMORY_PROTOCOLS
    retrieval_pose_source = (
        "direct six-state residual-head prediction"
        if direct_state_prediction
        else "final recorded action target"
    )
    destination_rows = choice.rows[predictions.destination_local]
    baseline_rows = choice.rows[predictions.baseline_local]
    input_rows = choice.rows[predictions.display_input_local]
    records = []
    for index in range(len(destination_rows)):
        retrieved_row = int(retrieval.rows[index])
        horizon_steps = max(
            1,
            int(
                (
                    predictions.destination_local[index]
                    - predictions.display_input_local[index]
                )
                // frameskip
            ),
        )
        records.append(
            {
                "prediction_index": index,
                "task": int(predictions.tasks[index]),
                "context_size": int(predictions.context_sizes[index]),
                "forecast_horizon_steps": horizon_steps,
                "forecast_horizon_raw_frames": horizon_steps * frameskip,
                "target_cache_row": int(destination_rows[index]),
                "target_frame_index": int(choice.cache.data.frame_indices[destination_rows[index]]),
                "copy_baseline_cache_row": int(baseline_rows[index]),
                "display_input_cache_row": int(input_rows[index]),
                "retrieval_geometry_supported": bool(retrieval.valid[index]),
                "retrieved_train_cache_row": (
                    retrieved_row if retrieval.valid[index] else None
                ),
                "retrieved_train_episode": (
                    int(choice.cache.data.episodes[retrieved_row])
                    if retrieval.valid[index]
                    else None
                ),
                "retrieved_train_frame_index": (
                    int(choice.cache.data.frame_indices[retrieved_row])
                    if retrieval.valid[index]
                    else None
                ),
                "predicted_to_retrieved_latent_mse": (
                    float(retrieval.latent_mse[index])
                    if retrieval.valid[index]
                    else None
                ),
                "retrieved_command_joint_rmse": float(
                    retrieval.command_joint_rmse[index]
                ),
                "retrieved_command_arm_max_error": (
                    float(retrieval.command_arm_max_error[index])
                    if retrieval.valid[index]
                    else None
                ),
                "retrieved_command_gripper_error": (
                    float(retrieval.command_gripper_error[index])
                    if retrieval.valid[index]
                    else None
                ),
            }
        )
    return {
        "format": REPORT_FORMAT,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "execution_device": str(device),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "step": int(checkpoint.get("step", -1)),
            "embedded_args_authoritative": not bool(args.cache),
            "cache_override": [str(path) for path in (args.cache or [])],
            "embedded_args": jsonable(embedded_args),
        },
        "model_contract": {
            "builder": (
                f"MultiScaleHistoryLeWM({protocol})"
                if is_memory_model
                else
                "StateResidualLeWM(train_so101_lewm.build_model)"
                if direct_state_prediction
                else "train_so101_lewm.build_model"
            ),
            "protocol": protocol,
            "frameskip": frameskip,
            "history": history,
            "action_dim": frameskip * 6,
            "memory_offsets": list(
                checkpoint.get("run_config", {})
                .get("contract", {})
                .get("memory_offsets", [])
            ),
            "prediction_path": (
                "full causal V9 memory path: recent visual/state/action history "
                "plus memory embeddings, memory actions, memory states, and ages"
                if is_memory_model
                else "direct state-residual prediction path"
            ),
            "action_normalization": action_normalization,
            "latent": "encoder CLS token after projector; prediction after pred_proj",
            "state_prediction": (
                "direct checkpoint six-state residual head"
                if direct_state_prediction
                else "train-only ridge decoder used for visualization"
            ),
        },
        "cache_contract": {
            "format": "so101-lewm-cache-v1",
            "ordered_resolved_caches": [
                {
                    "cache_id": cache.cache_id,
                    "source": cache.source,
                    "path": str(cache.root),
                    "frames": int(cache.manifest["frames"]),
                    "camera": cache.manifest.get("camera"),
                    "fps": cache.manifest.get("fps"),
                    "joint_names": list(cache.joint_names),
                }
                for cache in caches
            ],
        },
        "split_contract": {
            "implementation": "train_so101_lewm.episode_splits",
            "split_seed": int(
                embedded_args.get(
                    "split_seed",
                    checkpoint.get("run_config", {}).get("contract", {}).get(
                        "split_seed", 0
                    ),
                )
            ),
            "val_fraction": float(
                embedded_args.get(
                    "val_fraction",
                    checkpoint.get("run_config", {}).get("contract", {}).get(
                        "val_fraction", 0.1
                    ),
                )
            ),
            "exclusions": list(
                embedded_args.get(
                    "exclude_episode",
                    checkpoint.get("run_config", {}).get("contract", {}).get(
                        "exclusions", []
                    ),
                )
            ),
            "selected_episode_is_held_out": True,
        },
        "evaluation": {
            "source": choice.cache.source,
            "cache_id": choice.cache.cache_id,
            "episode": choice.episode,
            "episode_frames": len(choice.rows),
            "mode": predictions.mode,
            "teacher_forced_contract": (
                "full episode, one frameskip-horizon prediction for every eligible raw-frame "
                "offset; each context RGB row is strictly earlier than its target"
                if predictions.mode == "teacher-forced"
                else None
            ),
            "open_loop_contract": (
                "one recursive frameskip phase seeded by exactly the checkpoint history of "
                "held-out RGB frames; no later held-out RGB enters prediction"
                if predictions.mode == "open-loop"
                else None
            ),
            "prediction_cycle_contract": (
                "the current held-out RGB history is paused while a fresh recursive forecast "
                "uses only that history and the recorded future action chunks; the observation "
                "then advances by one frameskip and the forecast is restarted"
                if predictions.mode == "prediction-cycle"
                else None
            ),
            "prediction_count": len(predictions.predicted_latents),
            "render_stride": args.render_stride,
            "rendered_frames": rendered_frames,
            "media_fps": media_fps,
        },
        "ridge_decoder": {
            "training_split_only": True,
            "scope": "all frames in non-held-out episodes across all checkpoint caches",
            "outputs": list(choice.cache.joint_names),
            "training_rows": decoder.training_rows,
            "alpha": decoder.alpha,
            "latent_standardization_fit_on_training_rows_only": True,
        },
        "retrieval": {
            "display_semantics": (
                "pose-gated train-episode nearest-neighbor RGB of predicted latent; "
                "not generated RGB"
            ),
            "pose_source": retrieval_pose_source,
            "distance_query": (
                f"{retrieval_pose_source} applies hard six-joint RMS, per-arm-joint, "
                "and gripper gates; predicted latent ranks only supported candidates"
            ),
            "distance": "mean squared error in projected LeWM latent coordinates",
            "restriction": (
                "same checkpoint cache source, same integer task index, training split, and "
                f"command-joint RMSE <= {args.retrieval_max_command_joint_rmse:g}, "
                f"arm max <= {args.retrieval_max_arm_joint_error:g}, "
                f"gripper error <= {args.retrieval_max_gripper_error:g}"
            ),
            "max_command_joint_rmse": args.retrieval_max_command_joint_rmse,
            "max_arm_joint_error": args.retrieval_max_arm_joint_error,
            "max_gripper_error": args.retrieval_max_gripper_error,
            "groups": bank_summaries,
        },
        "leakage_controls": {
            "prediction_inputs": "strictly prior held-out RGB latents plus recorded action chunks",
            "retrieval_bank": "RGB/latents from hash-split training episodes only",
            "retrieval_query": (
                f"predicted latent plus {retrieval_pose_source}; no target latent, observed "
                "target state, or target RGB"
            ),
            "target_rgb_access": (
                "after all predictions and nearest-neighbor row selections; then used only for "
                "target-latent metrics and display"
            ),
            "right_panel": (
                "retrieved same-source/same-task training RGB, explicitly labelled not generated"
            ),
        },
        "prediction_records": records,
        "artifacts": {
            "index": "index.html",
            "manifest": "manifest.json",
            "metrics": "metrics.json",
            "media": media_files,
        },
    }


def create_episode_report(
    args: argparse.Namespace,
    output: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint: dict[str, Any],
    embedded_args: dict[str, Any],
    caches: list[CacheView],
    assignments: SplitAssignments,
    choice: EpisodeChoice,
    frameskip: int,
    history: int,
    action_center: np.ndarray,
    action_scale: np.ndarray,
    action_normalization: str,
    model: torch.nn.Module,
    decoder: RidgeDecoder,
    bank_groups: dict[tuple[int, int], BankGroup],
    bank_summaries: list[dict[str, Any]],
    device: torch.device,
    kinematics: SO101MuJoCoKinematics | None,
) -> dict[str, Any]:
    prepare_output(output, args.overwrite)
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "teacher-forced":
        predictions = teacher_forced_predictions(
            model,
            choice,
            frameskip,
            history,
            action_center,
            action_scale,
            args.batch_size,
            action_center,
            action_scale,
        )
    elif args.mode == "open-loop":
        predictions = open_loop_predictions(
            model,
            choice,
            frameskip,
            history,
            action_center,
            action_scale,
            args.batch_size,
            action_center,
            action_scale,
        )
    else:
        predictions = prediction_cycle_predictions(
            model,
            choice,
            frameskip,
            history,
            args.cycle_horizon,
            action_center,
            action_scale,
            args.batch_size,
            action_center,
            action_scale,
            tuple(
                int(value)
                for value in checkpoint.get("run_config", {})
                .get("contract", {})
                .get("memory_offsets", ())
            ),
        )

    # Neighbor rows are finalized before any held-out target RGB or observed
    # target state is encoded/materialized. The recorded action targets used
    # by the legacy model are already predictor inputs. The v4 checkpoint uses
    # its own deployable state prediction, never the observed target state.
    expected_joint_states, retrieval_pose_source = retrieval_expected_joint_states(
        choice, predictions
    )
    retrieval = retrieve_neighbors(
        predictions,
        choice.cache,
        bank_groups,
        assignments,
        expected_joint_states,
        args.retrieval_max_command_joint_rmse,
        args.retrieval_max_arm_joint_error,
        args.retrieval_max_gripper_error,
    )
    destination_rows = choice.rows[predictions.destination_local]
    baseline_rows = choice.rows[predictions.baseline_local]
    target_latents = encode_rows(model, choice.cache, destination_rows, args.batch_size)
    baseline_latents = encode_rows(model, choice.cache, baseline_rows, args.batch_size)
    metrics, actual_states, predicted_states = build_metrics(
        choice,
        predictions,
        retrieval,
        target_latents,
        baseline_latents,
        decoder,
        retrieval_pose_source,
        kinematics,
    )

    source_fps = float(choice.cache.manifest.get("fps") or 30.0)
    prediction_frame_step = (
        frameskip
        if predictions.mode in {"open-loop", "prediction-cycle"}
        else 1
    )
    media_fps = args.output_fps or (
        source_fps / (prediction_frame_step * args.render_stride)
    )
    media_files, rendered_frames = render_media(
        output,
        args.media,
        media_fps,
        args.render_stride,
        choice,
        predictions,
        retrieval,
        actual_states,
        predicted_states,
        frameskip,
    )
    manifest = build_manifest(
        args,
        checkpoint_path,
        checkpoint_sha256,
        checkpoint,
        embedded_args,
        caches,
        choice,
        frameskip,
        history,
        predictions,
        retrieval,
        bank_summaries,
        decoder,
        media_files,
        rendered_frames,
        media_fps,
        device,
        action_normalization,
    )
    write_json(output / "metrics.json", metrics)
    write_json(output / "manifest.json", manifest)
    index = build_html(
        manifest,
        metrics,
        actual_states,
        predicted_states,
        choice.cache.joint_names,
    )
    atomic_write_text(output / "index.html", index)
    return {
        "source": choice.cache.source,
        "dataset_id": choice.cache.dataset_id,
        "task": episode_task_key(choice),
        "episode": choice.episode,
        "directory": output.name,
        "prediction_mse": metrics["latent"]["prediction_mse"],
        "copy_baseline_mse": metrics["latent"]["copy_baseline_mse"],
        "state_prediction_kind": metrics["joint_decoder"]["kind"],
        "retrieval_pose_source": metrics["retrieval"]["pose_source"],
        "media": media_files,
    }


def build_gallery_index(reports: list[dict[str, Any]], mode: str) -> str:
    sections = []
    for report in reports:
        task = html.escape(str(report["task"]))
        source = html.escape(str(report["source"]))
        directory = html.escape(str(report["directory"]), quote=True)
        improvement = report["copy_baseline_mse"] - report["prediction_mse"]
        media = report.get("media", [])
        mp4 = next((str(name) for name in media if str(name).endswith(".mp4")), None)
        gif = next((str(name) for name in media if str(name).endswith(".gif")), None)
        if mp4:
            media_markup = (
                '<video controls muted loop playsinline preload="metadata">'
                f'<source src="{directory}/{html.escape(mp4, quote=True)}" '
                'type="video/mp4"></video>'
            )
        elif gif:
            media_markup = (
                f'<img class="animation" loading="lazy" '
                f'src="{directory}/{html.escape(gif, quote=True)}" '
                f'alt="{task} prediction animation">'
            )
        else:
            media_markup = '<p class="missing">No media artifact.</p>'
        gif_link = (
            f'<a href="{directory}/{html.escape(gif, quote=True)}">GIF</a> · '
            if gif
            else ""
        )
        sections.append(
            f"""<section><div class="head"><div><h2>{task}</h2>
              <p>{source} | held-out episode {report['episode']}</p></div>
              <div class="metrics">latent MSE <b>{report['prediction_mse']:.5f}</b><br>
              copy improvement <b>{improvement:+.5f}</b></div></div>
              {media_markup}
              <p class="links">{gif_link}<a href="{directory}/index.html">Full report</a></p>
            </section>"""
        )
    cycle_note = (
        "The current observation pauses while H1-H8 predictions play; it then advances "
        "one frameskip and starts a new forecast. "
        if mode == "prediction-cycle"
        else ""
    )
    pose_sources = {
        str(report.get("retrieval_pose_source", "commanded joint target"))
        for report in reports
    }
    pose_note = (
        next(iter(pose_sources))
        if len(pose_sources) == 1
        else "the report-specific deployable pose source"
    )
    retrieval_note = (
        "command-pose-gated"
        if pose_note == "commanded joint target"
        else f"pose-gated by {pose_note}"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 LeWM task gallery</title><style>
*{{box-sizing:border-box}}body{{font-family:system-ui,sans-serif;background:#edf1f3;color:#20242a;margin:0;letter-spacing:0}}
header{{background:#202124;color:white;padding:24px max(20px,calc((100vw - 1120px)/2));border-bottom:6px solid #f4b942}}
h1,h2,p{{margin:0}}header p{{color:#d7d9dc;margin-top:8px;line-height:1.5}}
main{{max-width:1120px;margin:auto;padding:10px 20px 60px}}section{{padding:28px 0 34px;border-bottom:1px solid #c8d0d5}}
.head{{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:12px}}.head p,.links{{color:#58606b;margin-top:5px}}
.metrics{{text-align:right;color:#58606b;font-size:13px}}.metrics b{{color:#20242a}}video,.animation{{display:block;width:100%;height:auto;background:#202124;border:1px solid #bbc5cb}}
.links{{margin-top:9px}}a{{color:#075e68}}.missing{{padding:20px;background:white}}@media(max-width:700px){{.head{{display:block}}.metrics{{text-align:left;margin-top:8px}}}}
</style></head><body><header><h1>SO-101 LeWM: 12-Task Prediction Gallery</h1>
<p>{html.escape(cycle_note)}Right-side RGB is a labelled, {html.escape(retrieval_note)} train-only neighbor ranked by predicted latent, never generated RGB.</p></header>
<main>{''.join(sections)}</main></body></html>"""


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    except RuntimeError:
        pass

    (
        checkpoint,
        embedded_args,
        cache_paths,
        frameskip,
        history,
        checkpoint_sha256,
    ) = checkpoint_contract(checkpoint_path, args.cache_base, args.cache)
    caches = load_caches(cache_paths)
    selection_caches = (
        [cache for cache in caches if cache.source == args.source]
        if args.source
        else caches
    )
    if not selection_caches:
        available = ", ".join(cache.source for cache in caches)
        raise ValueError(
            f"unknown --source {args.source!r}; checkpoint sources: {available}"
        )
    contract = checkpoint.get("run_config", {}).get("contract", {})
    exclusions = parse_exclusions(
        embedded_args.get("exclude_episode", contract.get("exclusions", []))
    )
    assignments = episode_splits(
        [cache.data for cache in caches],
        exclusions,
        split_seed=int(embedded_args.get("split_seed", contract.get("split_seed", 0))),
        val_fraction=float(
            embedded_args.get("val_fraction", contract.get("val_fraction", 0.1))
        ),
    )
    if args.all_tasks:
        choices = choose_one_episode_per_task(
            selection_caches, frameskip, history, assignments
        )
    else:
        choices = [
            choose_episode(
                selection_caches, frameskip, history, args.episode, assignments
            )
        ]
    action_center, action_scale, action_normalization = checkpoint_action_normalizer(
        checkpoint, embedded_args
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = build_checkpoint_model(checkpoint, frameskip, history, device)
    kinematics = (
        SO101MuJoCoKinematics(args.so101_model)
        if args.so101_model is not None
        else None
    )

    decoder, bank_groups, bank_summaries = build_training_artifacts(
        model,
        caches,
        assignments,
        args.batch_size,
        args.ridge_alpha,
        args.bank_max_per_group,
    )
    reports = []
    for choice in choices:
        report_output = output / report_slug(choice) if args.all_tasks else output
        report = create_episode_report(
            args,
            report_output,
            checkpoint_path,
            checkpoint_sha256,
            checkpoint,
            embedded_args,
            caches,
            assignments,
            choice,
            frameskip,
            history,
            action_center,
            action_scale,
            action_normalization,
            model,
            decoder,
            bank_groups,
            bank_summaries,
            device,
            kinematics,
        )
        reports.append(report)
        print(
            f"report: {report_output / 'index.html'} "
            f"({choice.cache.source}:{choice.episode})",
            flush=True,
        )
    if args.all_tasks:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "gallery.json", {"format": REPORT_FORMAT, "reports": reports})
        atomic_write_text(output / "index.html", build_gallery_index(reports, args.mode))
        print(f"gallery: {output / 'index.html'}")


if __name__ == "__main__":
    main()
