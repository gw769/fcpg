#!/usr/bin/env python3
"""Train a small SO-101 FCPG corrector on synthetic camera drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fcpg.modules import sample_feature_map_at_pixels
from fcpg.so101_fcpg_lewm_bridge import (
    SO101_FCPG_CORRECTOR_VARIANTS,
    SO101FCPGConditionedLeWM,
    build_so101_fcpg_corrector,
)
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from fcpg.so101_synthetic_dataset import SO101SyntheticFCPGDataset
from scripts.train_so101_lewm import IMAGENET_MEAN, IMAGENET_STD
from scripts.visualize_so101_lewm import (
    build_checkpoint_model,
    checkpoint_contract,
)
from v10.config import FORMAT as V10_FORMAT, V10Config
from v10.data import JOINT_SCALE
from v10.model import SO101V10


SYNTHETIC_TRAINING_SCOPE = "synthetic_degradation"
TRAINABLE_STATE_PREFIX = "corrector."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--initial-corrector-checkpoint",
        type=Path,
        help="Warm-start only model.corrector from a provenance-matched checkpoint.",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        help="Fail unless the frozen source checkpoint has this SHA-256.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=53102)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--corrector-variant",
        choices=SO101_FCPG_CORRECTOR_VARIANTS,
        default="independent3",
    )
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.feature_batch_size, args.log_every) <= 0:
        parser.error("step and batch arguments must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("optimizer arguments are invalid")
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    if args.expected_checkpoint_sha256 is not None and (
        len(args.expected_checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.expected_checkpoint_sha256)
    ):
        parser.error("--expected-checkpoint-sha256 must be 64 lowercase hex characters")
    return args


def normalize_rgb(rgb: np.ndarray) -> torch.Tensor:
    pixels = torch.from_numpy(np.asarray(rgb).copy()).permute(0, 3, 1, 2)
    pixels = pixels.float().div_(255.0)
    return (pixels - IMAGENET_MEAN) / IMAGENET_STD


def unit_rgb(rgb: np.ndarray) -> torch.Tensor:
    pixels = torch.from_numpy(np.asarray(rgb).copy()).permute(0, 3, 1, 2)
    return pixels.float().div_(255.0)


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, metadata, and exact bytes in deterministic order."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def load_checkpoint_and_sha256(path: Path) -> tuple[dict[str, Any], str]:
    before = file_identity(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    after_load = file_identity(path)
    if before != after_load:
        raise RuntimeError(f"checkpoint changed while loading: {path}")
    checkpoint_sha = sha256_file(path)
    after_hash = file_identity(path)
    if after_load != after_hash:
        raise RuntimeError(f"checkpoint changed while hashing: {path}")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    return checkpoint, checkpoint_sha


def snapshot_frozen_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
        if not name.startswith(TRAINABLE_STATE_PREFIX)
    }


def assert_frozen_state_unchanged(
    module: torch.nn.Module,
    before: Mapping[str, torch.Tensor],
) -> str:
    after = snapshot_frozen_state(module)
    if set(after) != set(before):
        raise RuntimeError("frozen state keys changed during FCPG-only training")
    changed = [name for name in sorted(before) if not torch.equal(before[name], after[name])]
    if changed:
        raise RuntimeError(f"frozen tensors changed during FCPG-only training: {changed[:8]}")
    before_sha = state_dict_sha256(before)
    after_sha = state_dict_sha256(after)
    if before_sha != after_sha:
        raise RuntimeError("frozen state hash changed despite tensor equality")
    return after_sha


def configure_corrector_only(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    module.requires_grad_(False)
    corrector = getattr(module, "corrector", None)
    if not isinstance(corrector, torch.nn.Module):
        raise TypeError("FCPG-only training requires a corrector module")
    corrector.requires_grad_(True)
    module.eval()
    corrector.train()
    named = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    if not named or any(not name.startswith(TRAINABLE_STATE_PREFIX) for name, _ in named):
        raise RuntimeError("trainable parameter scope escaped model.corrector")
    return [parameter for _, parameter in named]


def assert_optimizer_scope(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    corrector = getattr(module, "corrector", None)
    if not isinstance(corrector, torch.nn.Module):
        raise TypeError("optimizer scope requires a corrector module")
    if any(not parameter.requires_grad for parameter in corrector.parameters()):
        raise RuntimeError("every corrector parameter must be trainable")
    expected = {id(parameter) for parameter in corrector.parameters()}
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if actual != expected:
        raise RuntimeError("optimizer parameters are not exactly model.corrector")


def assert_frozen_gradients_absent(module: torch.nn.Module) -> None:
    offenders = [
        name
        for name, parameter in module.named_parameters()
        if not name.startswith(TRAINABLE_STATE_PREFIX) and parameter.grad is not None
    ]
    if offenders:
        raise RuntimeError(f"frozen parameters received gradients: {offenders[:8]}")


def assert_corrector_gradients(module: torch.nn.Module) -> float:
    gradients = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if name.startswith(TRAINABLE_STATE_PREFIX) and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("corrector received no gradients")
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("corrector received non-finite gradients")
    squared_norm = sum(gradient.detach().float().square().sum() for gradient in gradients)
    norm = float(squared_norm.sqrt().cpu())
    if norm <= 0.0:
        raise RuntimeError("corrector gradient norm is zero")
    return norm


def _v10_config(checkpoint: Mapping[str, Any]) -> V10Config:
    raw = dict(checkpoint.get("config") or {})
    for key in ("history_offsets", "future_offsets"):
        if isinstance(raw.get(key), list):
            raw[key] = tuple(int(value) for value in raw[key])
    config = V10Config(**raw)
    config.validate()
    return config


def build_v12_checkpoint_model(
    checkpoint: Mapping[str, Any],
    device: torch.device,
    *,
    vit_source_override: Path | None = None,
) -> SO101V10:
    if checkpoint.get("format") != V10_FORMAT:
        raise ValueError("checkpoint is not in the V10/V12 model family")
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, Mapping) or architecture.get("family") != "v12":
        raise ValueError("FCPG-only V12 training requires architecture.family='v12'")
    if architecture.get("structured_goal") != "episode_terminal_state":
        raise ValueError("unsupported V12 structured-goal construction")
    if architecture.get("task_class_conditioning") is not True:
        raise ValueError("unsupported V12 task-class construction")
    recorded_vit_source = checkpoint.get("vit_source")
    if not isinstance(recorded_vit_source, str):
        raise FileNotFoundError(
            f"V12 ViT source is unavailable: {recorded_vit_source}"
        )
    vit_source = (
        Path(recorded_vit_source)
        if vit_source_override is None
        else Path(vit_source_override)
    ).expanduser().resolve()
    if not vit_source.is_file():
        raise FileNotFoundError(f"V12 ViT source is unavailable: {vit_source}")
    model = SO101V10(
        config=_v10_config(checkpoint),
        vit_checkpoint=Path(vit_source),
        structured_goal=True,
        structured_goal_source="episode_terminal_state",
        goal_query_residual=False,
        task_class_conditioning=True,
        state_residual_mode=str(architecture.get("state_residual_mode", "uniform")),
        state_residual_steps=float(architecture.get("state_residual_steps", 1.0)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    contract = checkpoint.get("action_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("V12 checkpoint is missing action_contract")
    model.set_action_contract(
        np.asarray(contract["step_cap"], dtype=np.float32),
        np.asarray(contract["lower"], dtype=np.float32),
        np.asarray(contract["upper"], dtype=np.float32),
    )
    model.to(device)
    model.eval()
    return model


def load_source_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    *,
    corrector_variant: str,
    vit_source_override: Path | None = None,
) -> tuple[dict[str, Any], torch.nn.Module, str, str]:
    path = checkpoint_path.resolve()
    checkpoint, checkpoint_sha = load_checkpoint_and_sha256(path)
    if checkpoint.get("format") == V10_FORMAT:
        if corrector_variant != "independent3":
            raise ValueError("V12 checkpoints only support their integrated independent3 corrector")
        return (
            checkpoint,
            build_v12_checkpoint_model(
                checkpoint,
                device,
                vit_source_override=vit_source_override,
            ),
            checkpoint_sha,
            "v12",
        )

    checkpoint, _, _, frameskip, history, verified_sha = checkpoint_contract(path, None)
    if verified_sha != checkpoint_sha:
        raise RuntimeError("checkpoint hash changed while loading")
    model = build_checkpoint_model(checkpoint, frameskip, history, device)
    if not isinstance(model, StateResidualLeWM):
        raise TypeError("legacy SO-101 FCPG training requires a state-residual LeWM checkpoint")
    corrector = build_so101_fcpg_corrector(
        model,
        camera_context_dim=5,
        variant=corrector_variant,
    )
    system = SO101FCPGConditionedLeWM(
        model,
        corrector=corrector,
        camera_context_dim=5,
        freeze_model=True,
    ).to(device)
    return checkpoint, system, checkpoint_sha, "state_residual_lewm"


def load_corrector_initializer(
    system: torch.nn.Module,
    path: Path,
    *,
    source_checkpoint_sha256: str,
    source_family: str,
    corrector_variant: str,
) -> dict[str, Any]:
    payload, file_sha = load_checkpoint_and_sha256(path.resolve())
    state = payload.get("corrector")
    if not isinstance(state, Mapping):
        raise ValueError("initial corrector checkpoint is missing its state dictionary")
    if payload.get("source_checkpoint_sha256") != source_checkpoint_sha256:
        raise ValueError("initial corrector was trained from a different source checkpoint")
    if payload.get("source_family") != source_family:
        raise ValueError("initial corrector source family does not match")
    if payload.get("corrector_variant") != corrector_variant:
        raise ValueError("initial corrector variant does not match")
    state_sha = state_dict_sha256(state)
    if payload.get("corrector_state_sha256_after") != state_sha:
        raise ValueError("initial corrector state SHA-256 disagrees with metadata")
    corrector = getattr(system, "corrector", None)
    if not isinstance(corrector, torch.nn.Module):
        raise TypeError("warm-start requires model.corrector")
    corrector.load_state_dict(state, strict=True)
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": file_sha,
        "corrector_state_sha256": state_sha,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_family": source_family,
        "corrector_variant": corrector_variant,
        "training_scope": payload.get("training_scope"),
        "has_physical_labels": payload.get("has_physical_labels"),
    }


def split_identity_audit(dataset: SO101SyntheticFCPGDataset) -> dict[str, int | bool]:
    arrays = dataset.arrays
    split = np.asarray(arrays["split"])
    train = split == 0
    val = split == 1
    train_identity = set(
        zip(
            np.asarray(arrays["source_index"])[train].astype(int).tolist(),
            np.asarray(arrays["episode"])[train].astype(int).tolist(),
            strict=True,
        )
    )
    val_identity = set(
        zip(
            np.asarray(arrays["source_index"])[val].astype(int).tolist(),
            np.asarray(arrays["episode"])[val].astype(int).tolist(),
            strict=True,
        )
    )
    overlap = train_identity & val_identity
    if overlap:
        raise ValueError(f"synthetic train/val physical identities overlap: {sorted(overlap)[:8]}")
    return {
        "train_physical_identities": len(train_identity),
        "val_physical_identities": len(val_identity),
        "overlap": len(overlap),
        "passed": True,
    }


def masked_smooth_l1(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if predicted_delta.shape != target_delta.shape or predicted_delta.shape[-1] != 2:
        raise ValueError("delta tensors must share shape [B, N, 2]")
    if valid.shape != predicted_delta.shape[:-1]:
        raise ValueError("valid mask must align with delta tensors")
    per_coordinate = F.smooth_l1_loss(
        predicted_delta,
        target_delta,
        beta=1.0,
        reduction="none",
    )
    weights = valid.to(dtype=per_coordinate.dtype).unsqueeze(-1)
    denominator = weights.sum() * predicted_delta.shape[-1]
    if float(denominator.detach().cpu()) <= 0:
        raise ValueError("training batch has no valid anchors")
    return (per_coordinate * weights).sum() / denominator


def metric_row(
    q_nominal: np.ndarray,
    q_corrected: np.ndarray,
    q_true: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    support = np.asarray(valid, dtype=bool)
    if not np.any(support):
        return {"anchors": 0, "nominal_epe_px": float("nan"), "corrected_epe_px": float("nan")}
    nominal = np.linalg.norm(q_nominal[support] - q_true[support], axis=-1)
    corrected = np.linalg.norm(q_corrected[support] - q_true[support], axis=-1)
    return {
        "anchors": int(len(nominal)),
        "nominal_epe_px": float(nominal.mean()),
        "corrected_epe_px": float(corrected.mean()),
    }


def aggregate_metrics(
    dataset: SO101SyntheticFCPGDataset,
    indices: np.ndarray,
    corrected: np.ndarray,
) -> dict[str, Any]:
    arrays = dataset.arrays
    views = dataset.manifest["camera"]["views"]
    buckets: dict[str, list[int]] = defaultdict(list)
    for raw_index in indices:
        index = int(raw_index)
        camera = str(views[int(arrays["camera_index"][index])]["name"])
        clean = not bool(np.any(np.abs(np.asarray(arrays["drift"][index])) > 1.0e-8))
        buckets["all"].append(index)
        buckets["clean" if clean else "drift"].append(index)
        buckets[f"{camera}/all"].append(index)
        buckets[f"{camera}/{'clean' if clean else 'drift'}"].append(index)

    metrics: dict[str, Any] = {}
    for name, rows in sorted(buckets.items()):
        valid = np.asarray(arrays["valid"][rows], dtype=bool)
        nominal = np.asarray(arrays["q_nominal"][rows])
        target = np.asarray(arrays["q_true"][rows])
        row = metric_row(nominal, corrected[rows], target, valid)
        nominal_epe = float(row["nominal_epe_px"])
        corrected_epe = float(row["corrected_epe_px"])
        row["frames"] = len(rows)
        row["relative_improvement"] = (
            (nominal_epe - corrected_epe) / nominal_epe
            if nominal_epe > 1.0e-8
            else float("nan")
        )
        metrics[name] = row
    return metrics


@torch.inference_mode()
def token_sampling_metrics(
    dataset: SO101SyntheticFCPGDataset,
    indices: np.ndarray,
    features: torch.Tensor,
    corrected: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    arrays = dataset.arrays
    width = int(dataset.manifest["width"])
    height = int(dataset.manifest["height"])
    nominal_mse: list[float] = []
    corrected_mse: list[float] = []
    nominal_cosine: list[float] = []
    corrected_cosine: list[float] = []
    for begin in range(0, len(indices), batch_size):
        rows = np.asarray(indices[begin : begin + batch_size], dtype=np.int64)
        row_tensor = torch.from_numpy(rows)
        feature = features[row_tensor].to(device=device, dtype=torch.float32)
        nominal = torch.from_numpy(np.asarray(arrays["q_nominal"][rows]).copy()).to(device)
        target = torch.from_numpy(np.asarray(arrays["q_true"][rows]).copy()).to(device)
        predicted = torch.from_numpy(np.asarray(corrected[rows]).copy()).to(device)
        valid = torch.from_numpy(np.asarray(arrays["valid"][rows]).copy()).to(device)
        nominal_tokens = sample_feature_map_at_pixels(
            feature, nominal, image_size=(width, height), padding_mode="border"
        )
        corrected_tokens = sample_feature_map_at_pixels(
            feature, predicted, image_size=(width, height), padding_mode="border"
        )
        target_tokens = sample_feature_map_at_pixels(
            feature, target, image_size=(width, height), padding_mode="border"
        )
        mse_nominal = (nominal_tokens - target_tokens).square().mean(dim=-1)
        mse_corrected = (corrected_tokens - target_tokens).square().mean(dim=-1)
        cosine_nominal = 1.0 - F.cosine_similarity(
            nominal_tokens, target_tokens, dim=-1
        )
        cosine_corrected = 1.0 - F.cosine_similarity(
            corrected_tokens, target_tokens, dim=-1
        )
        nominal_mse.extend(mse_nominal[valid].cpu().tolist())
        corrected_mse.extend(mse_corrected[valid].cpu().tolist())
        nominal_cosine.extend(cosine_nominal[valid].cpu().tolist())
        corrected_cosine.extend(cosine_corrected[valid].cpu().tolist())
    if not nominal_mse:
        raise ValueError("token metric split has no valid anchors")
    nominal_mse_mean = float(np.mean(nominal_mse))
    corrected_mse_mean = float(np.mean(corrected_mse))
    nominal_cosine_mean = float(np.mean(nominal_cosine))
    corrected_cosine_mean = float(np.mean(corrected_cosine))
    return {
        "anchors": len(nominal_mse),
        "nominal_feature_mse": nominal_mse_mean,
        "corrected_feature_mse": corrected_mse_mean,
        "feature_mse_relative_improvement": (
            (nominal_mse_mean - corrected_mse_mean) / nominal_mse_mean
            if nominal_mse_mean > 1.0e-12
            else float("nan")
        ),
        "nominal_cosine_distance": nominal_cosine_mean,
        "corrected_cosine_distance": corrected_cosine_mean,
        "cosine_relative_improvement": (
            (nominal_cosine_mean - corrected_cosine_mean) / nominal_cosine_mean
            if nominal_cosine_mean > 1.0e-12
            else float("nan")
        ),
    }


@torch.inference_mode()
def cache_feature_maps(
    system: torch.nn.Module,
    dataset: SO101SyntheticFCPGDataset,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    rgb = dataset.arrays["rgb"]
    features: list[torch.Tensor] = []
    system.eval()
    for begin in range(0, len(dataset), batch_size):
        end = min(begin + batch_size, len(dataset))
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            if isinstance(system, SO101V10):
                pixels = unit_rgb(np.asarray(rgb[begin:end])).to(device)
                _, patches = system._encode_flat(pixels)
                feature_map = (
                    patches.reshape(-1, 16, 16, system.config.latent_dim)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            else:
                pixels = normalize_rgb(np.asarray(rgb[begin:end])).to(device).unsqueeze(1)
                _, _, feature_map = system._encode_shared_vision(pixels)
        features.append(feature_map.detach().to(device="cpu", dtype=torch.float16))
        print(f"encoded features {end}/{len(dataset)}", flush=True)
    return torch.cat(features, dim=0)


def correct_all(
    system: torch.nn.Module,
    dataset: SO101SyntheticFCPGDataset,
    features: torch.Tensor,
    normalized_state: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    arrays = dataset.arrays
    result = np.empty_like(np.asarray(arrays["q_nominal"]), dtype=np.float32)
    system.corrector.eval()
    with torch.inference_mode():
        for begin in range(0, len(dataset), batch_size):
            end = min(begin + batch_size, len(dataset))
            rows = slice(begin, end)
            feature = features[rows].to(device=device, dtype=torch.float32)
            nominal = torch.from_numpy(np.asarray(arrays["q_nominal"][rows]).copy()).to(device)
            xyz = torch.from_numpy(np.asarray(arrays["anchors_xyz"][rows]).copy()).to(device)
            state = torch.from_numpy(normalized_state[rows]).to(device)
            camera = torch.from_numpy(np.asarray(arrays["camera_context"][rows]).copy()).to(device)
            observation = torch.cat((state, camera), dim=-1)
            output = system.corrector(
                feature,
                nominal,
                xyz,
                observation,
                image_size=(
                    int(dataset.manifest["width"]),
                    int(dataset.manifest["height"]),
                ),
            )
            result[rows] = output.q_corrected_px.detach().cpu().numpy()
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset = SO101SyntheticFCPGDataset(args.dataset)
    split_audit = split_identity_audit(dataset)
    checkpoint, system, checkpoint_sha, source_family = load_source_checkpoint(
        args.checkpoint,
        device,
        corrector_variant=args.corrector_variant,
    )
    if (
        args.expected_checkpoint_sha256 is not None
        and checkpoint_sha != args.expected_checkpoint_sha256
    ):
        raise ValueError(
            "frozen source checkpoint SHA-256 mismatch: "
            f"expected {args.expected_checkpoint_sha256}, got {checkpoint_sha}"
        )
    if isinstance(system, SO101V10) and (
        int(dataset.manifest["width"]) != system.config.image_size
        or int(dataset.manifest["height"]) != system.config.image_size
    ):
        raise ValueError(
            "V12 FCPG data must match the checkpoint image size "
            f"{system.config.image_size}x{system.config.image_size}"
        )
    initial_corrector = None
    if args.initial_corrector_checkpoint is not None:
        initial_corrector = load_corrector_initializer(
            system,
            args.initial_corrector_checkpoint,
            source_checkpoint_sha256=checkpoint_sha,
            source_family=source_family,
            corrector_variant=args.corrector_variant,
        )
    trainable_parameters = configure_corrector_only(system)
    frozen_before = snapshot_frozen_state(system)
    frozen_sha_before = state_dict_sha256(frozen_before)
    corrector_sha_before = state_dict_sha256(system.corrector.state_dict())
    features = cache_feature_maps(
        system, dataset, device, args.feature_batch_size
    )

    states = np.asarray(dataset.arrays["states"], dtype=np.float32)
    if source_family == "v12":
        normalized_state = states / JOINT_SCALE
        state_normalization = "joint_degrees_divided_by_100"
    else:
        contract = checkpoint["run_config"]["contract"]
        center = np.asarray(contract["normalization_center"], dtype=np.float32)
        scale = np.asarray(contract["normalization_scale"], dtype=np.float32)
        normalized_state = (states - center) / scale
        state_normalization = str(contract.get("normalization", "checkpoint contract"))
    train_indices = torch.from_numpy(dataset.indices("train")).long()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    assert_optimizer_scope(system, optimizer)
    width = int(dataset.manifest["width"])
    height = int(dataset.manifest["height"])
    arrays = dataset.arrays
    history_rows: list[dict[str, float | int]] = []
    system.eval()
    system.corrector.train()
    for step in range(1, args.steps + 1):
        sampled = train_indices[
            torch.randint(
                len(train_indices),
                (args.batch_size,),
                generator=generator,
            )
        ]
        rows = sampled.numpy()
        feature = features[sampled].to(device=device, dtype=torch.float32)
        nominal = torch.from_numpy(np.asarray(arrays["q_nominal"][rows]).copy()).to(device)
        target = torch.from_numpy(np.asarray(arrays["q_true"][rows]).copy()).to(device)
        xyz = torch.from_numpy(np.asarray(arrays["anchors_xyz"][rows]).copy()).to(device)
        valid = torch.from_numpy(np.asarray(arrays["valid"][rows]).copy()).to(device)
        state = torch.from_numpy(normalized_state[rows]).to(device)
        camera = torch.from_numpy(np.asarray(arrays["camera_context"][rows]).copy()).to(device)
        observation = torch.cat((state, camera), dim=-1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            prediction = system.corrector(
                feature,
                nominal,
                xyz,
                observation,
                image_size=(width, height),
            )
            loss = masked_smooth_l1(
                prediction.delta_px,
                target - nominal,
                valid,
            )
        loss.backward()
        assert_frozen_gradients_absent(system)
        gradient_norm = assert_corrector_gradients(system)
        torch.nn.utils.clip_grad_norm_(system.corrector.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "corrector_gradient_norm": gradient_norm,
            }
            history_rows.append(row)
            print(json.dumps(row), flush=True)

    frozen_sha_after = assert_frozen_state_unchanged(system, frozen_before)
    corrector_sha_after = state_dict_sha256(system.corrector.state_dict())
    if corrector_sha_after == corrector_sha_before:
        raise RuntimeError("corrector state did not change during training")

    corrected = correct_all(
        system,
        dataset,
        features,
        normalized_state,
        device,
        args.feature_batch_size,
    )
    np.save(output / "corrected_q.npy", corrected)
    metrics = {
        "train": aggregate_metrics(dataset, dataset.indices("train"), corrected),
        "val": aggregate_metrics(dataset, dataset.indices("val"), corrected),
    }
    token_metrics = {
        "train": token_sampling_metrics(
            dataset,
            dataset.indices("train"),
            features,
            corrected,
            device,
            args.feature_batch_size,
        ),
        "val": token_sampling_metrics(
            dataset,
            dataset.indices("val"),
            features,
            corrected,
            device,
            args.feature_batch_size,
        ),
    }
    dataset_manifest_path = dataset.root / "manifest.json"
    dataset_manifest_sha = sha256_file(dataset_manifest_path)
    dataset_scope = str(
        dataset.manifest.get("scope", "synthetic camera degradation supervision")
    )
    vit_source = checkpoint.get("vit_source") if source_family == "v12" else None
    vit_source_sha = sha256_file(Path(vit_source)) if isinstance(vit_source, str) else None
    freeze_contract = {
        "trainable_prefixes": [TRAINABLE_STATE_PREFIX],
        "frozen_prefixes": sorted(
            {name.split(".", 1)[0] for name in frozen_before}
        ),
        "frozen_state_sha256_before": frozen_sha_before,
        "frozen_state_sha256_after": frozen_sha_after,
        "bitwise_unchanged": frozen_sha_before == frozen_sha_after,
        "corrector_state_sha256_before": corrector_sha_before,
        "corrector_state_sha256_after": corrector_sha_after,
        "corrector_changed": corrector_sha_before != corrector_sha_after,
    }
    payload = {
        "format": (
            "so101-v12-fcpg-only-synthetic-v1"
            if source_family == "v12"
            else "so101-synthetic-fcpg-corrector-v1"
        ),
        "step": args.steps,
        "seed": args.seed,
        "dataset": str(dataset.root),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_checkpoint_sha256_verified": (
            args.expected_checkpoint_sha256 == checkpoint_sha
        ),
        "source_family": source_family,
        "source_local_format": checkpoint.get("local_format"),
        "source_step": checkpoint.get("step"),
        "vit_source": vit_source,
        "vit_source_sha256": vit_source_sha,
        "training_scope": SYNTHETIC_TRAINING_SCOPE,
        "has_physical_labels": False,
        "claim_boundary": f"{dataset_scope}; no physical grounding labels",
        "dataset_scope": dataset_scope,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "dataset_generation_source_checkpoint": dataset.manifest.get("source_checkpoint"),
        "dataset_generation_source_checkpoint_sha256": dataset.manifest.get(
            "source_checkpoint_sha256"
        ),
        "state_normalization": state_normalization,
        "split_identity_audit": split_audit,
        "freeze_contract": freeze_contract,
        "camera_context_dim": 5,
        "corrector_variant": args.corrector_variant,
        "initial_corrector": initial_corrector,
        "corrector": {
            "parameters": sum(parameter.numel() for parameter in system.corrector.parameters()),
            "local_radius_px": system.corrector.local_radius_px,
            "max_residual_px": system.corrector.max_residual_px,
        },
        "scope": f"{dataset_scope}; no physical-grounding evidence",
        "metrics": metrics,
        "token_sampling_metrics": token_metrics,
        "history": history_rows,
    }
    torch.save(
        {
            "corrector": system.corrector.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "token_sampling_metrics": token_metrics,
            "source_checkpoint_sha256": checkpoint_sha,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_checkpoint_sha256_verified": (
                args.expected_checkpoint_sha256 == checkpoint_sha
            ),
            "corrector_variant": args.corrector_variant,
            "initial_corrector": initial_corrector,
            "source_family": source_family,
            "source_local_format": checkpoint.get("local_format"),
            "source_step": checkpoint.get("step"),
            "vit_source": vit_source,
            "vit_source_sha256": vit_source_sha,
            "training_scope": SYNTHETIC_TRAINING_SCOPE,
            "has_physical_labels": False,
            "dataset_scope": dataset_scope,
            "frozen_state_sha256": frozen_sha_after,
            "freeze_contract": freeze_contract,
            "corrector_state_sha256_before": corrector_sha_before,
            "corrector_state_sha256_after": corrector_sha_after,
            "dataset_manifest_sha256": dataset_manifest_sha,
        },
        output / "latest.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(metrics["val"], indent=2), flush=True)
    print(json.dumps({"val_token_sampling": token_metrics["val"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
