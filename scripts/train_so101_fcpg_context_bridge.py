#!/usr/bin/env python3
"""Train an anchor bridge from FCPG-corrected evidence to clean LeWM context."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fcpg.so101_fcpg_lewm_bridge import (
    SO101FCPGConditionedLeWM,
    build_so101_fcpg_corrector,
    infer_so101_fcpg_corrector_variant,
)
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from fcpg.so101_synthetic_dataset import (
    SO101SyntheticFCPGDataset,
    clean_partner_indices,
)
from scripts.train_so101_synthetic_fcpg import normalize_rgb
from scripts.visualize_so101_lewm import build_checkpoint_model, checkpoint_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corrector-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=6123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.feature_batch_size, args.log_every) <= 0:
        parser.error("step and batch arguments must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("optimizer arguments are invalid")
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    return args


def matching_visual_backbone(
    target_checkpoint: dict[str, Any], source_path: Path
) -> bool:
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    target_state = target_checkpoint["model"]
    source_state = source["model"]
    keys = [
        key
        for key in target_state
        if key.startswith("base.encoder.") or key.startswith("base.projector.")
    ]
    return bool(keys) and all(
        key in source_state and torch.equal(target_state[key], source_state[key])
        for key in keys
    )


@torch.inference_mode()
def cache_vision(
    system: SO101FCPGConditionedLeWM,
    dataset: SO101SyntheticFCPGDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    contexts: list[torch.Tensor] = []
    patches: list[torch.Tensor] = []
    system.eval()
    rgb = dataset.arrays["rgb"]
    for begin in range(0, len(dataset), batch_size):
        end = min(begin + batch_size, len(dataset))
        pixels = normalize_rgb(np.asarray(rgb[begin:end])).to(device).unsqueeze(1)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            context, patch, _ = system._encode_shared_vision(pixels)
        contexts.append(context[:, 0].to(device="cpu", dtype=torch.float16))
        patches.append(patch[:, 0].to(device="cpu", dtype=torch.float16))
        print(f"encoded bridge features {end}/{len(dataset)}", flush=True)
    return torch.cat(contexts), torch.cat(patches)


def patch_feature_map(patches: torch.Tensor, latent_dim: int) -> torch.Tensor:
    grid = int(round(float(patches.shape[1]) ** 0.5))
    if grid * grid != patches.shape[1] or patches.shape[2] != latent_dim:
        raise ValueError("patch tensor does not form the expected feature map")
    return patches.reshape(len(patches), grid, grid, latent_dim).permute(0, 3, 1, 2)


@torch.inference_mode()
def correct_points(
    system: SO101FCPGConditionedLeWM,
    dataset: SO101SyntheticFCPGDataset,
    patches: torch.Tensor,
    normalized_state: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    arrays = dataset.arrays
    result = np.empty_like(np.asarray(arrays["q_nominal"]), dtype=np.float32)
    system.corrector.eval()
    for begin in range(0, len(dataset), batch_size):
        end = min(begin + batch_size, len(dataset))
        rows = slice(begin, end)
        patch = patches[rows].to(device=device, dtype=torch.float32)
        nominal = torch.from_numpy(
            np.asarray(arrays["q_nominal"][rows]).copy()
        ).to(device)
        xyz = torch.from_numpy(np.asarray(arrays["anchors_xyz"][rows]).copy()).to(device)
        state = torch.from_numpy(normalized_state[rows]).to(device)
        camera = torch.from_numpy(
            np.asarray(arrays["camera_context"][rows]).copy()
        ).to(device)
        output = system.corrector(
            patch_feature_map(patch, system.model.latent_dim),
            nominal,
            xyz,
            torch.cat((state, camera), dim=-1),
            image_size=(
                int(dataset.manifest["width"]),
                int(dataset.manifest["height"]),
            ),
        )
        result[rows] = output.q_corrected_px.cpu().numpy()
    return result


def aggregate_context_metrics(
    dataset: SO101SyntheticFCPGDataset,
    indices: np.ndarray,
    base_context: np.ndarray,
    bridged_context: np.ndarray,
    clean_partners: np.ndarray,
) -> dict[str, Any]:
    arrays = dataset.arrays
    views = dataset.manifest["camera"]["views"]
    buckets: dict[str, list[int]] = defaultdict(list)
    for raw_index in indices:
        index = int(raw_index)
        view = str(views[int(arrays["camera_index"][index])]["name"])
        clean = int(arrays["drift_variant"][index]) == 0
        buckets["all"].append(index)
        buckets["clean" if clean else "drift"].append(index)
        buckets[f"{view}/all"].append(index)
        buckets[f"{view}/{'clean' if clean else 'drift'}"].append(index)
    result: dict[str, Any] = {}
    for name, rows in sorted(buckets.items()):
        row_array = np.asarray(rows, dtype=np.int64)
        target = base_context[clean_partners[row_array]]
        base_mse = np.mean(np.square(base_context[row_array] - target), axis=1)
        bridge_mse = np.mean(np.square(bridged_context[row_array] - target), axis=1)
        base_mean = float(base_mse.mean())
        bridge_mean = float(bridge_mse.mean())
        result[name] = {
            "frames": len(rows),
            "base_context_mse": base_mean,
            "bridged_context_mse": bridge_mean,
            "relative_improvement": (
                (base_mean - bridge_mean) / base_mean if base_mean > 1.0e-12 else None
            ),
            "bridged_p95_mse": float(np.quantile(bridge_mse, 0.95)),
        }
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
    clean_partners = clean_partner_indices(dataset.arrays)
    checkpoint, _, _, frameskip, history, checkpoint_sha = checkpoint_contract(
        args.checkpoint.resolve(), None
    )
    model = build_checkpoint_model(checkpoint, frameskip, history, device)
    if not isinstance(model, StateResidualLeWM):
        raise TypeError("context bridge requires a state-residual LeWM checkpoint")
    corrector_payload = torch.load(
        args.corrector_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    corrector_variant = infer_so101_fcpg_corrector_variant(corrector_payload)
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
    source_checkpoint = Path(corrector_payload["args"]["checkpoint"]).resolve()
    if not matching_visual_backbone(checkpoint, source_checkpoint):
        raise ValueError("corrector and target LeWM use different visual backbones")
    system.corrector.load_state_dict(corrector_payload["corrector"], strict=True)
    system.corrector.requires_grad_(False)
    system.bridge.requires_grad_(True)

    contexts, patches = cache_vision(
        system, dataset, device, args.feature_batch_size
    )
    contract = checkpoint["run_config"]["contract"]
    center = np.asarray(contract["normalization_center"], dtype=np.float32)
    scale = np.asarray(contract["normalization_scale"], dtype=np.float32)
    states = np.asarray(dataset.arrays["states"], dtype=np.float32)
    normalized_state = (states - center) / scale
    corrected = correct_points(
        system,
        dataset,
        patches,
        normalized_state,
        device,
        args.feature_batch_size,
    )

    arrays = dataset.arrays
    width = int(dataset.manifest["width"])
    height = int(dataset.manifest["height"])
    train_indices = torch.from_numpy(dataset.indices("train")).long()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    optimizer = torch.optim.AdamW(
        system.bridge.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history_rows: list[dict[str, float | int]] = []
    system.bridge.train()
    for step in range(1, args.steps + 1):
        sampled = train_indices[
            torch.randint(
                len(train_indices), (args.batch_size,), generator=generator
            )
        ]
        rows = sampled.numpy()
        partners = clean_partners[rows]
        context = contexts[sampled].to(device=device, dtype=torch.float32).unsqueeze(1)
        patch = patches[sampled].to(device=device, dtype=torch.float32).unsqueeze(1)
        nominal = torch.from_numpy(
            np.asarray(arrays["q_nominal"][rows]).copy()
        ).to(device).unsqueeze(1)
        corrected_batch = torch.from_numpy(corrected[rows].copy()).to(device).unsqueeze(1)
        xyz = torch.from_numpy(np.asarray(arrays["anchors_xyz"][rows]).copy()).to(device).unsqueeze(1)
        valid = torch.from_numpy(np.asarray(arrays["valid"][rows]).copy()).to(device).unsqueeze(1)
        target = contexts[torch.from_numpy(partners)].to(
            device=device, dtype=torch.float32
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output_bridge = system.bridge(
                context,
                patch,
                nominal,
                corrected_batch,
                xyz,
                valid,
                image_size=(width, height),
            )
            loss = F.mse_loss(output_bridge.context_embedding[:, 0], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.bridge.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "mean_abs_gate": float(
                    torch.tanh(system.bridge.gate).abs().mean().detach().cpu()
                ),
            }
            history_rows.append(row)
            print(json.dumps(row), flush=True)

    system.bridge.eval()
    bridged = np.empty((len(dataset), model.latent_dim), dtype=np.float32)
    with torch.inference_mode():
        for begin in range(0, len(dataset), args.feature_batch_size):
            end = min(begin + args.feature_batch_size, len(dataset))
            rows = slice(begin, end)
            output_bridge = system.bridge(
                contexts[rows].to(device=device, dtype=torch.float32).unsqueeze(1),
                patches[rows].to(device=device, dtype=torch.float32).unsqueeze(1),
                torch.from_numpy(np.asarray(arrays["q_nominal"][rows]).copy()).to(device).unsqueeze(1),
                torch.from_numpy(corrected[rows].copy()).to(device).unsqueeze(1),
                torch.from_numpy(np.asarray(arrays["anchors_xyz"][rows]).copy()).to(device).unsqueeze(1),
                torch.from_numpy(np.asarray(arrays["valid"][rows]).copy()).to(device).unsqueeze(1),
                image_size=(width, height),
            )
            bridged[rows] = output_bridge.context_embedding[:, 0].cpu().numpy()
    base_context = contexts.float().numpy()
    metrics = {
        "train": aggregate_context_metrics(
            dataset, dataset.indices("train"), base_context, bridged, clean_partners
        ),
        "val": aggregate_context_metrics(
            dataset, dataset.indices("val"), base_context, bridged, clean_partners
        ),
    }
    payload = {
        "format": "so101-fcpg-context-bridge-v1",
        "step": args.steps,
        "seed": args.seed,
        "dataset": str(dataset.root),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_sha,
        "corrector_checkpoint": str(args.corrector_checkpoint.resolve()),
        "corrector_variant": corrector_variant,
        "parameters": sum(parameter.numel() for parameter in system.bridge.parameters()),
        "scope": "real appearance plus synthetic drift development; no robot commands",
        "metrics": metrics,
        "history": history_rows,
    }
    torch.save(
        {
            "bridge": system.bridge.state_dict(),
            "corrector": system.corrector.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "source_checkpoint_sha256": checkpoint_sha,
            "corrector_variant": corrector_variant,
        },
        output / "latest.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics["val"], indent=2), flush=True)


if __name__ == "__main__":
    main()
