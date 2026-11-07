"""Immutable run receipts for the pro6000D -> V100 Git/artifact handoff."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from fcpg.offline_pipeline_manifest import PROTOCOL_ID, sha256_file, sha256_json


RECEIPT_FORMAT = "so101-fcpg-lewm-run-receipt-v1"
ARTIFACT_FORMAT = "so101-fcpg-lewm-artifact-manifest-v1"


def git_state(root: Path, *, require_clean: bool) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ("git", *args), cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()

    head = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain")
    if require_clean and status:
        raise RuntimeError("formal/canary runs require a clean Git worktree")
    return {"head": head, "branch": branch, "clean": not bool(status), "status": status}


def parameter_inventory(module: nn.Module) -> dict[str, object]:
    frozen = []
    trainable = []
    frozen_count = 0
    trainable_count = 0
    for name, parameter in module.named_parameters():
        row = {"name": name, "shape": list(parameter.shape), "count": parameter.numel()}
        if parameter.requires_grad:
            trainable.append(row)
            trainable_count += parameter.numel()
        else:
            frozen.append(row)
            frozen_count += parameter.numel()
    return {
        "trainable_count": trainable_count,
        "frozen_count": frozen_count,
        "trainable": trainable,
        "frozen": frozen,
    }


def runtime_identity(device: torch.device) -> dict[str, object]:
    result: dict[str, object] = {
        "hostname": platform.node(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["cuda_runtime"] = torch.version.cuda
    return result


def write_receipt(output: Path, payload: Mapping[str, object]) -> dict[str, object]:
    receipt = {
        "format": RECEIPT_FORMAT,
        "protocol_id": PROTOCOL_ID,
        **dict(payload),
    }
    receipt["content_sha256"] = sha256_json(receipt)
    path = output / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def write_artifact_manifest(
    output: Path,
    *,
    git_head: str,
    v12_checkpoint_sha256: str,
    data_manifest_sha256: str,
    artifact_paths: list[Path],
) -> dict[str, object]:
    artifacts = []
    for path in artifact_paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        artifacts.append(
            {
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    manifest = {
        "format": ARTIFACT_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "git_head": git_head,
        "v12_checkpoint_sha256": v12_checkpoint_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "artifacts": artifacts,
    }
    manifest["content_sha256"] = sha256_json(manifest)
    (output / "artifacts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
