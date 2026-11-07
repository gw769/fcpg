#!/usr/bin/env python3
"""Train one action-conditioned LeWM across cached SO-101 datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from stable_pretraining.backbone.utils import vit_hf


SIGREG_WEIGHT = 0.02
JOINT_SCALE = 100.0
DEFAULT_MOTION_THRESHOLD = 0.01
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def resolve_cache_source_root(cache_root: Path, raw_source: str | Path) -> Path:
    embedded = Path(raw_source).expanduser()
    if embedded.is_absolute():
        original = embedded.resolve()
    else:
        original = (cache_root / embedded).resolve()
    if original.is_dir():
        return original

    # Standard relocated layout:
    # data/lerobot/so101/cache/<cache> -> data/lerobot/so101/raw/<dataset>
    relocated = (cache_root.parent.parent / "raw" / embedded.name).resolve()
    if relocated.is_dir():
        return relocated
    return original


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=5120)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=32)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=7301)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--joint-scale", type=float, default=JOINT_SCALE)
    parser.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD)
    parser.add_argument(
        "--sampling",
        choices=("source-task-motion-balanced", "source-task-natural"),
        default="source-task-motion-balanced",
    )
    parser.add_argument(
        "--action-normalization",
        choices=("physical-scale", "global-train-zscore"),
        default="physical-scale",
    )
    parser.add_argument(
        "--exclude-episode",
        action="append",
        default=[],
        metavar="DATASET:EPISODE",
        help="Exclude a known-bad physical episode; may be repeated.",
    )
    parser.add_argument(
        "--init",
        type=Path,
        help="Optional matching LeWM checkpoint. Omit to train SO-101 from scratch.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


class Cache:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != "so101-lewm-cache-v1":
            raise ValueError(f"unsupported cache format: {self.root}")
        self.frames = np.load(self.root / "frames.npy", mmap_mode="r")
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r")
        self.states = np.load(self.root / "states.npy", mmap_mode="r")
        self.episodes = np.load(self.root / "episodes.npy", mmap_mode="r")
        self.tasks = np.load(self.root / "tasks.npy", mmap_mode="r")
        self.frame_indices = np.load(self.root / "frame_indices.npy", mmap_mode="r")
        expected = int(self.manifest["frames"])
        if any(len(x) != expected for x in (self.frames, self.actions, self.states, self.episodes, self.tasks)):
            raise ValueError(f"cache arrays disagree with manifest: {self.root}")
        self.action_mean = np.asarray(self.manifest["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(self.manifest["action_std"], dtype=np.float32)
        self.name = str(self.manifest["name"])
        self.source_root = resolve_cache_source_root(
            self.root,
            self.manifest.get("source_root", self.root),
        )
        self.dataset_id = str(self.manifest.get("dataset_id", self.source_root.name))
        self.robot_type = str(self.manifest.get("robot_type", "unknown"))
        self.camera = str(self.manifest.get("camera", "unknown"))
        self.task_names = self._load_task_names()

    def _load_task_names(self) -> dict[int, str]:
        embedded = self.manifest.get("task_names")
        if embedded:
            return {int(key): str(value) for key, value in embedded.items()}
        task_path = self.source_root / "meta" / "tasks.jsonl"
        if not task_path.is_file():
            return {}
        rows = [json.loads(line) for line in task_path.read_text().splitlines() if line.strip()]
        return {int(row["task_index"]): str(row["task"]) for row in rows}

    def task_key(self, task: int) -> str:
        text = self.task_names.get(int(task), f"task_{int(task)}").lower()
        # The public fruit dataset contains both "a orange" and "an orange".
        text = re.sub(r"\b(?:a|an|the)\b", " ", text)
        return " ".join(text.split())


def parse_exclusions(values: Iterable[str]) -> set[tuple[str, int]]:
    exclusions: set[tuple[str, int]] = set()
    for value in values:
        try:
            source, episode = value.rsplit(":", 1)
            exclusions.add((source, int(episode)))
        except ValueError as error:
            raise ValueError(f"invalid --exclude-episode {value!r}; expected DATASET:EPISODE") from error
    return exclusions


def _is_excluded(cache: Cache, episode: int, exclusions: set[tuple[str, int]]) -> bool:
    return (cache.dataset_id, episode) in exclusions or (cache.name, episode) in exclusions


def episode_splits(
    caches: list[Cache],
    exclusions: set[tuple[str, int]],
    split_seed: int,
    val_fraction: float,
) -> dict[tuple[str, int], str]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    grouped: dict[tuple[str, str], set[int]] = defaultdict(set)
    for cache in caches:
        episodes = np.asarray(cache.episodes)
        tasks = np.asarray(cache.tasks)
        for episode in np.unique(episodes):
            episode = int(episode)
            if _is_excluded(cache, episode, exclusions):
                continue
            rows = np.flatnonzero(episodes == episode)
            task_ids = np.unique(tasks[rows])
            if len(task_ids) != 1:
                raise ValueError(f"episode spans multiple tasks: {cache.name}:{episode}")
            grouped[(cache.dataset_id, cache.task_key(int(task_ids[0])))].add(episode)

    assignments: dict[tuple[str, int], str] = {}
    for (dataset_id, task_key), episode_ids in sorted(grouped.items()):
        ordered = sorted(
            episode_ids,
            key=lambda episode: hashlib.sha256(
                f"{split_seed}:{dataset_id}:{task_key}:{episode}".encode()
            ).digest(),
        )
        if len(ordered) < 2:
            raise ValueError(f"task has fewer than two episodes: {dataset_id}/{task_key}")
        val_count = max(1, int(round(len(ordered) * val_fraction)))
        val_count = min(val_count, len(ordered) - 1)
        validation = set(ordered[:val_count])
        for episode in ordered:
            key = (dataset_id, episode)
            split = "val" if episode in validation else "train"
            previous = assignments.setdefault(key, split)
            if previous != split:
                raise ValueError(f"multi-camera split disagreement for {dataset_id}:{episode}")
    return assignments


def global_train_action_stats(
    caches: list[Cache],
    assignments: dict[tuple[str, int], str],
    exclusions: set[tuple[str, int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit one deployable six-joint normalizer on training episodes only."""
    count = 0
    total = np.zeros(6, dtype=np.float64)
    total_square = np.zeros(6, dtype=np.float64)
    seen_datasets: set[str] = set()
    for cache in caches:
        # Multiple camera caches for one physical dataset share actions. Count
        # them once so adding a view cannot change the action normalizer.
        if cache.dataset_id in seen_datasets:
            continue
        seen_datasets.add(cache.dataset_id)
        episodes = np.asarray(cache.episodes)
        for raw_episode in np.unique(episodes):
            episode = int(raw_episode)
            if _is_excluded(cache, episode, exclusions):
                continue
            if assignments.get((cache.dataset_id, episode)) != "train":
                continue
            rows = np.flatnonzero(episodes == episode)
            actions = np.asarray(cache.actions[rows], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] != 6:
                raise ValueError(f"invalid actions in {cache.name}:{episode}")
            if not np.all(np.isfinite(actions)):
                raise ValueError(f"non-finite actions in {cache.name}:{episode}")
            count += len(actions)
            total += actions.sum(axis=0)
            total_square += np.square(actions).sum(axis=0)
    if count < 2:
        raise ValueError("not enough training actions to fit global normalization")
    center = total / count
    variance = np.maximum(total_square / count - np.square(center), 1e-6)
    scale = np.sqrt(variance)
    return center.astype(np.float32), scale.astype(np.float32), count


class FixedIndexSampler(Sampler[int]):
    def __init__(self, indices: torch.Tensor):
        self.indices = indices.to(dtype=torch.int64, device="cpu")

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self) -> int:
        return len(self.indices)


class SO101Sequences(Dataset):
    def __init__(
        self,
        cache_roots: list[Path],
        split: str,
        frameskip: int = 5,
        history: int = 3,
        *,
        exclusions: set[tuple[str, int]],
        split_seed: int,
        val_fraction: float,
        joint_scale: float,
        motion_threshold: float,
        action_center: np.ndarray | None = None,
        action_scale: np.ndarray | None = None,
        episode_assignments: dict[tuple[str, int], str] | None = None,
        memory_offsets: tuple[int, ...] = (),
    ):
        if split not in {"train", "val"}:
            raise ValueError(split)
        self.caches = [Cache(root) for root in cache_roots]
        self.frameskip = frameskip
        self.history = history
        self.sequence_frames = history + 1
        self.span = frameskip * history
        self.joint_scale = float(joint_scale)
        self.motion_threshold = float(motion_threshold)
        self.split_seed = int(split_seed)
        self.memory_offsets = tuple(int(value) for value in memory_offsets)
        if any(
            value < self.frameskip for value in self.memory_offsets
        ) or tuple(sorted(set(self.memory_offsets))) != self.memory_offsets:
            raise ValueError(
                "memory_offsets must be unique, increasing, and at least one frameskip"
            )
        self.action_center = np.zeros(6, dtype=np.float32) if action_center is None else np.asarray(action_center, dtype=np.float32)
        self.action_scale = (
            np.full(6, self.joint_scale, dtype=np.float32)
            if action_scale is None
            else np.asarray(action_scale, dtype=np.float32)
        )
        if self.joint_scale <= 0:
            raise ValueError("--joint-scale must be positive")
        if self.action_center.shape != (6,) or self.action_scale.shape != (6,):
            raise ValueError("action normalization vectors must have shape (6,)")
        if not np.all(np.isfinite(self.action_center)) or not np.all(np.isfinite(self.action_scale)):
            raise ValueError("action normalization vectors must be finite")
        if np.any(self.action_scale <= 0):
            raise ValueError("action normalization scale must be positive")
        self.samples: list[tuple[int, int]] = []
        self.group_keys: list[tuple[str, str]] = []
        self.moving: list[bool] = []
        assignments = (
            episode_splits(
                self.caches,
                exclusions,
                split_seed=split_seed,
                val_fraction=val_fraction,
            )
            if episode_assignments is None
            else dict(episode_assignments)
        )
        if any(value not in {"train", "val"} for value in assignments.values()):
            raise ValueError("episode assignments must contain only 'train' or 'val'")

        for cache_id, cache in enumerate(self.caches):
            source = str(cache.manifest["name"])
            episodes = np.asarray(cache.episodes)
            frame_indices = np.asarray(cache.frame_indices)
            tasks = np.asarray(cache.tasks)
            for episode in np.unique(episodes):
                episode = int(episode)
                if _is_excluded(cache, episode, exclusions):
                    continue
                if (cache.dataset_id, episode) not in assignments:
                    raise ValueError(
                        f"missing episode assignment for {cache.dataset_id}:{episode}"
                    )
                is_val = assignments[(cache.dataset_id, episode)] == "val"
                if (split == "val") != is_val:
                    continue
                rows = np.flatnonzero(episodes == episode)
                if len(rows) <= self.span:
                    continue
                if not np.all(np.diff(rows) == 1):
                    raise ValueError(f"non-contiguous episode {source}:{episode}")
                local_frames = frame_indices[rows]
                for offset in range(0, len(rows) - self.span):
                    if self.memory_offsets and offset < self.memory_offsets[-1]:
                        continue
                    selected = local_frames[offset : offset + self.span + 1]
                    if not np.all(np.diff(selected) == 1):
                        continue
                    start = int(rows[offset])
                    self.samples.append((cache_id, start))
                    self.group_keys.append((cache.dataset_id, cache.task_key(int(tasks[start]))))
                    delta = (
                        np.asarray(cache.states[start + self.span], dtype=np.float32)
                        - np.asarray(cache.states[start], dtype=np.float32)
                    ) / self.joint_scale
                    self.moving.append(float(np.sqrt(np.mean(delta * delta))) >= self.motion_threshold)
        if not self.samples:
            raise ValueError(f"no {split} sequences found")
        unique_groups = sorted(set(self.group_keys))
        self.group_to_id = {group: index for index, group in enumerate(unique_groups)}
        self.group_ids = [self.group_to_id[group] for group in self.group_keys]
        self.group_labels = {
            index: f"{dataset}/{task}" for (dataset, task), index in self.group_to_id.items()
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        cache_id, start = self.samples[index]
        cache = self.caches[cache_id]
        frame_rows = start + np.arange(self.sequence_frames) * self.frameskip
        frames_np = np.asarray(cache.frames[frame_rows]).copy()
        frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().div_(255.0)
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        action_chunks = []
        for step in range(self.history):
            begin = start + step * self.frameskip
            raw = np.asarray(cache.actions[begin : begin + self.frameskip], dtype=np.float32)
            raw = (raw - self.action_center) / self.action_scale
            action_chunks.append(raw.reshape(-1))
        state = np.asarray(cache.states[frame_rows], dtype=np.float32).copy()
        sample = {
            "pixels": frames,
            "action": torch.from_numpy(np.stack(action_chunks)),
            "state": torch.from_numpy(state),
            "source": torch.tensor(cache_id, dtype=torch.long),
            "task": torch.tensor(int(cache.tasks[start]), dtype=torch.long),
            "episode": torch.tensor(int(cache.episodes[start]), dtype=torch.long),
            "group": torch.tensor(self.group_ids[index], dtype=torch.long),
            "moving": torch.tensor(self.moving[index], dtype=torch.bool),
        }
        if self.memory_offsets:
            memory_rows = start - np.asarray(self.memory_offsets, dtype=np.int64)
            expected_episode = int(cache.episodes[start])
            if not np.all(np.asarray(cache.episodes[memory_rows]) == expected_episode):
                raise AssertionError("multi-scale memory crossed an episode boundary")
            memory_frames_np = np.asarray(cache.frames[memory_rows]).copy()
            memory_frames = (
                torch.from_numpy(memory_frames_np).permute(0, 3, 1, 2).float().div_(255.0)
            )
            memory_frames = (memory_frames - IMAGENET_MEAN) / IMAGENET_STD
            memory_actions = []
            for row in memory_rows:
                raw = np.asarray(
                    cache.actions[row : row + self.frameskip], dtype=np.float32
                )
                if raw.shape != (self.frameskip, 6):
                    raise AssertionError("multi-scale memory action chunk is incomplete")
                raw = (raw - self.action_center) / self.action_scale
                memory_actions.append(raw.reshape(-1))
            sample.update(
                {
                    "memory_pixels": memory_frames,
                    "memory_action": torch.from_numpy(np.stack(memory_actions)),
                    "memory_state": torch.from_numpy(
                        np.asarray(cache.states[memory_rows], dtype=np.float32).copy()
                    ),
                    "memory_age": torch.as_tensor(
                        np.asarray(self.memory_offsets, dtype=np.float32)
                        / float(self.frameskip)
                    ),
                    "memory_valid": torch.ones(
                        len(self.memory_offsets), dtype=torch.bool
                    ),
                }
            )
        return sample

    def balanced_weights(self, sampling: str) -> torch.Tensor:
        if sampling == "source-task-natural":
            group_counts = Counter(self.group_keys)
            tasks_per_source = Counter(dataset for dataset, _ in set(self.group_keys))
            return torch.tensor(
                [
                    1.0 / (tasks_per_source[group[0]] * group_counts[group])
                    for group in self.group_keys
                ],
                dtype=torch.double,
            )
        if sampling != "source-task-motion-balanced":
            raise ValueError(f"unsupported sampling mode {sampling!r}")
        bin_counts = Counter(zip(self.group_keys, self.moving))
        bins_per_group = Counter(group for group, _ in bin_counts)
        tasks_per_source = Counter(dataset for dataset, _ in set(self.group_keys))
        weights = []
        for group, moving in zip(self.group_keys, self.moving, strict=True):
            dataset = group[0]
            weight = 1.0 / (
                tasks_per_source[dataset]
                * bins_per_group[group]
                * bin_counts[(group, moving)]
            )
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.double)

    def balanced_eval_indices(self, limit: int | None) -> list[int]:
        by_group: dict[int, list[int]] = defaultdict(list)
        for index, group in enumerate(self.group_ids):
            by_group[group].append(index)
        for group, indices in by_group.items():
            indices.sort(
                key=lambda index: hashlib.sha256(
                    (
                        f"eval:{self.split_seed}:{group}:"
                        f"{self.samples[index][0]}:{int(self.caches[self.samples[index][0]].episodes[self.samples[index][1]])}:"
                        f"{int(self.caches[self.samples[index][0]].frame_indices[self.samples[index][1]])}"
                    ).encode()
                ).digest()
            )
        ordered: list[int] = []
        cursor = 0
        while True:
            added = False
            for group in sorted(by_group):
                rows = by_group[group]
                if cursor < len(rows):
                    ordered.append(rows[cursor])
                    added = True
                    if limit is not None and len(ordered) >= limit:
                        return ordered
            if not added:
                return ordered
            cursor += 1


def build_model(action_dim: int, history: int) -> JEPA:
    return JEPA(
        encoder=vit_hf(
            size="tiny",
            patch_size=14,
            image_size=224,
            pretrained=False,
            use_mask_token=False,
        ),
        predictor=ARPredictor(
            num_frames=history,
            depth=6,
            heads=16,
            mlp_dim=2048,
            input_dim=192,
            hidden_dim=192,
            output_dim=192,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.0,
        ),
        action_encoder=Embedder(input_dim=action_dim, emb_dim=192),
        projector=MLP(
            input_dim=192,
            hidden_dim=2048,
            output_dim=192,
            norm_fn=nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=192,
            hidden_dim=2048,
            output_dim=192,
            norm_fn=nn.BatchNorm1d,
        ),
    )


def initialize_from_checkpoint(model: nn.Module, path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    current = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
    incompatible = sorted(key for key, value in state.items() if key in current and current[key].shape != value.shape)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "path": str(path.resolve()),
        "loaded_tensors": len(compatible),
        "shape_mismatch": incompatible,
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
    }


def prediction_tensors(model: JEPA, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = model.encode({"pixels": batch["pixels"], "action": batch["action"]})
    embedding = encoded["emb"]
    target = embedding[:, 1:]
    prediction = model.predict(embedding[:, :-1], encoded["act_emb"])
    return embedding, prediction, target


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    result = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    result["action_shuffle_delta"] = result["shuffled_action_mse"] - result["prediction_mse"]
    result["no_action_delta"] = result["no_action_mse"] - result["prediction_mse"]
    result["copy_improvement"] = result["copy_mse"] - result["prediction_mse"]
    return result


def _shuffle_actions_within_group(
    actions: torch.Tensor, groups: torch.Tensor
) -> torch.Tensor:
    shuffled = actions.clone()
    group_values = groups.detach().cpu().numpy()
    for group in np.unique(group_values):
        rows = np.flatnonzero(group_values == group)
        if len(rows) > 1:
            source = np.roll(rows, max(1, len(rows) // 2))
            shuffled[torch.as_tensor(rows, device=actions.device)] = actions[
                torch.as_tensor(source, device=actions.device)
            ]
        else:
            shuffled[rows[0]] = actions[rows[0]].flip(0)
    return shuffled


@torch.no_grad()
def evaluate(
    model: JEPA,
    loader: DataLoader,
    device: torch.device,
    group_labels: dict[int, str],
) -> dict:
    model.eval()
    per_group: dict[int, list[dict[str, float]]] = defaultdict(list)
    per_motion: dict[bool, list[dict[str, float]]] = defaultdict(list)
    all_rows: list[dict[str, float]] = []
    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        actions = batch["action"].to(device, non_blocking=True)
        groups = batch["group"]
        moving = batch["moving"]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            encoded = model.encode({"pixels": pixels, "action": actions})
            embedding = encoded["emb"]
            target = embedding[:, 1:]
            factual = model.predict(embedding[:, :-1], encoded["act_emb"])
            shuffled_actions = _shuffle_actions_within_group(actions, groups)
            shuffled = model.predict(embedding[:, :-1], model.action_encoder(shuffled_actions))
            # Remove action conditioning at the predictor interface. Passing a
            # zero command through the biased action encoder is a different,
            # out-of-distribution intervention and must not be called no-action.
            no_action = model.predict(
                embedding[:, :-1], torch.zeros_like(encoded["act_emb"])
            )
            zero_command = model.predict(
                embedding[:, :-1], model.action_encoder(torch.zeros_like(actions))
            )
            factual_mse = (factual - target).square().mean(dim=(1, 2))
            shuffled_mse = (shuffled - target).square().mean(dim=(1, 2))
            no_action_mse = (no_action - target).square().mean(dim=(1, 2))
            zero_command_mse = (zero_command - target).square().mean(dim=(1, 2))
            copy_mse = (embedding[:, :-1] - target).square().mean(dim=(1, 2))
        for index in range(len(pixels)):
            row = {
                "prediction_mse": float(factual_mse[index]),
                "shuffled_action_mse": float(shuffled_mse[index]),
                "no_action_mse": float(no_action_mse[index]),
                "zero_command_mse": float(zero_command_mse[index]),
                "copy_mse": float(copy_mse[index]),
            }
            group = int(groups[index])
            motion = bool(moving[index])
            per_group[group].append(row)
            per_motion[motion].append(row)
            all_rows.append(row)
    if not all_rows:
        raise RuntimeError("validation loader produced no batches")
    group_metrics = {
        group_labels[group]: {**_mean_metrics(rows), "sequences": len(rows)}
        for group, rows in sorted(per_group.items())
    }
    macro = _mean_metrics(
        [
            {key: value for key, value in metrics.items() if key in {
                "prediction_mse",
                "shuffled_action_mse",
                "no_action_mse",
                "zero_command_mse",
                "copy_mse",
            }}
            for metrics in group_metrics.values()
        ]
    )
    result = {
        **macro,
        "micro": {**_mean_metrics(all_rows), "sequences": len(all_rows)},
        "active_motion": {
            **_mean_metrics(per_motion.get(True, [])),
            "sequences": len(per_motion.get(True, [])),
        },
        "near_static": {
            **_mean_metrics(per_motion.get(False, [])),
            "sequences": len(per_motion.get(False, [])),
        },
        "by_source_task": group_metrics,
        "macro_groups": len(group_metrics),
    }
    return result


def learning_rate(step: int, total: int, warmup: int, peak: float) -> float:
    if step <= warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    args: argparse.Namespace,
    run_config: dict,
    init_report: dict | None,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "sigreg_weight": SIGREG_WEIGHT,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "run_config": run_config,
            "init_report": init_report,
            "metrics": metrics,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    temporary.replace(path)


def _group_counts(dataset: SO101Sequences) -> dict[str, dict[str, int]]:
    rows = {
        label: {"sequences": 0, "moving": 0, "near_static": 0}
        for label in dataset.group_labels.values()
    }
    for group_id, moving in zip(dataset.group_ids, dataset.moving, strict=True):
        row = rows[dataset.group_labels[group_id]]
        row["sequences"] += 1
        row["moving" if moving else "near_static"] += 1
    return rows


def _make_loader(
    dataset: SO101Sequences,
    indices: torch.Tensor | list[int],
    args: argparse.Namespace,
    device: torch.device,
    *,
    train: bool,
) -> DataLoader:
    workers = args.workers if train else max(0, min(2, args.workers))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=FixedIndexSampler(torch.as_tensor(indices, dtype=torch.int64)),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=train,
    )


def _restore_rng(checkpoint: dict, device: torch.device) -> None:
    required = ("python_rng_state", "numpy_rng_state", "torch_rng_state")
    if any(key not in checkpoint for key in required):
        raise ValueError("resume checkpoint lacks deterministic RNG state")
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])


def main() -> None:
    args = parse_args()
    if args.device == "cuda:1" or args.device.startswith("cuda:1"):
        raise ValueError("CUDA1 is forbidden for this project")
    if args.history != 3:
        raise ValueError("the released LeWM architecture is fixed to history=3")
    if SIGREG_WEIGHT != 0.02:
        raise AssertionError("SO-101 runs must use SIGReg=0.02")
    if args.steps <= 0 or args.batch_size <= 0 or args.val_batches < 0:
        raise ValueError("steps/batch-size must be positive and val-batches non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    exclusions = parse_exclusions(args.exclude_episode)
    normalization_caches = [Cache(root) for root in args.cache]
    assignments = episode_splits(
        normalization_caches,
        exclusions,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
    )
    if args.action_normalization == "global-train-zscore":
        action_center, action_scale, normalization_frames = global_train_action_stats(
            normalization_caches, assignments, exclusions
        )
    else:
        action_center = np.zeros(6, dtype=np.float32)
        action_scale = np.full(6, args.joint_scale, dtype=np.float32)
        normalization_frames = 0
    dataset_kwargs = {
        "exclusions": exclusions,
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "joint_scale": args.joint_scale,
        "motion_threshold": args.motion_threshold,
        "action_center": action_center,
        "action_scale": action_scale,
    }
    train_set = SO101Sequences(
        args.cache, "train", args.frameskip, args.history, **dataset_kwargs
    )
    val_set = SO101Sequences(
        args.cache, "val", args.frameskip, args.history, **dataset_kwargs
    )

    action_dim = args.frameskip * 6
    model = build_model(action_dim, args.history).to(device)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_step = 0
    init_report = None
    checkpoint = None

    legacy_contract = (
        args.sampling == "source-task-motion-balanced"
        and args.action_normalization == "physical-scale"
    )
    contract = {
        "protocol": "so101-shared-lewm-v2" if legacy_contract else "so101-shared-lewm-v3",
        "caches": [str(path.resolve()) for path in args.cache],
        "cache_datasets": [cache.dataset_id for cache in train_set.caches],
        "robot_types": [cache.robot_type for cache in train_set.caches],
        "cameras": [cache.camera for cache in train_set.caches],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "frameskip": args.frameskip,
        "history": args.history,
        "action_dim": action_dim,
        "joint_scale": args.joint_scale,
        "motion_threshold": args.motion_threshold,
        "exclusions": [f"{source}:{episode}" for source, episode in sorted(exclusions)],
        "sigreg_weight": SIGREG_WEIGHT,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    if not legacy_contract:
        contract.update(
            {
                "sampling": args.sampling,
                "action_normalization": args.action_normalization,
                "action_center": action_center.tolist(),
                "action_scale": action_scale.tolist(),
                "action_normalization_frames": normalization_frames,
            }
        )
    run_config = {
        "contract": contract,
        "train_sequences": len(train_set),
        "val_sequences": len(val_set),
        "train_groups": _group_counts(train_set),
        "val_groups": _group_counts(val_set),
        "sampling": (
            "source-balanced/task-balanced/motion-balanced"
            if args.sampling == "source-task-motion-balanced"
            else "source-balanced/task-balanced/natural-within-task"
        ),
        "action_normalization": (
            "absolute joint targets divided by joint_scale"
            if args.action_normalization == "physical-scale"
            else {
                "kind": "global training-split z-score",
                "center": action_center.tolist(),
                "scale": action_scale.tolist(),
                "training_frames": normalization_frames,
            }
        ),
        "init_report": None,
    }

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if float(checkpoint.get("sigreg_weight")) != SIGREG_WEIGHT:
            raise ValueError("resume checkpoint does not use SIGReg=0.02")
        previous_config = checkpoint.get("run_config")
        if previous_config is None or previous_config.get("contract") != contract:
            raise ValueError("resume checkpoint training contract differs from requested run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        if start_step >= args.steps and not args.eval_only:
            raise ValueError(f"checkpoint already reached step {start_step}")
        init_report = checkpoint.get("init_report")
        run_config["init_report"] = init_report
    elif args.init is not None:
        if not args.init.is_file():
            raise FileNotFoundError(args.init)
        init_report = initialize_from_checkpoint(model, args.init)
        run_config["init_report"] = init_report

    if not args.eval_only and args.resume is None and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"fresh run output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if args.resume and config_path.is_file():
        existing = json.loads(config_path.read_text())
        if existing.get("contract") != contract:
            raise ValueError("output config differs from resume checkpoint contract")
    config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_config, indent=2, sort_keys=True), flush=True)

    quick_limit = None if args.val_batches == 0 else args.val_batches * args.batch_size
    quick_indices = val_set.balanced_eval_indices(quick_limit)
    full_indices = val_set.balanced_eval_indices(None)
    quick_val_loader = _make_loader(val_set, quick_indices, args, device, train=False)
    full_val_loader = _make_loader(val_set, full_indices, args, device, train=False)

    if args.eval_only:
        if checkpoint is None:
            raise ValueError("--eval-only requires --resume")
        loader = full_val_loader if args.val_batches == 0 else quick_val_loader
        metrics = evaluate(model, loader, device, val_set.group_labels)
        (args.output / "evaluation.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    generator = torch.Generator().manual_seed(args.seed)
    all_train_indices = torch.multinomial(
        train_set.balanced_weights(args.sampling),
        num_samples=args.steps * args.batch_size,
        replacement=True,
        generator=generator,
    )
    train_indices = all_train_indices[start_step * args.batch_size :]
    train_loader = _make_loader(train_set, train_indices, args, device, train=True)
    if checkpoint is not None:
        _restore_rng(checkpoint, device)

    model.train()
    metrics_path = args.output / "metrics.jsonl"
    train_iterator = iter(train_loader)
    last_metrics: dict = {}
    started = time.time()
    for step in range(start_step + 1, args.steps + 1):
        batch = next(train_iterator)
        pixels = batch["pixels"].to(device, non_blocking=True)
        actions = batch["action"].to(device, non_blocking=True)
        lr = learning_rate(step, args.steps, args.warmup_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            embedding, prediction, target = prediction_tensors(
                model, {"pixels": pixels, "action": actions}
            )
            prediction_loss = (prediction - target).square().mean()
            sigreg_loss = sigreg(embedding.transpose(0, 1))
            loss = prediction_loss + SIGREG_WEIGHT * sigreg_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            eval_loader = full_val_loader if step == args.steps else quick_val_loader
            last_metrics = evaluate(model, eval_loader, device, val_set.group_labels)
            record = {
                "step": step,
                "train_prediction_loss": float(prediction_loss.detach()),
                "train_sigreg_loss": float(sigreg_loss.detach()),
                "train_loss": float(loss.detach()),
                "lr": lr,
                "elapsed_seconds": time.time() - started,
                **last_metrics,
            }
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if step == args.steps:
                (args.output / "evaluation.json").write_text(
                    json.dumps(last_metrics, indent=2, sort_keys=True) + "\n"
                )
            model.train()

        if step % args.save_every == 0 or step == args.steps:
            for path in (
                args.output / f"step_{step:06d}.pt",
                args.output / "latest.pt",
            ):
                save_checkpoint(
                    path,
                    model,
                    optimizer,
                    scaler,
                    step,
                    args,
                    run_config,
                    init_report,
                    last_metrics,
                )


if __name__ == "__main__":
    main()
