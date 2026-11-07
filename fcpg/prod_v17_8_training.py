"""Training, cache, and evaluation contracts for Prod-V17.8.

V17.8 deliberately reclassifies the complete 320K archive as training data.
Every Actor example is an exact row/horizon join of current visual features,
belief state, completed readback history, task text, and the factual five-step
command chunk.  Fish episode 2 remains inaccessible to every optimizer and is
opened only by an explicitly authorised evaluation call.

The implementation reuses the proven V17.7 image/cache and recurrent belief
plumbing without modifying its files.  Component checkpoints and belief/task
caches have a separate V17.8 contract, so old artifacts cannot be loaded by
accident.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
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

from fcpg.prod_v17_5 import JOINT_POSITION_UNIT, restore_companion_joint_units
from fcpg.prod_v17_8 import (
    TASK_ENCODER,
    TASK_EMBEDDING_DIM,
    V178Core,
    V178RawHistory,
    hard_permutation_pair_cross_entropy,
)
from fcpg.prod_v18_1_data import (
    FISH_COMMAND_STRIDE,
    FISH_ENDPOINT_STRIDE,
    FISH_TRAIN_EPISODES,
    TRANSITIONS,
    V181FeatureCache,
    V181FishArchive,
    V181FishPair,
    V181TemporalArchive,
)
from fcpg import prod_v17_7_training as _v177
from fcpg import prod_v18_1_training as _v181


VERSION = "17.8.0"
SEED = 17_800
ARCHITECTURE_CONTRACT = "task-conditioned-historical-belief-actor-v1"
FISH_TASK_TEXT = "Pick up the red fish toy and place it in the cardboard box"
VIEWS = 2
HYPOTHESES = 2
GRID_SIZE = 56
BELIEF_CELLS = GRID_SIZE * GRID_SIZE + 1
FEATURE_DIM = 384
VIT_PATCHES = 256
STATE_DIM = 6
COMMAND_STEPS = 5
PATH_STEPS = 6
VIEW_ORDER = ("wrist", "column")
CHECKPOINT_EVERY = 500
GRADIENT_CLIP = 1.0
CAD_PASSES = 4
REAL_REFIT_UPDATES = 600
ACTOR_ARCHIVE_PASSES = 4
ACTOR_FISH_REPLAY_UPDATES = 4_000
ACTOR_ARCHIVE_LR = 1.0e-4
ACTOR_FISH_REPLAY_LR = 3.0e-5
JOINT_LOWER = np.asarray([-100, -100, -100, -100, -100, 0], dtype=np.float32)
JOINT_UPPER = np.full(STATE_DIM, 100.0, dtype=np.float32)


class V178TrainingError(RuntimeError):
    """Raised before malformed V17.8 data can change model state."""


def seed_v178(seed: int = SEED) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


configure_cuda = _v177.configure_cuda


def require_physical_gpu1(device: torch.device) -> None:
    if device.type == "cuda" and (
        str(device) != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1"
    ):
        raise V178TrainingError(
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    cursor: Mapping[str, Any] | None = None,
) -> Path:
    """Save an ordinary, resumable V17.8 checkpoint."""

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
        raise V178TrainingError(f"V17.8 component is absent: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise V178TrainingError("V17.8 checkpoint must be a mapping")
    if (
        payload.get("version") != VERSION
        or payload.get("architecture") != ARCHITECTURE_CONTRACT
        or payload.get("component") != str(component)
        or payload.get("seed") != SEED
        or payload.get("config") != _model_config(module)
        or not isinstance(payload.get("model"), dict)
    ):
        raise V178TrainingError("V17.8 checkpoint contract differs")
    module.load_state_dict(payload["model"], strict=True)
    module.to(device)
    return payload


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


def _progress(stage: str, step: int, total: int, loss: float, started: float) -> None:
    if step != 1 and step != total and step % 50:
        return
    elapsed = max(time.monotonic() - started, 1.0e-6)
    rate = step / elapsed
    print(
        json.dumps(
            {
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
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _batches(values: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(values), int(batch_size)):
        yield values[start : start + int(batch_size)]


def all_archive_rows(archive: V181TemporalArchive) -> np.ndarray:
    """The explicit V17.8 training population, including the old 48K DEV."""

    rows = np.arange(int(archive.rows), dtype=np.int64)
    if len(rows) != 320_000:
        raise V178TrainingError(
            f"formal V17.8 archive must contain 320000 rows, observed {len(rows)}"
        )
    return rows


def actor_archive_exposure_contract(
    rows: int = 320_000,
    *,
    horizons: int = TRANSITIONS,
    passes: int = ACTOR_ARCHIVE_PASSES,
) -> dict[str, int]:
    if rows <= 0 or horizons != 4 or passes != 4:
        raise V178TrainingError("V17.8 Actor exposure contract is fixed at N x 4 x 4")
    return {
        "rows": int(rows),
        "horizons": int(horizons),
        "passes": int(passes),
        "base_pair_exposures": int(rows * horizons * passes),
    }


def _canonical_task(value: str) -> str:
    return " ".join(str(value).strip().split()).casefold()


@dataclass(frozen=True)
class V178TaskVocabulary:
    """Exact archive-row to frozen-text-embedding identity."""

    task_texts: tuple[str, ...]
    row_embedding_index: np.ndarray
    manifest_task_index: np.ndarray
    fish_index: int
    manifest_task_count: int

    @classmethod
    def from_archive(
        cls,
        archive: V181TemporalArchive,
        endpoint_root: str | Path | None = None,
        temporal_root: str | Path | None = None,
    ) -> "V178TaskVocabulary":
        endpoint = Path(endpoint_root or archive.endpoint_root).expanduser().resolve()
        temporal = Path(temporal_root or archive.temporal_root).expanduser().resolve()
        task_path = endpoint / "task_index.npy"
        manifest_path = temporal / "manifest.json"
        try:
            task_index = np.load(task_path, mmap_mode="r", allow_pickle=False)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise V178TrainingError("archive task vocabulary is unreadable") from error
        if task_index.shape != (archive.rows,) or task_index.dtype != np.int32:
            raise V178TrainingError("task_index.npy must be int32 [320000]")
        names = manifest.get("task_names")
        if not isinstance(names, list) or not names or not all(
            isinstance(value, str) and value.strip() for value in names
        ):
            raise V178TrainingError("temporal manifest task_names are malformed")
        task_values = np.asarray(task_index, dtype=np.int64)
        if bool(((task_values < 0) | (task_values >= len(names))).any()):
            raise V178TrainingError("task_index leaves temporal task_names")

        # A row task must belong to the exact source episode of the row's H0
        # endpoint.  This prevents a visually plausible but cross-chain text
        # join from silently becoming an Actor shortcut.
        episode = np.asarray(
            archive.endpoint_episode_index[
                np.asarray(archive.row_endpoint_index[:, 0], dtype=np.int64)
            ],
            dtype=np.int64,
        )
        canonical_names = tuple(_canonical_task(value) for value in names)
        allowed_by_episode: list[set[str]] = []
        for record in archive.episodes:
            row = record.get("manifest_row")
            tasks = row.get("tasks") if isinstance(row, Mapping) else None
            if not isinstance(tasks, list) or not tasks:
                raise V178TrainingError("temporal episode has no task text")
            allowed = {
                _canonical_task(value)
                for value in tasks
                if isinstance(value, str) and value.strip()
            }
            if not allowed:
                raise V178TrainingError("temporal episode task text is empty")
            allowed_by_episode.append(allowed)
        for start in range(0, archive.rows, 16_384):
            stop = min(archive.rows, start + 16_384)
            for row_index in range(start, stop):
                chosen = canonical_names[int(task_values[row_index])]
                if chosen not in allowed_by_episode[int(episode[row_index])]:
                    raise V178TrainingError(
                        f"task_index/episode mismatch at archive row {row_index}"
                    )

        unique_text: list[str] = []
        by_canonical: dict[str, int] = {}
        manifest_to_embedding = np.empty(len(names), dtype=np.int64)
        for index, raw in enumerate(names):
            canonical = canonical_names[index]
            if canonical not in by_canonical:
                by_canonical[canonical] = len(unique_text)
                unique_text.append(" ".join(raw.strip().split()))
            manifest_to_embedding[index] = by_canonical[canonical]
        # Fish always receives the literal frozen instruction, even if an
        # archive task differs only in case or whitespace.  Runtime and replay
        # must encode exactly the same user-visible task string.
        try:
            fish_index = unique_text.index(FISH_TASK_TEXT)
        except ValueError:
            fish_index = len(unique_text)
            unique_text.append(FISH_TASK_TEXT)
        row_embedding = manifest_to_embedding[task_values]
        return cls(
            task_texts=tuple(unique_text),
            row_embedding_index=np.asarray(row_embedding, dtype=np.int64),
            manifest_task_index=np.asarray(task_values, dtype=np.int64),
            fish_index=int(fish_index),
            manifest_task_count=len(names),
        )

    @property
    def size(self) -> int:
        return len(self.task_texts)

    def indices_for_rows(self, rows: Any) -> np.ndarray:
        selected = np.asarray(rows, dtype=np.int64)
        if bool(((selected < 0) | (selected >= len(self.row_embedding_index))).any()):
            raise V178TrainingError("task embedding row leaves the archive")
        return np.asarray(self.row_embedding_index[selected], dtype=np.int64)

    def to_json(self, *, include_texts: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "encoder": TASK_ENCODER,
            "embedding_dim": TASK_EMBEDDING_DIM,
            "manifest_task_count": int(self.manifest_task_count),
            "unique_task_count": int(self.size),
            "fish_task_text": FISH_TASK_TEXT,
            "fish_index": int(self.fish_index),
            "row_count": int(len(self.row_embedding_index)),
        }
        if include_texts:
            value["task_texts"] = list(self.task_texts)
        return value


class V178TaskEmbeddingCache:
    """Read-only frozen MiniLM embeddings addressed by archive row or text id."""

    def __init__(
        self,
        root: str | Path,
        vocabulary: V178TaskVocabulary,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        try:
            config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
            self.embeddings = np.load(
                self.root / "embeddings.npy", mmap_mode="r", allow_pickle=False
            )
            cached_row_index = np.load(
                self.root / "task_index.npy", mmap_mode="r", allow_pickle=False
            )
            cached_names = json.loads(
                (self.root / "task_names.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise V178TrainingError("task embedding cache is unreadable") from error
        expected = {
            "version": VERSION,
            "contract": ARCHITECTURE_CONTRACT,
            "encoder": TASK_ENCODER,
            "pooling": "attention_mask_mean_then_l2",
            "embedding_dim": TASK_EMBEDDING_DIM,
            "rows": vocabulary.size,
            "row_count": int(len(vocabulary.row_embedding_index)),
            "manifest_task_count": int(vocabulary.manifest_task_count),
            "complete": True,
            "task_texts": list(vocabulary.task_texts),
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise V178TrainingError(f"task embedding cache {key} differs")
        if self.embeddings.shape != (vocabulary.size, TASK_EMBEDDING_DIM):
            raise V178TrainingError("task embedding cache shape differs")
        if self.embeddings.dtype != np.float32:
            raise V178TrainingError("task embedding cache must be float32")
        sample = np.asarray(self.embeddings, dtype=np.float32)
        if not bool(np.isfinite(sample).all()):
            raise V178TrainingError("task embedding cache contains non-finite values")
        norms = np.linalg.norm(sample, axis=1)
        if float(np.max(np.abs(norms - 1.0))) > 2.0e-4:
            raise V178TrainingError("task embeddings must be L2 normalized")
        if (
            cached_row_index.shape != vocabulary.row_embedding_index.shape
            or cached_row_index.dtype != np.int64
            or not bool(
                np.array_equal(
                    np.asarray(cached_row_index, dtype=np.int64),
                    vocabulary.row_embedding_index,
                )
            )
        ):
            raise V178TrainingError("cached task_index row mapping differs")
        if cached_names != list(vocabulary.task_texts):
            raise V178TrainingError("cached task_names differ")
        self.vocabulary = vocabulary

    def for_rows(self, rows: Any, *, device: torch.device) -> torch.Tensor:
        index = self.vocabulary.indices_for_rows(rows)
        return self.for_indices(index, device=device)

    def for_indices(self, indices: Any, *, device: torch.device) -> torch.Tensor:
        index = np.asarray(indices, dtype=np.int64)
        if bool(((index < 0) | (index >= self.vocabulary.size)).any()):
            raise V178TrainingError("task embedding index leaves the vocabulary")
        value = np.asarray(self.embeddings[index], dtype=np.float32)
        return torch.as_tensor(np.ascontiguousarray(value), device=device)

    def fish(self, batch: int, *, device: torch.device) -> torch.Tensor:
        value = np.asarray(
            self.embeddings[[self.vocabulary.fish_index]], dtype=np.float32
        )
        return torch.as_tensor(value, device=device).expand(int(batch), -1)

    def fish_donor(self, batch: int, *, device: torch.device) -> torch.Tensor:
        if self.vocabulary.size < 2:
            raise V178TrainingError("task shuffle needs at least two unique texts")
        donor = (int(self.vocabulary.fish_index) + 1) % self.vocabulary.size
        return self.for_indices(
            np.full(int(batch), donor, dtype=np.int64), device=device
        )


@torch.inference_mode()
def _encode_task_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise V178TrainingError(
            "transformers is required to build the frozen MiniLM task cache"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.requires_grad_(False)
    model.eval()
    rows: list[np.ndarray] = []
    for start in range(0, len(texts), int(batch_size)):
        batch = list(texts[start : start + int(batch_size)])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = model(**encoded).last_hidden_state.float()
        mask = encoded["attention_mask"].to(hidden)[..., None]
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        pooled = F.normalize(pooled, dim=-1)
        if pooled.shape[1] != TASK_EMBEDDING_DIM:
            raise V178TrainingError(
                f"MiniLM returned dimension {pooled.shape[1]}, expected 384"
            )
        rows.append(pooled.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(rows, axis=0)


def prepare_task_embedding_cache(
    vocabulary: V178TaskVocabulary,
    output_root: str | Path,
    *,
    device: torch.device,
    batch_size: int = 128,
    model_name: str = TASK_ENCODER,
) -> Path:
    """Encode every unique training task once with a frozen MiniLM."""

    if model_name != TASK_ENCODER:
        raise V178TrainingError(f"V17.8 task encoder is fixed at {TASK_ENCODER}")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    expected = {
        "version": VERSION,
        "contract": ARCHITECTURE_CONTRACT,
        "encoder": TASK_ENCODER,
        "pooling": "attention_mask_mean_then_l2",
        "embedding_dim": TASK_EMBEDDING_DIM,
        "rows": vocabulary.size,
        "row_count": int(len(vocabulary.row_embedding_index)),
        "manifest_task_count": int(vocabulary.manifest_task_count),
        "task_texts": list(vocabulary.task_texts),
    }
    expected_files = (
        config_path,
        output / "embeddings.npy",
        output / "task_index.npy",
        output / "task_names.json",
    )
    if all(path.is_file() for path in expected_files):
        V178TaskEmbeddingCache(output, vocabulary)
        return output
    if any(path.is_file() for path in expected_files):
        raise V178TrainingError("task cache is partially present")
    _write_json(config_path, {**expected, "complete": False})
    embeddings = _encode_task_texts(
        vocabulary.task_texts,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    temporary = output / "embeddings.tmp.npy"
    np.save(temporary, embeddings, allow_pickle=False)
    temporary.replace(output / "embeddings.npy")
    index_temporary = output / "task_index.tmp.npy"
    np.save(
        index_temporary,
        np.asarray(vocabulary.row_embedding_index, dtype=np.int64),
        allow_pickle=False,
    )
    index_temporary.replace(output / "task_index.npy")
    names_temporary = output / "task_names.json.tmp"
    names_temporary.write_text(
        json.dumps(list(vocabulary.task_texts), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    names_temporary.replace(output / "task_names.json")
    _write_json(config_path, {**expected, "complete": True})
    V178TaskEmbeddingCache(output, vocabulary)
    return output


@dataclass(frozen=True)
class V178ActorLoss:
    total: torch.Tensor
    command: torch.Tensor
    velocity: torch.Tensor


def _command_deltas(commands: torch.Tensor, current_q: torch.Tensor) -> torch.Tensor:
    return torch.diff(torch.cat((current_q[:, None], commands), dim=1), dim=1)


def audit_archive_joint_bounds(
    archive: V181TemporalArchive,
    *,
    vocabulary: V178SourceVocabulary | None = None,
    chunk_rows: int = 4096,
) -> dict[str, Any]:
    """Report, but never rewrite, archive values outside deployment bounds.

    The companion's restored unit is LeRobot calibrated position, not a
    physical angle.  Source calibrations can legitimately extend beyond the
    current arm's deployment envelope, so this audit is descriptive and all
    factual targets remain unchanged.
    """

    rows = np.arange(int(archive.rows), dtype=np.int64)
    row_source = (
        np.zeros(len(rows), dtype=np.int64)
        if vocabulary is None
        else np.asarray(vocabulary.row_index(archive), dtype=np.int64)
    )
    if row_source.shape != (len(rows),) or bool((row_source < 0).any()):
        raise V178TrainingError("archive source index is malformed")
    source_count = int(row_source.max(initial=0)) + 1
    source_rows = np.zeros(source_count, dtype=np.int64)
    source_command_oob = np.zeros((source_count, STATE_DIM), dtype=np.int64)
    source_q_oob = np.zeros((source_count, STATE_DIM), dtype=np.int64)
    source_row_oob = np.zeros(source_count, dtype=np.int64)
    command_oob = np.zeros(STATE_DIM, dtype=np.int64)
    q_oob = np.zeros(STATE_DIM, dtype=np.int64)
    command_min = np.full(STATE_DIM, np.inf, dtype=np.float64)
    command_max = np.full(STATE_DIM, -np.inf, dtype=np.float64)
    q_min = np.full(STATE_DIM, np.inf, dtype=np.float64)
    q_max = np.full(STATE_DIM, -np.inf, dtype=np.float64)
    oob_chunks = 0
    oob_rows = 0
    q_oob_rows = 0
    for selected in _batches(rows, int(chunk_rows)):
        commands = restore_companion_joint_units(
            np.asarray(archive.completed_commands[selected], dtype=np.float32)
        )
        current_q = restore_companion_joint_units(
            np.asarray(
                archive.measured_state_path[selected, :, 0], dtype=np.float32
            )
        )
        if not bool(np.isfinite(commands).all() and np.isfinite(current_q).all()):
            raise V178TrainingError("archive joint-bound audit found non-finite values")
        commands_bad = (commands < JOINT_LOWER) | (commands > JOINT_UPPER)
        q_bad = (current_q < JOINT_LOWER) | (current_q > JOINT_UPPER)
        command_oob += commands_bad.sum(axis=(0, 1, 2), dtype=np.int64)
        q_oob += q_bad.sum(axis=(0, 1), dtype=np.int64)
        command_min = np.minimum(command_min, commands.min(axis=(0, 1, 2)))
        command_max = np.maximum(command_max, commands.max(axis=(0, 1, 2)))
        q_min = np.minimum(q_min, current_q.min(axis=(0, 1)))
        q_max = np.maximum(q_max, current_q.max(axis=(0, 1)))
        chunk_bad = commands_bad.any(axis=(2, 3))
        row_bad = chunk_bad.any(axis=1)
        q_row_bad = q_bad.any(axis=(1, 2))
        oob_chunks += int(chunk_bad.sum())
        oob_rows += int(row_bad.sum())
        q_oob_rows += int(q_row_bad.sum())
        sources = row_source[selected]
        for source in np.unique(sources):
            mask = sources == source
            index = int(source)
            source_rows[index] += int(mask.sum())
            source_command_oob[index] += commands_bad[mask].sum(
                axis=(0, 1, 2), dtype=np.int64
            )
            source_q_oob[index] += q_bad[mask].sum(
                axis=(0, 1), dtype=np.int64
            )
            source_row_oob[index] += int(
                (row_bad[mask] | q_row_bad[mask]).sum()
            )

    names = tuple(getattr(vocabulary, "names", ()))
    by_source = []
    for index in range(source_count):
        if source_rows[index] == 0:
            continue
        name = "unknown" if index == 0 else (
            names[index - 1] if index - 1 < len(names) else f"source-{index}"
        )
        by_source.append(
            {
                "index": index,
                "name": name,
                "rows": int(source_rows[index]),
                "rows_with_any_oob": int(source_row_oob[index]),
                "command_oob_per_joint": source_command_oob[index].tolist(),
                "current_q_oob_per_joint": source_q_oob[index].tolist(),
            }
        )
    return {
        "unit": JOINT_POSITION_UNIT,
        "deployment_lower": JOINT_LOWER.tolist(),
        "deployment_upper": JOINT_UPPER.tolist(),
        "training_targets_clipped": False,
        "deployment_projection_only": True,
        "command_min": command_min.tolist(),
        "command_max": command_max.tolist(),
        "command_oob_per_joint": command_oob.tolist(),
        "command_oob_transition_chunks": int(oob_chunks),
        "command_oob_rows": int(oob_rows),
        "current_q_min": q_min.tolist(),
        "current_q_max": q_max.tolist(),
        "current_q_oob_per_joint": q_oob.tolist(),
        "current_q_oob_rows": int(q_oob_rows),
        "by_source": by_source,
    }


def actor_action_loss(
    predicted_commands: torch.Tensor,
    target_commands: torch.Tensor,
    current_q: torch.Tensor,
    joint_delta_scale: torch.Tensor | np.ndarray | Sequence[float],
) -> V178ActorLoss:
    predicted = torch.as_tensor(predicted_commands).float()
    target = torch.as_tensor(target_commands, device=predicted.device).float()
    q = torch.as_tensor(current_q, device=predicted.device).float()
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[1:] != (5, 6):
        raise V178TrainingError("Actor commands must be matching [B,5,6]")
    if q.shape != (len(predicted), 6):
        raise V178TrainingError("Actor current_q must be [B,6]")
    scale = torch.as_tensor(
        joint_delta_scale, device=predicted.device, dtype=torch.float32
    )
    if scale.shape != (6,) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise V178TrainingError("joint delta loss scale must be positive finite [6]")
    command = F.smooth_l1_loss(
        (predicted - target) / scale,
        torch.zeros_like(predicted),
    )
    velocity_residual = (
        _command_deltas(predicted, q) - _command_deltas(target, q)
    ) / scale
    velocity = F.smooth_l1_loss(
        velocity_residual, torch.zeros_like(velocity_residual)
    )
    return V178ActorLoss(total=command + 0.2 * velocity, command=command, velocity=velocity)


def compute_joint_delta_loss_scale(
    archive: V181TemporalArchive,
    *,
    chunk_rows: int = 4096,
) -> np.ndarray:
    """Population std of all 6.4M factual per-command joint increments.

    The result scales losses only.  It is never copied into the Actor output
    projection and therefore imposes no per-command motion bound.
    """

    rows = all_archive_rows(archive)
    total = 0
    summed = np.zeros(STATE_DIM, dtype=np.float64)
    squared = np.zeros(STATE_DIM, dtype=np.float64)
    for selected in _batches(rows, chunk_rows):
        commands = restore_companion_joint_units(
            np.asarray(archive.completed_commands[selected], dtype=np.float32)
        )
        commands = commands.astype(np.float64)
        q = restore_companion_joint_units(
            np.asarray(archive.measured_state_path[selected, :, 0], dtype=np.float32)
        ).astype(np.float64)
        previous = np.concatenate((q[:, :, None, :], commands[:, :, :-1]), axis=2)
        delta = commands - previous
        flat = delta.reshape(-1, STATE_DIM)
        summed += flat.sum(axis=0, dtype=np.float64)
        squared += np.square(flat).sum(axis=0, dtype=np.float64)
        total += len(flat)
    if total != archive.rows * TRANSITIONS * COMMAND_STEPS:
        raise V178TrainingError("joint delta statistic exposure count differs")
    mean = summed / float(total)
    variance = np.maximum(squared / float(total) - np.square(mean), 0.0)
    scale = np.sqrt(variance).astype(np.float32)
    if not bool(np.isfinite(scale).all()) or bool((scale <= 1.0e-6).any()):
        raise V178TrainingError("archive joint delta scale is degenerate")
    return scale


@contextmanager
def _legacy_v178_contract() -> Iterator[None]:
    """Run reused V17.7 plumbing under the independent V17.8 artifact tag."""

    names = {
        "VERSION": VERSION,
        "SEED": SEED,
        "ARCHITECTURE_CONTRACT": ARCHITECTURE_CONTRACT,
    }
    previous = {key: getattr(_v177, key) for key in names}
    previous_loader = _v177.load_component
    try:
        for key, value in names.items():
            setattr(_v177, key, value)
        # Reused V17.7 stage loops must obey the stronger V17.8 seed/config
        # contract when resuming or returning an already completed artifact.
        _v177.load_component = load_component
        yield
    finally:
        _v177.load_component = previous_loader
        for key, value in previous.items():
            setattr(_v177, key, value)


class _AllRowsArchive:
    """Read-only adapter that exposes the former split as one training set."""

    def __init__(self, archive: V181TemporalArchive) -> None:
        self._archive = archive
        self.split = np.zeros(archive.rows, dtype=np.uint8)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._archive, name)

    def rows_for_partition(self, partition: str) -> np.ndarray:
        if partition == "train":
            return all_archive_rows(self._archive)
        if partition == "dev":
            return np.empty(0, dtype=np.int64)
        raise V178TrainingError(f"unknown archive partition {partition}")


class V178FishArchive:
    """Fish causal pairs with the factual command/readback stream unchanged.

    Source/target frames and measured paths are identical to V18.1.  Neither a
    legacy per-step cap nor an absolute-bound clip is applied to supervision or
    history; the Actor's deployment output performs the final hardware-bound
    projection.
    """

    def __init__(self, archive: V181FishArchive) -> None:
        self._archive = archive

    def __getattr__(self, name: str) -> Any:
        return getattr(self._archive, name)

    @staticmethod
    def _factual_commands(commands: np.ndarray) -> np.ndarray:
        value = np.asarray(commands, dtype=np.float32)
        if not bool(np.isfinite(value).all()):
            raise V178TrainingError("fish factual commands are non-finite")
        return value

    def pair_batch(
        self,
        ordinals: Any,
        *,
        partition: str = "train",
        allow_held: bool = False,
    ) -> V181FishPair:
        sequences = self._archive.sequences_for_partition(partition)  # type: ignore[arg-type]
        selected = np.asarray(ordinals, dtype=np.int64)
        population = len(sequences) * TRANSITIONS
        if (
            selected.ndim != 1
            or len(selected) == 0
            or bool(((selected < 0) | (selected >= population)).any())
        ):
            raise V178TrainingError("fish pair ordinals leave their partition")
        if partition == "held" and not allow_held:
            raise V178TrainingError("fish episode 2 is sealed during optimization")
        local_sequence = selected // TRANSITIONS
        transition = selected % TRANSITIONS
        sequence_rows = self._archive._sequences[sequences[local_sequence]]
        episode = sequence_rows[:, 0]
        start = sequence_rows[:, 1]
        source = start + transition * FISH_ENDPOINT_STRIDE
        target = source + FISH_ENDPOINT_STRIDE
        command_offsets = (
            np.arange(COMMAND_STEPS, dtype=np.int64) * FISH_COMMAND_STRIDE
        )
        path_offsets = np.arange(PATH_STEPS, dtype=np.int64) * FISH_COMMAND_STRIDE
        commands = self._factual_commands(
            self._archive.actions[source[:, None] + command_offsets[None]]
        )
        current_q = np.asarray(self._archive.states[source], dtype=np.float32)
        measured_path = np.asarray(
            self._archive.states[source[:, None] + path_offsets[None]],
            dtype=np.float32,
        )
        history_source = np.full((len(selected), 3), -1, dtype=np.int64)
        history_valid = np.zeros((len(selected), 3), dtype=np.bool_)
        history_commands = np.zeros(
            (len(selected), 3, COMMAND_STEPS, STATE_DIM), dtype=np.float32
        )
        history_paths = np.zeros(
            (len(selected), 3, PATH_STEPS, STATE_DIM), dtype=np.float32
        )
        for row, horizon in enumerate(transition.tolist()):
            count = min(int(horizon), 3)
            for slot in range(count):
                past = int(horizon) - count + slot
                past_source = int(start[row] + past * FISH_ENDPOINT_STRIDE)
                history_source[row, slot] = past_source
                history_valid[row, slot] = True
                history_commands[row, slot] = self._factual_commands(
                    self._archive.actions[past_source + command_offsets]
                )
                history_paths[row, slot] = self._archive.states[
                    past_source + path_offsets
                ]
        return V181FishPair(
            ordinal=selected,
            episode=episode.astype(np.int64, copy=False),
            transition=transition.astype(np.int64, copy=False),
            source_frame=source.astype(np.int64, copy=False),
            target_frame=target.astype(np.int64, copy=False),
            history_source_frames=history_source,
            history_valid=history_valid,
            current_q=current_q,
            target_commands=commands,
            measured_q_path=measured_path,
            history_commands=history_commands,
            history_paths=history_paths,
        )


def as_v178_fish_archive(value: V181FishArchive | V178FishArchive) -> V178FishArchive:
    return value if isinstance(value, V178FishArchive) else V178FishArchive(value)


# Proven data/cache sources retained without modifying their V17.7 definitions.
V178CombinedFeatureSource = _v177.V177CombinedFeatureSource
V178CombinedBeliefSource = _v177.V177CombinedBeliefSource
V178CanonicalRealRGB = _v177.V177CanonicalRealRGB
V178DenseMeasurementSource = _v177.V177DenseMeasurementSource
V178FishMeasurementSource = _v177.V177FishMeasurementSource


def fish_training_frame_indices(archive: V181FishArchive) -> np.ndarray:
    """Global fish frame identities restricted to optimizer episodes 0/1/3."""

    episodes = np.asarray(archive.episodes, dtype=np.int64)
    selected = np.flatnonzero(np.isin(episodes, FISH_TRAIN_EPISODES)).astype(
        np.int64, copy=False
    )
    if len(selected) == 0 or bool((np.diff(selected) <= 0).any()):
        raise V178TrainingError("fish training frame identities are malformed")
    if not bool(np.isin(episodes[selected], FISH_TRAIN_EPISODES).all()):
        raise V178TrainingError("fish training cache selected a sealed episode")
    return selected


@torch.inference_mode()
def prepare_train_fish_dino_cache(
    archive: V181FishArchive,
    output_root: str | Path,
    backbone: nn.Module | None,
    *,
    device: torch.device,
    dino_batch_size: int = 512,
    existing_full_cache: V181FeatureCache | None = None,
) -> Path:
    """Build a sparse DINO cache without reading episode-2 RGB/features.

    When the already-existing full cache is available, only the selected
    episode-0/1/3 rows are copied.  Otherwise DINO is run only on those train
    RGB rows.  The destination never contains an episode-2 endpoint identity.
    """

    endpoint = fish_training_frame_indices(archive)
    valid = np.ones((len(endpoint), VIEWS), dtype=np.bool_)
    output = Path(output_root).expanduser().resolve()
    tokens, done = _v181._create_feature_arrays(
        output,
        endpoint,
        valid,
        kind="fish-v178-train",
    )
    pending = np.flatnonzero(~done.all(axis=1))
    for start in range(0, len(pending), int(dino_batch_size)):
        cache_rows = pending[start : start + int(dino_batch_size)]
        frame_ids = endpoint[cache_rows]
        if existing_full_cache is not None:
            source_rows = existing_full_cache.rows_for(frame_ids)
            source_valid = np.asarray(
                existing_full_cache.view_valid[source_rows], dtype=np.bool_
            )
            if not bool(np.asarray(source_valid, dtype=np.bool_).all()):
                raise V178TrainingError(
                    "existing fish DINO cache has an invalid training view"
                )
            encoded = np.asarray(
                existing_full_cache.tokens[source_rows], dtype=np.float16
            )
            if not bool(np.isfinite(encoded).all()):
                raise V178TrainingError(
                    "existing fish DINO cache has non-finite training features"
                )
        else:
            if backbone is None:
                raise V178TrainingError(
                    "fish train cache needs a frozen DINO backbone or full cache"
                )
            rgb = archive.rgb(frame_ids).reshape(-1, 224, 224, 3)
            encoded = _v181.extract_dinov2_tokens(
                backbone,
                rgb,
                device=device,
                batch_size=dino_batch_size,
            ).reshape(len(cache_rows), VIEWS, VIT_PATCHES, FEATURE_DIM)
        tokens[cache_rows] = encoded
        done[cache_rows] = True
        _v181._flush_and_release_pages(tokens, done)
    if not bool(done.all()):
        raise V178TrainingError("fish training DINO cache remains incomplete")
    config = dict(
        json.loads((output / "config.json").read_text(encoding="utf-8"))
    )
    config.update(
        {
            "complete": True,
            "fish_episodes": list(map(int, FISH_TRAIN_EPISODES)),
            "episode_2_rgb_read": False,
            "episode_2_feature_read": False,
            "population_source": (
                "selected_rows_from_existing_full_cache"
                if existing_full_cache is not None
                else "selected_train_rgb_dino_encoding"
            ),
            "endpoint_identity": "global_fish_frame_index",
        }
    )
    _v181._json_write(output / "config.json", config)
    return output
V178GeometryIdentityProvider = _v177.V177GeometryIdentityProvider
V178DeviceBeliefBank = _v177.V177DeviceBeliefBank
dense_endpoint_map = _v177.dense_endpoint_map


class V178SourceVocabulary(_v177.V177SourceVocabulary):
    def to_json(self) -> dict[str, Any]:
        value = super().to_json()
        value["version"] = VERSION
        return value


class V178BeliefCache:
    """Stopped V17.8 visual measurements addressed by endpoint identity."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_kind: str | None = None,
        expected_model_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        try:
            config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise V178TrainingError("belief cache config is malformed") from error
        if (
            config.get("version") != VERSION
            or config.get("contract") != ARCHITECTURE_CONTRACT
            or tuple(config.get("view_order", ())) != VIEW_ORDER
            or not bool(config.get("complete"))
        ):
            raise V178TrainingError("belief cache contract is incomplete or differs")
        measurement = config.get("measurement_contract")
        if (
            not isinstance(measurement, Mapping)
            or measurement.get("version") != VERSION
            or measurement.get("architecture") != ARCHITECTURE_CONTRACT
            or measurement.get("component") != "measurement"
            or measurement.get("seed") != SEED
            or not isinstance(measurement.get("model_config"), Mapping)
        ):
            raise V178TrainingError(
                "belief cache is not bound to a V17.8 Measurement contract"
            )
        if (
            expected_model_config is not None
            and dict(measurement["model_config"])
            != json.loads(json.dumps(dict(expected_model_config), sort_keys=True))
        ):
            raise V178TrainingError("belief cache Measurement config differs")
        self.kind = str(config.get("kind", ""))
        if expected_kind is not None and self.kind != expected_kind:
            raise V178TrainingError(
                f"belief cache kind is {self.kind}, expected {expected_kind}"
            )
        try:
            self.endpoint_index = np.load(
                self.root / "endpoint_index.npy", mmap_mode="r", allow_pickle=False
            )
            self.anchor_hypothesis_log_likelihood = np.load(
                self.root / "anchor_hypothesis_log_likelihood.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            self.view_valid = np.load(
                self.root / "view_valid.npy", mmap_mode="r", allow_pickle=False
            )
        except (OSError, ValueError) as error:
            raise V178TrainingError("belief cache arrays are unreadable") from error
        rows = len(self.endpoint_index)
        expected = {
            "endpoint_index": ((rows,), np.dtype(np.int64)),
            "anchor_hypothesis_log_likelihood": (
                (rows, VIEWS, HYPOTHESES, BELIEF_CELLS),
                np.dtype(np.float16),
            ),
            "view_valid": ((rows, VIEWS), np.dtype(np.bool_)),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise V178TrainingError(f"belief cache {name} shape/dtype differs")
        if rows == 0 or bool((np.diff(self.endpoint_index) <= 0).any()):
            raise V178TrainingError("belief endpoint identities must be sorted unique")
        sample = np.unique(
            np.linspace(0, rows - 1, min(rows, 128), dtype=np.int64)
        )
        logp = np.asarray(
            self.anchor_hypothesis_log_likelihood[sample], dtype=np.float32
        )
        if not bool(np.isfinite(logp).all()):
            raise V178TrainingError("belief cache contains non-finite values")
        if float(np.max(np.abs(np.logaddexp.reduce(logp, axis=-1)))) > 3.0e-2:
            raise V178TrainingError("belief cache is not normalized")

    def rows_for(self, endpoint_indices: Any) -> np.ndarray:
        requested = np.asarray(endpoint_indices, dtype=np.int64)
        flat = requested.reshape(-1)
        position = np.searchsorted(self.endpoint_index, flat)
        safe = np.minimum(position, len(self.endpoint_index) - 1)
        if bool((position >= len(self.endpoint_index)).any()) or not bool(
            np.array_equal(self.endpoint_index[safe], flat)
        ):
            raise V178TrainingError("endpoint is absent from belief cache")
        return position.reshape(requested.shape)

    def get_numpy(self, endpoint_indices: Any) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(endpoint_indices, dtype=np.int64)
        rows = self.rows_for(requested)
        shape = requested.shape
        return (
            np.asarray(
                self.anchor_hypothesis_log_likelihood[rows], dtype=np.float32
            ).reshape(*shape, VIEWS, HYPOTHESES, BELIEF_CELLS),
            np.asarray(self.view_valid[rows], dtype=np.bool_).reshape(*shape, VIEWS),
        )


@torch.inference_mode()
def prepare_belief_cache(
    core: V178Core,
    feature_source: Any,
    endpoint_index: np.ndarray,
    output_root: str | Path,
    *,
    device: torch.device,
    kind: str,
    batch_size: int = 64,
    rgb_source: Any | None = None,
) -> Path:
    """Cache only stopped measurements under the V17.8 cache contract."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    measurement_contract = {
        "version": VERSION,
        "architecture": ARCHITECTURE_CONTRACT,
        "component": "measurement",
        "seed": SEED,
        "model_config": json.loads(
            json.dumps(_model_config(core), sort_keys=True)
        ),
    }
    if config_path.is_file():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V178TrainingError("belief cache config is malformed") from exc
        bound = existing.get("measurement_contract")
        if bound is not None and bound != measurement_contract:
            raise V178TrainingError("belief cache Measurement contract differs")
        if bool(existing.get("complete")) and bound is None:
            raise V178TrainingError(
                "complete belief cache lacks Measurement provenance"
            )
    else:
        endpoint = np.asarray(endpoint_index, dtype=np.int64)
        _write_json(
            config_path,
            {
                "version": VERSION,
                "contract": ARCHITECTURE_CONTRACT,
                "kind": str(kind),
                "view_order": list(VIEW_ORDER),
                "rows": int(len(endpoint)),
                "contents": "stopped current-image measurement only",
                "identity_mode": "exchangeable_raw_measurement",
                "measurement_contract": measurement_contract,
                "complete": False,
            },
        )

    with _legacy_v178_contract():
        result = _v177.prepare_belief_cache(
            core,
            feature_source,
            endpoint_index,
            output_root,
            device=device,
            kind=kind,
            batch_size=batch_size,
            rgb_source=rgb_source,
        )
    # The reused writer replaces its config at finalization, so restore the
    # stronger semantic binding after all arrays are durable.
    completed_config = json.loads(config_path.read_text(encoding="utf-8"))
    _write_json(
        config_path,
        {**completed_config, "measurement_contract": measurement_contract},
    )
    V178BeliefCache(
        result,
        expected_kind=kind,
        expected_model_config=_model_config(core),
    )
    return result


def _rewrite_final_metrics(
    core: V178Core,
    output_root: str | Path,
    component: str,
    metrics: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    final = Path(output_root).expanduser().resolve() / component / "final.pt"
    payload = load_component(final, core, component=component, device=device)
    value = dict(metrics)
    save_component(
        final,
        core,
        component=component,
        step=int(payload.get("step", value.get("steps", 0))),
        metrics=value,
    )
    _merge_metrics(
        Path(output_root).expanduser().resolve() / "train-metrics.json",
        component,
        value,
    )
    return value


def train_measurement(
    core: V178Core,
    real_rgb: Any,
    cad: Any,
    temporal_features: Any,
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
    """Train V17.8 appearance and hard-set spatial measurement from scratch."""

    if measurement_init is not None:
        raise V178TrainingError(
            "V17.8 requires a complete Measurement retrain; old weights are forbidden"
        )
    old_loss = _v177.exchangeable_pair_nll
    old_penalty = _v177.hypothesis_separation_penalty
    old_parameters = _v177._measurement_spatial_parameters

    def zero_penalty(
        predicted: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        del target, valid
        return predicted.float().sum() * 0.0

    def spatial_parameters(module: V178Core, *, update_rgb: bool) -> list[nn.Parameter]:
        # CAD fused batches must send spatial gradients into the high-resolution
        # RGB stem/fusion.  DINO projection remains frozen exactly as before.
        return old_parameters(module, update_rgb=True)

    try:
        _v177.exchangeable_pair_nll = hard_permutation_pair_cross_entropy
        _v177.hypothesis_separation_penalty = zero_penalty
        _v177._measurement_spatial_parameters = spatial_parameters
        with _legacy_v178_contract():
            result = _v177.train_measurement(
                core,
                real_rgb,
                cad,
                temporal_features,
                cad_features,
                archive,
                backbone,
                real_label_cache,
                output_root,
                device=device,
                appearance_batch=appearance_batch,
                cad_batch=cad_batch,
                real_batch=real_batch,
                smoke=smoke,
                measurement_init=None,
            )
    finally:
        _v177.exchangeable_pair_nll = old_loss
        _v177.hypothesis_separation_penalty = old_penalty
        _v177._measurement_spatial_parameters = old_parameters
    metrics = {
        key: value
        for key, value in result.items()
        if "overlap" not in key
        and "separation" not in key
        and key != "cad_fused_batches_train_readout_over_fixed_rgb_features"
    }
    if "cad_exchangeable_soft_target_cross_entropy" in metrics:
        metrics["cad_hard_permutation_set_cross_entropy"] = metrics.pop(
            "cad_exchangeable_soft_target_cross_entropy"
        )
    if "real_training_fit_exchangeable_soft_target_cross_entropy" in metrics:
        metrics["real_training_fit_hard_permutation_set_cross_entropy"] = metrics.pop(
            "real_training_fit_exchangeable_soft_target_cross_entropy"
        )
    metrics.update(
        {
            "set_matching": "detached_hard_min_of_direct_and_swap",
            "hypothesis_overlap_penalty": False,
            "cad_passes": 0 if smoke else CAD_PASSES,
            "real_refit_updates": 2 if smoke else REAL_REFIT_UPDATES,
            "cad_fused_batches_update_rgb_stem_and_fusion": True,
            "cad_fused_batches_train_readout_over_fixed_rgb_features": False,
            "cad_rgb_spatial_supervision_fraction": 0.5,
            "measurement_init": None,
        }
    )
    return _rewrite_final_metrics(
        core, output_root, "measurement", metrics, device=device
    )


def train_lewm(
    core: V178Core,
    archive: V181TemporalArchive,
    temporal_beliefs: Any,
    dense_beliefs: V178BeliefCache,
    dense_rows: np.ndarray,
    output_root: str | Path,
    *,
    device: torch.device,
    vocabulary: V178SourceVocabulary,
    fish: V181FishArchive | None = None,
    fish_source: Any | None = None,
    identity: Any | None = None,
    effective_batch: int = 256,
    microbatch: int = 64,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train recurrent transport on archive, dense RGB and the calibrated rig.

    The final fish phase remains task-free: it teaches only the otherwise
    unseen fish camera-source embedding and the analytic-residual branch.  No
    task text or Actor target is visible to the LeWM.
    """

    fish_inputs = (fish is not None, fish_source is not None, identity is not None)
    if not all(fish_inputs):
        raise V178TrainingError(
            "V17.8 LeWM is incomplete without the calibrated-rig phase; fish, "
            "measurement source, and identity are all required"
        )
    proxy = _AllRowsArchive(archive)
    with _legacy_v178_contract():
        result = _v177.train_lewm(
            core,
            proxy,
            temporal_beliefs,
            dense_beliefs,
            dense_rows,
            output_root,
            device=device,
            vocabulary=vocabulary,
            fish=as_v178_fish_archive(fish),
            fish_source=fish_source,
            identity=identity,
            effective_batch=effective_batch,
            microbatch=microbatch,
            smoke=smoke,
        )
    metrics = dict(result)
    metrics.update(
        {
            "archive_partition": "all_320k_reclassified_training",
            "archive_train_rows": int(archive.rows),
            "former_dev_rows_reclassified_training": int(
                np.count_nonzero(np.asarray(archive.split) == 1)
            ),
            "former_dev_rows_excluded": 0,
            "task_conditioning": False,
            "causal_recursive_unroll": True,
            "stop_gradient_between_transitions": True,
            "predicted_belief_writeback": False,
            "calibrated_fish_rig_phase": True,
            "calibrated_fish_rig_task_conditioning": False,
            "fish_train_episodes": [0, 1, 3],
        }
    )
    return _rewrite_final_metrics(core, output_root, "lewm", metrics, device=device)


def _set_trainable(module: nn.Module, selected: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    active_ids = {id(value) for value in selected}
    active: list[nn.Parameter] = []
    for value in module.parameters():
        enabled = id(value) in active_ids
        value.requires_grad_(enabled)
        if enabled:
            active.append(value)
    if not active:
        raise V178TrainingError("optimizer has no trainable parameters")
    return active


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(value), device=device, dtype=torch.float32)


def _archive_target(
    archive: V181TemporalArchive,
    rows: np.ndarray,
    horizon: int,
) -> np.ndarray:
    selected = np.asarray(rows, dtype=np.int64)
    if int(horizon) not in range(TRANSITIONS):
        raise V178TrainingError("archive Actor horizon is outside H0-H3")
    target = restore_companion_joint_units(
        np.asarray(
            archive.completed_commands[selected, int(horizon)], dtype=np.float32
        )
    )
    if target.shape != (len(selected), COMMAND_STEPS, STATE_DIM) or not bool(
        np.isfinite(target).all()
    ):
        raise V178TrainingError("factual archive commands are malformed")
    # No legacy step-cap or absolute-bound label clipping is allowed here.
    # These are the factual calibrated-position commands for this row/horizon.
    return np.asarray(target, dtype=np.float32)


def _actor_schedule(
    archive: V181TemporalArchive,
    features: Any,
    *,
    batch_size: int,
    smoke: bool,
) -> list[tuple[int, np.ndarray]]:
    rows = all_archive_rows(archive)
    if smoke:
        return [(0, rows[: min(len(rows), batch_size)])]
    schedule: list[tuple[int, np.ndarray]] = []
    for pass_index in range(ACTOR_ARCHIVE_PASSES):
        for horizon in range(TRANSITIONS):
            endpoints = np.asarray(
                archive.row_endpoint_index[rows, horizon], dtype=np.int64
            )
            localized = _v177._localized_feature_rows(
                rows,
                endpoints,
                features,
                seed=SEED + 700 + 10 * pass_index + horizon,
            )
            if len(localized) != archive.rows or len(np.unique(localized)) != archive.rows:
                raise V178TrainingError("Actor schedule is not one exact archive pass")
            schedule.append((horizon, localized))
    return schedule


def _history_blank(history: V178RawHistory) -> V178RawHistory:
    return V178RawHistory(
        commands=history.commands,
        paths=history.paths,
        scene_pooled=history.scene_pooled,
        valid=torch.zeros_like(history.valid),
    )


def _history_reverse(history: V178RawHistory) -> V178RawHistory:
    return V178RawHistory(
        commands=history.commands.flip(1),
        paths=history.paths.flip(1),
        scene_pooled=history.scene_pooled.flip(1),
        valid=history.valid.flip(1),
    )


def _history_shuffle(history: V178RawHistory) -> V178RawHistory:
    if len(history.valid) <= 1:
        return history
    order = torch.roll(
        torch.arange(len(history.valid), device=history.valid.device), shifts=1
    )
    return V178RawHistory(
        commands=history.commands[order],
        paths=history.paths[order],
        scene_pooled=history.scene_pooled[order],
        valid=history.valid[order],
    )


def train_actor(
    core: V178Core,
    archive: V181TemporalArchive,
    fish: V181FishArchive,
    archive_beliefs: Any,
    archive_features: Any,
    fish_source: V178FishMeasurementSource,
    identity: V178GeometryIdentityProvider,
    task_vocabulary: V178TaskVocabulary,
    task_cache: V178TaskEmbeddingCache,
    output_root: str | Path,
    *,
    device: torch.device,
    vocabulary: V178SourceVocabulary,
    archive_effective_batch: int = 256,
    fish_effective_batch: int = 64,
    microbatch: int = 64,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train one unified task/history/belief Actor.

    The base phase contains exactly ``320K x 4 horizons x 4 passes`` examples.
    The final 4000 updates jointly optimize a fish batch and an archive replay
    batch so task specialization cannot silently replace the generic policy.
    """

    if archive_effective_batch % microbatch or fish_effective_batch % microbatch:
        raise V178TrainingError("Actor effective batches must divide into microbatches")
    fish_v178 = as_v178_fish_archive(fish)
    if len(task_vocabulary.row_embedding_index) != archive.rows:
        raise V178TrainingError("task vocabulary row count differs from archive")
    _v177._check_source_vocabulary(core, vocabulary)
    row_source = vocabulary.row_index(archive)
    joint_bound_audit = audit_archive_joint_bounds(
        archive, vocabulary=vocabulary
    )
    output = Path(output_root).expanduser().resolve()
    stage_dir = output / "actor"
    latest_path = stage_dir / "latest.pt"
    final_path = stage_dir / "final.pt"
    if final_path.is_file():
        payload = load_component(final_path, core, component="actor", device=device)
        return dict(payload.get("metrics", {}))

    core.to(device)
    core.measurement.eval()
    core.grounding_encoder.eval()
    core.lewm.eval()
    trainable = _set_trainable(
        core,
        list(core.actor.parameters()) + list(core.innovation_encoder.parameters()),
    )
    loss_scale = compute_joint_delta_loss_scale(archive)
    schedule = _actor_schedule(
        archive,
        archive_features,
        batch_size=archive_effective_batch,
        smoke=smoke,
    )
    archive_steps = sum(
        math.ceil(len(rows) / archive_effective_batch) for _, rows in schedule
    )
    replay_steps = 2 if smoke else ACTOR_FISH_REPLAY_UPDATES
    total_steps = archive_steps + replay_steps

    payload: dict[str, Any] | None = None
    if latest_path.is_file():
        payload = load_component(latest_path, core, component="actor", device=device)
    resume_step = int(payload.get("step", 0)) if payload is not None else 0
    previous_metrics = dict(payload.get("metrics", {})) if payload is not None else {}
    global_step = 0
    started = time.monotonic()
    last_total = float(previous_metrics.get("loss", float("nan")))
    last_command = float(previous_metrics.get("command_smooth_l1", float("nan")))
    last_velocity = float(previous_metrics.get("velocity_smooth_l1", float("nan")))

    def optimize(
        optimizer: torch.optim.Optimizer,
        batches: Sequence[
            tuple[
                torch.Tensor,
                Any,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                V178RawHistory,
                torch.Tensor,
            ]
        ],
        group_weights: Sequence[float] | None = None,
    ) -> tuple[float, float, float]:
        optimizer.zero_grad(set_to_none=True)
        if not batches:
            raise V178TrainingError("Actor optimizer received no microbatches")
        if group_weights is None:
            examples = float(sum(len(item[0]) for item in batches))
            weights = [float(len(item[0])) / examples for item in batches]
        else:
            if len(group_weights) != len(batches):
                raise V178TrainingError("Actor group weights differ from batches")
            denominator = float(sum(group_weights))
            weights = [float(value) / denominator for value in group_weights]
        accumulated = [torch.zeros((), device=device) for _ in range(3)]
        scale_tensor = torch.as_tensor(loss_scale, device=device)
        for item, weight in zip(batches, weights, strict=True):
            scene, grounded, innovation, q, target, history, task = item
            with _autocast(device):
                prediction = core.act(
                    scene,
                    grounded,
                    innovation,
                    q,
                    history=history,
                    task_embedding=task,
                )
                # Supervise the unconstrained cumulative output.  The returned
                # ``commands`` remain the deployment-safe absolute projection,
                # but source-calibration overshoot must not rewrite labels.
                losses = actor_action_loss(
                    prediction.desired_commands, target, q, scale_tensor
                )
            (losses.total * weight).backward()
            for accumulator, value in zip(
                accumulated,
                (losses.total, losses.command, losses.velocity),
                strict=True,
            ):
                accumulator.add_(value.detach() * weight)
        torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
        optimizer.step()
        return tuple(float(value.cpu()) for value in accumulated)  # type: ignore[return-value]

    optimizer = torch.optim.AdamW(trainable, lr=ACTOR_ARCHIVE_LR)
    if (
        payload is not None
        and payload.get("cursor", {}).get("phase") == "archive"
        and isinstance(payload.get("optimizer"), Mapping)
    ):
        optimizer.load_state_dict(payload["optimizer"])

    archive_exposures = 0
    for horizon, rows in schedule:
        for selected in _batches(rows, archive_effective_batch):
            if global_step < resume_step:
                archive_exposures += len(selected)
                global_step += 1
                continue
            micro_values = []
            for micro in _batches(selected, microbatch):
                # The belief/GDM parameters are frozen, but the Actor-facing
                # innovation encoder is trained here; do not wrap this update
                # in no_grad or the innovation branch would remain random.
                with _autocast(device):
                    scene, update, q, history = _v177._archive_cached_actor_context(
                        core,
                        archive,
                        archive_beliefs,
                        archive_features,
                        micro,
                        horizon,
                        device=device,
                        row_source=row_source,
                        collect_history=True,
                    )
                micro_values.append(
                    (
                        scene,
                        update.grounded,
                        update.innovation.innovation,
                        q,
                        _tensor(_archive_target(archive, micro, horizon), device),
                        history,
                        task_cache.for_rows(micro, device=device),
                    )
                )
            last_total, last_command, last_velocity = optimize(
                optimizer, micro_values
            )
            archive_exposures += len(selected)
            global_step += 1
            _progress("actor-archive-full", global_step, total_steps, last_total, started)
            if global_step % CHECKPOINT_EVERY == 0 or global_step == archive_steps:
                save_component(
                    latest_path,
                    core,
                    component="actor",
                    step=global_step,
                    metrics={
                        "loss": last_total,
                        "command_smooth_l1": last_command,
                        "velocity_smooth_l1": last_velocity,
                        "archive_pair_exposures": int(archive_exposures),
                        "joint_delta_loss_scale_calibrated_position": loss_scale.tolist(),
                    },
                    optimizer=optimizer,
                    cursor={"phase": "archive", "horizon": int(horizon)},
                )

    if not smoke and archive_exposures != 5_120_000:
        raise V178TrainingError(
            f"Actor base exposure count is {archive_exposures}, expected 5120000"
        )

    optimizer = torch.optim.AdamW(trainable, lr=ACTOR_FISH_REPLAY_LR)
    if (
        payload is not None
        and payload.get("cursor", {}).get("phase") == "fish_archive_replay"
        and isinstance(payload.get("optimizer"), Mapping)
    ):
        optimizer.load_state_dict(payload["optimizer"])
    train_sequence_count = len(fish_v178.sequences_for_partition("train"))
    if train_sequence_count <= 0:
        raise V178TrainingError("fish training partition has no sequence")
    replay_archive_exposures = 0
    fish_exposures = 0
    for replay_step in range(replay_steps):
        if global_step < resume_step:
            global_step += 1
            replay_archive_exposures += fish_effective_batch
            fish_exposures += fish_effective_batch
            continue
        horizon = replay_step % TRANSITIONS
        generator = np.random.default_rng(SEED + 10_000 + replay_step)
        archive_rows = generator.integers(
            0, archive.rows, size=fish_effective_batch, dtype=np.int64
        )
        fish_sequences = generator.integers(
            0,
            train_sequence_count,
            size=fish_effective_batch,
            dtype=np.int64,
        )
        fish_ordinals = fish_sequences * TRANSITIONS + horizon
        archive_batches = []
        fish_batches = []
        for micro in _batches(archive_rows, microbatch):
            with _autocast(device):
                scene, update, q, history = _v177._archive_cached_actor_context(
                    core,
                    archive,
                    archive_beliefs,
                    archive_features,
                    micro,
                    horizon,
                    device=device,
                    row_source=row_source,
                    collect_history=True,
                )
            archive_batches.append(
                (
                    scene,
                    update.grounded,
                    update.innovation.innovation,
                    q,
                    _tensor(_archive_target(archive, micro, horizon), device),
                    history,
                    task_cache.for_rows(micro, device=device),
                )
            )
        for micro in _batches(fish_ordinals, microbatch):
            with _autocast(device):
                pair, scene, update, q, history = _v177._fish_actor_context(
                    core,
                    identity,
                    fish_v178,
                    fish_source,
                    micro,
                    device=device,
                    vocabulary=vocabulary,
                    collect_history=True,
                )
            fish_batches.append(
                (
                    scene,
                    update.grounded,
                    update.innovation.innovation,
                    q,
                    _tensor(_v177._fish_commands_degrees(pair), device),
                    history,
                    task_cache.fish(len(micro), device=device),
                )
            )
        combined = archive_batches + fish_batches
        # Half of every update is archive replay and half is fish, independent
        # of the chosen microbatch partition.
        weights = [
            0.5 * len(value[0]) / fish_effective_batch
            for value in archive_batches
        ] + [
            0.5 * len(value[0]) / fish_effective_batch
            for value in fish_batches
        ]
        last_total, last_command, last_velocity = optimize(
            optimizer, combined, weights
        )
        global_step += 1
        replay_archive_exposures += len(archive_rows)
        fish_exposures += len(fish_ordinals)
        _progress(
            "actor-fish-archive-replay",
            global_step,
            total_steps,
            last_total,
            started,
        )
        if global_step % CHECKPOINT_EVERY == 0 or replay_step + 1 == replay_steps:
            save_component(
                latest_path,
                core,
                component="actor",
                step=global_step,
                metrics={
                    "loss": last_total,
                    "command_smooth_l1": last_command,
                    "velocity_smooth_l1": last_velocity,
                    "archive_pair_exposures": int(archive_exposures),
                    "archive_replay_exposures": int(replay_archive_exposures),
                    "fish_pair_exposures": int(fish_exposures),
                    "joint_delta_loss_scale_calibrated_position": loss_scale.tolist(),
                },
                optimizer=optimizer,
                cursor={
                    "phase": "fish_archive_replay",
                    "replay_step": replay_step + 1,
                },
            )

    contract = actor_archive_exposure_contract(archive.rows)
    metrics = {
        "steps": int(global_step),
        "loss": last_total,
        "command_smooth_l1": last_command,
        "velocity_smooth_l1": last_velocity,
        "velocity_loss_weight": 0.2,
        "joint_delta_loss_scale_calibrated_position": loss_scale.tolist(),
        "joint_position_unit": JOINT_POSITION_UNIT,
        "actor_loss_output": "unprojected_cumulative_desired_commands",
        "deployment_output": "absolute_joint_bound_projection_only",
        "archive_joint_bound_audit": joint_bound_audit,
        "joint_delta_scale_role": "loss_only_not_output_limit",
        "archive_role": "all_320k_reclassified_training",
        "former_dev_rows_reclassified_training": int(
            np.count_nonzero(np.asarray(archive.split) == 1)
        ),
        "archive_pair_exposures": int(archive_exposures),
        "expected_archive_pair_exposures": int(contract["base_pair_exposures"]),
        "archive_contract_complete": bool(
            smoke or archive_exposures == contract["base_pair_exposures"]
        ),
        "archive_rows": int(archive.rows),
        "archive_horizons": TRANSITIONS,
        "archive_passes": ACTOR_ARCHIVE_PASSES,
        "archive_replay_exposures": int(replay_archive_exposures),
        "fish_pair_exposures": int(fish_exposures),
        "fish_archive_replay_updates": int(replay_steps),
        "fish_train_episodes": [0, 1, 3],
        "fish_task_text": FISH_TASK_TEXT,
        "task_encoder": TASK_ENCODER,
        "task_embedding_dim": TASK_EMBEDDING_DIM,
        "task_embedding_frozen_precomputed": True,
        "raw_actual_history_steps": 3,
        "actor_context": [
            "current_dual_view_scene",
            "paired_grounding_and_innovation",
            "last_three_completed_actual_transitions",
            "current_q",
            "frozen_task_text",
        ],
        "hard_xy_actor_input": False,
        "future_goal_input": False,
        "per_command_step_cap": False,
        "absolute_joint_limits": True,
        "run_kind": "smoke" if smoke else "full",
    }
    core.requires_grad_(False)
    save_component(
        final_path,
        core,
        component="actor",
        step=global_step,
        metrics=metrics,
    )
    _merge_metrics(output / "train-metrics.json", "actor", metrics)
    return metrics


def _accumulate(
    store: dict[str, list[float | int]],
    name: str,
    value: torch.Tensor,
    active: torch.Tensor | None = None,
) -> None:
    tensor = torch.as_tensor(value).float()
    mask = torch.ones_like(tensor, dtype=torch.bool) if active is None else torch.as_tensor(
        active, device=tensor.device, dtype=torch.bool
    )
    if mask.shape != tensor.shape:
        mask = torch.broadcast_to(mask, tensor.shape)
    selected = tensor[mask]
    row = store.setdefault(name, [0.0, 0])
    row[0] = float(row[0]) + float(selected.sum().cpu())
    row[1] = int(row[1]) + int(selected.numel())


def _summarize(store: Mapping[str, Sequence[float | int]]) -> dict[str, Any]:
    return {
        key: {
            "mean": (float(value[0]) / int(value[1]) if int(value[1]) else None),
            "sum": float(value[0]),
            "count": int(value[1]),
        }
        for key, value in store.items()
    }


def _hard_pair_ce_per_view(
    predicted_log: torch.Tensor,
    target_probability: torch.Tensor,
) -> torch.Tensor:
    predicted = torch.log_softmax(predicted_log.float(), dim=-1)
    target = target_probability.float().to(predicted)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    cost = -torch.einsum("bvac,bvsc->bvas", target, predicted)
    direct = cost[:, :, 0, 0] + cost[:, :, 1, 1]
    swapped = cost[:, :, 0, 1] + cost[:, :, 1, 0]
    return torch.where(direct <= swapped, direct, swapped)


def _commands_mae_per_example(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return (predicted.float() - target.float()).abs().mean(dim=(1, 2))


def _absolute_bound_action_shuffle(
    current_q: torch.Tensor,
    commands: torch.Tensor,
    core: V178Core,
) -> torch.Tensor:
    delta = _command_deltas(commands.float(), current_q.float())
    if len(delta) > 1:
        delta = torch.roll(delta, shifts=1, dims=0)
    else:
        delta = delta.flip(1)
    desired = current_q[:, None].float() + delta.cumsum(dim=1)
    lower = core.actor.action_lower.to(desired)
    upper = core.actor.action_upper.to(desired)
    return torch.maximum(torch.minimum(desired, upper), lower)


def _q_only_grounded(grounded: Any) -> Any:
    return replace(
        grounded,
        ordered_tokens=torch.zeros_like(grounded.ordered_tokens),
        dynamics_grounding=torch.zeros_like(grounded.dynamics_grounding),
        belief_statistics=torch.zeros_like(grounded.belief_statistics),
        view_valid=torch.zeros_like(grounded.view_valid),
        appearance_available=torch.zeros_like(grounded.appearance_available),
    )


def _exposed_fish_access_record(*, event: str) -> dict[str, Any]:
    if event not in {"evaluation_started", "evaluation_completed"}:
        raise V178TrainingError("unknown episode-2 access event")
    return {
        "unix_time_ns": time.time_ns(),
        "event": event,
        "version": VERSION,
        "dataset": "fish_episode_2",
        "data_role": "frozen_once_exposed_development_evaluation",
        "strict_held_evidence": False,
        "smoke": False,
        "process_id": os.getpid(),
    }


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        count = os.write(descriptor, view)
        if count <= 0:
            raise V178TrainingError("failed to write episode-2 access record")
        view = view[count:]


def _claim_exposed_fish_access(access_ledger: str | Path) -> Path:
    """Atomically and permanently claim the dataset-level episode-2 access."""

    path = Path(access_ledger).expanduser().resolve()
    if not path.parent.is_dir():
        raise V178TrainingError(
            "episode-2 access-ledger parent must be the existing dataset parent"
        )
    record = _exposed_fish_access_record(event="evaluation_started")
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise V178TrainingError(
            "V17.8 fish episode-2 evaluation was already opened; refusing reselection"
        ) from error
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def _complete_exposed_fish_access(access_ledger: str | Path) -> None:
    path = Path(access_ledger).expanduser().resolve()
    if not path.is_file():
        raise V178TrainingError(
            "episode-2 access was not atomically claimed before completion"
        )
    record = _exposed_fish_access_record(event="evaluation_completed")
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@torch.inference_mode()
def evaluate_v178(
    core: V178Core,
    cad: Any,
    cad_features: Any,
    archive: V181TemporalArchive,
    archive_beliefs: Any,
    fish: V181FishArchive,
    identity: V178GeometryIdentityProvider,
    task_vocabulary: V178TaskVocabulary,
    task_cache: V178TaskEmbeddingCache,
    output_root: str | Path,
    *,
    device: torch.device,
    vocabulary: V178SourceVocabulary,
    archive_features: Any,
    fish_feature_cache_root: str | Path,
    access_ledger: str | Path,
    batch_size: int = 32,
    probe_rows: int = 2048,
    smoke: bool = False,
    authorize_exposed_development: bool = False,
) -> dict[str, Any]:
    """Run fixed V17.8 mechanism probes without selecting a checkpoint.

    Archive results are labelled training-fit because all 320K rows train the
    Actor.  Fish episode 2 is opened only after explicit authorization and is
    logged durably before any frame is read.
    """

    if authorize_exposed_development is not True:
        raise V178TrainingError(
            "fish episode 2 requires a separate explicitly authorised evaluation"
        )
    if smoke:
        raise V178TrainingError(
            "fish episode-2 cannot be opened by a smoke evaluation"
        )
    wrist_parent = Path(fish.wrist_root).expanduser().resolve().parent
    column_parent = Path(fish.column_root).expanduser().resolve().parent
    expected_ledger = (
        wrist_parent / ".prod-v17-8-fish-episode2-access.jsonl"
    ).resolve()
    if wrist_parent != column_parent or Path(access_ledger).expanduser().resolve() != expected_ledger:
        raise V178TrainingError(
            "episode-2 access ledger is not bound to the paired fish dataset"
        )
    _v177._check_source_vocabulary(core, vocabulary)
    core.eval().to(device)
    output = Path(output_root).expanduser().resolve()
    evaluation = output / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)

    # CAD validation: the optimized hard-set objective plus spatial/calibration
    # metrics.  This split never enters Measurement optimization.
    cad_store: dict[str, list[float | int]] = {}
    cad_rows = np.asarray(cad.validation_rows, dtype=np.int64)
    if smoke:
        cad_rows = cad_rows[:batch_size]
    for selected in _batches(cad_rows, batch_size):
        raw_rgb = np.asarray(cad.rgb[selected], dtype=np.uint8)
        rgb = torch.from_numpy(np.ascontiguousarray(raw_rgb[:, ::-1])).to(device)
        rgb = rgb.permute(0, 1, 4, 2, 3)
        dino, feature_valid = cad_features.get(selected)
        valid = torch.as_tensor(feature_valid, device=device, dtype=torch.bool)
        measurement = core.measure(
            torch.as_tensor(dino, device=device, dtype=torch.float32),
            valid,
            rgb,
        )
        target_px = torch.from_numpy(
            np.ascontiguousarray(
                np.asarray(cad.keypoints_px[selected], dtype=np.float32)[
                    :, ::-1, 4:6
                ]
            )
        ).to(device)
        target_valid = torch.from_numpy(
            np.ascontiguousarray(
                np.asarray(cad.valid[selected], dtype=np.bool_)[:, ::-1, 4:6]
            )
        ).to(device)
        target = _v177._bilinear_point_belief(target_px, target_valid)
        predicted = measurement.anchor_hypothesis_log_likelihood
        pair_ce = _hard_pair_ce_per_view(predicted, target)
        _accumulate(cad_store, "hard_permutation_pair_cross_entropy", pair_ce, valid)
        epe, span, active = _v177._unordered_epe_per_view(
            predicted, target_px, target_valid
        )
        _accumulate(cad_store, "unordered_epe_px", epe, active)
        _accumulate(cad_store, "predicted_span_px", span, active)
        _accumulate(cad_store, "collapse_below_2px", (span < 2.0).float(), active)
        probability = predicted.float().exp()
        confident = probability[..., :-1].amax(-1).gt(0.5).all(-1)
        _accumulate(
            cad_store,
            "high_confidence_collapse_below_2px",
            ((span < 2.0) & confident).float(),
            active,
        )
        _accumulate(
            cad_store,
            "hypothesis_bhattacharyya_overlap",
            _v177.hypothesis_overlap(predicted),
            valid,
        )
        coverage = _v177.credible_coverage(predicted, target_px, target_valid)
        coverage_active = coverage["active"] & valid[:, :, None]
        for key in (
            "coverage_at_50",
            "coverage_at_90",
            "needed_mass",
            "absent_probability",
        ):
            _accumulate(cad_store, key, coverage[key], coverage_active)

    # Action-conditioned belief dynamics on a deterministic all-archive panel.
    all_rows = all_archive_rows(archive)
    sample_count = min(int(probe_rows), len(all_rows))
    sampled = all_rows[
        np.linspace(0, len(all_rows) - 1, sample_count, dtype=np.int64)
    ]
    if smoke:
        sampled = sampled[:batch_size]
    row_source = vocabulary.row_index(archive)
    dynamics_store: dict[str, list[float | int]] = {}
    for selected in _batches(sampled, batch_size):
        endpoint, commands_np, paths_np = _v177._archive_chain_arrays(
            archive, selected
        )
        grounded = _v177._grounded_from_raw(
            core,
            *_v177._cache_tensors(archive_beliefs, endpoint[:, 0], device),
            track_direct=_v177._anonymous_track_direct(
                archive, selected, device=device
            ),
        )
        commands_all = _tensor(commands_np, device)
        paths_all = _tensor(paths_np, device)
        q = paths_all[:, 0, 0]
        source_index = _v177._source_tensor(
            row_source, selected, device=device
        )
        recent_deltas = None
        recent_errors = None
        for horizon in range(TRANSITIONS):
            commands = commands_all[:, horizon]
            path = paths_all[:, horizon]
            target_log, target_valid = _v177._cache_tensors(
                archive_beliefs, endpoint[:, horizon + 1], device
            )
            factual = _v177._predict(
                core,
                grounded,
                q,
                commands,
                recent_deltas,
                recent_errors,
                source_index=source_index,
            )
            hold_commands = q[:, None].expand(-1, COMMAND_STEPS, -1)
            hold = _v177._predict(
                core,
                grounded,
                q,
                hold_commands,
                recent_deltas,
                recent_errors,
                source_index=source_index,
            )
            shuffled_commands = _absolute_bound_action_shuffle(q, commands, core)
            shuffled = _v177._predict(
                core,
                grounded,
                q,
                shuffled_commands,
                recent_deltas,
                recent_errors,
                source_index=source_index,
            )
            conditions = {
                "factual": factual.prior_log_belief,
                "persistence": grounded.log_posterior,
                "hold": hold.prior_log_belief,
                "legal_action_shuffle": shuffled.prior_log_belief,
            }
            for name, prior in conditions.items():
                value, active = _v177._exchangeable_soft_target_ce_per_view(
                    prior, target_log.exp(), target_valid
                )
                _accumulate(dynamics_store, name, value, active)
            ordered, _ = _v177.identity_or_track_assignment(
                target_log, factual.prior_log_belief, target_valid
            )
            posterior = _v177.fuse_measurement_prior(
                factual.prior_log_belief, ordered, target_valid
            )
            grounded = _v177._grounded_from_ordered(
                core, posterior, target_valid
            )
            recent_deltas, recent_errors = _v177._completed_readback_feedback(
                commands, path
            )
            q = path[:, -1]

    # Task/history causal attribution on the exact same archive contexts.
    actor_store: dict[str, list[float | int]] = {}
    sensitivity_store: dict[str, list[float | int]] = {}
    task_shuffle_changed = 0
    task_shuffle_total = 0
    for horizon in range(TRANSITIONS):
        for selected in _batches(sampled, batch_size):
            scene, update, q, history = _v177._archive_cached_actor_context(
                core,
                archive,
                archive_beliefs,
                archive_features,
                selected,
                horizon,
                device=device,
                row_source=row_source,
                collect_history=True,
            )
            target = _tensor(_archive_target(archive, selected, horizon), device)
            task = task_cache.for_rows(selected, device=device)
            task_id = task_vocabulary.indices_for_rows(selected)
            if task_vocabulary.size < 2:
                raise V178TrainingError("task shuffle needs at least two unique texts")
            donor_id = (task_id + 1) % task_vocabulary.size
            shuffled_task = task_cache.for_indices(donor_id, device=device)
            task_shuffle_changed += int(
                (shuffled_task - task).abs().amax(dim=-1).gt(1.0e-6).sum().cpu()
            )
            task_shuffle_total += len(task)
            variants = {
                "full": (scene, update.grounded, update.innovation.innovation, history, task),
                "task_blank": (
                    scene,
                    update.grounded,
                    update.innovation.innovation,
                    history,
                    torch.zeros_like(task),
                ),
                "task_shuffle": (
                    scene,
                    update.grounded,
                    update.innovation.innovation,
                    history,
                    shuffled_task,
                ),
                "history_zero": (
                    scene,
                    update.grounded,
                    update.innovation.innovation,
                    _history_blank(history),
                    task,
                ),
                "history_reverse": (
                    scene,
                    update.grounded,
                    update.innovation.innovation,
                    _history_reverse(history),
                    task,
                ),
                "history_shuffle": (
                    scene,
                    update.grounded,
                    update.innovation.innovation,
                    _history_shuffle(history),
                    task,
                ),
                "innovation_zero": (
                    scene,
                    update.grounded,
                    torch.zeros_like(update.innovation.innovation),
                    history,
                    task,
                ),
                "q_only": (
                    torch.zeros_like(scene),
                    _q_only_grounded(update.grounded),
                    torch.zeros_like(update.innovation.innovation),
                    _history_blank(history),
                    torch.zeros_like(task),
                ),
            }
            outputs: dict[str, torch.Tensor] = {}
            for name, (scene_i, grounded_i, innovation_i, history_i, task_i) in variants.items():
                outputs[name] = core.act(
                    scene_i,
                    grounded_i,
                    innovation_i,
                    q,
                    history=history_i,
                    task_embedding=task_i,
                ).commands
                _accumulate(
                    actor_store,
                    name,
                    _commands_mae_per_example(outputs[name], target),
                )
            repeat = q[:, None].expand(-1, COMMAND_STEPS, -1)
            _accumulate(
                actor_store, "repeat", _commands_mae_per_example(repeat, target)
            )
            for name in (
                "task_blank",
                "task_shuffle",
                "history_zero",
                "history_reverse",
                "history_shuffle",
                "innovation_zero",
                "q_only",
            ):
                _accumulate(
                    sensitivity_store,
                    f"full_vs_{name}_action_delta_calibrated_position",
                    (outputs["full"] - outputs[name]).abs().mean(dim=(1, 2)),
                )

    full_cache_root = Path(fish_feature_cache_root).expanduser().resolve()
    try:
        full_cache_config = json.loads(
            (full_cache_root / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise V178TrainingError(
            "complete full fish DINO cache is required before consuming episode 2"
        ) from error
    if (
        full_cache_config.get("kind") != "fish"
        or not bool(full_cache_config.get("complete"))
    ):
        raise V178TrainingError(
            "complete full fish DINO cache is required before consuming episode 2"
        )
    ledger = _claim_exposed_fish_access(access_ledger)
    fish_cache = V181FeatureCache(full_cache_root, expected_kind="fish")
    if not bool(
        np.array_equal(
            np.asarray(fish_cache.endpoint_index, dtype=np.int64),
            np.arange(fish.frames, dtype=np.int64),
        )
    ):
        raise V178TrainingError("full fish DINO cache endpoint clock differs")
    fish_features: Any = (
        _v181.V181DeviceFeatureBank(
            fish_cache, device=device, max_bytes=4 * 1024**3
        )
        if device.type == "cuda"
        else fish_cache
    )
    fish_source = V178FishMeasurementSource(fish, fish_features)
    fish_v178 = as_v178_fish_archive(fish)
    held = fish_v178.pair_ordinals("held")
    if smoke:
        held = held[:batch_size]
    fish_store: dict[str, list[float | int]] = {}
    for horizon in range(TRANSITIONS):
        horizon_ordinals = held[held % TRANSITIONS == horizon]
        for selected in _batches(horizon_ordinals, batch_size):
            pair, scene, update, q, history = _v177._fish_actor_context(
                core,
                identity,
                fish_v178,
                fish_source,
                selected,
                device=device,
                partition="held",
                vocabulary=vocabulary,
                collect_history=True,
            )
            task = task_cache.fish(len(q), device=device)
            target = _tensor(_v177._fish_commands_degrees(pair), device)
            fish_variants = {
                "full": (history, task),
                "task_blank": (history, torch.zeros_like(task)),
                "task_shuffle": (
                    history,
                    task_cache.fish_donor(len(q), device=device),
                ),
                "history_zero": (_history_blank(history), task),
                "history_reverse": (_history_reverse(history), task),
                "history_shuffle": (_history_shuffle(history), task),
            }
            for name, (history_i, task_i) in fish_variants.items():
                commands = core.act(
                    scene,
                    update.grounded,
                    update.innovation.innovation,
                    q,
                    history=history_i,
                    task_embedding=task_i,
                ).commands
                _accumulate(
                    fish_store,
                    name,
                    _commands_mae_per_example(commands, target),
                )
            innovation_zero = core.act(
                scene,
                update.grounded,
                torch.zeros_like(update.innovation.innovation),
                q,
                history=history,
                task_embedding=task,
            ).commands
            _accumulate(
                fish_store,
                "innovation_zero",
                _commands_mae_per_example(innovation_zero, target),
            )
            repeat = q[:, None].expand(-1, COMMAND_STEPS, -1)
            _accumulate(
                fish_store, "repeat", _commands_mae_per_example(repeat, target)
            )

    result = {
        "version": VERSION,
        "selection_role": "descriptive_only_no_checkpoint_selection",
        "cad_validation": {
            "rows": int(len(cad_rows)),
            "metrics": _summarize(cad_store),
        },
        "archive_belief_dynamics_training_fit": {
            "rows": int(len(sampled)),
            "all_320k_were_training": True,
            "action_shuffle": "swap_deltas_then_reanchor_and_absolute_bound_only",
            "metrics": _summarize(dynamics_store),
        },
        "archive_actor_training_fit_attribution": {
            "rows": int(len(sampled)),
            "all_horizons": True,
            "first_five_mae_calibrated_position": _summarize(actor_store),
            "prediction_output": "deployment_absolute_bound_clipped_commands",
            "target_output": "raw_source_calibrated_position_commands",
            "comparable_to_unprojected_training_loss": False,
            "action_sensitivity": _summarize(sensitivity_store),
            "task_shuffle_changed_fraction": float(
                task_shuffle_changed / max(1, task_shuffle_total)
            ),
            "metric_reduction": "strict_per_example_sum_count",
        },
        "fish_episode2_exposed_development": {
            "rows": int(len(held)),
            "first_five_mae_calibrated_position": _summarize(fish_store),
            "prediction_output": "deployment_absolute_bound_clipped_commands",
            "target_output": "raw_source_calibrated_position_commands",
            "comparable_to_unprojected_training_loss": False,
            "task_text": FISH_TASK_TEXT,
            "metric_reduction": "strict_per_example_sum_count",
            "strict_held_evidence": False,
        },
        "acceptance_checks": {
            "measurement_collapse_reported": True,
            "lewm_factual_persistence_hold_shuffle_reported": True,
            "actor_repeat_task_history_q_only_reported": True,
        },
    }
    _write_json(evaluation / "metrics.json", result)
    _complete_exposed_fish_access(ledger)
    _merge_metrics(output / "train-metrics.json", "evaluation", result)
    return result


__all__ = [
    "ACTOR_ARCHIVE_PASSES",
    "ACTOR_FISH_REPLAY_UPDATES",
    "ARCHITECTURE_CONTRACT",
    "CAD_PASSES",
    "FISH_TASK_TEXT",
    "REAL_REFIT_UPDATES",
    "SEED",
    "TASK_EMBEDDING_DIM",
    "TASK_ENCODER",
    "VERSION",
    "V178ActorLoss",
    "V178BeliefCache",
    "V178CanonicalRealRGB",
    "V178CombinedBeliefSource",
    "V178CombinedFeatureSource",
    "V178DenseMeasurementSource",
    "V178DeviceBeliefBank",
    "V178FishArchive",
    "V178FishMeasurementSource",
    "V178GeometryIdentityProvider",
    "V178SourceVocabulary",
    "V178TaskEmbeddingCache",
    "V178TaskVocabulary",
    "V178TrainingError",
    "actor_action_loss",
    "actor_archive_exposure_contract",
    "all_archive_rows",
    "as_v178_fish_archive",
    "compute_joint_delta_loss_scale",
    "configure_cuda",
    "dense_endpoint_map",
    "evaluate_v178",
    "fish_training_frame_indices",
    "load_component",
    "prepare_belief_cache",
    "prepare_task_embedding_cache",
    "prepare_train_fish_dino_cache",
    "require_physical_gpu1",
    "save_component",
    "seed_v178",
    "train_actor",
    "train_lewm",
    "train_measurement",
]
