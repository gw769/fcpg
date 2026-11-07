"""Existing-cache-only data contracts for Prod-V18.1.

The 320K archive loader opens endpoint identities, view validity, factual
commands, and measured paths.  It never opens the legacy visual-token array.
The fish loader uses all 2,183 cached frames and constructs the fixed 30 Hz to
10 Hz command clock without a hindsight image or task identifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np
import torch

from fcpg.lerobot_archive import video_frame_request_view
from fcpg.v21_encoder import DIM as ENCODER_DIM, ENCODER, PATCHES as ENCODER_PATCHES
from fcpg.prod_v18_1 import (
    COMMAND_STEPS,
    PATH_STEPS,
    STATE_DIM,
    VIEW_ORDER,
    VIEWS,
    VIT_DIM,
    VIT_PATCHES,
    project_v181_action_trajectory,
    v181_default_action_contract,
)


OBSERVATIONS = 5
TRANSITIONS = 4
FORMAL_ROWS = 320_000
FORMAL_TRAIN_ROWS = 272_000
FORMAL_DEV_ROWS = 48_000
DENSE_SUBSET_ROWS = 20_000
DENSE_SUBSET_TRAIN_ROWS = 17_000
DENSE_SUBSET_DEV_ROWS = 3_000
FISH_FRAMES = 2_183
FISH_TRAIN_EPISODES = (0, 1, 3)
FISH_HELD_EPISODES = (2,)
FISH_ENDPOINT_STRIDE = 15
FISH_COMMAND_STRIDE = 3
FISH_SEQUENCE_STRIDE = 3
FISH_SEQUENCE_SPAN = TRANSITIONS * FISH_ENDPOINT_STRIDE

Partition = Literal["train", "dev"]
FishPartition = Literal["train", "held"]


class V181DataContractError(ValueError):
    """Raised before a split, view, or clock mismatch reaches training."""


@dataclass(frozen=True)
class V181SourceExposureAudit:
    """Ordinary source/task overlap accounting for held-policy claims."""

    fish_dataset_id: str
    fish_source_name: str
    fish_tasks: tuple[str, ...]
    temporal_train_episodes: int
    temporal_dev_episodes: int
    exact_source_train_episodes: int
    exact_source_dev_episodes: int
    exact_task_train_episodes: int
    exact_task_dev_episodes: int

    @property
    def fish_source_seen_in_world_train(self) -> bool:
        return bool(self.exact_source_train_episodes or self.exact_task_train_episodes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fish_dataset_id": self.fish_dataset_id,
            "fish_source_name": self.fish_source_name,
            "fish_tasks": list(self.fish_tasks),
            "temporal_train_episodes": self.temporal_train_episodes,
            "temporal_dev_episodes": self.temporal_dev_episodes,
            "exact_source_train_episodes": self.exact_source_train_episodes,
            "exact_source_dev_episodes": self.exact_source_dev_episodes,
            "exact_task_train_episodes": self.exact_task_train_episodes,
            "exact_task_dev_episodes": self.exact_task_dev_episodes,
            "fish_source_seen_in_world_train": self.fish_source_seen_in_world_train,
            "held_episode_claim": (
                "policy-held-only"
                if self.fish_source_seen_in_world_train
                else "source-unseen-within-the-audited-320k-manifest"
            ),
        }


def _root(value: str | Path, *, label: str) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise V181DataContractError(f"{label} directory is absent: {root}")
    return root


def _array(root: Path, name: str) -> np.ndarray:
    path = root / f"{name}.npy"
    if not path.is_file():
        raise V181DataContractError(f"required array is absent: {path}")
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise V181DataContractError(f"cannot open array: {path}") from error


def _shape(
    name: str,
    value: np.ndarray,
    expected: tuple[int, ...],
    *,
    dtype: np.dtype[Any] | type[np.generic] | None = None,
) -> np.ndarray:
    if tuple(value.shape) != expected:
        raise V181DataContractError(
            f"{name} shape drifted: observed={tuple(value.shape)} expected={expected}"
        )
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise V181DataContractError(
            f"{name} dtype drifted: observed={value.dtype} expected={np.dtype(dtype)}"
        )
    return value


def _finite(name: str, value: np.ndarray) -> None:
    if not bool(np.isfinite(value).all()):
        raise V181DataContractError(f"{name} contains non-finite values")


def _indices(value: Any, *, upper: int, label: str) -> np.ndarray:
    selected = np.asarray(value, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise V181DataContractError(f"{label} must be non-empty int64 [N]")
    if bool(((selected < 0) | (selected >= int(upper))).any()):
        raise V181DataContractError(f"{label} leaves its population")
    return selected


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V181DataContractError(f"{label} is malformed: {path}") from error
    if not isinstance(payload, dict):
        raise V181DataContractError(f"{label} must be a JSON object")
    return payload


@dataclass(frozen=True)
class V181TemporalBatch:
    rows: np.ndarray
    transition: int
    source_endpoint_index: np.ndarray
    target_endpoint_index: np.ndarray
    source_view_valid: np.ndarray
    target_view_valid: np.ndarray
    current_q: np.ndarray
    commands: np.ndarray
    measured_q_path: np.ndarray


@dataclass(frozen=True)
class V181TemporalActorBatch:
    """Strictly causal Actor examples from ordinary 320K transition arrays."""

    ordinal: np.ndarray
    row: np.ndarray
    transition: np.ndarray
    source_endpoint_index: np.ndarray
    source_view_valid: np.ndarray
    history_source_endpoint_index: np.ndarray
    history_target_endpoint_index: np.ndarray
    history_valid: np.ndarray
    current_q: np.ndarray
    target_commands: np.ndarray
    history_commands: np.ndarray
    history_paths: np.ndarray


class V181TemporalArchive:
    """TRAIN/DEV identities and factual transitions from the 320K archive."""

    def __init__(
        self,
        endpoint_root: str | Path,
        companion_root: str | Path,
        temporal_root: str | Path,
        *,
        expected_rows: int | None = FORMAL_ROWS,
        expected_endpoints: int | None = 572_521,
    ) -> None:
        self.endpoint_root = _root(endpoint_root, label="endpoint cache")
        self.companion_root = _root(companion_root, label="transition companion")
        self.temporal_root = _root(temporal_root, label="temporal index")
        endpoint_config = _load_json(
            self.endpoint_root / "manifest.json", label="endpoint cache manifest"
        )
        if tuple(endpoint_config.get("view_order", ())) != VIEW_ORDER:
            raise V181DataContractError(
                f"endpoint cache view order must be {VIEW_ORDER}"
            )

        self.row_endpoint_index = _array(self.endpoint_root, "row_endpoint_index")
        rows = len(self.row_endpoint_index)
        if expected_rows is not None and rows != int(expected_rows):
            raise V181DataContractError(
                f"row count drifted: observed={rows} expected={expected_rows}"
            )
        _shape(
            "row_endpoint_index",
            self.row_endpoint_index,
            (rows, OBSERVATIONS),
            dtype=np.int32,
        )
        self.split = _shape(
            "split", _array(self.endpoint_root, "split"), (rows,), dtype=np.uint8
        )
        if bool((~np.isin(self.split, (0, 1))).any()):
            raise V181DataContractError("320K split must contain only TRAIN/DEV")
        self.chain_id = _shape(
            "endpoint chain_id",
            _array(self.endpoint_root, "chain_id"),
            (rows,),
            dtype=np.int64,
        )
        self.sidecar_row = _shape(
            "sidecar_row",
            _array(self.endpoint_root, "sidecar_row"),
            (rows,),
            dtype=np.int32,
        )
        if len(np.unique(self.sidecar_row)) != rows:
            raise V181DataContractError("320K sidecar rows must be unique")
        self.endpoint_episode_index = _array(self.endpoint_root, "episode_index")
        self.endpoint_local_frame_index = _array(
            self.endpoint_root, "local_frame_index"
        )
        self.endpoint_view_valid = _array(self.endpoint_root, "view_valid")
        endpoints = len(self.endpoint_episode_index)
        if expected_endpoints is not None and endpoints != int(expected_endpoints):
            raise V181DataContractError(
                f"endpoint count drifted: observed={endpoints} expected={expected_endpoints}"
            )
        _shape(
            "endpoint episode_index",
            self.endpoint_episode_index,
            (endpoints,),
            dtype=np.int32,
        )
        _shape(
            "endpoint local_frame_index",
            self.endpoint_local_frame_index,
            (endpoints,),
            dtype=np.int32,
        )
        _shape(
            "endpoint view_valid",
            self.endpoint_view_valid,
            (endpoints, VIEWS),
            dtype=np.bool_,
        )
        if bool((self.row_endpoint_index < 0).any()) or int(
            self.row_endpoint_index.max()
        ) >= endpoints:
            raise V181DataContractError("row endpoint identity leaves endpoint bank")
        if bool((~self.endpoint_view_valid).all(axis=1).any()):
            raise V181DataContractError("every archived endpoint needs one valid view")

        companion_chain = _shape(
            "companion chain_id",
            _array(self.companion_root, "chain_id"),
            (rows,),
            dtype=np.int64,
        )
        if not bool(np.array_equal(companion_chain, self.chain_id)):
            raise V181DataContractError("endpoint and transition companion clocks differ")
        self.completed_commands = _shape(
            "completed_commands",
            _array(self.companion_root, "completed_commands"),
            (rows, TRANSITIONS, COMMAND_STEPS, STATE_DIM),
            dtype=np.float32,
        )
        self.measured_state_path = _shape(
            "measured_state_path",
            _array(self.companion_root, "measured_state_path"),
            (rows, TRANSITIONS, PATH_STEPS, STATE_DIM),
            dtype=np.float32,
        )
        temporal_config = _load_json(
            self.temporal_root / "manifest.json", label="temporal manifest"
        )
        episodes = temporal_config.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise V181DataContractError("temporal manifest has no canonical episodes")
        self.episodes: tuple[Mapping[str, Any], ...] = tuple(episodes)
        if bool((self.endpoint_episode_index < 0).any()) or int(
            self.endpoint_episode_index.max()
        ) >= len(self.episodes):
            raise V181DataContractError("endpoint references an unknown episode")
        if expected_rows == FORMAL_ROWS:
            observed = {
                "train": int((self.split == 0).sum()),
                "dev": int((self.split == 1).sum()),
            }
            if observed != {"train": FORMAL_TRAIN_ROWS, "dev": FORMAL_DEV_ROWS}:
                raise V181DataContractError(
                    f"formal split drifted: observed={observed}"
                )

    @property
    def rows(self) -> int:
        return len(self.split)

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoint_episode_index)

    def rows_for_partition(self, partition: Partition) -> np.ndarray:
        code = 0 if partition == "train" else 1
        return np.flatnonzero(self.split == code).astype(np.int64, copy=False)

    def dense_subset_rows(
        self,
        rgb_subset_root: str | Path,
        *,
        partition: Partition,
    ) -> np.ndarray:
        """Join the existing 20K RGB subset by ordinary sidecar-row identity."""

        root = _root(rgb_subset_root, label="20K RGB subset")
        subset = _shape(
            "20K sidecar_row",
            _array(root, "sidecar_row"),
            (DENSE_SUBSET_ROWS,),
            dtype=np.int32,
        )
        if len(np.unique(subset)) != DENSE_SUBSET_ROWS:
            raise V181DataContractError("20K sidecar rows must be unique")
        positions = np.searchsorted(np.sort(self.sidecar_row), subset)
        sorted_rows = np.sort(self.sidecar_row)
        if bool((positions >= len(sorted_rows)).any()) or not bool(
            np.array_equal(sorted_rows[positions], subset)
        ):
            raise V181DataContractError("20K rows do not belong to the 320K archive")
        selected = np.flatnonzero(np.isin(self.sidecar_row, subset)).astype(
            np.int64, copy=False
        )
        code = 0 if partition == "train" else 1
        selected = selected[self.split[selected] == code]
        expected = (
            DENSE_SUBSET_TRAIN_ROWS if partition == "train" else DENSE_SUBSET_DEV_ROWS
        )
        if len(selected) != expected:
            raise V181DataContractError(
                f"20K {partition} row count drifted: observed={len(selected)} "
                f"expected={expected}"
            )
        return selected

    def actor_pair_ordinals(
        self,
        *,
        partition: Partition = "train",
        rows: np.ndarray | None = None,
        allow_dev: bool = False,
    ) -> np.ndarray:
        if partition == "dev" and not allow_dev:
            raise V181DataContractError("DEV is sealed during Actor optimization")
        population_rows = (
            self.rows_for_partition(partition)
            if rows is None
            else _indices(rows, upper=self.rows, label="Actor rows")
        )
        expected = 0 if partition == "train" else 1
        if bool((self.split[population_rows] != expected).any()):
            raise V181DataContractError("Actor rows cross their declared split")
        return np.arange(len(population_rows) * TRANSITIONS, dtype=np.int64)

    @staticmethod
    def _project_targets(current_q: np.ndarray, commands: np.ndarray) -> np.ndarray:
        cap, low, high = v181_default_action_contract()
        with torch.no_grad():
            projected = project_v181_action_trajectory(
                torch.from_numpy(np.ascontiguousarray(current_q)).float(),
                torch.from_numpy(np.ascontiguousarray(commands)).float(),
                step_cap=cap,
                action_lower=low,
                action_upper=high,
            )
        return projected.numpy().astype(np.float32, copy=False)

    def actor_pair_batch(
        self,
        ordinals: Any,
        *,
        partition: Partition = "train",
        rows: np.ndarray | None = None,
        allow_dev: bool = False,
    ) -> V181TemporalActorBatch:
        """Build H0--H3 behavior-cloning examples without exposing H4 as goal."""

        if partition == "dev" and not allow_dev:
            raise V181DataContractError("DEV is sealed during Actor optimization")
        population_rows = (
            self.rows_for_partition(partition)
            if rows is None
            else _indices(rows, upper=self.rows, label="Actor rows")
        )
        expected = 0 if partition == "train" else 1
        if bool((self.split[population_rows] != expected).any()):
            raise V181DataContractError("Actor rows cross their declared split")
        population = len(population_rows) * TRANSITIONS
        selected = _indices(ordinals, upper=population, label="Actor pair ordinals")
        local_row = selected // TRANSITIONS
        transition = selected % TRANSITIONS
        row = population_rows[local_row]
        batch_index = np.arange(len(selected), dtype=np.int64)
        paths = np.asarray(self.measured_state_path[row], dtype=np.float32)
        raw_commands = np.asarray(self.completed_commands[row], dtype=np.float32)
        current_q = paths[batch_index, transition, 0]
        target = self._project_targets(
            current_q, raw_commands[batch_index, transition]
        )

        history_source = np.zeros((len(selected), 3), dtype=np.int64)
        history_target = np.zeros((len(selected), 3), dtype=np.int64)
        history_valid = np.zeros((len(selected), 3), dtype=np.bool_)
        history_commands = np.zeros(
            (len(selected), 3, COMMAND_STEPS, STATE_DIM), dtype=np.float32
        )
        history_paths = np.zeros(
            (len(selected), 3, PATH_STEPS, STATE_DIM), dtype=np.float32
        )
        endpoint = np.asarray(self.row_endpoint_index[row], dtype=np.int64)
        for slot in range(3):
            active = transition > slot
            history_source[active, slot] = endpoint[active, slot]
            history_target[active, slot] = endpoint[active, slot + 1]
            history_valid[active, slot] = True
            if bool(active.any()):
                # LeWM was trained against the factual completed command
                # stream.  Only the Actor target is canonicalized; completed
                # history remains the actual archive command/readback pair.
                history_commands[active, slot] = raw_commands[active, slot]
                history_paths[active, slot] = paths[active, slot]

        source = endpoint[batch_index, transition]
        # Inactive history entries still need an addressable gather identity;
        # their validity mask keeps the duplicated current endpoint out of all
        # losses and memory.  This avoids requiring endpoint zero in a subset.
        for slot in range(3):
            inactive = ~history_valid[:, slot]
            history_source[inactive, slot] = source[inactive]
            history_target[inactive, slot] = source[inactive]
        return V181TemporalActorBatch(
            ordinal=selected,
            row=row.astype(np.int64, copy=False),
            transition=transition.astype(np.int64, copy=False),
            source_endpoint_index=source,
            source_view_valid=np.asarray(
                self.endpoint_view_valid[source], dtype=np.bool_
            ),
            history_source_endpoint_index=history_source,
            history_target_endpoint_index=history_target,
            history_valid=history_valid,
            current_q=np.asarray(current_q, dtype=np.float32),
            target_commands=target,
            history_commands=history_commands,
            history_paths=history_paths,
        )

    def audit_fish_source_exposure(
        self,
        fish: "V181FishArchive",
    ) -> V181SourceExposureAudit:
        """Count exact source/task exposure without opening pixels or held actions.

        The audit uses ordinary manifest names and task strings. It is a
        reporting boundary, not content matching or checkpoint binding.
        A nonzero count is allowed, but then episode 2 can only be described as
        held from policy optimization.
        """

        source_names = {
            value
            for value in (
                fish.dataset_id.strip().lower(),
                fish.source_name.strip().lower(),
            )
            if value
        }
        tasks = {value.strip().lower() for value in fish.task_names if value.strip()}
        counts = {
            "temporal_train_episodes": 0,
            "temporal_dev_episodes": 0,
            "exact_source_train_episodes": 0,
            "exact_source_dev_episodes": 0,
            "exact_task_train_episodes": 0,
            "exact_task_dev_episodes": 0,
        }
        for episode in self.episodes:
            row = episode.get("manifest_row")
            if not isinstance(row, dict):
                raise V181DataContractError("temporal episode omits its manifest row")
            split = str(row.get("split", ""))
            if split not in {"train", "development"}:
                raise V181DataContractError("temporal manifest contains an unknown split")
            suffix = "train" if split == "train" else "dev"
            counts[f"temporal_{suffix}_episodes"] += 1
            ordinary_sources = {
                str(row.get("dataset_id", "")).strip().lower(),
                Path(str(row.get("dataset_root", ""))).name.strip().lower(),
                str(row.get("dataset_provenance", "")).strip().lower(),
            }
            ordinary_sources.discard("")
            source_match = any(
                fish_name == source or fish_name in source.split("/")
                for fish_name in source_names
                for source in ordinary_sources
            )
            row_tasks = {
                str(value).strip().lower()
                for value in row.get("tasks", ())
                if str(value).strip()
            }
            if source_match:
                counts[f"exact_source_{suffix}_episodes"] += 1
            if tasks.intersection(row_tasks):
                counts[f"exact_task_{suffix}_episodes"] += 1
        return V181SourceExposureAudit(
            fish_dataset_id=fish.dataset_id,
            fish_source_name=fish.source_name,
            fish_tasks=fish.task_names,
            **counts,
        )

    def transition_batch(
        self,
        rows: Any,
        transition: int,
        *,
        partition: Partition = "train",
        allow_dev: bool = False,
    ) -> V181TemporalBatch:
        selected = _indices(rows, upper=self.rows, label="temporal rows")
        if not 0 <= int(transition) < TRANSITIONS:
            raise V181DataContractError("transition must be H0-H1 through H3-H4")
        expected = 0 if partition == "train" else 1
        if partition == "dev" and not allow_dev:
            raise V181DataContractError("DEV is sealed during optimization")
        if bool((self.split[selected] != expected).any()):
            raise V181DataContractError("selected rows cross their declared split")
        source = np.asarray(
            self.row_endpoint_index[selected, int(transition)], dtype=np.int64
        )
        target = np.asarray(
            self.row_endpoint_index[selected, int(transition) + 1], dtype=np.int64
        )
        commands = np.asarray(
            self.completed_commands[selected, int(transition)], dtype=np.float32
        )
        path = np.asarray(
            self.measured_state_path[selected, int(transition)], dtype=np.float32
        )
        _finite("temporal commands", commands)
        _finite("temporal measured path", path)
        return V181TemporalBatch(
            rows=selected,
            transition=int(transition),
            source_endpoint_index=source,
            target_endpoint_index=target,
            source_view_valid=np.asarray(
                self.endpoint_view_valid[source], dtype=np.bool_
            ),
            target_view_valid=np.asarray(
                self.endpoint_view_valid[target], dtype=np.bool_
            ),
            current_q=path[:, 0].copy(),
            commands=commands,
            measured_q_path=path,
        )

    def logical_views(self, episode_index: int) -> tuple[str, str]:
        record = self.episodes[int(episode_index)]
        row = record.get("manifest_row")
        if not isinstance(row, dict):
            raise V181DataContractError("temporal episode omits its manifest row")
        slots = tuple(map(str, row.get("lewm_v12_slot_views", ())))
        if len(slots) != VIEWS or not all(slots) or slots[0] == slots[1]:
            raise V181DataContractError("temporal episode lacks two distinct views")
        # The archive's legacy slot order was [external, wrist].  V18.1 never
        # reads its tokens; only the role declaration is used to decode RGB.
        return slots[1], slots[0]

    def media_requests(
        self, endpoint_indices: Iterable[int]
    ) -> list[tuple[int, int, Path, int, bool]]:
        """Resolve endpoint/view identities to ordinary archived video frames."""

        selected = _indices(
            list(endpoint_indices), upper=self.endpoint_count, label="endpoint indices"
        )
        result: list[tuple[int, int, Path, int, bool]] = []
        for endpoint in selected.tolist():
            episode_index = int(self.endpoint_episode_index[endpoint])
            local_frame = int(self.endpoint_local_frame_index[endpoint])
            record = self.episodes[episode_index]
            row = record.get("manifest_row")
            if not isinstance(row, dict):
                raise V181DataContractError("temporal episode omits its manifest row")
            split = str(row.get("split", ""))
            if split not in {"train", "development"}:
                raise V181DataContractError("endpoint media is outside TRAIN/DEV")
            for view, raw_view in enumerate(self.logical_views(episode_index)):
                valid = bool(self.endpoint_view_valid[endpoint, view])
                path, frame = video_frame_request_view(row, raw_view, local_frame)
                result.append((endpoint, view, path, int(frame), valid))
        return result


@dataclass(frozen=True)
class V181FishPair:
    ordinal: np.ndarray
    episode: np.ndarray
    transition: np.ndarray
    source_frame: np.ndarray
    target_frame: np.ndarray
    history_source_frames: np.ndarray
    history_valid: np.ndarray
    current_q: np.ndarray
    target_commands: np.ndarray
    measured_q_path: np.ndarray
    history_commands: np.ndarray
    history_paths: np.ndarray


class V181FishArchive:
    """All-frame fixed-task fish demonstrations with a strict causal clock."""

    def __init__(
        self,
        wrist_root: str | Path,
        column_root: str | Path,
        *,
        expected_frames: int | None = FISH_FRAMES,
    ) -> None:
        self.wrist_root = _root(wrist_root, label="fish wrist cache")
        self.column_root = _root(column_root, label="fish column cache")
        wrist_config = _load_json(
            self.wrist_root / "manifest.json", label="fish wrist manifest"
        )
        column_config = _load_json(
            self.column_root / "manifest.json", label="fish column manifest"
        )
        if str(wrist_config.get("camera")) != "observation.images.wrist":
            raise V181DataContractError("first fish cache must be wrist")
        if str(column_config.get("camera")) != "observation.images.column":
            raise V181DataContractError("second fish cache must be column")
        for field in ("dataset_id", "source_root", "task_names"):
            if wrist_config.get(field) != column_config.get(field):
                raise V181DataContractError(
                    f"fish dual-view {field} provenance differs"
                )
        self.dataset_id = str(wrist_config.get("dataset_id", "")).strip()
        self.source_name = Path(str(wrist_config.get("source_root", ""))).name.strip()
        raw_tasks = wrist_config.get("task_names", {})
        if not isinstance(raw_tasks, dict):
            raise V181DataContractError("fish task_names must be a mapping")
        self.task_names = tuple(
            sorted(
                str(value).strip()
                for value in raw_tasks.values()
                if str(value).strip()
            )
        )
        if int(wrist_config.get("fps", 0)) != 30 or int(column_config.get("fps", 0)) != 30:
            raise V181DataContractError("fish clock requires 30 Hz cached frames")
        self.wrist_frames = _array(self.wrist_root, "frames")
        self.column_frames = _array(self.column_root, "frames")
        frames = len(self.wrist_frames)
        if expected_frames is not None and frames != int(expected_frames):
            raise V181DataContractError(
                f"fish frame count drifted: observed={frames} expected={expected_frames}"
            )
        _shape(
            "wrist frames", self.wrist_frames, (frames, 224, 224, 3), dtype=np.uint8
        )
        _shape(
            "column frames", self.column_frames, (frames, 224, 224, 3), dtype=np.uint8
        )
        self.episodes = _shape(
            "episodes", _array(self.wrist_root, "episodes"), (frames,), dtype=np.int32
        )
        self.frame_indices = _shape(
            "frame_indices",
            _array(self.wrist_root, "frame_indices"),
            (frames,),
            dtype=np.int32,
        )
        self.states_degrees = _shape(
            "states", _array(self.wrist_root, "states"), (frames, STATE_DIM), dtype=np.float32
        )
        self.actions_degrees = _shape(
            "actions", _array(self.wrist_root, "actions"), (frames, STATE_DIM), dtype=np.float32
        )
        for name in ("episodes", "frame_indices", "states", "actions"):
            other = _array(self.column_root, name)
            reference = {
                "episodes": self.episodes,
                "frame_indices": self.frame_indices,
                "states": self.states_degrees,
                "actions": self.actions_degrees,
            }[name]
            if not bool(np.array_equal(reference, other)):
                raise V181DataContractError(f"fish dual-view {name} clocks differ")
        _finite("fish states", self.states_degrees)
        _finite("fish actions", self.actions_degrees)
        self.states = np.asarray(self.states_degrees, dtype=np.float32) / 100.0
        self.actions = np.asarray(self.actions_degrees, dtype=np.float32) / 100.0
        if bool((self.states < -1.01).any() or (self.states > 1.01).any()):
            raise V181DataContractError("normalized fish state leaves joint contract")
        self._sequences = self._build_sequences()

    @property
    def frames(self) -> int:
        return len(self.episodes)

    def _build_sequences(self) -> np.ndarray:
        rows: list[tuple[int, int]] = []
        for episode in sorted(map(int, np.unique(self.episodes))):
            positions = np.flatnonzero(self.episodes == episode)
            if not bool(np.array_equal(self.frame_indices[positions], np.arange(len(positions)))):
                raise V181DataContractError("fish episode frame clock is not contiguous")
            last_start = len(positions) - FISH_SEQUENCE_SPAN - 1
            for local in range(0, last_start + 1, FISH_SEQUENCE_STRIDE):
                rows.append((episode, int(positions[local])))
        if not rows:
            raise V181DataContractError("fish archive has no complete H0-H4 sequence")
        return np.asarray(rows, dtype=np.int64)

    def sequences_for_partition(self, partition: FishPartition) -> np.ndarray:
        episodes = FISH_TRAIN_EPISODES if partition == "train" else FISH_HELD_EPISODES
        return np.flatnonzero(np.isin(self._sequences[:, 0], episodes)).astype(
            np.int64, copy=False
        )

    def pair_ordinals(self, partition: FishPartition) -> np.ndarray:
        sequences = self.sequences_for_partition(partition)
        return np.arange(len(sequences) * TRANSITIONS, dtype=np.int64)

    @staticmethod
    def _project_targets(current_q: np.ndarray, commands: np.ndarray) -> np.ndarray:
        cap, low, high = v181_default_action_contract()
        with torch.no_grad():
            projected = project_v181_action_trajectory(
                torch.from_numpy(np.ascontiguousarray(current_q)).float(),
                torch.from_numpy(np.ascontiguousarray(commands)).float(),
                step_cap=cap,
                action_lower=low,
                action_upper=high,
            )
        return projected.numpy().astype(np.float32, copy=False)

    def pair_batch(
        self,
        ordinals: Any,
        *,
        partition: FishPartition = "train",
        allow_held: bool = False,
    ) -> V181FishPair:
        sequences = self.sequences_for_partition(partition)
        population = len(sequences) * TRANSITIONS
        selected = _indices(ordinals, upper=population, label="fish pair ordinals")
        if partition == "held" and not allow_held:
            raise V181DataContractError("held fish episode is sealed during optimization")
        local_sequence = selected // TRANSITIONS
        transition = selected % TRANSITIONS
        sequence_rows = self._sequences[sequences[local_sequence]]
        episode = sequence_rows[:, 0]
        start = sequence_rows[:, 1]
        source = start + transition * FISH_ENDPOINT_STRIDE
        target = source + FISH_ENDPOINT_STRIDE

        command_offsets = np.arange(COMMAND_STEPS, dtype=np.int64) * FISH_COMMAND_STRIDE
        path_offsets = np.arange(PATH_STEPS, dtype=np.int64) * FISH_COMMAND_STRIDE
        raw_commands = self.actions[source[:, None] + command_offsets[None]]
        current_q = self.states[source]
        commands = self._project_targets(current_q, raw_commands)
        measured_path = self.states[source[:, None] + path_offsets[None]]

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
                raw = self.actions[past_source + command_offsets]
                history_commands[row, slot] = self._project_targets(
                    self.states[past_source : past_source + 1], raw[None]
                )[0]
                history_paths[row, slot] = self.states[past_source + path_offsets]
        return V181FishPair(
            ordinal=selected,
            episode=episode.astype(np.int64, copy=False),
            transition=transition.astype(np.int64, copy=False),
            source_frame=source.astype(np.int64, copy=False),
            target_frame=target.astype(np.int64, copy=False),
            history_source_frames=history_source,
            history_valid=history_valid,
            current_q=np.asarray(current_q, dtype=np.float32),
            target_commands=commands,
            measured_q_path=np.asarray(measured_path, dtype=np.float32),
            history_commands=history_commands,
            history_paths=history_paths,
        )

    def rgb(self, frame_indices: Any) -> np.ndarray:
        selected = np.asarray(frame_indices, dtype=np.int64)
        if selected.size == 0 or bool(((selected < 0) | (selected >= self.frames)).any()):
            raise V181DataContractError("fish RGB indices leave the frame cache")
        flattened = selected.reshape(-1)
        rgb = np.stack(
            (
                np.asarray(self.wrist_frames[flattened], dtype=np.uint8),
                np.asarray(self.column_frames[flattened], dtype=np.uint8),
            ),
            axis=1,
        )
        return rgb.reshape(*selected.shape, VIEWS, 224, 224, 3)


def deterministic_fish_view_valid(
    identities: Any, *, seed: int
) -> np.ndarray:
    """Fixed 70% both, 15% wrist-only, 15% column-only sampling."""

    values = np.asarray(identities, dtype=np.int64)
    if values.ndim != 1:
        raise V181DataContractError("view sampling identities must be int64 [N]")
    bucket = (values * 37 + int(seed) * 17 + 11) % 100
    valid = np.ones((len(values), VIEWS), dtype=np.bool_)
    valid[(bucket >= 70) & (bucket < 85), 1] = False
    valid[bucket >= 85, 0] = False
    return valid


class V181FeatureCache:
    """Plain fp16 DINOv2 patch-token cache addressed by endpoint identity."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_kind: str | None = None,
    ) -> None:
        self.root = _root(root, label="V18.1 feature cache")
        config = _load_json(self.root / "config.json", label="feature cache config")
        if tuple(config.get("view_order", ())) != VIEW_ORDER:
            raise V181DataContractError("feature cache view order drifted")
        if str(config.get("backbone")) != ENCODER:
            raise V181DataContractError(f"feature cache requires the process encoder {ENCODER} (V21_ENCODER)")
        self.kind = str(config.get("kind", ""))
        if expected_kind is not None and self.kind != expected_kind:
            raise V181DataContractError(
                f"feature cache kind is {self.kind}, expected {expected_kind}"
            )
        self.endpoint_index = _array(self.root, "endpoint_index")
        self.tokens = _array(self.root, "tokens")
        self.view_valid = _array(self.root, "view_valid")
        rows = len(self.endpoint_index)
        _shape(
            "feature endpoint_index", self.endpoint_index, (rows,), dtype=np.int64
        )
        _shape(
            "feature tokens",
            self.tokens,
            (rows, VIEWS, ENCODER_PATCHES, ENCODER_DIM),
            dtype=np.float16,
        )
        _shape(
            "feature view_valid", self.view_valid, (rows, VIEWS), dtype=np.bool_
        )
        if rows == 0 or bool((np.diff(self.endpoint_index) <= 0).any()):
            raise V181DataContractError("feature endpoint identities must be sorted unique")

    def rows_for(self, endpoint_indices: Any) -> np.ndarray:
        requested = np.asarray(endpoint_indices, dtype=np.int64)
        positions = np.searchsorted(self.endpoint_index, requested)
        safe = np.minimum(positions, len(self.endpoint_index) - 1)
        if bool((positions >= len(self.endpoint_index)).any()) or not bool(
            np.array_equal(self.endpoint_index[safe], requested)
        ):
            raise V181DataContractError("requested endpoint is absent from feature cache")
        return positions.astype(np.int64, copy=False)

    def get(self, endpoint_indices: Any) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(endpoint_indices, dtype=np.int64)
        rows = self.rows_for(requested.reshape(-1))
        tokens = np.asarray(self.tokens[rows], dtype=np.float32).reshape(
            *requested.shape, VIEWS, ENCODER_PATCHES, ENCODER_DIM
        )
        valid = np.asarray(self.view_valid[rows], dtype=np.bool_).reshape(
            *requested.shape, VIEWS
        )
        _finite("feature tokens", tokens)
        invalid = ~valid
        if bool((tokens * invalid[..., None, None]).any()):
            raise V181DataContractError("invalid feature views must be exact zero")
        return tokens, valid


__all__ = [
    "FISH_HELD_EPISODES",
    "FISH_TRAIN_EPISODES",
    "OBSERVATIONS",
    "TRANSITIONS",
    "V181DataContractError",
    "V181FeatureCache",
    "V181FishArchive",
    "V181FishPair",
    "V181SourceExposureAudit",
    "V181TemporalArchive",
    "V181TemporalActorBatch",
    "V181TemporalBatch",
    "deterministic_fish_view_valid",
]
