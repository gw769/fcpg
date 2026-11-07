"""Design and materialize the local fish-to-box actor dataset.

The four cleaned demos are full go-and-return cycles. Their episode terminals
are folded home, so a V11.5 ``episode_terminal_state`` goal trained on the raw
episodes learns "go home". This module cuts each demo into approach/grasp
slices whose terminals stay in the workspace.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from v10.data import V10SourceSpec, WRIST_CAMERA_ROLE, EXTERNAL_CAMERA_ROLE
from v10.task_class import (
    FISH_TO_BOX,
    require_catalog_task_class,
    require_source_task_class,
)

TASK_NAME = "Pick up the red fish toy and place it in the cardboard box"
DATASET_ID = "fish_to_box_task_v1"
TASK_CLASS = FISH_TO_BOX
SOURCE_DATASET_ID = "my_first_task_clean_v1"
SOURCE_TRAIN_EPISODES = (0, 1, 3)
SOURCE_VAL_EPISODES = (2,)
HOME_LIFT_DEG = -80.0
WORKSPACE_LIFT_DEG = -50.0
HISTORY_PAD = 5
MIN_SLICE_FRAMES = 26
MIN_WORKSPACE_FRAMES = 20
CLOSE_DROP_DEG = 8.0
CLOSE_LOOKAHEAD = 40
GRASP_LOOKBACK = 40
GRASP_HOLD = 15
PARK_STILL_DEG = 1.0
PARK_PHASES = ("leading_park", "trailing_park")
CATALOG_FORMAT = "so101-fish-to-box-task-v1"
FISH_LETTERBOX_PROJECTION_DIRNAME = "fish_to_box_task_v1_dual_view_letterbox_v1"


@dataclass(frozen=True)
class TaskSlice:
    slice_id: int
    source_episode: int
    phase: str
    split: str
    start: int
    end: int
    terminal_local: int
    frames: int
    gripper_min: float
    terminal_state: tuple[float, ...]
    closes: bool
    task_class: int = TASK_CLASS


def _first_true(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    if not hits.size:
        return None
    return int(hits[0])


def _return_home_index(lift: np.ndarray, t_leave: int) -> int:
    seen = 0
    for index in range(t_leave, len(lift)):
        if float(lift[index]) > HOME_LIFT_DEG:
            seen += 1
        elif seen >= MIN_WORKSPACE_FRAMES:
            return index
    return len(lift)


def joint_range_deg(states: np.ndarray) -> float:
    if not len(states):
        return 0.0
    span = np.asarray(states, dtype=np.float32)
    return float(np.max(span.max(axis=0) - span.min(axis=0)))


def _still_prefix(states: np.ndarray) -> int:
    """Frames the arm sat parked before the demonstrator started moving it.

    Grown on the running joint range so the returned span satisfies the same
    ``joint_range_deg <= PARK_STILL_DEG`` test the catalog is validated against.
    """

    if not len(states):
        return 0
    span = np.maximum.accumulate(states, axis=0) - np.minimum.accumulate(
        states, axis=0
    )
    moved = np.flatnonzero(np.max(span, axis=1) > PARK_STILL_DEG)
    return int(moved[0]) if moved.size else len(states)


def _still_suffix(states: np.ndarray) -> int:
    if not len(states):
        return len(states)
    return len(states) - _still_prefix(states[::-1])


def _last_close_event(
    grip: np.ndarray, start: int, stop: int
) -> tuple[int, int] | None:
    last: tuple[int, int] | None = None
    if stop - start < 10:
        return None
    for index in range(start, stop - 5):
        ahead = grip[index : min(index + CLOSE_LOOKAHEAD, stop)]
        drop = float(ahead[0] - ahead.min())
        if drop >= CLOSE_DROP_DEG:
            last = (index, index + int(np.argmin(ahead)))
    if last is None:
        return None
    close_start, settle = last
    if float(grip[settle]) > float(np.quantile(grip[start:stop], 0.35)):
        return None
    return close_start, settle


def segment_episode(states: np.ndarray, *, source_episode: int) -> list[TaskSlice]:
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError("episode states must have shape [T, 6]")
    lift = np.asarray(states[:, 1], dtype=np.float32)
    grip = np.asarray(states[:, 5], dtype=np.float32)
    t_leave = _first_true(lift > HOME_LIFT_DEG)
    if t_leave is None:
        return []
    t_return = _return_home_index(lift, t_leave)
    start = max(0, t_leave - HISTORY_PAD)
    close = _last_close_event(grip, t_leave, t_return)
    split = "val" if source_episode in SOURCE_VAL_EPISODES else "train"
    raw: list[tuple[str, int, int, bool]] = []
    workspace = [
        index
        for index in range(t_leave, t_return)
        if float(lift[index]) > WORKSPACE_LIFT_DEG
    ]
    last_workspace = (workspace[-1] + 1) if workspace else t_return
    if close is not None:
        close_start, settle = close
        grasp_end = min(last_workspace, t_return, settle + GRASP_HOLD)
        if close_start - start >= MIN_SLICE_FRAMES:
            raw.append(("approach", start, close_start, False))
        grasp_begin = max(start, close_start - GRASP_LOOKBACK)
        if (
            grasp_end - grasp_begin >= MIN_SLICE_FRAMES
            and float(lift[grasp_end - 1]) > WORKSPACE_LIFT_DEG
        ):
            raw.append(("grasp", grasp_begin, grasp_end, True))
    elif last_workspace - start >= MIN_SLICE_FRAMES:
        raw.append(("approach", start, last_workspace, False))

    # The parked head and tail of every demo. Their terminal is the pose they
    # are already sitting in, so they supervise "the goal is where you are, stay
    # there" -- not "go home", which is why the travelling return_home segment
    # stays dropped. Without them nothing in the task teaches holding still, and
    # the actor loses to repeating its own state on exactly those windows.
    lead_end = min(_still_prefix(states), start)
    if lead_end >= MIN_SLICE_FRAMES:
        raw.append(("leading_park", 0, lead_end, False))
    trail_start = max(_still_suffix(states), t_return)
    if len(states) - trail_start >= MIN_SLICE_FRAMES:
        raw.append(("trailing_park", trail_start, len(states), False))

    slices: list[TaskSlice] = []
    for phase, begin, end, closes in raw:
        terminal = end - 1
        slices.append(
            TaskSlice(
                slice_id=-1,
                source_episode=source_episode,
                phase=phase,
                split=split,
                start=begin,
                end=end,
                terminal_local=terminal,
                frames=end - begin,
                gripper_min=float(np.min(grip[begin:end])),
                terminal_state=tuple(
                    round(float(value), 3) for value in states[terminal]
                ),
                closes=closes,
            )
        )
    return slices


def catalog_from_states(
    episodes: np.ndarray, states: np.ndarray
) -> list[TaskSlice]:
    slices: list[TaskSlice] = []
    for source_episode in sorted(int(value) for value in np.unique(episodes)):
        rows = np.flatnonzero(episodes == source_episode)
        for item in segment_episode(states[rows], source_episode=source_episode):
            slices.append(
                TaskSlice(
                    slice_id=len(slices),
                    source_episode=item.source_episode,
                    phase=item.phase,
                    split=item.split,
                    start=int(rows[item.start]),
                    end=int(rows[item.end - 1]) + 1,
                    terminal_local=int(rows[item.terminal_local]),
                    frames=item.frames,
                    gripper_min=item.gripper_min,
                    terminal_state=item.terminal_state,
                    closes=item.closes,
                )
            )
    if not slices:
        raise ValueError("fish-to-box task catalog is empty")
    return slices


def validate_catalog(
    slices: list[TaskSlice],
    *,
    n_source_frames: int,
    states: np.ndarray | None = None,
) -> None:
    if any(item.frames < MIN_SLICE_FRAMES for item in slices):
        raise ValueError("a designed slice is shorter than the V11.5 window")
    if {item.source_episode for item in slices if item.split == "train"} != set(
        SOURCE_TRAIN_EPISODES
    ):
        raise ValueError("train slices must come from source episodes 0/1/3")
    if {item.source_episode for item in slices if item.split == "val"} != set(
        SOURCE_VAL_EPISODES
    ):
        raise ValueError("val slices must come from source episode 2")
    if not any(item.phase == "grasp" and item.source_episode == 0 for item in slices):
        raise ValueError("source episode 0 must contribute a grasp slice")
    if not any(item.phase == "grasp" and item.source_episode == 3 for item in slices):
        raise ValueError("source episode 3 must contribute a grasp slice")
    for phase in PARK_PHASES:
        if not any(item.phase == phase for item in slices):
            raise ValueError(f"the design lost its {phase} hold slices")
    for item in slices:
        if not 0 <= item.start < item.end <= n_source_frames:
            raise ValueError(f"slice {item.slice_id} is outside the source cache")
        if item.phase in PARK_PHASES:
            # A slice that never moves cannot supervise go-home, which is the
            # only reason the workspace terminal rule exists. Staying still is
            # the lesson, so stillness is what has to be checked instead.
            if states is None:
                continue
            if joint_range_deg(states[item.start : item.end]) > PARK_STILL_DEG:
                raise ValueError(
                    f"park slice {item.slice_id} moves; it would supervise travel"
                )
            continue
        if item.terminal_state[1] <= WORKSPACE_LIFT_DEG:
            raise ValueError(
                f"slice {item.slice_id} terminal left the workspace "
                f"(shoulder_lift={item.terminal_state[1]})"
            )
        if item.task_class != TASK_CLASS:
            raise ValueError(
                f"slice {item.slice_id} task_class is not {TASK_CLASS}"
            )


def _write_trajectory_cache(
    output: Path,
    *,
    name: str,
    camera: str,
    frames: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    source_root: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "frames.npy", np.asarray(frames, dtype=np.uint8))
    np.save(output / "states.npy", np.asarray(states, dtype=np.float32))
    np.save(output / "actions.npy", np.asarray(actions, dtype=np.float32))
    np.save(output / "episodes.npy", np.asarray(episode_ids, dtype=np.int32))
    np.save(output / "tasks.npy", np.zeros(len(frames), dtype=np.int16))
    np.save(
        output / "frame_indices.npy",
        np.arange(len(frames), dtype=np.int32),
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "format": "so101-lewm-cache-v1",
                "name": name,
                "dataset_id": DATASET_ID,
                "source_root": source_root,
                "source_dataset_id": SOURCE_DATASET_ID,
                "robot_type": "so101_follower",
                "camera": camera,
                "image_size": 224,
                "fps": 30,
                "frames": int(len(frames)),
                "episodes": int(len(np.unique(episode_ids))),
                "tasks": 1,
                "task_names": {"0": TASK_NAME},
                "actor_task_class": TASK_CLASS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_task_dataset(
    *,
    data_root: Path,
    source_name: str = SOURCE_DATASET_ID,
) -> dict[str, object]:
    data_root = data_root.expanduser().resolve()
    wrist = data_root / "cache" / f"{source_name}_wrist_224"
    column = data_root / "cache" / f"{source_name}_column_224"
    geometry = (
        data_root / "predictive_context" / f"{source_name}_wrist_expanded5_10k"
    )
    episodes = np.load(wrist / "episodes.npy")
    states = np.load(wrist / "states.npy")
    actions = np.load(wrist / "actions.npy")
    wrist_frames = np.load(wrist / "frames.npy", mmap_mode="r")
    column_frames = np.load(column / "frames.npy", mmap_mode="r")
    anchors = np.load(geometry / "anchors_xyz.npy", mmap_mode="r")
    if len(wrist_frames) != len(column_frames) or len(states) != len(wrist_frames):
        raise ValueError("source wrist/column caches are not paired")
    if len(anchors) != len(states):
        raise ValueError("source geometry rows do not match the trajectory")

    slices = catalog_from_states(episodes, states)
    validate_catalog(slices, n_source_frames=len(states), states=states)

    keep = np.concatenate(
        [np.arange(item.start, item.end, dtype=np.int64) for item in slices]
    )
    episode_ids = np.concatenate(
        [
            np.full(item.frames, item.slice_id, dtype=np.int32)
            for item in slices
        ]
    )
    out_wrist = data_root / "cache" / f"{DATASET_ID}_wrist_224"
    out_column = data_root / "cache" / f"{DATASET_ID}_column_224"
    out_geometry = data_root / "predictive_context" / f"{DATASET_ID}_wrist_expanded5_10k"
    dataset_dir = data_root / "datasets" / DATASET_ID
    dataset_dir.mkdir(parents=True, exist_ok=True)

    _write_trajectory_cache(
        out_wrist,
        name=f"{DATASET_ID}_wrist",
        camera="observation.images.wrist",
        frames=np.asarray(wrist_frames[keep]),
        states=states[keep],
        actions=actions[keep],
        episode_ids=episode_ids,
        source_root=str(wrist),
    )
    _write_trajectory_cache(
        out_column,
        name=f"{DATASET_ID}_column",
        camera="observation.images.column",
        frames=np.asarray(column_frames[keep]),
        states=states[keep],
        actions=actions[keep],
        episode_ids=episode_ids,
        source_root=str(column),
    )
    out_geometry.mkdir(parents=True, exist_ok=True)
    np.save(out_geometry / "anchors_xyz.npy", np.asarray(anchors[keep]))
    (out_geometry / "manifest.json").write_text(
        json.dumps(
            {
                "format": "so101-predictive-action-context-v1",
                "rows": int(len(keep)),
                "dataset_id": DATASET_ID,
                "source_name": f"{DATASET_ID}_wrist",
                "sliced_from": str(geometry),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    train_ids = [item.slice_id for item in slices if item.split == "train"]
    val_ids = [item.slice_id for item in slices if item.split == "val"]
    catalog = {
        "format": CATALOG_FORMAT,
        "name": DATASET_ID,
        "task_class": TASK_CLASS,
        "task": TASK_NAME,
        "public_mix": False,
        "source_dataset": SOURCE_DATASET_ID,
        "source_train_episodes": list(SOURCE_TRAIN_EPISODES),
        "source_val_episodes": list(SOURCE_VAL_EPISODES),
        "design": {
            "unit": "phase_slice",
            "keep": ["approach", "grasp", *PARK_PHASES],
            "drop": ["return_home"],
            "goal": "slice_terminal_state",
            "why": (
                "raw episode terminals are folded home; V11.5 episode_terminal_state "
                "would otherwise supervise go-home. The parked head and tail are "
                "kept as their own still slices so the goal there is the pose the "
                "arm already holds; only the travelling return is dropped"
            ),
            "home_lift_deg": HOME_LIFT_DEG,
            "workspace_lift_deg": WORKSPACE_LIFT_DEG,
            "min_slice_frames": MIN_SLICE_FRAMES,
            "park_still_deg": PARK_STILL_DEG,
        },
        "slices": [asdict(item) for item in slices],
        "train_slice_ids": train_ids,
        "val_slice_ids": val_ids,
        "frames": int(len(keep)),
        "source_frames": int(len(states)),
        "dropped_frames": int(len(states) - len(keep)),
    }
    (dataset_dir / "catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    source = {
        "format": "so101-v10-local-source-v1",
        "name": DATASET_ID,
        "locked": True,
        "public_mix": False,
        "task_class": TASK_CLASS,
        "task": TASK_NAME,
        "wrist_cache": str(out_wrist),
        "column_cache": str(out_column),
        "geometry": str(out_geometry),
        "train_episodes": train_ids,
        "validation_episodes": val_ids,
        "catalog": str(dataset_dir / "catalog.json"),
        "column_role": "global_rgb_context_only",
        "wrist_role": "fcpg_three_anchor_grounding",
    }
    source_path = data_root / "local_sources" / f"{DATASET_ID}.json"
    source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    repo_source = (
        Path(__file__).resolve().parent / "local_sources" / f"{DATASET_ID}.json"
    )
    repo_source.parent.mkdir(parents=True, exist_ok=True)
    repo_source.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    catalog["source"] = source
    return catalog


def default_fish_letterbox_projection_path(data_root: str | Path) -> Path:
    """On-disk location of the opt-in dual-view letterbox_v1 cache."""

    return (
        Path(data_root).expanduser().resolve()
        / "projection"
        / FISH_LETTERBOX_PROJECTION_DIRNAME
    )


def designed_fish_to_box_task_source(
    data_root: str | Path,
    *,
    task_class: int,
    projection: Path | None = None,
) -> V10SourceSpec:
    """Local-only designed fish-to-box slices. Public mix is forbidden.

    The default return (``projection=None``) is unchanged: no projection or
    real_grounding field is set. Pass ``projection=`` or call
    ``designed_fish_to_box_task_source_with_letterbox_projection`` to opt in.
    """

    root = Path(data_root).expanduser().resolve()
    source_path = root / "local_sources" / f"{DATASET_ID}.json"
    payload = json.loads(source_path.read_text())
    if payload.get("name") != DATASET_ID:
        raise ValueError(f"unexpected task source name: {source_path}")
    require_source_task_class(payload, task_class)
    catalog = json.loads(Path(payload["catalog"]).read_text())
    tagged = require_catalog_task_class(catalog, task_class)
    spec = V10SourceSpec(
        trajectory=Path(payload["wrist_cache"]),
        secondary_trajectory=Path(payload["column_cache"]),
        geometry=Path(payload["geometry"]),
        primary_view=0,
        validation_episodes=tuple(int(value) for value in payload["validation_episodes"]),
        primary_camera_role=WRIST_CAMERA_ROLE,
        secondary_camera_role=EXTERNAL_CAMERA_ROLE,
        task_class=tagged,
    )
    if projection is None:
        return spec
    return replace(spec, projection=Path(projection).expanduser().resolve())


def designed_fish_to_box_task_source_with_letterbox_projection(
    data_root: str | Path,
    *,
    task_class: int,
    projection: Path | None = None,
) -> V10SourceSpec:
    """Opt-in variant that attaches the dual-view letterbox_v1 nominal cache.

    Builds the default spec first, then attaches ``projection``. The cache
    must already exist; this never invents ``q_true`` and never changes the
    default source constructor.
    """

    spec = designed_fish_to_box_task_source(data_root, task_class=task_class)
    cache = (
        Path(projection).expanduser().resolve()
        if projection is not None
        else default_fish_letterbox_projection_path(data_root)
    )
    if not (cache / "manifest.json").is_file() or not (cache / "q_nominal_px.npy").is_file():
        raise FileNotFoundError(
            f"letterbox dual-view projection cache missing at {cache}; "
            "build it with scripts/build_fish_dual_view_letterbox_projection.py"
        )
    return replace(spec, projection=cache)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/james/so101-data-disk/lerobot/so101"),
    )
    return parser.parse_args()


def main() -> int:
    catalog = build_task_dataset(data_root=parse_args().data_root)
    print(json.dumps(
        {
            "name": catalog["name"],
            "task_class": catalog["task_class"],
            "frames": catalog["frames"],
            "dropped_frames": catalog["dropped_frames"],
            "train_slice_ids": catalog["train_slice_ids"],
            "val_slice_ids": catalog["val_slice_ids"],
            "slices": [
                {
                    "slice_id": item["slice_id"],
                    "source_episode": item["source_episode"],
                    "phase": item["phase"],
                    "split": item["split"],
                    "frames": item["frames"],
                    "closes": item["closes"],
                    "terminal_lift": item["terminal_state"][1],
                    "gripper_min": item["gripper_min"],
                }
                for item in catalog["slices"]
            ],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
