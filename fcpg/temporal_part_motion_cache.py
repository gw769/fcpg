"""Schedule-bound six-part CAD motion signatures for candidate-first FCPG-H."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from fcpg.fcpg_h_candidate_first import (
    FCPGH_V5_METHOD_REVISION,
    PART_COUNT,
    PART_MOTION_DIM,
    build_part_motion_signature_rows,
)
from fcpg.offline_pipeline_manifest import sha256_file, sha256_json
from fcpg.so101_kinematics import SO101MuJoCoKinematics
from fcpg.temporal_cache import assert_disk_budget
from fcpg.temporal_grounding_protocol import TEMPORAL_PROTOCOL_ID
from fcpg.temporal_rgb_cache import RGB_CACHE_FORMAT


PART_MOTION_CACHE_FORMAT = "so101-temporal-six-part-motion-cache-v1"
PART_MOTION_DTYPE = np.dtype("float16")


def selected_episode_offsets_from_rgb_cache(
    index: object,
    rgb_cache_root: Path,
) -> tuple[dict[int, np.ndarray], Mapping[str, object]]:
    """Union view-specific RGB rows into one schedule-bound frame set."""

    root = Path(rgb_cache_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != RGB_CACHE_FORMAT or not manifest.get("complete"):
        raise ValueError("part-motion sidecar requires a completed temporal RGB cache")
    index_manifest_path = Path(index.root) / "manifest.json"
    if manifest.get("index_manifest_sha256") != sha256_file(index_manifest_path):
        raise ValueError("RGB cache and temporal index SHA do not match")
    if not bool(manifest.get("complete_schedule_coverage")):
        raise ValueError("part-motion sidecar requires complete schedule coverage")

    offsets_by_episode: dict[int, set[int]] = {}
    for pack in manifest["packs"]:
        episode_index = int(pack["episode_index"])
        if not 0 <= episode_index < len(index.episodes):
            raise ValueError("RGB cache refers to an unknown temporal episode")
        record = index.episodes[episode_index]
        if str(pack["canonical_episode_id"]) != str(record["canonical_episode_id"]):
            raise ValueError("RGB cache episode identity drifted from temporal index")
        offsets = np.load(
            root / "packs" / str(pack["offsets_file"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        if offsets.ndim != 1 or offsets.dtype != np.int32:
            raise ValueError("RGB cache offsets must be one-dimensional int32")
        offsets_by_episode.setdefault(episode_index, set()).update(map(int, offsets))
    selected = {
        episode_index: np.asarray(sorted(offsets), dtype=np.int32)
        for episode_index, offsets in offsets_by_episode.items()
    }
    return selected, manifest


def _episode_window_range(index: object, episode_index: int) -> tuple[int, int]:
    episode = index.arrays["window_episode"]
    start = int(np.searchsorted(episode, episode_index, side="left"))
    stop = int(np.searchsorted(episode, episode_index, side="right"))
    if stop <= start or not bool(np.all(episode[start:stop] == episode_index)):
        raise ValueError("selected episode has no temporally indexed windows")
    return start, stop


def materialize_temporal_part_motion_cache(
    index: object,
    *,
    rgb_cache_root: Path,
    scene: Path,
    output: Path,
    generator: Mapping[str, object],
) -> dict[str, object]:
    """Build fp16 `[selected_frame,6,44]` signatures without reading RGB."""

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    selected, rgb_manifest = selected_episode_offsets_from_rgb_cache(
        index, rgb_cache_root
    )
    frame_count = sum(len(offsets) for offsets in selected.values())
    if frame_count <= 0:
        raise ValueError("RGB schedule selected no frames for part-motion cache")
    planned_bytes = frame_count * (
        np.dtype("int64").itemsize
        + PART_COUNT * PART_MOTION_DIM * PART_MOTION_DTYPE.itemsize
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    assert_disk_budget(output.parent, planned_bytes=planned_bytes)
    output.mkdir()
    global_indices = np.lib.format.open_memmap(
        output / "global_frame_indices.npy",
        mode="w+",
        dtype="int64",
        shape=(frame_count,),
    )
    signatures = np.lib.format.open_memmap(
        output / "part_motion_signatures.npy",
        mode="w+",
        dtype=PART_MOTION_DTYPE,
        shape=(frame_count, PART_COUNT, PART_MOTION_DIM),
    )
    kinematics = SO101MuJoCoKinematics(scene)
    cursor = 0
    for episode_index in sorted(selected):
        record = index.episodes[episode_index]
        frame_start = int(record["frame_start"])
        frame_stop = int(record["frame_stop"])
        episode_length = frame_stop - frame_start
        requested = selected[episode_index]
        if (
            requested.size == 0
            or int(requested[0]) < 0
            or int(requested[-1]) >= episode_length
        ):
            raise ValueError("part-motion cache requested an out-of-range frame")
        window_start, window_stop = _episode_window_range(index, episode_index)
        canonical = np.unique(
            np.asarray(
                index.arrays["window_image_indices"][window_start:window_stop],
                dtype=np.int64,
            ).reshape(-1)
        )
        positions = np.searchsorted(canonical, requested)
        if (
            bool(np.any(positions >= len(canonical)))
            or not bool(np.array_equal(canonical[positions], requested))
        ):
            raise ValueError("RGB schedule frame is absent from the canonical 10 Hz grid")
        previous_positions = np.maximum(positions - 1, 0)
        previous_offsets = canonical[previous_positions]
        needed_offsets = np.unique(np.concatenate((requested, previous_offsets)))
        needed_global = frame_start + needed_offsets
        needed_states = np.asarray(
            index.arrays["states"][needed_global], dtype=np.float32
        )
        needed_xyz, needed_jacobians = (
            kinematics.forward_calibration_keypoints_and_jacobians(needed_states)
        )
        current_lookup = np.searchsorted(needed_offsets, requested)
        previous_lookup = np.searchsorted(needed_offsets, previous_offsets)
        current_states = needed_states[current_lookup]
        previous_states = needed_states[previous_lookup]
        completed_actions = np.asarray(
            index.arrays["actions"][frame_start + previous_offsets],
            dtype=np.float32,
        )
        selected_signatures = build_part_motion_signature_rows(
            torch.from_numpy(current_states),
            torch.from_numpy(previous_states),
            torch.from_numpy(completed_actions),
            torch.from_numpy(needed_xyz[current_lookup]).float(),
            torch.from_numpy(needed_xyz[previous_lookup]).float(),
            torch.from_numpy(needed_jacobians[current_lookup]).float(),
            torch.from_numpy(positions > 0),
        ).numpy()
        stop = cursor + len(requested)
        global_indices[cursor:stop] = frame_start + requested
        signatures[cursor:stop] = selected_signatures.astype(
            PART_MOTION_DTYPE, copy=False
        )
        cursor = stop
    if cursor != frame_count:
        raise AssertionError("part-motion cache cursor did not fill allocated arrays")
    global_indices.flush()
    signatures.flush()
    if frame_count > 1 and not bool(np.all(np.diff(global_indices) > 0)):
        raise ValueError("part-motion cache global indices must be unique and sorted")

    index_manifest_path = Path(index.root) / "manifest.json"
    rgb_manifest_path = Path(rgb_cache_root).expanduser().resolve() / "manifest.json"
    manifest = {
        "format": PART_MOTION_CACHE_FORMAT,
        "protocol_id": TEMPORAL_PROTOCOL_ID,
        "method_revision": FCPGH_V5_METHOD_REVISION,
        "complete": True,
        "frame_count": frame_count,
        "part_count": PART_COUNT,
        "signature_dim": PART_MOTION_DIM,
        "dtype": PART_MOTION_DTYPE.name,
        "planned_payload_bytes": planned_bytes,
        "actual_payload_bytes": (
            (output / "global_frame_indices.npy").stat().st_size
            + (output / "part_motion_signatures.npy").stat().st_size
        ),
        "index_manifest_sha256": sha256_file(index_manifest_path),
        "rgb_cache_manifest_sha256": sha256_file(rgb_manifest_path),
        "rgb_cache_content_sha256": rgb_manifest["content_sha256"],
        "scene_sha256": sha256_file(scene),
        "arrays": {
            "global_frame_indices": {
                "shape": [frame_count],
                "sha256": sha256_file(output / "global_frame_indices.npy"),
            },
            "part_motion_signatures": {
                "shape": [frame_count, PART_COUNT, PART_MOTION_DIM],
                "sha256": sha256_file(output / "part_motion_signatures.npy"),
            },
        },
        "generator": dict(generator),
    }
    manifest["content_sha256"] = sha256_json(manifest)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


class TemporalPartMotionDiskCache:
    """Read-only mmap lookup for one exact training schedule."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        if (
            self.manifest.get("format") != PART_MOTION_CACHE_FORMAT
            or not self.manifest.get("complete")
        ):
            raise ValueError("part-motion cache is incomplete or has the wrong format")
        if self.manifest.get("protocol_id") != TEMPORAL_PROTOCOL_ID:
            raise ValueError("part-motion cache protocol mismatch")
        if (
            int(self.manifest.get("part_count", -1)) != PART_COUNT
            or int(self.manifest.get("signature_dim", -1)) != PART_MOTION_DIM
            or self.manifest.get("dtype") != PART_MOTION_DTYPE.name
        ):
            raise ValueError("part-motion cache tensor contract mismatch")
        self.global_indices = np.load(
            self.root / "global_frame_indices.npy", mmap_mode="r", allow_pickle=False
        )
        self.signatures = np.load(
            self.root / "part_motion_signatures.npy", mmap_mode="r", allow_pickle=False
        )
        expected = int(self.manifest["frame_count"])
        if self.global_indices.shape != (expected,) or self.signatures.shape != (
            expected,
            PART_COUNT,
            PART_MOTION_DIM,
        ):
            raise ValueError("part-motion cache array shape mismatch")

    def assert_binding(self, *, index_manifest: Path, rgb_cache_manifest: Path) -> None:
        if sha256_file(index_manifest) != self.manifest["index_manifest_sha256"]:
            raise ValueError("part-motion cache temporal-index SHA mismatch")
        if sha256_file(rgb_cache_manifest) != self.manifest["rgb_cache_manifest_sha256"]:
            raise ValueError("part-motion cache RGB-schedule SHA mismatch")

    def get_many(self, global_frame_indices: np.ndarray) -> np.ndarray:
        requested = np.asarray(global_frame_indices, dtype=np.int64)
        positions = np.searchsorted(self.global_indices, requested.reshape(-1))
        valid = positions < len(self.global_indices)
        matched = np.zeros(len(positions), dtype=bool)
        matched[valid] = self.global_indices[positions[valid]] == requested.reshape(-1)[valid]
        if not bool(matched.all()):
            raise KeyError(
                f"part-motion cache missed {int((~matched).sum())} scheduled frames"
            )
        return np.asarray(self.signatures[positions]).reshape(
            *requested.shape, PART_COUNT, PART_MOTION_DIM
        )

    def close(self) -> None:
        for array in (self.global_indices, self.signatures):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
