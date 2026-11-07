"""V20 source identities, explicit calibrated-position units and real RGB.

The archive's companion arrays store positions divided by 100.  This module
is the only V20 boundary that reverses that storage transform.  CAD and raw
temporal ``states`` already use calibrated positions and are never multiplied.
No split here claims unseen evidence: every 320K archive row is training data.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

POSITION_UNIT = "lerobot_calibrated_position"
VIEW_ORDER = ("wrist", "column")


class V20DataError(ValueError):
    pass


def _finite(name: str, array: np.ndarray) -> np.ndarray:
    if not np.isfinite(array).all():
        raise V20DataError(f"{name} contains non-finite values")
    return array


def companion_to_position(value: Any) -> np.ndarray:
    """Reverse the documented companion /100 storage operation exactly once."""
    return _finite("companion position", np.asarray(value, dtype=np.float32) * 100.0)


def _indices(value: Any, upper: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64)
    if result.ndim != 1 or not len(result) or np.any((result < 0) | (result >= upper)):
        raise V20DataError("indices must be a nonempty in-range vector")
    return result


def _array(root: Path, name: str) -> np.ndarray:
    return np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)


@dataclass(frozen=True)
class V20ChainBatch:
    rows: np.ndarray
    chain_id: np.ndarray
    endpoint_index: np.ndarray              # B,5
    q_position: np.ndarray                  # B,5,6: actual endpoint readbacks
    commands_position: np.ndarray           # B,4,5,6
    measured_q_paths_position: np.ndarray   # B,4,6,6
    command_time_offsets_s: np.ndarray      # B,4,5 relative to each Hh
    readback_time_offsets_s: np.ndarray     # B,4,6 relative to each Hh
    endpoint_time_s: np.ndarray             # B,5 source episode clock
    view_valid: np.ndarray                  # B,5,2
    task_text: tuple[str, ...]
    position_unit: str = POSITION_UNIT
    time_source: str = "source-frame-index / declared-source-fps"


@dataclass(frozen=True)
class V20ActorBatch:
    rows: np.ndarray
    horizon: np.ndarray
    endpoint_index: np.ndarray
    q_position: np.ndarray
    target_commands_position: np.ndarray
    measured_q_path_position: np.ndarray
    command_time_offsets_s: np.ndarray
    readback_time_offsets_s: np.ndarray
    history_source_endpoint_index: np.ndarray
    history_endpoint_index: np.ndarray
    history_commands_position: np.ndarray
    history_readbacks_position: np.ndarray
    history_command_time_offsets_s: np.ndarray
    history_readback_time_offsets_s: np.ndarray
    history_valid: np.ndarray
    task_text: tuple[str, ...]


class V20TemporalArchive:
    """Read the existing archive without old step-cap or split filtering."""

    def __init__(self, endpoint_root: str | Path, companion_root: str | Path,
                 temporal_root: str | Path, *, expected_rows: int | None = 320_000,
                 expected_endpoints: int | None = 572_521) -> None:
        from fcpg.prod_v18_1_data import V181TemporalArchive
        self.archive = V181TemporalArchive(endpoint_root, companion_root, temporal_root,
            expected_rows=expected_rows, expected_endpoints=expected_endpoints)
        self.endpoint_root = Path(endpoint_root)
        self.temporal_root = Path(temporal_root)
        self.companion_root = Path(companion_root)
        self.rows = self.archive.rows
        self.endpoint_count = self.archive.endpoint_count
        self.training_rows = np.arange(self.rows, dtype=np.int64)
        self.episodes = self.archive.episodes
        self.row_endpoint_index = self.archive.row_endpoint_index
        self.endpoint_view_valid = self.archive.endpoint_view_valid
        self.endpoint_episode_index = self.archive.endpoint_episode_index
        self.endpoint_local_frame_index = self.archive.endpoint_local_frame_index
        self.chain_id = self.archive.chain_id
        self.sidecar_row = self.archive.sidecar_row
        manifest = json.loads((self.temporal_root / "manifest.json").read_text())
        self.task_names = tuple(manifest["task_names"])
        self.task_index = _array(self.endpoint_root, "task_index")
        self.action_frame_indices = _array(self.temporal_root, "window_future_action_indices")
        self.window_episode = _array(self.temporal_root, "window_episode")
        self.window_task_index = _array(self.temporal_root, "window_task_index")
        if self.task_index.shape != (self.rows,):
            raise V20DataError("archive task index shape drifted")
        chain = np.asarray(self.chain_id, dtype=np.int64)
        if not np.array_equal(self.task_index, self.window_task_index[chain]):
            raise V20DataError("task text is not from the same temporal chain")
        eps = self.endpoint_episode_index[self.row_endpoint_index]
        if not np.all(eps == eps[:, :1]) or not np.array_equal(eps[:, 0], self.window_episode[chain]):
            raise V20DataError("chain image endpoints cross source episodes")
        self.fps = np.asarray([float(x["manifest_row"]["fps_declared"]) for x in self.episodes])
        if not np.isfinite(self.fps).all() or np.any(self.fps <= 0):
            raise V20DataError("source fps missing: do not invent endpoint times")
        for row in range(self.rows):
            task_index = int(self.task_index[row])
            if not 0 <= task_index < len(self.task_names):
                raise V20DataError("task index out of range")
            task = " ".join(self.task_names[task_index].split()).casefold()
            allowed = {" ".join(str(x).split()).casefold() for x in
                       self.episodes[int(eps[row, 0])]["manifest_row"]["tasks"]}
            if not task or task not in allowed:
                raise V20DataError(f"task / episode disagreement at row {row}")
        # Ordinary endpoint identity, not occurrence or predicted state.
        flat = np.asarray(self.row_endpoint_index).reshape(-1)
        unique, first = np.unique(flat, return_index=True)
        if not np.array_equal(unique, np.arange(self.endpoint_count)):
            raise V20DataError("endpoint bank is not fully covered")
        self._endpoint_row = first // 5
        self._endpoint_horizon = first % 5

    def chain_batch(self, rows: Any) -> V20ChainBatch:
        selected = _indices(rows, self.rows)
        endpoint = np.asarray(self.row_endpoint_index[selected], dtype=np.int64)
        paths = companion_to_position(self.archive.measured_state_path[selected])
        commands = companion_to_position(self.archive.completed_commands[selected])
        if not np.allclose(paths[:, 1:, 0], paths[:, :-1, -1], atol=0.02, rtol=0):
            raise V20DataError("Hh start is not previous actual readback endpoint")
        q = np.concatenate((paths[:, :1, 0], paths[:, :, -1]), axis=1)
        frames = np.asarray(self.endpoint_local_frame_index[endpoint], dtype=np.int64)
        command_frames = np.asarray(self.action_frame_indices[self.chain_id[selected]], dtype=np.int64).reshape(-1, 4, 5)
        if (not np.array_equal(command_frames[:, :, 0], frames[:, :-1]) or
                np.any(np.diff(command_frames, axis=-1) <= 0) or
                np.any(command_frames[:, :, -1] >= frames[:, 1:])):
            raise V20DataError("commands and RGB endpoints violate causal source clock")
        fps = self.fps[self.endpoint_episode_index[endpoint[:, 0]]]
        ct = (command_frames - frames[:, :-1, None]) / fps[:, None, None]
        rt = np.concatenate((ct, ((frames[:, 1:] - frames[:, :-1]) / fps[:, None])[..., None]), axis=-1)
        return V20ChainBatch(selected, np.asarray(self.chain_id[selected]), endpoint, q,
            commands, paths, ct.astype(np.float32), rt.astype(np.float32),
            (frames / fps[:, None]).astype(np.float64),
            np.asarray(self.endpoint_view_valid[endpoint], dtype=bool),
            tuple(self.task_names[int(x)] for x in self.task_index[selected]))

    chains = chain_batch

    def actor_batch(self, ordinals: Any, *, rows: Any | None = None) -> V20ActorBatch:
        population = self.training_rows if rows is None else _indices(rows, self.rows)
        index = _indices(ordinals, len(population) * 4)
        selected, h = population[index // 4], index % 4
        chain = self.chain_batch(selected)
        b = np.arange(len(index))
        current_endpoint = chain.endpoint_index[b, h]
        hist_endpoint = np.repeat(current_endpoint[:, None], 3, axis=1)
        hist_source = hist_endpoint.copy()
        hc = np.zeros((len(index), 3, 5, 6), dtype=np.float32)
        hp = np.zeros((len(index), 3, 6, 6), dtype=np.float32)
        hct = np.zeros((len(index), 3, 5), dtype=np.float32)
        hrt = np.zeros((len(index), 3, 6), dtype=np.float32)
        valid = np.zeros((len(index), 3), dtype=bool)
        for slot in range(3):
            active = h > slot
            valid[active, slot] = True
            hist_source[active, slot] = chain.endpoint_index[active, slot]
            hist_endpoint[active, slot] = chain.endpoint_index[active, slot + 1]
            hc[active, slot] = chain.commands_position[active, slot]
            hp[active, slot] = chain.measured_q_paths_position[active, slot]
            hct[active, slot] = chain.command_time_offsets_s[active, slot]
            hrt[active, slot] = chain.readback_time_offsets_s[active, slot]
        return V20ActorBatch(selected, h, current_endpoint, chain.q_position[b, h],
            chain.commands_position[b, h], chain.measured_q_paths_position[b, h],
            chain.command_time_offsets_s[b, h], chain.readback_time_offsets_s[b, h],
            hist_source, hist_endpoint, hc, hp, hct, hrt, valid, chain.task_text)

    def endpoint_state(self, endpoint_indices: Any) -> dict[str, np.ndarray]:
        endpoint = _indices(endpoint_indices, self.endpoint_count)
        row, horizon = self._endpoint_row[endpoint], self._endpoint_horizon[endpoint]
        paths = companion_to_position(self.archive.measured_state_path[row])
        q = paths[np.arange(len(endpoint)), np.maximum(horizon - 1, 0), -1].copy()
        h0 = horizon == 0
        q[h0] = paths[h0, 0, 0]
        ep = self.endpoint_episode_index[endpoint]
        return {"q_position": q, "q_available": np.ones(len(endpoint), dtype=bool),
                "endpoint_time_s": self.endpoint_local_frame_index[endpoint] / self.fps[ep],
                "camera_available": np.zeros((len(endpoint), 2), dtype=bool)}

    def dense_rows(self, rgb_root: str | Path) -> np.ndarray:
        subset = np.asarray(_array(Path(rgb_root), "sidecar_row"), dtype=np.int64)
        if len(np.unique(subset)) != len(subset):
            raise V20DataError("20K sidecar identity is not unique")
        selected = np.flatnonzero(np.isin(self.sidecar_row, subset))
        if len(selected) != len(subset):
            raise V20DataError("20K does not join exactly into full archive")
        return selected.astype(np.int64)

    def media_requests(self, endpoint_indices: Any):
        return self.archive.media_requests(endpoint_indices)


V20Archive = V20TemporalArchive


class ArchiveRGBSource:
    """Decode actual source videos through the same 224px letterbox as DINO.

    There is deliberately no fallback to token-only input. Missing valid-view
    media raises; unavailable views remain explicitly invalid and zero-filled.
    A supplied decoder follows IsolatedVideoDecodePool.map_ordered, useful for
    tests and for a shared persistent pool in the production cache builder.
    """
    def __init__(self, archive: V20TemporalArchive, *, workers: int = 4,
                 decoder: Any | None = None, timeout_seconds: float = 300.0) -> None:
        self.archive = archive
        self.decoder = decoder
        self.workers = workers
        self.timeout_seconds = timeout_seconds
        self._owned = decoder is None

    def _pool(self):
        if self.decoder is None:
            from fcpg.isolated_video_decode import IsolatedVideoDecodePool
            self.decoder = IsolatedVideoDecodePool(workers=self.workers)
        return self.decoder

    def get_numpy(self, endpoint_indices: Any) -> dict[str, np.ndarray]:
        endpoint = _indices(endpoint_indices, self.archive.endpoint_count)
        rgb = np.zeros((len(endpoint), 2, 3, 224, 224), dtype=np.uint8)
        destinations: dict[int, list[int]] = defaultdict(list)
        for i, value in enumerate(endpoint):
            destinations[int(value)].append(i)
        grouped: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        for identity, view, path, frame, valid in self.archive.media_requests(np.unique(endpoint)):
            if valid:
                if not Path(path).is_file():
                    raise V20DataError(f"valid-view RGB is missing: {path}")
                for index in destinations[int(identity)]:
                    grouped[str(path)][int(frame)].append((index, int(view)))
        jobs = [(path, tuple(sorted(frames)), 224, 1) for path, frames in sorted(grouped.items())]
        decoded = self._pool().map_ordered(jobs, timeout=self.timeout_seconds) if jobs else []
        for request, value in zip(jobs, decoded, strict=True):
            path, frames, images = value
            if str(path) != request[0] or not np.array_equal(frames, request[1]):
                raise V20DataError("decoder changed endpoint order")
            images = np.asarray(images)
            if images.shape != (len(frames), 224, 224, 3) or images.dtype != np.uint8:
                raise V20DataError("RGB decoder must return uint8 letterbox224")
            for offset, frame in enumerate(frames):
                for index, view in grouped[str(path)][int(frame)]:
                    rgb[index, view] = np.moveaxis(images[offset], -1, 0)
        result = self.archive.endpoint_state(endpoint)
        result.update(rgb=rgb, view_valid=np.asarray(self.archive.endpoint_view_valid[endpoint], dtype=bool),
                      endpoint_index=endpoint)
        return result

    def audit_media(self, *, endpoint_indices: Any | None = None) -> dict[str, Any]:
        endpoints = np.arange(self.archive.endpoint_count) if endpoint_indices is None else _indices(endpoint_indices, self.archive.endpoint_count)
        paths: set[str] = set()
        failures: list[dict[str, Any]] = []
        valid_requests = 0
        for endpoint in endpoints.tolist():
            try:
                for _, view, path, _, valid in self.archive.media_requests([endpoint]):
                    if valid:
                        valid_requests += 1
                        paths.add(str(path))
                        if not Path(path).is_file():
                            failures.append({"endpoint_index": endpoint, "view": view, "path": str(path)})
            except (OSError, ValueError, IndexError) as error:
                failures.append({"endpoint_index": endpoint, "reason": str(error)})
        return {"checked_endpoints": len(endpoints), "valid_view_requests": valid_requests,
                "unique_video_paths": len(paths), "missing_or_unresolvable_count": len(failures),
                "missing_or_unresolvable_examples": failures[:20],
                "decoded_video_integrity_checked": False,
                "camera_geometry_inferred_from_dataset_id": False,
                "complete_path_audit": len(endpoints) == self.archive.endpoint_count}

    def close(self) -> None:
        if self._owned and self.decoder is not None:
            self.decoder.close()
            self.decoder = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class V20CADArchive:
    """Structured CAD labels; the public head only supervises the jaw pair."""
    def __init__(self, root: str | Path, *, expected_rows: int | None = 200_000) -> None:
        self.root = Path(root)
        names = ("rgb", "state", "keypoints_px", "keypoints_base_xyz", "camera_from_reference", "intrinsic", "valid", "split")
        for name in names:
            setattr(self, name, _array(self.root, name))
        self.rows = len(self.rgb)
        if expected_rows is not None and self.rows != expected_rows:
            raise V20DataError("CAD population changed")
        shapes = {"rgb": (self.rows, 2, 224, 224, 3), "state": (self.rows, 6),
                  "keypoints_px": (self.rows, 2, 6, 2), "keypoints_base_xyz": (self.rows, 6, 3),
                  "camera_from_reference": (self.rows, 2, 4, 4), "intrinsic": (self.rows, 2, 3, 3),
                  "valid": (self.rows, 2, 6), "split": (self.rows,)}
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise V20DataError(f"CAD {name} shape mismatch")
        if self.rgb.dtype != np.uint8 or self.valid.dtype != np.bool_:
            raise V20DataError("CAD RGB/valid dtype mismatch")
        self.training_rows = np.flatnonzero(self.split == 0).astype(np.int64)
        if expected_rows == 200_000 and len(self.training_rows) != 190_000:
            raise V20DataError("CAD TRAIN must contain 190K rows")

    def batch(self, rows: Any) -> dict[str, np.ndarray]:
        index = _indices(rows, self.rows)
        # The CAD generator's storage order is [column,wrist]; all spatial
        # arrays move together into the public [wrist,column] contract.
        all_points = np.asarray(self.keypoints_px[index], dtype=np.float32)[:, ::-1].copy()
        all_valid = np.asarray(self.valid[index], dtype=bool)[:, ::-1].copy()
        return {"row_index": index, "dino_index": index,
                "rgb": np.transpose(np.asarray(self.rgb[index])[:, ::-1], (0, 1, 4, 2, 3)).copy(),
                "q_position": _finite("CAD q", np.asarray(self.state[index], dtype=np.float32)),
                "q_available": np.ones(len(index), dtype=bool),
                "camera_from_reference": _finite("CAD T", np.asarray(self.camera_from_reference[index])[:, ::-1].copy()),
                "intrinsic": _finite("CAD K", np.asarray(self.intrinsic[index])[:, ::-1].copy()),
                "image_transform": np.broadcast_to(np.eye(3, dtype=np.float32), (len(index), 2, 3, 3)).copy(),
                "camera_available": np.ones((len(index), 2), dtype=bool),
                "view_valid": np.ones((len(index), 2), dtype=bool),
                "all_keypoints_px": all_points, "all_keypoints_valid": all_valid,
                "keypoints_base_xyz": np.asarray(self.keypoints_base_xyz[index]),
                "fm_px": all_points[:, :, 4:6], "fm_valid": all_valid[:, :, 4:6],
                "ordered": np.ones(len(index), dtype=bool)}


def recover_real_q_from_temporal(panel_path: str | Path, temporal_root: str | Path,
                                 annotation_manifest_path: str | Path | None = None) -> dict[tuple[str, int], np.ndarray]:
    """Exact entry+endpoint -> original source-frame q, with no image guessing."""
    panel = json.loads(Path(panel_path).read_text())
    root = Path(temporal_root)
    manifest = json.loads((root / "manifest.json").read_text())
    states, offsets = _array(root, "states"), _array(root, "episode_offsets")
    records = panel.get("records")
    if not isinstance(records, list):
        raise V20DataError("real panel omits exact source records")
    declared_entries = None
    if annotation_manifest_path is not None:
        samples = json.loads(Path(annotation_manifest_path).read_text())["samples"]
        declared_entries = {str(x["entry_id"]) for x in samples}
    result: dict[tuple[str, int], np.ndarray] = {}
    for record in records:
        entry, episode = str(record["entry_id"]), int(record["episode_index"])
        if declared_entries is not None and entry not in declared_entries:
            raise V20DataError("recovery panel is not the real annotation corpus")
        ep = manifest["episodes"][episode]
        start, stop = int(offsets[episode]), int(offsets[episode + 1])
        if int(ep["frame_start"]) != start or int(ep["frame_stop"]) != stop:
            raise V20DataError("real source episode and temporal offsets disagree")
        if record.get("canonical_episode_id") is not None and record["canonical_episode_id"] != ep.get("canonical_episode_id"):
            raise V20DataError("real source episode identity differs")
        for kind, field in enumerate(("query_offset", "target_offset")):
            offset = int(record[field])
            if not 0 <= offset < stop - start or (entry, kind) in result:
                raise V20DataError("real source endpoint missing or ambiguous")
            result[entry, kind] = _finite("recovered real q", np.asarray(states[start + offset], dtype=np.float32).copy())
    return result


def load_real_tip_cache(path: str | Path, *, recovered_q: Mapping[tuple[str, int], Any] | None = None) -> dict[str, Any]:
    """Reorder legacy [column,wrist] labels and reject fabricated q semantics."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return real_tip_payload(payload, recovered_q=recovered_q)


def real_tip_payload(payload: Mapping[str, Any], *, recovered_q: Mapping[tuple[str, int], Any] | None = None) -> dict[str, Any]:
    def arr(name: str, dtype=None):
        value = payload[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)
    rgb, source = arr("rgb", np.uint8), arr("source", np.int8)
    count = len(rgb)
    q = arr("current_q", np.float32).copy()
    ordered = arr("ordered", bool)
    endpoint_kind, entry_id = arr("endpoint_kind", np.int8), arr("entry_id")
    if rgb.shape != (count, 2, 3, 224, 224) or q.shape != (count, 6):
        raise V20DataError("real label tensor shapes changed")
    if not np.isin(source, (0, 1)).all():
        raise V20DataError("unknown real-label provenance")
    # source 0 is the measured ordered panel. source 1's zeros were placeholders.
    available = (source == 0) & np.isfinite(q).all(axis=-1)
    recovered = np.zeros(count, dtype=bool)
    for i in np.flatnonzero(source == 1):
        key = str(entry_id[i]), int(endpoint_kind[i])
        if recovered_q is not None and key in recovered_q:
            value = _finite("real recovered q", np.asarray(recovered_q[key], dtype=np.float32))
            if value.shape != (6,):
                raise V20DataError("recovered q is not six calibrated positions")
            q[i], available[i], recovered[i] = value, True, True
        else:
            q[i] = 0.0  # masked missing value, never a closed-gripper label
    valid = arr("view_valid", bool)[:, ::-1].copy()
    return {"rgb": rgb[:, ::-1].copy(), "q_position": q, "q_available": available,
            "fm_px": arr("target_fm_px", np.float32)[:, ::-1].copy(),
            "nominal_fm_px": arr("nominal_fm_px", np.float32)[:, ::-1].copy(),
            "view_valid": valid, "fm_valid": np.repeat(valid[..., None], 2, axis=-1),
            "camera_available": np.zeros((count, 2), dtype=bool), "ordered": ordered,
            "source": source, "endpoint_kind": endpoint_kind, "entry_id": entry_id,
            "audit": {"rows": count, "valid_views": int(valid.sum()),
                      "valid_views_wrist_column": valid.sum(axis=0).tolist(),
                      "original_measured_q_rows": int((source == 0).sum()),
                      "placeholder_provenance_rows": int((source == 1).sum()),
                      "exact_q_recovered_rows": int(recovered.sum()),
                      "q_unavailable_rows": int((~available).sum()),
                      "evidence_role": "training-fit", "position_unit": POSITION_UNIT}}


def balanced_real_indices(view_valid: np.ndarray, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Equal view probability; natural uniform valid rows within each view."""
    if count <= 0 or view_valid.ndim != 2 or view_valid.shape[1] != 2:
        raise V20DataError("invalid real refit sampler input")
    views = np.arange(count, dtype=np.int64) % 2
    rng.shuffle(views)
    rows = np.empty(count, dtype=np.int64)
    for view in range(2):
        candidates = np.flatnonzero(view_valid[:, view])
        if not len(candidates):
            raise V20DataError("both views require real spatial labels")
        active = views == view
        rows[active] = rng.choice(candidates, int(active.sum()), replace=True)
    return rows, views


def deterministic_real_indices(view_valid: np.ndarray, ordinals: Any, seed: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Resume/OOM-safe natural sampling, independent of microbatch partition."""
    ids = np.asarray(ordinals, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0 or (ids < 0).any():
        raise V20DataError("real sample ordinals must be non-negative")
    views = ids % 2
    candidates = [np.flatnonzero(view_valid[:, view]) for view in range(2)]
    if not all(len(value) for value in candidates):
        raise V20DataError("both views require real spatial labels")
    rows = np.asarray([np.random.default_rng(np.random.SeedSequence([seed, int(identity)])).choice(candidates[int(view)])
                       for identity, view in zip(ids, views, strict=True)], dtype=np.int64)
    return rows, views


def audit_archive_contract(archive: V20TemporalArchive, *, block: int = 4096) -> dict[str, Any]:
    max_continuity_error = 0.0
    delta_sum = np.zeros(6, dtype=np.float64)
    delta_sq = delta_sum.copy()
    count = 0
    for start in range(0, archive.rows, block):
        batch = archive.chain_batch(np.arange(start, min(start + block, archive.rows)))
        path = batch.measured_q_paths_position
        max_continuity_error = max(max_continuity_error, float(np.abs(path[:, 1:, 0] - path[:, :-1, -1]).max(initial=0)))
        delta = np.diff(path, axis=-2).reshape(-1, 6).astype(np.float64)
        delta_sum += delta.sum(axis=0)
        delta_sq += np.square(delta).sum(axis=0)
        count += len(delta)
    variance = np.maximum(delta_sq / count - np.square(delta_sum / count), 0)
    return {"rows": archive.rows, "unique_endpoints": archive.endpoint_count,
            "actor_pair_exposures": archive.rows * 4 * 4,
            "all_archive_rows_are_training": True, "unseen_validation_claim": False,
            "joint_position_unit": POSITION_UNIT, "companion_restore_scale": 100.0,
            "max_readback_endpoint_continuity_error": max_continuity_error,
            "joint_measured_delta_std": np.sqrt(variance).tolist(),
            "joint_measured_delta_count": count,
            "time_source": "source-frame-index / declared-source-fps",
            "camera_geometry_available_from_dataset_id": False,
            "fish_episode_2_opened": False}
