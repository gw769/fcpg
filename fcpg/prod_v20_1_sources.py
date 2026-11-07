"""Production V20.1 minibatches from existing, identity-aligned source data.

This is not a second model path: every observation contains real RGB AND DINO.
The frozen encoders can be reused; no old jaw or policy checkpoint is loaded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from .prod_v20_timing import FISH_FPS, FISH_COMMAND_STRIDE_FRAMES, FISH_COMMAND_STEPS, FISH_COMMAND_PERIOD_S
from .v21_dequantize import dequantizes_on_device, quantized_rows_on_device
from .v21_encoder import DIM as ENCODER_DIM, PATCHES as ENCODER_PATCHES

from .prod_v20_data import (
    ArchiveRGBSource, V20CADArchive, V20DataError, V20TemporalArchive,
    _array, _indices, audit_archive_contract, balanced_real_indices,
    deterministic_real_indices, load_real_tip_cache, recover_real_q_from_temporal,
)


@dataclass(frozen=True)
class V201SourcePaths:
    endpoint_root: str = "/data/experiments/prod_v16_5_endpoint_cache_b7909f6"
    companion_root: str = "/data/experiments/prod_v16_2_minimal_v1/predictive-fcpg-companion-320k-v1"
    temporal_root: str = "/data/experiments/fcpg_h_temporal_v2/temporal-index"
    rgb_root: str = "/data/experiments/prod_v16_2_teacher_discarded_v1/rgb-formal-20k"
    sidecar_root: str = "/data/experiments/prod_v16_2_minimal_v1/sidecar-320k"
    cad_root: str = "/data/experiments/fcpg_forward_v1/teacher-render-200k"
    real_label_cache: str = "/data/experiments/prod_v17_4_real_tip_refit_320k/real-tip-cache.pt"
    real_recovery_panel: str = "/data/experiments/fcpg_h_query_initialized_displacement_measurement_v3/development-panel.json"
    real_annotation_manifest: str = "/data/experiments/fcpg_h_query_initialized_displacement_measurement_v3/annotation-ui/manifest.json"
    real_calibration: str = "/data/feedback-corrected-grounding/outputs/lerobot/so101/my_first_task_clean_v1_dual_camera_calibration/calibration_letterbox_v1.json"
    fish_wrist_root: str = "/data/experiments/prod_v18_1_tip_free/source/fish-wrist"
    fish_column_root: str = "/data/experiments/prod_v18_1_tip_free/source/fish-column"
    dino_cache_root: str = "/data/experiments/prod_v18_1_tip_free/dino-cache"
    task_cache_root: str = "/data/experiments/prod_v17_8_task_history_belief/task-cache"
    dino_repository: str = "/data/models/torch-hub/hub/facebookresearch_dinov2_main"
    dino_weights: str = "/data/models/torch-hub/hub/checkpoints/dinov2_vits14_pretrain.pth"


def validate_dino_cache_completion(root: str | Path, view_valid: np.ndarray) -> dict[str, Any]:
    """Require completed real feature extraction, not merely allocated arrays.

    Existing temporal/fish caches publish per-view completion [N,2]; the CAD
    cache publishes per-row completion [N]. Some older writers did not publish
    ``config.complete``: their explicit completion array remains authoritative.
    A published False is never overridden by apparently complete array data.
    """
    root = Path(root)
    try:
        config = json.loads((root / "config.json").read_text())
        done = _array(root, "done")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise V20DataError(f"DINO completion metadata is unreadable: {root}") from error
    valid = np.asarray(view_valid, dtype=bool)
    if valid.ndim != 2 or valid.shape[1] != 2 or done.dtype != np.bool_:
        raise V20DataError("DINO completion/validity dtype or shape differs")
    if done.shape == (len(valid),):
        completed_views = np.broadcast_to(done[:, None], valid.shape)
        representation = "per-endpoint"
    elif done.shape == valid.shape:
        completed_views = done
        representation = "per-endpoint-view"
    else:
        raise V20DataError("DINO completion array does not match endpoint identities")
    if "complete" in config and config["complete"] is not True:
        raise V20DataError(f"DINO cache is explicitly incomplete: {root}")
    missing = int((valid & ~completed_views).sum())
    if missing:
        raise V20DataError(f"DINO cache has {missing} unfinished valid endpoint-views: {root}")
    return {"cache": root.name, "completion_representation": representation,
            "config_complete_present": "complete" in config,
            "config_complete": config.get("complete"), "endpoints": len(valid),
            "valid_endpoint_views": int(valid.sum()), "unfinished_valid_endpoint_views": missing,
            "unfinished_invalid_endpoint_views": int((~valid & ~completed_views).sum()),
            "completed_for_training": True}


def _canonical_source_value(*values: Any) -> str | None:
    """Canonicalize recorded provenance, not a task label or a guessed source."""
    for value in values:
        # Dataset manifests use strings; integer dataset identifiers are also
        # legitimate. Containers, bools and empty placeholders are not IDs.
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            text = str(value).strip()
            if text:
                return text
    return None


def canonical_archive_source(episode: dict[str, Any]) -> str | None:
    """The same dataset_id -> recorded root fallback used by development strata."""
    record = episode.get("manifest_row") or {}
    return _canonical_source_value(record.get("dataset_id"), record.get("dataset_root"),
                                   episode.get("dataset_root"))


def build_source_identity_metadata(archive: Any, dense_rows: Any, *,
                                   fish_source_identity: str | None = None,
                                   fish_rows: int = 0) -> dict[str, Any]:
    """Build deterministic, source-local regularizer labels from chain metadata.

    These integers are grouping metadata only: they are not model inputs, task
    IDs, camera calibration or a learned domain embedding. Missing provenance
    stays -1 and must be excluded from grouped regularization, never pooled as
    an invented common unknown domain. Fish has a separate provenance namespace
    even if its recorded dataset name happens to equal an archive dataset name.
    """
    endpoints = np.asarray(archive.row_endpoint_index)
    episode_ids = np.asarray(archive.endpoint_episode_index)
    if endpoints.ndim != 2 or endpoints.shape != (archive.rows, 5):
        raise V20DataError("source identity requires five endpoints from each archive row")
    if not np.issubdtype(endpoints.dtype, np.integer) or episode_ids.ndim != 1:
        raise V20DataError("source endpoint identity shape/dtype differs")
    if np.any(endpoints < 0) or np.any(endpoints >= len(episode_ids)):
        raise V20DataError("source endpoint identity is out of range")
    row_episodes = episode_ids[endpoints]
    if not np.issubdtype(row_episodes.dtype, np.integer) or np.any(row_episodes < 0) or np.any(row_episodes >= len(archive.episodes)):
        raise V20DataError("source episode identity is out of range")
    if not np.all(row_episodes == row_episodes[:, :1]):
        raise V20DataError("source identity cannot group a chain crossing source episodes")
    episode_sources = [canonical_archive_source(episode) for episode in archive.episodes]
    row_sources = np.asarray(episode_sources, dtype=object)[row_episodes[:, 0]]
    # Only represented archive sources receive an integer. Ordering is lexical,
    # independent of episode order, sampled rows, task labels and random seed.
    names = sorted({source for source in row_sources if source is not None})
    lookup = {source: index for index, source in enumerate(names)}
    row_index = np.asarray([lookup.get(source, -1) for source in row_sources], dtype=np.int64)
    dense = _indices(dense_rows, archive.rows)
    dense_index = row_index[dense]
    fish_identity = _canonical_source_value(fish_source_identity)
    if fish_rows < 0:
        raise V20DataError("fish source row count must be nonnegative")
    fish_index = len(names) if fish_identity is not None else -1
    vocabulary = []
    for index, name in enumerate(names):
        vocabulary.append(dict(index=index, namespace="archive", source_id=name,
                               archive_rows=int((row_index == index).sum()),
                               dense_rows=int((dense_index == index).sum()), fish_rows=0))
    if fish_identity is not None:
        vocabulary.append(dict(index=fish_index, namespace="fish-rig-data", source_id=fish_identity,
                               archive_rows=0, dense_rows=0, fish_rows=int(fish_rows)))

    def counts(values):
        known = values >= 0
        return dict(rows=len(values), identified_rows=int(known.sum()),
                    missing_identity_rows=int((~known).sum()),
                    represented_sources=len(np.unique(values[known])))

    report = dict(contract="recorded-chain-source-local-grouping-v1", metadata_only=True,
                  model_input=False, task_id=False, camera_geometry=False,
                  missing_identity_index=-1, missing_identity_policy="exclude; never group unknown rows together",
                  canonical_source_precedence=["manifest_row.dataset_id", "manifest_row.dataset_root", "episode.dataset_root"],
                  fish_source_precedence=["paired_manifest.dataset_id", "paired_manifest.source_root"],
                  fish_source_index=fish_index, vocabulary=vocabulary,
                  counts={"archive": counts(row_index), "dense": counts(dense_index),
                          "fish": counts(np.full(int(fish_rows), fish_index, dtype=np.int64))})
    return dict(archive_row_source_index=row_index, fish_source_index=fish_index, report=report)


class V201FishTrainingSource:
    """Only indexes episode 0/1/3 pixels, actions and states; never episode 2."""
    def __init__(self, wrist_root: str | Path, column_root: str | Path):
        self.roots = Path(wrist_root), Path(column_root)
        manifests = [json.loads((root / "manifest.json").read_text()) for root in self.roots]
        if [x.get("camera") for x in manifests] != ["observation.images.wrist", "observation.images.column"]:
            raise V20DataError("fish camera roles must be wrist,column")
        if any(float(x.get("fps", 0)) != FISH_FPS for x in manifests):
            raise V20DataError("fish command/readback schedule requires declared 30 Hz")
        for key in ("dataset_id", "source_root", "task_names"):
            if manifests[0].get(key) != manifests[1].get(key):
                raise V20DataError(f"fish paired-view {key} provenance differs")
        self.source_identity = _canonical_source_value(manifests[0].get("dataset_id"),
                                                       manifests[0].get("source_root"))
        self.episodes = _array(self.roots[0], "episodes")
        self.frame_indices = _array(self.roots[0], "frame_indices")
        self.frames = tuple(_array(root, "frames") for root in self.roots)
        self.q = _array(self.roots[0], "states")
        self.actions = _array(self.roots[0], "actions")
        if self.q.shape != (len(self.episodes), 6) or self.actions.shape != self.q.shape:
            raise V20DataError("fish six-joint state/action shape differs")
        for frames in self.frames:
            if frames.shape != (len(self.episodes), 224, 224, 3) or frames.dtype != np.uint8:
                raise V20DataError("fish RGB cache must preserve letterbox224")
        # Metadata may enumerate held episode identity. No held RGB/action/q
        # values are materialized or included in equality/finite reductions.
        self.training_frames = np.flatnonzero(np.isin(self.episodes, (0, 1, 3)))
        for name in ("episodes", "frame_indices", "states", "actions"):
            other = _array(self.roots[1], name)
            reference = {"episodes": self.episodes, "frame_indices": self.frame_indices,
                         "states": self.q, "actions": self.actions}[name]
            if not np.array_equal(reference[self.training_frames], other[self.training_frames]):
                raise V20DataError(f"fish train dual-view {name} differs")
        self.starts = []
        for episode in (0, 1, 3):
            positions = np.flatnonzero(self.episodes == episode)
            if not len(positions) or not np.array_equal(self.frame_indices[positions], np.arange(len(positions))):
                raise V20DataError("fish train episode clock is not contiguous")
            self.starts.extend(int(positions[local]) for local in range(0, len(positions) - 60, 3))
        self.starts = np.asarray(self.starts, dtype=np.int64)

    def validate_frames(self, identities: Any) -> np.ndarray:
        ids = _indices(identities, len(self.episodes))
        if not np.isin(self.episodes[ids], (0, 1, 3)).all():
            raise V20DataError("fish episode 2 is not open to training or caching")
        return ids

    def observation(self, identities: Any) -> dict[str, np.ndarray]:
        ids = self.validate_frames(identities)
        rgb = np.stack([frames[ids] for frames in self.frames], axis=1).transpose(0, 1, 4, 2, 3)
        return {"rgb": np.ascontiguousarray(rgb), "q_position": np.asarray(self.q[ids], dtype=np.float32),
                "q_available": np.ones(len(ids), dtype=bool), "endpoint_index": ids,
                "endpoint_time_s": self.frame_indices[ids].astype(np.float64) / 30,
                "view_valid": np.ones((len(ids), 2), dtype=bool),
                "camera_available": np.zeros((len(ids), 2), dtype=bool)}

    def chains(self, rows: Any) -> dict[str, np.ndarray]:
        rows = _indices(rows, len(self.starts))
        endpoint = self.starts[rows, None] + np.arange(5)[None] * (FISH_COMMAND_STEPS * FISH_COMMAND_STRIDE_FRAMES)
        action_indices = endpoint[:, :-1, None] + np.arange(FISH_COMMAND_STEPS)[None, None] * FISH_COMMAND_STRIDE_FRAMES
        path_indices = np.concatenate((action_indices, endpoint[:, 1:, None]), axis=-1)
        self.validate_frames(path_indices.reshape(-1))
        q = np.asarray(self.q[endpoint], dtype=np.float32)
        commands = np.asarray(self.actions[action_indices], dtype=np.float32)
        paths = np.asarray(self.q[path_indices], dtype=np.float32)
        if not np.isfinite(q).all() or not np.isfinite(commands).all() or not np.isfinite(paths).all():
            raise V20DataError("fish training positions are non-finite")
        return {"rows": rows, "endpoint_index": endpoint, "q_position": q,
                "commands_position": commands, "measured_q_paths_position": paths,
                "time_deltas": np.full((len(rows), 4, 5), FISH_COMMAND_PERIOD_S, dtype=np.float32),
                "view_valid": np.ones((len(rows), 5, 2), dtype=bool),
                "initial_time_s": self.frame_indices[endpoint[:, 0]].astype(np.float64) / 30,
                "geometry": None}


class V201Sources:
    def __init__(self, paths: V201SourcePaths = V201SourcePaths(), *, device: str | torch.device = "cpu",
                 decode_workers: int = 4, seed: int = 20100):
        from .prod_v18_1_data import V181FeatureCache
        from .prod_v17_8_training import V178TaskVocabulary, V178TaskEmbeddingCache
        from .prod_v17_5_training import V175AllRealRGB
        if not isinstance(paths, V201SourcePaths):
            paths = V201SourcePaths(**{field.name: str(getattr(paths, field.name))
                                     for field in fields(V201SourcePaths)
                                     if getattr(paths, field.name, None) is not None})
        self.paths, self.device = paths, torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.archive = V20TemporalArchive(paths.endpoint_root, paths.companion_root, paths.temporal_root)
        self.rgb_source = ArchiveRGBSource(self.archive, workers=decode_workers)
        self.dense = V175AllRealRGB(paths.rgb_root, paths.sidecar_root)
        self.cad = V20CADArchive(paths.cad_root)
        self.fish = V201FishTrainingSource(paths.fish_wrist_root, paths.fish_column_root)
        recovery = recover_real_q_from_temporal(paths.real_recovery_panel, paths.temporal_root,
                                               paths.real_annotation_manifest)
        self.real = load_real_tip_cache(paths.real_label_cache, recovered_q=recovery)
        cache = Path(paths.dino_cache_root)
        self.archive_dino = [V181FeatureCache(cache / name) for name in ("temporal-train", "temporal-dev")]
        self.cad_dino = V181FeatureCache(cache / "cad-200k-structured")
        self.fish_dino = V181FeatureCache(cache / "fish")
        self.dino_completion_audit = [validate_dino_cache_completion(value.root, value.view_valid)
                                      for value in (*self.archive_dino, self.cad_dino, self.fish_dino)]
        self.vocabulary = V178TaskVocabulary.from_archive(self.archive.archive)
        self.tasks = V178TaskEmbeddingCache(paths.task_cache_root, self.vocabulary)
        self.dense_rows = self.archive.dense_rows(paths.rgb_root)
        sidecar_to_row = {int(value): row for row, value in enumerate(self.archive.sidecar_row)}
        self._dense_ordered_rows = np.asarray([sidecar_to_row[int(value)] for value in self.dense.sidecar_row], dtype=np.int64)
        if not np.array_equal(self.archive.chain_id[self._dense_ordered_rows],
                              _array(Path(paths.sidecar_root), "chain_id")[self.dense.sidecar_row]):
            raise V20DataError("20K RGB and 320K chain identities disagree")
        canonical_valid = self.archive.endpoint_view_valid[self.archive.row_endpoint_index[self._dense_ordered_rows]]
        legacy_valid = self.dense.view_valid[self.dense.sidecar_row]
        self.dense_validity_audit = {
            "legacy_vs_canonical_mismatch": int((legacy_valid != canonical_valid).sum()),
            "legacy_true_canonical_false": int((legacy_valid & ~canonical_valid).sum()),
            "canonical_valid_wrist_column": canonical_valid.sum((0, 1)).tolist(),
            "validity_authority": "canonical endpoint identity and matching frozen DINO cache",
        }
        self.counts = {"archive": self.archive.rows, "dense": len(self.dense_rows),
                       "fish": len(self.fish.starts), "appearance": len(self.dense.sidecar_row) * 5,
                       "cad": len(self.cad.training_rows), "real": len(self.real["rgb"])}
        self._initialize_source_identity()
        self._backbone = None
        self._real_dino = None
        self._fish_geometry = None
        self._kinematics = None
        self.device_cache = None

    def _initialize_source_identity(self):
        metadata = build_source_identity_metadata(self.archive, self.dense_rows,
            fish_source_identity=getattr(self.fish, "source_identity", None), fish_rows=self.counts["fish"])
        self._archive_source_index = metadata["archive_row_source_index"]
        self.fish_source_index = metadata["fish_source_index"]
        self.source_identity_audit = metadata["report"]

    def _feature_values(self, cache, endpoint_ids):
        """Frozen fp16 stays fp16 across PCIe; model-facing conversion is unchanged."""
        if self.device_cache is None:
            if getattr(self, "compact_host_features", False):
                rows = cache.rows_for(endpoint_ids)
                value = np.array(cache.tokens[rows], copy=True)
                available = np.asarray(cache.view_valid[rows], dtype=bool)
                # Same validity contract as cache.get, without expanding the
                # full field and allocating a second masked float32 field.
                if not np.isfinite(value).all() or np.any(value[~available] != 0):
                    raise V20DataError("frozen DINO data are nonfinite or invalid views are nonzero")
                return value, available
            return cache.get(endpoint_ids)
        rows = cache.rows_for(endpoint_ids)
        valid = np.asarray(cache.view_valid[rows], dtype=bool)
        if dequantizes_on_device(self.device_cache, cache.tokens):
            # section 17.65: an int8 store's rows reach the GPU as int8 values and scales and are multiplied there
            # (fcpg/v21_dequantize.py; bit for bit the float16 values below, same integrity contract)
            def load(selected):
                return quantized_rows_on_device(cache.tokens, selected, self.device_cache.device, cache.view_valid,
                                                error=V20DataError)
            return self.device_cache.get_rows((str(cache.root), "tokens"), rows, load), valid
        def load(selected):
            value = np.array(cache.tokens[selected], copy=True)
            available = np.asarray(cache.view_valid[selected], dtype=bool)
            if not np.isfinite(value).all() or np.any(value[~available] != 0):
                raise V20DataError("frozen DINO data are nonfinite or invalid views are nonzero")
            return value
        return self.device_cache.get_rows((str(cache.root), "tokens"), rows, load), valid

    def _get_archive_dino(self, endpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(endpoints, dtype=np.int64)
        host_dtype = np.float16 if getattr(self, "compact_host_features", False) else np.float32
        tokens = (np.zeros((len(requested), 2, ENCODER_PATCHES, ENCODER_DIM), dtype=host_dtype) if self.device_cache is None
                  else torch.zeros((len(requested), 2, ENCODER_PATCHES, ENCODER_DIM), dtype=torch.float16, device=self.device))
        valid = np.zeros((len(requested), 2), dtype=bool)
        done = np.zeros(len(requested), dtype=bool)
        for cache in self.archive_dino:
            index = np.searchsorted(cache.endpoint_index, requested)
            in_range = index < len(cache.endpoint_index)
            match = in_range & (cache.endpoint_index[np.minimum(index, len(cache.endpoint_index)-1)] == requested) & ~done
            if match.any():
                loaded, valid[match] = self._feature_values(cache, requested[match])
                if isinstance(tokens, torch.Tensor):
                    tokens.index_copy_(0, torch.as_tensor(np.flatnonzero(match), device=self.device), loaded)
                else:
                    tokens[match] = loaded
                done[match] = True
        if not done.all():
            raise V20DataError(f"missing frozen DINO endpoints: {requested[~done][:10].tolist()}")
        return tokens, valid

    def observation_batch(self, endpoint_ids: Any, kind: str = "archive") -> dict[str, Any]:
        if kind in ("archive", "dense"):
            result = self.rgb_source.get_numpy(endpoint_ids)
            dino, valid = self._get_archive_dino(result["endpoint_index"])
        elif kind == "fish":
            result = self.fish.observation(endpoint_ids)
            dino, valid = self._feature_values(self.fish_dino, result["endpoint_index"])
        else:
            raise V20DataError(f"unknown observation source {kind}")
        if not np.array_equal(valid, result["view_valid"]):
            raise V20DataError("DINO/RGB view validity disagrees")
        result["dino"] = dino
        result["geometry"] = self.fish_geometry() if kind == "fish" else None
        if kind == "fish":
            result["camera_available"] = np.ones((len(result["q_position"]), 2), dtype=bool)
        return result

    def dino_batch(self, endpoint_ids: Any, kind: str = "archive") -> dict[str, Any]:
        if kind in ("archive", "dense"):
            tokens, valid = self._get_archive_dino(np.asarray(endpoint_ids, dtype=np.int64))
        elif kind == "fish":
            ids = self.fish.validate_frames(endpoint_ids)
            tokens, valid = self._feature_values(self.fish_dino, ids)
        else:
            raise V20DataError("unknown DINO source")
        return {"dino": tokens, "view_valid": valid}

    def fish_geometry(self):
        if self._fish_geometry is None:
            from .prod_v20_1_geometry import V201FittedMechanicalProjection
            self._fish_geometry = V201FittedMechanicalProjection(self.paths.real_calibration, device=self.device,
                                                               kinematics=self.kinematics())
        return self._fish_geometry

    def kinematics(self):
        if self._kinematics is None:
            from .robot_kinematic_graph import DifferentiableRobotKinematics, so101_robot_spec
            self._kinematics = DifferentiableRobotKinematics(so101_robot_spec()).to(self.device).eval().requires_grad_(False)
        return self._kinematics

    def appearance_batch(self, indices: Any) -> dict[str, Any]:
        selected = _indices(indices, self.counts["appearance"])
        row, horizon = selected // 5, selected % 5
        rgb, legacy_valid = self.dense.batch(selected)
        endpoint = self.archive.row_endpoint_index[self._dense_ordered_rows[row], horizon]
        dino, valid = self._get_archive_dino(endpoint)
        # The 20K builder already reverses legacy slots into [wrist,column].
        if not np.array_equal(valid, self.archive.endpoint_view_valid[endpoint]):
            raise V20DataError("20K DINO view validity disagrees with canonical endpoint")
        rgb = rgb.copy()
        rgb[~valid] = 0
        return {"rgb": rgb, "dino": dino, "view_valid": valid, "endpoint_index": endpoint}

    def appearance_weights(self, indices):
        selected = _indices(indices, self.counts["appearance"])
        endpoints = self.archive.row_endpoint_index[self._dense_ordered_rows[selected // 5], selected % 5]
        return self.archive.endpoint_view_valid[endpoints].sum(-1).astype(np.float64)

    def spatial_normalization(self, indices, *, real=False):
        """Cheap labels-only effective-batch denominators, including OOM replay."""
        if real:
            rows, views = deterministic_real_indices(self.real["view_valid"], indices, self.seed)
            target = self.real["fm_px"][rows]
            labels = np.zeros((len(rows), 2, 2), dtype=bool)
            labels[np.arange(len(rows)), views] = True
            visible = np.ones_like(labels)
        else:
            rows = self.cad.training_rows[_indices(indices, self.counts["cad"])]
            target = np.asarray(self.cad.keypoints_px[rows], dtype=np.float32)[:, ::-1, 4:6]
            labels = np.ones(target.shape[:-1], dtype=bool)
            visible = np.asarray(self.cad.valid[rows], dtype=bool)[:, ::-1, 4:6]
        center = np.nan_to_num(target).mean(-2)
        center_known = ((labels & visible).all(-1) & np.isfinite(target).all((-1, -2))
                        & (center >= -.5).all(-1) & (center <= 223.5).all(-1))
        return dict(pair_view_counts=labels.any(-1).sum(0), center_view_counts=center_known.sum(0),
                    effective_rows=len(rows))

    def _cad_batch_cached(self, rows):
        """Same V20 CAD values/order, with RGB reads only for cache misses."""
        cad = self.cad
        def load_rgb(selected):
            return np.asarray(cad.rgb[selected])[:, ::-1].transpose(0, 1, 4, 2, 3).copy()
        rgb = self.device_cache.get_rows((str(cad.root), "rgb-wrist-column"), rows, load_rgb)
        points = np.asarray(cad.keypoints_px[rows], dtype=np.float32)[:, ::-1].copy()
        valid = np.asarray(cad.valid[rows], dtype=bool)[:, ::-1].copy()
        result = dict(row_index=rows, dino_index=rows, rgb=rgb,
            q_position=np.asarray(cad.state[rows], dtype=np.float32), q_available=np.ones(len(rows), dtype=bool),
            camera_from_reference=np.asarray(cad.camera_from_reference[rows])[:, ::-1].copy(),
            intrinsic=np.asarray(cad.intrinsic[rows])[:, ::-1].copy(),
            image_transform=np.broadcast_to(np.eye(3, dtype=np.float32), (len(rows), 2, 3, 3)).copy(),
            camera_available=np.ones((len(rows), 2), dtype=bool), view_valid=np.ones((len(rows), 2), dtype=bool),
            all_keypoints_px=points, all_keypoints_valid=valid, keypoints_base_xyz=np.asarray(cad.keypoints_base_xyz[rows]),
            fm_px=points[:, :, 4:6], fm_valid=valid[:, :, 4:6], ordered=np.ones(len(rows), dtype=bool))
        for name in ("q_position", "camera_from_reference", "intrinsic"):
            if not np.isfinite(result[name]).all():
                raise V20DataError(f"CAD {name} is nonfinite")
        return result

    def _encode_real_dino(self):
        from .prod_v18_1_training import load_frozen_dinov2, extract_dinov2_tokens
        if self._backbone is None:
            # Constructing the frozen backbone initializes parameters before
            # replacing them with pretrained weights. That must not consume
            # the resumed learner's RNG stream (e.g. dropout/prior noise).
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                self._backbone = load_frozen_dinov2(self.paths.dino_repository, self.paths.dino_weights, device=self.device)
        images = self.real["rgb"].transpose(0, 1, 3, 4, 2).reshape(-1, 224, 224, 3)
        value = extract_dinov2_tokens(self._backbone, images, device=self.device, batch_size=64)
        self._real_dino = np.asarray(value).reshape(len(self.real["rgb"]), 2, ENCODER_PATCHES, ENCODER_DIM).astype(np.float32)
        self._real_dino *= self.real["view_valid"][..., None, None]

    def cad_batch(self, indices: Any) -> dict[str, Any]:
        from .prod_v20_1_geometry import V201MechanicalProjection
        rows = self.cad.training_rows[_indices(indices, self.counts["cad"])]
        result = self.cad.batch(rows) if self.device_cache is None else self._cad_batch_cached(rows)
        result["dino"], valid = self._feature_values(self.cad_dino, rows)
        if not np.array_equal(valid, result["view_valid"]):
            raise V20DataError("CAD RGB/DINO validity differs")
        result["target_fm_px"] = result["fm_px"]
        # Renderer labels every point, but ``valid`` only tests projection
        # depth/in-frame. It is not an occlusion-tested positive visibility.
        result["point_valid"] = np.ones_like(result["fm_valid"])
        result["label_valid"] = result["point_valid"]
        result["visibility"] = result["fm_valid"].astype(np.float32)
        result["visibility_valid"] = ~result["fm_valid"]
        result["geometry"] = V201MechanicalProjection(self.kinematics(),
            torch.as_tensor(result["camera_from_reference"], device=self.device),
            torch.as_tensor(result["intrinsic"], device=self.device),
            torch.as_tensor(result["image_transform"], device=self.device),
            torch.as_tensor(result["camera_available"], device=self.device))
        return result

    def real_batch(self, indices: Any = None, *, batch_size: int | None = None) -> dict[str, Any]:
        if self._real_dino is None:
            self._encode_real_dino()
        count = int(batch_size if batch_size is not None else len(indices) if indices is not None else 64)
        rows, views = (deterministic_real_indices(self.real["view_valid"], indices, self.seed)
                       if indices is not None else balanced_real_indices(self.real["view_valid"], count, self.rng))
        result = {key: value[rows] for key, value in self.real.items() if isinstance(value, np.ndarray) and len(value) == self.counts["real"]}
        result["dino"] = self._real_dino[rows]
        # Forward both actual images; only the sampled view contributes loss.
        valid = np.zeros((count, 2, 2), dtype=bool)
        valid[np.arange(count), views] = True
        result["target_fm_px"], result["point_valid"] = result["fm_px"], valid
        result["label_valid"] = valid
        result["visibility"] = np.ones_like(valid, dtype=np.float32)
        result["visibility_valid"] = valid
        # Only the ordered panel belongs to this physical camera rig. Restored
        # q from eight other sources does NOT establish their camera geometry.
        from .prod_v20_1_geometry import V201FittedMechanicalProjection
        camera_valid = np.repeat(((result["source"] == 0) & result["q_available"])[:, None], 2, axis=1)
        result["geometry"] = V201FittedMechanicalProjection(self.paths.real_calibration,
            device=self.device, camera_valid=camera_valid, kinematics=self.fish_geometry().kinematics)
        result["camera_available"] = camera_valid
        return result

    def chain_batch(self, rows: Any, kind: str = "archive") -> dict[str, Any]:
        if kind == "fish":
            result = self.fish.chains(rows)
            result["geometry"] = self.fish_geometry()
            result["task_embedding"] = self.tasks.fish(len(result["q_position"]), device=torch.device("cpu")).numpy()
            result["source_index"] = np.full(len(result["q_position"]), self.fish_source_index, dtype=np.int64)
            return result
        if kind not in ("archive", "dense"):
            raise V20DataError("unknown chain source")
        selected = _indices(rows, self.counts[kind])
        archive_rows = selected if kind == "archive" else self.dense_rows[selected]
        chain = self.archive.chains(archive_rows)
        return {"rows": archive_rows, "q_position": chain.q_position,
                "source_index": self._archive_source_index[archive_rows].copy(),
                "commands_position": chain.commands_position,
                "measured_q_paths_position": chain.measured_q_paths_position,
                "time_deltas": np.diff(chain.readback_time_offsets_s, axis=-1),
                "view_valid": chain.view_valid, "endpoint_index": chain.endpoint_index,
                "task_embedding": self.tasks.for_rows(archive_rows, device=torch.device("cpu")).numpy(),
                "geometry": None, "initial_time_s": chain.endpoint_time_s[:, 0]}

    def real_evaluation_batch(self, indices: Any) -> dict[str, Any]:
        """Exact exposed label rows once, not a resampled training-fit batch."""
        from .prod_v20_1_geometry import V201FittedMechanicalProjection
        rows = _indices(indices, self.counts["real"])
        if self._real_dino is None:
            self._encode_real_dino()
        result = {key: value[rows] for key, value in self.real.items()
                  if isinstance(value, np.ndarray) and len(value) == self.counts["real"]}
        result["dino"] = self._real_dino[rows]
        result["target_fm_px"] = result["fm_px"]
        result["point_valid"] = result["fm_valid"]
        available = np.repeat(((result["source"] == 0) & result["q_available"])[:, None], 2, axis=1)
        result["geometry"] = V201FittedMechanicalProjection(self.paths.real_calibration,
            device=self.device, camera_valid=available, kinematics=self.fish_geometry().kinematics)
        result["camera_available"] = available
        return result

    def preflight(self, *, media: bool = False) -> dict[str, Any]:
        result = audit_archive_contract(self.archive)
        result.update(counts=self.counts, real_labels=self.real["audit"], paths=asdict(self.paths),
                      source_identity=self.source_identity_audit,
                      dense_view_validity=self.dense_validity_audit,
                      reused_dino_cache_completion=self.dino_completion_audit,
                      real_camera_geometry_available_rows=int((self.real["source"] == 0).sum()),
                      fish_camera_geometry_available=True,
                      real_camera_geometry_kind="existing-affine-pixel-fit-not-metric-extrinsics",
                      archive_rgb_path="source-video-letterbox224 + existing DINO", fish_episode_2_opened=False,
                      geometry_availability={
                          "archive_chain_rows": {"available": 0, "unavailable": self.counts["archive"]},
                          "dense_chain_rows": {"available": 0, "unavailable": self.counts["dense"]},
                          "fish_chain_rows": {"available": self.counts["fish"], "unavailable": 0},
                          "real_rows": {"available": int(((self.real["source"] == 0) & self.real["q_available"]).sum()),
                                        "total": self.counts["real"]},
                          "dataset_id_is_not_camera_geometry": True})
        if media:
            result["media"] = self.rgb_source.audit_media()
        return result

    def development_manifest(self, count: int, *, excluded_rows=(), seed: int | None = None):
        """Stratify only exposed archive metadata; never open fish episode 2."""
        archive = self.archive
        episode_index = archive.endpoint_episode_index[archive.row_endpoint_index[:, 0]]
        sources = []
        for episode in archive.episodes:
            source = canonical_archive_source(episode)
            if source is None:
                raise V20DataError("development strata require actual dataset source metadata")
            sources.append(str(source))
        row_sources = np.asarray(sources, dtype=object)[episode_index]
        tasks = np.asarray(archive.task_names, dtype=object)[archive.task_index]
        return stratified_development_manifest(row_sources, tasks, archive.chain_id, count,
            seed=self.seed if seed is None else int(seed), excluded_rows=excluded_rows,
            episode_indices=episode_index)

    def servo_training_prefix(self, evaluation_rows, count: int = 256):
        """Early training chains only; same-episode prefixes precede evaluation."""
        archive = self.archive
        selected = _indices(evaluation_rows, archive.rows)
        ids = archive.row_endpoint_index
        episode = archive.endpoint_episode_index[ids[:, 0]]
        frames = archive.endpoint_local_frame_index[ids]
        boundaries = {}
        for ep in np.unique(episode):
            evaluation = selected[episode[selected] == ep]
            if len(evaluation):
                boundaries[int(ep)] = int(frames[evaluation, 0].min())
            else:
                epframes = frames[episode == ep]
                boundaries[int(ep)] = int(epframes[:, 0].min() + .1 * (epframes[:, -1].max() - epframes[:, 0].min()))
        eligible = frames[:, -1] < np.asarray([boundaries[int(ep)] for ep in episode])
        forbidden = np.unique(ids[selected])
        eligible &= ~np.isin(ids, forbidden).any(-1)
        groups = {}
        for row in np.flatnonzero(eligible):
            source = canonical_archive_source(archive.episodes[int(episode[row])])
            if not source:
                raise V20DataError("servo training prefix requires source metadata")
            groups.setdefault((str(source), int(archive.task_index[row])), []).append(int(row))
        for rows in groups.values():
            rows.sort(key=lambda row: (int(episode[row]), int(frames[row, 0]), row))
        result, depth, keys = [], 0, sorted(groups)
        while len(result) < min(count, int(eligible.sum())):
            for key in keys:
                if depth < len(groups[key]):
                    result.append(groups[key][depth])
                    if len(result) == count:
                        break
            depth += 1
        return dict(rows=result, role="servo-fit-training-prefix", held_episode_opened=False,
                    evaluation_endpoint_overlap=False, same_episode_prefix_precedes_evaluation=True,
                    other_episode_prefix_fraction=.1, source_task_strata=len(groups),
                    evidence="training-development")

    def close(self):
        self.rgb_source.close()


def stratified_development_manifest(source_ids, task_texts, chain_ids, count: int, *,
                                    seed: int = 20100, excluded_rows=(), episode_indices=None):
    """Deterministic round-robin over true source/task strata, not first-N rows.

    This remains exposed development evidence. Sampling order and every chosen
    identity are published, and missing strata are counted rather than hidden.
    """
    source_ids, task_texts, chain_ids = map(np.asarray, (source_ids, task_texts, chain_ids))
    if count < 1 or source_ids.ndim != 1 or task_texts.shape != source_ids.shape or chain_ids.shape != source_ids.shape:
        raise V20DataError("development metadata/count shapes differ")
    excluded = set(int(x) for x in excluded_rows)
    if any(x < 0 or x >= len(source_ids) for x in excluded):
        raise V20DataError("excluded development row is out of range")
    groups = {}
    for row, (source, task) in enumerate(zip(source_ids, task_texts)):
        source, task = str(source).strip(), " ".join(str(task).split())
        if not source or not task:
            raise V20DataError("development source/task metadata cannot be blank")
        if row not in excluded:
            groups.setdefault((source, task), []).append(row)
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    groups = {key: rng.permutation(groups[key]).tolist() for key in keys}
    selected, depth = [], 0
    while len(selected) < min(count, sum(len(value) for value in groups.values())):
        for key in keys:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                if len(selected) == count:
                    break
        depth += 1
    records = []
    for row in selected:
        record = dict(row=int(row), chain_id=int(chain_ids[row]), source_id=str(source_ids[row]),
                      task_text=str(task_texts[row]), camera_geometry_available=False)
        if episode_indices is not None:
            record["episode_index"] = int(episode_indices[row])
        records.append(record)
    covered = {(str(source_ids[row]), " ".join(str(task_texts[row]).split())) for row in selected}
    return dict(protocol="exposed-source-task-stratified-v201", seed=int(seed),
                evidence="training-development", unseen_claim=False, held_episode_opened=False,
                requested_rows=int(count), selected_rows=len(selected), total_strata=len(keys),
                represented_strata=len(covered), uncovered_strata=len(keys) - len(covered),
                records=records, geometry_available_rows=0, geometry_unavailable_rows=len(selected))


__all__ = ["V201SourcePaths", "V201Sources", "V201FishTrainingSource", "validate_dino_cache_completion", "stratified_development_manifest", "canonical_archive_source", "build_source_identity_metadata"]
