"""V21 uses native text tokens and existing frozen DINO caches, never MiniLM.

Full recorded command traces are restored from canonical temporal arrays. A
five-command BC target is not silently treated as the complete actuator input.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import re
import numpy as np
import torch

from .v21_encoder import DIM, PATCHES
from .prod_v20_data import V20TemporalArchive, _array
from .prod_v20_1_sources import V201FishTrainingSource, V201Sources, validate_dino_cache_completion
from .prod_v18_1_data import V181FeatureCache
from .v21_models import CommandTrace
from .v21_p0_features import open_archive_cache


@dataclass(frozen=True)
class TaskVocabulary:
    words: tuple[str, ...]
    max_length: int = 64

    @staticmethod
    def tokenize(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("each chain needs a nonempty task instruction")
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]|[^\s]", text.casefold())

    @classmethod
    def build(cls, texts):
        texts = list(texts)
        tokens = sorted({word for text in texts for word in cls.tokenize(text)})
        result = cls(("<pad>", "<unk>", "<bos>", "<eos>", *tokens))
        result.encode(list(texts))
        return result

    def encode(self, texts, *, device="cpu"):
        if self.words[:4] != ("<pad>", "<unk>", "<bos>", "<eos>") or len(set(self.words)) != len(self.words):
            raise ValueError("invalid vocabulary contract")
        lookup = {word: i for i, word in enumerate(self.words)}
        rows = [[2, *[lookup.get(word, 1) for word in self.tokenize(text)], 3] for text in texts]
        if not rows or max(map(len, rows)) > self.max_length:
            raise ValueError("empty text batch or instruction too long; no silent truncation")
        values = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long, device=device)
        for i, row in enumerate(rows):
            values[i, :len(row)] = torch.tensor(row, device=device)
        return values


@dataclass(frozen=True)
class TrainingChain:
    q: torch.Tensor                    # [B,5,6], actual H0..H4
    endpoint_times: torch.Tensor       # [B,5], float64
    commands: torch.Tensor             # [B,4,5,6], BC targets
    paths: torch.Tensor                # [B,4,6,6], actual q-path
    readback_offsets: torch.Tensor     # [B,4,6], float64
    traces: tuple[CommandTrace, ...]    # four FULL recorded command sequences
    features: torch.Tensor | None      # [B,5,2,PATCHES,DIM], frozen actual RGB features (fcpg/v21_encoder.py)
    view_valid: torch.Tensor           # [B,5,2]
    text_ids: torch.Tensor
    source_index: torch.Tensor         # missing source = -1
    object_relative: torch.Tensor | None = None  # [B,5,8] wrist object-vs-jaw grounding; None = zeros
    actor_visual_drop: torch.Tensor | None = None  # [B], hide Actor feature branch, not camera delivery
    capture_times: torch.Tensor | None = None  # [B,5,2], actual decoded exposure clock
    capture_q: torch.Tensor | None = None      # [B,5,2,6], causal readback at exposure
    camera_types: torch.Tensor | None = None   # [B,2], reviewed static sensor metadata
    place_relative: torch.Tensor | None = None   # [B,5,8] wrist place-vs-jaw grounding (fish rows); None = zeros
    place_goal: torch.Tensor | None = None       # [B,5,8] hindsight place goal vs the jaws (fcpg/v21_place_goal.py)

    def slice(self, start, stop):
        index = slice(start, stop)
        return TrainingChain(self.q[index], self.endpoint_times[index], self.commands[index], self.paths[index],
            self.readback_offsets[index], tuple(CommandTrace(t.positions[index], t.offsets[index], t.valid[index], t.source, t.complete) for t in self.traces),
            None if self.features is None else self.features[index], self.view_valid[index],
            self.text_ids[index], self.source_index[index],
            None if self.object_relative is None else self.object_relative[index],
            None if self.actor_visual_drop is None else self.actor_visual_drop[index],
            None if self.capture_times is None else self.capture_times[index],
            None if self.capture_q is None else self.capture_q[index],
            None if self.camera_types is None else self.camera_types[index],
            None if self.place_relative is None else self.place_relative[index],
            None if self.place_goal is None else self.place_goal[index])


class V21Sources:
    # Reuse the established endpoint-index gather/LRU, without constructing
    # its old observer, RGB decoder, CAD loader or task-embedding cache.
    _feature_values = V201Sources._feature_values

    def _get_archive_dino(self, ids):
        tokens, valid = V201Sources._get_archive_dino(self, ids)
        roles = getattr(self, 'archive_camera_roles', None)
        if roles is not None:
            return roles.remap(tokens, ids), roles.remap(valid, ids)
        return tokens, valid

    def __init__(self, paths, *, device="cpu", expected_rows=320000, expected_endpoints=572521, object_relative_table=None, archive_object_relative_table=None, motion_only=False, place_relative_table=None,
                 place_goal_tables=None):
        if type(motion_only) is not bool:
            raise ValueError("motion_only must be an explicit boolean")
        self.motion_only = motion_only
        self.device = torch.device(device)
        self.device_cache = None
        self.compact_host_features = True
        self.paths = dict(paths)
        # The 1.2M source config states its own row/endpoint counts (scripts/build_v21_1p2m_training_roots.py);
        # every other config keeps the 320k contract.
        expected_rows = int(paths.get("expected_rows", expected_rows))
        expected_endpoints = int(paths.get("expected_endpoints", expected_endpoints))
        self.archive = V20TemporalArchive(paths["endpoint_root"], paths["companion_root"], paths["temporal_root"],
                                         expected_rows=expected_rows, expected_endpoints=expected_endpoints)
        root = Path(paths["temporal_root"])
        self.raw_actions, self.raw_q, self.offsets = (_array(root, name) for name in ("actions", "states", "episode_offsets"))
        if self.raw_actions.shape != self.raw_q.shape or self.raw_actions.shape[-1] != 6:
            raise ValueError("complete canonical command/readback arrays required")
        # The 20K RGB subset joins only the 320k/1.2M chains; the half-second Archive (scripts/build_v21_halfsecond_
        # training_roots.py) has no such subset (rgb_root null), so its dense population is empty.
        self.dense_rows = (self.archive.dense_rows(paths["rgb_root"]) if paths.get("rgb_root")
                           else np.zeros(0, np.int64))
        self.fish = V201FishTrainingSource(paths["fish_wrist_root"], paths["fish_column_root"])
        # Optional precomputed wrist object-vs-jaw grounding per fish frame; archive never has one.
        self.object_relative_table = None
        if object_relative_table is not None:
            table = np.load(object_relative_table)
            if table.shape != (len(self.fish.q), 8) or not np.isfinite(table).all():
                raise ValueError("object-relative table must be finite [fish_frames, 8]")
            self.object_relative_table = np.asarray(table, np.float32)
        # Optional place table, one row per fish frame like the object table (docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_
        # 20261012.md section 17); the archive never has one.
        self.place_relative_table = None
        if place_relative_table is not None:
            table = np.load(place_relative_table)
            if table.shape != (len(self.fish.q), 8) or not np.isfinite(table).all():
                raise ValueError("place-relative table must be finite [fish_frames, 8]")
            self.place_relative_table = np.asarray(table, np.float32)
        # Optional place-goal tables (training overhaul doc section 17.7; scripts/build_v21_place_goal_tables.py): one
        # row per global Archive endpoint id and one per fish frame, zeros where no goal is labelled.
        self.archive_place_goal_table = self.fish_place_goal_table = None
        if place_goal_tables is not None:
            directory = Path(place_goal_tables)
            archive_goals = np.load(directory / 'archive_place_goal.npy')
            fish_goals = np.load(directory / 'fish_place_goal.npy')
            if (archive_goals.shape != (expected_endpoints, 8) or fish_goals.shape != (len(self.fish.q), 8)
                    or not np.isfinite(archive_goals).all() or not np.isfinite(fish_goals).all()):
                raise ValueError('place-goal tables must be finite [archive endpoints, 8] and [fish frames, 8]')
            self.archive_place_goal_table = np.asarray(archive_goals, np.float32)
            self.fish_place_goal_table = np.asarray(fish_goals, np.float32)
        # Optional archive table, one row per global endpoint id (zeros = no object grounding).
        self.archive_object_relative_table = None
        if archive_object_relative_table is not None:
            table = np.load(archive_object_relative_table)
            if table.ndim != 2 or table.shape[1] != 8 or table.shape[0] < expected_endpoints or not np.isfinite(table).all():
                raise ValueError("archive object-relative table must be finite [>=endpoints, 8]")
            self.archive_object_relative_table = np.asarray(table, np.float32)
        manifest = json.loads((Path(paths["fish_wrist_root"]) / "manifest.json").read_text())
        names = manifest["task_names"]
        names = list(names.values()) if isinstance(names, dict) else list(names)
        if len(names) != 1:
            raise ValueError("fish task-to-row mapping is not uniquely specified")
        self.fish_task = names[0]
        self.vocabulary = TaskVocabulary.build([*self.archive.task_names, self.fish_task])
        # LeWM consumes q, complete commands and actual motion history only.
        # Explicit motion-only sources never open an incomplete visual cache;
        # they also cannot be used as a zero-feature substitute for the Actor.
        if not motion_only:
            cache = Path(paths["dino_cache_root"])
            # Combined P0 partitions (old 320k tokens + fresh lossless blocks) or the plain V18.1 cache.
            self.archive_dino = [open_archive_cache(cache / name) for name in ("temporal-train", "temporal-dev")]
            self.fish_dino = V181FeatureCache(cache / "fish")
            for value in (*self.archive_dino, self.fish_dino):
                validate_dino_cache_completion(value.root, value.view_valid)
        self.recorded_clock = None
        if paths.get('dino_clock_version'):
            if motion_only:
                raise ValueError('recorded-clock migration requires the complete paired visual cache')
            from .v21_clock_cache import RecordedClockCache
            self.recorded_clock = RecordedClockCache(paths, self.archive, self.fish,
                                                     [*self.archive_dino, self.fish_dino])
            self.archive.endpoint_view_valid = self.recorded_clock.tables['archive']['valid']
        elif not motion_only and (Path(paths['dino_cache_root'])/'manifest.json').is_file():
            meta = json.loads((Path(paths['dino_cache_root'])/'manifest.json').read_text())
            if meta.get('clock_version'):
                raise ValueError('new DINO cache requires explicit clock/extraction version; no legacy fallback')
        self.archive_camera_roles = None
        self.camera_types = None
        if paths.get('camera_type_contract') is not None:
            if self.recorded_clock is None or paths.get('archive_camera_role_adapter') is not None:
                raise ValueError('typed cameras require recorded clocks and original, unpermuted slots')
            from .v21_camera_types import ReviewedCameraTypes
            self.camera_types = ReviewedCameraTypes(paths['camera_type_contract'], self.archive, paths)
        if paths.get('archive_camera_role_adapter') is not None:
            if self.recorded_clock is None:
                raise ValueError('reviewed role adapter requires paired capture-time/q clock data')
            from .v21_archive_camera_roles import ReviewedArchiveRoles, RoleAwareClock
            self.archive_camera_roles = ReviewedArchiveRoles(paths['archive_camera_role_adapter'], self.archive)
            self.recorded_clock = RoleAwareClock(self.recorded_clock, self.archive_camera_roles)
        labels = [str(e.get("manifest_row", {}).get("dataset_root", "")) for e in self.archive.episodes]
        fish_label = str(manifest.get("source_root", ""))
        vocabulary = {name: i for i, name in enumerate(sorted(set(labels + [fish_label]) - {""}))}
        self.episode_sources = np.asarray([vocabulary.get(name, -1) for name in labels], dtype=np.int64)
        self.fish_source = vocabulary.get(fish_label, -1)
        self.counts = dict(archive=self.archive.rows, dense=len(self.dense_rows), fish=len(self.fish.starts))

    def _tensor(self, value, dtype=None):
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=dtype or value.dtype)
        result = torch.from_numpy(np.array(value, copy=True))
        return result.to(device=self.device, dtype=dtype or result.dtype)

    def chain(self, rows, kind="archive", *, images=True):
        if images and getattr(self, "motion_only", False):
            raise ValueError("motion-only sources cannot provide Actor visual features")
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or not len(rows) or (rows < 0).any() or (rows >= self.counts[kind]).any():
            raise ValueError("invalid chain rows")
        if kind == "dense":
            rows = self.dense_rows[rows]
        b = len(rows)
        if kind == "fish":
            legacy = self.fish.chains(rows)
            endpoint = legacy["endpoint_index"]
            # Every access to fish pixels/q/actions is after this exclusion.
            self.fish.validate_frames(endpoint.reshape(-1))
            local = np.asarray(self.fish.frame_indices[endpoint])
            starts = endpoint[:, :-1]
            ends = endpoint[:, 1:]
            qbank, abank = self.fish.q, self.fish.actions
            fps = np.full(b, 30., dtype=np.float64)
            path_ids = np.concatenate((starts[..., None] + np.arange(5)*3, ends[..., None]), -1)
            self.fish.validate_frames(path_ids.reshape(-1))
            text = [self.fish_task] * b
            source = np.full(b, self.fish_source, np.int64)
            expected_commands, expected_q = legacy["commands_position"], legacy["q_position"]
            image_ids = endpoint
            absolute_endpoint = endpoint
            valid = legacy["view_valid"]
        else:
            legacy = self.archive.chain_batch(rows)
            endpoint = legacy.endpoint_index
            local = np.asarray(self.archive.endpoint_local_frame_index[endpoint])
            ep = self.archive.endpoint_episode_index[endpoint[:, 0]]
            bases = np.asarray(self.offsets[ep], dtype=np.int64)
            if any(int(self.archive.episodes[int(e)]["frame_start"]) != int(base) for e, base in zip(ep, bases)):
                raise ValueError("canonical episode offsets disagree")
            absolute_endpoint = bases[:, None] + local
            starts, ends = absolute_endpoint[:, :-1], absolute_endpoint[:, 1:]
            act_local = np.asarray(self.archive.action_frame_indices[self.archive.chain_id[rows]]).reshape(b, 4, 5)
            path_ids = np.concatenate((bases[:, None, None] + act_local, ends[..., None]), -1)
            qbank, abank = self.raw_q, self.raw_actions
            fps = self.archive.fps[ep]
            text, source = legacy.task_text, self.episode_sources[ep]
            expected_commands, expected_q = legacy.commands_position, legacy.q_position
            image_ids, valid = endpoint, legacy.view_valid
        q = np.asarray(qbank[absolute_endpoint], dtype=np.float32)
        paths = np.asarray(qbank[path_ids], dtype=np.float32)
        commands = np.asarray(abank[path_ids[:, :, :-1]], dtype=np.float32)
        if not np.allclose(q, expected_q, atol=.02, rtol=0) or not np.allclose(commands, expected_commands, atol=.02, rtol=0):
            raise ValueError("canonical raw positions disagree with restored companion/chain units")
        if not np.isfinite(q).all() or not np.isfinite(paths).all() or not np.isfinite(commands).all():
            raise ValueError("nonfinite recorded motion")
        clock = getattr(self, 'recorded_clock', None)
        timebank = None if clock is None else clock.times(kind)
        endpoint_times = local/fps[:, None] if timebank is None else np.asarray(timebank[absolute_endpoint])
        offsets = ((path_ids-starts[..., None]) / fps[:, None, None] if timebank is None
                   else timebank[path_ids]-timebank[starts][..., None])
        capture_times = capture_q = None
        if clock is not None:
            valid, capture_times, capture_q = clock.observations(image_ids, kind)
            if not np.isfinite(endpoint_times).all() or (capture_times > endpoint_times[..., None]).any():
                raise ValueError('future or missing capture time in current visual input')
        traces = []
        for h in range(4):
            lengths = ends[:, h]-starts[:, h]
            if (lengths < 1).any() or (lengths > 128).any():
                raise ValueError("unsupported recorded command span; no truncation allowed")
            m = int(lengths.max())
            values = np.zeros((b, m, 6), np.float32); times = np.zeros((b, m), np.float64)
            available = np.zeros((b, m), bool)
            for i, (begin, end) in enumerate(zip(starts[:, h], ends[:, h])):
                ids = np.arange(begin, end)
                if kind == "fish":
                    self.fish.validate_frames(ids)
                values[i, :len(ids)] = abank[ids]
                times[i, :len(ids)] = (np.arange(len(ids))/fps[i] if timebank is None
                                      else timebank[ids]-timebank[begin])
                available[i, :len(ids)] = True
            trace = CommandTrace(self._tensor(values), self._tensor(times), self._tensor(available), "recorded", True)
            trace.validate(b); traces.append(trace)
        features = None
        if images:
            flat = image_ids.reshape(-1)
            unique, inverse = np.unique(flat, return_inverse=True)
            if kind == "fish":
                self.fish.validate_frames(unique)
                tokens, checked = self._feature_values(self.fish_dino, unique)
            else:
                tokens, checked = self._get_archive_dino(unique)
            if not np.array_equal(checked[inverse].reshape(b, 5, 2), valid):
                raise ValueError("DINO and current chain view identities differ")
            tensor = self._tensor(tokens)
            features = tensor[torch.as_tensor(inverse, device=self.device)].reshape(b, 5, 2, PATCHES, DIM)
        relative = np.zeros((b, 5, 8), np.float32)
        table = getattr(self, "object_relative_table", None)
        if kind == "fish" and table is not None:
            relative = table[endpoint]
        place_table = getattr(self, "place_relative_table", None)
        place = None
        if place_table is not None:
            # section 17.117: Archive rows carry the place cue when fcpg/v21_outline_cue.attach_archive_cues gave one
            archive_place = getattr(self, "archive_place_relative_table", None)
            place = (place_table[endpoint] if kind == "fish" else
                     archive_place[np.asarray(endpoint, np.int64)] if archive_place is not None else
                     np.zeros((b, 5, 8), np.float32))
        goal = None
        if getattr(self, "archive_place_goal_table", None) is not None:
            goal = (self.fish_place_goal_table if kind == "fish" else self.archive_place_goal_table)[
                np.asarray(endpoint, np.int64)]
        archive_table = getattr(self, "archive_object_relative_table", None)
        if kind != "fish" and archive_table is not None:
            relative = archive_table[np.asarray(endpoint, np.int64)]
            roles = getattr(self, 'archive_camera_roles', None)
            # outline cues (fcpg/v21_outline_cue.attach_archive_cues, section 17.30) were drawn on the camera this
            # layout calls the wrist, so they stay with it when the layout reorders the views
            if roles is not None and not getattr(self, 'archive_object_cues_follow_layout', False):
                swapped = (roles.orders(endpoint) == [1, 0]).all(-1)
                if np.any(relative[swapped] != 0):
                    raise ValueError('wrist-only object cues cannot be remapped without a paired-view contract')
        return TrainingChain(self._tensor(q), self._tensor(endpoint_times), self._tensor(commands),
            self._tensor(paths), self._tensor(offsets), tuple(traces), features, self._tensor(valid),
            self.vocabulary.encode(text, device=self.device), self._tensor(source), self._tensor(relative),
            capture_times=None if capture_times is None else self._tensor(capture_times),
            capture_q=None if capture_q is None else self._tensor(capture_q),
            camera_types=(None if getattr(self, 'camera_types', None) is None else
                          self._tensor(self.camera_types.for_chain(endpoint, kind, self.fish))),
            place_relative=None if place is None else self._tensor(place),
            place_goal=None if goal is None else self._tensor(goal))

    def audit(self):
        reports = {}
        images = not getattr(self, "motion_only", False)
        for kind, size in self.counts.items():
            rows = np.unique(np.asarray([0, size//2, size-1]))
            batch = self.chain(rows, kind, images=images)
            reports[kind] = dict(rows_checked=rows.tolist(), count=size,
                command_counts=[x.valid.sum(-1).tolist() for x in batch.traces],
                real_features=None if batch.features is None else list(batch.features.shape), task_tokens=list(batch.text_ids.shape))
        return dict(contract="v21-full-recorded-command-traces-v1", counts=self.counts,
                    samples=reports, full_population_audited=False, episode_2_values_read=False,
                    all_archive_rows_training=True, actor_archive_pair_exposures=self.counts["archive"]*4*4,
                    vocabulary_size=len(self.vocabulary.words), text_encoder="native-word-embedding")

    def prefetch(self, rows, kind="archive", block=128):
        """Sorted bounded reads, one CPU block ahead, upload on the owning thread."""
        if getattr(self, "motion_only", False):
            raise ValueError("motion-only sources cannot prefetch Actor visual features")
        if self.device_cache is None:
            return
        from concurrent.futures import ThreadPoolExecutor
        rows = np.asarray(rows, dtype=np.int64)
        if kind == "dense":
            rows = self.dense_rows[rows]
        if kind == "fish":
            ids = np.unique(self.fish.chains(rows)["endpoint_index"])
            self.fish.validate_frames(ids)
            caches = [self.fish_dino]
        else:
            ids = np.unique(self.archive.row_endpoint_index[rows])
            caches = self.archive_dino
        for cache in caches:
            positions = np.searchsorted(cache.endpoint_index, ids)
            inside = positions < len(cache.endpoint_index)
            matched = inside & (cache.endpoint_index[np.minimum(positions, len(cache.endpoint_index)-1)] == ids)
            selected = np.unique(positions[matched])
            chunks = [selected[i:i+block] for i in range(0, len(selected), block)]
            def read(index):
                value = np.array(cache.tokens[index], copy=True)
                valid = np.asarray(cache.view_valid[index])
                if not np.isfinite(value).all() or np.any(value[~valid] != 0):
                    raise ValueError("invalid cached visual features")
                return value
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(read, chunks[0]) if chunks else None
                for i, index in enumerate(chunks):
                    value = future.result()
                    future = pool.submit(read, chunks[i+1]) if i+1 < len(chunks) else None
                    self.device_cache.put_rows((str(cache.root), "tokens"), index, value)
