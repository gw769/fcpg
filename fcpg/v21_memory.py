"""Actual-only visual memory and readback-addressed attention, no hardware I/O."""
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .prod_v20_policy import ActualHistory, DEFAULT_LOWER, DEFAULT_UPPER, _validate_history
from . import v21_compiled
from .v21_checks import checking, deferring, fail_if
from .v21_compiled import region
from .v21_encoder import DIM, PATCHES, patch_xy        # DINOv2-S 256 x 384, DINOv3-L 196 x 1024 (V21_ENCODER)

ADDRESS = 27        # concat_inputs: the read address as the keys read it (joints at capture, their change since,
                    # recent travel and error, log age, camera); the patch position comes with the picture
POSE_POSITION = 72  # pose_position (11/03): each patch's position basis (x, y, sin and cos of pi x and pi y: 6) times
                    # its picture's joints at capture and their change since (12), joined to the keys


def pose_position_basis():
    """[PATCHES, 6]: each patch centre's x, y, sin and cos of pi x and pi y (patch_xy in [-1, 1])."""
    xy = patch_xy()
    return torch.cat((xy, torch.sin(math.pi * xy), torch.cos(math.pi * xy)), -1)


def pose_position(metadata):
    """Per-patch metadata [..., ADDRESS + 2] (the read address, then the patch position) -> [..., POSE_POSITION]: the
    patch's position basis times its picture's joints at capture and their change since, zero where the metadata is.
    In a key read by one linear layer, the read address alone adds the same amount to every patch of a picture, so the
    joints at capture could only weigh whole pictures (11/03: swapping them moved the release Actor's commands 0.0000);
    these products let them move where in a picture the reader looks."""
    xy = metadata[..., ADDRESS:ADDRESS + 2]
    basis = torch.cat((xy, torch.sin(math.pi * xy), torch.cos(math.pi * xy)), -1)
    return (basis[..., :, None] * metadata[..., None, :12]).flatten(-2)


def finite(name, value, shape=None):
    if not isinstance(value, torch.Tensor) or (shape is not None and tuple(value.shape) != tuple(shape)):
        raise ValueError(f"{name}: invalid tensor shape")
    if not value.is_floating_point():
        raise ValueError(f"{name}: expected finite floating values")
    if checking():
        fail_if(~torch.isfinite(value).all(), f"{name}: expected finite floating values")


def finite_where(name, value, mask):
    """finite(name, value[mask]) without selecting the rows (a selection makes the CPU wait for the GPU)."""
    if not value.is_floating_point():
        raise ValueError(f"{name}: expected finite floating values")
    mask = mask.reshape(*mask.shape, *([1] * (value.ndim - mask.ndim)))
    if checking():
        fail_if(~(torch.isfinite(value) | ~mask).all(), f"{name}: expected finite floating values")


def clock(name, value, shape):
    finite(name, value, shape)
    if value.dtype != torch.float64:
        raise ValueError(f"{name}: timestamps must be float64 seconds")


# Section 17.84: the memory's largest element-wise steps, each one compiled program inside a training update
# (fcpg/v21_compiled.py); the same values (they only compare, select and move values).
@region
def _pictures_finite(features, mask):
    """finite_where()'s condition for [B,2,PATCHES,DIM] pictures and a [B,2] mask."""
    return (torch.isfinite(features) | ~mask[..., None, None]).all()


@region
def _changed(features, memory_features, q, memory_q):
    """[B,2]: the exposure's pictures or pose differ from the newest ones in the memory."""
    return ((features.to(memory_features) != memory_features).flatten(2).any(-1)
            | (q.to(memory_q) != memory_q).any(-1))


def _shift(old, new, select):
    """_shifted_in's values outside the compiled regions (17.173): torch.where already makes a new tensor, so without
    the copy of it that followed (a third of the memory shifts' GPU time in a b128 LeWM update)."""
    return torch.where(select, torch.cat((old[:, :, 1:], new[:, :, None]), dim=2), old)


@region
def _shifted_in(old, new, select):
    """The memory's [B,2,3,...] values with `new` [B,2,...] as the newest slot of the selected views."""
    shifted = torch.cat((old[:, :, 1:], new[:, :, None]), dim=2)
    return torch.where(select, shifted, old).clone()


@dataclass(frozen=True)
class VisualObservation:
    features: torch.Tensor                 # [B,2,PATCHES,DIM], frozen RGB encoder (fcpg/v21_encoder.py)
    q_at_image: torch.Tensor               # [B,2,6], actual q at each exposure
    image_times: torch.Tensor              # [B,2], float64
    measurement_available: torch.Tensor    # [B,2]
    object_relative: torch.Tensor | None = None  # [B,8], wrist object vs jaws; None = zeros
    source: str = "actual_rgb"
    view_order: tuple[str, str] = ("wrist", "column")
    camera_types: torch.Tensor | None = None  # research opt-in [B,2], static per stream
    place_relative: torch.Tensor | None = None  # [B,8], where the object goes (wrist drop zone) vs jaws, same layout; None = zeros
    place_goal: torch.Tensor | None = None      # [B,8], the place goal vs the jaws (fcpg/v21_place_goal.py); None = zeros


@dataclass(frozen=True)
class VisualMemory:
    features: torch.Tensor                 # [B,2,3,PATCHES,DIM], actual values only
    q_at_image: torch.Tensor               # [B,2,3,6]
    image_times: torch.Tensor              # [B,2,3]
    valid: torch.Tensor                    # [B,2,3], oldest -> newest
    readback_indices: torch.Tensor | None = None  # [B,2,3], actual completed q1..q5 sample index at capture
    current_readback_index: torch.Tensor | None = None  # [B], advances only after completed actual block
    camera_types: torch.Tensor | None = None  # type metadata, never an RGB observation/target

    @classmethod
    def empty(cls, batch, *, device="cpu", dtype=torch.float32):
        with torch.inference_mode(False), torch.no_grad():
            return cls(torch.zeros(batch, 2, 3, PATCHES, DIM, device=device, dtype=dtype),
                       torch.zeros(batch, 2, 3, 6, device=device),
                       torch.zeros(batch, 2, 3, device=device, dtype=torch.float64),
                       torch.zeros(batch, 2, 3, device=device, dtype=torch.bool),
                       torch.zeros(batch, 2, 3, device=device, dtype=torch.long),
                       torch.zeros(batch, device=device, dtype=torch.long))

    @property
    def available(self):
        return self.valid.any(-1)

    def validate_observation(self, observation, now):
        """Validate before either insertion OR same-pose replacement, without mutation."""
        if not isinstance(observation, VisualObservation) or observation.source != "actual_rgb":
            raise ValueError("memory accepts actual RGB observations only, never forecasts")
        b = len(self.features)
        if observation.camera_types is None:
            if observation.view_order != ("wrist", "column") or self.camera_types is not None:
                raise ValueError('legacy view order or camera type stream changed')
        else:
            from .v21_camera_types import validate_camera_types
            types = validate_camera_types(observation.camera_types, b)
            if observation.view_order != ('slot0', 'slot1'):
                raise ValueError('typed cameras retain original slot0,slot1 order')
            if self.camera_types is None:
                if self.valid.any():
                    raise ValueError('cannot relabel an existing untyped visual history')
            elif not torch.equal(types.to(self.camera_types.device), self.camera_types):
                raise ValueError('camera types cannot change inside one visual history')
        clock("query time", now, (b,))
        if observation.features.shape != (b, 2, PATCHES, DIM):
            raise ValueError(f"observation features must be [B,2,{PATCHES},{DIM}]")
        mask = observation.measurement_available
        if mask.shape != (b, 2) or mask.dtype != torch.bool:
            raise ValueError("measurement availability must be bool [B,2]")
        if observation.q_at_image.shape != (b, 2, 6) or observation.image_times.shape != (b, 2):
            raise ValueError("per-view image q/clock shape differs")
        # section 17.80: the same checks on the selected views, written without selecting them
        finite_where("image time", observation.image_times, mask)
        if observation.image_times.dtype != torch.float64:
            raise ValueError("image time: timestamps must be float64 seconds")
        finite_where("image q", observation.q_at_image, mask)
        if not observation.features.is_floating_point():
            raise ValueError("actual image features: expected finite floating values")
        if checking():
            fail_if(~_pictures_finite(observation.features, mask), "actual image features: expected finite floating values")
        if checking():
            fail_if((mask & (observation.image_times > now[:, None] + 1e-9)).any(), "future image cannot enter actual memory")
        latest = self.image_times[:, :, -1]
        if checking():
            fail_if((mask & self.available & (observation.image_times < latest)).any(), "out-of-order exposure for the same view")
        duplicate = mask & self.available & (observation.image_times == latest)
        if checking():
            changed = _changed(observation.features, self.features[:, :, -1], observation.q_at_image,
                               self.q_at_image[:, :, -1])
            fail_if((duplicate & changed).any(), "same exposure identity has changed features or q")
        return mask & ~duplicate

    def write(self, observation, now, *, readback_index=None):
        """Write one genuinely new exposure per view; never advance on readback alone."""
        inserted = self.validate_observation(observation, now)
        camera_types = self.camera_types
        if camera_types is None and observation.camera_types is not None:
            camera_types = observation.camera_types.to(self.features.device).detach().clone()
        if not deferring() and not inserted.any():
            # A cold/missing-image stream still carries types, not fake pixels. (Section 17.82: while checks are
            # deferred the rows are pushed anyway; with nothing inserted every value stays as it was.)
            return (self if camera_types is self.camera_types else
                    replace(self, camera_types=camera_types)), inserted
        b = len(self.features)
        def push(old, new):
            with torch.inference_mode(False), torch.no_grad():
                new = new.to(old).detach()
                select = inserted.reshape(b, 2, *([1] * (old.ndim - 2)))
                # compiled for the pictures only (section 17.84); the small pose, clock and flag tensors stay as they are;
                # outside the compiled regions every tensor is shifted without the extra copy (17.173)
                return (_shifted_in if old.ndim == 5 and v21_compiled.active() else _shift)(old, new, select)
        indices = getattr(self, 'readback_indices', None)
        current = getattr(self, 'current_readback_index', None)
        if readback_index is not None:
            if readback_index.shape != (b,) or readback_index.dtype != torch.long:
                raise ValueError('completed readback index must be nonnegative int64 [B]')
            if checking():
                fail_if((readback_index < 0).any(), 'completed readback index must be nonnegative int64 [B]')
            if indices is None:
                raise ValueError('legacy memory progress must be explicitly reconstructed before a new capture')
            indices = push(indices, readback_index[:, None].expand(-1, 2))
            current = (readback_index.detach().clone() if not deferring()
                       else torch.where(inserted.any(), readback_index, current).detach().clone())
        return VisualMemory(push(self.features, observation.features),
                            push(self.q_at_image, observation.q_at_image),
                            push(self.image_times, observation.image_times),
                            push(self.valid, torch.ones_like(observation.measurement_available)),
                            indices, current, camera_types), inserted


class ReadLayer(nn.Module):
    def __init__(self, concat=False, joints=False, pose_position=False):
        super().__init__()
        # concat (V21Config.concat_inputs; James 11/02 「能concat的就concat」「能合并的就合并网络 不要把linear放在外面」
        # 「或者是直接输入到fcpg呢」): the layer reads everything as it is, with no network of its own before it: its keys
        # project [picture features, patch position, read address], its values [picture features, patch position] and,
        # with joints, its queries [query, current joints]
        self.concat = concat
        self.q_norm = nn.LayerNorm(192)
        if not concat:
            self.kv_norm = nn.LayerNorm(192)
        self.q_projection = nn.Linear(192 + 6 if joints else 192, 192)
        # pose_position (11/03): the keys also join the patch position x joints products (POSE_POSITION)
        if pose_position and not concat:
            raise ValueError('the pose x position products join a concatenating read layer\'s keys')
        self.key = nn.Linear(DIM + 2 + ADDRESS + (POSE_POSITION if pose_position else 0) if concat else 192, 192)
        self.value = nn.Linear(DIM + 2 if concat else 192, 192)
        self.output = nn.Linear(192, 192)
        self.ff = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 768), nn.GELU(), nn.Linear(768, 192))

    def project(self, features):
        if getattr(self, 'concat', False):          # the keys join the read address in forward
            return features, self.value(features)
        x = self.kv_norm(features)
        return self.key(x), self.value(x)

    def forward(self, query, keys, values, address, available, joints=None):
        b, n, _ = keys.shape
        has = available.any(-1)
        # avoid all-masked softmax; result is explicitly unavailable (section 17.84: the same mask without an indexed
        # assignment, which torch.compile cannot keep in one program)
        safe = available | ((~has)[:, None] & (torch.arange(n, device=available.device) == 0))
        split = lambda x: x.reshape(b, -1, 6, 32).transpose(1, 2)
        q = self.q_norm(query)
        if joints is not None:      # concat: the current joints [B, 6] go into every query's projection
            q = torch.cat((q, joints.to(q)[:, None].expand(-1, q.shape[1], -1)), -1)
        q = split(self.q_projection(q))
        keys = self.key(torch.cat((keys, address), -1)) if getattr(self, 'concat', False) else keys + address
        update = F.scaled_dot_product_attention(q, split(keys), split(values),
                                               attn_mask=safe[:, None, None], dropout_p=0.)
        update = update.transpose(1, 2).reshape(b, len(query[0]), 192)
        query = query + self.output(update)
        query = query + self.ff(query)
        return torch.where(has[:, None, None], query, torch.zeros_like(query))


class ReadbackMemoryReader(nn.Module):
    """Two shared per-view layers; 29->64->192 changes read addresses, never values.

    Motion path/error statistics summarize the most recent three COMPLETED
    transitions, not an invented continuous path since the image exposure.

    task_vocabulary (docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md, C): the
    read queries also depend on the task words and the current joints, so the
    reader can look for the instructed object. None keeps the original reader.
    history_address=False (same doc, B): the read addresses keep each exposure's
    pose and its displacement to the current pose, but not the recent travel and
    tracking-error statistics of the last completed transitions.
    jaw_address=False (same doc, section 8): the jaw opening (joint 6) is not read:
    not in the exposure poses and displacements of the addresses, not in the
    pose that shapes the task queries.
    """
    def __init__(self, *, mode="readback", task_vocabulary=None, history_address=True, jaw_address=True, queries=8,
                 pose_queries=False, concat=False, pose_position=False):
        super().__init__()
        if type(queries) is not int or queries not in (8, 32):
            raise ValueError("8 or 32 read queries per view")
        self.queries_per_view = queries
        if mode not in ("readback", "temporal"):
            raise ValueError("memory mode must be readback or temporal")
        if type(history_address) is not bool or type(jaw_address) is not bool:
            raise ValueError("history_address and jaw_address must be explicit booleans")
        if not jaw_address and history_address:
            raise ValueError("the recent travel statistics carry the jaw; jaw_address=False needs history_address=False")
        # concat (V21Config.concat_inputs, Fig. 1 b): no network before the layers: they read [picture features, patch
        # position] (values), [picture features, patch position, read address] (keys) and, with pose_queries, [learned
        # query, current joints] (queries) as they are
        if type(concat) is not bool or (concat and task_vocabulary is not None):
            raise ValueError("concat is an explicit boolean, and a concatenating reader reads no task words")
        self.concat = concat
        self.mode = mode
        self.history_address = history_address
        self.jaw_address = jaw_address
        if not concat:
            self.feature_projection = nn.Linear(DIM, 192)
            self.position = nn.Linear(2, 192)
        self.queries = nn.Parameter(torch.randn(2, queries, 192) * .02)
        if not concat:
            self.address = nn.Sequential(nn.Linear(29, 64), nn.GELU(), nn.Linear(64, 192))
        self.reads_joints = concat and pose_queries and task_vocabulary is None
        # pose_position (V21Config.pose_position, 11/03): the keys also read each patch's position x its picture's joints
        if type(pose_position) is not bool or (pose_position and not concat):
            raise ValueError("pose_position is an explicit boolean and needs the concatenating reader")
        self.pose_position = pose_position
        self.layers = nn.ModuleList([ReadLayer(concat, self.reads_joints, pose_position),
                                     ReadLayer(concat, self.reads_joints, pose_position)])
        self.register_buffer("lower", torch.tensor(DEFAULT_LOWER))
        self.register_buffer("ranges", torch.tensor(DEFAULT_UPPER) - self.lower)
        self.register_buffer("patch_xy", patch_xy())
        self.task_vocabulary = None if task_vocabulary is None else int(task_vocabulary)
        if self.task_vocabulary is not None:
            if self.task_vocabulary < 4:
                raise ValueError("task vocabulary must include four reserved tokens")
            self.task_words = nn.Embedding(self.task_vocabulary, 192, padding_idx=0)
            self.task_queries = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 2 * queries * 192))
            self.pose_queries = nn.Sequential(nn.Linear(6, 192), nn.GELU(), nn.Linear(192, 2 * queries * 192))
        elif pose_queries and not concat:
            # V21Config.task_text=False (James 11/02): the read queries follow the current joints, not the task words
            self.pose_queries = nn.Sequential(nn.Linear(6, 192), nn.GELU(), nn.Linear(192, 2 * queries * 192))
        self._projection_cache = None
        # section 17.84: 1 = only the first (wrist) view can ever hold a picture (V21Config.wrist_only; the core sets it),
        # so a compiled training read skips the other view, whose read is zero anyway
        self.views = 2

    def _queries(self, b, actual_q, task_ids):
        query = self.queries[None].expand(b, -1, -1, -1)
        if self.task_vocabulary is None:
            if not hasattr(self, 'pose_queries'):
                return query
            pose = (actual_q.to(self.lower) - self.lower) / self.ranges * 2 - 1
            if not self.jaw_address:
                pose = torch.cat((pose[:, :5], torch.zeros_like(pose[:, 5:])), -1)
            return query + self.pose_queries(pose.to(query)).reshape(b, 2, self.queries_per_view, 192)
        if (not isinstance(task_ids, torch.Tensor) or task_ids.ndim != 2 or task_ids.shape[0] != b
                or task_ids.dtype != torch.long):
            raise ValueError("this reader reads for a task: valid task token ids [B,T] are required")
        if checking():
            fail_if((task_ids < 0).any() | (task_ids >= self.task_vocabulary).any() | ~(task_ids != 0).any(-1).all(),
                    "this reader reads for a task: valid task token ids [B,T] are required")
        words = self.task_words(task_ids.to(self.queries.device))
        present = (task_ids != 0).to(words)[..., None]
        task = (words * present).sum(1) / present.sum(1)
        pose = (actual_q.to(self.lower) - self.lower) / self.ranges * 2 - 1
        if not self.jaw_address:
            pose = torch.cat((pose[:, :5], torch.zeros_like(pose[:, 5:])), -1)
        return query + (self.task_queries(task) + self.pose_queries(pose.to(task))).reshape(b, 2, self.queries_per_view, 192)

    def _joints(self, actual_q, views):
        """concat with pose_queries: the current joints on -1..1 (no jaw without jaw_address) for the layers' queries,
        one row per (row, view) as the layers read them -> [B x views, 6]; None for every other reader."""
        if not getattr(self, 'reads_joints', False):
            return None
        pose = (actual_q.to(self.lower) - self.lower) / self.ranges * 2 - 1
        if not self.jaw_address:
            pose = torch.cat((pose[:, :5], torch.zeros_like(pose[:, 5:])), -1)
        return pose.repeat_interleave(views, 0)

    def train(self, mode=True):
        self._projection_cache = None
        return super().train(mode)

    def _project(self, memory):
        # Eval-only cache. Tensor versions invalidate it after optimizer steps,
        # load_state_dict or device/dtype conversion. No learned K/V is stored
        # in the actual observation memory or reused through training updates.
        version = lambda x: (x.data_ptr(), None if x.is_inference() else x._version)
        signature = tuple(version(p) for p in self.parameters())
        signature += tuple(version(t) for t in
                           (self.patch_xy, memory.features, memory.q_at_image, memory.image_times, memory.valid))
        cacheable = (not self.training and not torch.is_grad_enabled()
                     and not memory.features.is_inference())
        if cacheable and self._projection_cache is not None:
            owner, versions, values = self._projection_cache
            if owner is memory and versions == signature:
                return values
        b = len(memory.features)
        concat = getattr(self, 'concat', False)

        def project(features):
            if concat:          # the pictures as the layers read them: [features, patch position]
                x = features.reshape(b * 2, 3 * PATCHES, DIM).to(self.queries)
                xy = self.patch_xy.to(x).repeat(3, 1)[None].expand(len(x), -1, -1)
                return torch.cat((F.layer_norm(x, (DIM,)), xy), -1)
            x = features.reshape(b * 2, 3 * PATCHES, DIM).to(self.feature_projection.weight)
            return self.feature_projection(F.layer_norm(x, (DIM,)))

        if getattr(self, 'checkpoint_projection', False) and torch.is_grad_enabled():
            # section 17.80 (Actor trainer --checkpoint-reader): the float32 pictures and their normalised copy are
            # made again in the backward pass instead of being kept (the same values and gradients, less memory)
            x = checkpoint(project, memory.features, use_reentrant=False, preserve_rng_state=False)
        else:
            x = project(memory.features)
        if not concat:
            x = x + self.position(self.patch_xy.to(x)).repeat(3, 1)[None]
        values = [layer.project(x) for layer in self.layers]
        if cacheable:
            self._projection_cache = (memory, signature, values)
        return values

    def frame_metadata(self, memory, actual_q, now, history):
        """Identical motion/time information for both readers' plain Actor tokens.

        [B,2,3,28] = exposure q, displacement, recent travel/error, age,
        view and validity. These are actual numeric context, never appearance.
        """
        b = len(memory.features)
        finite("actual q", actual_q, (b, 6)); clock("read time", now, (b,))
        _validate_history(history, b)
        if checking():
            fail_if((memory.valid & (memory.image_times > now[:, None, None] + 1e-9)).any(), "cannot read a future exposure")
        ranges = self.ranges.to(actual_q)
        q0 = (memory.q_at_image.to(actual_q) - self.lower.to(actual_q)) / ranges * 2 - 1
        delta = (actual_q[:, None, None] - memory.q_at_image.to(actual_q)) / ranges
        mask = history.valid.to(actual_q.device)
        paths = history.readbacks_position.to(actual_q)
        commands = history.commands_position.to(actual_q)
        dt = history.time_deltas.to(actual_q) * mask[:, :, None]
        travel = (paths.diff(dim=2).abs() * mask[:, :, None, None]).sum((1, 2)) / ranges
        error = ((paths[:, :, 1:] - commands) * dt[..., None]).sum((1, 2))
        error = error / dt.sum((1, 2)).clamp_min(1e-6)[:, None] / ranges
        recent = lambda x: x[:, None, None].expand(-1, 2, 3, -1)
        motion = torch.cat((q0, delta, recent(travel), recent(error)), -1)
        age = torch.log1p((now[:, None, None] - memory.image_times).clamp_min(0)).float()[..., None]
        view = torch.eye(2, device=actual_q.device)[None, :, None].expand(b, 2, 3, 2)
        meta = torch.cat((motion, age, view, memory.valid[..., None].to(actual_q)), -1)
        return torch.where(memory.valid[..., None], meta, torch.zeros_like(meta))

    def forward(self, memory, actual_q, now, history, *, address_mode=None, task_ids=None):
        b = len(memory.features)
        meta = self.frame_metadata(memory, actual_q, now, history)[..., :27]
        if not deferring() and not memory.available.any():   # section 17.82: deferred, rows without a picture
            return actual_q.new_zeros(b, 2, self.queries_per_view, 192)    # are zeroed by the layers instead
        mode = self.mode if address_mode is None else address_mode
        if mode not in ("readback", "temporal"):
            raise ValueError("unsupported address intervention")
        if mode == "temporal":
            meta = torch.cat((torch.zeros_like(meta[..., :24]), meta[..., 24:]), -1)
        elif not self.history_address:
            meta = torch.cat((meta[..., :12], torch.zeros_like(meta[..., 12:24]), meta[..., 24:]), -1)
            if not self.jaw_address:
                # columns 5 and 11: the jaw opening at the exposure and its change since
                column = torch.arange(27, device=meta.device)    # ones with 0 at 5 and 11, made on the GPU (17.80)
                keep = ((column != 5) & (column != 11)).to(meta.dtype)
                meta = meta * keep
        if v21_compiled.active() and not (getattr(self, 'checkpoint_projection', False) and torch.is_grad_enabled()):
            # section 17.84: a training update's read, one compiled program (fcpg/v21_compiled.py)
            query = self._queries(b, actual_q, task_ids)
            if getattr(self, 'views', 2) == 1:
                if checking():
                    fail_if(memory.valid[:, 1].any(), 'a wrist-only memory holds a picture of the other view')
                read = self._read(memory.features[:, :1], memory.valid[:, :1], meta[:, :1], query[:, :1],
                                  self._joints(actual_q, 1))
                return torch.cat((read, torch.zeros_like(read)), 1)    # what the layers give a view without pictures
            return self._read(memory.features, memory.valid, meta, query, self._joints(actual_q, 2))
        meta = meta[..., None, :].expand(-1, -1, -1, PATCHES, -1)
        xy = self.patch_xy[None, None, None].expand(b, 2, 3, -1, -1)
        metadata = torch.cat((meta.to(xy), xy), -1).reshape(b * 2, 3 * PATCHES, 29)
        metadata = torch.where(memory.valid.reshape(b * 2, 3).repeat_interleave(PATCHES, 1)[..., None],
                               metadata, torch.zeros_like(metadata))
        if getattr(self, 'concat', False):     # the read address as it is (the patch position comes with the picture)
            bias = metadata[..., :ADDRESS]
            if getattr(self, 'pose_position', False):
                bias = torch.cat((bias, pose_position(metadata)), -1)
        else:
            bias = self.address(metadata.to(self.address[0].weight))
        query = self._queries(b, actual_q, task_ids).reshape(b * 2, self.queries_per_view, 192)
        available = memory.valid.reshape(b * 2, 3).repeat_interleave(PATCHES, 1)
        joints = self._joints(actual_q, 2)
        for layer, (keys, values) in zip(self.layers, self._project(memory)):
            query = layer(query, keys, values, bias.to(keys), available, joints)
        return query.reshape(b, 2, self.queries_per_view, 192)

    @region
    def _read(self, features, valid, meta, query, joints=None):
        """forward()'s read after its checks, of [B,V,3,...] memory values (V = 2, or 1: the wrist view alone) and the
        [B,V,queries,192] queries (and, concat with pose_queries, the current joints [B x V, 6]): the same operations
        as the uncompiled read, without the evaluation-only cache."""
        b, v = features.shape[:2]
        meta = meta[..., None, :].expand(-1, -1, -1, PATCHES, -1)
        xy = self.patch_xy[None, None, None].expand(b, v, 3, -1, -1)
        metadata = torch.cat((meta.to(xy), xy), -1).reshape(b * v, 3 * PATCHES, 29)
        available = valid.reshape(b * v, 3).repeat_interleave(PATCHES, 1)
        metadata = torch.where(available[..., None], metadata, torch.zeros_like(metadata))
        if getattr(self, 'concat', False):
            bias = metadata[..., :ADDRESS]
            if getattr(self, 'pose_position', False):
                bias = torch.cat((bias, pose_position(metadata)), -1)
            x = features.reshape(b * v, 3 * PATCHES, DIM).to(self.queries)
            patches = self.patch_xy.to(x).repeat(3, 1)[None].expand(len(x), -1, -1)
            x = torch.cat((F.layer_norm(x, (DIM,)), patches), -1)
        else:
            bias = self.address(metadata.to(self.address[0].weight))
            x = features.reshape(b * v, 3 * PATCHES, DIM).to(self.feature_projection.weight)
            x = self.feature_projection(F.layer_norm(x, (DIM,)))
            x = x + self.position(self.patch_xy.to(x)).repeat(3, 1)[None]
        query = query.reshape(b * v, self.queries_per_view, 192)
        for layer in self.layers:
            keys, values = layer.project(x)
            query = layer(query, keys, values, bias.to(keys), available, joints)
        return query.reshape(b, v, self.queries_per_view, 192)
