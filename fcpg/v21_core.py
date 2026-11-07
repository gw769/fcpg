"""V21 research core. Distinct from prod_v21's V20.3.2 deployment repair."""
from dataclasses import dataclass, replace
from uuid import uuid4
import torch
from torch import nn

from .prod_v20_policy import empty_history, append_completed
from .v21_checks import checking, fail_if
from .v21_memory import VisualMemory, VisualObservation, ReadbackMemoryReader, finite, clock
from .v21_models import V21Actor, V21LeWM, CommandTrace, ARMOutput
from .v21_encoder import DIM, ENCODER, ENCODERS, PATCHES
from .v21_picture_memory import empty_pictures, write_pictures
from .v21_temporal_contract import (effective_memory, temporal_actor_state, record_training_temporal_features,
    prepare_progress_state, actor_temporal_features, READBACKS_PER_COMPLETED_BLOCK)

# Canonical paper/runtime name; the 3k checkpoint was originally serialized as 21.0.1.
# Naming only: the tensor architecture and v2 contract are unchanged.
VERSION = "21.0.2"
COMPATIBLE_VERSIONS = ("21.0.1", VERSION, "21.0.4")
CONTRACT = "v21-readback-addressed-visual-memory-v2"
POSITION_UNIT = "lerobot_calibrated_position"
WORLD_HEAD_INIT_SEED = 1713   # section 17.113: the world head's own generator


def current_scene_pool(features):
    """Latest frozen DINO tokens, pooled over patches. Shape [B, 2, DIM]."""
    if features.ndim != 5 or features.shape[1:] != (2, 3, PATCHES, DIM):
        raise ValueError(f'visual memory features must be [B,2,3,{PATCHES},{DIM}]')
    return features[:, :, -1].float().mean(-2)


def current_object_cue(observation, q):
    """Current wrist coordinates are not the historical visual-memory bank."""
    return _current_wrist_cue(observation, q, None if observation is None else observation.object_relative,
                              'object-relative grounding')


def current_place_cue(observation, q):
    """Where the object goes (the drop zone in the wrist) against the jaws; layout and currency rule of current_object_cue."""
    return _current_wrist_cue(observation, q, None if observation is None else getattr(observation, 'place_relative', None),
                              'place-relative grounding')


def current_goal_cue(observation, q):
    """The place goal against the jaws now (fcpg/v21_place_goal.py; training overhaul doc section 17.7). Geometry, not a
    picture: kept when a camera delivery is missing; zeros without an observation, like the other current cues."""
    empty = q.new_zeros(len(q), 8)
    value = None if observation is None else getattr(observation, 'place_goal', None)
    if value is None:
        return empty
    value = value.to(q.device, torch.float32)
    finite('place goal', value, (len(q), 8))
    return value.detach().clone()


def _current_wrist_cue(observation, q, value, name):
    empty = q.new_zeros(len(q), 8)
    if value is None:
        return empty
    value = value.to(q.device, torch.float32)
    finite(name, value, (len(q), 8))
    return torch.where(observation.measurement_available[:, :1].to(q.device),
                       value, empty).detach().clone()


@dataclass(frozen=True)
class V21Config:
    vocabulary_size: int
    memory_mode: str = "readback"
    dimension: int = 192
    memory_frames: int = 3
    queries_per_view: int = 8
    seed: int = 20000
    object_relative: bool = False  # wrist object-vs-jaw token for the Actor
    prediction_horizon: int = 5
    execution_horizon: int = 5
    # docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md: defaults keep every earlier model unchanged
    actor_history: bool = True     # False: no motion tokens / execution feedback for the Actor, no recent
                                   # travel/error statistics in the FCPG read addresses
    reader_task: bool = False      # True: FCPG read queries depend on the task words and current joints
    # same doc, section 8 (second round); defaults again keep every earlier model unchanged
    gripper_relative: bool = False  # section 17.39 (K2), needs gripper_separate: the sixth output is a change from the
                                    # current jaw reading (reading + output x half range); the reading is still no input
    gripper_absolute_aux: bool = False  # section 17.46, needs gripper_relative: an extra head predicts the absolute jaw
                                        # command for the training loss only; commands are unchanged
    gripper_separate: bool = False  # True: the jaw opening is not an input anywhere, and the Actor outputs the
                                    # opening itself instead of a change added to the jaw reading
    wrist_only: bool = False        # True: the external camera (slot 1) never enters the visual memory
    actor_width: int = 192          # 384: a wider Actor with the same layers; FCPG tokens stay 192-d.
                                    # queries_per_view may then also be 32 (a longer summary of each view)
    # docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md section 17; the default keeps every earlier model unchanged
    place_relative: bool = False      # True: the Actor also reads where the object goes (wrist drop zone) vs the jaws, a second
                                    # 8-number token in the object token's layout; needs object_relative
    # same doc, section 17.7 (the B route); the default keeps every earlier model unchanged
    place_goal: bool = False        # True: the Actor also reads the place goal against its jaws in the arm's polar frame
                                    # (fcpg/v21_place_goal.py), added to the pose token while available
    # James 10/21 「DINOv3 L，8 位 全部替换成这个」: the frozen picture encoder (fcpg/v21_encoder.py); the default keeps
    # every earlier model unchanged, and a core is only built in a process running the same encoder (V21_ENCODER)
    encoder: str = "dinov2_vits14"
    # same doc, section 17.75 (James 10/23 「actor根本沒有設計過」); the default keeps every earlier model unchanged
    actor_layers: str = "v1"        # "v2": 4 context encoder layers instead of 1, the 40 command queries attend to each
                                    # other in both directions, dropout 0.1 in every Actor layer
    # same doc, section 17.113 (James 10/27 「训练的时候可以actor+lewn合二为一吗」); the default keeps every earlier model
    world_head: bool = False        # True: a world head (fcpg/v21_world_head.py) predicts, from the reader's wrist tokens,
                                    # the pose and a block of commands, the wrist picture's change and the read-backs
    world_components: int = 64      # principal components of the wrist patches the world head predicts the change of
                                    # (world_target 'patches' only)
    world_target: str = "patches"   # 'tokens' (17.113, second version): the reader's own wrist tokens of the next state
    # James 11/02 「fcpg和actor 不要处理了文字了 没用 它看不懂的」; the default keeps every earlier model unchanged
    task_text: bool = True          # False: neither the FCPG reader nor the Actor reads the task words (the object
                                    # finder still does); a task-reading reader's queries then follow the current
                                    # joints only
    # James 11/02 「能concat的就concat 然后改代码」「能合并的就合并网络 不要把linear放在外面」; the default keeps every
    # earlier model unchanged
    concat_inputs: bool = False     # True: the FCPG reader's attention reads everything as it is, with no network
                                    # before it: [picture features, patch position] for its values, with the read
                                    # address for its keys, [learned query, current joints] for its queries
                                    # (Fig. 1 b); needs task_text=False
    # 11/03: in the release Actor (concat_inputs, no task words) the place cue, added to the pose token, steers the pan
    # (13 of 70 robot states), the grasp cue, a token of its own, does not (0 of 267; Archive development rows 15 of
    # 117, the earlier two-camera base 76); the default keeps every earlier model unchanged
    object_on_pose: bool = False    # True: the wrist object cue is added to the pose token while seen, as the place cue,
                                    # not a token of its own; needs object_relative
    # 11/03 (James 「AOK阿」): with concat_inputs the joints at capture enter the keys through one linear layer, which
    # adds the same amount to every patch of a picture: they could weigh whole pictures, never move where in one the
    # reader looks (swapping them moved the release Actor's commands by 0.0000); the default keeps every earlier model
    pose_position: bool = False     # True: the FCPG keys also join each patch's position basis x its picture's joints at
                                    # capture and their change since (fcpg/v21_memory.pose_position); needs concat_inputs

    def __post_init__(self):
        if self.vocabulary_size < 4 or self.memory_mode not in ("readback", "temporal"):
            raise ValueError("invalid vocabulary or comparison mode")
        if any(type(value) is not bool for value in (self.actor_history, self.reader_task, self.gripper_separate,
                                                      self.gripper_relative, self.gripper_absolute_aux,
                                                      self.wrist_only, self.place_relative, self.place_goal,
                                                      self.task_text, self.concat_inputs, self.object_on_pose,
                                                      self.pose_position)):
            raise ValueError("actor_history, reader_task, gripper_separate, gripper_relative, wrist_only, place_relative, "
                             "place_goal, task_text, concat_inputs, object_on_pose and pose_position must be explicit "
                             "booleans")
        if self.pose_position and not self.concat_inputs:
            raise ValueError("pose_position joins the concatenating reader's keys: it needs concat_inputs")
        if self.object_on_pose and not self.object_relative:
            raise ValueError("object_on_pose adds the object cue to the pose token: it needs object_relative")
        if self.concat_inputs and self.reader_task and self.task_text:
            raise ValueError("a concatenating FCPG reader reads no task words: concat_inputs needs task_text=False")
        if self.encoder not in ENCODERS:
            raise ValueError(f"unknown picture encoder {self.encoder!r}")
        if self.actor_layers not in ("v1", "v2"):
            raise ValueError("actor_layers is 'v1' (every earlier model) or 'v2' (section 17.75)")
        if self.place_relative and not self.object_relative:
            raise ValueError("the place token comes with the object token: place_relative needs object_relative")
        if self.gripper_separate and self.actor_history:
            raise ValueError("the motion history carries the jaw reading; gripper_separate needs actor_history=False")
        if self.gripper_relative and not self.gripper_separate:
            raise ValueError("gripper_relative changes how the separate jaw output becomes a command: it needs gripper_separate")
        if type(self.world_head) is not bool or type(self.world_components) is not int or self.world_components not in (16, 32, 64, 128):
            raise ValueError("world_head is an explicit boolean and world_components 16, 32, 64 or 128")
        if self.world_target not in ("patches", "tokens"):
            raise ValueError("world_target is 'patches' or 'tokens'")
        if self.gripper_absolute_aux and not self.gripper_relative:
            raise ValueError("gripper_absolute_aux predicts the absolute jaw beside a relative one: it needs gripper_relative")
        if ((self.dimension, self.memory_frames) != (192, 3) or type(self.queries_per_view) is not int
                or self.queries_per_view not in (8, 32) or type(self.actor_width) is not int
                or self.actor_width not in (192, 384)):
            raise ValueError("V21 fixes 192D tokens and three actual frames; 8 or 32 queries per view; Actor width 192 or 384")
        if (type(self.prediction_horizon) is not int or self.prediction_horizon not in (5, 40)
                or type(self.execution_horizon) is not int or self.execution_horizon != 5):
            raise ValueError("V21 predicts 5 or 40 steps and completes five actual readbacks")


@dataclass(frozen=True)
class ActualState:
    memory: VisualMemory
    q: torch.Tensor
    time: torch.Tensor
    history: object
    history_ends: torch.Tensor
    grounding: torch.Tensor
    visual_error: torch.Tensor
    measurement_available: torch.Tensor
    measurement_pending: torch.Tensor
    visual_feedback_available: torch.Tensor
    visual_feedback_time: torch.Tensor
    q_error: torch.Tensor
    feedback_available: torch.Tensor
    feedback_time: torch.Tensor
    feedback_offsets: torch.Tensor
    feedback_retimed: torch.Tensor
    object_relative: torch.Tensor | None = None  # [B,8], current exposure
    feedback_readback_index: torch.Tensor | None = None
    visual_feedback_readback_indices: torch.Tensor | None = None
    progress_session_id: str | None = None
    progress_origin: str | None = None
    task_ids: torch.Tensor | None = None   # [B,T] task words, for a task-reading FCPG (V21Config.reader_task)
    place_relative: torch.Tensor | None = None  # [B,8], current exposure (V21Config.place_relative); None = zeros
    place_goal: torch.Tensor | None = None      # [B,8], current observation (V21Config.place_goal); None = zeros
    pictures: object | None = None              # 17.167: [B,2,N] PictureMemory of a picture-memory LeWM (a runtime's)


@dataclass(frozen=True)
class Forecast:
    source_q: torch.Tensor
    source_time: torch.Tensor
    trace: CommandTrace
    query_offsets: torch.Tensor
    response: ARMOutput


class V21ResearchCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.encoder != ENCODER:
            raise ValueError(f"this model reads {config.encoder} features but the process runs {ENCODER} "
                             "(set V21_ENCODER before importing fcpg)")
        self.config = config
        reads_words = config.reader_task and config.task_text
        self.reader = ReadbackMemoryReader(mode=config.memory_mode,
                                           task_vocabulary=config.vocabulary_size if reads_words else None,
                                           history_address=config.actor_history,
                                           jaw_address=not config.gripper_separate,
                                           queries=config.queries_per_view,
                                           pose_queries=config.reader_task and not config.task_text,
                                           concat=config.concat_inputs, pose_position=config.pose_position)
        # section 17.84: observe() never lets the external view into a wrist-only memory, so a compiled training read
        # skips that view (fcpg/v21_memory.py)
        self.reader.views = 1 if getattr(config, 'wrist_only', False) else 2
        self.actor = V21Actor(config.vocabulary_size, object_relative=config.object_relative,
                              place_relative=config.place_relative, place_goal=config.place_goal,
                              horizon=config.prediction_horizon, history_inputs=config.actor_history,
                              gripper_separate=config.gripper_separate, gripper_relative=config.gripper_relative,
                              gripper_absolute_aux=config.gripper_absolute_aux,
                              queries=config.queries_per_view,
                              width=config.actor_width, layers=config.actor_layers,
                              text_tokens=config.task_text, object_on_pose=config.object_on_pose)
        self.lewm = V21LeWM(horizon=config.prediction_horizon)
        if getattr(config, 'world_head', False):
            # section 17.113: drawn from its own generator, so every other module starts exactly as in the same seed's
            # model without it
            from .v21_world_head import WorldHead
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(WORLD_HEAD_INIT_SEED)
                self.world = WorldHead(components=config.world_components,
                                       target=getattr(config, 'world_target', 'patches'),
                                       queries=config.queries_per_view)
        self.register_buffer("q_loss_scale", torch.tensor([2.] * 6))

    def set_loss_scale(self, value):
        value = torch.as_tensor(value, device=self.q_loss_scale.device, dtype=torch.float32)
        finite("loss scale", value, (6,))
        if (value <= 0).any():
            raise ValueError("positive per-joint loss scales required")
        self.q_loss_scale.copy_(value)
        self.actor.increment_scale.copy_(value)
        self.lewm.increment_scale.copy_(value)

    def initial_state(self, q, now, *, feature_dtype=torch.float32, task_ids=None):
        b = len(q)
        finite("q", q, (b, 6)); clock("actual time", now, (b,))
        if task_ids is not None and (not isinstance(task_ids, torch.Tensor) or task_ids.ndim != 2
                                     or task_ids.shape[0] != b or task_ids.dtype != torch.long):
            raise ValueError("task ids must be int64 [B,T]")
        mask = torch.zeros(b, 2, device=q.device, dtype=torch.bool)
        slots = self.picture_slots()
        return ActualState(memory=VisualMemory.empty(b, device=q.device, dtype=feature_dtype),
            q=q.detach().clone(), time=now.detach().clone(),
            history=empty_history(b, device=q.device, dtype=q.dtype), history_ends=now.new_zeros(b, 3),
            grounding=q.new_zeros(b, 2, self.config.queries_per_view, 192),
            visual_error=q.new_zeros(b, 2, self.config.queries_per_view, 192),
            measurement_available=mask, measurement_pending=mask.clone(),
            visual_feedback_available=mask.clone(), visual_feedback_time=now.new_zeros(b, 2),
            q_error=q.new_zeros(b, 5, 6), feedback_available=mask[:, 0].clone(),
            feedback_time=now.new_zeros(b), feedback_offsets=now.new_zeros(b, 5),
            feedback_retimed=mask[:, 0].clone(), object_relative=q.new_zeros(b, 8), place_relative=q.new_zeros(b, 8),
            place_goal=q.new_zeros(b, 8),
            feedback_readback_index=torch.zeros(b, device=q.device, dtype=torch.long),
            visual_feedback_readback_indices=torch.zeros(b, 2, device=q.device, dtype=torch.long),
            progress_session_id=uuid4().hex, progress_origin='actual_state_reset; completed readback count starts at zero',
            task_ids=None if task_ids is None else task_ids.detach().clone(),
            pictures=empty_pictures(b, slots, device=q.device, dtype=feature_dtype) if slots else None)

    def observe(self, previous, observation, q, now, *, task_ids=None):
        """Current readback + optional newly arrived images; no invented execution.

        task_ids starts a new state's task (a task-reading FCPG needs it); a later
        observation keeps the state's task and may only repeat it."""
        b = len(q)
        finite("actual q", q, (b, 6)); clock("actual time", now, (b,))
        if self.config.wrist_only and observation is not None:
            # the external camera is not a model input: its exposures never enter the memory
            wrist = observation.measurement_available.clone()
            wrist[:, 1] = False
            observation = replace(observation, measurement_available=wrist)
        if previous is None:
            dtype = observation.features.dtype if observation is not None else q.dtype
            previous = self.initial_state(q, now, feature_dtype=dtype, task_ids=task_ids)
        elif task_ids is not None:
            if previous.task_ids is None or previous.task_ids.shape != task_ids.shape:
                raise ValueError("an actual state keeps its task; start a new state for another task")
            if checking():
                fail_if((previous.task_ids != task_ids.to(previous.task_ids.device)).any(),
                        "an actual state keeps its task; start a new state for another task")
        if checking():
            fail_if((now < previous.time).any() | (previous.history.valid & (previous.history_ends > now[:, None])).any(),
                    "actual state/history cannot travel backwards or use future execution")
        previous = prepare_progress_state(previous)
        before = self.reader(previous.memory, q, now, previous.history, task_ids=previous.task_ids)
        inserted = torch.zeros(b, 2, device=q.device, dtype=torch.bool)
        memory = previous.memory
        pictures = previous.pictures
        if observation is not None:
            memory, inserted = memory.write(observation, now, readback_index=memory.current_readback_index)
            if pictures is not None:        # 17.167: the same new exposures, into the last N
                pictures = write_pictures(pictures, observation, inserted)
        after = before if memory is previous.memory else self.reader(memory, q, now, previous.history,
                                                                     task_ids=previous.task_ids)
        previous_available = effective_memory(self, previous.memory, now).available
        fresh_error = torch.where(previous_available[..., None, None], after-before, torch.zeros_like(after))
        # New actual evidence replaces its view's event. Readback-only and
        # duplicate refreshes retain unconsumed feedback, with its own age.
        error = torch.where(inserted[..., None, None], fresh_error, previous.visual_error)
        visual_valid = torch.where(inserted, previous_available, previous.visual_feedback_available)
        visual_time = torch.where(inserted, now[:, None].expand(-1, 2), previous.visual_feedback_time)
        visual_progress = torch.where(inserted, memory.current_readback_index[:, None].expand(-1, 2),
                                      previous.visual_feedback_readback_indices)
        relative = current_object_cue(observation, q)
        return replace(previous, memory=memory, pictures=pictures, q=q.detach().clone(), time=now.detach().clone(),
                       object_relative=relative,
                       place_relative=current_place_cue(observation, q), place_goal=current_goal_cue(observation, q),
                       grounding=after, visual_error=error, measurement_available=inserted,
                       measurement_pending=previous.measurement_pending | inserted,
                       visual_feedback_available=visual_valid, visual_feedback_time=visual_time,
                       visual_feedback_readback_indices=visual_progress)

    def act(self, state, text_ids, command_offsets):
        inputs = self.actor_inputs(state)
        return self.actor(inputs.pop('grounding'), inputs.pop('visual_error'), inputs.pop('q_position'),
                          text_ids, command_offsets=command_offsets, **inputs)

    def actor_inputs(self, state):
        """Factual inputs before any trainable Actor projection."""
        record_training_temporal_features(self, state)
        state = temporal_actor_state(self, state)
        feedback_feature, visual_feedback_feature = actor_temporal_features(self, state)
        return dict(grounding=state.grounding, visual_error=state.visual_error, q_position=state.q,
            motion_context=self.reader.frame_metadata(state.memory, state.q, state.time, state.history),
            motion_available=state.memory.valid,
            memory_available=state.memory.available, measurement_pending=state.measurement_pending,
            visual_feedback_available=state.visual_feedback_available, q_error=state.q_error,
            visual_feedback_age=visual_feedback_feature,
            feedback_offsets=state.feedback_offsets,
            feedback_available=state.feedback_available, feedback_age=feedback_feature,
            object_relative=state.object_relative,
            **(dict(place_relative=getattr(state, 'place_relative', None)) if self.config.place_relative else {}),
            **(dict(place_goal=getattr(state, 'place_goal', None)) if getattr(self.config, 'place_goal', False) else {}))

    def picture_slots(self):
        """17.167: pictures per camera a new state keeps for a picture-memory LeWM: its memory_pictures in a runtime
        (load_runtime sets live_pictures), else 0 (training builds every row's memory from the recordings)."""
        if not getattr(self, 'live_pictures', False) or not hasattr(self.lewm, 'memory_pictures'):
            return 0
        return int(self.lewm.memory_pictures)

    def lewm_pictures(self, state):
        """The pictures the LeWM reads from a state: a picture-memory LeWM the state's own last N per camera
        (17.167), every other LeWM the three-picture memory."""
        if not hasattr(self.lewm, 'memory_pictures'):
            return state.memory
        if getattr(state, 'pictures', None) is None:
            raise ValueError("a picture-memory LeWM reads the state's picture memory, which a runtime keeps "
                             '(load_runtime: core.live_pictures)')
        return state.pictures

    def lewm_controller(self, batch, like):
        """17.153: [batch, 2] = the arm's control setting for a LeWM that reads it, as the runtime declared it
        (load_runtime's arm_p_coefficient; fcpg/v21_arm_gain.lewm_arm_setting). No default."""
        setting = getattr(self, 'arm_setting', None)
        if setting is None:
            raise ValueError("this LeWM reads the arm's control setting: load the runtime with the service's "
                             'arm_p_coefficient')
        return setting.to(like).expand(batch, -1)

    def predict(self, state, trace, query_offsets):
        pictures = self.lewm_pictures(state)
        visual = dict(patches=pictures.features, patch_valid=pictures.valid)
        if getattr(self.lewm, 'reads_visual_memory', False):
            visual.update(memory=pictures, now_time=state.time)
        if getattr(self.lewm, 'controller_input', False):  # 17.153: the arm's control setting of this runtime
            visual['controller'] = self.lewm_controller(len(state.q), state.q)
        response = self.lewm(state.q, trace, state.history, query_offsets, **visual)
        owned = CommandTrace(trace.positions.detach().clone(), trace.offsets.detach().clone(),
                             trace.valid.detach().clone(), trace.source, trace.complete)
        return Forecast(state.q.detach().clone(), state.time.detach().clone(), owned,
                        query_offsets.detach().clone(), response)

    def complete(self, previous, prediction, actual_trace, measured_path, readback_times, observation=None):
        """Only a complete factual trace may advance execution history."""
        b = len(previous.q)
        finite("actual q-path", measured_path, (b, 6, 6))
        clock("actual readback times", readback_times, (b, 6))
        actual_trace.validate(b)
        if actual_trace.source != "recorded":
            raise ValueError("completion requires recorded commands, not proposed actions")
        # section 17.80: allclose(a, b) is isclose(a, b).all(); the trace check is written without selecting rows
        if checking():
            fail_if(~torch.isclose(prediction.source_q, previous.q, atol=1e-4, rtol=0).all()
                    | ~torch.isclose(prediction.source_time, previous.time, atol=1e-9, rtol=0).all()
                    | ~torch.isclose(measured_path[:, 0], previous.q, atol=1e-4, rtol=0).all()
                    | ~torch.isclose(readback_times[:, 0], previous.time, atol=1e-9, rtol=0).all(),
                    "prediction, actual trace and source endpoint disagree")
        if checking():
            fail_if((readback_times.diff(dim=1) <= 0).any(), "actual readback times must strictly increase")
        if (actual_trace.positions.shape != prediction.trace.positions.shape
                or actual_trace.valid.shape != prediction.trace.valid.shape):
            raise ValueError("recorded command values differ from the predicted trace")
        if checking():
            fail_if((actual_trace.valid != prediction.trace.valid).any()
                    | ~(torch.isclose(actual_trace.positions, prediction.trace.positions, atol=1e-4, rtol=0)
                        | ~actual_trace.valid[..., None]).all(),
                    "recorded command values differ from the predicted trace")
        offsets = readback_times - readback_times[:, :1]
        if checking():
            fail_if((actual_trace.offsets[:, 0] >= offsets[:, 1]-1e-9).any(),
                    "first readback requires a preceding recorded command")
        if checking():
            fail_if((actual_trace.valid & (actual_trace.offsets >= offsets[:, -1:])).any(),
                    "completion precedes a recorded command")
        command_clock_equal = (torch.isclose(actual_trace.offsets, prediction.trace.offsets, atol=1e-8, rtol=0)
                               | ~actual_trace.valid).all(-1)
        query_clock_equal = torch.isclose(offsets[:, 1:], prediction.query_offsets, atol=1e-8, rtol=0).all(-1)
        retimed = ~(command_clock_equal & query_clock_equal)
        response = prediction.response
        # a forecaster whose answer does not depend on the clock (retime_invariant) would answer the same again
        if not getattr(self.lewm, 'retime_invariant', False) and retimed.any():
            # Re-evaluate only at factual command/readback times, using the
            # SOURCE q/history. No later q or image may enter this forecast.
            with torch.no_grad():
                pictures = self.lewm_pictures(previous)
                visual = dict(patches=pictures.features, patch_valid=pictures.valid)
                if getattr(self.lewm, 'reads_visual_memory', False):
                    visual.update(memory=pictures, now_time=previous.time)
                if getattr(self.lewm, 'controller_input', False):    # 17.153
                    visual['controller'] = self.lewm_controller(len(previous.q), previous.q)
                response = self.lewm(previous.q, actual_trace, previous.history, offsets[:, 1:], **visual)
        active = actual_trace.active_at(offsets[:, 1:], previous.q)
        # For dense recordings these are the actual commands active before
        # each readback, not the five Actor targets chosen by downsampling.
        pooled = previous.memory.features[:, :, -1].float().mean(-2)
        history = append_completed(previous.history, active.detach(), measured_path.detach(), pooled,
                                   offsets.diff(dim=1).to(measured_path))
        ended = readback_times[:, -1]
        previous = prepare_progress_state(previous)
        next_index = previous.memory.current_readback_index + READBACKS_PER_COMPLETED_BLOCK
        current = replace(previous, history=history,
            memory=replace(previous.memory, current_readback_index=next_index),
            history_ends=torch.cat((previous.history_ends[:, 1:], ended[:, None]), -1),
            q_error=(measured_path[:, 1:]-response.predicted_q_path[:, 1:]).detach(),
            feedback_available=torch.ones(b, device=ended.device, dtype=torch.bool),
            feedback_time=ended.detach().clone(), feedback_offsets=offsets[:, 1:].detach().clone(),
            feedback_readback_index=next_index.detach().clone(),
            feedback_retimed=retimed.detach(),
            # A successfully completed action consumes the previous visual
            # event. The newly measured endpoint below starts the next one.
            visual_error=torch.zeros_like(previous.visual_error),
            visual_feedback_available=torch.zeros_like(previous.visual_feedback_available),
            measurement_pending=torch.zeros_like(previous.measurement_pending))
        return self.observe(current, observation, measured_path[:, -1], ended)
