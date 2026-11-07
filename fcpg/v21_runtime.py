"""Model-only V21 research runtime. Hardware ownership and STOP stay external."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import threading
import time
import math
import torch
from torch import nn

from .v21_encoder import DIM, PATCHES
from .v21_core import V21Config, V21ResearchCore, CONTRACT, VERSION, current_goal_cue, current_place_cue, current_object_cue
from .v21_data import TaskVocabulary
from .v21_memory import VisualObservation, finite, clock
from .v21_models import CommandTrace
from .v21_temporal_contract import prepare_progress_state


class PreExecutionStateChanged(RuntimeError):
    """No command was dispatched: the Actor must see the new actual state."""
    def __init__(self, source_q, actual_q, age_s):
        super().__init__('pose changed or plan expired before execution')
        self.diagnostic=dict(source_q_position=source_q.detach().cpu().tolist(),
            actual_q_position=actual_q.detach().cpu().tolist(),plan_age_s=float(age_s),
            maximum_pose_change=float((actual_q-source_q).abs().max()))


@dataclass(frozen=True)
class EncodedEndpoint:
    q: torch.Tensor
    time: torch.Tensor
    observation: VisualObservation | None


class FrozenDINOFrames(nn.Module):
    """Encode only new actual exposures; no network load/download in this wrapper."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode=True):
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode(self, rgb, q_at_image, image_times, measurement_available, memory, now):
        b = len(rgb)
        if rgb.shape != (b, 2, 3, 224, 224) or rgb.dtype != torch.uint8:
            raise ValueError("encoder expects letterboxed RGB uint8 [B,2,3,224,224]")
        device = self.mean.device
        q_at_image = q_at_image.to(device)
        image_times, now = image_times.to(device), now.to(device)
        measurement_available = measurement_available.to(device)
        clock("image timestamps", image_times, (b, 2)); clock("actual time", now, (b,))
        if measurement_available.shape != (b, 2) or measurement_available.dtype != torch.bool:
            raise ValueError("invalid camera availability")
        finite("image q", q_at_image[measurement_available])
        if (measurement_available & (image_times > now[:, None]+1e-9)).any():
            raise ValueError("future camera frame")
        new = measurement_available.clone()
        if memory is not None:
            last = memory.image_times[:, :, -1]
            if (new & memory.available & (image_times < last)).any():
                raise ValueError("out-of-order camera frame")
            duplicate = new & memory.available & (image_times == last)
            if duplicate.any() and not torch.equal(q_at_image[duplicate].to(memory.q_at_image), memory.q_at_image[:, :, -1][duplicate]):
                raise ValueError("duplicate exposure with different q")
            new &= ~duplicate
        features = torch.zeros(b, 2, PATCHES, DIM, device=self.mean.device)
        if new.any():
            pixels = rgb[new.to(rgb.device)].to(device).float()/255.
            result = self.backbone.forward_features((pixels-self.mean)/self.std)
            tokens = result["x_norm_patchtokens"]
            finite("DINO patch tokens", tokens, (int(new.sum()), PATCHES, DIM))   # a DINOv2 backbone fails here under DINOv3-L
            features[new] = tokens.float()
        return VisualObservation(features, q_at_image.to(features.device), image_times.to(features.device), new.to(features.device))


class V21ResearchRuntime:
    def __init__(self, core, text_ids, *, physical_check, lease_check, clock_fn=time.monotonic, max_plan_age_s=1.):
        self.core = core.eval().requires_grad_(False)
        self.text_ids, self.physical_check, self.lease_check = text_ids, physical_check, lease_check
        self.clock, self.max_plan_age_s = clock_fn, float(max_plan_age_s)
        if not math.isfinite(self.max_plan_age_s) or self.max_plan_age_s <= 0:
            raise ValueError("plan lifetime must be finite and positive")
        self.state = self.pending = None
        self.max_plan_dwell_s = None
        self.pending_ready_time_s = None
        # An unexecuted plan owns a candidate read of actual images, not a
        # committed visual-history update. Replanning must not evict history.
        self._pending_state = None
        self.pending_action_source = None
        self.stopped = True
        self.executing = False
        self.requires_home_after_incomplete_execution = False
        self.lock = threading.RLock()
        self.source_update_tolerance = 0.0  # model-only callers retain strict matching
        self.last_dispatch_update = None
        self.last_actor_input = None
        # 'bounded': a moved pose beyond the quiet allowance or an aged plan aborts
        # dispatch (PreExecutionStateChanged). 'always': recomputable plans are
        # re-derived from the actual q instead; nothing stale is ever sent and
        # no block is ever abandoned for drift or age.
        self.pre_execution_recompute = 'bounded'
        self._pending_builder = None
        self.recomputing_at_dispatch = False
        self.last_rebase = None

    # Sources whose commands are a pure function of (state, offsets) and may be
    # re-derived at dispatch. Assisted / alignment sources keep strict matching.
    # The two C4 sources re-derive at dispatch without a second C4 plan (fcpg/v21_c4_service.py):
    # shadow = the Actor at the measured pose; control = that plus the already planned C4 offset.
    RECOMPUTABLE_SOURCES = ('v21_actor', 'v21_actor_execution_geometry',
                            'v21_actor_centroid_gated','fcpg_carry_visual_probe',
                            'v21_actor_c4_shadow', 'v21_c4_cem',
                            # paper CEM on the system LeWM (fcpg/v21_alignment_cem.py): same dispatch rule
                            'v21_actor_align_shadow', 'v21_lewm_align_cem',
                            # exploration and the learned-cue CEM (fcpg/v21_explore.py, fcpg/v21_cue_cem.py)
                            'v21_actor_explore', 'v21_actor_cue_shadow', 'v21_lewm_cue_cem')

    def configure_pre_execution_recompute(self, mode):
        """'bounded' (default) or 'always'. See ``begin_execution``."""
        if mode not in ('bounded', 'always'):
            raise ValueError("pre-execution recompute mode must be 'bounded' or 'always'")
        if self.pending is not None or self.executing:
            raise RuntimeError('cannot configure an active transaction')
        self.pre_execution_recompute = mode

    def configure_plan_dwell_limit(self, seconds):
        """Optional legacy wait limit; explicit ROS debugging leaves this unset."""
        if seconds is not None and (isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError('plan dwell limit must be positive and finite or None')
        with self.lock:
            if self.pending is not None or self.executing:
                raise RuntimeError('cannot configure an active transaction')
            self.max_plan_dwell_s = None if seconds is None else float(seconds)

    def pending_is_recently_ready(self, now=None):
        if self.pending is None:
            return False
        if self.max_plan_dwell_s is None:
            return True
        now = float(self.clock() if now is None else now)
        ready = self.pending_ready_time_s
        return (ready is not None and math.isfinite(ready) and math.isfinite(now)
                and 0 <= now-ready <= self.max_plan_dwell_s)

    def _recompute_commands(self, source, offsets):
        builder = self._pending_builder
        if builder is None:
            return self.core.act(source, self.text_ids, offsets).commands_position
        self.recomputing_at_dispatch = True
        try:
            if getattr(builder, 'wants_command_offsets', False):
                commands = builder(source, offsets)
            else:
                commands = builder(source)
        finally:
            self.recomputing_at_dispatch = False
        finite('recomputed commands', commands, (1, 5, 6))
        return commands

    def pending_matches(self, actual_q):
        """True when the pending plan's source q equals this readback (1e-4)."""
        with self.lock:
            if self.pending is None or self.stopped:
                return False
            return bool(torch.allclose(actual_q, self.pending.source_q, atol=1e-4, rtol=0))

    @torch.inference_mode()
    def rebase_pending(self, actual_q, actual_time):
        """Re-derive the pending plan from a newer readback BEFORE the dispatch epoch.

        Heavy work (reader, Actor/builder, physical audit, LeWM forecast) happens
        here with no motor deadline. The same actual image bank is read at the
        new pose; no image is appended or re-timestamped, and no command from
        the old pose survives. Only recomputable sources in 'always' mode.
        """
        with self.lock:
            self._lease()
            if self.stopped or self.pending is None or self._pending_state is None or self.executing:
                raise RuntimeError('no rebasable V21 plan')
            if self.pre_execution_recompute != 'always' or self.pending_action_source not in self.RECOMPUTABLE_SOURCES:
                raise RuntimeError('pending plan is not recomputable')
            try:
                finite('rebase actual q', actual_q, (1, 6))
                clock('rebase readback time', actual_time, (1,))
                self._actual_time(actual_time)
                age = actual_time.item()-self.pending.source_time.item()
                delta = float((actual_q-self.pending.source_q).abs().max())
                old_q = self.pending.source_q.detach().cpu().tolist()
                source = replace(self._pending_state, q=actual_q.detach().clone(), time=actual_time.detach().clone())
                source = replace(source, grounding=self.core.reader(source.memory, actual_q, actual_time, source.history,
                                                                    task_ids=source.task_ids))
                trace = self.pending.trace
                commands = self._recompute_commands(source, trace.offsets)
                trace = replace(trace, positions=commands.detach().clone())
                self._physical(actual_q, trace.positions)
                self.pending = self.core.predict(source, trace, self.pending.query_offsets)
                self._pending_state = source
                self.last_rebase = dict(maximum_pose_change=delta, plan_age_s=float(age),
                    from_q_position=old_q, to_q_position=actual_q.detach().cpu().tolist(),
                    action_source=self.pending_action_source, commands_recomputed=True,
                    images_reused_without_retimestamp=True)
                return trace.positions.detach().clone()
            except BaseException:
                self.stop(); raise

    def configure_source_update_tolerance(self, value):
        """Use at most the deployed hardware's existing quiet-step allowance.

        This permits recomputation, not execution of a stale command. Joint,
        clearance, timing and command-transport checks remain unchanged.
        """
        value=float(value)
        if not math.isfinite(value) or not 0 <= value <= .25:
            raise ValueError('source update allowance must be within [0, 0.25]')
        if self.pending is not None or self.executing:
            raise RuntimeError('cannot configure an active transaction')
        self.source_update_tolerance=value

    def _lease(self):
        try:
            self.lease_check()
        except BaseException:
            self.stop(); raise

    def _actual_time(self, value):
        if not torch.isfinite(value).all() or (value > self.clock()+1e-6).any():
            self.stop()
            raise RuntimeError("actual timestamp is nonfinite or from the future")

    # Existing tolerance for near-identical acquisition poses. It does not
    # assert that the scene or any independently moving object is unchanged.
    SAME_POSE_TOLERANCE = .5

    def _refresh_memory(self, previous, endpoint):
        """Same-pose refresh replaces the newest exposure instead of appending.

        Training writes one exposure per view per transaction, and the Actor
        acts when the newest frame has age zero. The resident dashboard,
        plan() and execute() each capture again while the arm is still; if
        those frames were appended, the three-slot memory would hold three
        copies of one pose and no motion. If they were dropped, the newest
        slot would age for seconds, which training never shows either. So an
        near-identical acquisition pose replaces the newest slot with the real
        new content/clock, while a changed exposure pose or empty view appends.
        """
        observation = endpoint.observation
        if previous is None or observation is None:
            return previous, observation
        previous = prepare_progress_state(previous)
        memory = previous.memory
        # Replacement must not bypass exposure identity, ordering, dtype or
        # finite-value checks by clearing the insertion mask first.
        memory.validate_observation(observation, endpoint.time)
        # Slot replacement compares the two actual exposure poses per view.
        # Current q may have changed during capture/encoding; it still addresses
        # the reader below, but cannot relabel an exposure as a different pose.
        newest_q = memory.q_at_image[:, :, -1].to(observation.q_at_image)
        moved = (observation.q_at_image - newest_q).abs().amax(-1) > self.SAME_POSE_TOLERANCE
        still = observation.measurement_available & memory.available & ~moved
        if not still.any():
            return previous, observation
        mask = still[..., None]
        features = torch.where(mask[..., None, None], observation.features.to(memory.features)[:, :, None], memory.features[:, :, -1:])
        q_at = torch.where(mask[..., None], observation.q_at_image.to(memory.q_at_image)[:, :, None], memory.q_at_image[:, :, -1:])
        times = torch.where(mask, observation.image_times.to(memory.image_times)[:, :, None], memory.image_times[:, :, -1:])
        progress = torch.where(still, memory.current_readback_index[:, None].expand(-1, 2),
                               memory.readback_indices[:, :, -1])
        replaced = replace(memory, features=torch.cat((memory.features[:, :, :-1], features), 2),
                           q_at_image=torch.cat((memory.q_at_image[:, :, :-1], q_at), 2),
                           image_times=torch.cat((memory.image_times[:, :, :-1], times), 2),
                           readback_indices=torch.cat((memory.readback_indices[:, :, :-1], progress[:, :, None]), 2))
        remaining = replace(observation, measurement_available=observation.measurement_available & ~still)
        return replace(previous, memory=replaced), remaining

    def _refresh_endpoint(self, previous, endpoint):
        """Refresh G, but do not replace an unconsumed execution event E.

        Training calls Actor directly after completion's actual image update.
        A resident camera refresh is not another completed robot transaction.
        It may change the current grounding, never relabel a still-frame delta
        as the previous action's visual feedback, and never commit a duplicate
        exposure of an unmoved pose. Completion below deliberately uses
        core.observe directly to produce the NEXT event.
        """
        if previous is not None:
            previous = prepare_progress_state(previous)
        base, observation = self._refresh_memory(previous, endpoint)
        # a task-reading FCPG (V21Config.reader_task) starts its state with the task words; earlier models unchanged
        reads_task = getattr(getattr(self.core, 'config', None), 'reader_task', False)
        task = dict(task_ids=self.text_ids) if base is None and reads_task else {}
        current = self.core.observe(base, observation, endpoint.q, endpoint.time, **task)
        # A same-pose replacement is not a missing camera. Use original delivery
        # availability here, not the temporary FIFO-insertion mask.
        current = replace(current, object_relative=current_object_cue(endpoint.observation, endpoint.q),
                          place_relative=current_place_cue(endpoint.observation, endpoint.q),
                          place_goal=current_goal_cue(endpoint.observation, endpoint.q))
        if previous is None:
            return current
        # A fresh frame counts as the current measurement even when it only
        # replaced the newest slot, exactly as a completion frame would.
        seen = endpoint.observation.measurement_available if endpoint.observation is not None else current.measurement_available
        current = replace(current, measurement_available=current.measurement_available | seen,
                          measurement_pending=current.measurement_pending | seen)
        # Once an execution event exists, EVERY view's feedback fields belong to
        # that event until the next completion. A view whose completion frame
        # was missing must stay "no visual feedback" instead of acquiring a
        # still-frame delta when its camera returns during a refresh.
        keep = previous.feedback_available[:, None].expand(-1, 2)
        return replace(current,
            visual_error=torch.where(keep[..., None, None], previous.visual_error, current.visual_error),
            visual_feedback_available=torch.where(keep, previous.visual_feedback_available,
                                                  current.visual_feedback_available),
            visual_feedback_time=torch.where(keep, previous.visual_feedback_time, current.visual_feedback_time),
            visual_feedback_readback_indices=torch.where(keep, previous.visual_feedback_readback_indices,
                                                        current.visual_feedback_readback_indices))

    @torch.inference_mode()
    def observe(self, endpoint):
        with self.lock:
            self._lease()
            if self.executing or self.pending is not None:
                raise RuntimeError("cannot replace an active transaction endpoint")
            self._actual_time(endpoint.time)
            self.state = self._refresh_endpoint(self.state, endpoint)
            # Observation never rearms a STOP latch.
            return self.status()

    def _physical(self, q, commands):
        before = commands.detach().clone()
        q_before = q.detach().clone()
        checked_q, checked_commands = q_before.clone(), before.clone()
        self.physical_check(checked_q, checked_commands)
        if not torch.equal(before, checked_commands) or not torch.equal(q_before, checked_q):
            raise RuntimeError("physical admission changed policy commands or actual q")

    def _planning_state(self, endpoint):
        state = self.state
        if endpoint is not None:
            if not isinstance(endpoint, EncodedEndpoint):
                raise TypeError("planning requires an actual EncodedEndpoint")
            self._actual_time(endpoint.time)
            state = self._refresh_endpoint(state, endpoint)
        if state is None or len(state.q) != 1 or not state.memory.available.any():
            raise RuntimeError("one observed real scene is required before physical planning")
        if not 0 <= self.clock()-state.time.item() <= self.max_plan_age_s:
            raise RuntimeError("actual readback endpoint is stale or from the future")
        return state

    @staticmethod
    def _plan_intervals(state, intervals):
        dt = torch.full((1, 5), .1, device=state.q.device) if intervals is None else intervals
        if dt.shape != (1, 5) or not torch.allclose(dt, torch.full_like(dt, .1), atol=1e-7, rtol=0):
            raise ValueError("runtime executes five commands at 0.1 seconds")
        return dt

    @torch.inference_mode()
    def preview(self, endpoint, intervals=None, *, command_builder=None):
        """Read fresh images and audit a candidate without committing either.

        ``command_builder`` follows :meth:`plan`: a builder marked with
        ``wants_command_offsets`` receives ``(state, command_offsets)`` and any
        other builder receives only ``state``.  The default remains the raw
        Actor.  Preview never creates ``pending`` state or writes its refreshed
        real-image candidate into committed memory.
        """
        with self.lock:
            self._lease()
            if self.executing or self.pending is not None:
                raise RuntimeError("cannot preview during an active transaction")
            if command_builder is not None and not callable(command_builder):
                raise TypeError("preview command builder must be callable")
            state = self._planning_state(endpoint)
            dt = self._plan_intervals(state, intervals)
            offsets = torch.cat((dt.new_zeros(1, 1).double(), dt.double().cumsum(-1)[:, :-1]), -1)
            if command_builder is None:
                commands = self.core.act(state, self.text_ids, offsets).commands_position
            elif getattr(command_builder, "wants_command_offsets", False):
                commands = command_builder(state, offsets)
            else:
                commands = command_builder(state)
            finite("preview commands", commands, (1, 5, 6))
            admission = dict(allowed=True, reason=None)
            try:
                self._physical(state.q, commands)
            except RuntimeError as exc:
                admission = dict(allowed=False, reason=str(exc))
            forecast = self.core.predict(state, CommandTrace.policy(commands, dt), dt.double().cumsum(-1))
            return commands.detach().clone(), forecast, admission

    @torch.inference_mode()
    def plan(self, intervals=None, *, endpoint=None, command_builder=None, action_source='v21_actor'):
        with self.lock:
            self._lease()
            if self.requires_home_after_incomplete_execution:
                raise RuntimeError('HOME required after incomplete execution before planning')
            if self.executing or self.pending is not None:
                raise RuntimeError("one actual endpoint and no outstanding transaction required")
            try:
                if action_source not in ('v21_actor','v21_actor_grounding_corrected','v21_actor_execution_geometry',
                                         'v21_lewm_cem',
                                         # C4 planner (V21_LEWM_C4_CONTROL_PLAN_20261011): phase A runs
                                         # the unmodified Actor's commands under its own label while C4
                                         # only records; phase B runs C4's selection.
                                         'v21_actor_c4_shadow', 'v21_c4_cem',
                                         'v21_actor_align_shadow', 'v21_lewm_align_cem',
                                         'v21_actor_explore', 'v21_actor_cue_shadow', 'v21_lewm_cue_cem',
                                         'v21_actor_centroid_gated',
                                         'fcpg_carry_visual_probe',
                                         'visual_alignment_probe',
                                         'visual_alignment_return','visual_alignment_servo','visual_alignment_open_departure',
                                         'visual_alignment_view_step','visual_alignment_reach',
                                         # Explicit, open-jaw calibration only.  It is never
                                         # attributed to Actor and intentionally remains
                                         # non-recomputable at dispatch like other alignment
                                         # interventions.
                                         'column_dino_fg_probe'):
                    raise ValueError('unknown action source')
                if (command_builder is None)!=(action_source=='v21_actor'):
                    raise ValueError('external commands must not be labelled as Actor output')
                state = self._planning_state(endpoint)
                dt = self._plan_intervals(state, intervals)
                command_offsets = torch.cat((dt.new_zeros(1, 1).double(), dt.double().cumsum(-1)[:, :-1]), -1)
                if command_builder is None:
                    action = self.core.act(state, self.text_ids, command_offsets).commands_position
                elif getattr(command_builder, 'wants_command_offsets', False):
                    action = command_builder(state, command_offsets)
                else:
                    action = command_builder(state)
                finite('planned commands',action,(1,5,6))
                self._physical(state.q, action)  # BEFORE LeWM, never ranking
                trace = CommandTrace.policy(action, dt)
                self.pending = self.core.predict(state, trace, dt.double().cumsum(-1))
                self._pending_state = state
                self.pending_action_source = action_source
                self._pending_builder = command_builder
                self.pending_ready_time_s = float(self.clock())
                self.last_rebase = None
                self.stopped = False
                return action.detach().clone()
            except BaseException:
                self.stop()
                raise

    @torch.inference_mode()
    def begin_execution(self, actual_q, actual_time):
        with self.lock:
            self._lease()
            if self.stopped or self.pending is None or self._pending_state is None or self.executing:
                raise RuntimeError("no executable V21 plan")
            try:
                finite("pre-execution actual q", actual_q, (1, 6))
                clock("execution readback time", actual_time, (1,))
                self._actual_time(actual_time)
                self.last_dispatch_update=None
                self.last_actor_input=None
                age=actual_time.item()-self.pending.source_time.item()
                if age < 0:
                    raise RuntimeError('execution clock precedes planned source time')
                changed=not torch.allclose(actual_q,self.pending.source_q,atol=1e-4,rtol=0)
                delta=float((actual_q-self.pending.source_q).abs().max())
                stale=age > self.max_plan_age_s
                if stale and self.pending_action_source == 'v21_lewm_cem':
                    # A fresh joint forecast cannot refresh an old visual
                    # ranking. The host must capture and plan again before IO.
                    raise PreExecutionStateChanged(self.pending.source_q,actual_q,age)
                if self.max_plan_dwell_s is not None and not self.pending_is_recently_ready(actual_time.item()):
                    error = PreExecutionStateChanged(self.pending.source_q, actual_q, age)
                    error.diagnostic['reason'] = 'post_planning_wait_expired'
                    error.diagnostic['pending_ready_time_s'] = self.pending_ready_time_s
                    error.diagnostic['maximum_post_planning_wait_s'] = self.max_plan_dwell_s
                    raise error
                always=(self.pre_execution_recompute=='always'
                        and self.pending_action_source in self.RECOMPUTABLE_SOURCES)
                cem_hold=(self.pending_action_source=='v21_lewm_cem'
                          and delta <= max(float(self.source_update_tolerance), 1.0))
                if always:
                    # Never abort: re-derive the commands from the actual q for
                    # any drift or age. Physical audit and LeWM forecast below
                    # always run against the actual q.
                    recompute=changed or stale
                else:
                    can_update=(self.pending_action_source=='v21_actor'
                                and delta<=self.source_update_tolerance and not stale)
                    if cem_hold:
                        # Only a still-fresh CEM ranking may retain its commands
                        # within this pose tolerance; never rewrite it as Actor.
                        recompute=False
                    elif (changed and not can_update) or stale:
                        raise PreExecutionStateChanged(self.pending.source_q,actual_q,age)
                    else:
                        recompute=changed
                source = replace(self._pending_state, q=actual_q.detach().clone(), time=actual_time.detach().clone())
                trace=self.pending.trace
                if recompute:
                    # The same actual image bank is read at the newer measured
                    # pose. Do not append images, change exposure q/time, invent
                    # a completed transition, or reuse stale Actor commands.
                    source=replace(source,grounding=self.core.reader(source.memory,actual_q,actual_time,source.history,
                                                                     task_ids=source.task_ids))
                    commands=self._recompute_commands(source,trace.offsets)
                    trace=replace(trace,positions=commands.detach().clone())
                self._physical(actual_q,trace.positions)
                forecast = self.core.predict(source, trace, self.pending.query_offsets)
                if self.clock() >= actual_time.item()+forecast.query_offsets[0, 0].item():
                    raise RuntimeError("first readback deadline has passed")
                if recompute:
                    self.last_dispatch_update=dict(maximum_pose_change=delta,plan_age_s=float(age),
                        from_q_position=self.pending.source_q.detach().cpu().tolist(),
                        to_q_position=actual_q.detach().cpu().tolist(),commands_recomputed=True,
                        expired_plan_recomputed=bool(stale),action_source=self.pending_action_source,
                        images_reused_without_retimestamp=True)
                self.state = source
                # Unchanged-q dispatch retimes LeWM but keeps the earlier Actor
                # commands. Preserve the state that REALLY generated them.
                self.last_actor_input = source if recompute else self._pending_state
                self.pending = forecast
                self._pending_state = None
                self._pending_builder = None
                self.pending_ready_time_s = None
                self.executing = True
                return self.pending.trace.positions.detach().clone()
            except BaseException:
                self.stop(); raise

    @torch.inference_mode()
    def complete(self, actual_trace, measured_path, readback_times, endpoint, *, feedback_path=None,feedback_times=None):
        with self.lock:
            self._lease()
            if self.stopped or not self.executing or self.pending is None:
                raise RuntimeError("no executing V21 transaction")
            try:
                self._actual_time(readback_times)
                self._actual_time(endpoint.time)
                nominal_prediction=None
                model_path,model_times=measured_path,readback_times
                if feedback_path is not None or feedback_times is not None:
                    if feedback_path is None or feedback_times is None:
                        raise ValueError('following feedback needs both path and clocks')
                    finite('following feedback path',feedback_path,measured_path.shape)
                    clock('following feedback clocks',feedback_times,readback_times.shape)
                    self._actual_time(feedback_times)
                    if (not torch.equal(feedback_path[:,:-1],measured_path[:,:-1])
                            or not torch.equal(feedback_times[:,:-1],readback_times[:,:-1])
                            or (feedback_times[:,-1]<=readback_times[:,-1]).any()
                            or (feedback_times[:,-1]>endpoint.time).any()):
                        raise ValueError('following feedback must preserve the nominal prefix and end at an actual later sample')
                    nominal_offsets=readback_times[:,1:]-readback_times[:,:1]
                    nominal_prediction=self.core.predict(self.state,actual_trace,nominal_offsets).response.predicted_q_path
                    model_path,model_times=feedback_path,feedback_times
                actual = self.core.complete(self.state, self.pending, actual_trace, model_path, model_times)
                # A later camera endpoint has its OWN measured q/time, not a
                # renamed q5 sample. Actual q-path feedback retains its time.
                self.state = self.core.observe(actual, endpoint.observation, endpoint.q, endpoint.time)
                self.pending = None; self.executing = False
                result=self.status()
                if nominal_prediction is not None:
                    result['nominal_retimed_prediction']=nominal_prediction.detach().cpu().tolist()
                return result
            except BaseException:
                self.stop(); raise

    def stop(self):
        with self.lock:
            self.stopped = True; self.pending = None; self.executing = False
            self._pending_state = None
            self.pending_action_source = None
            self._pending_builder = None
            self.pending_ready_time_s = None

    def status(self):
        result = dict(version=VERSION, contract=CONTRACT, stopped=self.stopped,
                      pending=self.pending is not None, executing=self.executing,
                      position_unit="lerobot_calibrated_position", hardware_owned=False,
                      requires_home_after_incomplete_execution=self.requires_home_after_incomplete_execution,
                      source_update_tolerance=self.source_update_tolerance,
                      pre_execution_recompute=self.pre_execution_recompute)
        result.update(max_plan_age_s=self.max_plan_age_s, max_plan_dwell_s=self.max_plan_dwell_s,
                      pending_ready_time_s=self.pending_ready_time_s)
        if self.state is not None:
            s = self.state
            result.update(q=s.q.tolist(), time=s.time.tolist(),
                memory_available=s.memory.available.tolist(), measurement_available=s.measurement_available.tolist(),
                measurement_pending=s.measurement_pending.tolist(),
                visual_feedback_available=s.visual_feedback_available.tolist(), feedback_available=s.feedback_available.tolist(),
                visual_feedback_time=s.visual_feedback_time.tolist(), feedback_retimed=s.feedback_retimed.tolist(),
                feedback_offsets=s.feedback_offsets.tolist(),
                history_chunks=s.history.valid.sum(-1).tolist(), image_times=s.memory.image_times.tolist())
            index = getattr(s.memory, 'current_readback_index', None)
            if index is not None:
                result['readback_progress'] = dict(unit='completed_policy_readback_sample_index',
                    current_index=index.tolist(), capture_indices=s.memory.readback_indices.tolist(),
                    feedback_index=s.feedback_readback_index.tolist(),
                    visual_feedback_indices=s.visual_feedback_readback_indices.tolist(),
                    session_id=s.progress_session_id, origin=s.progress_origin,
                    real_timestamps_preserved=True)
        return result


def _expected_training_pair_exposures(config, schedule):
    """Keep legacy budgets and admit the declared original TRAIN-only split."""
    profile = config.get("training_data_contract")
    archive_rows, dense_rows = 320000, 20000
    if "training_data_contract" in config:
        expected = dict(schema="v21-original-source-split-v1",
            archive_train_rows=272000, archive_development_rows=48000,
            dense_train_rows=17000, dense_development_rows=3000,
            fish_train_episodes=[0, 1], fish_development_episodes=[3],
            fish_test_episode_used=False)
        if not isinstance(profile, dict) or any(
                type(profile.get(key)) is not type(value) or profile[key] != value
                or (isinstance(value, list) and any(type(item) is not int for item in profile[key]))
                for key, value in expected.items()):
            raise ValueError("unsupported formal training data population contract")
        archive_rows, dense_rows = profile["archive_train_rows"], profile["dense_train_rows"]
    return dict(lewm=archive_rows*4*schedule.archive_passes,
                actor=archive_rows*4*schedule.archive_passes,
                dense=dense_rows*4*schedule.dense_passes,
                fish=schedule.fish_updates*schedule.fish_chains_per_batch*4)


OVERHAUL_KIND = 'v21-fcpg-actor-overhaul-v1'   # scripts/train_v21_fcpg_actor.py


def _load_overhaul_runtime(root, config, task_text, *, device, physical_check, lease_check, **kwargs):
    """A completed run of scripts/train_v21_fcpg_actor.py: its own FCPG/Actor, the frozen full-patch LeWM."""
    from .v21_training import load_checkpoint
    from .v21_full_patch_lewm import V21FullPatchLeWM, architecture_from_state
    if config.get("complete") is not True or type(config.get("fish_updates")) is not int:
        raise ValueError("incomplete FCPG/Actor overhaul run")
    model_config = V21Config(**config["model"])
    vocabulary = TaskVocabulary(tuple(config["vocabulary"]))
    core = V21ResearchCore(model_config).to(device)
    blob = torch.load(root/"fish"/"final.pt", map_location="cpu", weights_only=False)
    architecture = architecture_from_state(blob.get("model", {}), "lewm.")     # 17.106: the LeWM's own size
    del blob
    core.lewm = V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40,
                                 **architecture).to(device)
    payload = load_checkpoint(root/"fish"/"final.pt", core, vocabulary, None, stage="fish")
    if payload.get("complete") is not True or payload.get("step") != config["fish_updates"]:
        raise ValueError("incomplete final Actor checkpoint")
    runtime = V21ResearchRuntime(core, vocabulary.encode([task_text], device=device),
        physical_check=physical_check, lease_check=lease_check, **kwargs)
    runtime.target_token_repair = None
    return runtime


def load_runtime(root, task_text, *, device, physical_check, lease_check, arm_p_coefficient=None, **kwargs):
    """A formal run's runtime. 17.153: arm_p_coefficient (None = LeRobot's 16, or 32: the service's
    --arm-p-coefficient) is the control setting a LeWM that reads it gets, with the planner input 1 (this runtime is
    the service's planner; fcpg/v21_arm_gain.lewm_arm_setting)."""
    from .v21_arm_gain import lewm_arm_setting
    runtime = _load_runtime(root, task_text, device=device, physical_check=physical_check, lease_check=lease_check,
                            **kwargs)
    runtime.core.arm_setting = lewm_arm_setting(arm_p_coefficient, device=next(runtime.core.parameters()).device)
    runtime.core.live_pictures = True     # 17.167: a picture-memory LeWM's states keep their last N pictures per camera
    if runtime.core.picture_slots() and getattr(runtime.core.config, 'wrist_only', False):
        # James 11/01 「雙視角雙視角」: the LeWMs learned to read both cameras; the wrist alone misleads them
        raise ValueError('a picture-memory LeWM reads both cameras: this package keeps the wrist camera only')
    return runtime


def _load_runtime(root, task_text, *, device, physical_check, lease_check, **kwargs):
    from .v21_training import load_checkpoint, V21Schedule
    root = Path(root)
    config = json.loads((root/"config.json").read_text())
    if config.get("contract") != CONTRACT or config.get("smoke") is not False:
        raise ValueError("only formal V21 research runs may load")
    if config.get("kind") == OVERHAUL_KIND:
        return _load_overhaul_runtime(root, config, task_text, device=device, physical_check=physical_check,
                                      lease_check=lease_check, **kwargs)
    model_config = V21Config(**config["model"])
    vocabulary = TaskVocabulary(tuple(config["vocabulary"]))
    schedule = V21Schedule(**config["schedule"])
    metrics = json.loads((root/"metrics.json").read_text())
    expected = _expected_training_pair_exposures(config, schedule)
    for stage, exposure in expected.items():
        if metrics.get(stage, {}).get("complete") is not True or metrics[stage].get("pair_exposures") != exposure:
            raise ValueError(f"formal {stage} exposure/completion contract is incomplete")
    core = V21ResearchCore(model_config).to(device)
    blob = torch.load(root/"fish"/"final.pt", map_location="cpu", weights_only=False)
    if any(str(key).startswith(("lewm.film.", "lewm.concat_inputs")) for key in blob.get("model", {})):
        from .v21_full_patch_lewm import V21FullPatchLeWM, architecture_from_state
        core.lewm = V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(),
                                     horizon=40, **architecture_from_state(blob["model"], "lewm.")).to(device)
    del blob
    payload = load_checkpoint(root/"fish"/"final.pt", core, vocabulary, schedule, stage="fish")
    if payload.get("complete") is not True or payload.get("step") != schedule.fish_updates:
        raise ValueError("incomplete final Actor checkpoint")
    runtime = V21ResearchRuntime(core, vocabulary.encode([task_text], device=device),
        physical_check=physical_check, lease_check=lease_check, **kwargs)
    runtime.target_token_repair = config.get("target_token_repair")
    return runtime
