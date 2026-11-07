"""C4 planner wiring for the live service (docs/V21_LEWM_C4_CONTROL_PLAN_20261011.md sections 2, 3, 5).

mode 'shadow' (phase A): the unmodified Actor's five commands execute; C4 plans the same block and is
only recorded. mode 'control' (phase B): C4's selection executes; any planner stop raises, so the
block is not executed and the service stops (no fallback to the Actor).

What runs outside the planning call, none of it holding a motor or writing in the control loop:
- a sampler thread keeps references to lerobot's latest camera frames (30 Hz look-ups);
- the bridge's write function is wrapped to note (time, command) of every transmitted command in
  memory only; the wrapper cannot change or delay the write beyond appending to a deque;
- while a capped trial runs, a recorder thread stores the C4 input of every 0.1 s (the letterboxed
  224 RGB pair and the worker's camera check) and the transmitted commands; they are kept in memory
  and written to <output>/c4-evidence/ when the trial ends, for the hindsight test after the trial.

Dispatch: the runtime re-derives a pending block at the pose it measures just before the first write
(the arm is often still settling). A second C4 plan would not fit before the first readback deadline,
so at dispatch shadow returns the Actor's commands for the measured pose and control returns those
plus the offset C4 already chose for this block (refused when that plan is older than
DISPATCH_MAX_AGE_S). 2026-10-11: without this every block whose pose had moved by more than 1e-4
between planning and dispatch was abandoned (two of the three phase A trials ended that way).
"""
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .v21_c4_planner import FUTURE, VIEWS, C4PlanError

MODES = ('shadow', 'control')
SOURCES = dict(shadow='v21_actor_c4_shadow', control='v21_c4_cem')
SAMPLE_PERIOD_S = 1 / 30
EVIDENCE_PERIOD_S = 0.1
EVIDENCE_FLUSH = 900        # instants (90 s, ~270 MB) before a write is forced in the middle of a trial
DISPATCH_MAX_AGE_S = 2.0    # control: the planned C4 offset may be re-based on the measured pose this long


def latest_frame(camera):
    """lerobot OpenCVCamera's newest frame reference (no copy, no new-frame event consumed)."""
    lock = getattr(camera, 'frame_lock', None)
    if lock is None:
        return getattr(camera, 'latest_frame', None)
    with lock:
        return camera.latest_frame


class C4ServiceAdapter:
    def __init__(self, mode, planner_factory):
        if mode not in MODES:
            raise ValueError('C4 planner mode must be shadow or control')
        self.mode, self.factory = mode, planner_factory
        self.source = SOURCES[mode]
        self.planner = None
        self.record = None
        self.identity = None
        self.stop_event = threading.Event()
        self.threads = []
        self.pending_commands = deque()
        self.sampler_errors = 0
        self.evidence_dir = None
        self.evidence = []
        self.evidence_error = None
        self.offset5 = None
        self.planned_at_s = None

    # ----------------------------------------------------------------- set up
    def attach(self, service):
        owner = service.owner
        rt = owner.runtime
        if getattr(rt.core.actor, 'horizon', None) != 40:
            raise ValueError('C4 planning needs the native 40-step Actor checkpoint')
        self.service, self.clock = service, service.clock
        self.planner, self.identity = self.factory(clock=self.clock)
        bridge = owner.bridge
        bridge_clock = getattr(bridge, 'clock', None)
        if callable(bridge_clock) and abs(float(bridge_clock()) - float(self.clock())) > .05:
            raise ValueError('C4 planner and bridge clocks differ; transmitted command times would not line up')
        original = bridge.write_q

        def noted_write(command, previous):
            actual = original(command, previous)
            try:
                stamp = float(self.clock())
                value = np.asarray(actual, np.float32).copy()
                self.planner.commands.add(stamp, value)
                self.pending_commands.append((stamp, value))
            except Exception:
                pass
            return actual
        bridge.write_q = noted_write
        self.evidence_dir = Path(service.output) / 'c4-evidence'
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._trial_running()          # a service without the trial interface is refused here, not hidden later
        cameras = getattr(getattr(owner.io, 'follower', None), 'cameras', None) or {}
        for target, name in ((self._sample, 'c4-frame-sampler'), (self._record_evidence, 'c4-evidence')):
            thread = threading.Thread(target=target, args=(cameras,), name=name, daemon=True)
            thread.start()
            self.threads.append(thread)

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=2.)
        self._flush(force=True)

    def _sample(self, cameras):
        while not self.stop_event.wait(SAMPLE_PERIOD_S):
            for view in VIEWS:
                camera = cameras.get(view)
                if camera is None:
                    continue
                try:
                    self.planner.frames.add(float(self.clock()), view, latest_frame(camera))
                except Exception:
                    self.sampler_errors += 1

    # ----------------------------------------------------------- evidence (A)
    def _trial_running(self):
        """CappedTrial.running is a property. 2026-10-11: it was called like a method and the error was
        swallowed, so phase A stored no frame at all; nothing is swallowed here any more."""
        running = self.service.trial.running
        return bool(running() if callable(running) else running)

    def _record_evidence(self, cameras):
        try:
            self._record_evidence_loop()
        except Exception as error:      # the recorder never takes the service down, but it says so
            self.evidence_error = f'{type(error).__name__}: {error}'

    def _record_evidence_loop(self):
        due = time.monotonic()
        while True:
            due = max(due + EVIDENCE_PERIOD_S, time.monotonic())      # an absolute 10 Hz schedule, no drift
            if self.stop_event.wait(max(0., due - time.monotonic())):
                return
            self._evidence_tick()

    def _evidence_tick(self):
        """One 0.1 s instant: stored while a capped trial runs, everything written out once it does not."""
        p = self.planner
        if not self._trial_running():
            self._flush(force=True)
            return
        now = float(self.clock())
        boxed = np.zeros((len(VIEWS), 224, 224, 3), np.uint8)
        reasons, seen = [], []
        for c, view in enumerate(VIEWS):
            frame, t = p.frames.nearest(view, now)
            if frame is None:
                reasons.append(-1)
                seen.append(np.nan)
                continue
            image = p.letterbox(np.asarray(frame))
            reason, _ = p.camera_check(np.asarray(frame), image)
            boxed[c] = image
            reasons.append(int(reason))
            seen.append(t)
        self.evidence.append((now, boxed, np.asarray(reasons, np.int8), np.asarray(seen, np.float64)))
        if len(self.evidence) >= EVIDENCE_FLUSH:
            self._flush()

    def _flush(self, force=False):
        if self.evidence_dir is None:
            return
        batch = []
        if self.evidence and (force or len(self.evidence) >= EVIDENCE_FLUSH):
            batch, self.evidence = self.evidence, []
        commands = []
        while self.pending_commands:
            commands.append(self.pending_commands.popleft())
        stamp = time.time_ns()
        if batch:
            np.savez(self.evidence_dir / f'frames-{stamp}.npz',
                     times=np.asarray([b[0] for b in batch], np.float64),
                     frames=np.stack([b[1] for b in batch]),
                     camera_reason=np.stack([b[2] for b in batch]),
                     frame_seen_s=np.stack([b[3] for b in batch]))
        if commands:
            with open(self.evidence_dir / 'commands.jsonl', 'a') as stream:
                for t, value in commands:
                    stream.write(json.dumps(dict(t=t, command=value.tolist())) + '\n')

    # ------------------------------------------------------------ planning
    def builder(self, state, command_offsets):
        """Runtime command builder: -> [1, 5, 6]. Sets self.record."""
        service = self.service
        rt = service.owner.runtime
        actor5 = rt.core.act(state, rt.text_ids, command_offsets).commands_position
        if getattr(rt, 'recomputing_at_dispatch', False):
            return self._at_dispatch(actor5)
        self.offset5, self.planned_at_s = None, float(self.clock())
        fresh = not bool(state.history.valid.any())       # after HOME: no completed block of this run yet
        if fresh:
            self.planner.commands.clear()
        dt40 = torch.full((1, FUTURE), .1, device=state.q.device, dtype=torch.float64)
        offsets40 = torch.cat((dt40.new_zeros(1, 1), dt40.cumsum(-1)[:, :-1]), -1)
        full = rt.core.act(state, rt.text_ids, offsets40).commands_position
        record = dict(mode=self.mode, action_source=self.source, identity=self.identity,
                      recorder=dict(sampler_errors=self.sampler_errors, evidence_error=self.evidence_error),
                      source_time_s=float(state.time[0]), timeline_cleared=fresh,
                      actor_first5_matches_native40=bool(torch.allclose(actor5.float(), full[:, :5].float(), atol=1e-4)))

        def admissible(commands5):
            candidate = torch.as_tensor(np.asarray(commands5), dtype=actor5.dtype, device=actor5.device)[None]
            try:
                rt._physical(state.q, candidate)
                return True
            except Exception:
                return False
        actor = rt.core.actor
        bounds = [None if getattr(actor, name, None) is None else getattr(actor, name).detach().float().cpu().numpy()
                  for name in ('position_lower', 'position_upper')]
        try:
            selected, plan = self.planner.plan(
                full[0].detach().float().cpu().numpy(), state.q[0].detach().float().cpu().numpy(),
                admissible=admissible, lower=bounds[0], upper=bounds[1])
            record.update(plan, available=True)
        except C4PlanError as error:
            record.update(available=False, reason=error.reason, detail=None if error.detail is None else str(error.detail))
            self.record = record
            if self.mode == 'control':
                raise RuntimeError(str(error)) from error
            record['executed'] = 'actor'
            return actor5
        self.record = record
        if self.mode == 'shadow':
            record['executed'] = 'actor'
            return actor5
        record['executed'] = 'c4'
        chosen5 = torch.as_tensor(selected[:5], dtype=actor5.dtype, device=actor5.device)[None]
        self.offset5 = chosen5 - full[:, :5].to(actor5.dtype)
        return chosen5
    builder.wants_command_offsets = True

    def _at_dispatch(self, actor5):
        """The pending block re-derived at the pose measured just before the first write; no C4 plan."""
        record = self.record if isinstance(self.record, dict) else {}
        note = record.setdefault('dispatch', dict(recomputed=0))
        note['recomputed'] += 1
        if self.mode == 'shadow':
            note['rule'] = 'actor_at_measured_pose'
            return actor5
        age = None if self.planned_at_s is None else float(self.clock()) - self.planned_at_s
        if self.offset5 is None or age is None or not 0. <= age <= DISPATCH_MAX_AGE_S:
            raise RuntimeError(f'C4 selection cannot be re-based at dispatch (plan age {age})')
        note.update(rule='actor_at_measured_pose_plus_planned_c4_offset', plan_age_s=age)
        return actor5 + self.offset5.to(actor5.device, actor5.dtype)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def real_planner_factory(checkpoint, library, *, dino_repo, dino_weights, device,
                         checkpoint_sha256=None, library_sha256=None):
    """-> factory(clock=...) -> (C4Planner, identity); loads once at service start, refuses other files."""
    device = torch.device(device)          # the DINO extractor reads device.type

    def factory(*, clock):
        from .v21_c4_planner import C4Planner, GoalLibrary
        from .v21_c4_shadow_worker import camera_check, letterbox, load_model, make_dino
        identity = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                        library=str(library), library_sha256=sha256_file(library),
                        dino_repo=str(dino_repo), dino_weights=str(dino_weights),
                        device=str(device), autocast='float16' if device.type == 'cuda' else None)
        for name, expected in (('checkpoint', checkpoint_sha256), ('library', library_sha256)):
            if expected is not None and identity[f'{name}_sha256'] != expected:
                raise ValueError(f'C4 {name} sha256 {identity[name + "_sha256"]} is not the expected {expected}')
        model, payload = load_model(checkpoint, device)
        identity.update(checkpoint_step=payload.get('step'), checkpoint_complete=payload.get('complete'))
        planner = C4Planner(model, GoalLibrary(library), make_dino(dino_repo, dino_weights, device),
                            letterbox=letterbox, camera_check=camera_check, device=device,
                            autocast_dtype=torch.float16 if device.type == 'cuda' else None, clock=clock)
        started = time.monotonic()
        planner.warm_up()
        identity['warm_up_s'] = time.monotonic() - started
        return planner, identity
    return factory
