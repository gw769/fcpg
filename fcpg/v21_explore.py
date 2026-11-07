"""Exploration for the LeWM's object-cue head (docs/V21_LEWM_EXPLORE_20261012.md): the robot's own experience with
varied moves, recorded for learning what a move does to where the fish appears in the wrist view.

Each block executes the frozen FCPG + Actor proposal plus a random offset of the five arm joints, ramped in over
the five commands; the gripper stays the Actor's. The offset is drawn from a fixed normal distribution and redrawn
(at most REDRAWS times) until the service's own physical admission accepts the commands; otherwise the proposal runs
unchanged. This is data collection, not a task strategy: the offset never depends on the fish or the task.
Dispatch re-bases like fcpg/v21_c4_service.py: the Actor at the measured pose plus the offset drawn for this block.
"""
import numpy as np
import torch

SOURCE = 'v21_actor_explore'
RECORD_KEY = 'explore'
ARM = 5
STD = (3., 3., 3., 3., 8.)          # units: pan, lift, elbow, wrist flex, wrist roll (8 units of roll is about 13 degrees)
RAMP = (.2, .4, .6, .8, 1.)
REDRAWS = 8
DISPATCH_MAX_AGE_S = 2.0


class ExploreAdapter:
    """Service wiring; the same small interface as fcpg/v21_c4_service.C4ServiceAdapter."""
    record_key = RECORD_KEY
    mode = 'explore'

    def __init__(self, *, seed=0):
        self.source = SOURCE
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.record = None
        self.offset5 = None
        self.planned_at_s = None

    def attach(self, service):
        self.service, self.clock = service, service.clock
        self.identity = dict(planner='exploration', std=list(STD), ramp=list(RAMP), redraws=REDRAWS, seed=self.seed)

    def close(self):
        pass

    def builder(self, state, command_offsets):
        """Runtime command builder: -> [1, 5, 6]. Sets self.record."""
        rt = self.service.owner.runtime
        actor5 = rt.core.act(state, rt.text_ids, command_offsets).commands_position
        if getattr(rt, 'recomputing_at_dispatch', False):
            return self._at_dispatch(actor5)
        self.offset5, self.planned_at_s = None, float(self.clock())
        ramp = torch.as_tensor(RAMP, dtype=actor5.dtype, device=actor5.device)

        def admissible(commands5):
            try:
                rt._physical(state.q, commands5)
                return True
            except Exception:
                return False
        chosen, offset, draws = actor5, np.zeros(ARM), 0
        for draws in range(1, REDRAWS + 1):
            drawn = self.rng.normal(0., STD)
            candidate = actor5.clone()
            candidate[0, :, :ARM] += ramp[:, None] * torch.as_tensor(drawn, dtype=actor5.dtype, device=actor5.device)
            if admissible(candidate):
                chosen, offset = candidate, drawn
                break
        fallback = chosen is actor5
        self.record = dict(mode='explore', action_source=SOURCE, identity=self.identity, available=True,
                           executed='explore', source_time_s=float(state.time[0]), offset=[float(v) for v in offset],
                           draws=draws, fallback_to_actor=fallback)
        arm_only = torch.tensor([1.] * ARM + [0.], dtype=actor5.dtype, device=actor5.device)
        self.offset5 = (chosen - actor5) * arm_only
        return chosen
    builder.wants_command_offsets = True

    def _at_dispatch(self, actor5):
        record = self.record if isinstance(self.record, dict) else {}
        note = record.setdefault('dispatch', dict(recomputed=0))
        note['recomputed'] += 1
        age = None if self.planned_at_s is None else float(self.clock()) - self.planned_at_s
        if self.offset5 is None or age is None or not 0. <= age <= DISPATCH_MAX_AGE_S:
            raise RuntimeError(f'exploration offset cannot be re-based at dispatch (plan age {age})')
        note.update(rule='actor_at_measured_pose_plus_drawn_offset', plan_age_s=age)
        return actor5 + self.offset5.to(actor5.device, actor5.dtype)
