"""C4 control planner (docs/V21_LEWM_C4_CONTROL_PLAN_20261011.md section 2).

Every 0.5 s the live service gives the Actor's native 40-step proposal. Candidates are smooth
perturbations of it on J1-J5 (J6 stays the Actor's), inside a widening envelope (step k of 40 at
most ENVELOPE * sqrt(k / 40) calibrated units from the Actor). C4 rolls each one out from the
actual t-1.0 / t-0.5 / t frames and the actually transmitted past 10 commands, and the cost is
the le-wm planning cost: the squared distance between the predicted t+4.0 s embedding and a goal
embedding. The goal is the t+4.0 s frame of the fish-demo window (ep0, ep1, ep3) whose
t-1.0 / t-0.5 / t embeddings are closest to now (scripts/build_v21_c4_goal_library.py).
CEM 32 / 8 / 3; the Actor's own proposal is candidate 0 of every round.

The planner holds no robot object, writes no motor command and keeps no model state between
plans: it returns a selected [40, 6] command sequence and a record, or raises C4PlanError with a
reason (the service then stops; it never falls back to the Actor in control mode).

Inputs it keeps:
- FrameHistory: references to lerobot's latest camera frames with the service clock time they
  were seen (lerobot replaces, never edits, a returned frame).
- CommandTimeline: every command the bridge actually transmitted, with its time. The past 10
  commands are the targets in force at t-1.0, t-0.9 ... t-0.1 s: the last command written at or
  before that time (the arm holds its target while the service plans). The service clears the
  timeline when a block starts from an empty actual history (after HOME, i.e. every trial start);
  a grid time before the first command since then uses the measured q.
"""
import threading
import time
from collections import deque

import numpy as np
import torch

from .v21_lewm_standalone import HISTORY, latent_goal_cost
from .v21_lewm_standalone_data import action_blocks

SCHEMA = 'v21-c4-control-plan-v1'
VIEWS = ('wrist', 'column')
PAST, FUTURE, JOINTS = 10, 40, 6
ARM = 5                          # J1..J5 may be changed; J6 (gripper) is the Actor's
CONTEXT_S = (-1.0, -0.5, 0.0)    # t-1.0, t-0.5, t
GOAL_FRAME = 10                  # library frame index of t+4.0 s
ENVELOPE = 10.0                  # calibrated units at step 40
STEP_STD = 10.0 / np.sqrt(40.)   # the offline noise candidates' per-step std
STEP_CLIP = 2 * STEP_STD
MIN_STD = 0.05
POPULATION, ELITES, ITERATIONS = 32, 8, 3
BUDGET_S = 0.5
FRAME_TOLERANCE_S = 0.1
ADMISSIBLE_TRIES = 10


class C4PlanError(RuntimeError):
    def __init__(self, reason, detail=None):
        super().__init__(f'C4 planner stopped: {reason}' + ('' if detail is None else f' ({detail})'))
        self.reason, self.detail = reason, detail


def envelope(steps=FUTURE):
    k = np.arange(1, steps + 1, dtype=np.float64)
    return ENVELOPE * np.sqrt(k / steps)


def context_distance(query, query_valid, library, library_valid):
    """Mean squared difference over the (frame, camera) pairs valid in both; inf where none.
    query [3, 2, D], library [N, 3, 2, D] (same rule as scripts/build_v21_c4_goal_library.py)."""
    both = query_valid[None] & library_valid
    error = ((library - query[None]) ** 2).mean(-1)
    count = both.sum((1, 2))
    total = np.where(both, error, 0.).sum((1, 2))
    return np.where(count > 0, total / np.maximum(count, 1), np.inf)


class FrameHistory:
    """Camera frames by the time they were first seen; references only."""

    def __init__(self, horizon_s=2.5):
        self.horizon_s = float(horizon_s)
        self.lock = threading.Lock()
        self.entries = {view: deque() for view in VIEWS}
        self.seen = {view: 0 for view in VIEWS}

    def add(self, t, view, frame):
        if frame is None:
            return False
        with self.lock:
            entries = self.entries[view]
            if entries and entries[-1][1] is frame:
                return False
            entries.append((float(t), frame))
            self.seen[view] += 1
            while entries and entries[0][0] < float(t) - self.horizon_s:
                entries.popleft()
        return True

    def nearest(self, view, t):
        with self.lock:
            best = min(self.entries[view], key=lambda e: abs(e[0] - t), default=None)
        if best is None or abs(best[0] - t) > FRAME_TOLERANCE_S:
            return None, None
        return best[1], best[0]


class CommandTimeline:
    """Commands the bridge actually transmitted, with the time each one was written."""

    def __init__(self, horizon_s=120.0):
        self.horizon_s = float(horizon_s)
        self.lock = threading.Lock()
        self.entries = deque()

    def add(self, t, command):
        command = np.asarray(command, np.float32).reshape(-1)
        if command.shape != (JOINTS,) or not np.isfinite(command).all():
            raise ValueError('a transmitted command must be six finite values')
        with self.lock:
            self.entries.append((float(t), command.copy()))
            while self.entries and self.entries[0][0] < float(t) - self.horizon_s:
                self.entries.popleft()

    def clear(self):
        with self.lock:
            self.entries.clear()

    def at(self, times, q_before):
        """[len(times), 6] targets in force at each time, and how many used q_before (no command yet)."""
        q_before = np.asarray(q_before, np.float32).reshape(JOINTS)
        with self.lock:
            entries = list(self.entries)
        out, held = np.empty((len(times), JOINTS), np.float32), 0
        for i, g in enumerate(times):
            last = next((c for t, c in reversed(entries) if t <= float(g) + 1e-9), None)
            if last is None:
                out[i], held = q_before, held + 1
            else:
                out[i] = last
        return out, held

    def past(self, t_now, q_now):
        """[10, 6] targets in force at t-1.0 ... t-0.1 and how many grid times used the measured q."""
        return self.at([float(t_now) - 1.0 + 0.1 * i for i in range(PAST)], q_now)


class GoalLibrary:
    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        emb, valid = np.asarray(data['emb'], np.float32), np.asarray(data['valid'], bool)
        if emb.ndim != 4 or emb.shape[1:3] != (11, len(VIEWS)) or valid.shape != emb.shape[:3]:
            raise ValueError('goal library must hold [N, 11, 2, D] embeddings with [N, 11, 2] validity')
        if 2 in set(np.asarray(data['recordings']).tolist()):
            raise ValueError('the sealed fish episode 2 is in the goal library')
        self.context, self.context_valid = emb[:, :HISTORY], valid[:, :HISTORY]
        self.goal, self.goal_valid = emb[:, GOAL_FRAME], valid[:, GOAL_FRAME]
        self.recordings = np.asarray(data['recordings'], np.int64)
        self.t_index = np.asarray(data['t_index'], np.int64)
        self.progress = np.asarray(data['progress'], np.float64)
        self.path = str(path)

    def retrieve(self, context, context_valid):
        d = context_distance(np.asarray(context, np.float32), np.asarray(context_valid, bool),
                             self.context, self.context_valid)
        i = int(np.argmin(d))
        if not np.isfinite(d[i]):
            raise C4PlanError('no_goal', 'no library window shares a valid camera with now')
        order = np.sort(d)
        return dict(index=i, distance=float(d[i]), second_distance=float(order[1]) if len(order) > 1 else None,
                    recording=int(self.recordings[i]), t_index=int(self.t_index[i]),
                    progress=float(self.progress[i]), goal=self.goal[i], goal_valid=self.goal_valid[i])


class C4Planner:
    def __init__(self, model, library, dino, *, letterbox, camera_check, device, autocast_dtype=None,
                 clock=time.monotonic, seed=21011, budget_s=BUDGET_S):
        if model.training:
            raise ValueError('C4 must be in eval mode')
        self.model, self.library, self.dino = model, library, dino
        self.letterbox, self.camera_check = letterbox, camera_check
        self.device, self.autocast_dtype = torch.device(device), autocast_dtype
        self.clock, self.seed, self.budget_s = clock, int(seed), float(budget_s)
        self.frames, self.commands = FrameHistory(), CommandTimeline()
        self.plans = 0

    def warm_up(self):
        """One throwaway pass through DINO, the encoder and a full candidate rollout, so the first real
        plan does not pay for CUDA start-up (0.46 s against 0.1 s in the rehearsal). Blank images and
        zero commands; no planner state changes."""
        blank = [np.zeros((224, 224, 3), np.uint8)] * (HISTORY * len(VIEWS))
        tokens = torch.as_tensor(self.dino(blank)).to(self.device, torch.float16).reshape(HISTORY, len(VIEWS), 256, 384)
        valid = np.ones((HISTORY, len(VIEWS)), bool)
        with torch.no_grad(), self._autocast():
            emb = self.model.encode(tokens[None], torch.as_tensor(valid)[None].to(self.device))[0].float()
        for _ in range(2):
            self.costs(emb, valid, np.zeros((PAST, JOINTS), np.float32),
                       np.zeros((POPULATION, FUTURE, JOINTS), np.float32), self.library.goal[0], self.library.goal_valid[0])

    # ----------------------------------------------------------------- inputs
    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None)

    def context(self, t_now):
        """-> (embeddings [3, 2, D] float32, validity [3, 2] bool, record)."""
        boxed, where, valid, reasons, offsets = [], [], np.zeros((HISTORY, len(VIEWS)), bool), {}, {}
        for k, dt in enumerate(CONTEXT_S):
            for c, view in enumerate(VIEWS):
                frame, seen = self.frames.nearest(view, t_now + dt)
                if frame is None:
                    reasons[f'{dt:+.1f}/{view}'] = 'no_frame'
                    continue
                image = self.letterbox(np.asarray(frame))
                reason, detail = self.camera_check(np.asarray(frame), image)
                offsets[f'{dt:+.1f}/{view}'] = float(seen - (t_now + dt))
                if reason:
                    reasons[f'{dt:+.1f}/{view}'] = f'{reason}:{detail}'
                    continue
                boxed.append(image)
                where.append((k, c))
                valid[k, c] = True
        if not valid.any(-1).all():
            raise C4PlanError('history_frames', reasons)
        patches = torch.zeros(HISTORY, len(VIEWS), 256, 384, dtype=torch.float16, device=self.device)
        tokens = torch.as_tensor(self.dino(boxed))
        if tokens.shape != (len(boxed), 256, 384):
            raise C4PlanError('dino_shape', tuple(tokens.shape))
        for (k, c), x in zip(where, tokens):
            patches[k, c] = x.to(self.device, torch.float16)
        with torch.no_grad(), self._autocast():
            emb = self.model.encode(patches[None], torch.as_tensor(valid)[None].to(self.device))[0]
        emb = emb.float()
        if not torch.isfinite(emb).all():
            raise C4PlanError('nonfinite_embedding')
        return emb, valid, dict(valid=valid.tolist(), frame_offsets_s=offsets, rejected=reasons)

    def costs(self, history_emb, history_valid, past, candidates, goal, goal_valid):
        """[N, 40, 6] candidates -> [N] le-wm goal costs (float64 numpy)."""
        n = len(candidates)
        commands = torch.cat((torch.as_tensor(past)[None].expand(n, -1, -1), torch.as_tensor(candidates)), 1).float()
        blocks, ok = action_blocks(commands, torch.ones(n, PAST + FUTURE, dtype=torch.bool))
        if not bool(ok.all()):
            raise C4PlanError('action_blocks')
        hv = torch.as_tensor(history_valid).to(self.device)
        with torch.no_grad(), self._autocast():
            act = self.model.action_encoder(blocks.to(self.device))
            predicted = self.model.rollout(history_emb[None].expand(n, -1, -1, -1), hv[None].expand(n, -1, -1), act)
        # The scorer's rule: every history time has a camera (context() refuses otherwise), so the
        # goal cameras are the library window's valid ones.
        goal_ok = torch.as_tensor(np.asarray(goal_valid, bool)).to(self.device)
        if not bool(goal_ok.any()):
            raise C4PlanError('goal_cameras')
        cost = latent_goal_cost(predicted[:, -1].float(), torch.as_tensor(goal).to(self.device).float(), goal_ok)
        return cost.double().cpu().numpy()

    # ------------------------------------------------------------------- plan
    def plan(self, actor40, q_now, *, admissible, lower=None, upper=None):
        """actor40 [40, 6] (calibrated units, 10 Hz), q_now [6] -> (selected [40, 6] float32, record).
        `admissible(commands5 [5, 6]) -> bool` is the service's physical check of the executed part."""
        started = float(self.clock())
        deadline = started + self.budget_s
        self.plans += 1
        actor40 = np.asarray(actor40, np.float32)
        if actor40.shape != (FUTURE, JOINTS) or not np.isfinite(actor40).all():
            raise C4PlanError('actor_proposal', tuple(actor40.shape))
        record = dict(schema=SCHEMA, plan_index=self.plans, planned_at_s=started, seed=self.seed + self.plans,
                      population=POPULATION, elites=ELITES, iterations=ITERATIONS, envelope_units=ENVELOPE)
        emb, valid, record['context'] = self.context(started)
        timing = dict(context=float(self.clock()) - started, rounds=[])     # where a slow plan spent its time
        goal = self.library.retrieve(emb.cpu().numpy(), valid)
        record['retrieval'] = {k: v for k, v in goal.items() if k not in ('goal', 'goal_valid')}
        past, held = self.commands.past(started, q_now)
        record['past_commands'] = dict(held_grid_times=int(held), values=past.tolist())
        lo = np.full(JOINTS, -100., np.float32) if lower is None else np.asarray(lower, np.float32)
        hi = np.full(JOINTS, 100., np.float32) if upper is None else np.asarray(upper, np.float32)
        env = torch.as_tensor(envelope(), dtype=torch.float32)[:, None]
        gen = torch.Generator().manual_seed(record['seed'])
        mean, std = torch.zeros(FUTURE, ARM), torch.full((FUTURE, ARM), STEP_STD)
        pool_cost, pool_cand, rounds = [], [], []
        for it in range(ITERATIONS):
            round_started = float(self.clock())
            inc = (mean + std * torch.randn(POPULATION - 1, FUTURE, ARM, generator=gen)).clamp(-STEP_CLIP, STEP_CLIP)
            inc = torch.cat((torch.zeros(1, FUTURE, ARM), inc))                     # candidate 0 = the Actor
            pert = torch.maximum(torch.minimum(inc.cumsum(1), env), -env)
            cand = torch.as_tensor(actor40)[None].repeat(POPULATION, 1, 1)
            cand[:, :, :ARM] += pert
            cand = torch.maximum(torch.minimum(cand, torch.as_tensor(hi)), torch.as_tensor(lo))
            cand[:, :, ARM:] = torch.as_tensor(actor40)[None, :, ARM:]
            cost = self.costs(emb, valid, past, cand.numpy(), goal['goal'], goal['goal_valid'])
            if not np.isfinite(cost).all():
                raise C4PlanError('nonfinite_cost')
            pool_cost.append(cost)
            pool_cand.append(cand.numpy())
            rounds.append(dict(actor_cost=float(cost[0]), best_cost=float(cost.min()),
                               median_cost=float(np.median(cost))))
            effective = torch.diff(torch.cat((torch.zeros(POPULATION, 1, ARM), cand[:, :, :ARM]
                                              - torch.as_tensor(actor40)[None, :, :ARM]), 1), dim=1)
            elite = torch.as_tensor(np.argsort(cost, kind='stable')[:ELITES])
            mean, std = effective[elite].mean(0), effective[elite].std(0).clamp_min(MIN_STD)
            timing['rounds'].append(float(self.clock()) - round_started)
            if float(self.clock()) > deadline:
                raise C4PlanError('over_budget', f'{float(self.clock()) - started:.3f} s after {it + 1} rounds; '
                                                 f'context {timing["context"]:.3f} s, rounds '
                                                 f'{[round(v, 3) for v in timing["rounds"]]}')
        costs, cands = np.concatenate(pool_cost), np.concatenate(pool_cand)
        order = np.argsort(costs, kind='stable')
        chosen, checks = None, 0
        for i in order[:ADMISSIBLE_TRIES]:
            checks += 1
            if admissible(cands[i, :5]):
                chosen = int(i)
                break
        if chosen is None:
            raise C4PlanError('no_admissible_candidate', f'{checks} cheapest candidates failed the physical check')
        selected = cands[chosen]
        diff = selected - actor40
        elapsed = float(self.clock()) - started
        if elapsed > self.budget_s:
            raise C4PlanError('over_budget', f'{elapsed:.3f} s; context {timing["context"]:.3f} s, rounds '
                                             f'{[round(v, 3) for v in timing["rounds"]]}')
        record.update(rounds=rounds, actor_cost=float(costs[0]), selected_cost=float(costs[chosen]),
                      selected_is_actor=bool(np.array_equal(selected, actor40)), selected_rank_tries=checks,
                      selected_round=int(chosen // POPULATION),
                      executed_deviation_max_abs=np.abs(diff[:5]).max(0).tolist(),
                      deviation_max_abs=np.abs(diff).max(0).tolist(),
                      # signed selection - Actor, per joint: at the last executed command and at 1 / 2 / 4 s
                      executed_deviation_end=diff[4].tolist(),
                      deviation_at_s={'1.0': diff[9].tolist(), '2.0': diff[19].tolist(), '4.0': diff[FUTURE - 1].tolist()},
                      timing_s=timing, elapsed_s=elapsed)
        return selected.astype(np.float32), record
