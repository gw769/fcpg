"""Stage goals for LeWM-scored CEM, NOT an IK/servo/grasp trajectory generator.

Stage transitions use actual observations. evaluate() scores only the future
q path predicted by LeWM. project_commands() sets joint permissions and the
five-step time shape BEFORE prediction; selected actions are never rewritten.
"""
from copy import deepcopy

import numpy as np
import torch


class FishTaskCost:
    strict_model_rank = True
    sensitivity_space = 'cost'
    CLOSE_BLUE_DISTANCE_PX = 16.
    WORKING_CLEARANCE_MM = 3.
    FLOOR_FINISH_MARGIN_MM = 2.
    CLOSE_CENTROID_PX = 16.
    CLOSE_ANGLE_DEG = 20.

    def __init__(self, q, previous, *, visual, wrist, clearance, midpoint,
                 floor_mm, closed, opened, release_mm=None, drop_visual=None):
        self.q = np.asarray(q, np.float64).reshape(6)
        self.visual, self.wrist = visual, wrist
        self.clearance, self.midpoint = clearance, midpoint
        self.floor_mm, self.closed, self.opened = float(floor_mm), float(closed), float(opened)
        # The collision tolerance below the table is NOT the motion target.
        self.working_clearance_mm = max(self.WORKING_CLEARANCE_MM, self.floor_mm + .5)
        self.release_mm, self.drop_visual = release_mm, drop_visual
        # A ~10 cm box is crossed/released above its rim, not at the table.
        self.release_height_mm = max(150., float(release_mm or 0.))
        self.drop_distance_px = None
        self.state = deepcopy(previous or dict(phase='approach'))
        phase = self.state['phase']
        height = float(clearance(self.q))
        mid = midpoint(self.q)
        blue_distance = float((wrist or {}).get('blue_segment_distance_px', np.inf))
        # A wide fish must not turn a 70px M/C miss into "aligned".
        self.precision_radius_px = 48.
        near = wrist is not None and wrist['dist_px'] <= self.precision_radius_px
        at_floor = self.floor_mm <= height <= self.working_clearance_mm + self.FLOOR_FINISH_MARGIN_MM
        low_latched = 'low_z_m' in self.state
        angle = float((wrist or {}).get('angle_deg', np.inf))
        self.close_ready = bool(wrist is not None and at_floor
            and wrist['dist_px'] <= self.CLOSE_CENTROID_PX and abs(angle) <= self.CLOSE_ANGLE_DEG)
        self.close_trigger = dict(ready=self.close_ready,
            rule='current_wrist_centroid_and_angle_within_coarse_tolerance_at_low_height',
            blue_segment_distance_px=blue_distance if np.isfinite(blue_distance) else None,
            current_centroid_distance_px=None if wrist is None else float(wrist['dist_px']),
            maximum_centroid_distance_px=self.CLOSE_CENTROID_PX,
            current_angle_deg=angle if np.isfinite(angle) else None,
            maximum_angle_deg=self.CLOSE_ANGLE_DEG,
            target_clearance_mm=self.working_clearance_mm,
            maximum_clearance_mm=self.working_clearance_mm + self.FLOOR_FINISH_MARGIN_MM,
            exact_centroid_or_angle_required=False)
        if phase in ('approach', 'align', 'lower', 'floor_roll', 'wrist_xy'):
            if (low_latched or at_floor) and wrist is None:
                raise RuntimeError('low XY needs the current Wrist fish; Column cannot replace it')
            phase = ('close' if self.close_ready else 'wrist_xy' if low_latched or at_floor
                     else 'lower' if near and blue_distance <= self.CLOSE_BLUE_DISTANCE_PX
                     else 'approach')
        elif phase == 'close':
            if self.q[5] < 20. and self.state.get('close_goal_issued') is True:
                phase = 'lift'
                self.state['lift_xy'] = mid[:2].tolist()
            elif not self.close_ready:
                raise RuntimeError('current low/near evidence lost before the pinch')
        if phase == 'lift' and height >= 150.:
            phase = 'carry'
        if phase == 'release':
            # Once a real opening transaction committed, finish opening in
            # place. The falling fish/empty jaws change the visual F/G anchor;
            # they must not send the arm back into carry or reclamp the fish.
            phase = 'done' if self.q[5] >= self.opened - 3. else 'release'
        elif phase in ('carry', 'release_height'):
            if drop_visual is None:
                raise RuntimeError('fish placement needs an established Column DINO drop goal and current visual tips')
            if wrist is None or not wrist.get('held_object_visible'):
                raise RuntimeError('fish is not currently observed between the jaws after lifting')
            self.drop_distance_px = float(drop_visual.evaluate(
                np.repeat(self.q[None], 6, axis=0))['distances_px'][0])
            over = self.drop_distance_px <= 8.
            if (phase == 'carry' and height < 150.) or (not over and height < 150.):
                phase = 'lift'
                self.state['lift_xy'] = mid[:2].tolist()
            elif not over:
                phase = 'carry'
            else:
                opened_q = self.q.copy(); opened_q[5] = self.opened
                error = float(clearance(opened_q)) - self.release_height_mm
                phase = 'release' if 0. <= error <= 16. else 'release_height'
        if phase in ('approach', 'lower', 'wrist_xy') and self.q[5] < 20.:
            raise RuntimeError('closed gripper without a committed grasp stage; HOME is required')
        if phase in ('approach', 'wrist_xy') and visual is None:
            raise RuntimeError('no current visual task for LeWM candidate ranking')
        if phase == 'wrist_xy' and not any(getattr(part, 'view', None) == 'wrist'
                for part in getattr(visual, 'parts', (visual,))):
            raise RuntimeError('low XY has no moving-Wrist scene score')
        fine_region = near
        self.phase = phase
        self.state['phase'] = phase
        self.actual_clearance_mm = height
        # These are permissions, not a target pose or a generated IK endpoint.
        locked = dict(approach=(3, 4, 5), wrist_xy=(0, 3, 5), lower=(0, 3, 4, 5),
                      close=(0, 1, 2, 3, 4), lift=(3, 4, 5), carry=(3, 4, 5),
                      release_height=(0, 3, 4, 5), release=(0, 1, 2, 3, 4),
                      done=(0, 1, 2, 3, 4, 5))
        self.fixed = {i: float(self.q[i]) for i in locked[phase]}
        if phase in ('lift', 'carry', 'release_height'):
            self.fixed[5] = self.closed  # Keep the same squeeze goal, not spring-loaded readback.
        self.low_z_m = self.state.get('low_z_m')
        self.z_gradient = None
        if phase == 'wrist_xy':
            self.low_z_m = float(self.state.setdefault('low_z_m', float(mid[2])))
            self.z_gradient = np.zeros(6)
            for j in (1, 2, 4):
                probe = self.q.copy(); probe[j] += .1
                self.z_gradient[j] = (float(midpoint(probe)[2])-float(mid[2]))/.1
            if np.linalg.norm(self.z_gradient[1:3]) < 1e-8:
                raise RuntimeError('low XY has no height-preserving J2/J3 direction')
            # Translate first. Roll only when close AND visibly outside the
            # loose angular tolerance. One realized roll, never +5/-5 hunting.
            if wrist['dist_px'] > 32. or abs(angle) <= self.CLOSE_ANGLE_DEG or self.state.get('low_roll_done'):
                self.fixed[4] = float(self.q[4])
        self.local_delta = (np.asarray([2., 2., 2., 2., 5., 2.], np.float64)
            if phase in ('approach', 'lower') and fine_region else None)
        if phase == 'wrist_xy':
            self.local_delta = np.asarray([2., 4., 4., 2., 5., 2.], np.float64)

    def command_bounds(self, low, high):
        """Local search bounds, BEFORE candidate rollout and scoring."""
        if self.local_delta is not None:
            q, delta = low.new_tensor(self.q), low.new_tensor(self.local_delta)
            low, high = torch.maximum(low, q-delta), torch.minimum(high, q+delta)
        # The shared CEM sampler requires a nonzero interval. Joint locks are
        # enforced by project_commands on EVERY candidate, not by zero bounds.
        return low, high

    def project_commands(self, commands):
        """Project CEM candidates before LeWM; preserve each free endpoint."""
        source = commands.new_tensor(self.q)
        alpha = torch.linspace(.2, 1., 5, device=commands.device, dtype=commands.dtype)
        endpoint = commands[..., -1:, :]
        if self.local_delta is not None:
            delta = commands.new_tensor(self.local_delta)
            endpoint = torch.maximum(source-delta, torch.minimum(endpoint, source+delta))
        out = source + alpha[:, None] * (endpoint - source)
        for joint, value in self.fixed.items():
            out[..., joint] = value
        if self.phase == 'wrist_xy':
            # A candidate constraint BEFORE LeWM, not an execution override.
            # Coupled J2/J3 motion cancels the candidate's vertical component.
            gradient = commands.new_tensor(self.z_gradient)
            dz = ((out-source)*gradient).sum(-1)
            target_dz = alpha * (self.low_z_m-float(self.midpoint(self.q)[2]))
            correction = (target_dz-dz) / gradient[1:3].square().sum()
            out[..., 1:3] += correction[..., None]*gradient[1:3]
        return out

    def require_commands(self, q, commands):
        if not torch.allclose(commands, self.project_commands(commands), atol=1e-5, rtol=0.):
            raise RuntimeError('CEM candidate violates current task joint permissions/time shape')
        if self.phase == 'wrist_xy':
            z = [float(self.midpoint(row)[2]) for row in commands.detach().cpu().numpy().reshape(-1, 6)]
            expected_z = np.linspace(float(self.midpoint(self.q)[2]), self.low_z_m, 6)[1:]
            if np.max(np.abs(np.asarray(z)-expected_z)) > .001:
                raise RuntimeError('low XY candidate does not preserve the held height')
        if self.phase == 'carry':
            # Use the same original physical checker as well; this only adds
            # the task's box-rim requirement, never relaxes table/joint limits.
            points = np.vstack((self.q, commands.detach().cpu().numpy().reshape(-1, 6)))
            for a, b in zip(points[:-1], points[1:]):
                n = max(2, int(np.ceil(np.max(np.abs(b - a)) / .5)) + 1)
                if min(self.clearance(row) for row in a + np.linspace(0., 1., n)[:, None] * (b - a)) < 150.:
                    raise RuntimeError('CEM carry candidate passes below box transit clearance')

    def _visual_cost(self, path):
        parts = getattr(self.visual, 'parts', (self.visual,))
        items = []
        cost = 0.
        for part in parts:
            if part is None:
                continue
            view = getattr(part, 'view', 'column')
            if self.phase == 'wrist_xy' and view != 'wrist':
                continue
            if self.phase != 'wrist_xy' and view != 'column':
                continue
            item = part.evaluate(path)
            dist = np.asarray(item.get('distances_px', ()), np.float64)
            angle = np.asarray(item.get('angles_rad', ()), np.float64)
            if dist.size > 1:
                cost += float(np.mean(dist[1:] / 32.))
            if self.phase == 'wrist_xy' and angle.size > 1 and self.wrist['dist_px'] <= 32.:
                cost += float(np.mean(np.maximum(angle[1:]-np.radians(self.CLOSE_ANGLE_DEG), 0.) / (np.pi/2.)))
            items.append(item)
        if not items:
            raise ValueError('current phase lacks candidate-sensitive visual evidence')
        return cost, items

    def evaluate(self, path):
        """All future quantities below derive from LeWM predicted readbacks."""
        path = np.asarray(path, np.float64)
        if path.shape != (6, 6) or not np.isfinite(path).all():
            raise ValueError('task scoring requires six finite LeWM q states')
        phase, cost, items = self.phase, 0., []
        future = path[1:]
        if phase in ('approach', 'lower', 'wrist_xy'):
            cost, items = self._visual_cost(path)
            if phase == 'wrist_xy':
                z = np.asarray([float(self.midpoint(q)[2]) for q in future])
                cost += float(np.mean(np.abs(z-self.low_z_m)/.003))
            elif phase == 'lower' or (self.wrist is not None and self.wrist['dist_px'] <= self.precision_radius_px):
                goal = self.working_clearance_mm
                cost += float(np.mean([abs(self.clearance(q) - goal) / 20. for q in future]))
        elif phase in ('close', 'release'):
            goal = self.closed if phase == 'close' else self.opened
            cost = float(np.mean(np.abs(future[:, 5] - goal) / 32.))
        elif phase in ('lift', 'carry', 'release_height'):
            if self.drop_visual is not None:
                item = self.drop_visual.evaluate(path)
                items.append(item)
                cost = float(item['cost'])
            elif phase != 'lift':
                raise ValueError('current DINO drop-zone goal is unavailable')
            if phase == 'release_height':
                opened = future.copy(); opened[:, 5] = self.opened
                heights = [self.clearance(q) for q in opened]
                target_height = self.release_height_mm + 8.
            else:
                heights = [self.clearance(q) for q in future]
                target_height = max(160., self.release_height_mm + 8.)
            cost += float(np.mean(np.abs(np.asarray(heights) - target_height) / 20.))
            cost += float(np.mean(np.abs(future[:, 5] - self.closed) / 32.))
            if phase == 'carry' and min(heights) < 150.:
                cost = 1e12
        return dict(cost=float(cost), phase=phase, prediction_source='lewm_predicted_readback',
                    predicted_midpoints_px=[p for item in items for p in item.get('predicted_midpoints_px', [])],
                    predicted_target_px=[p for item in items for p in item.get('predicted_target_px', [])],
                    distances_px=[p for item in items for p in item.get('distances_px', [])],
                    angles_rad=[p for item in items for p in item.get('angles_rad', [])])

    def receipt(self):
        return dict(available=True, source='four_stage_lewm_task', phase=self.phase,
                    next_state=deepcopy(self.state), actual_clearance_mm=self.actual_clearance_mm,
                    precision_align=self.phase == 'wrist_xy',
                    finish_rule='low_wrist_only_xy_hold_z_then_coarse_pinch',
                    working_clearance_mm=self.working_clearance_mm,
                    held_midpoint_z_m=self.low_z_m, column_cost_active=self.phase not in ('wrist_xy', 'close'),
                    low_roll_done=bool(self.state.get('low_roll_done')),
                    precision_radius_px=self.precision_radius_px,
                    close_trigger=deepcopy(self.close_trigger),
                    release_reference_clearance_mm=self.release_mm,
                    release_clearance_mm=self.release_height_mm,
                    drop_distance_px=self.drop_distance_px,
                    drop_goal=None if self.drop_visual is None else self.drop_visual.receipt(),
                    fixed_joints={f'J{i+1}': value for i, value in self.fixed.items()},
                    command_projection='before_lewm_only',
                    trajectory_source=('python_near_blue_close' if self.phase == 'close'
                                       else getattr(self, 'lowering_source', 'python_aligned_vertical_lower') if self.phase == 'lower'
                                       else 'cem_selected_unchanged'),
                    python_lower_rejection=getattr(self, 'python_lower_rejection', None),
                    score_state_source='lewm_predicted_readback', uniform_five_commands=True,
                    local_search_delta=None if self.local_delta is None else self.local_delta.tolist(),
                    views=([self.drop_visual.receipt()]
                           if self.phase in ('lift', 'carry', 'release_height', 'release') and self.drop_visual is not None
                           else [] if self.visual is None else [part.receipt() for part in
                               getattr(self.visual, 'parts', (self.visual,)) if
                               (getattr(part, 'view', 'column') == ('wrist' if self.phase == 'wrist_xy' else 'column'))]),
                    physical_task_success_verified=False)
