"""Experimental, hardware-free Actor proposal / LeWM consequence evaluator.

Geometry is a relative visual-state readout, never a command generator.
Scores are engineering distance proxies, NOT learned task values or probabilities.
This module has no dispatch API and cannot enable hardware.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np


def finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite {shape}')
    return result.copy()


@dataclass(frozen=True)
class VisualGoal:
    target_px: tuple
    observed_midpoint_px: tuple
    capture_q: tuple
    target_time_s: float
    midpoint_time_s: float
    source_time_s: float
    source: str = 'operator_colour_body'
    geometry_validated: bool = False
    contact_verified: bool = False
    body_axis_px: tuple | None = None
    observed_jaw_axis_px: tuple | None = None
    observed_height_m: float | None = None
    target_height_m: float | None = None
    goal_region: str = 'body'
    measured_target_region: str = 'body'

    def validate(self):
        finite_array(self.target_px, (2,), 'target')
        finite_array(self.observed_midpoint_px, (2,), 'observed midpoint')
        finite_array(self.capture_q, (6,), 'capture q')
        times = finite_array([self.target_time_s, self.midpoint_time_s, self.source_time_s], (3,), 'times')
        if abs(times[0]-times[1]) > 1e-6:
            raise ValueError('target and midpoint must share one actual image')
        if times[0] > times[2] + 1e-6 or times[2]-times[0] > .5:
            raise ValueError('goal image is future or older than one five-step horizon')
        if self.source != 'operator_colour_body':
            raise ValueError('pickup goal must use the registered colour body, not DINO station')
        if self.goal_region not in ('body', 'head') or self.measured_target_region != self.goal_region:
            raise ValueError('target region must match requested grasp region; body centre is not head centre')
        for axis in (self.body_axis_px, self.observed_jaw_axis_px):
            if axis is not None and np.linalg.norm(finite_array(axis, (2,), 'axis')) < 1e-8:
                raise ValueError('zero axis is not an orientation measurement')


def actor_prefix_candidates(q, commands):
    q = finite_array(q, (6,), 'q')
    commands = finite_array(commands, (5,6), 'Actor commands')
    candidates = {'actor_5': commands}
    for prefix in (3, 1):
        value = commands.copy()
        value[prefix:] = commands[prefix-1]
        candidates[f'actor_{prefix}_then_hold'] = value
    candidates['hold'] = np.repeat(q[None], 5, axis=0)
    return candidates


def evaluate_candidates(q, actor_commands, goal: VisualGoal, *,
                        predict: Callable, mechanical_check: Callable, project: Callable, pose=None):
    """predict(q, commands) -> (q_path[6,6], sigma[5,6]); project(q)->pixel[2].

    The source, time grid and actual history are closed over by predict's caller.
    No trial state is changed. Geometry errors reject a candidate, not substitute FK
    for a missing observation. A validated proxy is still not physical success.
    """
    q = finite_array(q, (6,), 'q')
    goal.validate()
    target = finite_array(goal.target_px, (2,), 'target')
    observed = finite_array(goal.observed_midpoint_px, (2,), 'observed midpoint')
    origin_projection = finite_array(project(np.asarray(goal.capture_q)), (2,), 'capture projection')
    current = observed + finite_array(project(q.copy()), (2,), 'current projection') - origin_projection
    full_pose = all(x is not None for x in (pose, goal.body_axis_px, goal.observed_jaw_axis_px,
                                          goal.observed_height_m, goal.target_height_m))
    origin_pose = pose(np.asarray(goal.capture_q)) if full_pose else None
    report = dict(executable=False, experimental=True, selected=None, action_dispatched=False,
                  score_kind='relative-visual-distance-plus-uncertainty-proxy',
                  geometry_validated=bool(goal.geometry_validated),
                  evaluation_scope='position-axis-height' if full_pose else 'position-only-diagnostic',
                  score_formula='(CM_px + uncertainty_proxy_px)/224 + axis_error_deg/90 + height_error_m/0.1; unavailable terms omitted only for position diagnostic',
                  current_distance_px=float(np.linalg.norm(current-target)), candidates=[])
    for name, commands in actor_prefix_candidates(q, actor_commands).items():
        row = dict(name=name, commands_position=commands.tolist(), eligible=False, reason=None)
        report['candidates'].append(row)
        try:
            checked = commands.copy()
            mechanical_check(q.copy(), checked)
            if not np.array_equal(checked, commands):
                raise ValueError('mechanical check must not rewrite a proposal')
            closing = (np.diff(np.r_[q[5], commands[:,5]]) < -1e-5).any()
            q_arg, u_arg = q.copy(), commands.copy()
            path, sigma = predict(q_arg, u_arg)
            if not np.array_equal(q_arg, q) or not np.array_equal(u_arg, commands):
                raise ValueError('predictor mutated candidate inputs')
            path = finite_array(path, (6,6), 'LeWM q path')
            sigma = finite_array(sigma, (5,6), 'LeWM uncertainty')
            if not np.allclose(path[0], q, atol=1e-4, rtol=0) or (sigma < 0).any():
                raise ValueError('LeWM source or uncertainty invalid')
            pixels = np.stack([observed + finite_array(project(x.copy()), (2,), 'projection')
                               - origin_projection for x in path])
            # First-order conservative magnitude proxy; no calibration/coverage claim.
            endpoint = path[-1]
            jac = np.zeros((2,6))
            for j in range(6):
                shifted = endpoint.copy(); shifted[j] += .1
                jac[:,j] = (finite_array(project(shifted), (2,), 'Jacobian projection')
                            - finite_array(project(endpoint.copy()), (2,), 'endpoint projection'))/.1
            risk = float(np.linalg.norm(np.abs(jac) @ sigma[-1]))
            distance = float(np.linalg.norm(pixels[-1]-target))
            cost = (distance+risk)/224.
            if full_pose:
                future_pose = pose(endpoint.copy())
                initial_axis = finite_array(origin_pose['jaw_axis_px'], (2,), 'initial geometry axis')
                future_axis = finite_array(future_pose['jaw_axis_px'], (2,), 'future geometry axis')
                if min(np.linalg.norm(initial_axis), np.linalg.norm(future_axis)) < 1e-8:
                    raise ValueError('degenerate projected jaw orientation')
                theta = np.arctan2(future_axis[1], future_axis[0])-np.arctan2(initial_axis[1], initial_axis[0])
                rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                jaw = rot @ np.asarray(goal.observed_jaw_axis_px)
                body = np.asarray(goal.body_axis_px)
                cosine = np.clip(abs(np.dot(jaw,body))/(np.linalg.norm(jaw)*np.linalg.norm(body)),0.,1.)
                axis_error = float(90.-np.degrees(np.arccos(cosine)))
                heights = finite_array([goal.observed_height_m,goal.target_height_m,
                    origin_pose['height_m'],future_pose['height_m']], (4,), 'heights in metres')
                predicted_height = float(heights[0]+heights[3]-heights[2])
                height_error = abs(predicted_height-heights[1])
                cost += axis_error/90. + height_error/.1
                row.update(predicted_axis_error_deg=axis_error,predicted_height_m=predicted_height,
                           height_error_m=height_error, predicted_jaw_axis_px=jaw.tolist())
            row.update(predicted_q_path=path.tolist(), predicted_midpoint_path_px=pixels.tolist(),
                       predicted_distance_px=distance, uncertainty_proxy_px=risk, score=float(cost))
            row['eligible'] = bool(goal.geometry_validated) and (not closing or goal.contact_verified)
            if closing and not goal.contact_verified:
                row['reason'] = 'contact is not verified; reject closure without modifying commands'
            elif not row['eligible']:
                row['reason'] = 'relative projection has not passed real direction validation'
        except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
            row['reason'] = str(exc)
    eligible = [r for r in report['candidates'] if r['eligible']]
    hold = next((r for r in eligible if r['name'] == 'hold'), None)
    # No validated hold reference => no selection, even if a motion looks attractive.
    if hold is not None:
        winner = min(eligible, key=lambda r: (r['score'], r['name'] != 'hold'))
        report['selected'] = winner['name'] if winner['score'] < hold['score'] else 'hold'
    return report
