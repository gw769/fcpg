"""Bounded, descriptive history from the existing owner's IO wrappers.

This is not a command source, a model input, or a complete bus-write journal.
Missing observations remain missing; no extra reads or waits are introduced.
"""
from copy import deepcopy

import numpy as np


def _position(value):
    try:
        q = np.asarray(value, dtype=np.float64)
        return q.tolist() if q.shape == (6,) and np.isfinite(q).all() else None
    except (TypeError, ValueError):
        return None


class ActuatorContextHistory:
    def __init__(self):
        self.reads = []
        self.last_nonzero_displacement = None
        self.last_operation = None
        self.boundary = None
        self.last_read_error = None

    def invalidate(self, reason, time_s):
        self.reads = []
        self.last_nonzero_displacement = None
        self.last_operation = None
        self.last_read_error = None
        self.boundary = dict(reason=str(reason), time_s=float(time_s))

    def observe_read(self, q, started_s, returned_s):
        values = _position(q)
        if values is None:
            self.observe_read_failure('nonfinite or non-six-vector readback', returned_s)
            return
        row = dict(q_position=values, started_s=float(started_s), returned_s=float(returned_s))
        self.reads = (self.reads + [row])[-2:]
        self.last_read_error = None
        displacement = self._displacement()
        if displacement is not None and any(value != 0. for value in displacement['delta_position']):
            self.last_nonzero_displacement = deepcopy(displacement)

    def observe_read_failure(self, error, time_s):
        # Do not infer a velocity across an unobserved/failed read interval.
        self.reads = []
        self.last_nonzero_displacement = None
        self.last_read_error = dict(error=str(error), time_s=float(time_s))

    def _displacement(self):
        if len(self.reads) != 2:
            return None
        previous, current = self.reads
        dt = current['returned_s'] - previous['returned_s']
        if dt <= 0 or previous['returned_s'] > current['started_s']:
            return None
        delta = np.asarray(current['q_position']) - previous['q_position']
        return dict(previous_read=deepcopy(previous), current_read=deepcopy(current),
                    delta_position=delta.tolist(), elapsed_host_s=dt,
                    average_velocity_position_per_host_s=(delta / dt).tolist(),
                    direction=np.sign(delta).astype(int).tolist(),
                    source='difference_of_two_actual_owner_readbacks',
                    instantaneous_servo_velocity=False)

    def observe_operation(self, kind, started_s, finished_s, *, command=None,
                          previous=None, returned=None, hold=None, error=None):
        row = dict(kind=str(kind), started_s=float(started_s), finished_s=float(finished_s),
                   returned=error is None, error=None if error is None else str(error),
                   complete_bus_write_history=False)
        if kind == 'position_goal_write':
            row.update(requested_position=_position(command), previous_position=_position(previous),
                       returned_position=_position(returned) if error is None else None,
                       actual_motion_verified=False, goal_register_verified=False)
        elif kind == 'hold' and isinstance(hold, dict):
            row.update(reason=hold.get('reason'),
                       motor_register_writes=hold.get('motor_register_writes', hold.get('goal_register_writes')),
                       reused_existing_hold=hold.get('reused_existing_hold'),
                       retained_existing_goal=hold.get('retained_existing_goal'))
            attempts = hold.get('attempts') or []
            if attempts:
                last = attempts[-1]
                row.update(goal_register_verified=last.get('goal_register_verified'),
                           goal_readback_raw=deepcopy(last.get('goal_readback_raw')),
                           held_position=_position(last.get('held_state')))
        self.last_operation = row

    def snapshot(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError('owner context snapshot requires a finite host time')
        def aged(value, key):
            if value is None:
                return None
            value = deepcopy(value)
            stamp = value[key] if key in value else value['current_read']['returned_s']
            # A clock/domain mismatch cannot silently become a causal feature.
            return dict(value, age_s=float(time_s)-stamp) if stamp <= time_s else None
        reads = [deepcopy(row) for row in self.reads if row['returned_s'] <= time_s]
        return dict(schema='v21-owner-actuator-context-v1', observed_at_s=float(time_s),
                    scope='this owner read/write/HOLD wrappers only; direct IO is not a complete bus history',
                    complete_bus_write_history=False, control_influence=False, lewm_input_used=False,
                    position_unit='lerobot_calibrated_position', recent_actual_readbacks=reads,
                    latest_observed_displacement=aged(self._displacement(), 'returned_s'),
                    last_nonzero_observed_displacement=aged(self.last_nonzero_displacement, 'returned_s'),
                    last_observed_operation=aged(self.last_operation, 'finished_s'),
                    last_read_error=aged(self.last_read_error, 'time_s'),
                    history_boundary=aged(self.boundary, 'time_s'))


def history_for(owner):
    history = getattr(owner, '_actuator_context_history', None)
    if history is None:
        history = owner._actuator_context_history = ActuatorContextHistory()
    return history
