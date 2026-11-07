"""A session record of the servo bus going silent and coming back, and of every failed service operation.

11/06 10:26:48-10:26:55 (session 20261106-release-ext50-mem30-p16-2): after a HOME chunk fault all six servos answered
no read for about 7 s (LeRobot: "Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 5 tries.
[TxRxResult] There is no status packet!"); in that time the arm moved with no command (the wrist roll 107 degrees, the
elbow 35, the wrist 45) and the next HOME started from there. The only trace was one debug-log line per failed idle
read. BusFaultLog appends to bus-faults.jsonl in the session folder: when the reads stop (with the last pose read and
its age), every failed operation (HOME included), and when the reads come back (how long they were silent, the pose
now and how far each joint moved in between). It reads nothing itself, never commands and never writes a register,
and a failed write of the record never reaches the service.
"""
import json
from pathlib import Path
import time

import numpy as np

NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
FILENAME = 'bus-faults.jsonl'


def _pose(q):
    return [round(float(v), 2) for v in np.asarray(q, np.float64).reshape(-1)]


class BusFaultLog:
    def __init__(self, directory, clock=time.monotonic, wall=time.time):
        self.path = None if directory is None else Path(directory) / FILENAME
        self.clock, self.wall = clock, wall
        self.last_q, self.last_q_s = None, None
        self.silent_since, self.failed_reads = None, 0

    def good(self, q):
        """A read that answered (q [6]): after a silence, how long it lasted and how far each joint moved."""
        now = float(self.clock())
        q = np.asarray(q, np.float64).reshape(-1)
        if self.silent_since is not None:
            moved = (None if self.last_q is None else
                     {name: round(float(b - a), 2) for name, a, b in zip(NAMES, self.last_q, q)})
            self._write(event='bus_back', silent_s=round(now - self.silent_since, 3), failed_reads=self.failed_reads,
                        q=_pose(q), moved_since_last_read=moved,
                        largest_move=None if moved is None else max(abs(v) for v in moved.values()))
            self.silent_since, self.failed_reads = None, 0
        self.last_q, self.last_q_s = q.copy(), now

    def failed(self, error, context, last_read=None):
        """A read the bus did not answer. last_read: the owner's last readback diagnostic (q_position, time_s on this
        clock), used when it is newer than the last read seen here (an operation's own reads, e.g. HOME's)."""
        now = float(self.clock())
        if isinstance(last_read, dict) and last_read.get('q_position') is not None and last_read.get('time_s') is not None:
            if self.last_q_s is None or float(last_read['time_s']) > self.last_q_s:
                self.last_q = np.asarray(last_read['q_position'], np.float64).reshape(-1)
                self.last_q_s = float(last_read['time_s'])
        self.failed_reads += 1
        if self.silent_since is None:
            self.silent_since = now
            self._write(event='bus_silent', context=context, error=f'{type(error).__name__}: {error}'[:400],
                        last_read_q=None if self.last_q is None else _pose(self.last_q),
                        last_read_age_s=None if self.last_q_s is None else round(now - self.last_q_s, 3))

    def operation_failed(self, action, error):
        """A service operation that raised (HOME, execute, ...), with whether the bus was silent at the time."""
        self._write(event='operation_failed', action=str(action), error=f'{type(error).__name__}: {error}'[:400],
                    bus_silent=self.silent_since is not None)

    def _write(self, **record):
        if self.path is None:
            return
        record.update(monotonic_s=round(float(self.clock()), 3), wall_time_s=round(float(self.wall()), 3))
        try:
            with open(self.path, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
        except OSError:
            pass
