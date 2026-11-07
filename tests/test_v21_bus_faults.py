"""fcpg/v21_bus_faults.BusFaultLog and the live service's idle readback (11/06: the servo bus silent for 7 s after a
HOME fault while the arm moved with no command): the record of the silence, of its end with how far each joint moved,
and of failed operations; the service's own behaviour is unchanged."""
import json

import numpy as np

from fcpg.v21_bus_faults import FILENAME, BusFaultLog
from fcpg.v21_live_service import V21LiveService
from test_v21_live_service import service  # noqa: F401

MONITOR_LOOP = V21LiveService._monitor_q          # before the fixture replaces it
SILENT = ConnectionError("Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 5 tries. "
                         "[TxRxResult] There is no status packet!")


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_silence_is_recorded_once_with_the_last_pose_and_its_end_with_the_motion(tmp_path):
    clock = Clock()
    log = BusFaultLog(tmp_path, clock=clock, wall=lambda: 1.)
    held = np.array([-24.4, -66.7, 65.0, 60.6, 46.5, 31.1])
    log.good(held)
    clock.now = 100.5
    for _ in range(7):
        log.failed(SILENT, 'idle readback')
        clock.now += 1.
    sagged = np.array([-5.1, -68.1, 99.6, 11.5, -62.0, 31.3])
    log.good(sagged)
    log.good(sagged)                                   # an ordinary read after that writes nothing
    silent, back = rows(tmp_path / FILENAME)
    assert silent['event'] == 'bus_silent' and silent['last_read_q'] == held.round(2).tolist()
    assert silent['last_read_age_s'] == .5 and 'no status packet' in silent['error'] and silent['context'] == 'idle readback'
    assert back['event'] == 'bus_back' and back['failed_reads'] == 7 and back['silent_s'] == 7.
    assert back['moved_since_last_read']['wrist_roll'] == -108.5 and back['largest_move'] == 108.5
    assert back['q'] == sagged.round(2).tolist()


def test_an_operations_own_newer_read_is_the_last_pose(tmp_path):
    clock = Clock()
    log = BusFaultLog(tmp_path, clock=clock)
    log.good(np.zeros(6))                              # the idle monitor at 100
    clock.now = 110.
    log.failed(SILENT, 'idle readback', dict(q_position=[1., 2., 3., 4., 46.6, 6.], time_s=109.5))
    silent, = rows(tmp_path / FILENAME)
    assert silent['last_read_q'] == [1., 2., 3., 4., 46.6, 6.] and silent['last_read_age_s'] == .5


def test_failed_operations_and_no_folder(tmp_path):
    log = BusFaultLog(tmp_path)
    log.operation_failed('home', RuntimeError('settled endpoint lies outside the safety envelope'))
    failed, = rows(tmp_path / FILENAME)
    assert failed['event'] == 'operation_failed' and failed['action'] == 'home' and failed['bus_silent'] is False
    BusFaultLog(None).failed(SILENT, 'idle readback')                         # no folder: nothing written
    BusFaultLog(tmp_path / 'missing' / 'dir').operation_failed('home', SILENT)  # an unwritable record never raises


def test_the_idle_loop_records_the_silence_and_still_stops_as_before(service):
    service._monitor_readback_once()                   # a good idle read first (the fake arm at zero)
    reads = [SILENT] * 4 + [np.full(6, 2., np.float32)]

    class Five:
        count = 0

        def wait(self, _):
            self.count += 1
            return self.count > 5

    def read():
        value = reads.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    old, service.closing = service.closing, Five()
    service.owner.read_checked = read
    try:
        MONITOR_LOOP(service)
    finally:
        service.closing = old
    assert service.owner.stop_event.is_set() and service.owner.holds == 1      # the idle guard tripped as before
    assert service.current_q == [2.] * 6 and not service.operation_lock.locked()
    events = rows(service.output / FILENAME)
    assert [e['event'] for e in events] == ['bus_silent', 'bus_back']
    assert events[0]['last_read_q'] == [0.] * 6
    assert events[1]['failed_reads'] == 4 and events[1]['moved_since_last_read']['wrist_roll'] == 2.
