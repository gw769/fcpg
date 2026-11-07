"""Fake bus and clock only; this helper is not connected to live execution."""
import json
from types import SimpleNamespace

import pytest

from fcpg.v21_j3_dynamic_telemetry import read_j3_dynamic_snapshot


class Bus:
    def __init__(self):
        self.motors = {'elbow_flex': SimpleNamespace(id=3, model='sts3215')}
        self.now = 100.
        self.values = dict(Present_Load=1244, Present_Current=32770,
                           Present_Voltage=74, Status=16)
        self.calls = []
        self.duration = .001
        self.cancelled = False

    def read(self, register, motor, *, normalize, num_retry):
        self.calls.append((register, motor, normalize, num_retry))
        self.now += self.duration
        value = self.values[register]
        if isinstance(value, Exception):
            raise value
        return value

    def write(self, *args, **kwargs):
        pytest.fail('telemetry attempted a motor write')

    sync_write = write
    connect = write

    def sample(self, **kwargs):
        return read_j3_dynamic_snapshot(self, enable=True, clock=lambda: self.now,
                                       cancelled=lambda: self.cancelled, **kwargs)


def test_disabled_invokes_no_bus_clock_or_cancellation_callback():
    class Forbidden:
        def __getattr__(self, key):
            pytest.fail('disabled helper inspected bus: ' + key)
    def forbidden():
        pytest.fail('disabled helper invoked callback')
    result = read_j3_dynamic_snapshot(Forbidden(), clock=forbidden, cancelled=forbidden)
    assert result['callbacks_invoked'] == 0 and result['readings'] == []
    with pytest.raises(TypeError):
        read_j3_dynamic_snapshot(Forbidden(), enable=1)


def test_sequential_actual_read_intervals_and_reused_signed_pwm_current_decode():
    bus = Bus()
    result = bus.sample(phase='after_goal_readback', command_index=4, deadline_s=100.1)
    assert result['bus_reads_attempted'] == result['bus_reads_returned'] == 4
    assert result['all_registers_valid'] and not result['deadline_missed']
    assert result['motor_register_writes'] == 0 and not result['simultaneous_sampling']
    assert not result['motion_verified'] and not result['model_feedback_label']
    assert result['command_index'] == 4 and result['phase'] == 'after_goal_readback'
    events = result['readings']
    assert all(c[1:] == ('elbow_flex', False, 0) for c in bus.calls)
    assert [e['started_s'] for e in events] == pytest.approx([100., 100.001, 100.002, 100.003])
    assert [e['returned_s'] for e in events] == pytest.approx([100.001, 100.002, 100.003, 100.004])
    assert [e['value'] for e in events] == pytest.approx([-22., -13., 7.4, 16])
    assert events[0]['raw_value'] == 1244 and events[0]['decoded']['magnitude'] == 220
    assert events[0]['unit'] == 'percent_pwm_duty'
    assert events[1]['raw_value'] == 32770 and events[1]['decoded']['sign_bit'] == 15
    json.dumps(result, allow_nan=False)


def test_per_register_failure_or_nonfinite_value_is_null_without_hiding_later_status():
    bus = Bus()
    bus.values.update(Present_Current=TimeoutError('no status packet'), Present_Voltage=float('nan'))
    result = bus.sample()
    current, voltage, status = result['readings'][1:]
    assert not current['read_returned'] and current['failed_s'] is not None
    assert current['raw_value'] is current['value'] is None
    assert current['error'] == 'TimeoutError: no status packet'
    assert voltage['read_returned'] and voltage['raw_value'] is voltage['value'] is None
    assert 'ValueError' in voltage['error']
    assert status['raw_value'] == 16 and not result['all_registers_valid']
    assert result['bus_reads_returned'] == 3
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('field,value', [('id', 2), ('model', 'different_servo')])
def test_wrong_motor_mapping_is_rejected_before_bus_read(field, value):
    bus = Bus()
    setattr(bus.motors['elbow_flex'], field, value)
    with pytest.raises(ValueError, match='STS3215 ID3'):
        bus.sample()
    assert not bus.calls


def test_tight_gap_skips_all_reads_without_waiting_or_borrowing_lateness():
    bus = Bus()
    result = bus.sample(deadline_s=100.05)
    assert not bus.calls and bus.now == 100.
    assert all(e['skip_reason'] == 'insufficient_read_budget' for e in result['readings'])
    assert not result['deadline_missed']
    def delayed_cancel_check():
        bus.now += .05
        return False
    result = read_j3_dynamic_snapshot(bus, enable=True, clock=lambda: bus.now,
                                     cancelled=delayed_cancel_check, deadline_s=100.1)
    assert not bus.calls  # Admission must use the clock after the cancellation check.
    assert result['readings'][0]['skip_reason'] == 'insufficient_read_budget'


def test_one_sdk_timeout_does_not_start_another_read_without_remaining_budget():
    bus = Bus()
    bus.duration = .051
    bus.values['Present_Load'] = TimeoutError('no packet')
    result = bus.sample(deadline_s=100.1)
    assert len(bus.calls) == 1 and result['readings'][0]['duration_s'] == pytest.approx(.051)
    assert all(e['skip_reason'] == 'insufficient_read_budget' for e in result['readings'][1:])
    assert not result['deadline_missed'] and not result['read_budget_exceeded']


def test_inflight_overrun_is_recorded_without_retiming_or_more_bus_reads():
    bus = Bus()
    bus.duration = .13
    result = bus.sample(deadline_s=100.1)
    assert len(bus.calls) == 1 and result['finished_s'] == pytest.approx(100.13)
    assert result['deadline_missed'] and result['read_budget_exceeded']
    assert result['readings'][0]['raw_value'] == 1244
    assert all(e['skip_reason'] == 'deadline_missed' for e in result['readings'][1:])


def test_stop_after_first_read_retains_that_read_and_skips_the_rest():
    bus = Bus()
    read = bus.read
    def cancel(*args, **kwargs):
        value = read(*args, **kwargs)
        bus.cancelled = True
        return value
    bus.read = cancel
    result = bus.sample(deadline_s=100.1)
    assert len(bus.calls) == 1 and result['readings'][0]['value'] == -22.
    assert all(e['skip_reason'] == 'cancelled' for e in result['readings'][1:])
