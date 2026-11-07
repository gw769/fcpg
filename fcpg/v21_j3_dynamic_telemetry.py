"""Opt-in sequential J3 telemetry; no device construction or bus ownership.

The caller must use the existing owner's lease, operation admission and bus
lock. Call only after the command and its Goal verification, in the remaining
gap before the next unchanged command deadline. Never run a second bus worker.

``read_budget_s`` reserves time before starting each optional read; it is not a
transport timeout and cannot interrupt an in-flight SDK call. The installed
Feetech single-read timeout is approximately 50 ms, even with zero retries.
The caller must retain the bridge's original 30 ms lateness check and STOP
behavior if a read or scheduling delay overruns. No sleep, write or retry is
performed here; persist these records outside the timed command interval.
"""
from __future__ import annotations

import math
import time

from .servo_register_decode import decode_present_current, decode_present_load


MOTOR = 'elbow_flex'
MOTOR_ID = 3
REGISTER_SPECS = (
    ('Present_Load', 60, 2, '0.1_percent_pwm_duty', 'percent_pwm_duty'),
    ('Present_Current', 69, 2, '6.5_mA', 'mA'),
    ('Present_Voltage', 62, 1, '0.1_V', 'V'),
    ('Status', 65, 1, 'bitmask', 'bitmask'),
)


def _interpret(register, raw):
    if register == 'Present_Load':
        decoded = decode_present_load(raw, source='raw_register')
        return decoded.decoded * .1, decoded.to_dict()
    if register == 'Present_Current':
        decoded = decode_present_current(raw, source='raw_register')
        return decoded.decoded * 6.5, decoded.to_dict()
    return (raw * .1 if register == 'Present_Voltage' else raw), None


def read_j3_dynamic_snapshot(bus=None, *, enable=False, phase='unspecified',
                             command_index=None, deadline_s=None, read_budget_s=.060,
                             clock=time.monotonic, cancelled=lambda: False):
    """Read four optional registers from the supplied existing ID3 bus only.

    ``phase`` and ``command_index`` are caller labels, not evidence of motion.
    Host read intervals are separate for every register, not simultaneous
    sensor exposures. Load measures PWM duty, not joint torque or force.
    A failed or skipped read retains a null value and its own reason.
    """
    if type(enable) is not bool:
        raise TypeError('telemetry enable must be an explicit bool')
    result = dict(schema='v21-j3-dynamic-telemetry-v1', enabled=enable,
                  motor=MOTOR, motor_id=MOTOR_ID, readings=[],
                  bus_reads_attempted=0, bus_reads_returned=0, motor_register_writes=0,
                  simultaneous_sampling=False, motion_verified=False,
                  control_influence=False, model_feedback_label=False)
    if not enable:
        return dict(result, reason='disabled', callbacks_invoked=0)
    if (not isinstance(phase, str) or not phase.strip()
            or (command_index is not None and
                (type(command_index) is not int or command_index < 0))):
        raise ValueError('phase and command index must be explicit caller labels')
    read_budget_s = float(read_budget_s)
    if not math.isfinite(read_budget_s) or read_budget_s <= 0:
        raise ValueError('read budget must be finite and positive')
    if deadline_s is not None:
        deadline_s = float(deadline_s)
        if not math.isfinite(deadline_s):
            raise ValueError('deadline must be a finite host monotonic time')
    motor = bus.motors[MOTOR]
    if motor.id != MOTOR_ID or motor.model != 'sts3215':
        raise ValueError('existing bus must map elbow_flex to STS3215 ID3')
    result.update(phase=phase, command_index=command_index,
                  context_provenance='caller labels only; motion is unassessed',
                  motor_identity_provenance='existing bus mapping, not a new identity read',
                  load_interpretation='drive PWM duty, not measured joint torque or force',
                  timing_semantics='sequential host read intervals, not simultaneous exposure',
                  started_s=float(clock()), deadline_s=deadline_s,
                  read_budget_s=read_budget_s, read_budget_exceeded=False,
                  deadline_missed=False)
    stop_reason = None
    for register, address, size, raw_unit, unit in REGISTER_SPECS:
        event = dict(register=register, address=address, size_bytes=size,
                     raw_unit=raw_unit, unit=unit, normalize=False, num_retry=0,
                     read_attempted=False, read_returned=False, started_s=None,
                     returned_s=None, failed_s=None, finished_s=None, duration_s=None,
                     raw_value=None, value=None, decoded=None, error=None, skip_reason=None)
        result['readings'].append(event)
        if stop_reason is None and cancelled():
            stop_reason = 'cancelled'
        now = float(clock())
        if (stop_reason is None and deadline_s is not None
                and now + read_budget_s > deadline_s):
            stop_reason = 'insufficient_read_budget'
        if stop_reason is not None:
            event.update(skip_reason=stop_reason, skipped_s=now)
            continue
        event.update(read_attempted=True, started_s=now)
        result['bus_reads_attempted'] += 1
        try:
            value = bus.read(register, MOTOR, normalize=False, num_retry=0)
        except Exception as error:
            event.update(error=f'{type(error).__name__}: {error}', failed_s=float(clock()))
            event['finished_s'] = event['failed_s']
        else:
            event.update(read_returned=True, returned_s=float(clock()))
            event['finished_s'] = event['returned_s']
            result['bus_reads_returned'] += 1
            try:
                numeric = float(value)
                if (isinstance(value, bool) or not math.isfinite(numeric)
                        or numeric != int(numeric) or not 0 <= numeric < 2 ** (8 * size)):
                    raise ValueError('readback is not an unsigned integer register value')
                raw = int(numeric)
                event['raw_value'] = raw
                event['value'], event['decoded'] = _interpret(register, raw)
            except (TypeError, ValueError, OverflowError) as error:
                event['error'] = f'{type(error).__name__}: {error}'
        event['duration_s'] = event['finished_s'] - event['started_s']
        if event['duration_s'] > read_budget_s:
            result['read_budget_exceeded'] = True
            stop_reason = 'read_budget_exceeded'
        if deadline_s is not None and event['finished_s'] > deadline_s:
            result['deadline_missed'] = True
            stop_reason = 'deadline_missed'
    result['finished_s'] = float(clock())
    if deadline_s is not None and result['finished_s'] > deadline_s:
        result['deadline_missed'] = True
    result['all_registers_valid'] = all(
        item['read_returned'] and item['error'] is None for item in result['readings'])
    return result
