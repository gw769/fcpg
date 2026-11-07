"""SO-101 connection/hold semantics only; no learned weights or motor tuning."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ORDER=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
LOWER=np.array([-100.]*5+[0.]);UPPER=np.full(6,100.)

_PENDING_HOME_SCHEMA = 'v21-home-physical-settled-receipt-v1'


def read_map(bus,register,*,normalize=False):
    values=bus.sync_read(register,normalize=normalize,num_retry=4)
    if not isinstance(values,dict) or set(values)!=set(ORDER) or not np.isfinite([values[k] for k in ORDER]).all():
        raise ValueError('invalid six-motor register read: '+register)
    if not normalize and any(isinstance(values[k],(bool,np.bool_)) or values[k]!=int(values[k]) for k in ORDER):
        raise ValueError('invalid raw integer register read: '+register)
    return {k:float(values[k]) if normalize else int(values[k]) for k in ORDER}


def verify_policy_goal_position(bus, goal, *, cancelled=lambda:False, clock=time.monotonic):
    """One raw Goal read, after a policy write, using its exact bus conversion.

    This never writes a register or waits for motion. A missing status packet
    is read again, up to five times. Any other read failure is not retried.
    Goal is a command register, not Present_Position and never a model feedback label.
    Failure metadata stays on the original exception for the transaction log.
    """
    record=dict(schema='v21-policy-goal-readback-v1',register='Goal_Position',normalize=False,
        started_s=float(clock()),position_unit='lerobot_calibrated_position',read_attempted=False,read_returned=False,
        goal_register_verified=False,motor_register_writes=0,num_retry=0,
        verification_scope='exact Goal_Position register cells after this policy write; not actual motion',
        individual_write_packet_acknowledged=False,
        repeated_identical_write_delivery_distinguishable=False,
        normalised_goal_source='derived from this same raw read, not a second register transaction',
        model_feedback_label=False)
    try:
        if cancelled():raise RuntimeError('policy goal verification cancelled before read')
        command=np.asarray(goal,dtype=np.float32)
        if command.shape!=(6,) or not np.isfinite(command).all():
            raise ValueError('policy goal verification requires a finite six-vector')
        if (command<LOWER).any() or (command>UPPER).any():
            raise ValueError('policy goal verification target outside joint bounds')
        ids={name:bus.motors[name].id for name in ORDER}
        requested={name:float(command[index]) for index,name in enumerate(ORDER)}
        # _write_position_command passes these same Python float values to
        # normalized sync_write, whose conversion is precisely _unnormalize.
        converted=bus._unnormalize({ids[name]:requested[name] for name in ORDER})
        expected={name:int(converted[ids[name]]) for name in ORDER}
        record.update(requested_goal_position=requested,expected_goal_raw=expected)
        if cancelled():raise RuntimeError('policy goal verification cancelled before read')
        record.update(read_started_s=float(clock()),read_attempted=True)
        raw=None
        packet_tries=0
        while True:
            if cancelled():raise RuntimeError('policy goal verification cancelled before read')
            try:
                raw=bus.sync_read('Goal_Position',normalize=False,num_retry=0)
                break
            except BaseException as error:
                packet_tries+=1
                # #region agent log
                try:
                    import json as _json, time as _time
                    with open('/home/james/feedback-corrected-grounding/.cursor/debug-55ff20.log','a') as _f:
                        _f.write(_json.dumps({'sessionId':'55ff20','runId':'post-fix','hypothesisId':'B','location':'v21_hardware_state.py:verify_policy_goal_position','message':'goal status read failed','data':{'try':packet_tries,'error':f'{type(error).__name__}: {error}'[:180]},'timestamp':int(_time.time()*1000)})+'\n')
                except Exception:
                    pass
                # #endregion
                if 'no status packet' not in str(error).lower() or packet_tries>=5:
                    raise
        record.update(read_returned_s=float(clock()),read_returned=True,status_packet_retries=packet_tries)
        if (not isinstance(raw,dict) or set(raw)!=set(ORDER) or not np.isfinite([raw[name] for name in ORDER]).all()
                or any(float(raw[name])!=int(raw[name]) for name in ORDER)):
            raise ValueError('policy Goal_Position readback is not six finite raw integer cells')
        actual={name:int(raw[name]) for name in ORDER}
        mismatched={name:dict(expected_raw=expected[name],observed_raw=actual[name])
                    for name in ORDER if actual[name]!=expected[name]}
        record.update(goal_raw=actual,mismatched_joints=mismatched,
                      goal_register_verified=not bool(mismatched))
        # STOP wins before any remaining diagnostics; the raw result above is
        # retained even if STOP arrived during the single in-flight bus read.
        if cancelled():raise RuntimeError('policy goal verification cancelled after read')
        normalized=bus._normalize({ids[name]:actual[name] for name in ORDER})
        record['goal_position_from_raw']={name:float(normalized[ids[name]]) for name in ORDER}
        if not np.isfinite(list(record['goal_position_from_raw'].values())).all():
            raise ValueError('policy Goal_Position normalization is nonfinite')
        if mismatched:
            raise RuntimeError('policy Goal_Position raw readback mismatch: '+', '.join(mismatched))
        if cancelled():raise RuntimeError('policy goal verification cancelled after read')
        record['returned_s']=float(clock())
        return record
    except BaseException as error:
        record.update(failed_s=float(clock()),error=f'{type(error).__name__}: {error}')
        error.goal_readback_verification=deepcopy(record)
        raise


def actuator_snapshot(bus,*,cancelled=lambda:False,read_events=None):
    """One sync_read call per register; decode positions from those raw cells.

    Present and Goal still have separate host read intervals. Neither the six
    motors nor different registers are claimed to have simultaneous exposure.
    Each interval includes any retries performed by the existing read helper.
    """
    position_intervals={}
    def read(register,*,normalize=False):
        if cancelled():raise RuntimeError('normal completion cancelled')
        event=dict(register=register,normalize=normalize,started_s=time.monotonic())
        if read_events is not None:read_events.append(event)
        try:
            values=read_map(bus,register,normalize=normalize)
            event.update(returned_s=time.monotonic(),values=values)
            if register in ('Present_Position','Goal_Position'):
                position_intervals[register]={k:event[k] for k in ('started_s','returned_s')}
        except BaseException as exc:
            event.update(failed_s=time.monotonic(),error=f'{type(exc).__name__}: {exc}')
            raise
        if cancelled():raise RuntimeError('normal completion cancelled')
        return values
    def normalize_position(raw):
        if cancelled():raise RuntimeError('normal completion cancelled')
        ids={name:bus.motors[name].id for name in ORDER}
        if len(set(ids.values()))!=len(ORDER):
            raise ValueError('position normalization requires six distinct motor IDs')
        normalized=bus._normalize({ids[name]:raw[name] for name in ORDER})
        if (not isinstance(normalized,dict) or set(normalized)!=set(ids.values())
                or any(isinstance(value,(bool,np.bool_)) for value in normalized.values())
                or not np.isfinite(list(normalized.values())).all()):
            raise ValueError('invalid six-motor position normalization')
        if cancelled():raise RuntimeError('normal completion cancelled')
        return {name:float(normalized[ids[name]]) for name in ORDER}
    started=time.monotonic()
    raw_present=read('Present_Position')
    raw_goal=read('Goal_Position')
    present=normalize_position(raw_present)
    goal=normalize_position(raw_goal)
    torque=read('Torque_Enable');moving=read('Moving')
    return dict(started_s=started,finished_s=time.monotonic(),position_unit='lerobot_calibrated_position',
        normalised_position_source='same_raw_register_read',position_read_intervals_s=position_intervals,
        timing_semantics='separate host register read boundaries; not simultaneous sensor exposure',
        present_raw=raw_present,goal_raw=raw_goal,present_position=present,goal_position=goal,
        present_minus_goal={k:present[k]-goal[k] for k in ORDER},
        torque=torque,moving=moving,motor_register_writes=0)


def actuator_diagnostics(bus):
    """Optional raw register reads for the stopped operator diagnostic only."""
    registers = ('ID', 'Model_Number', 'Firmware_Major_Version', 'Firmware_Minor_Version',
                 'P_Coefficient', 'I_Coefficient', 'D_Coefficient', 'Lock', 'Operating_Mode', 'Phase',
                 'CW_Dead_Zone', 'CCW_Dead_Zone', 'Minimum_Startup_Force',
                 'Min_Position_Limit', 'Max_Position_Limit', 'Max_Torque_Limit',
                 'Torque_Limit', 'Protection_Current', 'Overload_Torque', 'Protective_Torque',
                 'Protection_Time', 'Over_Current_Protection_Time',
                 'Unloading_Condition', 'LED_Alarm_Condition',
                 'Goal_Time', 'Goal_Velocity', 'Acceleration',
                 'Maximum_Acceleration', 'Maximum_Velocity_Limit',
                 'Homing_Offset', 'Present_Velocity', 'Present_Voltage',
                 'Present_Temperature', 'Status')
    result = dict(started_s=time.monotonic(), registers={}, errors={}, motor_register_writes=0)
    for register in registers:
        try:
            values = bus.sync_read(register, normalize=False, num_retry=0)
            if set(values) != set(ORDER) or not np.isfinite([values[k] for k in ORDER]).all():
                raise ValueError('invalid six-motor diagnostic register')
            result['registers'][register] = {k:int(values[k]) for k in ORDER}
        except Exception as exc:
            result['registers'][register] = None
            result['errors'][register] = f'{type(exc).__name__}: {exc}'
    result['finished_s'] = time.monotonic()
    return result


def _pre_dispatch_register_read(bus, register, *, cancelled, clock):
    """Read one optional register and retain the actual SDK call boundaries.

    ``read_map`` intentionally uses the normal four-retry state-read policy.
    These optional context reads are diagnostic and use one SDK call with no
    retry. An in-flight SDK call still has its own transport timeout. STOP is
    checked before admission; the caller records this result then checks STOP
    again before another read. ``normalize=False`` does not disable LeRobot's
    register sign decode (notably Present_Velocity).
    """
    event = dict(register=register, normalize=False, num_retry=0,
                 read_attempted=False, read_returned=False,
                 read_started_s=None, read_finished_s=None,
                 missing_reason=None, error=None)
    if cancelled():
        raise RuntimeError('STOP during pre-dispatch observation')
    event['read_started_s'] = float(clock())
    event['read_attempted'] = True
    try:
        values = bus.sync_read(register, normalize=False, num_retry=0)
    except Exception as error:
        event['read_finished_s'] = float(clock())
        event['missing_reason'] = 'read_failed'
        event['error'] = f'{type(error).__name__}: {error}'
        return None, event
    event['read_finished_s'] = float(clock())
    event['read_returned'] = True
    try:
        if (not isinstance(values, dict) or set(values) != set(ORDER)
                or not np.isfinite([values[k] for k in ORDER]).all()
                or any(isinstance(values[k], (bool, np.bool_))
                       or values[k] != int(values[k]) for k in ORDER)):
            raise ValueError('invalid six-motor integer register read: ' + register)
        parsed = {k: int(values[k]) for k in ORDER}
    except Exception as error:
        event['missing_reason'] = 'invalid_readback'
        event['error'] = f'{type(error).__name__}: {error}'
        return None, event
    return parsed, event


def _local_bus_calibration(bus):
    """Describe the loaded bus calibration without inventing an identity.

    This is loaded host configuration, not a fresh hardware ID or EEPROM read.
    Keep motor mapping IDs and calibration IDs distinct, and never infer
    missing calibration fields from a motor name or another object.
    """
    motors = getattr(bus, 'motors', {})
    calibration = getattr(bus, 'calibration', {})
    result = {}
    for name in ORDER:
        motor = motors.get(name) if isinstance(motors, dict) else None
        cal = calibration.get(name) if isinstance(calibration, dict) else None
        errors = {}

        def field(source, key, label=None, *, integer=True):
            label = label or key
            value = getattr(source, key, None) if source is not None else None
            if value is None:
                errors[label] = 'absent_from_loaded_bus_configuration'
                return None
            if not integer:
                return str(getattr(value, 'value', value))
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                errors[label] = 'loaded_configuration_value_is_not_an_integer'
                return None
            return int(value)

        motor_id = field(motor, 'id', 'motor_id')
        calibration_id = field(cal, 'id', 'calibration_id')
        norm_mode = field(motor, 'norm_mode', integer=False)
        description = {
            'range_m100_100': 'calibrated range [-100, 100], clamped to loaded encoder bounds',
            'range_0_100': 'calibrated range [0, 100], clamped to loaded encoder bounds',
            'degrees': 'degrees about the loaded range midpoint using motor model resolution',
        }.get(norm_mode)
        result[name] = dict(
            motor_id=motor_id, model=field(motor, 'model', integer=False),
            calibration_id=calibration_id,
            calibration_id_matches_motor_id=(None if motor_id is None or calibration_id is None
                                             else motor_id == calibration_id),
            range_min=field(cal, 'range_min'), range_max=field(cal, 'range_max'),
            homing_offset=field(cal, 'homing_offset'), drive_mode=field(cal, 'drive_mode'),
            norm_mode=norm_mode, norm_mode_description=description,
            field_errors=errors,
        )
    apply_drive_mode = getattr(bus, 'apply_drive_mode', None)
    return dict(source='bus.motors + bus.calibration', motors=result,
                apply_drive_mode=apply_drive_mode if type(apply_drive_mode) is bool else None,
                apply_drive_mode_missing_reason=(None if type(apply_drive_mode) is bool
                                                 else 'absent_or_nonboolean_bus_setting'),
                hardware_calibration_read=False,
                complete=all(not row['field_errors'] for row in result.values())
                         and type(apply_drive_mode) is bool)


def pre_dispatch_actuator_state(bus, *, cancelled=lambda: False, observation_phase='pre_dispatch'):
    """Causal pre-block registers, sampled before the five-command clock.

    Raw current/load is descriptive only. Zero readings are explicitly not a
    physical witness; no register is written and no model input is fabricated.
    Optional diagnostics retain a real host read interval and a null value on
    unsupported registers. Configuration is limited to the installed
    P/I/D/Operating_Mode values and is never changed here.
    """
    if observation_phase not in ('pre_dispatch', 'stopped_observation'):
        raise ValueError('unsupported actuator observation phase')
    from .servo_register_decode import encoding_provenance
    state = actuator_snapshot(bus, cancelled=cancelled)
    registers = {}
    errors = {}
    read_events = {}
    optional_registers = ('Present_Velocity', 'Present_Load', 'Present_Current')
    configuration_registers = ('P_Coefficient', 'I_Coefficient', 'D_Coefficient',
                               'Operating_Mode')
    state.update(schema='v21-pre-dispatch-actuator-state-v1',
        observation_phase=observation_phase, registers=registers,
        optional_register_errors=errors, register_read_events=read_events,
        local_calibration=_local_bus_calibration(bus),
        register_encoding=encoding_provenance(),
        configuration_register_unit='P/I/D: register coefficients; Operating_Mode: register enum',
        control_influence=False,lewm_input_used=False,
        timing_semantics='host read boundaries; not simultaneous per-motor sensor exposure',
        provenance=('existing_single_owner_before_first_policy_write' if observation_phase == 'pre_dispatch'
                    else 'existing_single_owner_stopped_observation'))
    try:
        for register in (*optional_registers, *configuration_registers):
            values, event = _pre_dispatch_register_read(
                bus, register, cancelled=cancelled, clock=time.monotonic)
            registers[register] = values
            read_events[register] = event
            if values is None:
                errors[register] = event['error']
            if cancelled():
                raise RuntimeError('STOP during pre-dispatch observation')
    except BaseException as error:
        state.update(finished_s=time.monotonic(), error=f'{type(error).__name__}: {error}')
        error.pre_dispatch_actuator_state = deepcopy(state)
        raise
    load = registers.get('Present_Load'); current = registers.get('Present_Current')
    state['load_current_nonzero_observed'] = bool(
        (load and any(load.values())) or (current and any(current.values())))
    state['load_current_nonzero_semantics'] = 'nonzero stored bus values; not a physical motion/contact witness'
    state['finished_s'] = time.monotonic()
    return state


def raw_stop(bus):
    """Cancel an unverified warm target without momentarily unloading the arm."""
    present=read_map(bus,'Present_Position')
    bus.sync_write('Goal_Position',present,normalize=False,num_retry=4)
    if read_map(bus,'Goal_Position')!=present:
        raise RuntimeError('warm-connection rejection HOLD could not be verified')


def _pending_raw_map(value, label, *, allow_zero=True):
    if (not isinstance(value, dict) or set(value) != set(ORDER)
            or any(isinstance(value[name], (bool, np.bool_)) for name in ORDER)):
        raise ValueError(label+' is not a six-motor raw map')
    try:
        result = {name: int(value[name]) for name in ORDER}
        numeric = {name: float(value[name]) for name in ORDER}
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(label+' is not a six-motor raw map') from error
    if any(not np.isfinite(numeric[name]) or numeric[name] != result[name] for name in ORDER):
        raise ValueError(label+' contains a non-integer raw cell')
    if not allow_zero and any(value == 0 for value in result.values()):
        raise ValueError(label+' contains a raw-zero Goal cell')
    return result


def _pending_position(value, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError(label+' is not a finite six-vector')
    if (array < LOWER).any() or (array > UPPER).any():
        raise ValueError(label+' is outside calibrated joint bounds')
    return array


def _pending_position_map(value, label):
    if not isinstance(value, dict) or set(value) != set(ORDER):
        raise ValueError(label+' is not a six-motor position map')
    return _pending_position([value[name] for name in ORDER], label)


def _home_journal_path(directory, value):
    """Resolve a pending HOME journal without following an untrusted link."""
    if not isinstance(value, str) or not value:
        raise ValueError('pending HOME receipt has no command journal path')
    root = (Path(directory) / 'command_journal').resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ValueError('pending HOME command journal path must be absolute')
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError('pending HOME command journal must be a regular file')
    resolved = candidate.resolve()
    try:
        inside = os.path.commonpath((str(root), str(resolved))) == str(root)
    except ValueError:
        inside = False
    if not inside:
        raise ValueError('pending HOME command journal escapes output directory')
    return resolved


def _finite_home_command(value, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError(label+' is not a finite six-vector')
    if (array < LOWER).any() or (array > UPPER).any():
        raise ValueError(label+' is outside calibrated joint bounds')
    return array


def _validate_home_command_journal(directory, value, *, expected_commands=None,
                                   settled_state=None, expected_sha256=None):
    """Validate the complete five-command HOME journal and return its summary."""
    path = _home_journal_path(directory, value)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('pending HOME command journal bytes changed')
    rows = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except Exception as error:
            raise ValueError('pending HOME command journal is unreadable') from error
        if not isinstance(row, dict):
            raise ValueError('pending HOME command journal contains a non-object event')
        rows.append(row)
    if len(rows) != 12:
        raise ValueError('pending HOME command journal does not contain exactly five commands')
    if rows[0].get('event') != 'chunk_intent' or rows[0].get('requested_commands') != 5:
        raise ValueError('pending HOME command journal has an invalid chunk intent')
    requested = np.asarray(rows[0].get('requested'), dtype=np.float64)
    if requested.shape != (5, 6) or not np.isfinite(requested).all():
        raise ValueError('pending HOME command journal has invalid requested commands')
    if (requested < LOWER).any() or (requested > UPPER).any():
        raise ValueError('pending HOME command journal command is outside joint bounds')
    if expected_commands is not None:
        expected = np.asarray(expected_commands, dtype=np.float64)
        if expected.shape != (5, 6) or not np.allclose(expected, requested, atol=1e-4, rtol=0):
            raise ValueError('pending HOME journal differs from the completed chunk')
    times = []
    for index in range(5):
        intent = rows[1 + index * 2]
        result = rows[2 + index * 2]
        if intent.get('event') != 'home_command_intent' or intent.get('command_index') != index:
            raise ValueError('pending HOME command journal intent sequence is invalid')
        if result.get('event') != 'home_command_result' or result.get('command_index') != index:
            raise ValueError('pending HOME command journal result sequence is invalid')
        requested_row = _finite_home_command(intent.get('requested'), 'pending HOME requested command')
        sent = _finite_home_command(result.get('sent'), 'pending HOME sent command')
        previous = _finite_home_command(intent.get('previous_command'), 'pending HOME previous command')
        observed = _finite_home_command(result.get('observed_state'), 'pending HOME observed state')
        if not np.allclose(requested_row, requested[index], atol=1e-4, rtol=0):
            raise ValueError('pending HOME journal intent does not match chunk intent')
        if not np.allclose(sent, requested_row, atol=1e-4, rtol=0):
            raise ValueError('pending HOME journal sent command changed')
        for row in (intent, result):
            stamp = row.get('monotonic_s')
            if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not np.isfinite(stamp):
                raise ValueError('pending HOME journal event has an invalid clock')
            times.append(float(stamp))
        if index and not np.allclose(previous, requested[index - 1], atol=1e-4, rtol=0):
            raise ValueError('pending HOME journal previous command changed')
        if index == 0 and not np.isfinite(previous).all():
            raise ValueError('pending HOME journal first previous command is invalid')
        if not np.isfinite(observed).all():
            raise ValueError('pending HOME journal observed state is invalid')
    completion = rows[-1]
    if completion.get('event') != 'home_chunk_completed' or completion.get('endpoint_outcome') != 'settled':
        raise ValueError('pending HOME journal lacks a settled completion event')
    completed = _finite_home_command(completion.get('settled_state'), 'pending HOME settled journal state')
    stamp = completion.get('monotonic_s')
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not np.isfinite(stamp):
        raise ValueError('pending HOME completion event has an invalid clock')
    if times and float(stamp) < max(times):
        raise ValueError('pending HOME journal completion precedes its commands')
    if settled_state is not None:
        expected_settled = _finite_home_command(settled_state, 'pending HOME settled state')
        if not np.allclose(expected_settled, completed, atol=1e-4, rtol=0):
            raise ValueError('pending HOME journal completion differs from receipt')
    return dict(path=str(path), sha256=digest, event_count=len(rows),
                requested_commands=requested.tolist(), settled_state=completed.tolist())


def _load_pending_home_receipt(directory, source):
    """Validate the newest physical HOME endpoint bound to ``last-attempt``."""
    path = Path(directory) / 'pending-home-receipt.json'
    if path.is_symlink():
        raise ValueError('pending HOME receipt must be a regular file')
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError('pending HOME receipt must be a regular file')
    try:
        raw = path.read_bytes()
        receipt = json.loads(raw)
    except Exception as error:
        raise ValueError('pending HOME receipt is unreadable') from error
    if not isinstance(receipt, dict) or receipt.get('schema') != _PENDING_HOME_SCHEMA:
        raise ValueError('unrecognized pending HOME receipt')
    if (receipt.get('kind') != 'pending-home-physical-settled'
            or receipt.get('requires_home') is not True
            or receipt.get('completed') is not True
            or receipt.get('physical_settled') is not True
            or receipt.get('model_transition') is not False
            or receipt.get('raw_readback_source') != 'same_raw_register_read'
            or receipt.get('raw_freeze_verified') is not False
            or receipt.get('endpoint_outcome') != 'settled'):
        raise ValueError('pending HOME receipt lacks physical-settled evidence')
    source_path = Path(directory) / 'last-attempt.json'
    if not source_path.exists():
        raise ValueError('pending HOME receipt has no source execution receipt')
    source_raw = source_path.read_bytes()
    try:
        source_time = source.get('source_time_s')
    except AttributeError as error:
        raise ValueError('pending HOME source execution receipt is invalid') from error
    if receipt.get('source_time_s') != source_time:
        raise ValueError('pending HOME receipt source execution changed')
    expected_hash = hashlib.sha256(source_raw).hexdigest()
    if receipt.get('source_sha256') != expected_hash:
        raise ValueError('pending HOME receipt source bytes changed')
    goal_raw = _pending_raw_map(receipt.get('goal_raw'), 'pending HOME Goal_Position', allow_zero=False)
    present_raw = _pending_raw_map(receipt.get('present_raw'), 'pending HOME Present_Position')
    goal_position = _pending_position(receipt.get('goal_position'), 'pending HOME Goal position')
    present_position = _pending_position(receipt.get('present_position'), 'pending HOME Present position')
    settled = _pending_position(receipt.get('settled_q_position'), 'pending HOME settled position')
    journal_settled = _pending_position(
        receipt.get('journal_settled_q_position'), 'pending HOME journal settled position')
    if not np.allclose(settled, present_position, atol=1e-5, rtol=0):
        raise ValueError('pending HOME settled position differs from Present readback')
    returned = receipt.get('policy_commands_returned')
    if isinstance(returned, bool) or returned != 5:
        raise ValueError('pending HOME receipt lacks completed command evidence')
    residual = receipt.get('present_minus_goal')
    if (not isinstance(residual, dict) or set(residual) != set(ORDER)
            or any(isinstance(residual[name], bool) for name in ORDER)):
        raise ValueError('pending HOME receipt lacks raw residual evidence')
    try:
        residual_values = {name: float(residual[name]) for name in ORDER}
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('pending HOME receipt has invalid raw residual evidence') from error
    if (not np.isfinite(list(residual_values.values())).all()
            or max(abs(present_raw[name] - goal_raw[name]) for name in ORDER) > 6):
        raise ValueError('pending HOME receipt has an excessive Present-to-Goal residual')
    journal = _validate_home_command_journal(
        directory, receipt.get('command_journal'),
        expected_commands=receipt.get('commands_position'),
        settled_state=journal_settled,
        expected_sha256=receipt.get('command_journal_sha256'))
    diagnostics = receipt.get('actuator_diagnostics')
    if (not isinstance(diagnostics, dict) or diagnostics.get('motor_register_writes') != 0
            or diagnostics.get('errors') not in ({}, None)
            or not isinstance(diagnostics.get('registers'), dict)):
        raise ValueError('pending HOME receipt lacks actuator identity/parameter evidence')
    registers = diagnostics['registers']
    required_registers = ('ID', 'Model_Number', 'Firmware_Major_Version',
                          'Firmware_Minor_Version', 'Lock', 'Operating_Mode',
                          'P_Coefficient', 'I_Coefficient', 'D_Coefficient',
                          'Torque_Limit', 'Max_Torque_Limit', 'Acceleration', 'Status')
    for register in required_registers:
        _pending_raw_map(registers.get(register), 'pending HOME '+register)
    for name in ORDER:
        if registers['ID'][name] != ORDER.index(name) + 1:
            raise ValueError('pending HOME actuator identity changed')
        if (registers['Model_Number'][name] != 777
                or registers['Firmware_Major_Version'][name] != 3
                or registers['Firmware_Minor_Version'][name] != 10
                or registers['Lock'][name] != 1
                or registers['Operating_Mode'][name] != 0):
            raise ValueError('pending HOME actuator operating mode changed')
        if registers['P_Coefficient'][name] != 16 or registers['I_Coefficient'][name] != 0 or registers['D_Coefficient'][name] != 0:
            raise ValueError('pending HOME actuator gains changed')
        if registers['Acceleration'][name] != 254 or registers['Status'][name] != 0:
            raise ValueError('pending HOME actuator status/configuration changed')
    snapshot = receipt.get('actuator_snapshot')
    if (not isinstance(snapshot, dict) or snapshot.get('moving') != {name: 0 for name in ORDER}
            or snapshot.get('torque') != {name: 1 for name in ORDER}):
        raise ValueError('pending HOME receipt is not a quiet energized snapshot')
    if (_pending_raw_map(snapshot.get('goal_raw'), 'pending HOME snapshot Goal_Position', allow_zero=False) != goal_raw
            or _pending_raw_map(snapshot.get('present_raw'), 'pending HOME snapshot Present_Position') != present_raw
            or not np.allclose(_pending_position_map(snapshot.get('goal_position'), 'pending HOME snapshot Goal position'),
                               goal_position, atol=1e-12, rtol=0)
            or not np.allclose(_pending_position_map(snapshot.get('present_position'), 'pending HOME snapshot Present position'),
                               present_position, atol=1e-12, rtol=0)):
        raise ValueError('pending HOME snapshot differs from its endpoint fields')
    receipt = deepcopy(receipt)
    receipt['goal_raw'] = goal_raw
    receipt['present_raw'] = present_raw
    receipt['goal_position'] = goal_position.tolist()
    receipt['present_position'] = present_position.tolist()
    receipt['settled_q_position'] = settled.tolist()
    receipt['journal_settled_q_position'] = journal_settled.tolist()
    receipt['journal'] = journal
    receipt['actuator_diagnostics'] = diagnostics
    receipt['actuator_snapshot'] = snapshot
    receipt['path'] = str(path.resolve())
    return receipt


def load_incomplete_arm_reconnection(output):
    """Same-output evidence only; never an override of an unknown live cap."""
    directory=Path(output)
    incomplete=directory/'incomplete-execution.json'
    if not incomplete.exists():return None
    state=json.loads(incomplete.read_text())
    if type(state.get('requires_home')) is not bool:
        raise ValueError('invalid incomplete-execution recovery record')
    if not state['requires_home']:return None
    source_path=directory/'last-attempt.json'
    if not source_path.exists():return None
    source=json.loads(source_path.read_text())
    from .v21_position_servo_recovery import (
        _failed_source, failed_position_servo_profile, latest_failed_position_servo_hold,
    )
    admitted=_failed_source(source)
    if admitted is None:return None
    touched,baseline,goal=admitted
    profile_state=failed_position_servo_profile(source,admitted)
    latest_hold=latest_failed_position_servo_hold(directory,source,admitted)
    if latest_hold is not None:
        goal=latest_hold['goal_raw']
        profile_state=latest_hold['profile_state']
    pending=_load_pending_home_receipt(directory,source)
    if pending is not None:
        # The physical HOME chunk is newer than the failed execution and any
        # older position-servo recovery receipt. Keep the failed profile for
        # parameter admission, but use this receipt's exact endpoint Goal.
        goal=deepcopy(pending['goal_raw'])
    compensation_restored=False
    compensation=None
    if (directory/'fixed-goal-compensation.json').exists():
        from .v21_fixed_goal_service import load_fixed_goal_compensation
        compensation=load_fixed_goal_compensation(directory,source)
        compensation_restored=bool(compensation and compensation.get('state')=='completed' and compensation.get('completed') is True
            and compensation.get('configuration_restored') is True)
        if compensation_restored:
            profile_state='baseline1000'
    if profile_state not in ('cleanup_cap200','baseline1000'):
        raise ValueError('incomplete ARM reconnect has no verified final parameter profile')
    if pending is not None:
        expected_parameters = deepcopy(baseline)
        if profile_state == 'cleanup_cap200':
            for name in touched:
                expected_parameters[name]['Torque_Limit'] = 200
        registers = pending['actuator_diagnostics']['registers']
        for name in ORDER:
            actual = {register: registers[register][name]
                      for register in expected_parameters[name]}
            if actual != expected_parameters[name]:
                raise ValueError('pending HOME actuator parameters differ from the evidenced profile')
    baseline_restored=profile_state=='baseline1000'
    held=np.asarray(pending['goal_position'] if pending is not None else
        latest_hold['held_position'] if latest_hold is not None else
        source['position_servo']['cleanup']['hold']['attempts'][-1]['held_state'],dtype=float)
    if held.shape!=(6,) or not np.isfinite(held).all():raise ValueError('invalid original HOLD actual position')
    cache=dict(goal_raw=deepcopy(goal),reference_present_raw=deepcopy(goal),
        raw_freeze_verified=False if pending is not None else True,
        reference_present=held.tolist(),held_position=held.tolist())
    if pending is not None:
        cache.update(actual_present_raw=deepcopy(pending['present_raw']),
                     actual_present_position=deepcopy(pending['present_position']),
                     present_goal_residual=deepcopy(pending['present_minus_goal']))
    return dict(source=source,touched_motors=touched,baseline=baseline,goal_raw=goal,
        effective_goal_raw=deepcopy(goal),held_position=held.tolist(),profile_state=profile_state,
        baseline_restored=baseline_restored,original_hold_cache=cache,
        hold_source=(pending['path'] if pending is not None else
                     None if latest_hold is None else latest_hold['path']),
        hold_predecessor=None if latest_hold is None else latest_hold['predecessor_path'],
        recovery_report_snapshot=None if latest_hold is None else latest_hold['report_snapshot'],
        pending_home_receipt_snapshot=None if pending is None else pending,
        compensation_receipt_snapshot=deepcopy(compensation),compensation_restored=compensation_restored)


def existing_owner_connection_initializer(owner,*,gripper_force='lerobot',arm_p_coefficient=None):
    # arm_p_coefficient (training overhaul doc 17.92): the warm checks also accept this service's own arm gain
    from .v21_arm_gain import check_arm_p_coefficient
    arm_p_coefficient=check_arm_p_coefficient(arm_p_coefficient)
    recovery=load_incomplete_arm_reconnection(owner.output)
    if recovery is None:
        if arm_p_coefficient is None:return connect_preserving_warm_hold
        def connect(io,**kwargs):return connect_preserving_warm_hold(io,arm_p_coefficient=arm_p_coefficient,**kwargs)
        return connect
    if gripper_force!='lerobot':raise RuntimeError('incomplete ARM reconnect preserves gripper500')
    owner.stop_event.set()
    owner.runtime.stop()
    owner.runtime.requires_home_after_incomplete_execution=True
    def initialize(io):
        owner._lease_check()
        if (not owner.stop_event.is_set() or owner.runtime.stopped is not True
                or owner.runtime.requires_home_after_incomplete_execution is not True
                or load_incomplete_arm_reconnection(owner.output)!=recovery):
            raise RuntimeError('incomplete ARM stopped reconnect admission changed')
        result=connect_preserving_warm_hold(io,incomplete_arm_output=owner.output,arm_p_coefficient=arm_p_coefficient)
        if (not owner.stop_event.is_set() or owner.runtime.stopped is not True
                or owner.runtime.requires_home_after_incomplete_execution is not True
                or load_incomplete_arm_reconnection(owner.output)!=recovery):
            raise RuntimeError('incomplete ARM reconnect admission changed before restoring HOLD evidence')
        # The original raw freeze is genuine saved evidence. Fresh warm checks
        # above reverify that exact goal; a bounded stable Present residual is
        # not a new freeze.
        owner._v21_hold_cache=deepcopy(recovery['original_hold_cache'])
        return result
    return initialize


def verify_existing_configuration(bus,*,arm_recovery=None,arm_p_coefficient=None):
    # These are the CURRENT installed SO101Follower.configure values, not tuning.
    expected={'Operating_Mode':0,'P_Coefficient':16,'I_Coefficient':0,'D_Coefficient':0,
              'Return_Delay_Time':0,'Acceleration':254}
    if bus.protocol_version==0:expected['Maximum_Acceleration']=254
    # 17.92: a service started with an arm gain also accepts that gain on shoulder lift and elbow flex (its own
    # earlier write, kept by a warm restart); every other motor and register stays exactly as configure() wrote it.
    from .v21_arm_gain import ARM_GAIN_JOINTS,check_arm_p_coefficient
    arm_p_coefficient=check_arm_p_coefficient(arm_p_coefficient)
    accepted_p={name:{16}|({arm_p_coefficient} if arm_p_coefficient is not None and name in ARM_GAIN_JOINTS else set())
                for name in ORDER}
    result={}
    for register,value in expected.items():
        result[register]=read_map(bus,register)
        if register=='P_Coefficient':
            if any(v not in accepted_p[name] for name,v in result[register].items()):
                raise RuntimeError('warm configuration differs: '+register)
        elif any(v!=value for v in result[register].values()):
            raise RuntimeError('warm configuration differs: '+register)
    for register,value in {'Max_Torque_Limit':500,'Protection_Current':250,'Overload_Torque':25}.items():
        actual=int(bus.read(register,'gripper',normalize=False,num_retry=4))
        result['gripper/'+register]=actual
        if actual!=value:raise RuntimeError('warm gripper protection differs: '+register)
    # A failed controller or HOME recovery may leave either its verified
    # cleanup cap or the original baseline. A restart must accept exactly the
    # final evidenced profile and must not write either profile here.
    actual=int(bus.read('Torque_Limit','elbow_flex',normalize=False,num_retry=4))
    result['elbow_flex/Torque_Limit']=actual
    state=None if arm_recovery is None else arm_recovery.get('profile_state')
    if arm_recovery is not None and state not in ('cleanup_cap200','baseline1000'):
        raise RuntimeError('incomplete ARM reconnect has an unknown parameter profile')
    reduced=state=='cleanup_cap200'
    expected_cap=(200 if reduced and 'elbow_flex' in arm_recovery['touched_motors']
                  else 1000 if arm_recovery is None else
                  arm_recovery['baseline']['elbow_flex']['Torque_Limit'])
    if actual!=expected_cap:raise RuntimeError('warm J3 torque limit differs from restored baseline')
    if arm_recovery is not None:
        from .v21_same_goal_integral import IDENTITY
        expected=deepcopy(arm_recovery['baseline'])
        if reduced:
            for name in arm_recovery['touched_motors']:expected[name]['Torque_Limit']=200
        columns={key:read_map(bus,key) for key in next(iter(expected.values()))}
        actual={name:{key:column[name] for key,column in columns.items()} for name in ORDER}
        if actual!=expected:raise RuntimeError('incomplete ARM reconnect parameter profile changed')
        identity={key:read_map(bus,key) for key in IDENTITY}
        for axis,name in enumerate(ORDER):
            if {key:values[name] for key,values in identity.items()}!=dict(IDENTITY,ID=axis+1):
                raise RuntimeError('incomplete ARM reconnect identity changed: '+name)
        result.update(arm_recovery=True,parameters_by_motor=actual,requires_home=True)
    return result


def stationary_existing_goal(io,*,expected_raw=None,wait=time.sleep,arm_recovery=None):
    samples=[]
    for index in range(3):
        if index:wait(.05)
        sample=actuator_snapshot(io.follower.bus)
        if any(v!=1 for v in sample['torque'].values()):
            raise RuntimeError('warm connection requires all motor torques already enabled')
        if any(v==0 for v in sample['goal_raw'].values()):
            raise RuntimeError('warm connection refuses a raw-zero goal')
        if expected_raw is not None and sample['goal_raw']!=expected_raw:
            raise RuntimeError('existing goal changed during warm connection')
        expected_raw=sample['goal_raw']
        present=np.array([sample['present_position'][k] for k in ORDER])
        goal=np.array([sample['goal_position'][k] for k in ORDER])
        for normalized,key in ((present,'present_raw'),(goal,'goal_raw')):
            reconstructed=io.calibration_reconstructed_encoder_cells(normalized)
            raw=np.array([sample[key][k] for k in ORDER])
            if np.abs(reconstructed-raw).max()>1:
                raise RuntimeError('warm normalization hides an out-of-range raw position')
        for q in (present,goal):
            if (q<LOWER).any() or (q>UPPER).any():raise RuntimeError('warm pose outside joint bounds')
        if np.abs(goal-present).max()>io.settle_position_tolerance:
            raise RuntimeError('existing goal has not reached the existing position tolerance')
        if arm_recovery is not None:
            from .servo_register_decode import decode_present_load
            from .v21_position_servo_recovery import HOLD_MAX_RAW_ERROR, HOLD_MAX_RESIDUAL
            other=[i for i,name in enumerate(ORDER) if name!='elbow_flex']
            other_raw_error=max(abs(sample['present_raw'][ORDER[i]]-sample['goal_raw'][ORDER[i]])
                                for i in other)
            other_q_error=float(np.abs(goal[other]-present[other]).max())
            j3_q_error=float(abs(goal[2]-present[2]))
            j3_raw_error=abs(sample['present_raw']['elbow_flex']-sample['goal_raw']['elbow_flex'])
            goal_q_error=float(np.abs(
                goal-np.asarray(arm_recovery['original_hold_cache']['held_position'])).max())
            violations=[]
            if sample['goal_raw']!=arm_recovery['goal_raw']:violations.append('goal_raw_changed')
            # Failed ARM cleanup and explicit HOME recovery both verify this
            # bounded residual on all six axes. Static encoder offset is not
            # motion: keep the separate .1 cross-sample/camera step checks.
            if other_raw_error>HOLD_MAX_RAW_ERROR:violations.append('other_raw_error')
            if other_q_error>HOLD_MAX_RESIDUAL+1e-12:violations.append('other_q_error')
            if j3_q_error>HOLD_MAX_RESIDUAL+1e-12:violations.append('j3_q_error')
            if j3_raw_error>HOLD_MAX_RAW_ERROR:violations.append('j3_raw_error')
            if any(sample['moving'].values()):violations.append('moving')
            if goal_q_error>1e-5:violations.append('goal_q_changed')
            if violations:
                details=dict(violations=violations,j3_raw_error=j3_raw_error,j3_q_error=j3_q_error,
                    other_raw_error=other_raw_error,other_q_error=other_q_error,
                    goal_q_error=goal_q_error,present_raw=sample['present_raw'],
                    goal_raw=sample['goal_raw'],moving=sample['moving'])
                raise RuntimeError('incomplete ARM reconnect differs from recorded quiet HOLD: '
                                   +json.dumps(details,sort_keys=True))
            load=read_map(io.follower.bus,'Present_Load')
            if (any(value<0 or value>2047 or abs(decode_present_load(value).decoded)>32 for value in load.values())
                    or any(read_map(io.follower.bus,'Status').values())
                    or any(read_map(io.follower.bus,'Present_Velocity').values())):
                raise RuntimeError('incomplete ARM reconnect is not quiet and low-load')
            sample['present_load_raw']=load
        io.clearance_gate.require_readback(present,label='warm actual position')
        io.clearance_gate.require_policy_path(present,np.repeat(goal[None],5,0))
        samples.append(sample)
    q=np.array([[s['present_position'][k] for k in ORDER] for s in samples])
    tolerance=.1 if arm_recovery is not None else io.settle_step_tolerance
    if np.abs(np.diff(q,axis=0)).max()>tolerance+1e-12:
        raise RuntimeError('warm actual position is not quiet')
    return samples


def connect_preserving_warm_hold(io,*,legacy_preseed=None,legacy_connect=None,wait=time.sleep,
                                 incomplete_arm_output=None,arm_p_coefficient=None):
    """Cold path unchanged; verified energized reconnect has ZERO register writes."""
    if legacy_preseed is None:
        from .so101_safe_energize import preseed_goal_position
        legacy_preseed=preseed_goal_position
    if legacy_connect is None:
        from scripts.serve_so101_realtime_fcpg_lewm import connect_follower_with_retry
        legacy_connect=connect_follower_with_retry
    recovery=None if incomplete_arm_output is None else load_incomplete_arm_reconnection(incomplete_arm_output)
    if incomplete_arm_output is not None and recovery is None:
        raise RuntimeError('incomplete ARM reconnect evidence disappeared')
    follower=io.follower;bus=follower.bus
    bus.connect()  # Existing ID/model/firmware handshake; no motor register writes.
    torque=read_map(bus,'Torque_Enable')
    if recovery is not None and any(value!=1 for value in torque.values()):
        raise RuntimeError('incomplete ARM reconnect requires already enabled torque; cold writes forbidden')
    if all(value==0 for value in torque.values()):
        report=legacy_preseed(bus)
        bus.disconnect(disable_torque=False)
        legacy_connect(follower)
        return dict(report,connection_mode='cold_existing_safety_path')
    try:
        if any(value!=1 for value in torque.values()):raise RuntimeError('mixed torque state; explicit recovery required')
        if not follower.is_calibrated:raise RuntimeError('warm hardware calibration does not match')
        config=verify_existing_configuration(bus,arm_recovery=recovery,arm_p_coefficient=arm_p_coefficient)
        before=stationary_existing_goal(io,wait=wait,arm_recovery=recovery)
        recovered_cameras=[]
        for view,camera in follower.cameras.items():
            try:
                camera.connect()
            except (TimeoutError,ConnectionError,RuntimeError) as error:
                recover=getattr(io,'_recover_camera',None)
                if not callable(recover):raise
                # Reuse the bounded, identity-checked camera-only recovery.
                # It never reconnects or configures the actuator bus.
                recover(view,error)
                recovered_cameras.append(view)
        if not follower.is_connected or not follower.is_calibrated:
            raise RuntimeError('warm camera/bus/calibration contract failed')
        after=stationary_existing_goal(io,expected_raw=before[-1]['goal_raw'],wait=wait,arm_recovery=recovery)
        delta=np.array([after[-1]['present_position'][k]-before[-1]['present_position'][k] for k in ORDER])
        tolerance=.1 if recovery is not None else io.settle_step_tolerance
        if np.abs(delta).max()>tolerance+1e-12:
            raise RuntimeError('physical pose changed while connecting cameras')
        if recovery is not None:
            from .v21_position_servo_recovery import HOLD_MAX_RAW_ERROR, HOLD_MAX_RESIDUAL
            verify_existing_configuration(bus,arm_recovery=recovery,arm_p_coefficient=arm_p_coefficient)
            if load_incomplete_arm_reconnection(incomplete_arm_output)!=recovery:
                raise RuntimeError('incomplete ARM reconnect source changed')
        return dict(connection_mode='warm_preserved_goal',configuration=config,before=before,after=after,
            position_delta=delta.tolist(),motor_register_writes=0,torque_changed=False,
            camera_recovery_views=recovered_cameras,
            arm_recovery=recovery is not None,requires_home=recovery is not None,
            needs_fixed_goal_compensation=bool(recovery is not None and (
                recovery.get('profile_state')!='baseline1000'
                or abs(after[-1]['present_raw']['elbow_flex']-after[-1]['goal_raw']['elbow_flex'])>HOLD_MAX_RAW_ERROR
                or abs(after[-1]['present_minus_goal']['elbow_flex'])>HOLD_MAX_RESIDUAL+1e-12)),
            profile_state=None if recovery is None else recovery['profile_state'],
            baseline_restored=bool(recovery is not None and recovery.get('baseline_restored',False)),
            compensation_restored=bool(recovery is not None and recovery.get('compensation_restored',False)),
            goal_raw=after[-1]['goal_raw'],holding_position=after[-1]['goal_position'],
            observed_position=after[-1]['present_position'])
    except BaseException as exc:
        if recovery is not None:raise  # This special reconnect must never replace the recorded HOLD Goal.
        try:raw_stop(bus)  # No torque-off or silent configuration fallback.
        except Exception as stop_exc:
            raise RuntimeError(f'warm connection rejected: {exc}; HOLD unverified: {stop_exc}') from exc
        raise


def verified_hold(owner,*,reason='v21_emergency_stop',actuator_events=None):
    """First STOP cancels motion; repeated STOP verifies the same held target."""
    io=owner.io;bus=io.follower.bus
    cached=getattr(owner,'_v21_hold_cache',None)
    # A warm connection only proves that its old goal is quiet and nearby.
    # It may still contain the normalized HOLD's one-cell truncation error.
    # Only a completed raw Present -> exact Goal freeze can suppress a STOP
    # write; subsequent Present drift must not ratchet this proven fixed goal.
    if (cached is not None and cached.get('raw_freeze_verified') is True
            and cached.get('reference_present_raw')==cached.get('goal_raw')):
        try:
            sample=actuator_snapshot(bus)
            ids={k:bus.motors[k].id for k in ORDER}
            raw=sample['present_raw']
            if any(not bus.calibration[k].range_min<=raw[k]<=bus.calibration[k].range_max for k in ORDER):
                raise ValueError('cached HOLD present raw outside calibration range')
            decoded=bus._normalize({ids[k]:raw[k] for k in ORDER})
            reconstructed=bus._unnormalize(decoded)
            q=np.array([decoded[ids[k]] for k in ORDER])
            if (not np.isfinite(q).all() or (q<LOWER).any() or (q>UPPER).any()
                    or any(abs(reconstructed[ids[k]]-raw[k])>1 for k in ORDER)):
                raise ValueError('cached HOLD normalization hides raw position')
            sample['present_position_from_raw']=dict(zip(ORDER,q.tolist()))
            unchanged=(sample['goal_raw']==cached['goal_raw']
                       and all(v==0 for v in sample['moving'].values())
                       and all(v==1 for v in sample['torque'].values())
                       and np.abs(q-np.array(cached['reference_present'])).max()<=io.settle_position_tolerance)
            if unchanged:
                io.clearance_gate.require_readback(q,label='repeated STOP actual readback')
                record=dict(event='safe_hold_verified',reason='existing_hold_reverified_no_new_goal',
                    reused_existing_hold=True,motor_register_writes=0,monotonic_s=time.monotonic(),
                    actual_readback=sample,attempts=[dict(held_state=np.array(cached['held_position']),
                    requested_goal_raw=dict(cached['reference_present_raw']),
                    goal_readback_raw=sample['goal_raw'],goal_register_verified=True,
                    goal_readback=sample['goal_position'],mismatched_joints={})])
                owner.last_hold_record=record
                if actuator_events is not None:
                    actuator_events.append(dict(event='existing_freeze_reverified',reason=reason,
                        actual_readback=sample,motor_register_writes=0))
                    record['actuator_events']=actuator_events
                return record
        except Exception:
            pass  # No cached evidence may suppress the established verified STOP.
    owner._v21_hold_cache=None
    kwargs={} if actuator_events is None else dict(actuator_events=actuator_events)
    record=io._hold_at_measured_pose(None,reason=reason,**kwargs)
    last=record['attempts'][-1]
    if (last.get('goal_register_verified') is not True or last.get('pose_validated') is not True
            or last.get('requested_goal_raw')!=last.get('goal_readback_raw')):
        raise RuntimeError('HOLD lacks exact raw measured-to-goal verification')
    owner._v21_hold_cache=dict(goal_raw=dict(last['goal_readback_raw']),
        reference_present_raw=dict(last['requested_goal_raw']),raw_freeze_verified=True,
        reference_present=np.asarray(last['held_state']).tolist(),held_position=np.asarray(last['held_state']).tolist())
    record=dict(record,reused_existing_hold=False,motor_register_writes=len(record['attempts']))
    if actuator_events is not None:record['actuator_events']=actuator_events
    owner.last_hold_record=record
    return record


def position_servo_selected_contract(follow, goal):
    """Bind deferred axes to their one actual admission pose, never a new goal."""
    fields=('selected_axes','selected_axes_arrival','deferred_axes',
            'deferred_anchor_q','deferred_position_tolerance','source_q0')
    if not any(key in follow for key in fields):
        return None
    selected=follow.get('selected_axes')
    deferred=follow.get('deferred_axes')
    anchor=np.asarray(follow.get('deferred_anchor_q'))
    source=np.asarray(follow.get('source_q0'))
    tolerance=follow.get('deferred_position_tolerance')
    if (not isinstance(selected,list) or selected!=[2,3,1]
            or any(type(axis) is not int for axis in selected)
            or not isinstance(deferred,list) or any(type(axis) is not int for axis in deferred)
            or deferred!=sorted(set(deferred)) or not set(deferred)<={0,4,5}
            or anchor.shape!=(6,) or anchor.dtype.kind not in 'iuf' or not np.isfinite(anchor).all()
            or source.shape!=(6,) or source.dtype.kind not in 'iuf' or not np.isfinite(source).all()
            or type(tolerance) not in (int,float) or tolerance!=.25):
        raise ValueError('invalid selected-axis finishing admission contract')
    target=np.asarray(goal,dtype=np.float64)
    if target.shape!=(6,) or not np.isfinite(target).all():
        raise ValueError('selected-axis finishing requires the original six-axis target')
    anchor=anchor.astype(np.float64,copy=True)
    expected=[axis for axis in (0,4,5) if abs(anchor[axis]-target[axis])>.2]
    if deferred!=expected:
        raise ValueError('deferred axes differ from actual admission residuals')
    return dict(selected_axes=selected.copy(),deferred_axes=deferred.copy(),
                deferred_anchor_q=anchor.tolist(),deferred_position_tolerance=.25,
                source_q0=source.astype(np.float64,copy=True).tolist())


def position_servo_deferred_corridor_valid(contract, goal, positions):
    """Use the helper's original q0-to-target interval or precise target band."""
    deferred=contract['deferred_axes']
    if not deferred:
        return True
    target=np.asarray(goal,dtype=np.float64)[deferred]
    source=np.asarray(contract['source_q0'],dtype=np.float64)[deferred]
    actual=np.asarray(positions,dtype=np.float64)[...,deferred]
    inside=(actual>=np.minimum(source,target)-1e-5)&(actual<=np.maximum(source,target)+1e-5)
    return bool(np.logical_or(inside,np.abs(actual-target)<=.2).all())


def position_servo_readback_proof(follow, goal, actual, *, previous_actual=None):
    """Check one later real position against the completed finishing contract."""
    contract=position_servo_selected_contract(follow,goal)
    target=np.asarray(goal,dtype=np.float64)
    actual=np.asarray(actual,dtype=np.float64)
    previous=np.asarray(follow['positions'][-1],dtype=np.float64)
    if (target.shape!=(6,) or actual.shape!=(6,) or previous.shape!=(6,)
            or not np.isfinite([target,actual,previous]).all()):
        raise ValueError('position finishing requires finite actual six-axis readbacks')
    deferred=[] if contract is None else contract['deferred_axes']
    strict=[axis for axis in range(6) if axis not in deferred or follow.get('goal_reached') is True]
    error=float(np.abs(actual-target).max())
    strict_error=float(np.abs(actual[strict]-target[strict]).max())
    step=float(np.abs(actual-previous).max())
    drift=(float(np.abs(actual[deferred]-np.asarray(contract['deferred_anchor_q'])[deferred]).max())
           if deferred else 0.)
    corridor=contract is None or position_servo_deferred_corridor_valid(contract,target,actual)
    proof=dict(original_target=target.tolist(),actual_position=actual.tolist(),max_abs_residual=error,
        required_axes=strict,required_axes_max_abs_residual=strict_error,
        step_from_final_finishing_readback=step,position_tolerance=.2,step_tolerance=.1,
        deferred_max_abs_drift=drift,deferred_original_corridor_valid=corridor,
        verified=bool(strict_error<=.2 and step<=.1 and drift<=.25 and corridor),
        model_feedback_label=False)
    if contract is not None:
        proof.update(contract)
    if previous_actual is not None:
        previous_actual=np.asarray(previous_actual,dtype=np.float64)
        if previous_actual.shape!=(6,) or not np.isfinite(previous_actual).all():
            raise ValueError('position finishing requires a finite actual completion HOLD position')
        hold_step=float(np.abs(actual-previous_actual).max())
        proof.update(completion_hold_actual_position=previous_actual.tolist(),
                     step_from_completion_hold_readback=hold_step,
                     verified=bool(proof['verified'] and hold_step<=.1))
    return proof


def position_servo_selected_completion_proof(follow, goal, positions, times):
    """Recompute all 22 selected-axis samples without replacing other coordinates."""
    from .v21_position_arrival import evaluate_selected_position_arrival

    contract=position_servo_selected_contract(follow,goal)
    if contract is None:
        return None
    if (positions.shape!=(22,6) or times.shape!=(22,) or times[-1]-times[0]<1.
            or (np.diff(times)<.05-1e-8).any()
            or np.abs(np.diff(positions,axis=0)).max()>.1):
        raise ValueError('selected-axis finishing lacks the full quiet second of actual samples')
    deferred=contract['deferred_axes']
    strict=[axis for axis in range(6) if axis not in deferred]
    target=np.asarray(goal,dtype=np.float64)
    if (np.abs(positions[:,strict]-target[strict]).max()>.2
            or deferred and np.abs(positions[:,deferred]-np.asarray(contract['deferred_anchor_q'])[deferred]).max()>.25
            or not position_servo_deferred_corridor_valid(contract,target,positions)):
        raise ValueError('selected-axis finishing left the original target or deferred admission bounds')
    selected=evaluate_selected_position_arrival(goal,positions,times-float(follow['source_time_s']),
        selected_axes=contract['selected_axes'],position_tolerance=.2,step_tolerance=.1,quiet_samples=22)
    reported=follow.get('selected_axes_arrival')
    if (not selected['reached'] or not isinstance(reported,dict)
            or any(reported.get(key)!=value for key,value in selected.items())):
        raise ValueError('selected-axis finishing arrival disagrees with its actual samples')
    return selected


def position_servo_completion_proof(follow, goal, *, quiet_samples=3):
    """Validate independent actual finishing samples, without accessing hardware.

    This receipt has its own real clock and no normal-follow budget. Its tail
    contains enough adjacent samples to verify quietness without treating the
    old model q5 as the predecessor of a much later servo readback.
    """
    from .v21_position_arrival import evaluate_position_arrival

    if (not isinstance(follow,dict)
            or follow.get('schema')!='v21-position-servo-completion-follow-v1'
            or follow.get('baseline_restored') is not True
            or type(follow.get('hold_eligible')) is not bool
            or type(follow.get('goal_reached')) is not bool):
        raise ValueError('invalid position-servo completion evidence')
    arrays=[]
    for name in ('original_target','positions','times_s'):
        value=np.asarray(follow.get(name))
        if value.dtype.kind not in 'iuf' or not np.isfinite(value).all():
            raise ValueError('position-servo completion requires finite actual '+name)
        arrays.append(value.astype(np.float64,copy=True))
    target,positions,times=arrays
    if target.shape!=(6,) or not np.array_equal(target,np.asarray(goal,dtype=np.float64)):
        raise ValueError('position-servo completion changed the original target')
    if positions.ndim!=2 or positions.shape[1]!=6 or len(positions)<quiet_samples+1:
        raise ValueError('position-servo completion lacks adjacent quiet samples')
    if times.shape!=(len(positions),) or not (np.diff(times)>0).all():
        raise ValueError('position-servo completion requires actual increasing sample times')
    source=follow.get('source_time_s')
    if (isinstance(source,(bool,np.bool_)) or not isinstance(source,(int,float,np.integer,np.floating))
            or not np.isfinite(source) or source<0 or (times<source).any()):
        raise ValueError('position-servo completion requires its original source clock')
    arrival=evaluate_position_arrival(target,positions,times-float(source),
        position_tolerance=.2,step_tolerance=.1,quiet_samples=3)
    if follow['goal_reached'] is not arrival['reached']:
        raise ValueError('position-servo completion arrival disagrees with actual samples')
    position_servo_selected_completion_proof(follow,goal,positions,times)
    return positions,times,arrival


def completed_command_hold(owner,goal,follow,q5,*,actuator_events):
    """Called under the bus lock. Retain only a quiet, verified fifth goal.

    No new waiting budget or servo writes on success. A rejected proof freezes
    through the established path; cancellation propagates to the STOP handler.
    """
    io=owner.io;bus=io.follower.bus
    verification=dict(event='normal_goal_verification',started_s=time.monotonic(),readbacks=[])
    actuator_events.append(verification)
    try:
        if owner.stop_event.is_set():raise RuntimeError('normal completion cancelled')
        goal=np.asarray(goal,dtype=np.float32)
        if goal.shape!=(6,) or not np.isfinite(goal).all():raise ValueError('invalid fifth goal')
        if (goal<LOWER).any() or (goal>UPPER).any():raise ValueError('fifth goal outside joint bounds')
        # Retaining an energized quiet goal uses the existing hold limits.
        # Precise original-target arrival is recorded separately by the bridge.
        servo_follow=follow.get('schema')=='v21-position-servo-completion-follow-v1'
        hold_eligible = follow.get('hold_eligible', follow.get('goal_reached'))
        # region agent log
        try:
            import json as _json, time as _time
            _load = read_map(bus, 'Present_Load')
            _cur = read_map(bus, 'Present_Current')
            _goal = read_map(bus, 'Goal_Position')
            _present = read_map(bus, 'Present_Position')
            _p = read_map(bus, 'P_Coefficient')
            with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                _f.write(_json.dumps({'sessionId':'75d952','hypothesisId':'L','location':'v21_hardware_state.py:completed_command_hold','message':'elbow pwm at hold decision','data':{'hold_eligible':hold_eligible is True,'elbow_load_raw':_load.get('elbow_flex'),'elbow_current':_cur.get('elbow_flex'),'elbow_goal_raw':_goal.get('elbow_flex'),'elbow_present_raw':_present.get('elbow_flex'),'elbow_p':_p.get('elbow_flex')},'timestamp':int(_time.time()*1000)})+'\n')
                _f.write(_json.dumps({'sessionId':'75d952','hypothesisId':'M','location':'v21_hardware_state.py:completed_command_hold','message':'wrist pwm at hold decision','data':{'flex_load_raw':_load.get('wrist_flex'),'roll_load_raw':_load.get('wrist_roll'),'flex_goal_raw':_goal.get('wrist_flex'),'flex_present_raw':_present.get('wrist_flex'),'roll_goal_raw':_goal.get('wrist_roll'),'roll_present_raw':_present.get('wrist_roll'),'flex_p':_p.get('wrist_flex'),'roll_p':_p.get('wrist_roll')},'timestamp':int(_time.time()*1000)})+'\n')
        except Exception:
            pass
        # endregion
        # region agent log
        try:
            import json as _json, time as _time
            _zone = {name: read_map(bus, name) for name in (
                'CW_Dead_Zone', 'CCW_Dead_Zone', 'Minimum_Startup_Force', 'Goal_Time', 'Goal_Velocity')}
            _pick = lambda reg, motor: _zone[reg].get(motor)
            with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                _f.write(_json.dumps({'sessionId':'75d952','hypothesisId':'N','location':'v21_hardware_state.py:completed_command_hold','message':'elbow profile and dead zone','data':{'elbow':{name:_pick(name,'elbow_flex') for name in _zone},'lift':{name:_pick(name,'shoulder_lift') for name in _zone},'flex':{name:_pick(name,'wrist_flex') for name in _zone}},'timestamp':int(_time.time()*1000)})+'\n')
        except Exception as _exc:
            try:
                import json as _json, time as _time
                with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                    _f.write(_json.dumps({'sessionId':'75d952','hypothesisId':'N','location':'v21_hardware_state.py:completed_command_hold','message':'profile read failed','data':{'error':f'{type(_exc).__name__}: {_exc}'},'timestamp':int(_time.time()*1000)})+'\n')
            except Exception:
                pass
        # endregion
        if hold_eligible is not True:
            # region agent log
            try:
                import json as _json, time as _time
                _cur = read_map(bus, 'Present_Current')
                _vel = read_map(bus, 'Present_Velocity')
                _goal = read_map(bus, 'Goal_Position')
                _present = read_map(bus, 'Present_Position')
                with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                    _f.write(_json.dumps({'sessionId':'75d952','hypothesisId':'I','location':'v21_hardware_state.py:completed_command_hold','message':'elbow effort before freeze','data':{'elbow_current':_cur.get('elbow_flex'),'lift_current':_cur.get('shoulder_lift'),'elbow_velocity':_vel.get('elbow_flex'),'lift_velocity':_vel.get('shoulder_lift'),'elbow_goal_raw':_goal.get('elbow_flex'),'elbow_present_raw':_present.get('elbow_flex'),'lift_goal_raw':_goal.get('shoulder_lift'),'lift_present_raw':_present.get('shoulder_lift')},'timestamp':int(_time.time()*1000)})+'\n')
            except Exception:
                pass
            # endregion
            # region agent log
            try:
                import json as _json, time as _time
                _load = read_map(bus, 'Present_Load')
                _limit = read_map(bus, 'Torque_Limit')
                _status = read_map(bus, 'Status')
                _p = read_map(bus, 'P_Coefficient')
                with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                    _f.write(_json.dumps({'sessionId':'75d952','hypothesisId':'J','location':'v21_hardware_state.py:completed_command_hold','message':'elbow pwm before freeze','data':{'elbow_load_raw':_load.get('elbow_flex'),'lift_load_raw':_load.get('shoulder_lift'),'elbow_torque_limit':_limit.get('elbow_flex'),'lift_torque_limit':_limit.get('shoulder_lift'),'elbow_status':_status.get('elbow_flex'),'elbow_p':_p.get('elbow_flex'),'lift_p':_p.get('shoulder_lift')},'timestamp':int(_time.time()*1000)})+'\n')
            except Exception:
                pass
            # endregion
            raise RuntimeError('normal follow did not establish quiet hold eligibility')
        count=io.settle_samples
        if servo_follow:
            positions,times,arrival=position_servo_completion_proof(follow,goal,quiet_samples=count)
            verification.update(completion_evidence_schema=follow['schema'],
                                completion_sample_times_s=times.tolist(),
                                original_target_arrival=arrival)
        else:
            if follow.get('schema') is not None or follow.get('budget_s') not in (.4,2.):
                raise RuntimeError('normal follow did not establish quiet hold eligibility')
            positions=np.asarray(follow['positions'],dtype=np.float64)
            if positions.ndim!=2 or positions.shape[1]!=6 or len(positions)<count or not np.isfinite(positions).all():
                raise RuntimeError('normal follow lacks the existing quiet sample quorum')
        q5=np.asarray(q5,dtype=np.float64)
        if q5.shape!=(6,) or not np.isfinite(q5).all():raise RuntimeError('invalid final model readback')
        path=positions if servo_follow else np.concatenate((q5[None],positions))
        if (np.abs(positions[-count:]-goal).max()>io.settle_position_tolerance
                or np.abs(np.diff(path,axis=0)[-count:]).max()>io.settle_step_tolerance):
            raise RuntimeError('normal follow is outside existing position or quiet tolerances')
        # This is the SAME encoder conversion used by normalized sync_write,
        # including integer truncation. Never accept a nearby but different cell.
        expected=dict(zip(ORDER,map(int,io.calibration_reconstructed_encoder_cells(goal))))
        verification.update(expected_goal_position=goal.tolist(),expected_goal_raw=expected)
        sample=actuator_snapshot(bus,cancelled=owner.stop_event.is_set,read_events=verification['readbacks'])
        verification['actual_readback']=sample
        if sample['goal_raw']!=expected:raise RuntimeError('fifth Goal_Position raw readback mismatch')
        if any(v!=1 for v in sample['torque'].values()) or any(v!=0 for v in sample['moving'].values()):
            raise RuntimeError('normal completion is not energized and stationary')
        present=np.array([sample['present_position'][k] for k in ORDER])
        held=np.array([sample['goal_position'][k] for k in ORDER])
        for q,key in ((present,'present_raw'),(held,'goal_raw')):
            if (q<LOWER).any() or (q>UPPER).any():raise RuntimeError('normal completion pose outside bounds')
            if np.abs(io.calibration_reconstructed_encoder_cells(q)-np.array([sample[key][k] for k in ORDER])).max()>1:
                raise RuntimeError('normal completion normalization hides raw position')
        if (np.abs(present-goal).max()>io.settle_position_tolerance
                or np.abs(present-positions[-1]).max()>io.settle_step_tolerance):
            raise RuntimeError('normal completion position moved after follow')
        if servo_follow and (arrival['reached'] or 'selected_axes' in follow):
            proof=position_servo_readback_proof(follow,goal,present)
            verification['position_servo_readback_validation']=proof
            if not proof['verified']:
                raise RuntimeError('position-servo arrival changed during completion verification')
        if owner.stop_event.is_set():raise RuntimeError('normal completion cancelled')
        io.clearance_gate.require_readback(present,label='normal completion actual readback')
        if owner.stop_event.is_set():raise RuntimeError('normal completion cancelled')
        io.clearance_gate.require_policy_path(present,np.repeat(held[None],5,0))
        if owner.stop_event.is_set():raise RuntimeError('normal completion cancelled')
        final_read=dict(register='Goal_Position',normalize=False,started_s=time.monotonic())
        verification['readbacks'].append(final_read)
        try:
            final_goal=read_map(bus,'Goal_Position')
            final_read.update(returned_s=time.monotonic(),values=final_goal)
        except BaseException as exc:
            final_read.update(failed_s=time.monotonic(),error=f'{type(exc).__name__}: {exc}')
            raise
        if owner.stop_event.is_set():raise RuntimeError('normal completion cancelled')
        if final_goal!=expected:raise RuntimeError('fifth Goal_Position changed during completion verification')
        verification.update(returned_s=time.monotonic(),goal_register_verified=True,
                            retained_existing_goal=True,motor_register_writes=0,
                            verification_scope='quiet_goal_retention_not_precise_arrival')
        record=dict(event='safe_hold_verified',reason='normal_completion_retained_fifth_goal',
            reused_existing_hold=True,retained_existing_goal=True,
            motor_register_writes=0,monotonic_s=time.monotonic(),
            actual_readback=sample,actuator_events=actuator_events,
            attempts=[dict(held_state=held,goal_readback=sample['goal_position'],
                           goal_register_verified=True,goal_readback_raw=dict(sample['goal_raw']),
                           mismatched_joints={})])
        # A normal retained command is not an emergency-freeze cache entry.
        owner.last_hold_record=record
        return record
    except BaseException as exc:
        verification.update(failed_s=time.monotonic(),error=f'{type(exc).__name__}: {exc}',
                            retained_existing_goal=False)
        if owner.stop_event.is_set() or not isinstance(exc,Exception):
            # Physical cancellation precedes even the independent disk journal.
            owner.stop_event.set()
            verified_hold(owner,reason='v21_normal_completion_cancelled',actuator_events=actuator_events)
            raise
        record=verified_hold(owner,reason='v21_normal_completion_fallback',actuator_events=actuator_events)
        record['normal_validation_error']=verification['error']
        return record
