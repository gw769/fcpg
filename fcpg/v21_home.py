"""The existing manual HOME route needs readbacks, not fabricated RGB frames.

The planner reuses the established route with a HOME-only clearance reserve.
The executor changes neither commands nor joint limits and creates no Actor
transition. Actual readback protection is never weakened.
"""
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

# HOME only: the measured failure had 0.29 mm nominal headroom and about
# 1.5 mm tracking-induced loss. This conservative reserve is not a guarantee
# for arbitrary tracking errors; the unchanged actual-readback guard remains
# authoritative. Never apply it by editing the shared policy clearance gate.
HOME_TRACKING_RESERVE_MM = 5.0
PENDING_HOME_RECEIPT_NAME = 'pending-home-receipt.json'
PENDING_HOME_RECEIPT_SCHEMA = 'v21-home-physical-settled-receipt-v1'


def home_clearance_gate(owner):
    gate = owner.clearance
    return replace(gate, planned_clearance_mm=max(
        gate.planned_clearance_mm,
        gate.readback_clearance_mm + HOME_TRACKING_RESERVE_MM))


# Training overhaul doc section 17.91 (James 10/24 「继续可以」): a trial that stops on the read-back floor (the arm ends
# 5-8 mm under its commands near the table and sinks further in its HOLD) leaves every motion refused, HOME included,
# and a person had to lift the arm. With the service's --home-lift-below-floor, HOME first lifts ONE joint, the way the
# 10/24 recovery did by hand (robot-20261024/lift_clear_warm_20261024.py): only from just under the floor, only with all
# torques on, from that joint's own Goal register (the measured pose sits under it), in small paced steps through the
# service's own write path, stopping when the clearance stops rising or another joint moves. plan_home_chunk itself
# still refuses any source under the floor.
LIFT_JOINTS = (1, 2, 3)                # shoulder lift, elbow flex, wrist flex: the gripper and roll cannot lift the jaws
LIFT_TARGET_ABOVE_FLOOR_MM = 15.       # lift until this far above the read-back floor (-10 mm -> +5 mm)
LIFT_LOWEST_UNDER_FLOOR_MM = 10.       # never from further under the floor than this: a person must look
LIFT_STEP = .5                         # goal change per step (units)
LIFT_PERIOD_S = .2
LIFT_MAX_CHANGE = 20.                  # largest goal change of the lifted joint (units)
LIFT_STALL_STEPS = 16                  # steps in a row that raise nothing (10/24 the loaded elbow needed seven to start)
LIFT_OTHER_JOINT_UNITS = 3.            # another joint moving this far stops the lift


def choose_lift(gate, q, target_mm, max_change=LIFT_MAX_CHANGE):
    """(joint, signed goal change): the smallest one-joint change reaching target_mm whose straight path keeps the
    clearance rising all the way, by the gate's own kinematics."""
    q = np.asarray(q, np.float64)
    best = None
    for joint in LIFT_JOINTS:
        for direction in (-1., 1.):
            for amount in np.arange(LIFT_STEP, max_change + 1e-9, LIFT_STEP):
                candidate = q.copy()
                candidate[joint] += direction * amount
                if gate.clearance_mm(candidate) >= target_mm:
                    along = np.array([gate.clearance_mm(q + t * (candidate - q)) for t in np.linspace(0., 1., 41)])
                    if np.all(np.diff(along) >= -1e-6) and (best is None or amount < best[0]):
                        best = (float(amount), joint, direction * float(amount))
                    break
    if best is None:
        raise RuntimeError(f'no single-joint lift of at most {max_change} units reaches {target_mm:.1f} mm while rising')
    return best[1], best[2]


def lift_below_floor(owner, output=None, *, clock=time.monotonic):
    """None when the measured pose is on or above the read-back floor (nothing is read beyond it, nothing moves); else
    the record of a one-joint lift to LIFT_TARGET_ABOVE_FLOOR_MM above the floor, also written to
    output/home-lift-<ns>.json. Raises (after writing the record) when the lift is refused or stops short."""
    from .v21_hardware_state import ORDER, actuator_snapshot
    gate = owner.clearance
    q = np.asarray(owner._read_q(), np.float64)
    start = gate.clearance_mm(q)
    floor = gate.readback_clearance_mm
    if start >= floor - 1e-9:
        return None
    target = floor + LIFT_TARGET_ABOVE_FLOOR_MM
    record = dict(kind='home-lift-below-floor', section='training overhaul doc 17.91', floor_mm=floor,
                  target_mm=target, start_q_position=q.tolist(), start_clearance_mm=start, steps=[], outcome=None)

    def finish(outcome, error=None):
        record.update(outcome=outcome, finished_s=float(clock()))
        if output is not None:
            from .v21_training import write_json
            path = Path(output) / f'home-lift-{time.time_ns()}.json'
            write_json(path, record)
            record['path'] = str(path)
        if error is not None:
            raise RuntimeError(f'HOME lift under the floor: {error}')
        return record

    record['started_s'] = float(clock())
    if start < floor - LIFT_LOWEST_UNDER_FLOOR_MM:
        return finish('refused', f'the jaws read {start:.2f} mm, more than {LIFT_LOWEST_UNDER_FLOOR_MM:.0f} mm under '
                                 f'the {floor:.1f} mm floor; a person must free the arm')
    owner._lease_check()
    with owner.bus_lock:
        snapshot = actuator_snapshot(owner.io.follower.bus, cancelled=owner.stop_event.is_set)
    record.update(torque=snapshot['torque'], goal_position=snapshot['goal_position'])
    if any(int(snapshot['torque'][name]) != 1 for name in ORDER):
        return finish('refused', 'not every torque is on: the HOLD is gone and nothing moved')
    try:
        joint, change = choose_lift(gate, q, target)
    except RuntimeError as error:
        return finish('refused', str(error))
    goals = np.asarray([snapshot['goal_position'][name] for name in ORDER], np.float64)
    record.update(joint=ORDER[joint], planned_change=change,
                  planned_clearance_mm=gate.clearance_mm(np.where(np.arange(6) == joint, q + change, q)))
    start_goal, best, stalled = float(goals[joint]), start, 0
    while abs(goals[joint] - start_goal) < LIFT_MAX_CHANGE - 1e-9:
        owner._lease_check()
        command = goals.copy()
        command[joint] += np.sign(change) * LIFT_STEP
        owner._write_q(command.astype(np.float32), goals.astype(np.float32))     # raises once STOP is latched
        goals = command
        if owner.stop_event.wait(LIFT_PERIOD_S):
            return finish('stopped', 'STOP during the lift')
        now_q = np.asarray(owner._read_q(), np.float64)
        reached = gate.clearance_mm(now_q)
        others = float(np.abs(np.delete(now_q - q, joint)).max())
        record['steps'].append(dict(goal=float(goals[joint]), q_joint=float(now_q[joint]), clearance_mm=reached,
                                    other_joints_moved=others))
        if others > LIFT_OTHER_JOINT_UNITS:
            return finish('another_joint_moved', f'another joint moved {others:.2f} units')
        if reached < best - .5:
            return finish('clearance_fell', f'the clearance fell to {reached:.2f} mm')
        stalled = stalled + 1 if reached < best + .05 else 0
        best = max(best, reached)
        if reached >= target:
            record.update(end_q_position=now_q.tolist(), end_clearance_mm=reached,
                          goal_change=float(goals[joint] - start_goal))
            return finish('lifted')
        if stalled >= LIFT_STALL_STEPS:
            return finish('no_rise', f'{LIFT_STALL_STEPS} steps in a row raised nothing ({reached:.2f} mm)')
    return finish('cap', f'the {LIFT_MAX_CHANGE:.0f}-unit cap reached at {best:.2f} mm')


def plan_home_chunk(owner, current, target):
    """Reuse the established HOME planner with a separate, stricter buffer.

    Below the hard readback floor, refuse automatic recovery. Between the
    floor and HOME buffer, only the existing monotonic-upward path is legal.
    """
    from .prod_v20_3_hardware_service import V203HardwareService
    gate = home_clearance_gate(owner)
    gate.require_readback(current, label='HOME source')
    # V21's explicitly authorized larger reset: at most four units per command.
    # The original audited route/fallback and stricter HOME buffer are unchanged.
    planner = SimpleNamespace(clearance=gate, io=owner.io,home_joint_step=20.)
    commands = V203HardwareService._home_step(planner, current, target)
    gate.require_policy_path(current, commands)
    return commands


def _settled_register_state(owner, final):
    """Read the final raw Goal/Present pair after the HOME HOLD.

    A pending receipt is only useful for a warm reconnect when it contains the
    actual register cells. Test-only owners without a bus use their calibrated
    inverse as a clearly labelled fallback; the live owner always takes the
    single raw-register snapshot path.
    """
    from .v21_hardware_state import ORDER, actuator_snapshot

    bus = getattr(getattr(owner.io, 'follower', None), 'bus', None)
    if bus is not None:
        from .v21_hardware_state import actuator_diagnostics
        with owner.bus_lock:
            sample = actuator_snapshot(bus)
            diagnostics = actuator_diagnostics(bus)
        if any(value != 0 for value in sample['moving'].values()):
            raise RuntimeError('HOME pending endpoint is still moving')
        if any(value != 1 for value in sample['torque'].values()):
            raise RuntimeError('HOME pending endpoint is not torque enabled')
        if max(abs(sample['present_raw'][name] - sample['goal_raw'][name])
               for name in sample['present_raw']) > 6:
            raise RuntimeError('HOME pending endpoint Present-to-Goal residual is excessive')
        if diagnostics.get('errors'):
            raise RuntimeError('HOME pending endpoint actuator diagnostics incomplete')
        return dict(goal_raw=sample['goal_raw'], present_raw=sample['present_raw'],
                    goal_position=sample['goal_position'], present_position=sample['present_position'],
                    raw_readback_source='same_raw_register_read',
                    present_minus_goal=sample['present_minus_goal'],
                    actuator_snapshot=dict(
                        present_raw=sample['present_raw'], goal_raw=sample['goal_raw'],
                        present_position=sample['present_position'], goal_position=sample['goal_position'],
                        present_minus_goal=sample['present_minus_goal'],
                        moving=sample['moving'], torque=sample['torque'],
                        motor_register_writes=sample['motor_register_writes']),
                    actuator_diagnostics=diagnostics)

    q = np.asarray(final, dtype=np.float64)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('HOME final readback is not a finite six-vector')
    inverse = getattr(owner.io, 'calibration_reconstructed_encoder_cells', None)
    if callable(inverse):
        cells = np.asarray(inverse(q), dtype=np.int64)
    else:
        # Lightweight unit-test doubles may have no encoder model. This branch
        # is never used by the live owner and is retained as explicit evidence.
        cells = np.rint(q).astype(np.int64)
    if cells.shape != (6,):
        raise ValueError('HOME final encoder reconstruction is not six cells')
    raw = dict(zip(ORDER, (int(value) for value in cells)))
    position = dict(zip(ORDER, (float(value) for value in q)))
    return dict(goal_raw=dict(raw), present_raw=dict(raw), goal_position=dict(position),
                present_position=dict(position), raw_readback_source='calibration_reconstruction',
                present_minus_goal={name: 0.0 for name in ORDER})


def _source_binding(output):
    """Bind a pending HOME receipt to the exact failed execution file."""
    path = Path(output) / 'last-attempt.json'
    if not path.exists():
        return dict(source_time_s=None, source_sha256=None)
    raw = path.read_bytes()
    source = json.loads(raw)
    if not isinstance(source, dict):
        raise ValueError('HOME source execution receipt is not an object')
    return dict(source_time_s=source.get('source_time_s'),
                source_sha256=hashlib.sha256(raw).hexdigest())


def _persist_pending_home_receipt(owner, record, final, *, clock):
    """Publish one atomically written, still-requires-HOME physical receipt."""
    output = getattr(owner, 'output', None)
    if output is None:
        return None
    output = Path(output)
    from .v21_hardware_state import ORDER, _validate_home_command_journal
    state = _settled_register_state(owner, final)
    journal = _validate_home_command_journal(
        output, record.get('command_journal'),
        expected_commands=record.get('commands_position'),
        settled_state=record.get('settled_q_position'))
    binding = _source_binding(output)
    receipt = dict(
        schema=PENDING_HOME_RECEIPT_SCHEMA,
        kind='pending-home-physical-settled',
        requires_home=True,
        completed=True,
        physical_settled=True,
        raw_freeze_verified=False,
        model_transition=False,
        action_source='manual_home_recovery',
        source_time_s=binding['source_time_s'],
        source_sha256=binding['source_sha256'],
        settled_time_s=float(record['settled_time_s']),
        persisted_time_s=float(clock()),
        # This is the pre-HOLD final readback written into the immutable
        # completion journal. Keep it separate from the physical post-HOLD
        # snapshot below because a stopped actuator can differ by a few ticks.
        journal_settled_q_position=np.asarray(record['settled_q_position'],
                                              dtype=np.float64).tolist(),
        # The register snapshot is taken after the final HOLD, so its Present
        # value is the authoritative physical endpoint for reconnect.
        settled_q_position=np.asarray([state['present_position'][name] for name in ORDER],
                                      dtype=np.float64).tolist(),
        goal_raw=state['goal_raw'],
        present_raw=state['present_raw'],
        goal_position=np.asarray([state['goal_position'][name] for name in ORDER],
                                 dtype=np.float64).tolist(),
        present_position=np.asarray([state['present_position'][name] for name in ORDER],
                                    dtype=np.float64).tolist(),
        present_minus_goal=state['present_minus_goal'],
        raw_readback_source=state['raw_readback_source'],
        endpoint_outcome=record.get('endpoint_outcome'),
        policy_commands_returned=int(record['policy_commands_returned']),
        commands_position=record['commands_position'],
        command_journal=journal['path'],
        command_journal_sha256=journal['sha256'],
        command_journal_event_count=journal['event_count'],
        actuator_snapshot=state.get('actuator_snapshot'),
        actuator_diagnostics=state.get('actuator_diagnostics'),
    )
    from .v21_training import write_json
    write_json(output / PENDING_HOME_RECEIPT_NAME, receipt)
    record['pending_home_receipt_path'] = str((output / PENDING_HOME_RECEIPT_NAME).resolve())
    record['pending_home_receipt_schema'] = PENDING_HOME_RECEIPT_SCHEMA
    return receipt


def clear_pending_home_receipt(output):
    """Remove the pending HOME receipt only after the durable HOME success marker."""
    path = Path(output) / PENDING_HOME_RECEIPT_NAME
    if path.exists() or path.is_symlink():
        path.unlink()


def execute_home_readback_chunk(owner, chunk, *, period_s=.2, clock=time.monotonic):
    record=dict(kind='manual-home-readback-only',model_transition=False,
                completed=False,action_source='manual_home_recovery',
                policy_commands_attempted=0,policy_commands_returned=0,
                completion_hold=dict(attempted=False,returned=False),
                completion_journal_returned=False,
                commands_position=[],measured_q_path=[],readback_times_s=[],command_times_s=[])
    owner.last_home_attempt=record
    try:
        commands=np.asarray(chunk,dtype=np.float32)
        if commands.shape!=(5,6) or not np.isfinite(commands).all():
            raise ValueError('HOME chunk must be finite [5,6]')
        if not np.isfinite(period_s) or period_s<=0:
            raise ValueError('HOME period must be positive')
        commands=commands.copy()
    except BaseException as exc:
        # Even a rejected request must not expose the preceding HOME's success.
        # Input validation still performs no IO or recovery HOLD.
        record.update(error_type=type(exc).__name__,error=str(exc),failure_phase='validation')
        raise
    journal=None
    phase='dispatch'
    try:
        owner._lease_check()
        if owner.stop_event.is_set():raise RuntimeError('HOME cancelled before dispatch')
        current=owner.read_checked()
        owner._physical_check(current,commands)
        home_gate=home_clearance_gate(owner)
        home_gate.require_policy_path(current,commands)
        record['home_planned_clearance_mm']=home_gate.planned_clearance_mm
        record['hard_readback_clearance_mm']=home_gate.readback_clearance_mm
        record['measured_q_path'].append(np.asarray(current).tolist())
        record['readback_times_s'].append(float(clock()))
        journal=owner.io._start_command_journal(commands)
        record['command_journal'] = None if journal is None else str(Path(journal).resolve())
        previous=np.asarray(current,dtype=np.float32).copy()
        for index,command in enumerate(commands):
            owner._lease_check()
            if owner.stop_event.is_set():raise RuntimeError('HOME cancelled before dispatch')
            actual_q=owner.read_checked()
            remaining=np.concatenate((commands[index:],np.repeat(commands[-1:],index,axis=0)),axis=0)
            owner._physical_check(actual_q,remaining)
            home_gate.require_policy_path(actual_q,remaining)
            owner.io._append_command_journal(journal,dict(event='home_command_intent',
                command_index=index,monotonic_s=float(clock()),requested=command,previous_command=previous))
            if owner.stop_event.is_set():raise RuntimeError('HOME cancelled before dispatch')
            sent_at=float(clock())
            record['policy_commands_attempted']+=1
            returned=owner._write_q(command,previous)
            record['policy_commands_returned']+=1
            sent=np.asarray(returned,dtype=np.float32)
            record['commands_position'].append(sent.tolist())
            record['command_times_s'].append(sent_at)
            if sent.shape!=(6,) or not np.allclose(sent,command,atol=1e-4,rtol=0):
                raise RuntimeError('HOME transport changed the audited command')
            previous=sent.copy()
            # HOME retains its manual 0.2 s pacing. Never burst late commands
            # to catch up, and let STOP interrupt the wait.
            if owner.stop_event.wait(period_s):raise RuntimeError('HOME cancelled after dispatch')
            measured=owner.read_checked();stamp=float(clock())
            record['measured_q_path'].append(np.asarray(measured).tolist())
            record['readback_times_s'].append(stamp)
            owner.io._append_command_journal(journal,dict(event='home_command_result',
                command_index=index,monotonic_s=stamp,sent=sent,observed_state=measured))
        # Reuse the established tolerance, quiet-sample quorum, timeout, and
        # stall handling. It reads joints only; it never calls io.observe().
        phase='settlement'
        with owner.bus_lock:
            try:
                settlement=owner.io._settle_commanded_endpoint(commands[-1],period_s=period_s,
                    stop_event=owner.stop_event,journal_path=journal)
            finally:
                # This legacy routine can read or write a protective HOLD
                # directly. Its effects must not inherit the prior wrapper's
                # command identity or an apparent continuous velocity sample.
                from .v21_actuator_history import history_for
                history_for(owner).invalidate('home_direct_io_settlement',float(clock()))
        phase='final_readback'
        final=owner.read_checked()
        record.update(settled_q_position=np.asarray(final).tolist(),
                      settled_time_s=float(clock()),endpoint_outcome=settlement.outcome)
        phase='completion_hold'
        record['completion_hold']['attempted']=True
        owner._halt()
        record['completion_hold']['returned']=True
        phase='completion_journal'
        owner.io._append_command_journal(journal,dict(event='home_chunk_completed',
            monotonic_s=float(clock()),settled_state=final,endpoint_outcome=settlement.outcome))
        # Persist the physical endpoint before exposing this chunk as complete.
        # A crash after this point can therefore reconnect to the newest raw
        # Goal/Present while the outer HOME interlock remains requires_home.
        _persist_pending_home_receipt(owner, record, final, clock=clock)
        # Success includes the final verified HOLD, completion journal, and
        # durable pending receipt.
        record['completion_journal_returned']=True
        record['completed']=True
        return record
    except BaseException as exc:
        record.update(completed=False,error_type=type(exc).__name__,error=str(exc),failure_phase=phase)
        if phase=='completion_hold':
            record['completion_hold'].update(error_type=type(exc).__name__,error=str(exc))
        owner.stop_event.set()
        cleanup_hold=dict(attempted=True,returned=False)
        record['fault_cleanup_hold']=cleanup_hold
        try:
            owner._halt()
            cleanup_hold['returned']=True
        except BaseException as cleanup_exc:
            cleanup_hold.update(error_type=type(cleanup_exc).__name__,error=str(cleanup_exc))
            add_note=getattr(exc,'add_note',None)
            if callable(add_note):
                add_note(f'HOME fault HOLD failed: {type(cleanup_exc).__name__}: {cleanup_exc}')
        fault_journal=dict(attempted=True,returned=False)
        record['fault_journal']=fault_journal
        try:
            owner.io._append_command_journal(journal,dict(event='home_chunk_fault',completed=False,
                monotonic_s=float(clock()),sent_commands_confirmed=len(record['commands_position']),
                error_type=type(exc).__name__,error=str(exc),failure_phase=phase,
                fault_cleanup_hold=dict(cleanup_hold)))
            fault_journal['returned']=True
        except BaseException as cleanup_exc:
            fault_journal.update(error_type=type(cleanup_exc).__name__,error=str(cleanup_exc))
            add_note=getattr(exc,'add_note',None)
            if callable(add_note):
                add_note(f'HOME fault journal failed: {type(cleanup_exc).__name__}: {cleanup_exc}')
        # Cleanup failures remain evidence; the caller receives the first fault.
        raise
