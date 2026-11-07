#!/usr/bin/env python3
"""Prepare J3 P16/P32 comparison; command-line execution is always offline.

The callback state machine constructs no bus, owner, service or live endpoint.
An independently admitted caller must inject existing-owner callbacks and
explicitly enable each phase. This helper never clears STOP or issues HOME.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import threading
import uuid
from typing import Callable

MOTOR = 'elbow_flex'
MOTORS = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
PID = ('P_Coefficient','I_Coefficient','D_Coefficient')
PROTECTED = ('Torque_Enable','Lock','Operating_Mode','Max_Torque_Limit','Torque_Limit',
    'Protection_Current','Overload_Torque','Protective_Torque','CW_Dead_Zone','CCW_Dead_Zone',
    'Minimum_Startup_Force','Homing_Offset','Min_Position_Limit','Max_Position_Limit')
IDENTITY = ('ID','Model_Number','Firmware_Major_Version','Firmware_Minor_Version')
RESTORE_GUARD = (*IDENTITY,'Torque_Enable','Lock','Operating_Mode')
REQUIRED_REGISTERS = (*IDENTITY,*PID,*PROTECTED,'Goal_Position','Present_Position','Moving')
HOME_PATH = Path('/home/james/feedback-corrected-grounding-v203/configs/prod_v17_7_legacy_v17_4_home.json')
ARTIFACT_ROOT = Path('/home/james/so101-vda-disk/feedback-corrected-grounding/outputs/lerobot/so101/v21-full-pipeline-repair-20260919')
PREFLIGHT_PATH = ARTIFACT_ROOT/'j3-pid-readonly-preflight/summary.json'
IDENTITY_SOURCE_PATH = ARTIFACT_ROOT/'j3-pid-live-adapter/readonly-summary.json'
MANUAL = 'https://www.feetechrc.com/Data/feetechrc/upload/file/20220618/%E5%85%88%E7%9C%8B%E8%BF%99%E9%87%8C-%E5%85%A5%E6%89%8B%E6%95%99%E7%A8%8B2-1.pdf'
MANUAL_SHA = '1db8cf7421b61c45f8d271ef8bb6fa58bb8fbbaa5e307a1b63b3c705ad328438'
ACTUAL_IDENTITY = dict(ID=3,Model_Number=777,Firmware_Major_Version=3,Firmware_Minor_Version=10)
ACTUAL_BASELINE = dict(P_Coefficient=16,I_Coefficient=0,D_Coefficient=0,Torque_Enable=1,Lock=1,
    Operating_Mode=0,Max_Torque_Limit=1000,Torque_Limit=1000,Protection_Current=310,
    Overload_Torque=80,Protective_Torque=20,CW_Dead_Zone=1,CCW_Dead_Zone=1,
    Minimum_Startup_Force=16,Homing_Offset=1743,Min_Position_Limit=920,Max_Position_Limit=3174)
MATCH_TOLERANCE = .25
HOME_TOLERANCE = 3.
FRESHNESS_S = .1
MAX_LATENESS_S = .03


def _vector(values,length,name):
    if not isinstance(values,(list,tuple)) or len(values)!=length:
        raise ValueError(name+': wrong shape')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        raise ValueError(name+': finite numeric values required')
    return [float(v) for v in values]


def j3_targets(source):
    q=_vector(source,6,'source q')
    if any(not -100<=v<=100 for v in q[:5]) or not 0<=q[5]<=100:
        raise ValueError('source outside original normalized joint bounds')
    result=[]
    for offset in (-1.5,-3.,-3.,-3.,-3.):
        row=q.copy();row[2]+=offset
        if not -100<=row[2]<=100:raise ValueError('diagnostic leaves original joint bounds')
        result.append(row)
    return result


def _config(snapshot):
    return dict(j3={k:snapshot['registers'][k] for k in (*IDENTITY,*PID,*PROTECTED)},
                other_motors=deepcopy(snapshot['other_motor_configuration']))


def _error(error):return f'{type(error).__name__}: {error}'


class J3PIDProtocolError(RuntimeError):pass
class J3PIDCancelled(J3PIDProtocolError):pass


@dataclass(frozen=True)
class J3PIDCallbacks:
    """Seven existing-owner callbacks; disabled phases invoke none.

    read_snapshot returns fresh q/clocks, raw J3 registers and other-motor
    configuration. read_restore_guard(*,include_p=False) reads only J3 identity,
    Torque/Lock/mode and optionally P; unrelated telemetry cannot block P16.
    write_p writes ONLY elbow_flex P, normalize=False,num_retry=0. The adapter
    rechecks the minimal guard inside that same bus lock/lease and allows P16
    restoration despite cancellation; it never bypasses identity/Torque/Lock.
    execute_five retains original geometry/Goal verification/clock/STOP handling.
    It is responsible for the separately admitted diagnostic arm operation.
    persist must durably commit the motion_sealed marker and return {saved:True}.
    Outer ordinary-policy admission MUST honor this marker until independently
    verified restoration. This helper never clears or releases the marker.
    """
    read_snapshot: Callable
    write_p: Callable
    stop_hold: Callable
    execute_five: Callable
    cancelled: Callable
    persist: Callable
    read_restore_guard: Callable | None = None


class J3PIDExperiment:
    """One baseline and one variant, with no automatic movement retry or reset."""
    def __init__(self,callbacks=None,*,expected_identity=None,firmware_applicability=None,
                 clock_fn=time.monotonic,home_q=None):
        self.callbacks=callbacks
        self.expected_identity=deepcopy(expected_identity)
        self.firmware_applicability=deepcopy(firmware_applicability)
        self.clock=clock_fn
        self.home_q=_vector(home_q if home_q is not None else json.loads(HOME_PATH.read_text())['current_state'],6,'HOME')
        self.phase='new';self.motion_sealed=True;self.reserved_commands=0
        self.baseline_source=None;self.baseline_configuration=None;self.reports=[]
        self.session_id=uuid.uuid4().hex
        self._operation_lock=threading.Lock()

    def baseline(self,*,enable=False):return self._phase('baseline',enable=enable)
    def variant(self,*,enable=False):return self._phase('variant',enable=enable)

    def require_policy_resume(self):
        raise J3PIDProtocolError('session remains sealed; external admission must verify restoration')

    def _check_contract(self):
        if not isinstance(self.callbacks,J3PIDCallbacks) or any(
                not callable(getattr(self.callbacks,k)) for k in J3PIDCallbacks.__dataclass_fields__):
            raise TypeError('seven explicit existing-owner callbacks are required')
        if self.expected_identity!=ACTUAL_IDENTITY or any(type(v) is not int for v in self.expected_identity.values()):
            raise ValueError('pin the measured J3 ID3/model777/firmware3.10 before enabling')
        proof=self.firmware_applicability
        if (not isinstance(proof,dict) or proof.get('reviewed') is not True
                or proof.get('manual_sha256')!=MANUAL_SHA
                or not isinstance(proof.get('basis'),str) or not proof['basis'].strip()):
            raise ValueError('explicit manufacturer-family applicability review is required; firmware P32 is not proven')

    def _cancel(self):
        if self.callbacks.cancelled():raise J3PIDCancelled('diagnostic cancelled; helper will not clear STOP')

    def _persist(self,report,event):
        report['events'].append(dict(event=event,time_s=float(self.clock())))
        report['motion_sealed']=True
        receipt=self.callbacks.persist(deepcopy(report))
        if not isinstance(receipt,dict) or receipt.get('saved') is not True:
            raise J3PIDProtocolError('durable sealed intent was not acknowledged by persist callback')

    def _read(self,report,label,*,recovery=False):
        if not recovery:self._cancel()
        started=float(self.clock());check=dict(label=label,call_started_s=started)
        report['read_checks'].append(check)
        try:snapshot=self.callbacks.read_snapshot()
        except BaseException as error:
            check.update(call_failed_s=float(self.clock()),error=_error(error));raise
        returned=float(self.clock());check.update(call_returned_s=returned,snapshot=deepcopy(snapshot))
        if not recovery:self._cancel()
        if not isinstance(snapshot,dict):raise ValueError('snapshot must be a mapping')
        q=_vector(snapshot.get('q_position'),6,'snapshot q')
        stamps=_vector([snapshot.get('started_s'),snapshot.get('q_time_s'),snapshot.get('returned_s')],3,'snapshot clocks')
        if (not started<=stamps[0]<=stamps[1]<=stamps[2]<=returned
                or returned-stamps[1]>FRESHNESS_S+1e-9 or stamps[2]-stamps[0]>FRESHNESS_S+1e-9):
            raise ValueError('snapshot is stale or actual read clocks are inconsistent')
        regs=snapshot.get('registers')
        if not isinstance(regs,dict) or any(k not in regs or type(regs[k]) is not int for k in REQUIRED_REGISTERS):
            raise ValueError('snapshot requires raw integer identity/PID/protected/held-pose registers')
        if any(regs[k]!=self.expected_identity[k] for k in IDENTITY):
            raise ValueError('observed J3 model or firmware differs from pinned identity')
        if regs['Torque_Enable']!=1 or regs['Lock']!=1 or regs['Operating_Mode']!=0:
            raise ValueError('require Torque1/Lock1/position mode; no torque/unlock fallback')
        others=snapshot.get('other_motor_configuration')
        if not isinstance(others,dict) or set(others)!=set(MOTORS)-{MOTOR}:
            raise ValueError('snapshot requires other motors configuration evidence')
        for values in others.values():
            if (not isinstance(values,dict) or set(values)!=set((*PID,*PROTECTED))
                    or any(type(v) is not int for v in values.values())
                    or (not recovery and values['Torque_Enable']!=1)):
                raise ValueError('other motor configuration incomplete or not energized')
        snapshot=deepcopy(snapshot);snapshot['q_position']=q
        return snapshot

    def _ready(self,snapshot,pid,*,paired=True):
        if snapshot.get('owner_idle') is not True or snapshot.get('exclusive_owner') is not True:
            raise ValueError('existing owner must be idle and exclusively admitted')
        if snapshot.get('source_geometry_validated') is not True:
            raise ValueError('actual source must pass the original readback geometry gate')
        regs=snapshot['registers']
        if [regs[k] for k in PID]!=pid:raise ValueError('PID differs from declared phase; no baseline repair')
        if any(regs[k]!=value for k,value in ACTUAL_BASELINE.items() if k not in PID):
            raise ValueError('J3 protected configuration differs from measured baseline')
        if regs['Moving']!=0 or abs(regs['Goal_Position']-regs['Present_Position'])>1:
            raise ValueError('fresh Goal/Present must be within one raw tick and Moving=0')
        q=snapshot['q_position']
        if max(abs(a-b) for a,b in zip(q,self.home_q))>HOME_TOLERANCE:
            raise ValueError('fresh source is outside existing HOME-near tolerance')
        if paired and self.baseline_source is not None and max(abs(a-b) for a,b in zip(q,self.baseline_source))>MATCH_TOLERANCE:
            raise ValueError('variant source differs by more than 0.25; no automatic HOME or comparison')
        j3_targets(q)

    def _read_restore_guard(self,report,label,*,include_p=False):
        """Read only facts needed for a P16 write, even if full telemetry failed."""
        started=float(self.clock());check=dict(label=label,call_started_s=started,minimal_restore_guard=True)
        report['read_checks'].append(check)
        try:snapshot=self.callbacks.read_restore_guard(include_p=include_p)
        except BaseException as error:
            check.update(call_failed_s=float(self.clock()),error=_error(error));raise
        returned=float(self.clock());check.update(call_returned_s=returned,snapshot=deepcopy(snapshot))
        if not isinstance(snapshot,dict):raise ValueError('restore guard must be a mapping')
        read_started,read_returned=_vector([snapshot.get('started_s'),snapshot.get('returned_s')],2,'restore guard clocks')
        if (not started<=read_started<=read_returned<=returned
                or returned-read_started>FRESHNESS_S+1e-9):
            raise ValueError('minimal restore guard is stale or has inconsistent clocks')
        if snapshot.get('exclusive_owner') is not True:
            raise ValueError('minimal restore guard requires the existing exclusive owner lease')
        regs=snapshot.get('registers');required=(*RESTORE_GUARD,'P_Coefficient') if include_p else RESTORE_GUARD
        if not isinstance(regs,dict) or any(type(regs.get(k)) is not int for k in required):
            raise ValueError('minimal restore guard lacks raw J3 identity/Torque/Lock/mode/P evidence')
        if any(regs[k]!=self.expected_identity[k] for k in IDENTITY):
            raise ValueError('minimal restore guard identity differs from pinned J3')
        if (regs['Torque_Enable'],regs['Lock'],regs['Operating_Mode'])!=(1,1,0):
            raise ValueError('minimal restore guard requires Torque1/Lock1/position mode')
        return snapshot

    def _restore_p(self,report,before):
        failure=None
        def record_failure(field,error):
            nonlocal failure
            report[field]=_error(error)
            if failure is None:failure=error;report['rollback_error']=_error(error)
            report['rollback_failed_s']=float(self.clock())
        try:self._read_restore_guard(report,'restore_guard')
        except BaseException as error:
            record_failure('restore_guard_error',error);return failure
        report['p_restore_started_s']=float(self.clock())
        try:
            self.callbacks.write_p(before['registers']['P_Coefficient'])
            report['p_restore_returned_s']=float(self.clock())
        except BaseException as error:record_failure('p_restore_write_error',error)
        # A failed ACK can still have applied P16. Verify independently, without
        # repeating the write and without depending on another motor's reads.
        try:
            restored=self._read_restore_guard(report,'after_P16',include_p=True)
            if restored['registers']['P_Coefficient']!=before['registers']['P_Coefficient']:
                raise ValueError('independent J3 P read did not verify original P16')
            report['p_restored_verified']=True
        except BaseException as error:record_failure('p_restore_verification_error',error)
        try:
            final=self._read(report,'after_restore',recovery=True)
            self._same_config(final,_config(before),16)
            report['p_restored_verified']=report['configuration_restored_verified']=True
        except BaseException as error:record_failure('configuration_verification_error',error)
        return failure

    @staticmethod
    def _same_config(snapshot,original,p):
        expected=deepcopy(original);expected['j3']['P_Coefficient']=p
        if _config(snapshot)!=expected:
            raise ValueError('PID/protected/other-motor configuration changed; only J3 P may differ')

    def _still_fresh(self,snapshot):
        age=float(self.clock())-snapshot['q_time_s']
        if not 0<=age<=FRESHNESS_S+1e-9:
            raise ValueError('source became stale before action; no retimestamping or retry')

    def _read_after_intent(self,report,label,source,pid):
        """Take one compulsory actual sample after logging, without replanning.

        Disk latency may age the frozen intent. A new bus read must still show
        that exact pose and configuration; an old sample never gets a new time.
        A mismatch or failed read aborts this phase instead of trying again.
        """
        current=self._read(report,label)
        self._same_config(current,_config(source),pid[0])
        self._ready(current,pid,paired=report['phase']=='variant')
        if max(abs(a-b) for a,b in zip(current['q_position'],source['q_position']))>1e-4:
            raise ValueError('actual source changed after durable intent; frozen commands are not regenerated')
        return current

    def _drive(self,report,snapshot,phase):
        self._cancel()
        if self.reserved_commands+5>10:raise J3PIDProtocolError('ten-command experiment budget exhausted')
        commands=j3_targets(snapshot['q_position']);self.reserved_commands+=5
        report.update(commands_position=deepcopy(commands),source=deepcopy(snapshot),
            source_role='frozen_command_intent',commands_reserved=self.reserved_commands)
        self._persist(report,'five_command_intent')
        dispatch=self._read_after_intent(report,'before_dispatch',snapshot,[16 if phase=='baseline' else 32,0,0])
        report['dispatch_source']=deepcopy(dispatch)
        self._cancel();self._still_fresh(dispatch)
        report['drive_attempted']=True;report['drive_started_s']=float(self.clock())
        result=self.callbacks.execute_five(deepcopy(commands),phase)
        report.update(drive_returned=True,drive_result=deepcopy(result),drive_returned_s=float(self.clock()))
        self._cancel();self._validate_drive(result,dispatch,commands,
            callback_started_s=report['drive_started_s'],callback_returned_s=report['drive_returned_s'])
        report['five_command_receipt_validated']=True
        path=result['measured_q_path'];endpoint=result.get('endpoint_q_position',path[-1])
        peak=max(0.,*(dispatch['q_position'][2]-q[2] for q in [*path,endpoint]))
        report.update(actual_J3_directional_peak=peak,actual_response_verified=peak>.25)
        self._persist(report,'five_command_result')

    @staticmethod
    def _validate_drive(result,snapshot,commands,*,callback_started_s,callback_returned_s):
        if not isinstance(result,dict):raise ValueError('driver returned no transaction evidence')
        for key in ('policy_commands_attempted','policy_commands_returned','policy_goal_reads_attempted',
                    'policy_goal_reads_returned','policy_goal_registers_verified'):
            if type(result.get(key)) is not int or result[key]!=5:
                raise ValueError('driver did not verify exactly five writes/readbacks: '+key)
        if result.get('completed') is not True or result.get('protective_hold_after_q5') is not True:
            raise ValueError('driver lacks original complete transaction and HOLD')
        if result.get('original_geometry_admitted') is not True:
            raise ValueError('driver must attest unchanged original geometry admission')
        if result.get('maximum_lateness_s')!=MAX_LATENESS_S or result.get('normal_follow',{}).get('budget_s')!=.4:
            raise ValueError('driver changed deadline or following budget')
        sent=result.get('commands_position');path=result.get('measured_q_path')
        if not isinstance(sent,list) or len(sent)!=5 or not isinstance(path,list) or len(path)!=6:
            raise ValueError('driver changed original command/readback sample count')
        sent=[_vector(row,6,'sent commands') for row in sent];path=[_vector(row,6,'q0..q5') for row in path]
        if any(abs(a-b)>1e-4 for want,actual in zip(commands,sent) for a,b in zip(want,actual)):
            raise ValueError('driver changed diagnostic commands')
        if max(abs(a-b) for a,b in zip(path[0],snapshot['q_position']))>1e-4:
            raise ValueError('source moved before dispatch; no matched comparison')
        ct=_vector(result.get('command_times_s'),5,'command clocks')
        wt=_vector(result.get('write_completed_times_s'),5,'write completion clocks')
        qt=_vector(result.get('readback_times_s'),6,'readback clocks')
        late=_vector(result.get('write_lateness_s'),5,'write lateness')
        callback_started_s,callback_returned_s=_vector(
            [callback_started_s,callback_returned_s],2,'driver callback clocks')
        if not snapshot['q_time_s']<=callback_started_s<=qt[0]<=qt[-1]<=callback_returned_s:
            raise ValueError('transaction evidence does not belong to this driver callback interval')
        if (any(a>=b for a,b in zip(ct,ct[1:])) or any(a>=b for a,b in zip(qt,qt[1:]))
                or any(not qt[i]<=ct[i]<=wt[i]<qt[i+1] for i in range(5))
                or min(late)<0 or max(late)>MAX_LATENESS_S
                or any(abs(late[i]-(wt[i]-(qt[0]+i*.1)))>1e-6 for i in range(5))
                or any(not -1e-9<=qt[i]-(qt[0]+i*.1)<=MAX_LATENESS_S+1e-9 for i in range(1,6))):
            raise ValueError('driver violated actual clock ordering or original lateness')
        if 'endpoint_q_position' in result:
            _vector(result['endpoint_q_position'],6,'endpoint q');stamp=result.get('endpoint_time_s')
            if (isinstance(stamp,bool) or not isinstance(stamp,(int,float)) or not math.isfinite(stamp)
                    or not qt[-1]<=stamp<=callback_returned_s):
                raise ValueError('endpoint clock is outside original q5/current callback interval')
        goals=result.get('policy_goal_readbacks')
        if not isinstance(goals,list) or len(goals)!=5:raise ValueError('missing five individual Goal readbacks')
        for i,event in enumerate(goals):
            if (not isinstance(event,dict) or type(event.get('command_index')) is not int
                    or event.get('command_index')!=i or event.get('goal_register_verified') is not True
                    or event.get('schema')!='v21-policy-goal-readback-v1'
                    or event.get('register')!='Goal_Position' or event.get('normalize') is not False
                    or event.get('read_attempted') is not True or event.get('read_returned') is not True
                    or type(event.get('num_retry')) is not int or event['num_retry']!=0
                    or type(event.get('motor_register_writes')) is not int or event['motor_register_writes']!=0
                    or not isinstance(event.get('goal_raw'),dict)
                    or set(event['goal_raw'])!=set(MOTORS) or any(type(v) is not int for v in event['goal_raw'].values())
                    or not isinstance(event.get('expected_goal_raw'),dict)
                    or any(type(v) is not int for v in event['expected_goal_raw'].values())
                    or event['goal_raw']!=event.get('expected_goal_raw')):
                raise ValueError('individual raw Goal evidence incomplete')
            requested=event.get('requested_goal_position')
            if not isinstance(requested,dict) or set(requested)!=set(MOTORS):
                raise ValueError('Goal receipt lacks the requested six-axis command')
            requested=_vector([requested[name] for name in MOTORS],6,'Goal requested command')
            if any(abs(a-b)>1e-4 for a,b in zip(requested,commands[i])):
                raise ValueError('Goal receipt belongs to a different requested command')
            read_started,read_returned,finished,io_lateness=_vector([
                event.get('read_started_s'),event.get('read_returned_s'),
                event.get('callback_returned_s'),event.get('policy_io_lateness_s')],4,'Goal read clocks')
            if (not wt[i]<=read_started<=read_returned<=finished<=qt[i+1]
                    or not 0<=io_lateness<=MAX_LATENESS_S
                    or abs(io_lateness-(finished-(qt[0]+i*.1)))>1e-6):
                raise ValueError('Goal verification exceeded original deadline or used inconsistent clocks')

    def _phase(self,phase,*,enable):
        if type(enable) is not bool:raise TypeError('enable must be an explicit bool')
        if not enable:
            return dict(schema='v21-j3-pid-phase-dry-run-v2',phase=phase,dry_run=True,
                        callbacks_invoked=0,protocol=build_protocol())
        if not self._operation_lock.acquire(blocking=False):
            raise J3PIDProtocolError('another diagnostic phase already owns this session')
        try:return self._enabled_phase(phase)
        finally:self._operation_lock.release()

    def _enabled_phase(self,phase):
        self._check_contract()
        expected='new' if phase=='baseline' else 'baseline_recorded'
        if self.phase!=expected:raise J3PIDProtocolError('phase unavailable or prior attempt sealed: '+self.phase)
        self.phase=phase+'_in_progress'
        report=dict(schema='v21-j3-pid-phase-result-v2',session_id=self.session_id,
            phase=phase,dry_run=False,motion_sealed=True,
            expected_identity=deepcopy(self.expected_identity),events=[],read_checks=[],
            firmware_applicability=deepcopy(self.firmware_applicability),
            p_write_intent_persisted=False,p_write_attempted=False,p_write_returned=False,
            drive_attempted=False,drive_returned=False,five_command_receipt_validated=False,
            actual_response_verified=False,primary_error=None,stop_error=None,rollback_error=None,
            persistence_error=None,p_restored_verified=False,configuration_restored_verified=False)
        primary=None;before=None;stop_error=None;restore_error=None
        try:
            self._cancel();self._persist(report,'session_sealed')
            before=self._read(report,'before');self._ready(before,[16,0,0],paired=phase=='variant')
            original=_config(before);report['before_configuration']=deepcopy(original)
            if phase=='variant':
                self._same_config(before,self.baseline_configuration,16)
                report['p_may_have_been_written']=True
                self._persist(report,'P32_write_intent');report['p_write_intent_persisted']=True
                p_source=self._read_after_intent(report,'before_P32_write',before,[16,0,0])
                report['p_write_source']=deepcopy(p_source)
                self._cancel();self._still_fresh(p_source)
                report['p_write_attempted']=True;report['p_write_started_s']=float(self.clock())
                try:self.callbacks.write_p(32)
                except BaseException:
                    report['p_write_failed_s']=float(self.clock());raise
                report.update(p_write_returned=True,p_write_returned_s=float(self.clock()));self._cancel()
                after=self._read(report,'after_P32');self._same_config(after,original,32);self._ready(after,[32,0,0])
                if max(abs(a-b) for a,b in zip(after['q_position'],before['q_position']))>MATCH_TOLERANCE:
                    raise ValueError('gain change moved source by more than 0.25; no diagnostic motion')
                before_drive=after
                # Freeze this proof and its exact source/commands together.
                # The mandatory post-intent read establishes dispatch freshness.
                report['events'].append(dict(event='P32_readback_verified',time_s=float(self.clock())))
            else:before_drive=before
            self._drive(report,before_drive,phase)
        except BaseException as error:
            primary=error;report['primary_error']=_error(error)
            attempt=getattr(error,'last_attempt',None)
            if isinstance(attempt,dict):report['failed_drive_attempt']=deepcopy(attempt)
        finally:
            if report['p_write_attempted'] or report['drive_attempted']:
                report['stop_started_s']=float(self.clock())
                try:self.callbacks.stop_hold();report['stop_returned_s']=float(self.clock())
                except BaseException as error:stop_error=error;report['stop_error']=_error(error)
            if report['p_write_attempted']:
                restore_error=self._restore_p(report,before)
            elif before is not None and report['drive_attempted']:
                try:
                    final=self._read(report,'baseline_final',recovery=True);self._same_config(final,_config(before),16)
                    report['p_restored_verified']=report['configuration_restored_verified']=True
                except BaseException as error:restore_error=error;report['rollback_error']=_error(error)
            report['further_motion_blocked']=True
            clean=(primary is None and stop_error is None and restore_error is None
                   and report['five_command_receipt_validated'] and report['configuration_restored_verified'])
            self.phase=('baseline_recorded' if phase=='baseline' else 'finished') if clean else 'failed_sealed'
            report['session_state']=self.phase
            try:self._persist(report,'phase_finished_still_sealed')
            except BaseException as error:
                report['persistence_error']=_error(error);self.phase='failed_sealed';report['session_state']=self.phase
                if primary is None:primary=error;report['primary_error']=_error(error)
            if self.phase=='baseline_recorded':
                self.baseline_source=report['dispatch_source']['q_position'].copy()
                self.baseline_configuration=deepcopy(report['before_configuration'])
            self.reports.append(deepcopy(report))
        failure=primary or stop_error or restore_error
        if failure is not None:failure.j3_pid_report=deepcopy(report);raise failure
        return deepcopy(report)


class FakeJ3Bus:
    """Memory-only existing-owner fixture. Values are not physical results."""
    def __init__(self,*,overrides=None,ignore_candidate=False,fail_after_apply=False,fail_restore=False,hook=None):
        self.values=dict(**ACTUAL_IDENTITY,**ACTUAL_BASELINE,Goal_Position=3154,Present_Position=3154,Moving=0)
        self.values.update(overrides or {});self.persisted=deepcopy(self.values)
        self.initial_persisted=deepcopy(self.persisted)
        self.others={name:{k:ACTUAL_BASELINE[k] for k in (*PID,*PROTECTED)} for name in MOTORS if name!=MOTOR}
        self.q=json.loads(HOME_PATH.read_text())['current_state'];self.events=[];self.saved=[];self.now=10.
        self.cancelled_flag=False;self.ignore_candidate=ignore_candidate;self.fail_after_apply=fail_after_apply
        self.fail_restore=fail_restore;self.hook=hook;self.last_report=None

    def _event(self,kind,**values):
        event=dict(kind=kind,**values);self.events.append(event)
        if self.hook is not None:self.hook(self,event)

    def read_snapshot(self):
        self._event('snapshot');started=self.now;self.now+=.001
        return dict(started_s=started,q_time_s=self.now,returned_s=self.now,q_position=self.q.copy(),
            registers=deepcopy(self.values),other_motor_configuration=deepcopy(self.others),
            owner_idle=True,exclusive_owner=True,source_geometry_validated=True)

    def write_p(self,value):
        if type(value) is not int or value not in (16,32):raise AssertionError('only P16/P32 writes allowed')
        if self.values['Torque_Enable']!=1 or self.values['Lock']!=1:raise RuntimeError('torque-on locked-volatile precondition absent')
        self._event('write',register='P_Coefficient',motor=MOTOR,value=value,phase='candidate' if value==32 else 'restore')
        if value==16 and self.fail_restore:raise ConnectionError('injected restore transport failure')
        if not (value==32 and self.ignore_candidate):self.values['P_Coefficient']=value
        if value==32 and self.fail_after_apply:raise ConnectionError('injected write failure after value may have applied')

    def read_restore_guard(self,*,include_p=False):
        self._event('restore_guard',include_p=include_p);started=self.now;self.now+=.001
        keys=(*RESTORE_GUARD,'P_Coefficient') if include_p else RESTORE_GUARD
        return dict(started_s=started,returned_s=self.now,exclusive_owner=True,
            registers={key:self.values[key] for key in keys})

    def stop_hold(self):self._event('simulated_existing_STOP_HOLD',fake_only=True)

    def persist(self,report):
        self.last_report=deepcopy(report);self._event('persist',event=report['events'][-1]['event'])
        self.saved.append(deepcopy(report));return dict(saved=True)

    def execute_five(self,commands,phase):
        self._event('execute_five',phase=phase,commands=deepcopy(commands));epoch=self.now;self.now+=.9
        goals=[{name:int(2000+value*10) for name,value in zip(MOTORS,row)} for row in commands]
        return dict(fake_only=True,completed=True,policy_commands_attempted=5,policy_commands_returned=5,
            policy_goal_reads_attempted=5,policy_goal_reads_returned=5,policy_goal_registers_verified=5,
            protective_hold_after_q5=True,original_geometry_admitted=True,maximum_lateness_s=.03,
            normal_follow=dict(budget_s=.4),commands_position=deepcopy(commands),
            measured_q_path=[self.q.copy() for _ in range(6)],command_times_s=[epoch+i*.1+.001 for i in range(5)],
            write_completed_times_s=[epoch+i*.1+.002 for i in range(5)],
            readback_times_s=[epoch+i*.1 for i in range(6)],write_lateness_s=[.002]*5,
            policy_goal_readbacks=[dict(command_index=i,goal_register_verified=True,motor_register_writes=0,
                schema='v21-policy-goal-readback-v1',register='Goal_Position',normalize=False,
                read_attempted=True,read_returned=True,num_retry=0,
                requested_goal_position=dict(zip(MOTORS,commands[i])),
                read_started_s=epoch+i*.1+.002,read_returned_s=epoch+i*.1+.003,
                callback_returned_s=epoch+i*.1+.003,policy_io_lateness_s=.003,
                goal_raw=goals[i].copy(),expected_goal_raw=goals[i].copy()) for i in range(5)])

    def callbacks(self):
        return J3PIDCallbacks(self.read_snapshot,self.write_p,self.stop_hold,self.execute_five,
            lambda:self.cancelled_flag,self.persist,self.read_restore_guard)

    def session(self):
        return J3PIDExperiment(self.callbacks(),expected_identity=ACTUAL_IDENTITY,
            firmware_applicability=dict(reviewed=True,manual_sha256=MANUAL_SHA,basis='FAKE callback fixture; real energized P32 not validated'),
            clock_fn=lambda:self.now)


def fake_transaction(bus,*,inject_trial_error=False):
    if type(bus) is not FakeJ3Bus:raise TypeError('fake-only helper refuses every real or foreign bus object')
    session=bus.session()
    if inject_trial_error:
        previous=bus.hook
        def trial_hook(owner,event):
            if previous is not None:previous(owner,event)
            if event['kind']=='execute_five' and event['phase']=='variant':raise RuntimeError('injected operator STOP during trial')
        bus.hook=trial_hook
    error=None
    try:session.baseline(enable=True);session.variant(enable=True)
    except BaseException as caught:error=caught
    report=deepcopy(session.reports[-1] if session.reports else getattr(error,'j3_pid_report',{}))
    report.update(fake_only=True,hardware_used=False,live_execution_enabled=False,physical_response=None,
        candidate_attempted=report.get('p_write_attempted',False),
        trial_marker_reached=report.get('drive_attempted',False) and report.get('phase')=='variant',
        rollback_verified=report.get('configuration_restored_verified',False),
        sealed_if_rollback_unverified=not report.get('configuration_restored_verified',False),
        persistent_state_unchanged=bus.persisted==bus.initial_persisted,fake_events=deepcopy(bus.events))
    return report


def build_protocol():
    home=json.loads(HOME_PATH.read_text())
    return dict(schema='v21-j3-pid-static-response-offline-protocol-v2',stage='offline_preparation_only',
        executable=False,live_execution_enabled=False,hardware_accessed=False,
        preferred_first_comparison=dict(register='P_Coefficient',baseline=16,candidate=32,restored=16,
            unchanged=dict(I_Coefficient=0,D_Coefficient=0),physical_cause_identified=False,
            reason='P tests static corrective-drive hypothesis. D0 is the local fork intentional anti-shake baseline; upstream D32 is not automatically restored.'),
        measured_baseline=deepcopy(ACTUAL_BASELINE),measured_identity=deepcopy(ACTUAL_IDENTITY),
        measured_baseline_source=str(PREFLIGHT_PATH),
        measured_identity_source=str(IDENTITY_SOURCE_PATH),
        measured_identity_source_sha256=hashlib.sha256(IDENTITY_SOURCE_PATH.read_bytes()).hexdigest() if IDENTITY_SOURCE_PATH.exists() else None,
        register_contract=dict(motor=MOTOR,expected_id=3,expected_model_number=777,firmware_major=3,firmware_minor=10,
            Torque_Enable=dict(address=40,required_value=1,writes_forbidden=True),
            Lock=dict(address=55,required_value=1,writes_forbidden=True),
            only_writable_register=dict(name='P_Coefficient',address=21,bytes=1,sequence=[32,16]),
            I_address=23,D_address=22,other_configuration_writes_forbidden=True,
            storage='Lock1 temporary parameter at EPROM-mapped address; no unlock or EEPROM-save sequence',
            energized_P32_on_firmware_3_10_verified=False),
        callbacks=dict(read_snapshot='fresh q/clocks passing original readback geometry; J3 raw identity/PID/protected/Goal/Present/Moving and other motor config',
            read_restore_guard='keyword include_p=False: raw J3 identity/Torque/Lock/mode and actual clocks under exclusive owner; include_p=True also reads P; no other motors or q required',
            write_p='one J3 P write, original bus lock/lease, normalize=False,num_retry=0; recheck minimal guard within same lock; P16 independent of cancellation',
            stop_hold='existing STOP/HOLD, never torque-off',
            execute_five='original geometry/Goal-read/30ms/STOP checks; actual clocks bound to current callback, requested Goal to exact commands, num_retry=0; no retry, no HOME, no label retiming',
            cancelled='admission cancellation, never cleared by helper',
            persist='durably commit report+motion_sealed marker before action; return {saved:true}'),
        default_dry_run=dict(baseline_enable=False,variant_enable=False,callbacks_invoked=0),
        phase_order=['baseline_P16_once','external_source_realignment_if_needed','variant_P32_once','finally_P16_verified'],
        budgets=dict(max_test_chunks=2,max_test_policy_writes=10,normal_gain_writes=2,automatic_retries=0,
            automatic_HOME=False,automatic_STOP_clear=False,automatic_policy_resume=False),
        source_pose=dict(home_file=str(HOME_PATH),home_sha256=hashlib.sha256(HOME_PATH.read_bytes()).hexdigest(),
            nominal_home=home['current_state'],home_tolerance=HOME_TOLERANCE,max_snapshot_age_s=FRESHNESS_S,
            max_snapshot_span_s=FRESHNESS_S,paired_source_max_abs_difference=MATCH_TOLERANCE,
            after_durable_intent='one compulsory actual read; unchanged configuration and source required; no retry or replan',
            intent_dispatch_source_max_abs_difference=1e-4,
            source_record='source is frozen intent; dispatch_source is the separate true post-intent bus sample',
            Moving=0,max_raw_goal_present_gap=1),
        motion=dict(joint_index=2,relative_J3_commands=[-1.5,-3.,-3.,-3.,-3.],max_delta=3.,max_step=1.5,
            example_from_saved_HOME=j3_targets(home['current_state']),position_unit='lerobot_calibrated_position',
            command_offsets_s=[0.,.1,.2,.3,.4],original_q0_q5_unchanged=True,normal_follow_budget_s=.4,
            maximum_lateness_s=MAX_LATENESS_S,other_axes='hold fresh actual q; only J3 targets vary',
            geometry='original nominal plus single-axis-stall geometry in existing execute_five callback',
            acceptance='existing actual directional J3 response >0.25; no response stays false'),
        rollback=dict(intent_persisted_before_write=True,STOP_before_rollback=True,STOP_error_does_not_skip_P_restore=True,
            failed_ACK_may_have_written=True,restore_only_P=True,I_D_or_protected_drift='report; never silently write it back',
            readable_identity_torque1_lock1_required=True,full_snapshot_failure_does_not_block_minimal_P_restore=True,
            P_restoration_and_full_configuration_verified_separately=True,restoration_verified_only_from_final_reads=True,
            restore_failure_blocks_following_motion=True,normal_policy_unseal='independent outer admission only'),
        unresolved=['separate final approval for temporary gain outside validated baseline',
            'reviewed existing-owner callback integration honoring durable seal',
            'actual energized P32 behavior remains untested on firmware3.10'],
        sources=[dict(url=MANUAL,sha256=MANUAL_SHA,pages=[12,13],meaning='P increases static holding, D braking; Lock1 edits are nonpersistent'),
            dict(url='https://github.com/huggingface/lerobot/issues/3400',meaning='SO-ARM100 shoulder_pan anecdote, not J3 diagnosis')],
        evidence_boundary='No real callbacks are connected here. Fake tests prove code behavior, not safe motor dynamics or PID improvement.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'protocol.json').write_text(json.dumps(build_protocol(),indent=2,allow_nan=False))
    cases={name:fake_transaction(FakeJ3Bus(**options),inject_trial_error=trial_error)
        for name,options,trial_error in [('success',{},False),('trial_stop',{},True),
            ('write_applied_then_error',dict(fail_after_apply=True),False),('candidate_ignored',dict(ignore_candidate=True),False),
            ('restore_failure',dict(fail_restore=True),True),('torque_off_refused',dict(overrides={'Torque_Enable':0}),False),
            ('unlocked_refused',dict(overrides={'Lock':0}),False)]}
    (args.output/'fake-transaction-results.json').write_text(json.dumps(cases,indent=2,allow_nan=False))
    print(json.dumps(dict(output=str(args.output),protocol_executable=False,fake_cases=len(cases),hardware_used=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
