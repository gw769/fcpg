"""Bounded measured tracking correction for the high-clearance view experiment.

This is engineering assistance, not a learned Actor change or motor tuning.
Only a completed, settled physical view step can supply an offset. Geometry
and joint checks must be repeated on the actual commands after correction.
"""
import numpy as np


def tracking_evidence(q,previous,*,pose_tolerance=.25):
    if previous is None or previous.get('action_source')!='visual_alignment_view_step':
        return None
    q=np.asarray(q,np.float64)
    follow=previous.get('normal_follow') or {}
    commands=np.asarray(previous.get('commands_position'),np.float64)
    measured=np.asarray(follow.get('positions'),np.float64)
    times=np.asarray(follow.get('times_s'),np.float64)
    if (q.shape!=(6,) or not np.isfinite(q).all()
            or previous.get('completed') is not True
            or previous.get('policy_commands_attempted')!=5 or previous.get('policy_commands_returned')!=5
            or follow.get('hold_eligible', follow.get('goal_reached')) is not True or follow.get('quiet_samples',0)<3
            or commands.shape!=(5,6) or measured.ndim!=2 or measured.shape[1:]!=(6,) or len(measured)<3
            or times.shape!=(len(measured),) or not np.isfinite(commands).all()
            or not np.isfinite(measured).all() or not np.isfinite(times).all()
            or (np.diff(times)<=0).any()):
        raise ValueError('tracking correction requires five real commands and three settled readbacks')
    if (not 0<pose_tolerance<=.25 or np.abs(q-measured[-1]).max()>pose_tolerance
            or np.abs(np.diff(measured[-3:],axis=0)).max()>pose_tolerance
            or np.abs(commands[-1]-measured[-1]).max()>3.):
        raise ValueError('tracking correction requires a matched quiet physical endpoint')
    plan=(previous.get('view_step') or {}).get('plan') or {}
    target=plan.get('nominal_commands_position')
    if target is None:
        rows=plan.get('waypoints') or []
        target=[row.get('q_position') for row in rows]
    nominal=np.asarray(target,np.float64)
    if nominal.shape!=(5,6) or not np.isfinite(nominal).all():
        raise ValueError('tracking correction has no genuine nominal view endpoint')
    offset=np.zeros(6);offset[1:4]=(commands[-1]-measured[-1])[1:4]
    error=float(np.abs(nominal[-1,1:4]-q[1:4]).max())
    recovering=error>.5
    if recovering and error>3.:
        raise ValueError('nominal endpoint error exceeds the bounded recovery range')
    if recovering and plan.get('stage')=='tracking_recovery':
        # A view waypoint is not a contact target. Do not repeat/integrate the
        # failed correction, but do not turn its precision into a new physical
        # veto: the caller may plan a NEW high-clearance path from actual q.
        return dict(usable_offset=False,recovering=False,
                    reason='discard_inexact_recovery_offset_replan_from_actual',nominal_error_position=error)
    return dict(offset=offset,nominal_target=nominal[-1].copy(),recovering=recovering,
                nominal_error_position=error,source_endpoint_time_s=float(previous['endpoint_time_s']),
                measured_follow_time_s=float(times[-1]))


def compensate(q,nominal,evidence,lower,upper,*,maximum_command_delta=6.):
    """Ramp one measured shortfall; never accumulate it or alter J1/J5/J6."""
    q=np.asarray(q,np.float64);nominal=np.asarray(nominal,np.float64).copy()
    lower=np.asarray(lower,np.float64);upper=np.asarray(upper,np.float64)
    if (q.shape!=(6,) or nominal.shape!=(5,6) or lower.shape!=(6,) or upper.shape!=(6,)
            or not np.isfinite(np.r_[q,nominal.ravel(),lower,upper]).all()
            or not 0<maximum_command_delta<=6. or np.any(lower>=upper)
            or not np.allclose(nominal[:,[0,4,5]],q[[0,4,5]],atol=1e-6,rtol=0)):
        raise ValueError('invalid fixed-jaw nominal view commands')
    offset=np.asarray(evidence['offset'],np.float64)
    if offset.shape!=(6,) or not np.isfinite(offset).all() or np.abs(offset).max()>3. or np.any(offset[[0,4,5]]):
        raise ValueError('invalid measured shortfall')
    ramp=np.linspace(.2,1.,5)[:,None]
    def legal(commands):
        return (np.all(commands>=lower) and np.all(commands<=upper)
                and np.abs(np.diff(np.vstack((q,commands)),axis=0)).max()<=maximum_command_delta+1e-9)
    if not legal(nominal):raise ValueError('nominal view path violates joint/step bounds')
    gain=1.
    if not legal(nominal+ramp*offset):
        lo,hi=0.,1.
        for _ in range(40):
            mid=(lo+hi)/2
            if legal(nominal+ramp*offset*mid):lo=mid
            else:hi=mid
        gain=lo
    issued=nominal+ramp*offset*gain
    return issued,dict(kind='bounded_measured_endpoint_shortfall',gain=gain,
        measured_offset_position=offset.tolist(),applied_offset_position=(offset*gain).tolist(),
        source_endpoint_time_s=evidence['source_endpoint_time_s'],
        measured_follow_time_s=evidence['measured_follow_time_s'],
        nominal_error_before_position=evidence['nominal_error_position'],
        nominal_commands_position=nominal.tolist(),motor_parameters_changed=False,
        learned_policy=False,physical_improvement_verified=False)
