"""FCPG diagnostic: bounded real motion pairs, not policy/task success."""
from copy import deepcopy
import numpy as np
import torch

from .v21_tip_display import tip_source_matches_view


def compare_actual_pair(before, after, joint, project):
    q0, q1 = np.asarray(before['q'], float), np.asarray(after['q'], float)
    delta = q1-q0
    others = np.delete(delta, joint)
    result = dict(actual_joint_delta=delta.tolist(), direction_verified=False,
                  height_verified=False, physical_grasp_verified=False)
    if abs(delta[joint]) < .5 or np.max(np.abs(others)) > .25:
        return dict(result, reason='insufficient selected-joint motion or coupled readback drift')
    a, b = before['views']['column'], after['views']['column']
    if not a['available'] or not b['available'] or b['time'] <= a['time']:
        return dict(result, reason='no matched current Column observations')
    actual = np.asarray(b['M'])-np.asarray(a['M'])
    predicted = np.asarray(project(q1))-np.asarray(project(q0))
    if actual.shape != (2,) or predicted.shape != (2,) or not np.isfinite([actual,predicted]).all():
        return dict(result, reason='invalid relative projection')
    norms = np.linalg.norm(actual), np.linalg.norm(predicted)
    result.update(observed_delta_px=actual.tolist(), predicted_delta_px=predicted.tolist(),
                  absolute_delta_error_px=float(np.linalg.norm(actual-predicted)))
    if min(norms) < 2.:
        return dict(result, reason='motion below two-pixel diagnostic resolution')
    cosine = float(actual@predicted/(norms[0]*norms[1]))
    result.update(direction_cosine=cosine, direction_verified=cosine>.5,
                  reason='single-pair direction diagnostic only; not global calibration')
    return result


def measure(service):
    endpoint = service.owner.last_endpoint
    service._publish()
    result = dict(q=endpoint.q[0].cpu().tolist(), time=float(endpoint.time[0]), views={})
    targets = (service.targets.get('targets') or {}).get('pickup') or {}
    for index, view in enumerate(('wrist','column')):
        tip=(service.fingertips or {}).get(view) or {}
        target=targets.get(view) or {}
        stamp=float(endpoint.observation.image_times[0,index])
        available=(bool(endpoint.observation.measurement_available[0,index])
                   and tip_source_matches_view(view, tip)
                   and bool(tip.get('available')) and bool(target.get('visible')))
        available=available and abs(float(tip.get('image_time_s',-1))-stamp)<1e-6 and abs(float(target.get('image_time_s',-1))-stamp)<1e-6
        result['views'][view]=dict(available=available, time=stamp,
            C=deepcopy(target.get('cog_px') or target.get('centroid_px')),
            M=deepcopy(tip.get('m_px')), M_source=tip.get('source'), C_source=target.get('source'),
            observation_accuracy_verified=bool(tip.get('accuracy_verified')))
    return result


def run_probe(service, joint, direction, *, returning=False, amplitude=2.):
    if type(joint) is not int or joint not in (0,1,2) or type(direction) is not int or direction not in (-1,1):
        raise ValueError('diagnostic joint must be 0/1/2 and direction +/-1')
    if isinstance(amplitude, bool) or not isinstance(amplitude, (float,int)) or not np.isfinite(amplitude) or not 0 < amplitude <= 4.:
        raise ValueError('explicit diagnostic amplitude must be finite in (0,4] calibrated units')
    previous=getattr(service,'motion_validation_record',None)
    if returning and (not previous or not previous.get('return_pending')):
        raise RuntimeError('no completed diagnostic to return from')
    snapshots=[]
    def build(state):
        sample=measure(service)
        if not sample['views']['column']['available']:
            raise RuntimeError('no current Column C/M pair for diagnostic')
        if sample['views']['column']['C_source']!='operator_colour_body':
            raise RuntimeError('Column body lock missing; no diagnostic motion')
        snapshots.append(sample)
        q=state.q[0]
        goal=q.clone()
        if returning:
            actual=np.asarray(sample['q'])
            last=np.asarray(previous['after']['q'])
            if np.max(np.abs(actual-last))>1.:
                raise RuntimeError('left the measured diagnostic neighbourhood')
            goal[joint]=float(previous['before']['q'][joint])
        else:
            goal[joint]+=float(amplitude)*float(direction) # calibrated units, not degrees
        fraction=torch.arange(1,6,device=q.device,dtype=q.dtype)[None,:,None]/5
        return q[None,None]+fraction*(goal-q)[None,None]
    source='visual_alignment_return' if returning else 'visual_alignment_probe'
    execution=service._run_alignment_motion(build,source)
    after=measure(service)
    try:
        assessment=compare_actual_pair(snapshots[-1],after,joint,service._column_tip_px)
    except (ValueError,TypeError,RuntimeError,AttributeError) as exc:
        assessment=dict(direction_verified=False,height_verified=False,
                        reason=f'projection assessment unavailable: {type(exc).__name__}')
    report=dict(kind='FCPG-real-motion-validation',joint=joint,direction=direction,requested_amplitude=float(amplitude),
        before=snapshots[-1],after=after,return_pending=not returning,
        action_source=source,source_time_s=execution['source_time_s'],completed=execution.get('completed'),
        actual_commands=execution.get('commands_position'),actual_readbacks=execution.get('measured_q_path'),
        assessment=assessment,
        not_task_execution=True)
    service.motion_validation_record=report
    return report
