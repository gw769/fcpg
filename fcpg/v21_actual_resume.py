"""Explicit resume of a saved, completed actual state; never resume pending motion.

Only tensors and ordinary metadata are serialized. Numerical replay checks are
interface checks, not task-success evidence or cryptographic provenance.
"""
from dataclasses import fields,is_dataclass
from copy import deepcopy
import json
from pathlib import Path

import torch

from .prod_v20_policy import ActualHistory
from .v21_core import ActualState,CONTRACT,VERSION,COMPATIBLE_VERSIONS
from .v21_memory import VisualMemory
from .v21_picture_memory import PictureMemory
from .v21_models import CommandTrace
from .v21_temporal_contract import prepare_progress_state


LEDGER_FIELDS=('active','trial_id','maximum_blocks','block_attempts',
    'policy_commands_attempted','policy_commands_returned','accounting_unknown',
    'pending_reservation','records')


PROGRESS_STR_FIELDS = {'progress_session_id', 'progress_origin'}
PROGRESS_STATE_FIELDS = PROGRESS_STR_FIELDS | {'feedback_readback_index', 'visual_feedback_readback_indices'}
PROGRESS_MEMORY_FIELDS = {'readback_indices', 'current_readback_index'}


def _progress_string(name, value):
    maximum = 96 if name == 'progress_session_id' else 256
    if (name not in PROGRESS_STR_FIELDS or not isinstance(value, str)
            or not 1 <= len(value) <= maximum or any(not 32 <= ord(c) <= 126 for c in value)):
        raise ValueError('invalid progress metadata string')
    return value


def pack(value, _field_name=None):
    if _field_name in PROGRESS_STR_FIELDS:
        return None if value is None else _progress_string(_field_name,value)
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if is_dataclass(value):
        return {field.name:pack(getattr(value,field.name),field.name) for field in fields(value)
                if not (isinstance(value,VisualMemory) and field.name=='camera_types'
                        and value.camera_types is None)}
    if value is None:return None
    if isinstance(value,str):return _progress_string(_field_name,value)
    raise ValueError('actual state contains an unsupported value')


def unpack_state(data,device):
    def decode(cls,value):
        expected={field.name for field in fields(cls)}
        optional=(PROGRESS_STATE_FIELDS | {'task_ids', 'pictures'} if cls is ActualState else
                  PROGRESS_MEMORY_FIELDS | {'camera_types'} if cls is VisualMemory else
                  {'ids', 'strides'} if cls is PictureMemory else set())
        if (not isinstance(value,dict) or set(value)-expected or (expected-set(value))-optional):
            raise ValueError('actual-state field contract differs')
        result={}
        for key in expected:
            item=value.get(key)
            if cls is ActualState and key in ('memory','history'):
                result[key]=decode(VisualMemory if key=='memory' else ActualHistory,item)
            elif cls is ActualState and key=='pictures' and item is not None:     # 17.167
                result[key]=decode(PictureMemory,item)
            elif cls is ActualState and key in PROGRESS_STR_FIELDS:
                result[key]=None if item is None else _progress_string(key,item)
            elif isinstance(item,torch.Tensor):result[key]=item.to(device)
            elif item is None and (key=='object_relative' or key in optional):result[key]=None
            else:raise ValueError('actual-state leaf must be a tensor')
        if cls is VisualMemory and result['camera_types'] is not None:
            from .v21_camera_types import validate_camera_types
            validate_camera_types(result['camera_types'],len(result['features']))
        return cls(**result)
    return decode(ActualState,data)


def ledger_view(budget):
    status=budget.status()
    if (not status['active'] or status['accounting_unknown'] or status['pending_reservation']
            or not status['records'] or not status['records'][-1].get('completed')):
        raise ValueError('resume requires known accounting after a completed auxiliary block')
    return {key:status[key] for key in LEDGER_FIELDS}


@torch.inference_mode()
def references(runtime,state):
    core=runtime.core
    state=prepare_progress_state(state)
    if any(module.training for module in (core,core.reader,core.actor,core.lewm)):
        raise ValueError('resume checks require frozen eval mode')
    old=core.reader._projection_cache
    try:
        g=core.reader(state.memory,state.q,state.time,state.history,task_ids=getattr(state,'task_ids',None))
        if not torch.allclose(g,state.grounding,atol=1e-4,rtol=0):
            raise ValueError('saved actual grounding does not match its bank')
        dt=torch.full((1,5),.1,device=state.q.device)
        trace=CommandTrace.policy(state.q[:,None].repeat(1,5,1),dt)
        actor=core.act(state,runtime.text_ids,trace.offsets).commands_position
        prediction=core.predict(state,trace,dt.double().cumsum(-1)).response.predicted_q_path
        if not torch.isfinite(actor).all() or not torch.isfinite(prediction).all():
            raise ValueError('nonfinite resume reference')
        return actor,prediction
    finally:
        core.reader._projection_cache=old


def validate_actual_response(response,state,ledger):
    """Bind optional physical telemetry to the exact completed history, not a forecast."""
    if response is None:return None
    if (not isinstance(response,dict) or response.get('completed') is not True
            or response.get('policy_commands_attempted')!=5 or response.get('policy_commands_returned')!=5
            or response.get('action_source')!=ledger['records'][-1].get('action_source')):
        raise ValueError('actual response is not the last completed ledger event')
    def tensor(key,shape):
        value=torch.as_tensor(response[key],dtype=torch.float64)
        if value.shape!=shape or not torch.isfinite(value).all():
            raise ValueError('invalid actual response '+key)
        return value
    commands=tensor('commands_position',(5,6))
    path=tensor('measured_q_path',(6,6))
    times=tensor('readback_times_s',(6,))
    end=tensor('endpoint_time_s',())
    if (not bool(state.history.valid[0,-1]) or bool((times.diff()<=0).any())
            or abs(float(end)-float(state.time[0]))>1e-6
            or abs(float(times[-1])-float(state.history_ends[0,-1]))>1e-6
            or not torch.allclose(commands,state.history.commands_position[0,-1].cpu().double(),atol=1e-5,rtol=0)
            or not torch.allclose(path,state.history.readbacks_position[0,-1].cpu().double(),atol=1e-5,rtol=0)):
        raise ValueError('actual response differs from saved actual history/time')
    pair=response.get('wrist_alignment')
    if pair is not None:
        for key in ('before','after'):
            row=pair[key];stamp=float(row['image_time_s']);qt=float(row['q_time_s'])
            matches=(state.memory.image_times[0,0].cpu()-stamp).abs()<1e-6
            matches &= state.memory.valid[0,0].cpu()
            q=torch.as_tensor(row['q_at_image'],dtype=torch.float64)
            if (q.shape!=(6,) or not torch.isfinite(q).all() or not matches.any()
                    or not stamp<=qt<=stamp+.05 or qt>float(state.time[0])
                    or not torch.allclose(q,state.memory.q_at_image[0,0].cpu().double()[matches][0],atol=1e-5,rtol=0)):
                raise ValueError('alignment observation differs from the actual memory exposure')
    return deepcopy(response)


def save_actual_resume(path,runtime,budget,model_output,last_execution=None):
    if not runtime.stopped or runtime.pending is not None or runtime.executing or runtime.state is None:
        raise ValueError('save only a stopped, committed actual state')
    ledger=ledger_view(budget)
    state=runtime.state
    if len(state.q)!=1 or not bool(state.feedback_available.all()):
        raise ValueError('a completed actual feedback event is required')
    actor,prediction=references(runtime,state)
    value=dict(kind='completed-actual-auxiliary-state',schema=1,version=VERSION,contract=CONTRACT,
        model_output=str(Path(model_output).resolve()),ledger=ledger,state=pack(state),
        text_ids=runtime.text_ids.detach().cpu().clone(),
        reference_actor_commands_not_executed=actor.detach().cpu(),
        reference_lewm_hold_forecast_not_actual=prediction.detach().cpu(),
        pending_motion_restored=False,controller_fit_restored=False,task_success_verified=False)
    value['last_execution']=validate_actual_response(last_execution,state,ledger)
    value['temporal_contract']=deepcopy(getattr(runtime.core,'temporal_contract_identity',None))
    path=Path(path);temporary=path.with_suffix('.tmp.pt')
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(value,temporary);temporary.replace(path)


def load_actual_resume(path,model_output):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if (not isinstance(value,dict) or value.get('kind')!='completed-actual-auxiliary-state' or value.get('schema')!=1
            or value.get('version') not in COMPATIBLE_VERSIONS or value.get('contract')!=CONTRACT
            or value.get('model_output')!=str(Path(model_output).resolve())
            or value.get('pending_motion_restored') is not False):
        raise ValueError('incompatible actual-state resume file')
    return value


@torch.inference_mode()
def resume_actual_state(value,runtime,budget,endpoint,*,pose_tolerance):
    """Fresh dual-view/readback validation precedes clearing the restart interlock."""
    if not runtime.stopped or runtime.pending is not None or runtime.executing:
        raise ValueError('resume requires a stopped runtime')
    ledger=ledger_view(budget)
    if value.get('ledger')!=ledger:
        raise ValueError('saved actual state and current budget differ')
    if value.get('temporal_contract')!=getattr(runtime.core,'temporal_contract_identity',None):
        raise ValueError('actual-state resume temporal compatibility profile differs')
    state=unpack_state(value['state'],endpoint.q.device)
    validate_actual_response(value.get('last_execution'),state,ledger)
    if not torch.equal(value['text_ids'].to(runtime.text_ids),runtime.text_ids):
        raise ValueError('resume task tokens differ')
    if (not isinstance(pose_tolerance,(int,float)) or not 0<pose_tolerance<=3.
            or endpoint.observation is None or not bool(endpoint.observation.measurement_available.all())
            or endpoint.time.dtype!=torch.float64 or not torch.isfinite(endpoint.q).all()
            or not torch.isfinite(endpoint.time).all() or bool((endpoint.time<state.time).any())
            or float((endpoint.q-state.q).abs().max())>pose_tolerance):
        raise ValueError('resume needs fresh dual views and a matched physical readback: '+json.dumps(dict(
            saved_q=state.q.detach().cpu().tolist(),actual_q=endpoint.q.detach().cpu().tolist(),
            maximum_pose_difference=float((endpoint.q-state.q).abs().max()),
            allowed_pose_difference=pose_tolerance,
            view_available=None if endpoint.observation is None else endpoint.observation.measurement_available.cpu().tolist(),
            saved_time=state.time.cpu().tolist(),actual_time=endpoint.time.cpu().tolist())))
    actor,prediction=references(runtime,state)
    errors={}
    for name,actual,key in (
            ('actor',actor,'reference_actor_commands_not_executed'),
            ('lewm',prediction,'reference_lewm_hold_forecast_not_actual')):
        expected=value[key].to(actual)
        if expected.shape!=actual.shape or not torch.isfinite(expected).all():
            raise ValueError('invalid numerical resume reference')
        errors[name]=float((actual-expected).abs().max())
        if errors[name]>1e-4:raise ValueError('model no longer reproduces the saved actual state')
    runtime._physical(endpoint.q,endpoint.q[:,None].repeat(1,5,1))
    previous=runtime.state
    try:
        runtime.state=state
        runtime.observe(endpoint)  # Actual new image/readback, never a retimestamped old image.
        if ledger_view(budget)!=ledger:raise ValueError('budget changed during actual-state restore')
        budget.accept_actual_restore(ledger)
    except BaseException:
        runtime.state=previous;runtime.stop();raise
    return dict(restored=True,trial_id=ledger['trial_id'],block_attempts=ledger['block_attempts'],
        history_chunks=int(runtime.state.history.valid.sum()),numeric_replay_max_error=errors,
        controller_fit_restored=False,pending_motion_restored=False,task_success_verified=False)
