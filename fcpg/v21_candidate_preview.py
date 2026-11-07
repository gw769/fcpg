"""Read-only integration of FCPG state, goal-free LeWM and Actor selection."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from .prod_v20_policy import ActualHistory, _validate_history
from .v21_candidate_evaluation import VisualGoal, evaluate_candidates
from .v21_lewm_shadow import _json_sha, _tensor_record
from .v21_models import CommandTrace
from .so101_camera_calibration import predict_projection
from .v21_tip_display import tip_source_matches_view


def snapshot_feedback_source(service, endpoint):
    """Export committed feedback with a separate fresh q; never form ActualState.

    There are no observation, planning, model or actuator calls here. The five
    history tensor records use the exact shadow-packet serialization, including
    genuine scene_pooled values and original dtypes, so history hashes compare.
    """
    rt = service.owner.runtime
    if not rt.stopped or rt.executing or rt.pending is not None:
        raise RuntimeError('feedback source snapshot requires STOP and no pending command')
    state = rt.state
    if state is None:
        raise RuntimeError('feedback source snapshot requires committed actual history')
    _validate_history(state.history, 1)

    def record(value, shape, name):
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or not torch.isfinite(value).all()):
            raise ValueError(f'feedback source {name} must be a finite tensor of shape {shape}')
        return _tensor_record(value.detach().cpu().clone())

    fresh_q = record(endpoint.q, (1, 6), 'fresh q')
    fresh_time = record(endpoint.time, (1,), 'fresh time')
    committed_q = record(state.q, (1, 6), 'committed q')
    committed_time = record(state.time, (1,), 'committed time')
    history_ends = record(state.history_ends, (1, 3), 'history ends')
    if (not endpoint.q.is_floating_point() or not endpoint.time.is_floating_point()
            or (endpoint.time < state.time.to(endpoint.time)).any()):
        raise ValueError('fresh measured endpoint must not precede committed state time')
    history = {field.name: _tensor_record(getattr(state.history, field.name).detach().cpu().clone())
               for field in fields(ActualHistory)}
    feedback_shapes = dict(q_error=(1, 5, 6), feedback_available=(1,),
                           feedback_time=(1,), feedback_offsets=(1, 5), feedback_retimed=(1,))
    actor_feedback = {name: record(getattr(state, name), shape, name)
                      for name, shape in feedback_shapes.items()}
    if rt.state is not state or not rt.stopped or rt.executing or rt.pending is not None:
        raise RuntimeError('feedback source changed while its read-only snapshot was collected')
    stamp = time.time_ns()
    payload = dict(
        schema='v21-feedback-source-observation-v1',
        source_semantics='committed actual history + fresh measured q, not full current ActualState',
        executable=False, model_state_updated=False, action_dispatched=False,
        position_unit='lerobot_calibrated_position', created_at_unix_ns=stamp,
        history=history, history_sha256=_json_sha(history), history_ends=history_ends,
        fresh_endpoint=dict(q=fresh_q, time=fresh_time),
        committed_state=dict(q=committed_q, time=committed_time),
        actor_feedback=actor_feedback,
        actor_feedback_source='unchanged committed state; not recomputed at the fresh endpoint',
    )
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    output = Path(service.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/f'feedback-source-observation-{stamp}.json'
    with path.open('xb') as handle:
        handle.write(raw)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                history_sha256=payload['history_sha256'], schema=payload['schema'],
                executable=False, model_state_updated=False)


def preview_actor_candidates(service):
    rt = service.owner.runtime
    if not rt.stopped or rt.executing or rt.pending is not None:
        raise RuntimeError('candidate preview requires STOP and no pending command')
    before = rt.state
    endpoint = service.owner.capture()
    feedback_snapshot = snapshot_feedback_source(service, endpoint)
    service._refresh_status()
    service._publish()
    tip = (service.fingertips or {}).get('column') or {}
    target = ((service.targets.get('targets') or {}).get('pickup') or {}).get('column') or {}
    unavailable = dict(executable=False, action_dispatched=False, selected=None, candidates=[],
                       available=False, reason=None, observation_sequence=service.sequence,
                       feedback_source_snapshot=feedback_snapshot)
    if (not tip_source_matches_view('column', tip) or not tip.get('available')
            or not target.get('visible')):
        return dict(unavailable, reason='current column-rgb-jaw-contour target or midpoint unavailable')
    if service.grasp_target != 'body':
        return dict(unavailable, reason='head target must be supplied explicitly; body C cannot stand in for head C')
    record = None

    def pose(q):
        # Relative geometric readout only. No inverse solve or command modification.
        kin = service.owner.kinematics
        world, local = kin.forward_anchor_frames(np.asarray(q, np.float32)[None])
        px = np.asarray(predict_projection(service._column_fit, world, local, np.asarray([q[-1]])))[0]
        return dict(jaw_axis_px=px[2]-px[1], height_m=float(kin.forward_anchors(q)[0,2]))

    def build(state, offsets):
        nonlocal record
        q = state.q[0].detach().cpu().numpy()
        capture_q = np.asarray(tip['q_at_image'], dtype=float)
        raw = rt.core.act(state, rt.text_ids, offsets).commands_position
        project = service._column_tip_px
        # Initialize/read existing projection; absence is an explicit failure.
        if project(capture_q) is None:
            raise ValueError('relative camera projection unavailable')
        pair = np.asarray(tip['fm_px'], dtype=float)
        goal = VisualGoal(tuple(target['cog_px']), tuple(tip['m_px']), tuple(capture_q),
            float(target['image_time_s']), float(tip['image_time_s']), float(state.time[0]),
            source=target.get('source'), geometry_validated=False, contact_verified=False,
            body_axis_px=tuple(target['axis_px']), observed_jaw_axis_px=tuple(pair[1]-pair[0]),
            observed_height_m=pose(capture_q)['height_m'],
            target_height_m=float(service._column_grasp_target_xyz()[2]))
        def predict(actual_q, commands):
            if not np.allclose(actual_q, q, atol=0, rtol=0):
                raise ValueError('candidate source q changed')
            trace = CommandTrace(torch.as_tensor(commands,device=state.q.device,dtype=state.q.dtype)[None],
                offsets, torch.ones_like(offsets,dtype=torch.bool), 'planned', True)
            response = rt.core.predict(state,trace,offsets+.1).response
            return response.predicted_q_path[0].cpu().numpy(), response.q_uncertainty[0].cpu().numpy()
        record = evaluate_candidates(q, raw[0].cpu().numpy(), goal,
            predict=predict, mechanical_check=service.owner.bridge.physical_check, project=project, pose=pose)
        record.update(available=True,observation_sequence=service.sequence,
            feedback_source_snapshot=feedback_snapshot,
            target_source=target.get('source'),midpoint_source=tip.get('source'),
            midpoint_accuracy_verified=bool(tip.get('accuracy_verified')),
            height_source='kinematic_prior_unverified',
            reason='preview only: real relative-motion and height validation are not established',
            ownership=dict(FCPG='state and spatial errors',LeWM='goal-free q response',Actor='proposal and internal selection'))
        # This return serves the existing non-executable preview contract only.
        # It is NOT selected or admitted for dispatch.
        return state.q[:,None].expand(-1,5,-1).clone()
    build.wants_command_offsets = True
    try:
        with torch.inference_mode():
            rt.preview(endpoint, command_builder=build)
    except (ValueError, RuntimeError, KeyError, TypeError, AttributeError) as exc:
        record = dict(unavailable, reason=f'{type(exc).__name__}: {exc}')
    if rt.state is not before or rt.pending is not None or rt.executing or not rt.stopped:
        raise RuntimeError('candidate preview violated actual-memory or STOP isolation')
    if record is None:
        record = dict(unavailable, reason='candidate preview returned no evaluation')
    return record
