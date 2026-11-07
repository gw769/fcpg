"""Local visual-alignment experiment using one existing motion owner.

No hardware constructors, independent serial handles, target-coordinate input,
or hidden grasp/height success decisions exist here. The injected runner must
enforce the same physical and five-command clock checks as other motion.
"""
import numpy as np
import torch

from .v21_visual_alignment import fit_measured_response,propose_alignment


def retain_unchanged_arm_goals(state,commands):
    """Holding an axis means holding its sent setpoint, not copying servo sag."""
    history=getattr(state,'history',None)
    if history is None or not bool(history.valid[0,-1]):return commands,[]
    previous=history.commands_position[0,-1,-1].to(commands)
    result=commands.clone();kept=[]
    for axis in (1,2,3):  # Never alter the selected pan, wrist roll, or gripper.
        if (bool(torch.all((result[0,:,axis]-state.q[0,axis]).abs()<1e-6))
                and bool(torch.isfinite(previous[axis]))
                and float((previous[axis]-state.q[0,axis]).abs())<=3.):
            result[0,:,axis]=previous[axis];kept.append(axis+1)
    return result,kept


def observation_payload(value):
    return dict(q_at_image=value.q_position.tolist(),image_time_s=value.image_time_s,
        q_time_s=value.q_time_s,target_center_px=value.target_center_px.tolist(),
        target_long_axis_px=value.target_axis_px.tolist(),
        body_truncated=value.body_truncated,
        target_short_axis_px=[-float(value.target_axis_px[1]),float(value.target_axis_px[0])],
        jaw_fm_reference_px=value.jaw_fm_px.tolist(),
        jaw_reference_source='existing-wrist-projection-not-V21-pixel-head',
        alignment_error=value.error.tolist(),angle_unit='radians',executable=False,
        physical_grasp_verified=False)


class AlignmentSession:
    def __init__(self,measure,run_motion,*,lower,upper,gripper_tolerance,probe_amplitude=2.):
        self.measure,self.run_motion=measure,run_motion
        self.lower=np.asarray(lower,np.float64);self.upper=np.asarray(upper,np.float64)
        if self.lower.shape!=(6,) or self.upper.shape!=(6,) or not np.isfinite([self.lower,self.upper]).all():
            raise ValueError('physical joint limits are required')
        if not np.isfinite(probe_amplitude) or not 0<probe_amplitude<=2.:
            raise ValueError('probe amplitude must be positive and at most two calibrated-position units')
        self.gripper_tolerance=float(gripper_tolerance);self.probe_amplitude=float(probe_amplitude)
        self.pairs=[];self.response=None;self.return_q=None;self.return_neighbourhood=None

    @staticmethod
    def _interpolate(state,goal):
        end=torch.as_tensor(goal,device=state.q.device,dtype=state.q.dtype)[None,None]
        fraction=torch.arange(1,6,device=state.q.device,dtype=state.q.dtype)[None,:,None]/5
        return state.q[:,None]+fraction*(end-state.q[:,None])

    def _collect(self,builder,source):
        observations=[]
        held_axes=[]
        self.response=None  # Even a partial/failed motion invalidates the old local fit.
        def bind(state):
            before=self.measure()
            observations.append(before)
            commands,kept=retain_unchanged_arm_goals(state,builder(state,before))
            held_axes.append(kept)
            return commands
        motion=self.run_motion(bind,source)
        if motion.get('completed') is not True:
            raise RuntimeError('incomplete motion cannot enter the visual response fit')
        after=self.measure()  # Actual post-transaction image/readback, not the prediction.
        before=observations[-1]
        self.pairs.append((before,after));self.pairs=self.pairs[-40:]
        self.response=None
        return dict(action_source=source,motion_completed=bool(motion['completed']),
            held_previous_actual_command_axes=held_axes[-1],
            before=observation_payload(before),after=observation_payload(after),
            model_response={key:motion.get(key) for key in ('source_time_s','command_times_s',
                'readback_times_s','commands_position','measured_q_path','retimed_predicted_q_path','q_path_residual')},
            external_task_before=motion.get('external_task_before'),
            external_task_after=motion.get('external_task_after'),
            response_pairs=len(self.pairs),physical_grasp_verified=False),before,after

    def probe(self,joint,direction):
        if type(joint) is not int or not 0<=joint<5 or type(direction) is not int or direction not in (-1,1):
            raise ValueError('probe needs arm joint 0..4 and direction -1 or +1')
        origin=[]
        def build(state,_before):
            q=state.q.detach().cpu().numpy()[0].copy();origin.append(q.copy())
            q[joint]+=direction*self.probe_amplitude
            if (q<self.lower).any() or (q>self.upper).any():raise ValueError('probe exceeds joint limits')
            return self._interpolate(state,q)
        self.return_q=None;self.return_neighbourhood=None
        result,_,after=self._collect(build,'visual_alignment_probe')
        self.return_q=origin[-1];self.return_neighbourhood=after.q_position.copy()
        return result

    def return_from_probe(self):
        if self.return_q is None:raise ValueError('no completed probe has a measured return pose')
        goal=self.return_q.copy();near=self.return_neighbourhood.copy()
        def build(state,_before):
            q=state.q.detach().cpu().numpy()[0]
            if (np.abs(q[:5]-near[:5])>self.probe_amplitude).any() or abs(q[5]-near[5])>self.gripper_tolerance:
                raise ValueError('robot moved outside the last probe neighbourhood')
            return self._interpolate(state,goal)
        result,_,_=self._collect(build,'visual_alignment_return')
        self.return_q=None;self.return_neighbourhood=None
        return result

    def fit(self):
        self.response=fit_measured_response(self.pairs,unmodelled_gripper_tolerance=self.gripper_tolerance)
        r=self.response
        return dict(response_pairs=r.sample_count,jacobian=r.jacobian.tolist(),
            residual_rms=r.residual_rms.tolist(),measured_support=r.measured_support.tolist(),
            reference_q=r.reference_q.tolist(),observed_radius=r.observed_radius.tolist(),
            design_condition=r.design_condition,physical_response_verified=False)

    def step(self):
        if self.response is None:raise ValueError('fit a measured local response before alignment motion')
        response=self.response;proposal=[]
        def build(state,before):
            # A stale image-time pose must not quietly become the command source.
            if np.max(np.abs(state.q.detach().cpu().numpy()[0]-before.q_position))>self.gripper_tolerance:
                raise ValueError('image and command-source readbacks disagree')
            planned=propose_alignment(before,response,self.lower,self.upper)
            proposal.append(planned)
            return torch.as_tensor(planned['commands_position'],device=state.q.device,dtype=state.q.dtype)[None]
        result,before,after=self._collect(build,'visual_alignment_servo')
        weights=np.array([1.,1.,before.jaw_span_px/2.])
        observed=float(np.linalg.norm(weights*after.error))
        start=float(np.linalg.norm(weights*before.error))
        noise=float(np.linalg.norm(weights*response.residual_rms))
        result.update(proposal=proposal[-1],observed_weighted_error=observed,
                      improvement_exceeds_fit_residual=bool(start-observed>2*noise),
                      physical_grasp_verified=False)
        # A step is never silently repeated on its old matrix. Refit/inspect the
        # actual response before the next command; a worsening step stays stopped.
        return result
