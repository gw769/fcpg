"""One measured J1 response, not a five-joint fit or a grasp controller.

This first auxiliary experiment only tests lateral image correction. It never
closes the gripper, claims head identity, or treats centering as a grasp.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .v21_alignment_session import AlignmentSession
from .v21_visual_alignment import AlignmentObservation,_array


def recorded_alignment_pair(payload):
    """Restore actual observations only, never a fitted response/controller state."""
    def observation(row):
        return AlignmentObservation(_array(row['q_at_image'],(6,),'q'),
            _array(row['target_center_px'],(2,),'target'),_array(row['target_long_axis_px'],(2,),'axis'),
            _array(row['jaw_fm_reference_px'],(2,2),'jaw'),float(row['image_time_s']),float(row['q_time_s']))
    return observation(payload['before']),observation(payload['after'])


def directional_proposal(before,after,current,lower,upper):
    """High-view engineering continuation of an observed improving direction.

    This is not an identified Jacobian: small unmodelled arm motion is logged,
    not attributed to J1. No closure, descent or other joint motion is proposed.
    The caller must re-audit high clearance and stop/reobserve after the block.
    """
    dq=after.q_position-before.q_position
    if (not after.image_time_s>before.image_time_s or current.image_time_s<after.image_time_s
            or abs(dq[0])<.5 or np.abs(dq[1:]).max()>1.
            or abs(before.error[0])-abs(after.error[0])<3.
            or before.error[0]*after.error[0]<=0
            or current.error[0]*after.error[0]<=0
            or np.abs(current.q_position-after.q_position).max()>1.
            or abs(current.error[0]-after.error[0])>8.):
        raise ValueError('no matched improving direction for high-view continuation')
    shift=after.target_center_px[0]-before.target_center_px[0]
    if abs(shift)<3. or abs(current.error[0])<8.:
        raise ValueError('no further coarse lateral reach is needed or observed')
    # Use a damped extrapolation, capped to eight units for the WHOLE block.
    # No low-clearance/contact behavior is enabled by this estimate.
    gain=shift/dq[0]
    delta=float(np.clip(-.6*current.error[0]/gain,-8.,8.))
    if delta*dq[0]<=0:raise ValueError('proposed direction differs from the observed improvement')
    q=current.q_position;end=q.copy()
    end[0]=np.clip(q[0]+delta,lower[0],upper[0])
    if abs(end[0]-q[0])<.05:raise ValueError('lateral reach exhausted joint range')
    commands=q+np.linspace(.2,1.,5)[:,None]*(end-q)
    return commands,dict(kind='directional_high_view_continuation',delta_j1=float(end[0]-q[0]),
        actual_previous_joint_delta=dq.tolist(),observed_shift_x_px=float(shift),
        error_before_x_px=float(current.error[0]),identified_jacobian=False,
        extrapolated_response=True,physical_grasp_verified=False)


@dataclass(frozen=True)
class LateralResponse:
    gain_px_per_position: float
    residual_rms_px: float
    reference_q: np.ndarray
    minimum_j1: float
    maximum_j1: float
    maximum_measured_delta: float
    sample_count: int

    def __post_init__(self):
        object.__setattr__(self,'reference_q',_array(self.reference_q,(6,),'lateral reference q'))


def fit_lateral_response(pairs, tolerance):
    if len(pairs) < 2:
        raise ValueError('lateral fit needs an actual probe and return observation')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid unmodelled-joint tolerance')
    x=[]; y=[]; positions=[]
    for before, after in pairs:
        if not isinstance(before,AlignmentObservation) or not isinstance(after,AlignmentObservation):
            raise TypeError('lateral fit requires paired actual alignment observations')
        if after.image_time_s <= before.image_time_s or after.q_time_s <= before.q_time_s:
            raise ValueError('actual observations must advance time')
        dq=after.q_position-before.q_position
        if np.max(np.abs(dq[1:])) > tolerance:
            raise ValueError('other joints moved; cannot attribute the image change to J1')
        if abs(dq[0]) <= max(tolerance, 1e-6):
            raise ValueError('actual J1 movement is insufficient; requested movement is not evidence')
        x.append(float(dq[0]))
        y.append(float(after.target_center_px[0]-before.target_center_px[0]))
        positions.extend((before.q_position, after.q_position))
    x=np.asarray(x); y=np.asarray(y)
    if not ((x > 0).any() and (x < 0).any()):
        raise ValueError('lateral fit needs actual motion in both directions')
    gain=float(x@y/(x@x))
    residual=float(np.sqrt(np.mean((y-gain*x)**2)))
    if (not np.isfinite([gain,residual]).all() or abs(gain) < 1e-6
            or not (x*y*gain > 0).all()
            or np.min(np.abs(y)) <= 2*residual):
        raise ValueError('lateral response direction is inconsistent or not above fit residual')
    positions=np.asarray(positions)
    if np.max(np.abs(positions[:,1:]-positions[0,1:])) > tolerance:
        raise ValueError('other joints changed configuration across the lateral fit')
    return LateralResponse(gain,residual,positions[0].copy(),float(positions[:,0].min()),
        float(positions[:,0].max()),float(np.abs(x).max()),len(x))


class LateralAlignmentSession(AlignmentSession):
    def advance(self):
        if not self.pairs:raise ValueError('an actual improving observation pair is required')
        previous=self.pairs[-1];proposals=[]
        def build(state,current):
            if np.max(np.abs(state.q[0].cpu().numpy()-current.q_position))>.25:
                raise ValueError('current image and dispatch pose differ')
            commands,diag=directional_proposal(*previous,current,self.lower,self.upper)
            proposals.append(diag)
            return torch.as_tensor(commands,device=state.q.device,dtype=state.q.dtype)[None]
        result,before,after=self._collect(build,'visual_alignment_reach')
        result.update(proposal=proposals[-1],actual_improvement_px=float(abs(before.error[0])-abs(after.error[0])),
                      physical_grasp_verified=False)
        return result

    def probe(self,joint,direction):
        if type(joint) is not int or joint != 0:
            raise ValueError('lateral experiment only probes J1')
        return super().probe(joint,direction)

    def fit(self):
        self.response=fit_lateral_response(self.pairs,self.gripper_tolerance)
        r=self.response
        return dict(scope='lateral_reacquisition_only',gain_px_per_position=r.gain_px_per_position,
            fit_residual_rms_px=r.residual_rms_px,response_pairs=r.sample_count,
            measured_j1_range=[r.minimum_j1,r.maximum_j1],
            physical_response_verified=False,physical_grasp_verified=False)

    def step(self):
        if not isinstance(self.response,LateralResponse):
            raise ValueError('fit an actual lateral response before correction')
        response=self.response; proposal=[]
        def build(state,before):
            q=state.q.detach().cpu().numpy()[0]
            if np.max(np.abs(q-before.q_position)) > self.gripper_tolerance:
                raise ValueError('image and command-source readbacks disagree')
            if np.max(np.abs(q[1:]-response.reference_q[1:])) > self.gripper_tolerance:
                raise ValueError('other joints left the measured lateral configuration')
            low=max(self.lower[0],response.minimum_j1)
            high=min(self.upper[0],response.maximum_j1)
            if q[0] < low-self.gripper_tolerance or q[0] > high+self.gripper_tolerance:
                raise ValueError('J1 is outside the measured lateral range')
            error=float(before.error[0])
            limit=.5*response.maximum_measured_delta
            delta=float(np.clip(-.5*error/response.gain_px_per_position,-limit,limit))
            goal=q.copy();goal[0]=np.clip(q[0]+delta,low,high)
            predicted=error+response.gain_px_per_position*(goal[0]-q[0])
            if abs(predicted) >= abs(error)-1e-6:
                raise ValueError('no lateral improvement inside measured and physical bounds')
            proposal.append(dict(error_before_x_px=error,setpoint_error_after_x_px=float(predicted),
                estimate_kind='command-setpoint-assuming-exact-tracking-not-LeWM',
                gain_px_per_position=response.gain_px_per_position,delta_j1=float(goal[0]-q[0]),
                scope='lateral_reacquisition_only',head_identity_verified=False))
            return self._interpolate(state,goal)
        result,before,after=self._collect(build,'visual_alignment_servo')
        improvement=abs(float(before.error[0]))-abs(float(after.error[0]))
        measured_delta=float(after.q_position[0]-before.q_position[0])
        observed_shift=float(after.target_center_px[0]-before.target_center_px[0])
        measured_response=response.gain_px_per_position*measured_delta
        result.update(proposal=proposal[-1],actual_error_before_x_px=float(before.error[0]),
            actual_error_after_x_px=float(after.error[0]),actual_improvement_px=improvement,
            improvement_exceeds_fit_residual=bool(improvement>2*response.residual_rms_px),
            y_error_before_after_px=[float(before.error[1]),float(after.error[1])],
            angle_error_before_after_rad=[float(before.error[2]),float(after.error[2])],
            exposure_paired_response=dict(actual_delta_j1=measured_delta,
                observed_target_shift_x_px=observed_shift,
                gain_times_actual_delta_x_px=measured_response,
                image_response_residual_x_px=observed_shift-measured_response,
                image_times_s=[before.image_time_s,after.image_time_s],
                q_times_s=[before.q_time_s,after.q_time_s],
                nominal_lewm_q5_comparison=False),
            physical_grasp_verified=False)
        return result
