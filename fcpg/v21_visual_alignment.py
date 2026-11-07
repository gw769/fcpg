"""Read-only short-axis alignment math; this module never owns robot IO.

The local response is fitted to ACTUAL joint/image changes, not requested
commands. A proposal is not a verified grasp or an executable authorization:
depth/contact, mechanical admission and fresh-pose binding remain separate.
"""
from dataclasses import dataclass

import numpy as np


def _array(value,shape,name):
    result=np.array(value,dtype=np.float64,copy=True)
    if result.shape!=shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite {shape}')
    result.setflags(write=False)
    return result


def _line_angle(angle):
    return .5*np.arctan2(np.sin(2.*angle),np.cos(2.*angle))


@dataclass(frozen=True)
class AlignmentObservation:
    q_position: np.ndarray
    target_center_px: np.ndarray
    target_axis_px: np.ndarray
    jaw_fm_px: np.ndarray
    image_time_s: float
    q_time_s: float
    body_truncated: bool | None = None

    @property
    def jaw_span_px(self):
        return float(np.linalg.norm(self.jaw_fm_px[1]-self.jaw_fm_px[0]))

    @property
    def scene_features(self):
        # The fixed Wrist camera moves with the arm. Arm-joint responses are
        # measured from the object's image motion; jaw opening is held fixed.
        return np.r_[self.target_center_px,np.arctan2(self.target_axis_px[1],self.target_axis_px[0])]

    @property
    def error(self):
        jaw=self.jaw_fm_px[1]-self.jaw_fm_px[0]
        a=self.target_axis_px
        b=np.array([-a[1],a[0]])
        angle=_line_angle(np.arctan2(b[1],b[0])-np.arctan2(jaw[1],jaw[0]))
        return np.r_[self.target_center_px-self.jaw_fm_px.mean(0),angle]


def alignment_observation(record,q_at_image,jaw_fm_px,*,image_time_s,q_time_s,
                          now_s,max_age_s,max_bracket_s=.05):
    """Bind a current registered pickup target to its actual image-time readback."""
    if (record.get('role')!='pickup' or record.get('view')!='wrist'
            or record.get('visible') is not True
            or record.get('view_source')!='current_observation'
            or record.get('source') not in ('frozen_dinov2_profile_prototype','operator_colour_body')):
        raise ValueError('a current observed Wrist pickup is required, never DROP/history')
    values=np.asarray([image_time_s,q_time_s,now_s,max_age_s,max_bracket_s],np.float64)
    if (not np.isfinite(values).all() or max_age_s<=0 or max_bracket_s<=0
            or not 0<=now_s-image_time_s<=max_age_s
            or not 0<=q_time_s-image_time_s<=max_bracket_s or q_time_s>now_s):
        raise ValueError('image/readback clocks are stale, future, or not paired')
    if not np.isclose(float(record.get('image_time_s',np.nan)),image_time_s,atol=1e-9,rtol=0):
        raise ValueError('target record belongs to a different image')
    truncated=record.get('body_truncated',record.get('support_clipped'))
    if truncated is not None and type(truncated) is not bool:
        raise ValueError('target completeness must be an observed boolean')
    q=_array(q_at_image,(6,),'actual q')
    center=_array(record.get('cog_px') if record.get('cog_px') is not None else record.get('centroid_px'),(2,),'target center')
    axis=_array(record.get('axis_px'),(2,),'target axis')
    fm=_array(jaw_fm_px,(2,2),'jaw reference')
    width=float(record.get('width_px',np.nan));length=float(record.get('length_px',np.nan))
    if not np.isfinite([width,length]).all() or width<=0 or length<=width+1e-6:
        raise ValueError('a nondegenerate long/short-axis region is required')
    minimum=np.array([0.,28.]);maximum=np.array([224.,196.])
    if (center<minimum).any() or (center>=maximum).any() or (fm<minimum).any() or (fm>=maximum).any():
        raise ValueError('target/jaw reference is outside the real letterboxed image content')
    norm=float(np.linalg.norm(axis))
    if norm<1e-8 or np.linalg.norm(fm[1]-fm[0])<1e-6:
        raise ValueError('axis or open-jaw reference is degenerate')
    return AlignmentObservation(q,center,_array(axis/norm,(2,),'unit axis'),fm,
                                float(image_time_s),float(q_time_s),truncated)


@dataclass(frozen=True)
class MeasuredImageResponse:
    jacobian: np.ndarray                  # [u, v, angle] / five actual arm positions
    reference_q: np.ndarray
    measured_support: np.ndarray
    observed_radius: np.ndarray
    residual_rms: np.ndarray
    design_condition: float
    sample_count: int


def fit_measured_response(pairs,*,unmodelled_gripper_tolerance):
    """Fit a local Wrist response from paired actual observations only.

    Returned residuals/conditioning are diagnostics, NOT physical validation.
    Reversal samples use actual achieved deltas, including cross-joint motion.
    """
    if not np.isfinite(unmodelled_gripper_tolerance) or unmodelled_gripper_tolerance<0:
        raise ValueError('gripper-motion tolerance must be declared and nonnegative')
    if len(pairs)<5:raise ValueError('at least five independent arm-motion observations are required')
    x=[];y=[];poses=[]
    for before,after in pairs:
        if not isinstance(before,AlignmentObservation) or not isinstance(after,AlignmentObservation):
            raise TypeError('response fit requires paired actual alignment observations')
        if after.image_time_s<=before.image_time_s or after.q_time_s<=before.q_time_s:
            raise ValueError('probe observations must advance actual time')
        dq=after.q_position-before.q_position
        if abs(dq[5])>unmodelled_gripper_tolerance:
            raise ValueError('gripper moved during an arm-response observation')
        dy=after.scene_features-before.scene_features
        dy[2]=_line_angle(dy[2])
        x.append(dq[:5]);y.append(dy);poses.extend((before.q_position,after.q_position))
    x=np.asarray(x);y=np.asarray(y)
    support=np.max(np.abs(x),axis=0)
    if (support<=1e-9).any():raise ValueError('an arm joint has no measured motion')
    design=x/support
    coefficient,_,rank,singular=np.linalg.lstsq(design,y,rcond=1e-6)
    if rank!=5:raise ValueError('actual arm probes are not independently identifiable')
    jacobian=coefficient.T/support
    reference=np.asarray(poses[0])
    radius=np.max(np.abs(np.asarray(poses)[:,:5]-reference[:5]),axis=0)
    residual=np.sqrt(np.mean((design@coefficient-y)**2,axis=0))
    return MeasuredImageResponse(_array(jacobian,(3,5),'measured response'),
        _array(reference,(6,),'reference q'),_array(support,(5,),'measured support'),
        _array(radius,(5,),'observed radius'),_array(residual,(3,),'fit residual'),
        float(singular[0]/singular[-1]),len(pairs))


def propose_alignment(observation,response,lower,upper,*,gain=.5,trust_fraction=.5,
                      relative_damping=1e-3):
    """Return an unexecuted five-step local correction, keeping the jaw open.

    Position is measured in pixels; angular error is scaled by half jaw span.
    In the ideal local linear model, 0 < gain <= 1 reduces the controllable
    error. This is not a claim of convergence for noisy/nonlinear hardware.
    """
    if (not np.isfinite([gain,trust_fraction,relative_damping]).all()
            or not 0<gain<=1 or not 0<trust_fraction<=1 or relative_damping<=0):
        raise ValueError('invalid local-servo gains')
    lower=_array(lower,(6,),'lower limits');upper=_array(upper,(6,),'upper limits')
    q=observation.q_position
    if (upper<=lower).any() or (q<lower).any() or (q>upper).any():
        raise ValueError('actual source pose violates the supplied joint limits')
    if (np.abs(q[:5]-response.reference_q[:5])>response.observed_radius+1e-9).any():
        raise ValueError('source pose is outside the measured local response region')
    weights=np.array([1.,1.,observation.jaw_span_px/2.])
    error=observation.error
    matrix=weights[:,None]*response.jacobian*response.measured_support[None]
    u,s,vt=np.linalg.svd(matrix,full_matrices=False)
    if len(s)!=3 or s[-1]<=s[0]*1e-6:raise ValueError('position and short-axis control are not observable')
    damping=relative_damping*s[0]
    normalized=vt.T@((s/(s*s+damping*damping))*(u.T@(-gain*weights*error)))
    delta=normalized*response.measured_support
    maximum=trust_fraction*response.measured_support
    local_lower=np.maximum(lower[:5],response.reference_q[:5]-response.observed_radius)
    local_upper=np.minimum(upper[:5],response.reference_q[:5]+response.observed_radius)
    alpha=1.
    for i,d in enumerate(delta):
        if abs(d)<=1e-12:continue
        room=local_upper[i]-q[i] if d>0 else q[i]-local_lower[i]
        alpha=min(alpha,float(maximum[i]/abs(d)),float(room/abs(d)))
    delta*=max(0.,alpha)
    predicted=error+response.jacobian@delta
    before=float(np.linalg.norm(weights*error));after=float(np.linalg.norm(weights*predicted))
    if before>1e-8 and after>=before-1e-8:raise ValueError('no predicted improvement inside the local/joint bounds')
    dq=np.r_[delta,0.]
    commands=q[None]+np.arange(1,6)[:,None]/5.*dq[None]
    if (commands<lower-1e-9).any() or (commands>upper+1e-9).any():
        raise ValueError('proposal escaped joint limits')
    return dict(executable=False,action_source='measured-dino-short-axis-alignment-proposal',
                source_q_position=q.tolist(),source_image_time_s=observation.image_time_s,
                commands_position=commands.tolist(),error_before=error.tolist(),
                predicted_error_after=predicted.tolist(),weighted_error_before=before,
                weighted_error_after=after,physical_grasp_verified=False)
