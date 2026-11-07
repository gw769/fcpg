"""Synthetic local-response math tests. Never open a camera or serial port."""
import numpy as np
import pytest

from fcpg.v21_visual_alignment import alignment_observation,fit_measured_response,propose_alignment


BASE_Q=np.array([0.,0.,0.,0.,0.,30.])
BASE_FEATURES=np.array([130.,135.,np.pi/2+.15])
J=np.array([[2.,0.,1.,0.,.2],[0.,3.,0.,1.,.3],[0.,0.,.02,-.01,.12]])
LOWER=np.array([-100.]*5+[0.])
UPPER=np.array([100.]*6)


def observation(q=BASE_Q,features=BASE_FEATURES,stamp=1.,**changes):
    record=dict(role='pickup',view='wrist',visible=True,view_source='current_observation',
        source='frozen_dinov2_profile_prototype',image_time_s=stamp,
        width_px=20.,length_px=40.,
        centroid_px=features[:2].tolist(),axis_px=[np.cos(features[2]),np.sin(features[2])])
    record.update(changes)
    return alignment_observation(record,q,[[90.,112.],[130.,112.]],
        image_time_s=stamp,q_time_s=stamp+.01,now_s=stamp+.02,max_age_s=.2)


def pairs(matrix=J):
    result=[]
    # Achieved motion includes cross-joint response. There is deliberately no
    # requested-command amplitude in the estimator's input.
    for i in range(5):
        for direction in (-1.,1.):
            delta=np.zeros(5);delta[i]=direction*(1.+i*.2)
            delta[(i+1)%5]=direction*.15
            q=BASE_Q.copy();q[:5]+=delta
            result.append((observation(stamp=2.*len(result)+1),
                observation(q,BASE_FEATURES+matrix@delta,2.*len(result)+2)))
    return result


def test_short_axis_not_long_axis_is_the_closing_direction():
    vertical=observation(features=np.array([110.,112.,np.pi/2]))
    horizontal=observation(features=np.array([110.,112.,0.]))
    np.testing.assert_allclose(vertical.error,[0,0,0],atol=1e-12)
    assert abs(horizontal.error[2])==pytest.approx(np.pi/2)


def test_axis_sign_flip_does_not_change_the_alignment_error():
    one=observation()
    two=observation(features=BASE_FEATURES+np.array([0.,0.,np.pi]))
    np.testing.assert_allclose(one.error,two.error,atol=1e-12)


@pytest.mark.parametrize('change',[
    {'role':'drop_zone'},{'view':'column'},{'visible':False},
    {'view_source':'historical_observation_not_live'},
    {'source':'manual_pixel'},{'image_time_s':.9}])
def test_noncurrent_or_wrong_target_cannot_create_a_reference(change):
    with pytest.raises(ValueError):observation(**change)


@pytest.mark.parametrize('change',[{'centroid_px':[110,210]},{'width_px':0.},{'length_px':20.}])
def test_padding_or_undefined_short_axis_cannot_be_a_grasp_reference(change):
    with pytest.raises(ValueError):observation(**change)


def test_response_uses_actual_achieved_cross_joint_motion():
    fit=fit_measured_response(pairs(),unmodelled_gripper_tolerance=0.)
    np.testing.assert_allclose(fit.jacobian,J,atol=1e-12)
    assert fit.sample_count==10 and np.max(fit.residual_rms)<1e-12


def test_rank_deficient_probes_are_rejected():
    sample=pairs()[0]
    with pytest.raises(ValueError,match='measured motion|identifiable'):
        fit_measured_response([sample]*10,unmodelled_gripper_tolerance=0.)


def test_unmodelled_gripper_motion_is_not_charged_to_an_arm_joint():
    samples=pairs();before,after=samples[0]
    q=after.q_position.copy();q[-1]+=.5
    samples[0]=(before,observation(q,after.scene_features,after.image_time_s))
    with pytest.raises(ValueError,match='gripper moved'):
        fit_measured_response(samples,unmodelled_gripper_tolerance=.1)


def test_proposal_reduces_local_error_without_closing_the_gripper():
    fit=fit_measured_response(pairs(),unmodelled_gripper_tolerance=0.)
    proposal=propose_alignment(observation(),fit,LOWER,UPPER)
    commands=np.asarray(proposal['commands_position'])
    assert commands.shape==(5,6) and np.isfinite(commands).all()
    np.testing.assert_array_equal(commands[:,5],np.repeat(BASE_Q[5],5))
    assert proposal['weighted_error_after']<proposal['weighted_error_before']
    assert not proposal['executable'] and not proposal['physical_grasp_verified']
    assert (np.abs(commands[-1,:5]-BASE_Q[:5])<=fit.measured_support*.5+1e-9).all()
    np.testing.assert_allclose(commands[0]-BASE_Q,(commands[-1]-BASE_Q)/5)


def test_joint_limits_are_not_replaced_by_visual_targets():
    fit=fit_measured_response(pairs(),unmodelled_gripper_tolerance=0.)
    lower=LOWER.copy();upper=UPPER.copy();lower[:5]=-.05;upper[:5]=.05
    proposal=propose_alignment(observation(),fit,lower,upper)
    commands=np.asarray(proposal['commands_position'])
    assert (commands>=lower-1e-9).all() and (commands<=upper+1e-9).all()


def test_outside_measured_neighbourhood_requires_new_evidence():
    fit=fit_measured_response(pairs(),unmodelled_gripper_tolerance=0.)
    q=BASE_Q.copy();q[0]=20
    with pytest.raises(ValueError,match='outside'):
        propose_alignment(observation(q),fit,LOWER,UPPER)


def test_angularly_unobservable_response_is_not_called_short_axis_control():
    matrix=J.copy();matrix[2]=0
    fit=fit_measured_response(pairs(matrix),unmodelled_gripper_tolerance=0.)
    with pytest.raises(ValueError,match='not observable'):
        propose_alignment(observation(),fit,LOWER,UPPER)


def test_input_arrays_are_owned_and_read_only():
    q=BASE_Q.copy();value=observation(q);q[:]=0
    assert value.q_position[5]==30
    with pytest.raises(ValueError):value.q_position[0]=1


def test_bad_image_readback_bracket_is_rejected():
    value=observation()
    record=dict(role='pickup',view='wrist',visible=True,view_source='current_observation',
        source='frozen_dinov2_profile_prototype',image_time_s=1.,centroid_px=[110,112],axis_px=[0,1])
    with pytest.raises(ValueError,match='paired'):
        alignment_observation(record,BASE_Q,value.jaw_fm_px,image_time_s=1.,q_time_s=1.08,
                              now_s=1.1,max_age_s=.2)


def test_registered_body_alignment_uses_the_actor_cog():
    value=observation(source="operator_colour_body",cog_px=[120.,140.],centroid_px=[80.,100.])
    np.testing.assert_allclose(value.target_center_px,[120.,140.])
    with pytest.raises(ValueError,match="current observed"):
        observation(source="operator_colour_body",visible=False)
