from dataclasses import replace

import numpy as np
import pytest

from fcpg.v21_lateral_alignment import LateralAlignmentSession,fit_lateral_response,directional_proposal
from test_v21_alignment_session import experiment
from test_v21_visual_alignment import observation,BASE_Q,BASE_FEATURES


def lateral_experiment():
    old,q,moves,matrix=experiment()
    session=LateralAlignmentSession(old.measure,old.run_motion,lower=old.lower,upper=old.upper,
                                    gripper_tolerance=old.gripper_tolerance)
    return session,q,moves,matrix


def test_probe_return_and_lateral_correction_use_three_blocks_not_a_five_joint_fit():
    session,q,moves,_=lateral_experiment()
    session.probe(0,-1);session.return_from_probe()
    result=session.fit()
    assert result['gain_px_per_position']==pytest.approx(2.)
    correction=session.step()
    assert len(moves)==3 and q[0]<0 and q[5]==30.
    np.testing.assert_array_equal(moves[-1][1][:,1:5],np.zeros((5,4)))
    assert correction['actual_improvement_px']>0
    assert not correction['physical_grasp_verified']
    assert correction['proposal']['scope']=='lateral_reacquisition_only'
    assert session.response is None


def test_other_joint_motion_cannot_be_assigned_to_j1():
    session,_,_,_=lateral_experiment();session.probe(0,-1);session.return_from_probe()
    before,after=session.pairs[0];q=after.q_position.copy();q[1]+=.5
    session.pairs[0]=(before,replace(after,q_position=q))
    with pytest.raises(ValueError,match='other joints'):
        session.fit()


def test_requested_motion_cannot_replace_insufficient_actual_motion():
    before=observation();q=BASE_Q.copy();q[0]=.05
    after=observation(q,BASE_FEATURES+np.array([4.,0.,0.]),stamp=2.)
    with pytest.raises(ValueError,match='insufficient'):
        fit_lateral_response([(before,after),(replace(after,image_time_s=3.,q_time_s=3.01),
            replace(before,image_time_s=4.,q_time_s=4.01))],.25)


def test_wrong_direction_return_does_not_validate_gain():
    session,_,_,_=lateral_experiment();session.probe(0,-1);session.return_from_probe()
    before,after=session.pairs[1]
    center=after.target_center_px.copy();center[0]=before.target_center_px[0]-4.
    session.pairs[1]=(before,replace(after,target_center_px=center))
    with pytest.raises(ValueError,match='inconsistent'):
        session.fit()


def test_lateral_scope_cannot_probe_wrist_roll():
    session,_,moves,_=lateral_experiment()
    with pytest.raises(ValueError,match='only probes J1'):session.probe(4,1)
    assert not moves


def test_directional_followup_uses_real_improvement_but_does_not_claim_a_jacobian():
    q=BASE_Q.copy();q[0]=-2.
    before=observation(q,np.array([40.,100.,0.]),stamp=1.)
    q=q.copy();q[0]-=2.;q[1]+=.4
    after=observation(q,np.array([54.,100.,0.]),stamp=2.)
    current=replace(after,image_time_s=3.,q_time_s=3.01)
    commands,diag=directional_proposal(before,after,current,np.array([-100.]*5+[0.]),np.full(6,100.))
    assert commands.shape==(5,6) and commands[-1,0]<q[0]
    assert abs(commands[-1,0]-q[0])<=8.
    np.testing.assert_array_equal(commands[:,1:],np.repeat(q[None,1:],5,0))
    assert diag['identified_jacobian'] is False and diag['extrapolated_response'] is True


@pytest.mark.parametrize('damage',['wrong_direction','large_other_motion','pose_change','object_change'])
def test_directional_followup_rejects_unsupported_continuations(damage):
    q=BASE_Q.copy();q[0]=-2.
    before=observation(q,np.array([40.,100.,0.]),stamp=1.)
    q=q.copy();q[0]-=2.
    after=observation(q,np.array([54.,100.,0.]),stamp=2.)
    current=replace(after,image_time_s=3.,q_time_s=3.01)
    if damage=='wrong_direction':after=replace(after,target_center_px=np.array([20.,100.]))
    if damage=='large_other_motion':
        x=after.q_position.copy();x[1]+=2;after=replace(after,q_position=x)
    if damage=='pose_change':
        x=current.q_position.copy();x[3]+=2;current=replace(current,q_position=x)
    if damage=='object_change':current=replace(current,target_center_px=np.array([170.,100.]))
    with pytest.raises(ValueError):directional_proposal(before,after,current,np.full(6,-100.),np.full(6,100.))


def test_small_local_drifts_cannot_mix_different_global_configurations():
    samples=[]
    for i,sign in enumerate((1.,-1.,1.,-1.)):
        q=BASE_Q.copy();q[1]=i*.15
        end=q.copy();end[0]+=sign
        samples.append((observation(q,stamp=2.*i+1.),
            observation(end,BASE_FEATURES+np.array([2.*sign,0.,0.]),stamp=2.*i+2.)))
    # Every within-pair non-J1 delta is zero, but the pool drifted by .45.
    with pytest.raises(ValueError,match='configuration'):fit_lateral_response(samples,.25)


def test_response_reference_is_owned_and_immutable():
    from fcpg.v21_lateral_alignment import LateralResponse
    q=BASE_Q.copy();response=LateralResponse(2.,0.,q,-2.,2.,2.,2)
    q[1]=30.
    assert response.reference_q[1]==BASE_Q[1]
    with pytest.raises(ValueError):response.reference_q[1]=40.


def test_actual_image_response_is_not_assumed_equal_to_the_command_delta():
    session,q,moves,_=lateral_experiment()
    session.probe(0,-1);session.return_from_probe();session.fit()
    move=session.run_motion
    def undertracking(builder,source):
        def reduced(state):
            commands=builder(state)
            return state.q[:,None]+.6*(commands-state.q[:,None])
        return move(reduced,source)
    session.run_motion=undertracking
    result=session.step();response=result['exposure_paired_response'];proposal=result['proposal']
    assert response['actual_delta_j1']==pytest.approx(.6*proposal['delta_j1'])
    assert response['observed_target_shift_x_px']==pytest.approx(response['gain_times_actual_delta_x_px'])
    assert response['image_response_residual_x_px']==pytest.approx(0.,abs=1e-12)
    assert result['actual_error_after_x_px']!=pytest.approx(proposal['setpoint_error_after_x_px'])
    assert 'predicted_after_x_px' not in proposal and 'not-LeWM' in proposal['estimate_kind']
