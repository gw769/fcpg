from copy import deepcopy

import numpy as np
import pytest

from fcpg.v21_view_feedback import tracking_evidence,compensate


Q=np.array([0.,-26.224783,45.430347,79.530205,-51.584415,31.143232])
TARGET=np.array([0.,-26.261694,43.119911,79.434341,-51.584415,31.143232])


def previous():
    nominal=np.repeat(TARGET[None],5,0)
    return dict(action_source='visual_alignment_view_step',completed=True,
        policy_commands_attempted=5,policy_commands_returned=5,endpoint_time_s=12.,
        commands_position=nominal.tolist(),normal_follow=dict(goal_reached=True,quiet_samples=3,
            positions=np.repeat(Q[None],3,0).tolist(),times_s=[11.8,11.85,11.9]),
        view_step=dict(plan=dict(stage='radial_view_probe',
            waypoints=[dict(q_position=row.tolist()) for row in nominal])))


def test_real_shortfall_case_recovers_previous_target_not_new_target_from_missed_pose():
    result=tracking_evidence(Q,previous())
    assert result['recovering']
    nominal=Q+np.linspace(.2,1,5)[:,None]*(result['nominal_target']-Q)
    issued,diag=compensate(Q,nominal,result,[-100]*5+[0],[100]*6)
    np.testing.assert_allclose(issued[-1],TARGET+(TARGET-Q))
    np.testing.assert_array_equal(issued[:,[0,4,5]],np.repeat(Q[None,[0,4,5]],5,0))
    assert diag['gain']==1. and diag['motor_parameters_changed'] is False
    assert np.abs(np.diff(np.vstack((Q,issued)),axis=0)).max()<6.


def test_quiet_shortfall_uses_hold_eligibility_without_claiming_precise_arrival():
    response=previous()
    response['normal_follow'].update(hold_eligible=True, goal_reached=False,
        goal_reached_scope='original_command_position_arrival')
    assert tracking_evidence(Q,response)['recovering']
    response['normal_follow'].update(hold_eligible=False,goal_reached=True)
    with pytest.raises(ValueError):tracking_evidence(Q,response)


@pytest.mark.parametrize('damage',['moving','stale_pose','unsettled','partial','fake_time','nan','missing'])
def test_unverified_response_cannot_supply_tracking_correction(damage):
    response=previous();q=Q.copy()
    if damage=='moving':response['normal_follow']['positions'][0][2]+=1
    elif damage=='stale_pose':q[1]+=1
    elif damage=='unsettled':response['normal_follow']['goal_reached']=False
    elif damage=='partial':response['policy_commands_returned']=4
    elif damage=='fake_time':response['normal_follow']['times_s']=[1.,1.,1.]
    elif damage=='nan':response['normal_follow']['positions'][-1][0]=float('nan')
    elif damage=='missing':response['normal_follow']['positions']=[]
    with pytest.raises(ValueError):tracking_evidence(q,response)


def test_inexact_recovery_cannot_be_reused_but_allows_fresh_high_view_planning():
    response=previous();response['view_step']['plan']['stage']='tracking_recovery'
    evidence=tracking_evidence(Q,response)
    assert evidence['usable_offset'] is False and evidence['recovering'] is False
    assert 'offset' not in evidence


def test_no_history_and_other_sources_do_not_invent_an_offset():
    assert tracking_evidence(Q,None) is None
    response=previous();response['action_source']='raw_actor'
    assert tracking_evidence(Q,response) is None


def test_compensation_scales_before_bounds_not_after_admission():
    e=tracking_evidence(Q,previous());nominal=np.repeat(Q[None],5,0)
    lower=np.array([-100.]*5+[0.]);lower[2]=Q[2]-.2
    commands,diag=compensate(Q,nominal,e,lower,[100]*6)
    assert 0<diag['gain']<1
    assert commands[:,2].min()>=lower[2]
    assert np.abs(np.diff(np.vstack((Q,commands)),axis=0)).max()<=6.


def test_recovery_does_not_change_previous_telemetry():
    p=previous();saved=deepcopy(p)
    e=tracking_evidence(Q,p);compensate(Q,np.repeat(Q[None],5,0),e,[-100]*5+[0],[100]*6)
    assert p==saved
