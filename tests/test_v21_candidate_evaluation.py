from dataclasses import replace
import numpy as np
import pytest
from fcpg.v21_candidate_evaluation import VisualGoal, actor_prefix_candidates, evaluate_candidates

Q = np.zeros(6)
U = np.zeros((5,6)); U[:,0] = np.arange(1,6)
GOAL = VisualGoal((10.,0.), (0.,0.), tuple(Q), 10.,10.,10., geometry_validated=True)
def forecast(q,u): return np.vstack([q,u]), np.zeros((5,6))
def project(q): return np.asarray(q)[:2]
def assess(goal=GOAL, **kw):
    return evaluate_candidates(Q,U,goal,predict=kw.get('predict',forecast),
        mechanical_check=kw.get('mechanical_check',lambda q,u:None),project=kw.get('project',project))

def test_actor_proposals_not_geometry_commands():
    candidates = actor_prefix_candidates(Q,U)
    np.testing.assert_array_equal(candidates['actor_5'],U)
    np.testing.assert_array_equal(candidates['actor_3_then_hold'][3:],np.repeat(U[2:3],2,axis=0))
    assert assess()['selected']=='actor_5'
    assert not assess()['executable']

def test_unvalidated_geometry_cannot_select():
    result=assess(replace(GOAL,geometry_validated=False))
    assert result['selected'] is None
    assert all(not r['eligible'] for r in result['candidates'])

def test_fixed_geometry_bias_cancels():
    a,b=assess(),assess(project=lambda q:project(q)+1000)
    assert a['selected']==b['selected']
    assert a['candidates'][0]['score']==b['candidates'][0]['score']

def test_bad_motion_prefers_hold():
    assert assess(replace(GOAL,target_px=(-10.,0.)))['selected']=='hold'

def test_mechanical_rejection_before_predict():
    calls=[]
    def check(q,u):
        if u[:,0].any():raise RuntimeError('unsafe')
    def pred(q,u):calls.append(u.copy()); return forecast(q,u)
    assert assess(mechanical_check=check,predict=pred)['selected']=='hold'
    assert len(calls)==1 and not calls[0].any()

def test_uncertain_motion_can_lose_to_hold():
    def pred(q,u):return np.vstack([q,u]),np.full((5,6),100. if u.any() else 0.)
    assert assess(predict=pred)['selected']=='hold'

@pytest.mark.parametrize('goal',[replace(GOAL,midpoint_time_s=9.),replace(GOAL,source_time_s=11.),
                                 replace(GOAL,source='DINO_support_mask')])
def test_invalid_goal_rejected(goal):
    with pytest.raises(ValueError): assess(goal)

def test_no_mutation():
    before=U.copy(); assess(); np.testing.assert_array_equal(U,before)

def test_unknown_contact_rejects_close_not_rewrites_it():
    commands=U.copy();commands[:,5]=-1.
    result=evaluate_candidates(Q,commands,GOAL,predict=forecast,mechanical_check=lambda q,u:None,project=project)
    assert result['selected']=='hold'
    assert result['candidates'][0]['commands_position'][-1][-1]==-1.

def test_invalid_hold_prediction_blocks_selection():
    def pred(q,u):
        if not u.any():raise ValueError('hold invalid')
        return forecast(q,u)
    assert assess(predict=pred)['selected'] is None


def test_full_pose_score_contains_orientation_and_height_without_changing_commands():
    goal=replace(GOAL,body_axis_px=(1.,0.),observed_jaw_axis_px=(0.,1.),
                 observed_height_m=.05,target_height_m=.02)
    result=evaluate_candidates(Q,U,goal,predict=forecast,mechanical_check=lambda q,u:None,
        project=project,pose=lambda q:dict(jaw_axis_px=[0.,1.],height_m=.05-.005*q[0]))
    assert result['evaluation_scope']=='position-axis-height'
    first=result['candidates'][0]
    assert first['predicted_axis_error_deg']==pytest.approx(0.)
    assert first['predicted_height_m']==pytest.approx(.025)
    assert first['commands_position']==U.tolist()


def test_body_cannot_be_used_as_head_goal():
    with pytest.raises(ValueError,match='region'):
        assess(replace(GOAL,goal_region='head'))
