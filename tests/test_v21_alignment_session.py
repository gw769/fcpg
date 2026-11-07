import numpy as np
import pytest
import torch
from types import SimpleNamespace

from fcpg.v21_alignment_session import AlignmentSession
from fcpg.v21_visual_alignment import alignment_observation


def experiment():
    q=np.array([0.,0.,0.,0.,0.,30.]);now=[1.];moves=[]
    matrix=np.array([[2.,0.,1.,0.,.2],[0.,3.,0.,1.,.3],[0.,0.,.02,-.01,.12]])
    def measure():
        feat=np.array([130.,135.,np.pi/2+.15])+matrix@q[:5]
        record=dict(role='pickup',view='wrist',visible=True,view_source='current_observation',
            source='frozen_dinov2_profile_prototype',image_time_s=now[0],centroid_px=feat[:2].tolist(),
            width_px=20.,length_px=40.,
            axis_px=[np.cos(feat[2]),np.sin(feat[2])])
        return alignment_observation(record,q,[[90,112],[130,112]],image_time_s=now[0],
            q_time_s=now[0],now_s=now[0],max_age_s=.5)
    def move(builder,source):
        command=builder(SimpleNamespace(q=torch.tensor(q)[None])).numpy()[0]
        q[:]=command[-1];now[0]+=.5;moves.append((source,command.copy()))
        return dict(completed=True,action_source=source)
    session=AlignmentSession(measure,move,lower=[-100]*5+[0],upper=[100]*6,gripper_tolerance=.25)
    return session,q,moves,matrix


def test_probe_and_return_are_bounded_and_do_not_close_jaw():
    session,q,moves,_=experiment()
    session.probe(0,1);assert q[0]==2 and q[5]==30
    session.return_from_probe();assert q[0]==0 and q[5]==30
    assert len(session.pairs)==2
    assert [r[0] for r in moves]==['visual_alignment_probe','visual_alignment_return']
    with pytest.raises(ValueError,match='no completed probe'):session.return_from_probe()


def test_fit_and_single_step_require_actual_pairs_and_invalidate_old_fit():
    session,q,moves,matrix=experiment()
    with pytest.raises(ValueError,match='fit a measured'):session.step()
    for joint in range(5):session.probe(joint,1);session.return_from_probe()
    fitted=session.fit()
    np.testing.assert_allclose(fitted['jacobian'],matrix,atol=1e-12)
    result=session.step()
    assert result['observed_weighted_error']<result['proposal']['weighted_error_before']
    assert moves[-1][0]=='visual_alignment_servo' and q[5]==30
    assert session.response is None and not result['physical_grasp_verified']


@pytest.mark.parametrize('joint,direction',[(5,1),(-1,1),(True,1),(0,0),(0,True)])
def test_bad_probe_parameters_never_call_motion(joint,direction):
    session,_,moves,_=experiment()
    with pytest.raises(ValueError):session.probe(joint,direction)
    assert not moves


def test_return_rejects_a_robot_moved_elsewhere():
    session,q,moves,_=experiment();session.probe(0,1);q[1]+=20
    with pytest.raises(ValueError,match='neighbourhood'):session.return_from_probe()
    assert len(moves)==1
