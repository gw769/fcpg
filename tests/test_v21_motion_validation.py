import numpy as np
import pytest
from fcpg.v21_motion_validation import compare_actual_pair
from fcpg.v21_motion_validation import run_probe

def obs(q,m,t):
    return dict(q=q,views={'column':dict(available=True,M=m,time=t)})

def test_actual_direction_and_not_height_certification():
    a=obs([0]*6,[0,0],1.)
    b=obs([1,0,0,0,0,0],[3,0],2.)
    out=compare_actual_pair(a,b,0,lambda q:np.array([3*q[0],0]))
    assert out['direction_verified'] and not out['height_verified']

def test_coupled_drift_rejected():
    a=obs([0]*6,[0,0],1.)
    b=obs([1,1,0,0,0,0],[3,0],2.)
    assert not compare_actual_pair(a,b,0,lambda q:q[:2])['direction_verified']

def test_wrong_direction_rejected():
    a=obs([0]*6,[0,0],1.)
    b=obs([1,0,0,0,0,0],[-3,0],2.)
    assert not compare_actual_pair(a,b,0,lambda q:np.array([3*q[0],0]))['direction_verified']

@pytest.mark.parametrize('amplitude',[0.,-1.,4.01,float('nan'),True])
def test_probe_amplitude_rejected_before_any_io(amplitude):
    with pytest.raises(ValueError,match='amplitude'):
        run_probe(None,2,-1,amplitude=amplitude)
