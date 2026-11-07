import numpy as np
import pytest
from dataclasses import replace

from fcpg.v21_visual_alignment import AlignmentObservation
from scripts.serve_so101_prod_v17_7 import V177ClearanceGate
from test_v21_home import FlatKinematics
from test_v21_live_service import service  # noqa: F401


def prepare(service):
    service.alignment_motion_enabled=True;service.alignment_scope='lateral'
    service.owner.q=np.array([-4.,12.,-7.,91.,-51.,31.],np.float32)
    service.owner.clearance=V177ClearanceGate(kinematics=FlatKinematics(),table_z_mm=0.)
    service.owner.bridge.extra_readback_check=None
    service.auxiliary_budget.reserve();service.auxiliary_budget.finish(None)
    def obs(q,t):
        return AlignmentObservation(q.copy(),np.array([54.-7.*(q[0]+4.),100.]),
            np.array([0.,1.]),np.array([[100.,140.],[160.,140.]]),t,t)
    service._alignment_measure=lambda:obs(service.owner.last_endpoint.q[0].numpy(),
                                         float(service.owner.last_endpoint.time[0]))
    q=service.owner.q.copy();old=q.copy();old[0]+=2.
    service._alignment_controller().pairs=[(obs(old,9.),obs(q,9.5))]


def test_directional_reach_uses_shared_transaction_budget_and_actual_memory(service):
    prepare(service)
    result=service.handle('alignment-reach')
    assert service.owner.writes==5 and result['auxiliary_trial']['block_attempts']==2
    assert result['last_execution']['action_source']=='visual_alignment_reach'
    assert result['alignment']['actual_improvement_px']>0
    assert result['runtime']['stopped'] and service.owner.bridge.extra_readback_check is None
    assert 'wrist_alignment' in result['last_execution']
    assert service.owner.q[5]==31.
    np.testing.assert_array_equal(service.owner.q[1:],[12.,-7.,91.,-51.,31.])


def test_directional_reach_has_no_low_clearance_exception(service):
    prepare(service)
    service.owner.clearance=replace(service.owner.clearance,table_z_mm=90.)
    with pytest.raises(RuntimeError):service.handle('alignment-reach')
    assert service.owner.writes==0 and service.owner.runtime.stopped
    assert service.owner.bridge.extra_readback_check is None
