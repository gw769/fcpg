from types import SimpleNamespace

import numpy as np
import pytest

from scripts.serve_so101_prod_v17_7 import V177ClearanceGate
from test_v21_home import FlatKinematics
from test_v21_live_service import service  # noqa: F401


@pytest.fixture
def view(service,monkeypatch):
    import fcpg.v21_live_service as live
    service.alignment_motion_enabled=True
    service.owner.q=np.array([0.,-30.,40.,60.,-50.,31.],np.float32)
    service.owner.kinematics=FlatKinematics()
    service.owner.clearance=V177ClearanceGate(kinematics=service.owner.kinematics,table_z_mm=0.)
    service.owner.bridge.extra_readback_check=None
    service.auxiliary_budget.reserve();service.auxiliary_budget.finish(None)
    class Solver:
        def __init__(self,*a,**kwargs):pass
        def propose_view_chunk(self,q,**kwargs):
            commands=np.repeat(q[None],5,0);commands[:,1]+=np.linspace(.2,1.,5)
            return SimpleNamespace(accepted=True,commands_position=commands,diagnostic={'stage':'fixture'})
    monkeypatch.setattr(live,'VerticalApproachSolver',Solver)
    monkeypatch.setattr(live,'approach_geometry',lambda *a:{'fixture':True})
    return service


def test_view_step_uses_budget_reobserves_stops_and_removes_temporary_guard(view):
    result=view.handle('alignment-view-step')
    assert result['runtime']['stopped'] and view.owner.writes==5
    assert result['auxiliary_trial']['block_attempts']==2
    assert result['auxiliary_trial']['policy_commands_returned']==5
    assert result['last_execution']['action_source']=='visual_alignment_view_step'
    assert view.owner.bridge.extra_readback_check is None
    assert view._carry is None and view.owner.q[5]==31.


def test_view_step_rejection_never_leaves_guard_or_sends_a_close(view):
    view.owner.q[5]=5.
    with pytest.raises(RuntimeError,match='open jaws'):view.handle('alignment-view-step')
    assert view.owner.writes==0 and view.owner.runtime.stopped
    assert view.owner.bridge.extra_readback_check is None
