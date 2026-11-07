from types import SimpleNamespace

import pytest
import torch

from test_v21_live_service import service  # noqa: F401


@pytest.mark.parametrize('failure',['missing_clearance','missing_solution'])
def test_missing_depth_or_descent_solution_never_becomes_release(service,failure):
    service._carry=dict(phase='lower',blocks=0)
    service.targets={'targets':{}}
    service.owner.last_object_relative=None
    def height(q):
        if failure=='missing_clearance':raise RuntimeError('no measured clearance')
        return 100.
    service.owner.clearance=SimpleNamespace(clearance_mm=height)
    service._approach_step=lambda *a:None
    q=torch.tensor([0.,0.,0.,0.,0.,5.])
    commands=q[None,None].repeat(1,5,1)
    with pytest.raises(RuntimeError,match='release is not authorized'):
        service._carry_builder(commands,q,{})
    assert service._carry['phase']=='lower'
    assert (commands[0,:,5]==5.).all()


@pytest.mark.parametrize('blocks',[40,41,-1,True,None])
def test_carry_cannot_run_a_forty_first_or_unaccounted_block(service,blocks):
    service._carry=dict(phase='lift',blocks=blocks)
    service.owner.last_object_relative=None
    q=torch.tensor([0.,0.,0.,0.,0.,5.])
    commands=q[None,None].repeat(1,5,1);before=commands.clone()
    with pytest.raises(RuntimeError,match='carry budget'):
        service._carry_builder(commands,q,{})
    torch.testing.assert_close(commands,before,rtol=0,atol=0)
    assert service._carry['blocks'] is blocks
