from dataclasses import replace

import pytest
import torch

from fcpg.v21_runtime import EncodedEndpoint
from fcpg.v21_memory import VisualObservation
from test_v21_planning_memory import runtime  # noqa: F401


@pytest.mark.parametrize('operation',['observe','plan','preview'])
@pytest.mark.parametrize('damage',['changed_same_exposure','out_of_order','nan_features','nan_image_q'])
def test_same_pose_replacement_cannot_bypass_memory_validation(runtime,operation,damage):
    rt,now=runtime;before=rt.state;memory=before.memory
    features=memory.features[:,:,-1].clone();q=memory.q_at_image[:,:,-1].clone()
    times=memory.image_times[:,:,-1].clone()
    if damage=='changed_same_exposure':features+=1.
    elif damage=='out_of_order':times-=.5
    elif damage=='nan_features':times+=.5;features[0,0,0,0]=float('nan')
    else:times+=.5;q[0,0,0]=float('nan')
    obs=VisualObservation(features,q,times,torch.ones(1,2,dtype=torch.bool))
    now[0]=11.;endpoint=EncodedEndpoint(before.q.clone(),torch.tensor([11.],dtype=torch.float64),obs)
    with pytest.raises(ValueError):
        if operation=='plan':rt.plan(endpoint=endpoint)
        else:getattr(rt,operation)(endpoint)
    assert rt.state is before and rt.state.memory is memory
    assert torch.isfinite(memory.features).all() and torch.isfinite(rt.state.grounding).all()
    assert memory.image_times[0,0,-1].item()==10.


def test_unavailable_view_does_not_poison_or_erase_its_actual_bank(runtime):
    rt,now=runtime;before=rt.state;memory=before.memory
    features=memory.features[:,:,-1].clone();features[:,1]=float('nan')
    q=memory.q_at_image[:,:,-1].clone();q[:,1]=float('nan')
    times=memory.image_times[:,:,-1].clone()+.5;times[:,1]=float('nan')
    obs=VisualObservation(features,q,times,torch.tensor([[True,False]]))
    now[0]=11.
    rt.observe(EncodedEndpoint(before.q.clone(),torch.tensor([11.],dtype=torch.float64),obs))
    assert torch.isfinite(rt.state.memory.features).all()
    torch.testing.assert_close(rt.state.memory.features[:,1],memory.features[:,1])
    torch.testing.assert_close(rt.state.memory.image_times[:,1],memory.image_times[:,1])
