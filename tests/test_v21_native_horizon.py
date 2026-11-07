import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_native_horizon import initialize_native40, native_outlook
from fcpg.v21_models import CommandTrace


def test_native40_is_one_actor_and_one_lewm_forward_with_one_real_source():
    core=initialize_native40(V21ResearchCore(V21Config(vocabulary_size=5))).eval()
    state=core.initial_state(torch.tensor([[0.,-50.,50.,50.,0.,30.]]),torch.tensor([10.],dtype=torch.float64))
    calls=[]
    handles=[m.register_forward_hook(lambda m,a,o,label=label:calls.append(label))
             for label,m in [('actor',core.actor),('lewm',core.lewm)]]
    result=native_outlook(core,state,torch.tensor([[1,2]]))
    for h in handles:h.remove()
    assert calls==['actor','lewm']
    assert len(result['commands_position'])==40 and len(result['predicted_q_path'])==41
    assert result['native_horizon']==40 and not result['executable']
    assert state.history.valid.sum()==0 and state.memory.current_readback_index.item()==0


def test_late_commands_do_not_change_earlier_native40_forecast():
    core=initialize_native40(V21ResearchCore(V21Config(vocabulary_size=5))).eval()
    q=torch.tensor([[0.,-50.,50.,50.,0.,30.]])
    state=core.initial_state(q,torch.tensor([10.],dtype=torch.float64))
    offsets=torch.arange(40,dtype=torch.float64)[None]*.1
    commands=q[:,None].expand(1,40,6).clone()
    trace=CommandTrace(commands,offsets,torch.ones_like(offsets,dtype=torch.bool),'planned',True)
    with torch.inference_mode():
        before=core.lewm(q,trace,state.history,offsets+.1).predicted_q_path
        changed=commands.clone();changed[:,20:,0]+=10
        trace=CommandTrace(changed,offsets,trace.valid,'planned',True)
        after=core.lewm(q,trace,state.history,offsets+.1).predicted_q_path
    torch.testing.assert_close(before[:,:21],after[:,:21],atol=1e-5,rtol=0)
    assert not torch.equal(before[:,21:],after[:,21:])
