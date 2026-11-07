from dataclasses import replace
import torch
import pytest

from fcpg.v21_core import V21Config,V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_runtime import V21ResearchRuntime,EncodedEndpoint
from fcpg.v21_memory import VisualObservation


def prepared():
    torch.set_num_threads(2);torch.manual_seed(21)
    vocabulary=TaskVocabulary.build(['pick red block'])
    core=V21ResearchCore(V21Config(len(vocabulary.words))).eval()
    now=[10.]
    runtime=V21ResearchRuntime(core,vocabulary.encode(['pick red block']),
        physical_check=lambda *a:None,lease_check=lambda:None,clock_fn=lambda:now[0])
    q=torch.zeros(1,6)
    obs=VisualObservation(torch.randn(1,2,256,384),q[:,None].expand(-1,2,-1),
        torch.full((1,2),10.,dtype=torch.float64),torch.ones(1,2,dtype=torch.bool))
    runtime.observe(EncodedEndpoint(q,torch.tensor([10.],dtype=torch.float64),obs))
    runtime.plan();now[0]=10.01
    commands=runtime.begin_execution(q,torch.tensor([10.01],dtype=torch.float64))
    trace=replace(runtime.pending.trace,source='recorded')
    path=torch.cat((q[:,None],commands),1)
    times=torch.tensor([[10.01,10.11,10.21,10.31,10.41,10.51]],dtype=torch.float64)
    follow=path.clone();follow[:,-1]+=.1
    ft=times.clone();ft[:,-1]=10.71
    now[0]=10.73
    image=replace(obs,q_at_image=follow[:,-1:, :].expand(-1,2,-1),
                  image_times=torch.full((1,2),10.73,dtype=torch.float64))
    endpoint=EncodedEndpoint(follow[:,-1],torch.tensor([10.73],dtype=torch.float64),image)
    return runtime,trace,path,times,follow,ft,endpoint


def test_follow_feedback_uses_actual_later_time_without_overwriting_nominal_q5():
    runtime,trace,path,times,follow,ft,endpoint=prepared()
    original=path.clone();original_times=times.clone()
    with torch.inference_mode():
        nominal=runtime.core.predict(runtime.state,trace,times[:,1:]-times[:,:1]).response.predicted_q_path
    result=runtime.complete(trace,path,times,endpoint,feedback_path=follow,feedback_times=ft)
    assert torch.equal(path,original) and torch.equal(times,original_times)
    assert runtime.state.feedback_offsets[0,-1].item()==pytest.approx(.7)
    assert runtime.state.history.time_deltas[0,-1,-1].item()==pytest.approx(.3)
    torch.testing.assert_close(torch.tensor(result['nominal_retimed_prediction']),nominal)
    torch.testing.assert_close(runtime.state.history.readbacks_position[0,-1,-1],follow[0,-1])


def test_live_default_keeps_original_q5_history_and_later_actual_image_state():
    runtime,trace,path,times,follow,ft,endpoint=prepared()
    original_q5=path[:,-1].clone()
    runtime.complete(trace,path,times,endpoint)
    assert runtime.state.feedback_offsets[0,-1].item()==pytest.approx(.5)
    assert runtime.state.history.time_deltas[0,-1,-1].item()==pytest.approx(.1)
    torch.testing.assert_close(runtime.state.history.readbacks_position[:,-1,-1],original_q5)
    torch.testing.assert_close(runtime.state.q,endpoint.q)
    torch.testing.assert_close(runtime.state.memory.q_at_image[:,:,-1],endpoint.observation.q_at_image)
    torch.testing.assert_close(runtime.state.memory.image_times[:,:,-1],endpoint.observation.image_times)
    assert runtime.state.time.item()==pytest.approx(10.73)
    assert runtime.state.feedback_time.item()==pytest.approx(10.51)


@pytest.mark.parametrize('damage',['prefix','clock','future'])
def test_invalid_follow_feedback_cannot_rewrite_the_nominal_prefix(damage):
    runtime,trace,path,times,follow,ft,endpoint=prepared()
    if damage=='prefix':follow[:,1,0]+=1
    elif damage=='clock':ft[:,1]+=.01
    else:ft[:,-1]=10.74
    with pytest.raises((ValueError,RuntimeError)):
        runtime.complete(trace,path,times,endpoint,feedback_path=follow,feedback_times=ft)
    assert runtime.stopped and runtime.pending is None
