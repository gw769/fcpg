"""Candidate planning uses fresh real inputs without committing UI refreshes.

All IO and clocks below are synthetic. No camera, robot, checkpoint, or GPU is
opened, and passing these tests is not evidence of a successful physical grasp.
"""
from dataclasses import replace

import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_memory import VisualObservation
from fcpg.v21_runtime import EncodedEndpoint, V21ResearchRuntime


def endpoint(stamp, marker, *, available=(True, True)):
    generator=torch.Generator().manual_seed(marker)
    q=torch.full((1,6),float(marker))
    obs=VisualObservation(torch.randn(1,2,256,384,generator=generator),
        q[:,None].expand(-1,2,-1),torch.full((1,2),stamp,dtype=torch.float64),
        torch.tensor([available],dtype=torch.bool))
    return EncodedEndpoint(q,torch.tensor([stamp],dtype=torch.float64),obs)


@pytest.fixture
def runtime():
    torch.set_num_threads(2);torch.manual_seed(21)
    vocab=TaskVocabulary.build(['pick red fish'])
    core=V21ResearchCore(V21Config(len(vocab.words)))
    now=[10.]
    value=V21ResearchRuntime(core,vocab.encode(['pick red fish']),
        physical_check=lambda q,u:None,lease_check=lambda:None,
        clock_fn=lambda:now[0],max_plan_age_s=300.)
    for stamp,marker in [(8.,1),(9.,2),(10.,3)]:value.observe(endpoint(stamp,marker))
    return value,now


def test_fresh_preview_does_not_commit_or_rearm(runtime):
    value,now=runtime;before=value.state
    for index in range(5):
        now[0]+=.1
        image=endpoint(now[0],index+4)
        commands,forecast,admission=value.preview(image)
        assert commands.shape==(1,5,6) and admission['allowed']
        torch.testing.assert_close(forecast.source_q,image.q)
        assert value.state is before and value.pending is None and value.stopped
    assert before.memory.q_at_image[0,0,:,0].tolist()==[1.,2.,3.]


def test_cancelled_plans_leave_original_actual_memory(runtime):
    value,now=runtime;before=value.state
    for index in range(5):
        now[0]+=.1
        image=endpoint(now[0],index+4)
        value.plan(endpoint=image)
        assert value.state is before
        torch.testing.assert_close(value.pending.source_q,image.q)
        value.stop()
        assert value._pending_state is None and value.state is before


def test_preview_count_cannot_change_the_final_same_input_action(runtime):
    value,now=runtime;before=value.state;final=endpoint(20.,9)
    with torch.inference_mode():
        candidate=value.core.observe(before,final.observation,final.q,final.time)
        dt=torch.full((1,5),.1)
        offsets=torch.cat((dt.new_zeros(1,1).double(),dt.double().cumsum(-1)[:,:-1]),-1)
        expected=value.core.act(candidate,value.text_ids,offsets).commands_position
    for index in range(5):
        now[0]=11.+index
        value.preview(endpoint(now[0],index+4))
        value.plan(endpoint=endpoint(now[0],index+4))
        value.stop()
    now[0]=20.
    actual=value.plan(endpoint=final)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    assert value.state is before


def test_only_the_executed_candidate_enters_actual_memory(runtime):
    value,now=runtime;before=value.state
    now[0]=10.1;value.plan(endpoint=endpoint(now[0],4));value.stop()
    now[0]=10.2;latest=endpoint(now[0],5)
    value.plan(endpoint=latest)
    assert value.state is before
    commands=value.begin_execution(latest.q,latest.time)
    assert commands.shape==(1,5,6) and value.executing
    assert value._pending_state is None
    assert value.state.memory.q_at_image[0,0,:,0].tolist()==[2.,3.,5.]
    assert not value.state.history.valid.any()  # begin is not completed motion


def test_changed_pose_rejects_candidate_without_committing_it(runtime):
    value,now=runtime;before=value.state
    now[0]=10.1;latest=endpoint(now[0],4);value.plan(endpoint=latest)
    with pytest.raises(RuntimeError,match='pose changed'):
        value.begin_execution(latest.q+1,latest.time)
    assert value.stopped and value.pending is None and value._pending_state is None
    assert value.state is before


def test_failed_start_deadline_does_not_commit_candidate(runtime,monkeypatch):
    value,now=runtime;before=value.state
    now[0]=10.1;latest=endpoint(now[0],4);value.plan(endpoint=latest)
    original=value.core.predict
    def slow(*args):
        result=original(*args);now[0]+=.2;return result
    monkeypatch.setattr(value.core,'predict',slow)
    with pytest.raises(RuntimeError,match='deadline'):
        value.begin_execution(latest.q,latest.time)
    assert value.state is before and value.stopped and value.pending is None


def test_rejected_preview_preserves_feedback_and_does_not_plan(runtime):
    value,now=runtime
    value.state=replace(value.state,visual_error=torch.ones_like(value.state.visual_error),
        visual_feedback_available=torch.ones_like(value.state.visual_feedback_available),
        visual_feedback_time=torch.full_like(value.state.visual_feedback_time,9.))
    before=value.state
    value.physical_check=lambda q,u:(_ for _ in ()).throw(RuntimeError('clearance'))
    now[0]=10.1
    _,_,admission=value.preview(endpoint(now[0],4))
    assert not admission['allowed'] and admission['reason']=='clearance'
    assert value.state is before and value.stopped and value.pending is None


def test_admission_cannot_mutate_committed_q_or_preview_commands(runtime):
    value,now=runtime;before=value.state;now[0]=10.1;image=endpoint(now[0],4)
    clean,_,_=value.preview(image)
    def mutate(q,u):q.add_(10);u.add_(10)
    value.physical_check=mutate
    commands,forecast,admission=value.preview(image)
    assert not admission['allowed'] and 'changed' in admission['reason']
    torch.testing.assert_close(commands,clean)
    torch.testing.assert_close(forecast.trace.positions,clean)
    assert value.state is before and before.q[0,0].item()==3.


def test_missing_column_never_gets_a_new_camera_time_from_preview(runtime):
    value,now=runtime;before=value.state;now[0]=10.1
    image=endpoint(now[0],4,available=(True,False))
    value.preview(image)
    assert value.state is before
    value.plan(endpoint=image)
    value.begin_execution(image.q,image.time)
    torch.testing.assert_close(value.state.memory.image_times[:,1],before.memory.image_times[:,1],atol=0,rtol=0)
    assert not value.state.measurement_available[0,1]


def test_rejected_plan_cannot_mutate_committed_state(runtime):
    value,now=runtime;before=value.state;now[0]=10.1
    def mutate(q,u):q.add_(10);u.add_(10)
    value.physical_check=mutate
    with pytest.raises(RuntimeError,match='changed'):
        value.plan(endpoint=endpoint(now[0],4))
    assert value.state is before and value.pending is None and value.stopped
    assert before.q[0,0].item()==3.


def test_future_preview_rejected_without_changing_actual_memory(runtime):
    value,now=runtime;before=value.state
    with pytest.raises(RuntimeError,match='future'):
        value.preview(endpoint(now[0]+1,4))
    assert value.state is before and value.stopped and value.pending is None


def test_external_proposal_is_checked_before_lewm_and_cannot_impersonate_actor(runtime,monkeypatch):
    value,now=runtime;before=value.state;image=endpoint(now[0],3)
    builder=lambda state:state.q[:,None].expand(-1,5,-1).clone()
    with pytest.raises(ValueError,match='labelled'):
        value.plan(endpoint=image,command_builder=builder)
    value.plan(endpoint=image,command_builder=builder,action_source='visual_alignment_probe')
    assert value.pending_action_source=='visual_alignment_probe' and value.state is before
    value.stop()
    value.plan(endpoint=image,command_builder=builder,action_source='column_dino_fg_probe')
    assert value.pending_action_source=='column_dino_fg_probe' and value.state is before
    value.stop()
    value.physical_check=lambda q,u:(_ for _ in ()).throw(RuntimeError('mechanical rejection'))
    monkeypatch.setattr(value.core,'predict',lambda *a:pytest.fail('LeWM ran before physical admission'))
    with pytest.raises(RuntimeError,match='mechanical rejection'):
        value.plan(endpoint=image,command_builder=builder,action_source='visual_alignment_servo')
    assert value.state is before and value.stopped
