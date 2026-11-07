from dataclasses import replace
from copy import deepcopy

import pytest
import torch

from fcpg.v21_actual_resume import load_actual_resume,resume_actual_state,save_actual_resume,validate_actual_response
from fcpg.v21_auxiliary_budget import AuxiliaryMotionBudget
from fcpg.v21_core import V21ResearchCore
from fcpg.v21_runtime import V21ResearchRuntime
from fcpg.v21_training import write_json
from test_v21_live_service import service  # noqa: F401


@pytest.fixture
def resume_case(service,tmp_path):
    service.handle('plan');service.handle('execute')
    original=service.owner.runtime
    budget=AuxiliaryMotionBudget();budget.reserve()
    budget.finish(dict(policy_commands_attempted=5,policy_commands_returned=5,completed=True,
                       action_source='visual_alignment_servo'))
    save_actual_resume(tmp_path/'actual.pt',original,budget,service.root)
    write_json(tmp_path/f'auxiliary-budget-{budget.trial_id}.json',budget.status())
    restored_budget=AuxiliaryMotionBudget.restore_latest(tmp_path)
    data=load_actual_resume(tmp_path/'actual.pt',service.root)
    core=V21ResearchCore(original.core.config).eval().requires_grad_(False)
    core.load_state_dict(original.core.state_dict(),strict=True)
    runtime=V21ResearchRuntime(core,original.text_ids,physical_check=lambda q,u:None,
                              lease_check=lambda:None,clock_fn=original.clock)
    endpoint=service.owner.capture()
    return data,runtime,restored_budget,endpoint,original,service


def test_actual_resume_preserves_history_budget_and_stop_not_pending_actions(resume_case):
    data,runtime,budget,endpoint,original,service=resume_case
    result=resume_actual_state(data,runtime,budget,endpoint,pose_tolerance=3.)
    assert result['restored'] and not result['pending_motion_restored']
    assert not result['controller_fit_restored'] and not result['task_success_verified']
    assert runtime.stopped and not runtime.executing and runtime.pending is None
    torch.testing.assert_close(runtime.state.history.readbacks_position,original.state.history.readbacks_position)
    torch.testing.assert_close(runtime.state.q_error,original.state.q_error)
    torch.testing.assert_close(runtime.state.memory.image_times[:,:,:-1],original.state.memory.image_times[:,:,:-1])
    assert (runtime.state.memory.image_times[:,:,-1]>original.state.memory.image_times[:,:,-1]).all()
    assert budget.block_attempts==1 and budget.returned==5 and not budget.requires_home
    assert service.owner.writes==5  # Only the fixture's original execution, never restore.


@pytest.mark.parametrize('damage',['budget','pose','missing_view','future','actor','lewm','text'])
def test_resume_fails_closed_on_mismatched_evidence(resume_case,damage):
    data,runtime,budget,endpoint,_,service=resume_case
    if damage=='budget':budget.block_attempts+=1
    elif damage=='pose':endpoint=replace(endpoint,q=endpoint.q+4)
    elif damage=='missing_view':
        endpoint=replace(endpoint,observation=replace(endpoint.observation,measurement_available=torch.tensor([[True,False]])))
    elif damage=='future':data['state']['time']+=100
    elif damage=='actor':
        with torch.no_grad():runtime.core.actor.output.bias.add_(1)
    elif damage=='lewm':
        with torch.no_grad():runtime.core.lewm.response.bias.add_(1)
    else:data['text_ids']=data['text_ids']+1
    with pytest.raises((ValueError,RuntimeError)):
        resume_actual_state(data,runtime,budget,endpoint,pose_tolerance=3.)
    assert runtime.state is None and runtime.stopped and budget.requires_home
    assert service.owner.writes==5


def test_save_rejects_pending_motion_even_when_old_state_is_real(service,tmp_path):
    service.handle('plan');service.handle('execute')
    budget=AuxiliaryMotionBudget();budget.reserve()
    budget.finish(dict(policy_commands_attempted=5,policy_commands_returned=5,completed=True))
    service.handle('plan')
    with pytest.raises(ValueError,match='stopped'):
        save_actual_resume(tmp_path/'bad.pt',service.owner.runtime,budget,service.root)
    assert not (tmp_path/'bad.pt').exists()


def test_file_contains_only_safe_loadable_tensors_and_regular_metadata(resume_case,tmp_path):
    data,_,_,_,_,service=resume_case
    assert isinstance(data['state'],dict) and isinstance(data['state']['history'],dict)
    assert data['reference_lewm_hold_forecast_not_actual'].shape==(1,6,6)
    assert 'reference_lewm_hold_forecast_not_actual' not in data['state']
    torch.save(data,tmp_path/'other.pt')
    with pytest.raises(ValueError,match='incompatible'):
        load_actual_resume(tmp_path/'other.pt',service.root/'different')


def test_physical_response_must_match_exact_saved_actual_event(resume_case):
    data,_,_,_,original,service=resume_case
    response=deepcopy(service.last_execution)
    response['action_source']=data['ledger']['records'][-1]['action_source']
    validated=validate_actual_response(response,original.state,data['ledger'])
    assert validated==response and validated is not response
    response['commands_position'][-1][2]+=1
    with pytest.raises(ValueError,match='actual history'):
        validate_actual_response(response,original.state,data['ledger'])


def test_resume_rejects_foreign_physical_response_before_state_changes(resume_case):
    data,runtime,budget,endpoint,_,service=resume_case
    data['last_execution']=deepcopy(service.last_execution)
    data['last_execution']['endpoint_time_s']-=1
    with pytest.raises(ValueError):resume_actual_state(data,runtime,budget,endpoint,pose_tolerance=3.)
    assert runtime.state is None and runtime.stopped and budget.requires_home


def test_rejected_resume_keeps_readonly_view_without_clearing_motion_interlock(resume_case):
    data,runtime,budget,endpoint,_,service=resume_case
    endpoint=replace(endpoint,q=endpoint.q+4.)
    service.owner.runtime=runtime;service.auxiliary_budget=budget
    service.owner.io.settle_position_tolerance=3.
    service._restore_or_observe_stopped(data,endpoint)
    assert service.actual_resume['restored'] is False
    assert service.actual_resume['motion_authorized'] is False
    assert 'maximum_pose_difference' in service.actual_resume['reason']
    assert budget.requires_home and budget.block_attempts==1 and budget.returned==5
    assert runtime.stopped and not runtime.state.history.valid.any()
    torch.testing.assert_close(runtime.state.q,endpoint.q)
    assert service.owner.writes==5  # Only original fixture execution; no resume commands.
    with pytest.raises(RuntimeError):budget.reserve()


def test_legacy_resume_version_alias_keeps_contract_checks(resume_case, tmp_path):
    data, runtime, budget, endpoint, original, service = resume_case
    path = tmp_path / "legacy.pt"
    data["version"] = "21.0.1"
    torch.save(data, path)
    assert load_actual_resume(path, service.root)["version"] == "21.0.1"
    data["contract"] = "foreign-contract"
    torch.save(data, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_actual_resume(path, service.root)
