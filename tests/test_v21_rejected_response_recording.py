"""Complete rejected responses remain factual training evidence; fake IO only."""
import json
from pathlib import Path

import pytest
import torch

import fcpg.v21_live_service as live
from test_v21_live_service import service  # noqa: F401
from test_v21_partial_failure_audit import install_bridge


def execute_rejected(service):
    with pytest.raises(ValueError, match='actual.*response|respond'):
        service.handle('execute')


def test_complete_rejection_saves_real_schema2_input_and_execution_after_hold(service, monkeypatch):
    bridge, writes, _ = install_bridge(service, monkeypatch, stall_j3=True)
    assert service.record_policy_inputs is False
    service.handle('plan')
    events=[]
    hold=service.owner._halt
    def halt():
        events.append('rejection_hold')
        return hold()
    monkeypatch.setattr(service.owner,'_halt',halt)
    pack=live.cpu_actual_state
    def snapshot(source):
        events.append('snapshot')
        return pack(source)
    monkeypatch.setattr(live,'cpu_actual_state',snapshot)
    save=torch.save
    def save_tensor(*args,**kwargs):
        events.append('torch_save')
        return save(*args,**kwargs)
    monkeypatch.setattr(live.torch,'save',save_tensor)
    write=live.write_json
    def save_json(path,value):
        events.append(Path(path).name.split('-')[0])
        return write(path,value)
    monkeypatch.setattr(live,'write_json',save_json)
    execute_rejected(service)
    assert events.index('rejection_hold') < events.index('snapshot') < events.index('torch_save') < events.index('execution')
    assert len(writes)==5
    files=list(service.output.glob('execution-*.json'))
    assert len(files)==1 and len(list(service.output.glob('policy-input-*.pt')))==1
    record=json.loads(files[0].read_text())
    assert record['completed'] is True and record['response_verified'] is False
    assert 'actual arm response' in record['response_error']
    diagnostic = record['actual_cem_response_diagnostic']
    assert diagnostic['blocked_arm_joint_indices'] == [2]
    assert diagnostic['minimum_motion'] == service.owner.io.settle_step_tolerance
    assert diagnostic['predictions_used_as_evidence'] is False
    assert record['execution_record']['saved'] is True
    assert record['execution_record']['path']==files[0].name
    assert record['execution_record']['task_success_inferred'] is False
    payload=torch.load(service.output/record['policy_input_snapshot'],weights_only=False)
    assert payload['kind']=='actual-actor-input' and payload['schema']==2 and payload['executable'] is False
    torch.testing.assert_close(payload['state'].q,bridge.last_actor_input.q)
    assert not payload['state'].history.valid.any()
    assert service.owner.runtime.state.history.valid.any()
    assert payload['commands_position']==record['commands_position']
    assert payload['dispatch_source_q_position']==record['source_q_position']
    assert service.execution_history[-1]['response_verified'] is False
    assert service.execution_history[-1]['response_error']==record['response_error']
    assert service.execution_history[-1]['actual_cem_response_diagnostic'] == diagnostic
    assert not service.status()['ready'] and service.owner.stop_event.is_set()


@pytest.mark.parametrize('returned',[0,2])
def test_partial_failure_never_fills_a_schema2_sample(service,monkeypatch,returned):
    bridge,writes,_=install_bridge(service,monkeypatch,fail_after=returned)
    service.record_policy_inputs=True
    service.handle('plan')
    with pytest.raises(RuntimeError,match='synthetic actual clearance'):
        service.handle('execute')
    assert len(writes)==returned and bridge.last_attempt['completed'] is False
    assert not list(service.output.glob('execution-*.json'))
    assert not list(service.output.glob('policy-input-*.pt'))
    assert 'execution_record' not in bridge.last_attempt


def test_success_and_saved_record_are_not_saved_twice(service,monkeypatch):
    bridge,writes,_=install_bridge(service,monkeypatch)
    service.record_policy_inputs=True
    service.handle('plan');service.handle('execute')
    record=bridge.last_attempt
    assert record['response_verified'] is True
    before=(len(list(service.output.glob('execution-*.json'))),len(list(service.output.glob('policy-input-*.pt'))))
    service._save_execution(record)
    service._save_rejected_complete_response(record)
    assert before==(1,1)
    assert before==(len(list(service.output.glob('execution-*.json'))),len(list(service.output.glob('policy-input-*.pt'))))
    assert len(service.execution_history)==1 and len(writes)==5


def test_execute_without_dispatch_cannot_resave_an_old_rejected_attempt(service,monkeypatch):
    bridge,writes,_=install_bridge(service,monkeypatch,stall_j3=True)
    service.handle('plan');execute_rejected(service)
    old=bridge.last_attempt
    old.pop('execution_record');old.pop('policy_input_snapshot')
    last_path=service.output/'last-attempt.json'
    old_bytes=last_path.read_bytes();old_mtime=last_path.stat().st_mtime_ns
    monkeypatch.setattr(service,'_save_execution',lambda *args,**kwargs:pytest.fail('old attempt must not be saved'))
    with pytest.raises(RuntimeError,match='explicit plan'):
        service.handle('execute')
    assert bridge.last_attempt is old and len(writes)==5
    assert last_path.read_bytes()==old_bytes and last_path.stat().st_mtime_ns==old_mtime
    assert len(list(service.output.glob('execution-*.json')))==1


@pytest.mark.parametrize('failure',['snapshot','execution_json','all_json'])
def test_logging_failure_never_masks_the_original_response_rejection(service,monkeypatch,failure):
    bridge,writes,_=install_bridge(service,monkeypatch,stall_j3=True)
    service.handle('plan')
    if failure=='snapshot':
        def fail(*args,**kwargs):raise OSError('synthetic snapshot disk full')
        monkeypatch.setattr(live.torch,'save',fail)
    else:
        original=live.write_json
        def fail(path,value):
            if failure=='all_json' or Path(path).name.startswith('execution-'):
                raise OSError('synthetic JSON disk full')
            return original(path,value)
        monkeypatch.setattr(live,'write_json',fail)
    execute_rejected(service)
    assert len(writes)==5 and service.owner.holds>=2
    assert service.error.startswith('CEMActualResponseError:') and 'actual arm response' in service.error
    assert bridge.last_attempt['response_verified'] is False
    assert bridge.last_attempt['execution_record']['saved'] is False
    assert 'disk full' in bridge.last_attempt['execution_recording_error']
    if failure=='all_json':assert 'disk full' in service.last_attempt['final_recording_error']
    assert not service.operation_lock.locked()


def test_error_after_execution_file_commit_does_not_duplicate_that_file(service,monkeypatch):
    bridge,writes,_=install_bridge(service,monkeypatch,stall_j3=True)
    service.handle('plan')
    def fail_publish():raise RuntimeError('synthetic presentation failure')
    monkeypatch.setattr(service,'_publish',fail_publish)
    execute_rejected(service)
    assert bridge.last_attempt['execution_record']['saved'] is True
    assert 'presentation failure' in bridge.last_attempt['execution_recording_error']
    service._save_rejected_complete_response(bridge.last_attempt)
    assert len(list(service.output.glob('execution-*.json')))==1
    assert len(list(service.output.glob('policy-input-*.pt')))==1
    assert len(writes)==5
