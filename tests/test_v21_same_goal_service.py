"""Same-Goal service lifecycle with fake owner/callbacks; no hardware/model IO."""
from copy import deepcopy
import json
import os
import stat
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from fcpg import v21_same_goal_service as module
from fcpg.v21_same_goal_service import SameGoalServiceCoordinator, SameGoalServiceError


class Clock:
    def __init__(self): self.now = 10.
    def __call__(self): return self.now
    def wait(self, delay): self.now += delay


def make(tmp_path, monkeypatch, mode='p32', integral_max_seconds=2):
    source = tmp_path/'source-execution.json'
    source_record = dict(completed=True, commands_position=[[0., 0., 3., 0., 0., 0.]]*5,
                         measured_q_path=[[0.]*6]*6)
    if mode == 'i1':
        source_record = dict(schema=module.SCHEMA, mode='p32', state='completed', completed=True,
            diagnostic_attempted=True, stopped=True, motion_blocked=False, cleanup={'restored':True},
            diagnostic_result=dict(schema=module.DIAGNOSTIC_SCHEMA, mode='p32', completed=True,
                original_target_raw={'elbow_flex':3018}, original_target_reached=False,
                stages={'candidate':{'present_raw':{'elbow_flex':3030}}},
                source_execution=source_record))
    source.write_text(json.dumps(source_record))
    clock = Clock()
    runtime = SimpleNamespace(stopped=True, pending=None, executing=False, state=object(),
                             requires_home_after_incomplete_execution=False)
    owner = SimpleNamespace(runtime=runtime, lease=object(), stop_event=threading.Event(),
                            bus_lock=threading.RLock(), bridge=SimpleNamespace(runtime=runtime))
    owner.stop_event.set()
    service = SimpleNamespace(owner=owner, output=tmp_path, operation_lock=threading.Lock(),
        display_lock=threading.RLock(), trial=SimpleNamespace(running=False),
        auxiliary_budget=SimpleNamespace(active=False), clock=clock, same_goal_wait=clock.wait,
        _trial_admission_in_progress=False, epoch=0, calls=[], clears=0,
        plan_record=object(), preview_record=object(), last_execution=object())
    service.ros_control_admission_epoch = lambda: service.epoch
    def lease_check():
        assert service.operation_lock.locked() and owner.bus_lock._is_owned()
        service.calls.append('lease')
    owner._lease_check = lease_check
    def read_checked():
        assert owner.stop_event.is_set() and owner.bus_lock._is_owned()
        service.calls.append('actual_q')
        return np.array([0., 0., 2.6, 0., 0., 0.], np.float32)
    owner.read_checked = read_checked
    owner._physical_check = lambda q, commands: service.calls.append('geometry')
    service._reset_camera_condition = lambda: service.calls.append('reset_condition')
    service._assistance = SimpleNamespace(reset=lambda: service.calls.append('reset_assistance'))
    def stop():
        service.calls.append('runtime_stop')
        runtime.stopped=True;runtime.pending=None;runtime.executing=False
    runtime.stop=stop
    def clear():
        service.calls.append('clear');service.clears+=1;owner.stop_event.clear()
    service._clear_stop_for_current_operation=clear
    def run(owner_arg, execution, **kwargs):
        assert owner_arg is owner and owner.bus_lock._is_owned() and service.operation_lock.locked()
        intent=json.loads((tmp_path/f'same-goal-{mode}-result.json').read_text())
        assert intent['state']=='started' and intent['diagnostic_attempted'] is False
        assert intent['integral_max_seconds']==integral_max_seconds
        assert kwargs['mode']==mode and kwargs['enabled'] is True and not kwargs['cancelled']()
        assert kwargs['integral_max_seconds']==integral_max_seconds
        assert execution==json.loads(source.read_text())
        service.calls.append('helper');clock.wait(6.)
        owner.stop_event.set()
        return dict(schema='v21-same-goal-gain-diagnostic-v1',mode=mode,completed=True,stopped=True,
            original_target=[0.,0.,3.,0.,0.,0.],original_target_raw={'elbow_flex':3018},
            original_target_arrival={'reached':False,'max_abs_residual':.4},original_target_reached=False,
            cleanup={'restored':True,'gain_restored':True,'integral_zero_verified':True,
                     'hold_verified':True,'torque_restored':True,'errors':[]},
            motor_register_writes=([{'register':'P_Coefficient','value':32},
                                    {'register':'P_Coefficient','value':16}]
                                  if mode=='p32' else [{'register':'I_Coefficient','value':1},
                                                       {'register':'I_Coefficient','value':0}]))
    monkeypatch.setattr(module,'_run_diagnostic',run)
    coordinator=SameGoalServiceCoordinator(service,enabled=True,source_execution_path=source,mode=mode,
        integral_max_seconds=integral_max_seconds,output=tmp_path)
    return service,coordinator,run


def invoke(service, coordinator):
    with service.operation_lock:return coordinator.handle()


@pytest.mark.parametrize('mode', ['p32', 'i1'])
def test_disabled_never_touches_owner_clock_or_creates_a_receipt(tmp_path,mode):
    class Forbidden:
        def __getattr__(self,name):pytest.fail('disabled touched '+name)
    coordinator=SameGoalServiceCoordinator(Forbidden(),mode=mode,output=tmp_path)
    with pytest.raises(SameGoalServiceError,match='disabled'):coordinator.handle()
    assert coordinator.status()['consumed'] is False and not list(tmp_path.iterdir())


@pytest.mark.parametrize('enabled',[1,0,None,'yes'])
def test_enable_requires_bool(tmp_path,enabled):
    with pytest.raises(TypeError):SameGoalServiceCoordinator(object(),enabled=enabled,output=tmp_path)


@pytest.mark.parametrize('enabled,mode,maximum,message',
    [(False,'p32',2,'disabled'),(False,'i1',2,'disabled'),(False,'i1',10,'disabled'),
     (True,'i2',2,'only explicit'),(True,'i1',5,'integral_max_seconds')])
def test_real_helper_import_rejects_disabled_or_unsupported_before_any_owner_io(enabled,mode,maximum,message):
    class Forbidden:
        def __getattr__(self,name):pytest.fail('disabled helper touched '+name)
    with pytest.raises(RuntimeError,match=message) as caught:
        module._run_diagnostic(Forbidden(),None,enabled=enabled,mode=mode,integral_max_seconds=maximum)
    report=caught.value.report
    assert report['schema']=='v21-same-goal-gain-diagnostic-v1'
    assert report['completed'] is False
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('mode', ['p32', 'i1'])
def test_malformed_helper_report_still_latches_stop_and_blocks_motion(tmp_path,monkeypatch,mode):
    service,coordinator,_=make(tmp_path,monkeypatch,mode)
    monkeypatch.setattr(module,'_run_diagnostic',lambda *a,**kw:['invalid report'])
    with pytest.raises(SameGoalServiceError,match='invalid report'):
        invoke(service,coordinator)
    report=json.loads(coordinator.path.read_text())
    assert report['state']=='failed' and report['motion_blocked'] and report['stopped']
    assert service.owner.stop_event.is_set() and service.owner.runtime.stopped
    assert service.calls.count('runtime_stop')==1


@pytest.mark.parametrize('mode', ['p32', 'i1'])
def test_supported_modes_require_an_explicit_source(tmp_path,mode):
    with pytest.raises(ValueError,match='source_execution_path'):
        SameGoalServiceCoordinator(object(),enabled=True,mode=mode,output=tmp_path)


@pytest.mark.parametrize('mode', ['i2', 'p64', None, ''])
def test_unsupported_mode_is_rejected_before_owner_access(tmp_path,mode):
    with pytest.raises(ValueError,match='only explicit p32 or i1'):
        SameGoalServiceCoordinator(object(),mode=mode,output=tmp_path)


@pytest.mark.parametrize('mode',['p32','i1'])
@pytest.mark.parametrize('maximum',[True,False,None,'2',0,1,5,-2,float('inf'),float('nan')])
def test_invalid_integral_maximum_is_rejected_before_service_access(tmp_path,mode,maximum):
    class Forbidden:
        def __getattr__(self,name):pytest.fail('invalid maximum touched '+name)
    with pytest.raises(ValueError,match='integral_max_seconds'):
        SameGoalServiceCoordinator(Forbidden(),mode=mode,integral_max_seconds=maximum,output=tmp_path)
    assert not list(tmp_path.iterdir())


def test_p32_cannot_use_extended_integral_maximum(tmp_path):
    with pytest.raises(ValueError,match='integral_max_seconds'):
        SameGoalServiceCoordinator(object(),mode='p32',integral_max_seconds=10,output=tmp_path)


@pytest.mark.parametrize('mode,maximum',[('p32',2),('p32',2.0),('i1',2),('i1',10),('i1',10.0)])
def test_integral_maximum_is_passed_to_helper_and_persisted(tmp_path,monkeypatch,mode,maximum):
    service,coordinator,_=make(tmp_path,monkeypatch,mode,maximum)
    assert coordinator.status()['integral_max_seconds']==maximum
    source_bytes=coordinator.source_execution_path.read_bytes()
    report=invoke(service,coordinator)
    assert report['completed'] and report['integral_max_seconds']==maximum
    assert json.loads(coordinator.path.read_text())['integral_max_seconds']==maximum
    assert coordinator.source_execution_path.read_bytes()==source_bytes
    assert service.calls.count('helper')==1


@pytest.mark.parametrize('fault',['unlocked','owner','lease','runtime','stopped','pending','executing',
                                  'unlatched','incomplete','trial','trial_admission','auxiliary'])
def test_bad_entry_has_no_owner_callback_clear_or_result(tmp_path,monkeypatch,fault):
    service,coordinator,_=make(tmp_path,monkeypatch);owner=service.owner
    if fault=='owner':service.owner=None
    elif fault=='lease':owner.lease=None
    elif fault=='runtime':owner.bridge.runtime=object()
    elif fault=='stopped':owner.runtime.stopped=False
    elif fault=='pending':owner.runtime.pending=object()
    elif fault=='executing':owner.runtime.executing=True
    elif fault=='unlatched':owner.stop_event.clear()
    elif fault=='incomplete':owner.runtime.requires_home_after_incomplete_execution=True
    elif fault=='trial':service.trial.running=True
    elif fault=='trial_admission':service._trial_admission_in_progress=True
    elif fault=='auxiliary':service.auxiliary_budget.active=True
    with pytest.raises(SameGoalServiceError):
        coordinator.handle() if fault=='unlocked' else invoke(service,coordinator)
    assert service.calls==[] and not coordinator.path.exists()


@pytest.mark.parametrize('mode', ['p32', 'i1'])
def test_restored_diagnostic_clears_history_stays_stopped_and_never_calls_home(tmp_path,monkeypatch,mode):
    service,coordinator,_=make(tmp_path,monkeypatch,mode)
    original=coordinator.source_execution_path.read_bytes()
    report=invoke(service,coordinator)
    assert report['completed'] and report['stopped'] and not report['motion_blocked']
    assert report['original_target_reached'] is False
    assert report['model_history_cleared'] and not report['model_transition']
    assert not report['diagnostic_samples_added_to_model_y']
    assert service.owner.runtime.state is None and service.owner.runtime.stopped
    assert not service.owner.runtime.requires_home_after_incomplete_execution
    assert service.owner.stop_event.is_set() and service.clears==1
    assert service.plan_record is service.preview_record is service.last_execution is None
    assert service.current_q==report['fresh_actual_state']['q_position']
    assert coordinator.source_execution_path.read_bytes()==original
    assert json.loads(coordinator.path.read_text())==report
    assert service.calls.count('helper')==1
    assert coordinator.path.name==f'same-goal-{mode}-result.json'
    coordinator.require_motion_unsealed('plan')
    restarted=SameGoalServiceCoordinator(service,enabled=True,source_execution_path=coordinator.source_execution_path,
                                         mode=mode,output=tmp_path)
    restarted.require_motion_unsealed('plan')
    for existing in [coordinator,restarted]:
        with pytest.raises(SameGoalServiceError,match='consumed'):invoke(service,existing)
    assert service.calls.count('helper')==1


def test_intent_file_and_directory_are_durable_before_any_owner_callback(tmp_path,monkeypatch):
    service,coordinator,_=make(tmp_path,monkeypatch);syncs=[];original=module.os.fsync
    def sync(fd):
        if len(syncs)<2:assert service.calls==[]
        syncs.append(stat.S_ISDIR(os.fstat(fd).st_mode));original(fd)
    monkeypatch.setattr(module.os,'fsync',sync)
    invoke(service,coordinator)
    assert syncs[:2]==[False,True]


@pytest.mark.parametrize('kind',['partial','directory','symlink'])
def test_any_prior_result_blocks_replay_without_reads(tmp_path,monkeypatch,kind):
    service,coordinator,_=make(tmp_path,monkeypatch)
    if kind=='partial':coordinator.path.write_text('{"state":"started"}')
    elif kind=='directory':coordinator.path.mkdir()
    else:coordinator.path.symlink_to(tmp_path/'absent')
    with pytest.raises(SameGoalServiceError,match='consumed'):invoke(service,coordinator)
    assert service.calls==[] and coordinator.status()['consumed']


def test_stop_epoch_changed_during_intent_does_not_rearm(tmp_path,monkeypatch):
    service,coordinator,_=make(tmp_path,monkeypatch);original=coordinator._reserve
    def reserve(report):original(report);service.epoch+=1
    monkeypatch.setattr(coordinator,'_reserve',reserve)
    with pytest.raises(SameGoalServiceError,match='cancelled'):invoke(service,coordinator)
    assert service.clears==0 and 'helper' not in service.calls
    assert service.owner.stop_event.is_set() and coordinator.last_result['stopped']
    assert json.loads(coordinator.path.read_text())['state']=='failed'


@pytest.mark.parametrize('restored',[False,True])
@pytest.mark.parametrize('mode',['p32','i1'])
def test_failed_helper_keeps_partial_report_and_blocks_only_unrestored_state(tmp_path,monkeypatch,restored,mode):
    service,coordinator,original=make(tmp_path,monkeypatch,mode)
    def fail(*args,**kwargs):
        report=original(*args,**kwargs);report.update(completed=False)
        report['cleanup']['restored']=restored
        error=RuntimeError('diagnostic read failed');error.report=report;raise error
    monkeypatch.setattr(module,'_run_diagnostic',fail)
    with pytest.raises(RuntimeError,match='diagnostic read failed'):invoke(service,coordinator)
    report=coordinator.last_result
    assert not report['completed'] and report['diagnostic_result']['completed'] is False
    assert report['cleanup']['restored'] is restored
    assert report['motion_blocked'] is (not restored)
    assert report['stopped'] and service.owner.runtime.state is None
    assert not service.owner.runtime.requires_home_after_incomplete_execution
    assert json.loads(coordinator.path.read_text())==report
    for action in ['status','observe','actuator-observe','stop']:coordinator.require_motion_unsealed(action)
    restarted=SameGoalServiceCoordinator(service,source_execution_path=coordinator.source_execution_path,output=tmp_path)
    if restored:restarted.require_motion_unsealed('plan')
    else:
        for action in ['plan','execute','home','trial','same-goal-p32','same-goal-i1']:
            with pytest.raises(SameGoalServiceError,match='blocked'):restarted.require_motion_unsealed(action)


def test_return_gap_stop_cannot_be_reported_as_success(tmp_path,monkeypatch):
    service,coordinator,original=make(tmp_path,monkeypatch)
    def stopped(*args,**kwargs):
        report=original(*args,**kwargs);service.epoch+=1;return report
    monkeypatch.setattr(module,'_run_diagnostic',stopped)
    with pytest.raises(SameGoalServiceError,match='cancelled'):invoke(service,coordinator)
    assert coordinator.last_result['completed'] is False
    assert coordinator.last_result['diagnostic_result']['completed'] is True
    assert coordinator.last_result['cleanup']['restored'] is True


def test_new_actual_geometry_failure_blocks_motion_without_home_or_history(tmp_path,monkeypatch):
    service,coordinator,_=make(tmp_path,monkeypatch)
    def invalid(*args):raise RuntimeError('fresh actual geometry failed')
    service.owner._physical_check=invalid
    with pytest.raises(RuntimeError,match='fresh actual geometry'):invoke(service,coordinator)
    assert coordinator.last_result['motion_blocked'] is True
    assert service.owner.runtime.state is None and service.owner.stop_event.is_set()
    assert 'fresh_actual_state' not in coordinator.last_result


def test_final_persistence_failure_preserves_primary_error_and_blocks(tmp_path,monkeypatch):
    service,coordinator,original=make(tmp_path,monkeypatch)
    def fail(*args,**kwargs):
        report=original(*args,**kwargs);report['completed']=False
        error=RuntimeError('primary read failure');error.report=report;raise error
    monkeypatch.setattr(module,'_run_diagnostic',fail)
    monkeypatch.setattr(module,'_durable_json',lambda *args:(_ for _ in ()).throw(OSError('final disk failure')))
    with pytest.raises(RuntimeError,match='primary read failure'):invoke(service,coordinator)
    assert coordinator.last_result['error']=='RuntimeError: primary read failure'
    assert 'final disk failure' in coordinator.last_result['persistence_error']
    assert coordinator.status()['motion_blocked'] and service.owner.stop_event.is_set()
    assert json.loads(coordinator.path.read_text())['state']=='started'


def test_nonobject_source_is_rejected_before_reservation_or_stop_clear(tmp_path,monkeypatch):
    service,coordinator,_=make(tmp_path,monkeypatch)
    coordinator.source_execution_path.write_text('[]')
    with pytest.raises(SameGoalServiceError,match='saved JSON object'):invoke(service,coordinator)
    assert service.calls==[] and not coordinator.path.exists()


def test_i1_consumes_complete_p32_json_without_overwriting_or_unwrapping_it(tmp_path,monkeypatch):
    service,configured,_=make(tmp_path,monkeypatch,mode='i1')
    source_bytes=configured.source_execution_path.read_bytes()
    p32_path=configured.result_paths['p32']
    p32_path.write_bytes(source_bytes)
    coordinator=SameGoalServiceCoordinator(service,enabled=True,source_execution_path=p32_path,
                                           mode='i1',output=tmp_path)
    status=coordinator.status()
    assert status['previous_result'] is None and not status['consumed']
    assert status['previous_results']['p32']==json.loads(source_bytes)
    coordinator.require_motion_unsealed('same-goal-i1')
    report=invoke(service,coordinator)
    assert report['completed'] and report['mode']=='i1'
    assert report['source_execution_path']==str(p32_path)
    assert report['original_target_raw']['elbow_flex']==3018
    assert report['original_target_reached'] is False
    assert p32_path.read_bytes()==source_bytes
    assert service.calls.count('helper')==1
    restarted=SameGoalServiceCoordinator(service,output=tmp_path)
    assert set(restarted.status()['previous_results'])=={'p32','i1'}
    restarted.require_motion_unsealed('plan')


@pytest.mark.parametrize('mode',['p32','i1'])
def test_source_cannot_be_its_own_mode_result(tmp_path,mode):
    with pytest.raises(ValueError,match='separate'):
        SameGoalServiceCoordinator(object(),enabled=True,mode=mode,output=tmp_path,
            source_execution_path=tmp_path/f'same-goal-{mode}-result.json')


@pytest.mark.parametrize('selected_mode',['p32','i1'])
@pytest.mark.parametrize('prior_kind',['failed','started','schema','mode','cleanup','directory','symlink'])
def test_sibling_result_failure_blocks_restart_even_when_disabled(tmp_path,monkeypatch,selected_mode,prior_kind):
    service,configured,_=make(tmp_path,monkeypatch,selected_mode)
    sibling='i1' if selected_mode=='p32' else 'p32'
    path=configured.result_paths[sibling]
    record=dict(schema=module.SCHEMA,mode=sibling,state='failed',diagnostic_attempted=True,
        cleanup={'restored':False},motion_blocked=True,stopped=True)
    if prior_kind=='started':
        record.update(state='started',cleanup={'restored':True},motion_blocked=False)
    elif prior_kind=='schema':record['schema']='unknown'
    elif prior_kind=='mode':record['mode']=selected_mode
    elif prior_kind=='cleanup':record['cleanup']=[]
    if prior_kind=='directory':path.mkdir()
    elif prior_kind=='symlink':path.symlink_to(tmp_path/'missing-result')
    else:path.write_text(json.dumps(record))
    restarted=SameGoalServiceCoordinator(service,mode=selected_mode,output=tmp_path)
    assert restarted.status()['motion_blocked'] and not restarted.status()['consumed']
    for action in ['plan','execute','home','trial','same-goal-p32','same-goal-i1']:
        with pytest.raises(SameGoalServiceError,match='ordinary motion is blocked'):
            restarted.require_motion_unsealed(action)
    for action in ['status','observe','actuator-observe','stop']:
        restarted.require_motion_unsealed(action)
    assert service.calls==[]


@pytest.mark.parametrize('mode',['p32','i1'])
@pytest.mark.parametrize('field',['schema','mode'])
def test_mismatched_helper_report_is_preserved_but_cannot_verify_restoration(tmp_path,monkeypatch,mode,field):
    service,coordinator,original=make(tmp_path,monkeypatch,mode)
    def mismatch(*args,**kwargs):
        report=original(*args,**kwargs)
        report[field]='other-schema' if field=='schema' else ('i1' if mode=='p32' else 'p32')
        return report
    monkeypatch.setattr(module,'_run_diagnostic',mismatch)
    with pytest.raises(SameGoalServiceError,match='without complete stage evidence'):
        invoke(service,coordinator)
    saved=json.loads(coordinator.path.read_text())
    expected=module.DIAGNOSTIC_SCHEMA if field=='schema' else mode
    assert saved['diagnostic_result'][field]!=expected
    assert saved['state']=='failed' and saved['motion_blocked'] and saved['stopped']
    restarted=SameGoalServiceCoordinator(service,output=tmp_path)
    with pytest.raises(SameGoalServiceError,match='ordinary motion is blocked'):
        restarted.require_motion_unsealed('plan')


@pytest.mark.parametrize('mode',['p32','i1'])
def test_malformed_cleanup_still_stops_and_blocks_motion(tmp_path,monkeypatch,mode):
    service,coordinator,original=make(tmp_path,monkeypatch,mode)
    def malformed(*args,**kwargs):
        report=original(*args,**kwargs);report['cleanup']=['unverified'];return report
    monkeypatch.setattr(module,'_run_diagnostic',malformed)
    with pytest.raises(SameGoalServiceError,match='restoration is incomplete'):
        invoke(service,coordinator)
    assert coordinator.last_result['state']=='failed' and coordinator.last_result['motion_blocked']
    assert coordinator.last_result['stopped'] and service.calls.count('runtime_stop')==1
    assert coordinator.last_result['diagnostic_result']['cleanup']==['unverified']
