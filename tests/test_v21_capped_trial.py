"""The continuous dispatcher is tested with callbacks, never with hardware."""
import math
import threading

import pytest

from fcpg.v21_camera_condition import ADDRESS_MODES as CAMERA_ADDRESS_MODES
from fcpg.v21_camera_condition import CONDITIONS
from fcpg.v21_capped_trial import (
    ACTION_SOURCE, ADDRESS_MODES, CAMERA_CONDITIONS, FAIL_TYPES, HUMAN_LABELS, CappedTrial)
try:
    from test_v21_live_service import service  # noqa: F401 -- fake IO fixture
except ImportError as exc:
    _LIVE_IMPORT_ERROR = exc
    @pytest.fixture
    def service():
        pytest.skip(f'live service extras unavailable: {_LIVE_IMPORT_ERROR}')


def callbacks(*, plan_hook=None, execute_hook=None):
    calls=[];saved=[];clock=[0.]
    def plan():
        calls.append('plan')
        if plan_hook:plan_hook()
        return dict(ready=True,plan=dict(action_source='v21_actor'))
    def execute():
        calls.append('execute');clock[0]+=1.
        if execute_hook:execute_hook()
        return dict(error=None,stop_error=None,last_execution=dict(
            completed=True,action_source='v21_actor',policy_commands_returned=5,
            commands_position=[[0.]*6]*5,source_time_s=clock[0],endpoint_time_s=clock[0]+.8))
    trial=CappedTrial(plan,execute,lambda:calls.append('halt'),saved.append)
    return trial,calls,saved


def test_exact_sixteen_block_cap_is_not_sixteen_successes():
    t,calls,saved=callbacks();t.start();t.join()
    d=t.status()
    assert calls==['plan','execute']*16+['halt']
    assert d['blocks_completed']==16 and d['policy_commands_returned']==80
    assert d['outcome']=='budget_exhausted' and not d['running']
    assert d['task_success_verified'] is False
    assert d['action_source']=='v21_actor'
    assert d['camera_condition'] is d['address_mode'] is d['reset_pose'] is None
    assert d['grasp']==d['place']==d['fail_type']=='unverified'
    assert all(row['action_source']=='v21_actor' for row in d['records'])
    assert saved[-1]==d
    assert d['records'][1]['inter_block_gap_s']==pytest.approx(.2)
    assert isinstance(d['trial_id'],str) and d['trial_id']


def test_stop_before_start_has_no_plan_or_execute():
    t,calls,_=callbacks();t.start(paused=True);t.cancel();t.join()
    assert calls==['halt'] and t.status()['blocks_completed']==0
    assert t.status()['outcome']=='operator_stopped'


def test_stop_during_plan_cannot_be_cleared_by_normal_block_stop():
    entered=threading.Event();release=threading.Event()
    def hook():entered.set();assert release.wait(2.)
    t,calls,_=callbacks(plan_hook=hook);t.start();assert entered.wait(1.)
    t.cancel();release.set();t.join()
    assert calls==['plan','halt']
    assert t.status()['outcome']=='operator_stopped'


def test_stop_after_execute_prevents_the_next_plan():
    t,calls,_=callbacks();original=t.execute
    def execute():
        result=original();t.cancel();return result
    t.execute=execute;t.start();t.join()
    assert calls==['plan','execute','halt']
    assert t.status()['blocks_completed']==1
    assert t.status()['outcome']=='operator_stopped'


def test_a_release_over_the_box_ends_the_trial_and_verifies_nothing():
    t,calls,_=callbacks();original=t.execute;count=[0]
    def execute():
        result=original();count[0]+=1
        result['last_execution']['alignment_cem']=dict(released_over_box=count[0]==3)
        return result
    t.execute=execute;t.start();t.join()
    d=t.status()
    assert calls==['plan','execute']*3+['halt']
    assert d['outcome']=='released_over_box' and d['blocks_completed']==3
    assert d['task_success_verified'] is False and d['place']=='unverified'
    assert [row['alignment_cem']['released_over_box'] for row in d['records']]==[False,False,True]


def test_failed_execution_stops_and_is_not_retried():
    def fail():raise RuntimeError('partial write / safety stop')
    t,calls,_=callbacks(execute_hook=fail);t.start();t.join()
    assert calls==['plan','execute','halt']
    assert t.status()['blocks_started']==1 and t.status()['blocks_completed']==0
    assert t.status()['outcome']=='execution_error'


def test_partial_block_returned_commands_are_not_lost_from_trial_count():
    t,calls,_=callbacks()
    original=t.execute
    count=[0]
    def execute():
        count[0]+=1
        if count[0]==4:
            raise RuntimeError('read timeout after fourth write')
        return original()
    t.execute=execute
    t.last_attempt=lambda:dict(policy_commands_attempted=4,policy_commands_returned=4,
                              completed=False,error='read timeout')
    t.start();t.join()
    d=t.status()
    assert d['blocks_completed']==3 and d['blocks_started']==4
    assert d['policy_commands_returned']==19
    assert d['partial_policy_commands_returned']==4
    assert d['outcome']=='execution_error' and count[0]==4


def test_assisted_or_unadmitted_plan_is_never_dispatched():
    t,calls,_=callbacks()
    t.plan=lambda:dict(ready=True,plan=dict(action_source='v21_actor_grounding_corrected'))
    t.start();t.join();assert calls==['halt']
    assert t.status()['outcome']=='execution_error'


def test_no_parallel_trial_and_status_is_a_copy():
    t,calls,_=callbacks();t.start(paused=True)
    with pytest.raises(RuntimeError,match='already running'):t.start()
    status=t.status();status['max_blocks']=100
    assert t.status()['max_blocks']==16
    t.cancel();t.join();assert calls==['halt']


def test_budget_cannot_exceed_requested_sixteen():
    with pytest.raises(ValueError,match='budget'):
        CappedTrial(None,None,None,None,max_blocks=33)


def test_service_excludes_other_actions_but_stop_never_waits(service):
    service.trial.start(paused=True)
    with pytest.raises(RuntimeError,match='owns motion'):
        service.handle('execute')
    with pytest.raises(RuntimeError,match='owns motion'):
        service.handle('home')
    service.handle('stop');service.trial.join()
    assert service.owner.writes==0
    assert service.trial.status()['outcome']=='operator_stopped'


def test_service_trial_refuses_action_override(service):
    service.grounding_correction=True
    with pytest.raises(RuntimeError,match='unmodified Actor'):
        service.handle('trial')
    assert service.owner.writes==0 and not service.trial.running


def test_service_runs_sixteen_real_fake_transactions_without_outer_lock_deadlock(service,monkeypatch):
    import fcpg.v21_live_service as live
    monkeypatch.setattr(live.established,'_load_v177_home_state',lambda *a,**k:service.owner.q.copy())
    service.owner.read_checked=service.owner._read_q
    service.owner.io.settle_position_tolerance=3.
    original=service.owner.bridge.execute
    def execute():
        result=original()
        result['policy_commands_returned']=5
        result['endpoint_time_s']=result['readback_times_s'][-1]
        return result
    service.owner.bridge.execute=execute
    service.handle('trial');service.trial.join(timeout=30.)
    result=service.trial.status()
    assert result['outcome']=='budget_exhausted',result
    assert result['blocks_completed']==16 and service.owner.writes==80
    assert service.owner.runtime.stopped
    assert result['task_success_verified'] is False
    assert result['action_source']=='v21_actor'
    assert result['grasp']==result['place']==result['fail_type']=='unverified'
    assert all(row['action_source']=='v21_actor' for row in result['records'])


def test_logging_vocab_matches_camera_condition_module():
    assert CAMERA_CONDITIONS==CONDITIONS
    assert ADDRESS_MODES==CAMERA_ADDRESS_MODES
    assert ACTION_SOURCE=='v21_actor'
    assert HUMAN_LABELS==('unverified','success','fail')
    assert 'unverified' in FAIL_TYPES


def test_start_record_keeps_bound_experiment_identity():
    t,calls,saved=callbacks()
    t.bind_experiment(camera_condition='wrist_h0',address_mode='factual',
                      reset_pose=(1.,2.,3.,4.,5.,6.),action_source='v21_actor')
    t.start();t.join()
    first,last=saved[0],saved[-1]
    assert first['trial_id']==last['trial_id']
    assert first['camera_condition']==last['camera_condition']=='wrist_h0'
    assert first['address_mode']==last['address_mode']=='factual'
    assert first['reset_pose']==last['reset_pose']==[1.,2.,3.,4.,5.,6.]
    assert first['action_source']==last['action_source']=='v21_actor'
    assert first['grasp']==first['place']==first['fail_type']=='unverified'
    assert first['task_success_verified'] is last['task_success_verified'] is False
    assert calls==['plan','execute']*16+['halt']


def test_start_kwargs_are_the_live_service_hook():
    t,_,saved=callbacks()
    t.start(camera_condition='external_cold',address_mode='zero',
            reset_pose=[0,0,0,0,0,31.34])
    t.join()
    d=saved[0]
    assert d['camera_condition']=='external_cold' and d['address_mode']=='zero'
    assert d['reset_pose']==[0.,0.,0.,0.,0.,31.34]
    assert d['action_source']=='v21_actor' and d['task_success_verified'] is False


def test_human_scores_stay_unverified_unless_recorded_and_never_verify_task():
    t,_,saved=callbacks()
    t.start();t.join()
    scored=t.record_human_scores(grasp='success',place='success',fail_type='unverified')
    assert scored['grasp']==scored['place']=='success'
    assert scored['fail_type']=='unverified'
    assert scored['task_success_verified'] is False
    assert saved[-1]['grasp']=='success' and saved[-1]['task_success_verified'] is False
    t.record_human_scores(grasp='fail',place='fail',fail_type='miss')
    assert t.status()['fail_type']=='miss' and t.status()['task_success_verified'] is False
    sheet=t.record_human_scores(grasp='yes',place='no',fail_type='')
    assert sheet['grasp']=='success' and sheet['place']=='fail'
    assert sheet['fail_type']=='unverified' and sheet['task_success_verified'] is False


def test_bind_and_scores_reject_unknown_labels():
    t,_,_=callbacks()
    with pytest.raises(ValueError,match='action_source'):
        t.bind_experiment(action_source='v21_actor_grounding_corrected')
    with pytest.raises(ValueError,match='camera_condition'):
        t.bind_experiment(camera_condition='wrist_only')
    with pytest.raises(ValueError,match='address_mode'):
        t.bind_experiment(address_mode='readback')
    with pytest.raises(ValueError,match='reset pose'):
        t.bind_experiment(reset_pose=[1.,2.,3.])
    with pytest.raises(ValueError,match='reset pose'):
        t.bind_experiment(reset_pose=[1.,2.,3.,4.,5.,math.inf])
    with pytest.raises(ValueError,match='grasp'):
        t.record_human_scores(grasp='maybe')
    with pytest.raises(ValueError,match='fail_type'):
        t.record_human_scores(fail_type='stuck')


def test_cannot_rebind_a_running_trial():
    t,_,_=callbacks();t.start(paused=True)
    with pytest.raises(RuntimeError,match='rebind'):
        t.bind_experiment(camera_condition='both_fresh')
    t.cancel();t.join()
