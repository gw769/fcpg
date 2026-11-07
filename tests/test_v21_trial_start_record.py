"""Trial-start persistence tests use callbacks only; no robot IO.

CLOSED_LOOP admission already lives in V21LiveService.handle('trial'): HOME
pose, empty history, no pending/executing, and no grounding/head assist.
CappedTrial.start only records that already-admitted empty ledger.
"""
import threading

import pytest

from fcpg.v21_capped_trial import CappedTrial


def test_started_record_is_saved_before_first_plan_and_remains_if_plan_fails():
    saved=[];calls=[]
    def plan():
        assert len(saved)==1
        assert saved[0]['running'] and saved[0]['blocks_started']==0
        assert saved[0]['task_success_verified'] is False
        calls.append('plan')
        raise RuntimeError('first plan refused')
    trial=CappedTrial(plan,lambda:calls.append('execute'),
                      lambda:calls.append('halt'),saved.append)
    trial.start();trial.join()
    assert calls==['plan','halt']
    assert saved[0]['trial_id']==saved[-1]['trial_id']
    assert saved[-1]['outcome']=='execution_error'
    assert saved[-1]['blocks_started']==saved[-1]['policy_commands_returned']==0
    # Post-admission ledger: no assist flag, no history, no pending motion.
    assert saved[0]['records']==[] and saved[0]['task_success_verified'] is False


def test_failed_start_record_never_starts_a_worker():
    calls=[]
    def save(_): raise OSError('disk unavailable')
    trial=CappedTrial(lambda:calls.append('plan'),lambda:calls.append('execute'),
                      lambda:calls.append('halt'),save)
    with pytest.raises(OSError,match='disk unavailable'): trial.start()
    assert not calls and trial.thread is None and not trial.running
    assert trial.status()['outcome']=='start_log_failed'


def test_cancel_during_start_record_is_not_cleared_before_worker_dispatch():
    saved=[];calls=[]
    def save(data):
        saved.append(data)
        if len(saved)==1: trial.cancel()
    trial=CappedTrial(lambda:calls.append('plan'),lambda:calls.append('execute'),
                      lambda:calls.append('halt'),save)
    trial.start();trial.join()
    assert calls==['halt']
    assert saved[-1]['outcome']=='operator_stopped'


def test_paused_start_is_on_disk_before_release():
    saved=[];calls=[]
    trial=CappedTrial(lambda:calls.append('plan'),lambda:calls.append('execute'),
                      lambda:calls.append('halt'),saved.append)
    trial.start(paused=True)
    try:
        assert len(saved)==1 and saved[0]['running'] and not calls
    finally:
        trial.cancel();trial.join()
    assert calls==['halt']


def test_worker_start_failure_updates_the_existing_trial_record(monkeypatch):
    saved=[];calls=[]
    def failed_start(_): raise RuntimeError('thread unavailable')
    monkeypatch.setattr(threading.Thread,'start',failed_start)
    trial=CappedTrial(lambda:calls.append('plan'),lambda:calls.append('execute'),
                      lambda:calls.append('halt'),saved.append)
    with pytest.raises(RuntimeError,match='thread unavailable'): trial.start()
    assert not calls and not trial.running
    trial.join()  # cleanup must not try to join a never-started thread
    assert len(saved)==2 and saved[0]['trial_id']==saved[1]['trial_id']
    assert saved[-1]['outcome']=='worker_start_failed'
