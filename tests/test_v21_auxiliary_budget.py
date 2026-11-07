import json

import pytest

from fcpg.v21_auxiliary_budget import AuxiliaryMotionBudget
from fcpg.v21_training import write_json
from test_v21_live_service import service  # noqa: F401


def test_ten_block_budget_counts_preparation_and_return_like_correction():
    budget=AuxiliaryMotionBudget()
    for _ in range(10):
        budget.reserve();budget.finish(dict(policy_commands_attempted=5,policy_commands_returned=5,completed=True))
    assert budget.status()['policy_commands_returned']==50
    with pytest.raises(RuntimeError,match='exhausted'):budget.reserve()


def test_explicit_thirty_extension_preserves_every_old_command_and_home_requirement(tmp_path):
    budget=AuxiliaryMotionBudget()
    for _ in range(10):
        budget.reserve();budget.finish(dict(policy_commands_attempted=5,policy_commands_returned=5,completed=True))
    budget.requires_home=True
    before=budget.status();budget.extend_to_thirty()
    assert budget.block_attempts==10 and budget.returned==50 and budget.requires_home
    assert budget.records==before['records'] and budget.maximum_blocks==30
    budget.reserve_home();budget.finish(dict(policy_commands_attempted=5,policy_commands_returned=5,completed=True))
    save_budget(tmp_path,budget)
    restored=AuxiliaryMotionBudget.restore_latest(tmp_path)
    assert restored.maximum_blocks==30 and restored.block_attempts==11 and restored.returned==55
    assert restored.requires_home


def test_thirty_is_still_a_bound_not_unlimited_motion():
    budget=AuxiliaryMotionBudget(maximum_blocks=30)
    for _ in range(30):budget.reserve();budget.finish(None)
    with pytest.raises(RuntimeError,match='exhausted'):budget.reserve()
    with pytest.raises(RuntimeError,match='no auxiliary budget'):budget.reserve_home()


def test_service_budget_extension_is_explicit_stopped_and_never_a_motor_command(service):
    b=service.auxiliary_budget;b.reserve();b.finish(None)
    with pytest.raises(ValueError):service.handle('auxiliary-budget',{'maximum_blocks':50})
    service.handle('auxiliary-budget',{'maximum_blocks':30})
    assert b.maximum_blocks==30 and b.block_attempts==1 and service.owner.writes==0
    restored=AuxiliaryMotionBudget.restore_latest(service.output)
    assert restored.maximum_blocks==30 and restored.block_attempts==1


def test_home_reset_keeps_authorized_cap_but_preserves_closed_old_record(service):
    b=service.auxiliary_budget;b.reserve();b.finish(None);b.extend_to_thirty()
    identity=b.trial_id;service._close_auxiliary_after_home()
    assert service.auxiliary_budget.maximum_blocks==30 and not service.auxiliary_budget.active
    old=json.loads((service.output/f'auxiliary-budget-{identity}.json').read_text())
    assert old['block_attempts']==1 and old['maximum_blocks']==30 and old['closed_by_home']


def test_failed_budget_write_rolls_back_without_clearing_counts(service,monkeypatch):
    b=service.auxiliary_budget;b.reserve();b.finish(None)
    def fail():raise OSError('disk full')
    monkeypatch.setattr(service,'_persist_auxiliary_budget',fail)
    with pytest.raises(OSError):service.handle('auxiliary-budget',{'maximum_blocks':30})
    assert b.maximum_blocks==10 and b.block_attempts==1 and service.owner.writes==0


def test_partial_unknown_and_zero_write_cases_remain_distinct():
    budget=AuxiliaryMotionBudget();budget.reserve();budget.finish(None)
    assert budget.status()['policy_commands_returned']==0
    budget.reserve();budget.finish(dict(policy_commands_attempted=4,policy_commands_returned=3,completed=False))
    assert budget.status()['policy_commands_attempted']==4 and budget.status()['policy_commands_returned']==3
    budget.reserve();budget.finish({})
    assert budget.status()['accounting_unknown']
    with pytest.raises(RuntimeError):budget.reserve()


def test_resetting_alignment_does_not_reset_motion_budget(service):
    service.auxiliary_budget.reserve()
    service.auxiliary_budget.finish(dict(policy_commands_attempted=5,policy_commands_returned=5,completed=True))
    service.handle('alignment-reset')
    assert service.status()['auxiliary_trial']['policy_commands_returned']==5


def test_active_auxiliary_budget_cannot_mix_in_unmetered_actor_actions(service):
    service.auxiliary_budget.reserve();service.auxiliary_budget.finish(None)
    for action in ('execute','trial'):
        with pytest.raises(RuntimeError,match='auxiliary experiment'):service.handle(action)
    assert service.owner.writes==0


def test_auxiliary_motion_requires_home_before_any_probe(service,monkeypatch):
    import numpy as np
    import fcpg.v21_live_service as live
    service.alignment_motion_enabled=True
    service.owner.read_checked=service.owner._read_q
    service.owner.io.settle_position_tolerance=3.
    monkeypatch.setattr(live.established,'_load_v177_home_state',lambda *a,**k:np.full(6,20.))
    with pytest.raises(RuntimeError,match='start at HOME'):
        service._run_alignment_motion(lambda state:None,'visual_alignment_probe')
    assert service.owner.writes==0 and not service.auxiliary_budget.active


def test_probe_cannot_masquerade_as_actor_output(service):
    service.alignment_motion_enabled=True
    with pytest.raises(ValueError,match='raw Actor'):
        service._run_alignment_motion(lambda state:None,'v21_actor')
    assert service.owner.writes==0


def save_budget(root,budget,value=None):
    path=root/f'auxiliary-budget-{budget.trial_id}.json'
    write_json(path,budget.status() if value is None else value)
    return path


def test_one_reservation_has_exactly_one_completion():
    budget=AuxiliaryMotionBudget()
    with pytest.raises(RuntimeError,match='no current'):budget.finish(None)
    budget.reserve()
    with pytest.raises(RuntimeError,match='already pending'):budget.reserve()
    budget.finish(None)
    with pytest.raises(RuntimeError,match='no current'):budget.finish(None)
    assert budget.block_attempts==1 and len(budget.records)==1


def test_restart_preserves_legacy_completed_counts_but_requires_home(tmp_path):
    budget=AuxiliaryMotionBudget()
    for _ in range(3):
        budget.reserve();budget.finish(dict(policy_commands_attempted=5,policy_commands_returned=5,completed=True))
    legacy=budget.status()
    for key in ('pending_reservation','requires_home_after_restart','closed_by_home'):legacy.pop(key)
    save_budget(tmp_path,budget,legacy)
    restored=AuxiliaryMotionBudget.restore_latest(tmp_path)
    assert restored.block_attempts==3 and restored.returned==15 and restored.attempted==15
    assert restored.active and restored.status()['requires_home_after_restart']
    with pytest.raises(RuntimeError,match='restart'):restored.reserve()


def test_interrupted_reservation_is_not_assumed_zero_and_only_home_closes_it(tmp_path):
    budget=AuxiliaryMotionBudget();budget.reserve();save_budget(tmp_path,budget)
    restored=AuxiliaryMotionBudget.restore_latest(tmp_path)
    assert restored.pending and restored.block_attempts==1
    with pytest.raises(RuntimeError):restored.reserve()
    with pytest.raises(RuntimeError):restored.finish(None)  # Cannot invent the interrupted result.
    closed=restored.closed_after_home_status()
    assert closed['accounting_unknown'] and len(closed['records'])==1
    assert restored.active and restored.pending  # Only persisted HOME clears the service budget.
    save_budget(tmp_path,restored,closed)
    assert not AuxiliaryMotionBudget.restore_latest(tmp_path).active


@pytest.mark.parametrize('change',[
    {'policy_commands_returned':4}, {'block_attempts':True}, {'records':[]},
    {'active':False}, {'trial_id':'different'}, {'maximum_blocks':50},
    {'accounting_unknown':True}, {'pending_reservation':True}, {'closed_by_home':True},
])
def test_corrupt_accounting_fails_closed(tmp_path,change):
    budget=AuxiliaryMotionBudget();budget.reserve()
    budget.finish(dict(policy_commands_attempted=5,policy_commands_returned=5,completed=True))
    value=budget.status();value.update(change);save_budget(tmp_path,budget,value)
    with pytest.raises(ValueError,match='ledger'):AuxiliaryMotionBudget.restore_latest(tmp_path)


def test_completed_flag_cannot_hide_partial_or_missing_command_counts(tmp_path):
    budget=AuxiliaryMotionBudget();budget.reserve()
    budget.finish(dict(policy_commands_attempted=4,policy_commands_returned=4,completed=True))
    assert budget.unknown
    save_budget(tmp_path,budget)
    restored=AuxiliaryMotionBudget.restore_latest(tmp_path)
    assert restored.unknown


def test_invalid_ledger_rejects_startup_before_hardware_is_acquired(tmp_path):
    from fcpg.v21_live_service import V21LiveService
    write_json(tmp_path/'auxiliary-budget-1.json',{})
    calls=[]
    with pytest.raises(ValueError,match='ledger'):
        V21LiveService(tmp_path/'model',tmp_path,owner_factory=lambda *a,**k:calls.append(True))
    assert not calls


def test_reservation_is_written_before_planning_and_zero_write_failure_is_counted(service,monkeypatch):
    import fcpg.v21_live_service as live
    service.alignment_motion_enabled=True
    service.owner.read_checked=service.owner._read_q
    service.owner.io.settle_position_tolerance=3.
    monkeypatch.setattr(live.established,'_load_v177_home_state',lambda *a,**k:service.owner.q.copy())
    seen=[]
    def fail_plan(**kwargs):
        path=service.output/f'auxiliary-budget-{service.auxiliary_budget.trial_id}.json'
        seen.append(json.loads(path.read_text()))
        raise ValueError('synthetic planning rejection')
    service.owner.bridge.plan=fail_plan
    with pytest.raises(ValueError,match='planning rejection'):
        service._run_alignment_motion(lambda state:None,'visual_alignment_probe')
    assert seen[0]['pending_reservation'] and seen[0]['block_attempts']==1
    assert not service.auxiliary_budget.pending and service.auxiliary_budget.returned==0
    assert service.owner.writes==0
    assert AuxiliaryMotionBudget.restore_latest(service.output).block_attempts==1


def test_failed_prewrite_persistence_never_reaches_motor_planning(service,monkeypatch):
    import fcpg.v21_live_service as live
    service.alignment_motion_enabled=True
    service.auxiliary_budget.reserve();service.auxiliary_budget.finish(None)
    plans=[];service.owner.bridge.plan=lambda **kw:plans.append(True)
    def fail_write(*a,**kw):raise OSError('disk unavailable')
    monkeypatch.setattr(live,'write_json',fail_write)
    with pytest.raises(OSError,match='disk unavailable'):
        service._run_alignment_motion(lambda state:None,'visual_alignment_probe')
    assert not plans and service.owner.writes==0 and service.owner.stop_event.is_set()


def test_stop_arriving_during_reservation_persistence_is_not_cleared(service,monkeypatch):
    service.alignment_motion_enabled=True
    service.auxiliary_budget.reserve();service.auxiliary_budget.finish(None)
    plans=[];service.owner.bridge.plan=lambda **kw:plans.append(True)
    persist=service._persist_auxiliary_budget
    def stop_while_writing():
        persist();service.owner.stop_event.set()
    monkeypatch.setattr(service,'_persist_auxiliary_budget',stop_while_writing)
    with pytest.raises(RuntimeError,match='cancelled before dispatch'):
        service._run_alignment_motion(lambda state:None,'visual_alignment_probe')
    assert not plans and service.owner.writes==0 and service.owner.stop_event.is_set()


@pytest.mark.parametrize('stop_during',['home_load','home_read'])
def test_stop_during_first_home_precheck_prevents_any_reservation_or_command(service,monkeypatch,stop_during):
    import fcpg.v21_live_service as live
    service.alignment_motion_enabled=True
    service.owner.io.settle_position_tolerance=3.
    plans=[];service.owner.bridge.plan=lambda **kw:plans.append(True)
    def home_target(*a,**kw):
        if stop_during=='home_load':service.owner.stop_event.set()
        return service.owner.q.copy()
    def read_home():
        if stop_during=='home_read':service.owner.stop_event.set()
        return service.owner.q.copy()
    monkeypatch.setattr(live.established,'_load_v177_home_state',home_target)
    service.owner.read_checked=read_home
    with pytest.raises(RuntimeError,match='cancelled during preparation'):
        service._run_alignment_motion(lambda state:None,'visual_alignment_probe')
    assert not plans and service.owner.writes==0 and service.owner.stop_event.is_set()
    assert service.auxiliary_budget.block_attempts==0 and not service.auxiliary_budget.active


def test_home_precheck_exception_preserves_stop_and_does_not_consume_budget(service,monkeypatch):
    import fcpg.v21_live_service as live
    service.alignment_motion_enabled=True
    def bad_home(*a,**kw):raise ValueError('missing HOME configuration')
    monkeypatch.setattr(live.established,'_load_v177_home_state',bad_home)
    with pytest.raises(ValueError,match='missing HOME'):
        service._run_alignment_motion(lambda state:None,'visual_alignment_probe')
    assert service.owner.writes==0 and service.owner.stop_event.is_set()
    assert service.auxiliary_budget.block_attempts==0


def test_home_closure_is_persisted_before_budget_is_reset(service,monkeypatch):
    import fcpg.v21_live_service as live
    budget=service.auxiliary_budget;budget.reserve();budget.finish(None)
    original=live.write_json
    def fail_write(*a,**kw):raise OSError('disk unavailable')
    monkeypatch.setattr(live,'write_json',fail_write)
    with pytest.raises(OSError):service._close_auxiliary_after_home()
    assert service.auxiliary_budget is budget and budget.active
    monkeypatch.setattr(live,'write_json',original)
    service._close_auxiliary_after_home()
    data=json.loads((service.output/f'auxiliary-budget-{budget.trial_id}.json').read_text())
    assert not data['active'] and data['closed_by_home']
    assert not service.auxiliary_budget.active
    assert not AuxiliaryMotionBudget.restore_latest(service.output).active


def test_wall_clock_rollback_cannot_hide_new_active_trial(tmp_path,monkeypatch):
    import fcpg.v21_auxiliary_budget as module
    monkeypatch.setattr(module.time,'time_ns',lambda:200)
    old=AuxiliaryMotionBudget();old.reserve();old.finish(None)
    save_budget(tmp_path,old,old.closed_after_home_status())
    monkeypatch.setattr(module.time,'time_ns',lambda:100)
    new=AuxiliaryMotionBudget.restore_latest(tmp_path);new.reserve()
    assert int(new.trial_id)>int(old.trial_id)
    new.finish(dict(policy_commands_attempted=5,policy_commands_returned=5,completed=True))
    save_budget(tmp_path,new)
    restored=AuxiliaryMotionBudget.restore_latest(tmp_path)
    assert restored.active and restored.returned==5 and restored.requires_home


def test_legacy_nonmonotonic_or_multiple_open_ledgers_fail_closed(tmp_path,monkeypatch):
    import fcpg.v21_auxiliary_budget as module
    monkeypatch.setattr(module.time,'time_ns',lambda:200)
    old=AuxiliaryMotionBudget();old.reserve();old.finish(None)
    save_budget(tmp_path,old,old.closed_after_home_status())
    monkeypatch.setattr(module.time,'time_ns',lambda:100)
    new=AuxiliaryMotionBudget();new.reserve();new.finish(None);save_budget(tmp_path,new)
    with pytest.raises(ValueError,match='ledger'):AuxiliaryMotionBudget.restore_latest(tmp_path)


def test_home_preserves_identity_floor_in_same_process(service,monkeypatch):
    import fcpg.v21_auxiliary_budget as module
    monkeypatch.setattr(module.time,'time_ns',lambda:200)
    service.auxiliary_budget.reserve();service.auxiliary_budget.finish(None)
    service._close_auxiliary_after_home()
    service._close_auxiliary_after_home()  # Repeating HOME cannot erase the floor.
    monkeypatch.setattr(module.time,'time_ns',lambda:100)
    service.auxiliary_budget.reserve()
    assert service.auxiliary_budget.trial_id=='201'


def test_home_recovery_after_restart_uses_remaining_budget_not_a_fresh_one(tmp_path):
    budget=AuxiliaryMotionBudget()
    for _ in range(9):
        budget.reserve();budget.finish(dict(policy_commands_attempted=5,policy_commands_returned=5,completed=True))
    save_budget(tmp_path,budget)
    restored=AuxiliaryMotionBudget.restore_latest(tmp_path)
    restored.reserve_home()
    restored.finish(dict(policy_commands_attempted=5,policy_commands_returned=5,completed=True,
                         action_source='manual_home_recovery'))
    assert restored.block_attempts==10 and restored.returned==50 and restored.requires_home
    with pytest.raises(RuntimeError,match='no auxiliary budget'):restored.reserve_home()
    with pytest.raises(RuntimeError):restored.reserve()


def test_interrupted_block_stays_unknown_and_home_does_not_refund_it(tmp_path):
    budget=AuxiliaryMotionBudget();budget.reserve();save_budget(tmp_path,budget)
    restored=AuxiliaryMotionBudget.restore_latest(tmp_path)
    restored.reserve_home()
    restored.finish(dict(policy_commands_attempted=3,policy_commands_returned=2,completed=False,
                         action_source='manual_home_recovery'))
    assert restored.block_attempts==2 and restored.returned==2 and restored.unknown
    assert restored.records[0]==dict(accounting_available=False)
    assert restored.requires_home


def test_home_budget_exhaustion_prevents_manual_motion(service,monkeypatch):
    import numpy as np
    import fcpg.v21_live_service as live
    for _ in range(10):service.auxiliary_budget.reserve();service.auxiliary_budget.finish(None)
    service.owner.read_checked=service.owner._read_q;service.owner.io.settle_position_tolerance=3.
    monkeypatch.setattr(live.established,'_load_v177_home_state',lambda *a,**k:np.ones(6)*20)
    monkeypatch.setattr(live,'plan_home_chunk',lambda *a:np.ones((5,6))*20)
    monkeypatch.setattr(live,'execute_home_readback_chunk',lambda *a,**k:pytest.fail('no budget for HOME'))
    with pytest.raises(RuntimeError,match='no auxiliary budget'):service.handle('home')
    assert service.auxiliary_budget.active and service.auxiliary_budget.block_attempts==10
    assert service.owner.writes==0 and service.owner.runtime.stopped


def test_successful_home_commands_are_charged_before_old_trial_is_closed(service,monkeypatch):
    import numpy as np
    import threading
    from types import SimpleNamespace
    import fcpg.v21_live_service as live
    from scripts.serve_so101_prod_v17_7 import V177ClearanceGate
    from test_v21_home import FlatKinematics
    owner=service.owner
    for _ in range(4):
        service.auxiliary_budget.reserve()
        service.auxiliary_budget.finish(dict(policy_commands_attempted=5,policy_commands_returned=5,completed=True))
    identity=service.auxiliary_budget.trial_id
    owner.read_checked=owner._read_q;owner._lease_check=lambda:None
    owner.bus_lock=threading.RLock()
    owner.clearance=V177ClearanceGate(kinematics=FlatKinematics(),table_z_mm=0.)
    owner.io.settle_position_tolerance=3.
    owner.io._start_command_journal=lambda commands:None
    owner.io._append_command_journal=lambda *a:None
    owner.io._settle_commanded_endpoint=lambda *a,**k:SimpleNamespace(outcome='settled')
    target=np.zeros(6,np.float32);target[1]=10.
    monkeypatch.setattr(live.established,'_load_v177_home_state',lambda *a,**k:target.copy())
    monkeypatch.setattr(live,'plan_home_chunk',lambda *a:np.repeat(target[None],5,0))
    def write(command,previous):
        owner.q=command.copy();owner.writes+=1
        return command.copy()
    owner._write_q=write
    service.handle('home')
    data=json.loads((service.output/f'auxiliary-budget-{identity}.json').read_text())
    assert data['closed_by_home'] and not data['active']
    assert data['block_attempts']==5 and data['policy_commands_returned']==25
    assert data['records'][-1]['action_source']=='manual_home_recovery'
    assert owner.writes==5 and not service.auxiliary_budget.active


def test_empty_resume_claim_cannot_clear_restart_interlock(tmp_path):
    budget=AuxiliaryMotionBudget();budget.reserve();budget.finish(None);save_budget(tmp_path,budget)
    restored=AuxiliaryMotionBudget.restore_latest(tmp_path)
    with pytest.raises(RuntimeError,match='does not match'):restored.accept_actual_restore({})
    assert restored.requires_home
