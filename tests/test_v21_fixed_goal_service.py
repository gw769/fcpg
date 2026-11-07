"""Service binding uses fake owners/registers only, never ROS or hardware IO."""
from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import pytest

from fcpg import v21_fixed_goal_service as module
from fcpg.v21_fixed_goal_service import FixedGoalServiceError
from test_v21_fixed_goal_compensation import force_failed_source_cap200
from test_v21_position_servo_recovery import setup


@pytest.fixture
def service(tmp_path):
    value = force_failed_source_cap200(setup(tmp_path))
    value.owner.arm_position_servo = True
    value.owner.output = tmp_path
    value.phase = 'fixed-goal-compensate'
    value.owner.runtime.state = object()
    value.fixed_goal_wait = value.owner.clock.wait
    value.trial = SimpleNamespace(running=False)
    value.auxiliary_budget = SimpleNamespace(active=False)
    value._trial_admission_in_progress = False
    value.plan_record = value.preview_record = value.last_execution = object()
    value.fixed_goal_compensation = value.fixed_goal_compensation_path = None
    (tmp_path/'last-attempt.json').write_text(json.dumps(value.last_attempt))
    (tmp_path/'incomplete-execution.json').write_text(json.dumps({'requires_home': True}))
    return value


def successful(execution):
    _, baseline, goal = module._admitted_source(execution)
    return dict(schema=module.HELPER_SCHEMA, completed=True, configuration_restored=True,
        source_time_s=execution['source_time_s'], baseline_parameters_by_motor=baseline,
        frozen_hold_goal_raw=goal, requires_home=True, model_transition=False,
        model_feedback_labels_added=0, readbacks=[{'actual': True}], parameter_writes=[],
        cleanup={'errors': []}, goal_writes=0)


def install(monkeypatch, service):
    calls = []
    def run(owner, execution, **kwargs):
        calls.append('compensate')
        assert owner is service.owner
        assert kwargs['enabled'] is True and kwargs['clock'] is service.clock
        assert kwargs['wait'] is service.fixed_goal_wait
        assert owner.bus_lock._is_owned() and service.operation_lock.locked()
        saved = json.loads((service.output/module.FILENAME).read_text())
        assert saved['state'] == 'started' and saved['completed'] is False
        assert saved['source_execution'] == execution
        assert owner.stop_event.is_set() and owner.runtime.state is None
        kwargs['admission_check']()
        return successful(execution)
    def verify(owner, execution, receipt, *, admission_check):
        calls.append('verify')
        admission_check()
        assert owner.stop_event.is_set() and service.operation_lock.locked()
        return dict(schema=module.VERIFICATION_SCHEMA, phase='fresh_compensated_HOME_admission',
            configuration_restored=True, requires_home=True, readback_time_s=service.clock(),
            present_raw=deepcopy(receipt['frozen_hold_goal_raw']),
            goal_raw=deepcopy(receipt['frozen_hold_goal_raw']),
            parameters_by_motor=deepcopy(receipt['baseline_parameters_by_motor']))
    monkeypatch.setattr(module, '_compensate', run)
    monkeypatch.setattr(module, '_verify', verify)
    return calls, run, verify


def test_success_keeps_source_stop_home_and_model_feedback_untouched(service, monkeypatch):
    calls, _, _ = install(monkeypatch, service)
    source = deepcopy(service.last_attempt)
    source_file = (service.output/'last-attempt.json').read_bytes()
    incomplete = (service.output/'incomplete-execution.json').read_bytes()
    history = service.owner.runtime.history
    result = module.handle_fixed_goal_compensation(service)
    assert calls == ['compensate']
    assert result['completed'] is True and result['configuration_restored'] is True
    assert service.owner.stop_event.is_set() and service.owner.runtime.stopped is True
    assert service.owner.runtime.requires_home_after_incomplete_execution is True
    assert service.owner.runtime.history is history and service.owner.runtime.state is None
    assert service.last_attempt == source
    assert (service.output/'last-attempt.json').read_bytes() == source_file
    assert (service.output/'incomplete-execution.json').read_bytes() == incomplete
    assert service.owner.bus.events == []
    saved = module.load_fixed_goal_compensation(service.output, source)
    assert saved['state'] == 'completed' and saved['source_execution'] == source
    assert saved['report'] == result == service.fixed_goal_compensation
    result['readbacks'].clear()
    assert service.fixed_goal_compensation['readbacks']


def test_restart_home_rechecks_full_actual_profile_and_goal_before_shortcut(service, monkeypatch):
    calls, _, _ = install(monkeypatch, service)
    result = module.handle_fixed_goal_compensation(service)
    service.last_attempt = None
    service.fixed_goal_compensation = None
    service.phase = 'home'
    verified = module.verify_fixed_goal_compensation_before_home(service)
    assert calls == ['compensate', 'verify']
    assert verified['schema'] == module.VERIFICATION_SCHEMA and verified['completed']
    assert verified['goal_raw'] == result['frozen_hold_goal_raw']
    assert verified['parameters_by_motor'] == result['baseline_parameters_by_motor']
    assert service.fixed_goal_compensation == result
    assert service.last_attempt is None and service.owner.runtime.requires_home_after_incomplete_execution


def test_repeat_explicit_request_verifies_success_without_repeating_compensation(service, monkeypatch):
    calls, _, _ = install(monkeypatch, service)
    result = module.handle_fixed_goal_compensation(service)
    again = module.handle_fixed_goal_compensation(service)
    assert calls == ['compensate', 'verify'] and again == result


@pytest.mark.parametrize('change', [
    lambda s:setattr(s.owner, 'arm_position_servo', False),
    lambda s:s.operation_lock.release(),
    lambda s:s.owner.stop_event.clear(),
    lambda s:setattr(s.owner.runtime, 'stopped', False),
    lambda s:setattr(s.owner.runtime, 'pending', object()),
    lambda s:setattr(s.owner.runtime, 'executing', True),
    lambda s:setattr(s.owner.runtime, 'requires_home_after_incomplete_execution', False),
    lambda s:setattr(s.trial, 'running', True),
    lambda s:setattr(s.auxiliary_budget, 'active', True),
    lambda s:setattr(s, '_trial_admission_in_progress', True),
    lambda s:setattr(s, 'phase', 'execute'),
    lambda s:setattr(s.owner.bridge, 'runtime', object()),
    lambda s:setattr(s.owner, 'lease', None),
])
def test_non_stopped_or_unowned_entry_never_calls_helper_or_writes_receipt(service, monkeypatch, change):
    calls, _, _ = install(monkeypatch, service)
    change(service)
    with pytest.raises(FixedGoalServiceError):
        module.handle_fixed_goal_compensation(service)
    assert calls == [] and not (service.output/module.FILENAME).exists()


@pytest.mark.parametrize('field', ['epoch', '_trial_stop_generation'])
def test_new_stop_cancels_operation_but_allows_same_owner_cleanup(service, monkeypatch, field):
    checks = []
    def stopped(owner, execution, *, admission_check, **kwargs):
        admission_check()
        setattr(service, field, getattr(service, field)+1)
        with pytest.raises(FixedGoalServiceError, match='newer STOP'):
            admission_check()
        admission_check(recovery=True)
        checks.append('cleanup_admitted')
        raise RuntimeError('injected new STOP')
    monkeypatch.setattr(module, '_compensate', stopped)
    with pytest.raises(FixedGoalServiceError, match='new STOP'):
        module.handle_fixed_goal_compensation(service)
    assert checks == ['cleanup_admitted']
    assert service.owner.stop_event.is_set() and service.owner.runtime.requires_home_after_incomplete_execution
    assert json.loads((service.output/module.FILENAME).read_text())['state'] == 'failed'


@pytest.mark.parametrize('change', [
    lambda s:setattr(s.owner, 'lease', object()),
    lambda s:setattr(s.owner, 'runtime', deepcopy(s.owner.runtime)),
    lambda s:setattr(s.owner, 'bridge', SimpleNamespace(runtime=s.owner.runtime)),
    lambda s:setattr(s.owner.io.follower, 'bus', object()),
    lambda s:setattr(s.owner, 'bus_lock', threading.RLock()),
    lambda s:setattr(s.owner, 'stop_event', threading.Event()),
    lambda s:setattr(s, 'operation_lock', threading.Lock()),
    lambda s:setattr(s, 'last_attempt', deepcopy(s.last_attempt)),
    lambda s:setattr(s, 'output', s.output/'other'),
])
def test_cleanup_never_ignores_owner_or_source_identity_change(service, monkeypatch, change):
    def changed(owner, execution, *, admission_check, **kwargs):
        change(service)
        with pytest.raises(FixedGoalServiceError, match='identity changed'):
            admission_check(recovery=True)
        raise RuntimeError('identity rejection proved')
    monkeypatch.setattr(module, '_compensate', changed)
    with pytest.raises(FixedGoalServiceError, match='identity rejection proved'):
        module.handle_fixed_goal_compensation(service)


def test_callback_from_another_thread_is_rejected_even_during_cleanup(service, monkeypatch):
    def changed(owner, execution, *, admission_check, **kwargs):
        errors = []
        def other_thread():
            try:admission_check(recovery=True)
            except FixedGoalServiceError as error:errors.append(str(error))
        thread = threading.Thread(target=other_thread)
        thread.start(); thread.join(timeout=1)
        assert not thread.is_alive() and errors and 'thread identity' in errors[0]
        return successful(execution)
    monkeypatch.setattr(module, '_compensate', changed)
    assert module.handle_fixed_goal_compensation(service)['completed']


def test_lease_verification_cannot_replace_owner_binding(service, monkeypatch):
    calls, _, _ = install(monkeypatch, service)
    service.owner.on_lease = lambda:setattr(service.owner, 'lease', object())
    with pytest.raises(FixedGoalServiceError, match='during lease verification'):
        module.handle_fixed_goal_compensation(service)
    assert calls == [] and not (service.output/module.FILENAME).exists()


@pytest.mark.parametrize('kind', ['disk_before', 'memory_during', 'disk_during', 'helper_copy'])
def test_source_contents_cannot_be_substituted_or_edited(service, monkeypatch, kind):
    calls, run, _ = install(monkeypatch, service)
    if kind == 'disk_before':
        saved = deepcopy(service.last_attempt); saved['extra'] = 'changed'
        (service.output/'last-attempt.json').write_text(json.dumps(saved))
    else:
        def changed(owner, execution, **kwargs):
            result = run(owner, execution, **kwargs)
            if kind == 'memory_during':service.last_attempt['extra'] = 'changed'
            elif kind == 'helper_copy':execution['extra'] = 'changed'
            else:
                saved = deepcopy(service.last_attempt); saved['extra'] = 'changed'
                (service.output/'last-attempt.json').write_text(json.dumps(saved))
            return result
        monkeypatch.setattr(module, '_compensate', changed)
    with pytest.raises(FixedGoalServiceError, match='changed|differs'):
        module.handle_fixed_goal_compensation(service)
    if kind == 'disk_before':assert calls == []


def test_start_receipt_must_be_durable_before_helper_is_called(service, monkeypatch):
    calls, _, _ = install(monkeypatch, service)
    old_state = service.owner.runtime.state
    def failed(*args):raise OSError('disk unavailable')
    monkeypatch.setattr(module, '_durable_json', failed)
    with pytest.raises(OSError, match='disk unavailable'):
        module.handle_fixed_goal_compensation(service)
    assert calls == [] and service.owner.runtime.state is old_state


@pytest.mark.parametrize('corrupt', [
    lambda report:report.update(completed=False),
    lambda report:report.update(configuration_restored=False),
    lambda report:report.update(source_time_s=-1),
    lambda report:report['frozen_hold_goal_raw'].update(elbow_flex=1234),
    lambda report:report['baseline_parameters_by_motor']['elbow_flex'].update(Torque_Limit=200),
])
def test_unproven_helper_result_is_never_persisted_as_success(service, monkeypatch, corrupt):
    def bad(owner, execution, **kwargs):
        result = successful(execution); corrupt(result); return result
    monkeypatch.setattr(module, '_compensate', bad)
    with pytest.raises(FixedGoalServiceError, match='restoration evidence'):
        module.handle_fixed_goal_compensation(service)
    assert json.loads((service.output/module.FILENAME).read_text())['state'] == 'failed'


@pytest.mark.parametrize('field', ['goal_raw', 'parameters_by_motor', 'configuration_restored', 'schema'])
def test_home_requires_fresh_exact_success_not_saved_success_alone(service, monkeypatch, field):
    _, _, verify = install(monkeypatch, service)
    module.handle_fixed_goal_compensation(service)
    def invalid(*args, **kwargs):
        result = verify(*args, **kwargs); result[field] = None; return result
    monkeypatch.setattr(module, '_verify', invalid)
    service.phase = 'home'
    with pytest.raises(FixedGoalServiceError, match='fresh HOME verification'):
        module.verify_fixed_goal_compensation_before_home(service)


def test_interrupted_receipt_cannot_authorize_ordinary_home_recovery(service, monkeypatch):
    install(monkeypatch, service)
    module.handle_fixed_goal_compensation(service)
    path = service.output/module.FILENAME
    record = json.loads(path.read_text())
    record.update(state='started', completed=False, configuration_restored=False)
    record['report'].update(completed=False, configuration_restored=False)
    path.write_text(json.dumps(record))
    service.phase = 'home'
    with pytest.raises(FixedGoalServiceError, match='interrupted'):
        module.verify_fixed_goal_compensation_before_home(service)


def test_failed_receipt_is_not_a_permanent_home_seal(service, monkeypatch):
    install(monkeypatch, service)
    module.handle_fixed_goal_compensation(service)
    path = service.output/module.FILENAME
    record = json.loads(path.read_text())
    record.update(state='failed', completed=False, configuration_restored=False)
    record['report'].update(completed=False, configuration_restored=False,
        cleanup=dict(integral_zero_verified=True, low_cap_verified=True, damping_zero_verified=True,
                     hold_verified=True, errors=[], hold=deepcopy(service.last_attempt['position_servo']['cleanup']['hold'])))
    path.write_text(json.dumps(record))
    service.phase = 'home'
    assert module.verify_fixed_goal_compensation_before_home(service) is None


def test_missing_receipt_keeps_existing_fake_and_ordinary_home_compatible(service):
    service.phase = 'home'
    assert module.verify_fixed_goal_compensation_before_home(service) is None
    assert module.verify_fixed_goal_compensation_before_home(SimpleNamespace()) is None


def test_stale_arm_receipt_is_ignored_for_current_ordinary_cem_failure(service):
    ordinary = deepcopy(service.last_attempt)
    ordinary.pop('position_servo', None)
    ordinary.pop('position_servo_error', None)
    service.last_attempt = ordinary
    (service.output/'last-attempt.json').write_text(json.dumps(ordinary))
    (service.output/module.FILENAME).write_text('{}')
    service.phase = 'home'
    assert module.verify_fixed_goal_compensation_before_home(service) is None


def test_receipt_cannot_be_rebound_to_a_different_output_directory(service, monkeypatch, tmp_path):
    install(monkeypatch, service)
    module.handle_fixed_goal_compensation(service)
    other = tmp_path/'other'; other.mkdir()
    (other/module.FILENAME).write_bytes((service.output/module.FILENAME).read_bytes())
    with pytest.raises(FixedGoalServiceError, match='source binding'):
        module.load_fixed_goal_compensation(other, service.last_attempt)


def test_old_success_for_a_different_real_execution_does_not_skip_home(service, monkeypatch):
    calls, _, _ = install(monkeypatch, service)
    module.handle_fixed_goal_compensation(service)
    service.last_attempt['source_time_s'] += 1
    (service.output/'last-attempt.json').write_text(json.dumps(service.last_attempt))
    service.phase = 'home'
    assert module.verify_fixed_goal_compensation_before_home(service) is None
    assert calls == ['compensate']


def test_real_helper_fake_bus_recipe_then_fresh_home_verification(service):
    from fcpg.v21_same_goal_integral import MOTOR, ORDER
    owner = service.owner
    owner.bus.after_halt = False
    owner.bus.steps[MOTOR] = 1
    owner.bus.values[MOTOR]['Present_Position'] = owner.bus.values[MOTOR]['Goal_Position']+33
    goals = {name:owner.bus.values[name]['Goal_Position'] for name in ORDER}
    source = deepcopy(service.last_attempt)
    result = module.handle_fixed_goal_compensation(service)
    assert result['completed'] and result['configuration_restored'] and result['goal_writes'] == 0
    writes = [(row['register'], row['value']) for row in owner.bus.events if row['kind'] == 'write']
    assert writes == [('D_Coefficient',32), ('Torque_Limit',300), ('I_Coefficient',1),
        ('I_Coefficient',0), ('Torque_Limit',200), ('D_Coefficient',0), ('Torque_Limit',1000)]
    event_count = len([row for row in owner.bus.events if row['kind'] in ('write', 'halt')])
    service.last_attempt = None
    service.phase = 'home'
    verified = module.verify_fixed_goal_compensation_before_home(service)
    assert verified['completed'] and verified['goal_raw'] == goals
    assert len([row for row in owner.bus.events if row['kind'] in ('write', 'halt')]) == event_count
    assert owner.runtime.requires_home_after_incomplete_execution and owner.stop_event.is_set()
    assert json.loads((service.output/'last-attempt.json').read_text()) == source


def failed_but_restored_receipt(service):
    from fcpg.v21_same_goal_integral import MOTOR
    owner = service.owner
    owner.bus.after_halt = False
    owner.bus.steps[MOTOR] = 1
    owner.bus.values[MOTOR]['Present_Position'] = owner.bus.values[MOTOR]['Goal_Position']+33
    def fail_active_damping(register,motor,value):
        if register == 'D_Coefficient' and value == 32:
            raise RuntimeError('injected fixed-Goal operation failure')
    owner.bus.on_write = fail_active_damping
    with pytest.raises(FixedGoalServiceError,match='operation failure'):
        module.handle_fixed_goal_compensation(service)
    owner.bus.on_write = None
    return json.loads((service.output/module.FILENAME).read_text())


def test_failed_operation_persists_verified_baseline_profile_and_new_hold(service):
    record = failed_but_restored_receipt(service)
    report, cleanup = record['report'], record['report']['cleanup']
    assert record['state']=='failed' and record['completed'] is False
    assert report['completed'] is False and record['configuration_restored'] is True
    assert cleanup['profile_state']=='baseline1000' and cleanup['baseline_restored']
    assert cleanup['effective_hold_goal_raw'] != report['frozen_hold_goal_raw']
    assert report['final_parameters_by_motor']==report['baseline_parameters_by_motor']
    assert module.load_fixed_goal_compensation(service.output,service.last_attempt)==record
    writes=len([row for row in service.owner.bus.events if row['kind'] in ('write','halt')])
    service.phase='home'
    assert module.verify_fixed_goal_compensation_before_home(service) is None
    assert len([row for row in service.owner.bus.events if row['kind'] in ('write','halt')])==writes


@pytest.mark.parametrize('damage',['quiet_window','baseline_ack'])
def test_failed_baseline_receipt_rejects_corrupt_restoration_evidence(service,damage):
    record=failed_but_restored_receipt(service)
    if damage=='quiet_window':
        record['report']['stages']['failed_baseline_restored']['sample_count']-=1
    else:
        event=next(row for row in reversed(record['report']['parameter_writes'])
            if row['register']=='Torque_Limit')
        event['readback']=200
    (service.output/module.FILENAME).write_text(json.dumps(record))
    with pytest.raises(FixedGoalServiceError,match='baseline|quiet'):
        module.load_fixed_goal_compensation(service.output,service.last_attempt)


@pytest.mark.parametrize('damage', ['missing_cleanup', 'hold_changed'])
def test_failed_receipt_with_unknown_or_different_hold_cannot_fall_through_to_cap_reset(service, monkeypatch, damage):
    install(monkeypatch, service)
    module.handle_fixed_goal_compensation(service)
    path = service.output/module.FILENAME
    record = json.loads(path.read_text())
    record.update(state='failed', completed=False, configuration_restored=False)
    cleanup = dict(integral_zero_verified=True, low_cap_verified=True, damping_zero_verified=True,
        hold_verified=True, errors=[], hold=deepcopy(service.last_attempt['position_servo']['cleanup']['hold']))
    if damage == 'missing_cleanup':cleanup.pop('integral_zero_verified')
    else:cleanup['hold']['attempts'][-1]['goal_readback_raw']['elbow_flex'] += 1
    record['report'].update(completed=False, configuration_restored=False, cleanup=cleanup)
    path.write_text(json.dumps(record))
    service.phase = 'home'
    with pytest.raises(FixedGoalServiceError, match='cleanup|HOLD'):
        module.verify_fixed_goal_compensation_before_home(service)


def reused_hold_failure(service, monkeypatch):
    install(monkeypatch, service)
    module.handle_fixed_goal_compensation(service)
    path = service.output/module.FILENAME
    record = json.loads(path.read_text())
    started = record['started_s']
    goal = module._admitted_source(service.last_attempt)[2]
    present = dict(goal)
    present['elbow_flex'] += 1
    sample = dict(started_s=started+.01, finished_s=started+.06,
        present_raw=present, goal_raw=deepcopy(goal),
        present_position_from_raw={name:(value-10000)/100. for name,value in present.items()},
        moving={name:0 for name in goal}, torque={name:1 for name in goal},
        normalised_position_source='same_raw_register_read', motor_register_writes=0,
        position_read_intervals_s={
            'Present_Position':dict(started_s=started+.02,returned_s=started+.03),
            'Goal_Position':dict(started_s=started+.04,returned_s=started+.05)})
    hold = dict(event='safe_hold_verified', reason='existing_hold_reverified_no_new_goal',
        reused_existing_hold=True, motor_register_writes=0, monotonic_s=started+.07,
        actual_readback=sample, attempts=[dict(goal_register_verified=True,
            requested_goal_raw=deepcopy(goal),goal_readback_raw=deepcopy(goal))])
    record.update(state='failed', completed=False, configuration_restored=False, finished_s=started+.1)
    record['report'].update(completed=False,configuration_restored=False,started_s=started,
        finished_s=started+.09,goal_writes=0,
        cleanup=dict(integral_zero_verified=True,low_cap_verified=True,damping_zero_verified=True,
            hold_verified=True,errors=[],hold=hold))
    return path, record


def test_failed_cleanup_with_fresh_reused_original_hold_allows_ordinary_home_recovery(service, monkeypatch):
    path, record = reused_hold_failure(service, monkeypatch)
    path.write_text(json.dumps(record))
    service.phase = 'home'
    assert module.verify_fixed_goal_compensation_before_home(service) is None
    assert service.owner.bus.events == []
    assert service.owner.stop_event.is_set() and service.owner.runtime.requires_home_after_incomplete_execution
    assert service.fixed_goal_compensation['configuration_restored'] is False


@pytest.mark.parametrize('damage', ['missing_readback','stale','future','missing_interval','bad_interval',
    'changed_goal','missing_present','moving','torque','writes','read_writes','total_writes',
    'unbound_normalization','missing_actual_position','wrong_reason','new_hold_unvalidated'])
def test_reused_hold_home_requires_fresh_same_goal_zero_write_evidence(service, monkeypatch, damage):
    path, record = reused_hold_failure(service, monkeypatch)
    hold = record['report']['cleanup']['hold']
    sample = hold['actual_readback']
    if damage == 'missing_readback':hold.pop('actual_readback')
    elif damage == 'stale':sample['started_s'] = record['started_s']-1
    elif damage == 'future':sample['finished_s'] = record['finished_s']+1
    elif damage == 'missing_interval':sample['position_read_intervals_s'].pop('Present_Position')
    elif damage == 'bad_interval':sample['position_read_intervals_s']['Present_Position']['returned_s'] = sample['finished_s']+1
    elif damage == 'changed_goal':sample['goal_raw']['elbow_flex'] += 1
    elif damage == 'missing_present':sample['present_raw'].pop('elbow_flex')
    elif damage == 'moving':sample['moving']['elbow_flex'] = 1
    elif damage == 'torque':sample['torque']['elbow_flex'] = 0
    elif damage == 'writes':hold['motor_register_writes'] = 1
    elif damage == 'read_writes':sample['motor_register_writes'] = 1
    elif damage == 'total_writes':record['report']['goal_writes'] = 1
    elif damage == 'unbound_normalization':sample['normalised_position_source'] = 'cached'
    elif damage == 'missing_actual_position':sample.pop('present_position_from_raw')
    elif damage == 'wrong_reason':hold['reason'] = 'new_unvalidated_hold'
    else:
        hold.update(reused_existing_hold=False, reason='new_hold')
        hold.pop('actual_readback')
    path.write_text(json.dumps(record))
    service.phase = 'home'
    with pytest.raises(FixedGoalServiceError, match='HOLD'):
        module.verify_fixed_goal_compensation_before_home(service)
    assert service.owner.bus.events == []
