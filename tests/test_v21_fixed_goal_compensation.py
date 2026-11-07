"""Fixed-Goal compensation recipe: fake clock/bus only, no hardware or models."""
from copy import deepcopy

import pytest

from fcpg.v21_fixed_goal_compensation import (
    FixedGoalCompensationError, compensate_failed_hold, verify_compensated_hold,
)
from fcpg.v21_same_goal_integral import MOTOR, ORDER
from test_v21_position_servo_recovery import setup as stopped_setup


def force_failed_source_cap200(service):
    """Make this fixture the low-cap predecessor required by fixed-Goal recovery."""
    servo = service.last_attempt['position_servo']
    cleanup = servo['cleanup']
    if cleanup.get('profile_state') == 'baseline1000':
        service.owner.bus.values[MOTOR]['Torque_Limit'] = 200
        servo['parameter_writes'].append(dict(motor=MOTOR,register='Torque_Limit',value=200,
            recovery=True,may_have_applied=True,verified=True,readback=200))
        servo.pop('final_parameters_by_motor',None)
        cleanup.update(restored=False,baseline_restored=False,torque_restored=False,
            post_restore_quiet_verified=False,profile_state='cleanup_cap200')
        cleanup['parameters_restored_by_motor'][MOTOR] = False
        cleanup['recovery_by_motor'][MOTOR]['low_cap_verified'] = True
    return service


def setup(tmp_path, error_cells=33):
    service = force_failed_source_cap200(stopped_setup(tmp_path))
    owner = service.owner
    owner.bus.after_halt = False
    owner.bus.steps[MOTOR] = 1
    owner.bus.values[MOTOR]['Present_Position'] = owner.bus.values[MOTOR]['Goal_Position']+error_cells
    raw_sync_read = owner.bus.sync_read
    owner.bus.requested_sync_retries = []
    def sync_read(register, *, normalize, num_retry):
        owner.bus.requested_sync_retries.append((register,num_retry))
        return raw_sync_read(register,normalize=normalize,num_retry=num_retry)
    owner.bus.sync_read = sync_read
    def admission(*, recovery=False):
        if service.owner is not owner or not service.operation_lock.locked():
            raise RuntimeError('owner or operation lock changed')
        if not recovery and service.epoch != 0:
            raise RuntimeError('new STOP cancelled compensation')
    service.admission = admission
    return service


def run(service):
    return compensate_failed_hold(service.owner,service.last_attempt,enabled=True,
        admission_check=service.admission,clock=service.owner.clock,wait=service.owner.clock.wait)


def writes(service):
    return [(row['register'],row['value']) for row in service.owner.bus.events if row['kind']=='write']


def failed(service,match=None):
    with pytest.raises(FixedGoalCompensationError,match=match) as caught:run(service)
    report = caught.value.report
    assert not report['completed']
    restored = report.get('cleanup',{}).get('baseline_restored') is True
    assert report['configuration_restored'] is restored
    if restored:
        assert report['cleanup']['profile_state'] == 'baseline1000'
        assert report['final_parameters_by_motor'] == report['baseline_parameters_by_motor']
    assert service.owner.stop_event.is_set()
    assert service.owner.runtime.requires_home_after_incomplete_execution is True
    return report


def test_only_fixed_j3_output_changes_original_goal_model_and_home_interlock_preserved(tmp_path):
    service = setup(tmp_path)
    owner = service.owner
    original = deepcopy(service.last_attempt)
    goals = {name:owner.bus.values[name]['Goal_Position'] for name in ORDER}
    history = owner.runtime.history
    report = run(service)
    assert report['completed'] and report['configuration_restored'] and report['goal_writes']==0
    assert writes(service)==[('D_Coefficient',32),('Torque_Limit',300),('I_Coefficient',1),
        ('I_Coefficient',0),('Torque_Limit',200),('D_Coefficient',0),('Torque_Limit',1000)]
    assert all(row['motor']==MOTOR for row in report['parameter_writes'])
    assert all(row['stopped'] for row in owner.bus.events)
    assert {name:owner.bus.values[name]['Goal_Position'] for name in ORDER}==goals
    assert all(row['kind']!='halt' for row in owner.bus.events)
    assert owner.runtime.history is history and service.last_attempt==original
    assert owner.runtime.requires_home_after_incomplete_execution is True
    assert report['integral_exposure']['within_limit'] and report['integral_exposure']['duration_s']<=10.
    for phase in ('retention','restored'):
        assert report['stages'][phase]['sample_count']==22 and report['stages'][phase]['actual_span_s']>=1.
    assert report['final_parameters_by_motor']['gripper']['Torque_Limit']==500
    current = verify_compensated_hold(owner,service.last_attempt,report,admission_check=service.admission)
    assert current['configuration_restored'] and current['requires_home']
    assert current['goal_raw']==goals


def test_disabled_never_opens_owner_or_clock():
    assert compensate_failed_hold(clock=lambda:pytest.fail('disabled touched clock'))['enabled'] is False


def test_sync_reads_request_one_transport_retry_and_writes_remain_zero(tmp_path):
    service = setup(tmp_path,error_cells=1)
    bus = service.owner.bus
    raw_write, raw_read = bus.write, bus.read
    writes_num_retry, reads_num_retry = [], []
    def write(register, motor, value, *, normalize, num_retry):
        writes_num_retry.append(num_retry)
        return raw_write(register,motor,value,normalize=normalize,num_retry=num_retry)
    def read(register, motor, *, normalize, num_retry):
        reads_num_retry.append(num_retry)
        return raw_read(register,motor,normalize=normalize,num_retry=num_retry)
    bus.write, bus.read = write, read
    report = run(service)
    assert bus.requested_sync_retries and {value for _,value in bus.requested_sync_retries} == {1}
    events = [event for row in report['readbacks']
        for event in row['register_reads'].values()]
    events += [event for boundary in report['stages'].get('candidate',{}).get('parameter_boundaries',[])
        for attempt in boundary['attempts'] for event in attempt['register_reads'].values()]
    assert events and all(event['num_retry'] == 1 for event in events)
    assert writes_num_retry and set(writes_num_retry) == {0}
    assert reads_num_retry and set(reads_num_retry) == {0}


@pytest.mark.parametrize('error_cells', [1, 3, 6])
def test_already_near_fixed_goal_only_restores_cap_with_full_windows(tmp_path, error_cells):
    service = setup(tmp_path,error_cells=error_cells)
    report = run(service)
    assert report['completed']
    assert writes(service)==[('Torque_Limit',1000)]
    assert 'integral_exposure' not in report


@pytest.mark.parametrize('error_cells,accepted', [(6, True), (7, False)])
def test_real_j3_scale_accepts_six_cells_but_rejects_seven(tmp_path, error_cells, accepted):
    service = setup(tmp_path,error_cells=error_cells)
    bus = service.owner.bus
    original_normalize, original_unnormalize = bus._normalize, bus._unnormalize
    bus._normalize = lambda values:{
        key:(value-10000)/11.27 if key == 3 else original_normalize({key:value})[key]
        for key,value in values.items()}
    bus._unnormalize = lambda values:{
        key:round(10000+11.27*value) if key == 3 else original_unnormalize({key:value})[key]
        for key,value in values.items()}
    if accepted:
        report = run(service)
        assert report['completed'] and report['initial_j3_residual'] == pytest.approx(error_cells/11.27)
    else:
        failed(service, 'no larger than .6')


@pytest.mark.parametrize('damage',['large_residual','wrong_goal','other_cap','wrong_gain','identity','untouched_motion'])
def test_bad_admission_never_raises_cap_or_enables_i1(tmp_path,damage):
    service = setup(tmp_path)
    bus = service.owner.bus
    if damage=='large_residual':bus.values[MOTOR]['Present_Position']=bus.values[MOTOR]['Goal_Position']+61
    elif damage=='wrong_goal':bus.values[MOTOR]['Goal_Position']+=1
    elif damage=='other_cap':bus.values[ORDER[3]]['Torque_Limit']=200
    elif damage=='wrong_gain':bus.values[MOTOR]['P_Coefficient']=32
    elif damage=='identity':bus.values[MOTOR]['Lock']=0
    else:bus.values[ORDER[0]]['Moving']=1
    report = failed(service)
    assert ('Torque_Limit',300) not in writes(service) and ('I_Coefficient',1) not in writes(service)
    assert (('Torque_Limit',1000) in writes(service)) is (
        report.get('cleanup',{}).get('baseline_restored') is True)


@pytest.mark.parametrize('register,value',[('D_Coefficient',32),('Torque_Limit',300),('I_Coefficient',1)])
def test_applied_but_lost_ack_clears_i_before_reducing_cap_and_removing_damping(tmp_path,register,value):
    service = setup(tmp_path)
    def lose_ack(actual_register,motor,actual_value):
        if actual_register==register and actual_value==value:raise RuntimeError('lost ack')
    service.owner.bus.on_write = lose_ack
    report = failed(service,'lost ack')
    assert writes(service)[-4:]==[('I_Coefficient',0),('Torque_Limit',200),
        ('D_Coefficient',0),('Torque_Limit',1000)]
    assert report['cleanup']['integral_zero_verified'] and report['cleanup']['low_cap_verified']
    assert report['cleanup']['damping_zero_verified'] and report['cleanup']['hold_verified']
    assert report['cleanup']['baseline_restored']
    assert service.owner.bus.values[MOTOR]['Torque_Limit']==1000


@pytest.mark.parametrize('at',[('I_Coefficient',1),('Torque_Limit',1000)])
def test_new_stop_during_parameter_write_runs_independent_cleanup(tmp_path,at):
    service = setup(tmp_path)
    def stop(register,motor,value):
        if (register,value)==at:service.epoch+=1
    service.owner.bus.on_write = stop
    report = failed(service,'STOP')
    assert writes(service)[-4:]==[('I_Coefficient',0),('Torque_Limit',200),
        ('D_Coefficient',0),('Torque_Limit',1000)]
    assert report['cleanup']['hold_verified']
    assert report['cleanup']['baseline_restored']


def test_unreached_fixed_goal_times_out_with_actual_i0_verified_within_budget(tmp_path):
    service = setup(tmp_path)
    service.owner.bus.steps[MOTOR]=0
    report = failed(service,'not reached')
    candidate = report['stages']['candidate']
    assert 0 < candidate['sample_count'] < 200
    assert candidate['wall_deadline_reached_s'] <= report['integral_exposure']['started_s']+10.
    assert report['integral_exposure']['within_limit']
    assert report['cleanup']['baseline_restored'] and ('Torque_Limit',1000) in writes(service)
    assert service.owner.bus.values[MOTOR]['I_Coefficient']==0


def test_large_loss_after_removing_i1_freezes_fresh_hold_then_restores_baseline(tmp_path):
    service = setup(tmp_path)
    active=False
    def sag(register,motor,value):
        nonlocal active
        if register=='I_Coefficient' and value==1:active=True
        if register=='I_Coefficient' and value==0 and active:
            service.owner.bus.values[MOTOR]['Present_Position']=service.owner.bus.values[MOTOR]['Goal_Position']+7
    service.owner.bus.on_write=sag
    report = failed(service,'retained')
    assert report['stages']['retention']['completed'] is False
    assert report['cleanup']['baseline_restored'] and ('Torque_Limit',1000) in writes(service)
    assert report['stages']['failed_hold_retention']['actual_span_s'] >= 1.


def test_active_overshoot_is_frozen_before_verified_baseline_cap_restore(tmp_path):
    service = setup(tmp_path)
    bus = service.owner.bus
    def overshoot(register,motor):
        if register=='Present_Position' and bus.values[MOTOR]['I_Coefficient']==1:
            bus.values[MOTOR]['Present_Position']=bus.values[MOTOR]['Goal_Position']-20
    bus.on_read=overshoot
    report = failed(service,'corridor')
    assert report['cleanup']['hold_verified'] and report['cleanup']['baseline_restored']
    assert ('Torque_Limit',1000) in writes(service)


def test_post_restore_actual_j3_load_failure_returns_cap200(tmp_path):
    service = setup(tmp_path)
    def loaded(register,motor,value):
        if register=='Torque_Limit' and value==1000:service.owner.bus.values[MOTOR]['Present_Load']=60
    service.owner.bus.on_write=loaded
    report = failed(service,'load')
    assert writes(service)[-3:]==[('D_Coefficient',0),('Torque_Limit',1000),('Torque_Limit',200)]
    assert not report['configuration_restored']
    assert report['cleanup']['profile_state']=='cleanup_cap200'
    assert report['cleanup']['fallback_low_cap_verified']
    assert service.owner.bus.values[MOTOR]['Torque_Limit']==200


def test_unverified_fallback_cap_never_claims_a_known_final_profile(tmp_path):
    service = setup(tmp_path)
    def fail_original_and_fallback(register,motor,value):
        if register=='D_Coefficient' and value==32:
            raise RuntimeError('injected original failure')
        if register=='Torque_Limit' and value==1000:
            service.owner.bus.values[MOTOR]['Present_Load']=60
        if (register=='Torque_Limit' and value==200
                and service.owner.bus.values[MOTOR]['Present_Load']==60):
            raise RuntimeError('injected fallback ACK loss')
    service.owner.bus.on_write=fail_original_and_fallback
    report=failed(service,'original failure')
    cleanup=report['cleanup']
    assert not report['configuration_restored']
    assert cleanup['profile_state']=='unknown'
    assert cleanup['fallback_low_cap_verified'] is False
    assert cleanup['low_cap_verified'] is False
    assert any('fallback cap200' in error for error in cleanup['errors'])


def test_full_source_change_during_compensation_cannot_publish_success(tmp_path):
    service = setup(tmp_path)
    def changed(register,motor,value):
        if register=='I_Coefficient' and value==1:service.last_attempt['source_time_s']=-1.
    service.owner.bus.on_write=changed
    failed(service,'source changed')


def test_compensated_home_verification_uses_fresh_profile_and_source(tmp_path):
    service = setup(tmp_path,error_cells=1)
    report = run(service)
    service.owner.bus.values[MOTOR]['Torque_Limit']=200
    with pytest.raises(RuntimeError,match='profile'):
        verify_compensated_hold(service.owner,service.last_attempt,report,admission_check=service.admission)
    service.last_attempt['source_time_s']=-1.
    with pytest.raises(ValueError,match='source'):
        verify_compensated_hold(service.owner,service.last_attempt,report,admission_check=service.admission)


def test_compensated_home_verification_rechecks_untouched_motor_identity(tmp_path):
    service=setup(tmp_path,error_cells=1)
    report=run(service)
    service.owner.bus.values[ORDER[3]]['Lock']=0
    with pytest.raises(RuntimeError,match='motor identity'):
        verify_compensated_hold(service.owner,service.last_attempt,report,admission_check=service.admission)


def inject_stationary_timeout(service, *, failures=1, elapsed=.05045, during=None):
    bus = service.owner.bus
    moving_reads = 0
    lost = 0
    def read(register, motor):
        nonlocal moving_reads, lost
        if register != 'Moving' or motor != 'all':return
        moving_reads += 1
        if moving_reads > 1 and lost < failures:
            lost += 1
            service.owner.clock.now += elapsed
            if during is not None:during()
            raise ConnectionError('injected Moving status packet timeout')
    bus.on_read = read


def test_one_transport_timeout_reacquires_whole_read_only_admission_window(tmp_path):
    service = setup(tmp_path)
    inject_stationary_timeout(service)
    report = run(service)
    window = report['stationary_admission']
    assert window['completed'] and len(window['attempts']) == 2
    first, second = window['attempts']
    assert first['discarded'] and not first['completed'] and first['sample_count'] == 0
    assert first['failed_register'] == 'Moving'
    assert second['completed'] and not second['discarded'] and second['sample_count'] == 3
    assert window['deadline_s'] == pytest.approx(window['started_s']+.5)
    assert second['finished_s'] <= window['deadline_s']
    partial = next(row for row in report['readbacks'] if row.get('admission_attempt') == 1)
    assert partial['present_raw'] and partial['goal_raw'] == report['frozen_hold_goal_raw']
    assert partial['q_position'] == report['admission_q']
    assert partial['readback_time_s'] <= partial['register_reads']['Goal_Position']['returned_s']
    assert partial['register_reads']['Moving']['failed_s'] > partial['readback_time_s']
    assert 'moving' not in partial and 'returned_s' not in partial
    fresh = [row for row in report['readbacks'] if row.get('admission_attempt') == 2]
    assert len(fresh) == 3 and all(row['started_s'] > partial['register_reads']['Moving']['failed_s'] for row in fresh)
    assert writes(service) == [('D_Coefficient',32),('Torque_Limit',300),('I_Coefficient',1),
        ('I_Coefficient',0),('Torque_Limit',200),('D_Coefficient',0),('Torque_Limit',1000)]
    assert report['goal_writes'] == 0


def test_second_transport_failure_never_starts_a_third_window_or_i1(tmp_path):
    service = setup(tmp_path)
    inject_stationary_timeout(service, failures=2)
    report = failed(service, 'status packet timeout')
    attempts = report['stationary_admission']['attempts']
    assert len(attempts) == 2 and all(row['discarded'] and not row['completed'] for row in attempts)
    assert writes(service) == [('I_Coefficient',0),('Torque_Limit',200),
        ('D_Coefficient',0),('Torque_Limit',1000)]
    assert 'integral_exposure' not in report


@pytest.mark.parametrize('change', ['stop', 'lease', 'owner'])
def test_timeout_does_not_retry_after_stop_or_owner_change(tmp_path, change):
    service = setup(tmp_path)
    def during():
        if change == 'stop':service.epoch += 1
        elif change == 'lease':service.owner.lease = object()
        else:service.owner.bridge.runtime = object()
    inject_stationary_timeout(service, during=during)
    report = failed(service, 'STOP|identity')
    assert len(report['stationary_admission']['attempts']) == 1
    assert ('I_Coefficient',1) not in writes(service) and ('Torque_Limit',300) not in writes(service)


@pytest.mark.parametrize('elapsed', [.35, .5])
def test_transport_recovery_cannot_extend_absolute_half_second_budget(tmp_path, elapsed):
    service = setup(tmp_path)
    inject_stationary_timeout(service, elapsed=elapsed)
    report = failed(service, '0.5s budget')
    window = report['stationary_admission']
    assert window['deadline_s'] == pytest.approx(window['started_s']+.5)
    assert window['completed'] is False
    assert writes(service) == [('I_Coefficient',0),('Torque_Limit',200),
        ('D_Coefficient',0),('Torque_Limit',1000)]
    if elapsed == .5:assert len(window['attempts']) == 1


@pytest.mark.parametrize('damage', ['goal', 'drift', 'shape', 'status', 'deadline', 'lease_connection'])
def test_nontransport_fault_never_retries_stationary_admission(tmp_path, damage):
    service = setup(tmp_path)
    bus = service.owner.bus
    seen = 0
    def read(register, motor):
        nonlocal seen
        if register != 'Present_Position' or motor != 'all':return
        seen += 1
        if seen != 2:return
        if damage == 'goal':bus.values[MOTOR]['Goal_Position'] += 1
        elif damage == 'drift':bus.values[MOTOR]['Present_Position'] += 11
        elif damage == 'shape':bus.values[MOTOR]['Present_Position'] = False
        elif damage == 'status':bus.values[MOTOR]['Status'] = 1
        elif damage == 'deadline':service.owner.clock.now += .031
        else:
            def lease_error():raise ConnectionError('lease is unavailable, not a register timeout')
            service.owner.on_lease = lease_error
    bus.on_read = read
    report = failed(service)
    assert len(report['stationary_admission']['attempts']) == 1
    assert ('I_Coefficient',1) not in writes(service)


@pytest.mark.parametrize('register', ['Goal_Position', 'Present_Velocity'])
def test_known_goal_or_status_fault_cannot_be_hidden_by_a_later_transport_failure(tmp_path, register):
    service = setup(tmp_path)
    bus = service.owner.bus
    seen = 0
    def read(actual, motor):
        nonlocal seen
        if actual == 'Present_Position' and motor == 'all':seen += 1
        if seen != 2:return
        if actual == register:
            if register == 'Goal_Position':bus.values[MOTOR]['Goal_Position'] += 1
            else:raise ConnectionError('must never read past a known Status fault')
        if actual == 'Moving' and register == 'Goal_Position':
            raise ConnectionError('must never read past a known Goal fault')
        if actual == 'Status' and register == 'Present_Velocity':bus.values[MOTOR]['Status'] = 1
    bus.on_read = read
    report = failed(service, 'Goal changed|status fault')
    assert len(report['stationary_admission']['attempts']) == 1
    assert report['stationary_admission']['attempts'][0].get('failed_register') is None
    assert ('I_Coefficient',1) not in writes(service)


def test_two_i1_transport_failures_run_cleanup_without_restarting_candidate(tmp_path):
    service = setup(tmp_path)
    bus = service.owner.bus
    def read(register, motor):
        if register == 'Moving' and bus.values[MOTOR]['I_Coefficient'] == 1:
            service.owner.clock.now += .05045
            raise ConnectionError('active I1 transport failure')
    bus.on_read = read
    report = failed(service, 'active I1')
    assert len(report['stationary_admission']['attempts']) == 1
    assert writes(service).count(('I_Coefficient',1)) == 1
    assert writes(service)[-4:] == [('I_Coefficient',0),('Torque_Limit',200),
        ('D_Coefficient',0),('Torque_Limit',1000)]
    assert report['integral_exposure']['within_limit']


def test_slow_active_read_advances_next_sample_from_actual_completion(tmp_path):
    service = setup(tmp_path)
    bus = service.owner.bus
    delayed = []
    def read(register, motor):
        if (register == 'Present_Load' and motor == 'all'
                and bus.values[MOTOR]['I_Coefficient'] == 1 and not delayed):
            delayed.append(True)
            service.owner.clock.now += .05045
    bus.on_read = read
    report = run(service)
    candidate = report['stages']['candidate']
    assert report['completed'] and delayed and not candidate['transport_retries']
    rows = [row for row in report['readbacks'] if row['phase'] == 'candidate' and row['completed']]
    assert len(rows) >= 2
    for previous, row in zip(rows,rows[1:]):
        assert row['scheduled_start_s'] == pytest.approx(previous['returned_s']+.05)
        assert row['started_s'] >= row['scheduled_start_s']
    assert candidate['schedule_basis'] == 'previous_attempt_completion_plus_0.05s'
    assert report['integral_exposure']['duration_s'] <= 10.


def test_one_active_load_packet_drop_reacquires_complete_sample_once(tmp_path):
    service = setup(tmp_path)
    bus = service.owner.bus
    active_load_reads = 0
    def read(register, motor):
        nonlocal active_load_reads
        if register != 'Present_Load' or motor != 'all' or bus.values[MOTOR]['I_Coefficient'] != 1:return
        active_load_reads += 1
        if active_load_reads == 2:
            service.owner.clock.now += .05045
            raise ConnectionError('injected active Present_Load packet timeout')
    bus.on_read = read
    report = run(service)
    candidate = report['stages']['candidate']
    assert report['completed'] and candidate['maximum_snapshot_attempts'] == 2
    assert len(candidate['transport_retries']) == 1
    retry = candidate['transport_retries'][0]
    assert retry['failed_register'] == 'Present_Load' and retry['sample_attempt'] == 1
    attempts = [row for row in report['readbacks']
        if row['phase'] == 'candidate' and row['sample_index'] == retry['sample_index']]
    assert [row['sample_attempt'] for row in attempts] == [1,2]
    assert attempts[0]['discarded'] and not attempts[0]['completed']
    assert attempts[1]['completed'] and not attempts[1]['discarded']
    assert attempts[1]['scheduled_start_s'] == pytest.approx(attempts[0]['failed_s']+.05)
    assert writes(service).count(('I_Coefficient',1)) == 1
    assert report['integral_exposure']['duration_s'] <= 10.


def test_candidate_profiles_are_checked_only_at_active_boundaries(tmp_path):
    service = setup(tmp_path)
    report = run(service)
    candidate = report['stages']['candidate']
    assert [item['label'] for item in candidate['parameter_boundaries']] == [
        'after_i1_verified','before_i0']
    assert all(item['completed'] and item['accepted_attempt'] == 1
        for item in candidate['parameter_boundaries'])
    boundary_reads = [event for item in candidate['parameter_boundaries']
        for attempt in item['attempts'] for event in attempt['register_reads'].values()]
    assert boundary_reads and all(event['num_retry'] == 1 for event in boundary_reads)
    rows = [row for row in report['readbacks'] if row['phase'] == 'candidate' and row['completed']]
    assert rows and all('parameters_by_motor' not in row for row in rows)


@pytest.mark.parametrize('fault',['stop_during_drop','lease','status','geometry'])
def test_active_stop_lease_status_and_geometry_faults_are_never_retried(tmp_path, fault):
    service = setup(tmp_path)
    bus = service.owner.bus
    fired = []
    if fault == 'geometry':
        def geometry():
            if bus.values[MOTOR]['I_Coefficient'] == 1 and not fired:
                fired.append(True)
                raise RuntimeError('injected active geometry rejection')
        service.owner.on_geometry = geometry
    else:
        def read(register, motor):
            if (register != ('Present_Load' if fault == 'stop_during_drop' else 'Present_Position')
                    or motor != 'all' or bus.values[MOTOR]['I_Coefficient'] != 1 or fired):return
            fired.append(True)
            if fault == 'stop_during_drop':
                service.epoch += 1
                raise ConnectionError('active packet timeout concurrent with STOP')
            if fault == 'lease':
                def lost_lease():raise ConnectionError('lease check failed outside bus sync_read')
                service.owner.on_lease = lost_lease
            else:bus.values[MOTOR]['Status'] = 1
        bus.on_read = read
    report = failed(service,'STOP|lease check|status fault|geometry rejection')
    candidate = report['stages']['candidate']
    assert fired and candidate['sample_count'] == 0
    candidate_rows = [row for row in report['readbacks'] if row['phase'] == 'candidate']
    assert len(candidate_rows) == 1 and candidate_rows[0]['sample_attempt'] == 1
    assert all(event['num_retry'] == 1 for event in candidate_rows[0]['register_reads'].values())
    if fault == 'stop_during_drop':
        assert len(candidate['transport_retries']) == 1
        assert candidate['transport_retries'][0]['retry_started'] is False
        assert 'STOP' in candidate['transport_retries'][0]['retry_blocked_error']
    else:assert candidate['transport_retries'] == []
