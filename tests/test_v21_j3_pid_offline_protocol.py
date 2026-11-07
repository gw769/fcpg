import pytest
from dataclasses import replace
from copy import deepcopy

from scripts.prepare_v21_j3_pid_protocol import (
    FakeJ3Bus,fake_transaction,j3_targets,build_protocol,J3PIDExperiment,J3PIDCallbacks,
    J3PIDProtocolError,J3PIDCancelled,ACTUAL_IDENTITY,ACTUAL_BASELINE,MANUAL_SHA)


def test_fake_helper_refuses_real_or_foreign_bus_before_any_call():
    class Foreign:
        def __getattr__(self,name):pytest.fail('foreign bus was touched')
    with pytest.raises(TypeError,match='fake-only'):
        fake_transaction(Foreign())


@pytest.mark.parametrize('failure',[False,True])
def test_only_j3_p_is_changed_and_finally_restored(failure):
    bus=FakeJ3Bus();before=bus.values.copy()
    result=fake_transaction(bus,inject_trial_error=failure)
    assert result['rollback_verified'] and bus.values==before
    writes=[e for e in bus.events if e['kind']=='write']
    assert [(e['register'],e['value']) for e in writes]==[('P_Coefficient',32),('P_Coefficient',16)]
    assert all(e['motor']=='elbow_flex' for e in writes)
    assert result['persistent_state_unchanged']
    assert not result['hardware_used'] and result['physical_response'] is None


def test_failed_ack_after_possible_write_still_restores_original():
    bus=FakeJ3Bus(fail_after_apply=True);result=fake_transaction(bus)
    assert 'write failure' in result['primary_error']
    assert result['rollback_verified'] and bus.values['P_Coefficient']==16
    assert not result['trial_marker_reached']


def test_ignored_runtime_write_never_triggers_unlock_or_torque_fallback():
    bus=FakeJ3Bus(ignore_candidate=True);result=fake_transaction(bus)
    assert not result['trial_marker_reached'] and result['rollback_verified']
    assert all(e['register']=='P_Coefficient' for e in bus.events if e['kind']=='write')


def test_restore_failure_keeps_primary_error_and_seals_result():
    bus=FakeJ3Bus(fail_restore=True);result=fake_transaction(bus,inject_trial_error=True)
    assert 'operator STOP' in result['primary_error']
    assert 'restore transport failure' in result['rollback_error']
    assert not result['rollback_verified'] and result['sealed_if_rollback_unverified']


@pytest.mark.parametrize('overrides',[{'Torque_Enable':0},{'Lock':0},{'D_Coefficient':32},{'Goal_Position':3103}])
def test_unknown_or_loaded_preconditions_refuse_all_parameter_writes(overrides):
    bus=FakeJ3Bus(overrides=overrides);result=fake_transaction(bus)
    assert result['primary_error'] and not result['candidate_attempted']
    assert not any(e['kind']=='write' for e in bus.events)


def test_protocol_is_nonexecutable_bounded_and_preserves_protection():
    protocol=build_protocol();assert not protocol['executable'] and not protocol['live_execution_enabled']
    assert protocol['preferred_first_comparison']['register']=='P_Coefficient'
    source=protocol['source_pose']['nominal_home'];targets=j3_targets(source)
    previous=source
    for target in targets:
        assert abs(target[2]-source[2])<=3 and abs(target[2]-previous[2])<=1.5
        assert all(target[i]==source[i] for i in (0,1,3,4,5))
        previous=target
    assert protocol['register_contract']['Torque_Enable']['writes_forbidden']
    assert protocol['register_contract']['Lock']['writes_forbidden']
    assert protocol['budgets']['max_test_chunks']==2
    assert protocol['budgets']['max_test_policy_writes']==10
    assert protocol['measured_identity']==ACTUAL_IDENTITY
    assert protocol['measured_baseline']==ACTUAL_BASELINE


def start_baseline(bus=None):
    bus=FakeJ3Bus() if bus is None else bus
    session=bus.session();report=session.baseline(enable=True)
    assert report['configuration_restored_verified']
    return bus,session


def writes(bus):return [e for e in bus.events if e['kind']=='write']
def drives(bus):return [e for e in bus.events if e['kind']=='execute_five']


def test_disabled_phase_invokes_no_callback_including_clock_and_cancel():
    def forbidden(*args):pytest.fail('default dry-run called an external callback')
    session=J3PIDExperiment(J3PIDCallbacks(*([forbidden]*7)),clock_fn=forbidden)
    assert session.baseline()['callbacks_invoked']==0
    assert session.variant()['callbacks_invoked']==0
    assert session.phase=='new' and session.reserved_commands==0


def test_explicit_enable_still_requires_pinned_firmware_and_document_review():
    bus=FakeJ3Bus()
    session=J3PIDExperiment(bus.callbacks())
    with pytest.raises(ValueError,match='pin the measured'):
        session.baseline(enable=True)
    assert not bus.events
    session=J3PIDExperiment(bus.callbacks(),expected_identity=ACTUAL_IDENTITY)
    with pytest.raises(ValueError,match='applicability'):
        session.baseline(enable=True)
    assert not bus.events


def test_one_baseline_and_one_variant_has_ten_command_budget_and_durable_seal_order():
    bus,session=start_baseline();baseline=deepcopy(session.reports[0])
    report=session.variant(enable=True)
    assert [event['phase'] for event in drives(bus)]==['baseline','variant']
    assert session.reserved_commands==10 and session.phase=='finished'
    assert report['configuration_restored_verified'] and bus.values['P_Coefficient']==16
    assert not report['actual_response_verified']  # fake fixture is deliberately motionless
    assert baseline['source']['q_position']==report['source']['q_position']
    before_write=next(i for i,e in enumerate(bus.events) if e['kind']=='write')
    assert any(e['kind']=='persist' and e['event']=='P32_write_intent' for e in bus.events[:before_write])
    assert all(saved['motion_sealed'] for saved in bus.saved)
    assert bus.saved[-1]['session_state']=='finished'
    for method in (session.baseline,session.variant):
        with pytest.raises(J3PIDProtocolError,match='prior attempt sealed'):
            method(enable=True)
    with pytest.raises(J3PIDProtocolError,match='remains sealed'):
        session.require_policy_resume()
    assert len(drives(bus))==2 and len(writes(bus))==2
    assert all('call_started_s' in check and 'call_returned_s' in check for check in report['read_checks'])


@pytest.mark.parametrize('overrides',[
    {'Model_Number':778},{'Firmware_Major_Version':4},{'Firmware_Minor_Version':9},{'ID':4},
    {'Moving':1},{'I_Coefficient':1},{'Protection_Current':500},{'Minimum_Startup_Force':0},
    {'Lock':0},{'Torque_Enable':0},{'Operating_Mode':1},{'Goal_Position':3156}])
def test_preflight_failures_do_not_issue_parameter_or_motion_writes(overrides):
    bus=FakeJ3Bus(overrides=overrides);session=bus.session()
    with pytest.raises(ValueError):session.baseline(enable=True)
    assert not writes(bus) and not drives(bus)
    assert session.phase=='failed_sealed'


def test_home_near_and_pairing_are_checked_without_automatic_home():
    bus=FakeJ3Bus();bus.q[1]+=3.01;session=bus.session()
    with pytest.raises(ValueError,match='HOME-near'):session.baseline(enable=True)
    assert not drives(bus)
    bus,session=start_baseline();bus.q[2]+=.251
    with pytest.raises(ValueError,match='more than 0.25'):session.variant(enable=True)
    assert len(drives(bus))==1 and not writes(bus)


def test_external_protected_configuration_change_between_phases_is_not_repaired():
    bus,session=start_baseline();bus.others['shoulder_pan']['P_Coefficient']=32
    with pytest.raises(ValueError,match='configuration changed'):session.variant(enable=True)
    assert not writes(bus) and len(drives(bus))==1


def test_cancel_before_action_never_starts_the_baseline():
    bus=FakeJ3Bus();bus.cancelled_flag=True;session=bus.session()
    with pytest.raises(J3PIDCancelled):session.baseline(enable=True)
    assert not drives(bus) and not writes(bus)
    assert session.phase=='failed_sealed'


@pytest.mark.parametrize('where',['intent','write','read_after_write','drive'])
def test_variant_cancel_points_restore_only_p_and_never_resume(where):
    bus,session=start_baseline()
    def hook(owner,event):
        trigger=((where=='intent' and event['kind']=='persist' and event.get('event')=='P32_write_intent')
            or (where=='write' and event['kind']=='write' and event['value']==32)
            or (where=='read_after_write' and event['kind']=='snapshot' and owner.values['P_Coefficient']==32)
            or (where=='drive' and event['kind']=='execute_five' and event['phase']=='variant'))
        if trigger:owner.cancelled_flag=True
    bus.hook=hook
    with pytest.raises(J3PIDCancelled) as caught:session.variant(enable=True)
    report=caught.value.j3_pid_report
    assert bus.values['P_Coefficient']==16 and session.phase=='failed_sealed'
    assert all(e['register']=='P_Coefficient' for e in writes(bus))
    assert len(drives(bus))==(2 if where=='drive' else 1)
    if where=='intent':assert not writes(bus)
    else:assert report['p_restored_verified'] and report['configuration_restored_verified']
    with pytest.raises(J3PIDProtocolError):session.variant(enable=True)


def test_stop_callback_error_does_not_skip_p_restore_or_mask_primary_error():
    bus,session=start_baseline()
    primary=ValueError('original driver error')
    def hook(owner,event):
        if event['kind']=='execute_five':raise primary
        if event['kind']=='simulated_existing_STOP_HOLD':raise RuntimeError('STOP transport error')
    bus.hook=hook
    with pytest.raises(ValueError) as caught:session.variant(enable=True)
    assert caught.value is primary
    report=primary.j3_pid_report
    assert 'STOP transport error' in report['stop_error']
    assert report['p_restored_verified'] and report['configuration_restored_verified']
    assert bus.values['P_Coefficient']==16 and session.phase=='failed_sealed'


def test_stop_error_without_primary_is_raised_after_verified_restoration():
    bus,session=start_baseline()
    def hook(owner,event):
        if event['kind']=='simulated_existing_STOP_HOLD':raise RuntimeError('STOP failed independently')
    bus.hook=hook
    with pytest.raises(RuntimeError,match='STOP failed independently') as caught:
        session.variant(enable=True)
    assert caught.value.j3_pid_report['configuration_restored_verified']
    assert bus.values['P_Coefficient']==16


def test_failed_ack_is_restored_and_primary_kept_even_when_restore_also_fails():
    bus,session=start_baseline();bus.fail_after_apply=True;bus.fail_restore=True
    with pytest.raises(ConnectionError,match='write failure after') as caught:session.variant(enable=True)
    report=caught.value.j3_pid_report
    assert 'restore transport failure' in report['rollback_error']
    assert not report['p_restored_verified'] and report['further_motion_blocked']
    assert len(drives(bus))==1 and bus.values['P_Coefficient']==32
    assert session.phase=='failed_sealed'


def test_read_error_after_p_write_still_restores_if_next_guard_read_works():
    bus,session=start_baseline();fired=[False]
    def hook(owner,event):
        if event['kind']=='snapshot' and owner.values['P_Coefficient']==32 and not fired[0]:
            fired[0]=True;raise ConnectionError('after-P read failure')
    bus.hook=hook
    with pytest.raises(ConnectionError,match='after-P read failure') as caught:session.variant(enable=True)
    report=caught.value.j3_pid_report
    assert report['p_restored_verified'] and bus.values['P_Coefficient']==16
    assert any('call_failed_s' in entry for entry in report['read_checks'])
    assert len(drives(bus))==1


def test_full_snapshot_failure_does_not_block_independent_j3_restore():
    bus,session=start_baseline()
    def hook(owner,event):
        if event['kind']=='snapshot' and owner.values['P_Coefficient']==32:raise ConnectionError('guard unreadable')
    bus.hook=hook
    with pytest.raises(ConnectionError,match='guard unreadable') as caught:session.variant(enable=True)
    report=caught.value.j3_pid_report
    assert [e['value'] for e in writes(bus)]==[32,16]
    assert report['p_restored_verified'] and report['configuration_restored_verified']
    assert session.phase=='failed_sealed'


@pytest.mark.parametrize('field,value',[('I_Coefficient',1),('D_Coefficient',32),('Torque_Limit',999)])
def test_external_pid_or_limit_change_restores_only_p_and_seals_anomaly(field,value):
    bus,session=start_baseline()
    def hook(owner,event):
        if event['kind']=='write' and event['value']==32:owner.values[field]=value
    bus.hook=hook
    with pytest.raises(ValueError,match='configuration changed') as caught:session.variant(enable=True)
    report=caught.value.j3_pid_report
    assert [e['value'] for e in writes(bus)]==[32,16]
    assert bus.values[field]==value and bus.values['P_Coefficient']==16
    assert report['p_restored_verified'] and not report['configuration_restored_verified']
    assert report['rollback_error']


@pytest.mark.parametrize('field',['Lock','Torque_Enable'])
def test_lost_lock_or_torque_never_causes_forbidden_configuration_recovery(field):
    bus,session=start_baseline()
    def hook(owner,event):
        if event['kind']=='write' and event['value']==32:owner.values[field]=0
    bus.hook=hook
    with pytest.raises(ValueError,match='Torque1/Lock1') as caught:session.variant(enable=True)
    assert len(writes(bus))==1 and bus.values[field]==0
    assert caught.value.j3_pid_report['rollback_error']


@pytest.mark.parametrize('event',['session_sealed','P32_write_intent','five_command_intent','five_command_result','phase_finished_still_sealed'])
def test_persist_faults_block_actions_or_restore_already_attempted_p(event):
    bus,session=start_baseline();fired=[False]
    def hook(owner,entry):
        if entry['kind']=='persist' and entry['event']==event and not fired[0]:
            fired[0]=True;raise OSError('durability failure '+event)
    bus.hook=hook
    with pytest.raises(OSError,match='durability failure'):session.variant(enable=True)
    assert session.phase=='failed_sealed' and bus.values['P_Coefficient']==16
    if event in ('session_sealed','P32_write_intent'):assert not writes(bus) and len(drives(bus))==1
    assert all(saved['motion_sealed'] for saved in bus.saved)


def test_persist_without_saved_receipt_never_writes_p():
    bus,session=start_baseline()
    session.callbacks=replace(session.callbacks,persist=lambda report:None)
    with pytest.raises(J3PIDProtocolError,match='durable sealed intent'):session.variant(enable=True)
    assert not writes(bus) and len(drives(bus))==1


def test_stale_or_future_snapshot_is_rejected_before_commands():
    bus=FakeJ3Bus();session=bus.session();original=session.callbacks.read_snapshot
    def stale():
        result=original();result['started_s']-=.2;result['q_time_s']-=.2;return result
    session.callbacks=replace(session.callbacks,read_snapshot=stale)
    with pytest.raises(ValueError,match='stale'):session.baseline(enable=True)
    assert not drives(bus)


def test_missing_readback_geometry_proof_stops_before_parameter_changes():
    bus=FakeJ3Bus();session=bus.session();original=session.callbacks.read_snapshot
    def unchecked():
        result=original();result['source_geometry_validated']=False;return result
    session.callbacks=replace(session.callbacks,read_snapshot=unchecked)
    with pytest.raises(ValueError,match='original readback geometry'):session.baseline(enable=True)
    assert not drives(bus) and not writes(bus)


@pytest.mark.parametrize('fault',['counts','commands','late','geometry','goal','source','endpoint',
                                'goal_late','goal_clock','q5_late','fake_lateness','q5_count'])
def test_bad_variant_receipt_retains_actual_result_restores_p_and_blocks_retry(fault):
    bus,session=start_baseline();original=session.callbacks.execute_five
    def bad(commands,phase):
        result=original(commands,phase)
        if fault=='counts':result['policy_commands_returned']=4
        elif fault=='commands':result['commands_position'][0][0]+=.1
        elif fault=='late':result['write_lateness_s'][2]=.031
        elif fault=='geometry':result['original_geometry_admitted']=False
        elif fault=='goal':result['policy_goal_readbacks'][2]['goal_raw']['elbow_flex']+=1
        elif fault=='source':result['measured_q_path'][0][2]+=.1
        elif fault=='endpoint':result.update(endpoint_q_position=bus.q.copy(),endpoint_time_s=0.)
        elif fault=='goal_late':result['policy_goal_readbacks'][2]['policy_io_lateness_s']=.031
        elif fault=='goal_clock':result['policy_goal_readbacks'][2]['read_started_s']-=.1
        elif fault=='q5_late':result['readback_times_s'][-1]+=.031
        elif fault=='fake_lateness':result['write_completed_times_s'][2]+=.04
        else:result['measured_q_path'].append(bus.q.copy())
        return result
    session.callbacks=replace(session.callbacks,execute_five=bad)
    with pytest.raises(ValueError) as caught:session.variant(enable=True)
    report=caught.value.j3_pid_report
    assert report['drive_returned'] and report['drive_result']
    assert report['p_restored_verified'] and bus.values['P_Coefficient']==16
    with pytest.raises(J3PIDProtocolError):session.variant(enable=True)
    assert len(drives(bus))==2


def test_baseexception_during_variant_still_restores_and_rethrows_same_exception():
    bus,session=start_baseline();interrupt=KeyboardInterrupt('operator interruption')
    def hook(owner,event):
        if event['kind']=='execute_five':raise interrupt
    bus.hook=hook
    with pytest.raises(KeyboardInterrupt) as caught:session.variant(enable=True)
    assert caught.value is interrupt and interrupt.j3_pid_report['configuration_restored_verified']
    assert bus.values['P_Coefficient']==16


def test_gain_change_pose_shift_refuses_variant_and_restores_p():
    bus,session=start_baseline()
    def hook(owner,event):
        if event['kind']=='write' and event['value']==32:owner.q[2]+=.3
    bus.hook=hook
    with pytest.raises(ValueError,match='more than 0.25'):session.variant(enable=True)
    assert len(drives(bus))==1 and bus.values['P_Coefficient']==16


def test_reentrant_attempt_is_rejected_before_second_driver():
    bus=FakeJ3Bus();session=bus.session();rejected=[]
    def hook(owner,event):
        if event['kind']=='execute_five':
            with pytest.raises(J3PIDProtocolError,match='already owns'):session.baseline(enable=True)
            rejected.append(True)
    bus.hook=hook;session.baseline(enable=True)
    assert rejected==[True] and len(drives(bus))==1


def test_other_axis_torque_anomaly_still_restores_j3_p_only_then_seals():
    bus,session=start_baseline()
    def hook(owner,event):
        if event['kind']=='write' and event['value']==32:owner.others['shoulder_pan']['Torque_Enable']=0
    bus.hook=hook
    with pytest.raises(ValueError,match='other motor configuration') as caught:session.variant(enable=True)
    report=caught.value.j3_pid_report
    assert [e['value'] for e in writes(bus)]==[32,16]
    assert report['p_restored_verified'] and not report['configuration_restored_verified']
    assert bus.others['shoulder_pan']['Torque_Enable']==0


def test_final_restoration_read_failure_remains_unverified_despite_p_value_being_restored():
    bus,session=start_baseline();restoring=[False]
    def hook(owner,event):
        if event['kind']=='write' and event['value']==16:restoring[0]=True
        if event['kind']=='snapshot' and restoring[0]:raise ConnectionError('final readback unavailable')
    bus.hook=hook
    with pytest.raises(ConnectionError,match='final readback unavailable') as caught:session.variant(enable=True)
    assert bus.values['P_Coefficient']==16
    report=caught.value.j3_pid_report
    assert report['p_restored_verified'] and not report['configuration_restored_verified']
    assert report['further_motion_blocked'] and session.phase=='failed_sealed'


def test_false_saved_receipt_on_final_report_leaves_durable_session_sealed():
    bus,session=start_baseline();original=session.callbacks.persist
    def persistence(report):
        if report['events'][-1]['event']=='phase_finished_still_sealed':return dict(saved=False)
        return original(report)
    session.callbacks=replace(session.callbacks,persist=persistence)
    with pytest.raises(J3PIDProtocolError,match='durable sealed intent'):session.variant(enable=True)
    assert session.phase=='failed_sealed' and bus.values['P_Coefficient']==16
    assert bus.saved[-1]['motion_sealed']


@pytest.mark.parametrize('where',['P32_write_intent','five_command_intent'])
def test_slow_persistence_uses_one_mandatory_actual_read_before_the_action(where):
    bus,session=start_baseline()
    def hook(owner,event):
        if event['kind']=='persist' and event['event']==where:owner.now+=.250
    bus.hook=hook
    report=session.variant(enable=True)
    assert len(drives(bus))==2 and [e['value'] for e in writes(bus)]==[32,16]
    snapshots={entry['label']:entry['snapshot'] for entry in report['read_checks']}
    current=snapshots['before_P32_write' if where=='P32_write_intent' else 'before_dispatch']
    original=snapshots['before' if where=='P32_write_intent' else 'after_P32']
    assert current['q_time_s']-original['q_time_s']==pytest.approx(.251)
    assert current['q_position']==original['q_position']
    assert report['p_write_source']==snapshots['before_P32_write']
    assert report['dispatch_source']==snapshots['before_dispatch']
    assert report['drive_started_s']==report['dispatch_source']['q_time_s']
    assert report['commands_position']==drives(bus)[-1]['commands']
    assert bus.values['P_Coefficient']==16 and session.phase=='finished'


@pytest.mark.parametrize('persist_delay_s',[.070,.101,.250])
def test_command_intent_keeps_its_original_source_and_dispatch_gets_a_distinct_actual_sample(persist_delay_s):
    bus,session=start_baseline();persist=session.callbacks.persist;checkpoints=[]
    def delayed(report):
        receipt=persist(report)
        if report['phase']=='variant' and report['events'][-1]['event'] in (
                'P32_readback_verified','five_command_intent'):
            checkpoints.append(deepcopy(report));bus.now+=persist_delay_s
        return receipt
    session.callbacks=replace(session.callbacks,persist=delayed)
    report=session.variant(enable=True)
    assert report['drive_attempted'] and report['five_command_receipt_validated']
    assert report['drive_started_s']-report['source']['q_time_s']==pytest.approx(persist_delay_s+.001)
    assert report['drive_started_s']==report['dispatch_source']['q_time_s']
    assert len(checkpoints)==1
    checkpoint=checkpoints[0]
    assert [event['event'] for event in checkpoint['events'][-2:]]==[
        'P32_readback_verified','five_command_intent']
    source=next(check['snapshot'] for check in report['read_checks'] if check['label']=='after_P32')
    assert checkpoint['source']==report['source']==source
    assert checkpoint['commands_position']==j3_targets(source['q_position'])
    assert checkpoint['commands_position']==drives(bus)[-1]['commands']
    assert 'dispatch_source' not in checkpoint
    dispatch=next(check['snapshot'] for check in report['read_checks'] if check['label']=='before_dispatch')
    assert report['dispatch_source']==dispatch and source['q_time_s']<dispatch['started_s']
    assert checkpoint['p_write_intent_persisted'] and checkpoint['p_write_returned']
    assert source['registers']['P_Coefficient']==32
    assert [check['label'] for check in report['read_checks']]==[
        'before','before_P32_write','after_P32','before_dispatch','restore_guard','after_P16','after_restore']
    assert [entry['value'] for entry in writes(bus)]==[32,16]
    assert report['p_restored_verified'] and report['configuration_restored_verified']
    assert len(drives(bus))==2


def test_baseline_slow_intent_does_not_require_a_retry_or_a_new_command_sequence():
    bus=FakeJ3Bus();session=bus.session()
    def hook(owner,event):
        if event['kind']=='persist' and event['event']=='five_command_intent':owner.now+=.250
    bus.hook=hook
    report=session.baseline(enable=True)
    assert report['drive_started_s']-report['source']['q_time_s']==pytest.approx(.251)
    assert report['drive_started_s']==report['dispatch_source']['q_time_s']
    assert [check['label'] for check in report['read_checks']]==['before','before_dispatch','baseline_final']
    intent=next(saved for saved in bus.saved if saved['events'][-1]['event']=='five_command_intent')
    assert intent['source']==report['source']
    assert intent['commands_position']==drives(bus)[0]['commands']==report['commands_position']
    assert len(drives(bus))==1 and not writes(bus)


@pytest.mark.parametrize('where',['P32_write_intent','five_command_intent'])
@pytest.mark.parametrize('axis',range(6))
def test_changed_actual_axis_after_slow_intent_refuses_frozen_commands_without_retry(where,axis):
    bus,session=start_baseline()
    def hook(owner,event):
        if event['kind']=='persist' and event['event']==where:
            owner.now+=.250;owner.q[axis]+=.01
    bus.hook=hook
    with pytest.raises(ValueError,match='source changed after durable intent') as caught:
        session.variant(enable=True)
    report=caught.value.j3_pid_report
    assert not report['drive_attempted'] and len(drives(bus))==1
    assert [entry['value'] for entry in writes(bus)]==([] if where=='P32_write_intent' else [32,16])
    label='before_P32_write' if where=='P32_write_intent' else 'before_dispatch'
    assert sum(check['label']==label for check in report['read_checks'])==1
    assert bus.values['P_Coefficient']==16
    with pytest.raises(J3PIDProtocolError):session.variant(enable=True)


@pytest.mark.parametrize('where',['P32_write_intent','five_command_intent'])
def test_returning_the_old_snapshot_after_logging_is_not_a_real_new_read(where):
    bus,session=start_baseline();original=session.callbacks.read_snapshot
    previous=[None];replay=[False]
    def read():
        if replay[0]:
            replay[0]=False
            return deepcopy(previous[0])
        result=original();previous[0]=deepcopy(result);return result
    def hook(owner,event):
        if event['kind']=='persist' and event['event']==where:
            owner.now+=.250;replay[0]=True
    bus.hook=hook;session.callbacks=replace(session.callbacks,read_snapshot=read)
    with pytest.raises(ValueError,match='stale or actual read clocks') as caught:
        session.variant(enable=True)
    report=caught.value.j3_pid_report
    assert not report['drive_attempted'] and len(drives(bus))==1
    assert [entry['value'] for entry in writes(bus)]==([] if where=='P32_write_intent' else [32,16])
    assert bus.values['P_Coefficient']==16
    label='before_P32_write' if where=='P32_write_intent' else 'before_dispatch'
    stale=next(check for check in report['read_checks'] if check['label']==label)
    assert stale['snapshot']['q_time_s']<stale['call_started_s']
    assert sum(check['label']==label for check in report['read_checks'])==1


@pytest.mark.parametrize('where',['P32_write_intent','five_command_intent'])
@pytest.mark.parametrize('target',['J3','other_motor'])
def test_configuration_changes_during_logging_block_the_new_source(where,target):
    bus,session=start_baseline()
    def hook(owner,event):
        if event['kind']=='persist' and event['event']==where:
            owner.now+=.250
            values=owner.values if target=='J3' else owner.others['shoulder_pan']
            values['I_Coefficient']=1
    bus.hook=hook
    with pytest.raises(ValueError,match='configuration changed') as caught:
        session.variant(enable=True)
    assert not caught.value.j3_pid_report['drive_attempted'] and len(drives(bus))==1
    assert [entry['value'] for entry in writes(bus)]==([] if where=='P32_write_intent' else [32,16])
    assert bus.values['P_Coefficient']==16


def test_stale_actual_predispatch_sample_is_rejected_without_a_third_read():
    bus=FakeJ3Bus();session=bus.session();original=session.callbacks.read_snapshot;calls=[0]
    def read():
        calls[0]+=1;result=original()
        if calls[0]==2:bus.now+=.101
        return result
    session.callbacks=replace(session.callbacks,read_snapshot=read)
    with pytest.raises(ValueError,match='stale or actual read clocks'):
        session.baseline(enable=True)
    assert calls[0]==2 and not drives(bus) and not writes(bus)


@pytest.mark.parametrize('fault',['goal_command','goal_retry','goal_expected_bool','early_q5',
                                'previous_receipt','future_receipt'])
def test_receipt_is_bound_to_current_commands_and_execution_interval(fault):
    bus,session=start_baseline();original=session.callbacks.execute_five
    def bad(commands,phase):
        result=original(commands,phase)
        if fault=='goal_command':
            result['policy_goal_readbacks'][0]['requested_goal_position']={
                name:0. for name in result['policy_goal_readbacks'][0]['goal_raw']}
        elif fault=='goal_retry':result['policy_goal_readbacks'][0]['num_retry']=1
        elif fault=='goal_expected_bool':
            event=result['policy_goal_readbacks'][0]
            event['goal_raw']['gripper']=1;event['expected_goal_raw']['gripper']=True
        elif fault=='early_q5':result['readback_times_s'][-1]-=.09
        else:
            shift=-10. if fault=='previous_receipt' else 10.
            for key in ('command_times_s','write_completed_times_s','readback_times_s'):
                result[key]=[stamp+shift for stamp in result[key]]
            for event in result['policy_goal_readbacks']:
                for key in ('callback_started_s','started_s','read_started_s','read_returned_s','returned_s','callback_returned_s'):
                    if key in event:event[key]+=shift
        return result
    session.callbacks=replace(session.callbacks,execute_five=bad)
    with pytest.raises(ValueError) as caught:session.variant(enable=True)
    assert caught.value.j3_pid_report['drive_returned']
    assert caught.value.j3_pid_report['p_restored_verified']
    assert bus.values['P_Coefficient']==16 and session.phase=='failed_sealed'


def test_enabled_session_requires_independent_restore_guard_before_touching_callbacks():
    bus=FakeJ3Bus();session=bus.session()
    session.callbacks=replace(session.callbacks,read_restore_guard=None)
    with pytest.raises(TypeError,match='seven explicit'):session.baseline(enable=True)
    assert not bus.events


@pytest.mark.parametrize('field,value',[
    ('ID',4),('Model_Number',778),('Firmware_Minor_Version',9),
    ('Torque_Enable',0),('Lock',0),('Operating_Mode',1)])
def test_minimal_restore_guard_never_blindly_writes_to_changed_or_unlocked_motor(field,value):
    bus,session=start_baseline()
    def hook(owner,event):
        if event['kind']=='write' and event['value']==32:owner.values[field]=value
    bus.hook=hook
    with pytest.raises(ValueError) as caught:session.variant(enable=True)
    report=caught.value.j3_pid_report
    assert [entry['value'] for entry in writes(bus)]==[32]
    assert report['restore_guard_error'] and not report['p_restored_verified']
    assert session.phase=='failed_sealed'


def test_minimal_restore_guard_read_failure_remains_sealed_without_blind_write():
    bus,session=start_baseline()
    def hook(owner,event):
        if event['kind']=='restore_guard':raise ConnectionError('J3 identity guard unreadable')
    bus.hook=hook
    with pytest.raises(ConnectionError,match='identity guard unreadable') as caught:session.variant(enable=True)
    assert [entry['value'] for entry in writes(bus)]==[32]
    assert caught.value.j3_pid_report['restore_guard_error']
    assert session.phase=='failed_sealed'


def test_failed_restore_ack_is_verified_without_second_write_but_session_stays_sealed():
    bus,session=start_baseline();original=session.callbacks.write_p
    def failed_ack(value):
        original(value)
        if value==16:raise ConnectionError('P16 ACK lost after application')
    session.callbacks=replace(session.callbacks,write_p=failed_ack)
    with pytest.raises(ConnectionError,match='P16 ACK lost') as caught:session.variant(enable=True)
    report=caught.value.j3_pid_report
    assert [entry['value'] for entry in writes(bus)]==[32,16]
    assert report['p_restored_verified'] and report['configuration_restored_verified']
    assert report['p_restore_write_error'] and session.phase=='failed_sealed'


def test_stop_and_final_persist_failure_leave_independently_verified_p16():
    bus,session=start_baseline()
    def hook(owner,event):
        if event['kind']=='simulated_existing_STOP_HOLD':raise RuntimeError('STOP callback failed')
        if event['kind']=='persist' and event['event']=='phase_finished_still_sealed':
            raise OSError('final journal unavailable')
    bus.hook=hook
    with pytest.raises(OSError,match='final journal unavailable') as caught:session.variant(enable=True)
    report=caught.value.j3_pid_report
    assert report['stop_error'] and report['persistence_error']
    assert report['p_restored_verified'] and bus.values['P_Coefficient']==16
    assert session.phase=='failed_sealed'
