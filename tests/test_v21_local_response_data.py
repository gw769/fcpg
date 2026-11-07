from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
import torch

from fcpg.prod_v20_policy import empty_history
from fcpg.v21_local_response_data import BLOCK01_RECORD_ID, MANIFEST_SCHEMA, LocalResponseDataset
from fcpg.v21_models import V21LeWM


def save_policy(tmp_path, identity='execution-1.json', *, fit=True, validation=False, holdout=False):
    q = torch.tensor([0., -90., 95., 60., -50., 30.])
    commands = q[None].repeat(5, 1)
    commands[:, 2] -= torch.arange(1., 6.)
    measured = q[None].repeat(6, 1)
    measured[:, 1] += torch.arange(6.) * .1
    record = dict(completed=True, policy_commands_attempted=5, policy_commands_returned=5,
        commands_position=commands.tolist(), measured_q_path=measured.tolist(),
        command_times_s=[100.0001, 100.1007, 100.2014, 100.3104, 100.416],
        write_completed_times_s=[100.0002, 100.1008, 100.2015, 100.3105, 100.4161],
        readback_times_s=[100., 100.1002, 100.201, 100.310, 100.415, 100.523],
        source_time_s=100., source_q_position=q.tolist(), action_source='v21_lewm_cem',
        predicted_q_path=[[999.] * 6] * 6, following_q_path=[[777.] * 6],
        completion_hold=dict(goal_register_verified=True, held_position=[666.] * 6))
    history = empty_history(1)
    history.valid[0, -1] = True
    history.commands_position[0, -1] = q
    history.readbacks_position[0, -1] = q + .2
    history.time_deltas[0, -1] = .1
    state = SimpleNamespace(q=q[None]+.5, time=torch.tensor([99.9], dtype=torch.float64),
        history=history, history_ends=torch.tensor([[0., 0., 99.5]], dtype=torch.float64))
    snapshot = dict(kind='actual-actor-input', schema=2, executable=False,
        commands_position=commands.tolist(), dispatch_source_time_s=100.,
        dispatch_source_q_position=q.tolist(), action_source='v21_lewm_cem', state=state)
    path = tmp_path / identity
    snapshot_path = path.with_suffix('.pt')
    path.write_text(json.dumps(record))
    torch.save(snapshot, snapshot_path)
    row = dict(record_id=identity, record_path=str(path), snapshot_path=str(snapshot_path),
        source_class='policy_execution', role='additional_diagnostic_holdout' if holdout else 'candidate',
        eligible_for_real_response_adapter=fit and not validation and not holdout,
        history=dict(valid_count=1), goal_verification=dict(status='unknown'),
        exclusion_reasons=['prior_validation_or_test_role_preserved'] if validation else [],
        provenance_declarations=[dict(declares_fit=fit, declares_validation_or_test=validation,
                                     split='train' if fit else 'recent_dev')])
    return row, record, snapshot


def manifest(tmp_path, rows):
    path = tmp_path / 'inventory.json'
    path.write_text(json.dumps(dict(schema=MANIFEST_SCHEMA, records=rows)))
    return path


def save_record(row, record):
    from pathlib import Path
    Path(row['record_path']).write_text(json.dumps(record))


def add_goal_proofs(record):
    count = len(record['commands_position'])
    record.update(policy_goal_reads_attempted=count, policy_goal_reads_returned=count,
                  policy_goal_registers_verified=count)
    record['policy_goal_readbacks'] = [dict(command_index=i, goal_register_verified=True,
        read_attempted=True, read_returned=True, motor_register_writes=0,
        goal_raw={'elbow_flex': 3000-i}, expected_goal_raw={'elbow_flex': 3000-i},
        read_started_s=end+.00001, read_returned_s=end+.00002)
        for i, end in enumerate(record['write_completed_times_s'])]


def test_split_roles_override_eligible_flag_and_keep_block01_out_of_training(tmp_path):
    train, _, _ = save_policy(tmp_path)
    validation, _, _ = save_policy(tmp_path, 'execution-2.json', fit=False, validation=True)
    validation['eligible_for_real_response_adapter'] = True  # Contradictory flag cannot move it to TRAIN.
    conflict, _, _ = save_policy(tmp_path, 'execution-3.json', validation=True)
    holdout, _, _ = save_policy(tmp_path, BLOCK01_RECORD_ID)  # Immutable diagnostic identity wins.
    sealed, _, _ = save_policy(tmp_path, 'execution-5.json', fit=False, validation=True)
    sealed['provenance_declarations'][0]['split'] = 'sealed_test'
    path = manifest(tmp_path, [train, validation, conflict, holdout, sealed])
    assert LocalResponseDataset(path).record_ids == (train['record_id'],)
    assert LocalResponseDataset(path, split='validation').record_ids == (validation['record_id'],)
    assert LocalResponseDataset(path, split='role_conflict').record_ids == (conflict['record_id'],)
    assert LocalResponseDataset(path, split='holdout').record_ids == (BLOCK01_RECORD_ID,)
    assert LocalResponseDataset(path).summary()['group_counts'] == dict(
        train=1, validation=1, role_conflict=1, holdout=1, excluded=1)


def test_batch_uses_actual_dispatch_y_clocks_and_saved_history(tmp_path):
    row, record, snapshot = save_policy(tmp_path)
    row['measured_q_path'] = [[888.] * 6] * 6  # Cached manifest values are not labels.
    ds = LocalResponseDataset(manifest(tmp_path, [row]))
    batch = ds.batch([0])
    assert torch.equal(batch.q_position[0], torch.tensor(record['source_q_position']))
    assert torch.equal(batch.y[0], torch.tensor(record['measured_q_path'][1:]))
    assert not torch.equal(batch.y, batch.trace.positions)
    assert torch.equal(batch.history.readbacks_position, snapshot['state'].history.readbacks_position)
    assert torch.equal(batch.query_offsets[0], torch.tensor(record['readback_times_s'][1:], dtype=torch.float64)-100.)
    assert torch.equal(batch.trace.offsets[0], torch.tensor(record['command_times_s'], dtype=torch.float64)-100.)
    assert batch.trace.positions.shape == (1, 5, 6) and batch.trace.source == 'recorded'
    assert batch.goal_status == ('unknown',) and batch.split == 'train'
    # Returned batches own their tensors and cannot mutate the CPU cache.
    batch.y.fill_(999)
    assert torch.equal(ds.batch([0]).y[0], torch.tensor(record['measured_q_path'][1:]))


@pytest.mark.parametrize('change,match', [
    (lambda r: r.update(policy_commands_returned=4), 'incomplete actual command'),
    (lambda r: r.update(measured_q_path=None), 'actual measured q path'),
    (lambda r: r['command_times_s'].__setitem__(0, 99.9), 'not causal'),
    (lambda r: r['write_completed_times_s'].__setitem__(4, 100.7), 'not causal'),
    (lambda r: r.update(actuator_events=[dict(started_s=100.25, motor_register_writes=1)]), 'inside supervised path'),
])
def test_partial_future_or_hold_contaminated_labels_are_rejected(tmp_path, change, match):
    row, record, _ = save_policy(tmp_path)
    change(record)
    save_record(row, record)
    with pytest.raises(ValueError, match=match):
        LocalResponseDataset(manifest(tmp_path, [row])).batch([0])


@pytest.mark.parametrize('future_history', [False, True])
def test_snapshot_binding_and_future_history_are_checked(tmp_path, future_history):
    row, _, snapshot = save_policy(tmp_path)
    if future_history:
        snapshot['state'].history_ends[0, -1] = 101.
    else:
        snapshot['commands_position'][0][2] += 1
    torch.save(snapshot, row['snapshot_path'])
    with pytest.raises(ValueError, match='history|commands differ'):
        LocalResponseDataset(manifest(tmp_path, [row])).batch([0])


def test_verified_goal_is_metadata_and_contradictory_proof_fails(tmp_path):
    row, record, _ = save_policy(tmp_path)
    add_goal_proofs(record)
    row['goal_verification']['status'] = 'verified'
    save_record(row, record)
    path = manifest(tmp_path, [row])
    batch = LocalResponseDataset(path).batch([0])
    assert batch.goal_status == ('verified',)
    assert torch.equal(batch.y[0], torch.tensor(record['measured_q_path'][1:]))
    record['policy_goal_readbacks'][0]['goal_raw'] = {'elbow_flex': 123}
    save_record(row, record)
    with pytest.raises(ValueError, match='Goal readback contradicts'):
        LocalResponseDataset(path).batch([0])


@pytest.mark.parametrize('legacy_pid', [False, True])
def test_native_first5_requires_real_h0_witnesses_and_excludes_second_half(tmp_path, legacy_pid):
    row, drive, _ = save_policy(tmp_path)
    row.pop('snapshot_path')
    row.update(source_class='native_diagnostic', runtime_session_id='session-native')
    row['history'] = dict(valid_count=0, h0_witnesses=[])
    drive.update(not_model_transition=True, model_feedback_label=False)
    if legacy_pid:
        report = dict(schema='v21-j3-pid-phase-result-v2', drive_started_s=99.99,
                      stop_returned_s=101., drive_result=drive)
    else:
        drive.update(policy_commands_attempted=10, policy_commands_returned=10)
        drive['commands_position'] += [[222.] * 6] * 5
        drive['measured_q_path'] += [[333.] * 6] * 5
        drive['command_times_s'] += [100.53+.1*i for i in range(5)]
        drive['write_completed_times_s'] += [100.531+.1*i for i in range(5)]
        drive['readback_times_s'] += [100.6+.1*i for i in range(5)]
        report = dict(started_s=99.99, finished_s=101., model_history_imported=False, drive_result=drive)
    save_record(row, report)
    for i, time in enumerate([99., 102.]):
        path = tmp_path / f'status-{i}.json'
        path.write_text(json.dumps(dict(published_monotonic_s=time, service=dict(runtime=dict(
            history_chunks=[0], readback_progress=dict(current_index=[0], session_id='session-native'),
            pending=False, executing=False)))))
        row['history']['h0_witnesses'].append(dict(path=str(path)))
    path = manifest(tmp_path, [row])
    batch = LocalResponseDataset(path).batch([0])
    assert not batch.history.valid.any()
    assert torch.equal(batch.y[0], torch.tensor(drive['measured_q_path'][1:6]))
    assert torch.equal(batch.trace.positions[0], torch.tensor(drive['commands_position'][:5]))
    row['history']['h0_witnesses'].pop()
    with pytest.raises(ValueError, match='two saved H0 witnesses'):
        LocalResponseDataset(manifest(tmp_path, [row])).batch([0])


def test_batch_passes_native_lewm_cpu_interface_without_images(tmp_path):
    row, _, _ = save_policy(tmp_path)
    batch = LocalResponseDataset(manifest(tmp_path, [row])).batch([0, 0], device='cpu')
    model = V21LeWM().eval()
    with torch.no_grad():
        output = model(batch.q_position, batch.trace, batch.history, batch.query_offsets)
    assert output.predicted_q_path.shape == (2, 6, 6)
    assert torch.isfinite(output.predicted_q_path).all()
    assert batch.query_offsets.dtype == batch.trace.offsets.dtype == torch.float64
    assert batch.y.dtype == batch.q_position.dtype == torch.float32
