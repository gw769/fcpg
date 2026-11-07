"""Fake registers and an advancing clock only; no hardware/model constructors."""
from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.v21_same_goal_integral import (
    AXIS, IDENTITY, MAX_STEP, MOTOR, ORDER, SameGoalIntegralError,
    _ramp, run_same_goal_integral,
)


def cells(q):
    q = np.asarray(q, dtype=np.float32)
    return {name: int(round(10000+100*float(q[i]))) for i, name in enumerate(ORDER)}


def source_execution():
    target = np.array([.1, .1, 0., .1, .1, 30.1], dtype=np.float32)
    start = np.array([0., 0., 5., 0., 0., 30.])
    tail = np.array([0., 0., 2., 0., 0., 30.])
    commands = _ramp(start, target)
    measured = np.repeat(tail[None], 6, axis=0)
    measured[0] = start
    times = 10.+.1*np.arange(6)
    receipts = [dict(command_index=i, goal_register_verified=True, model_feedback_label=False,
        expected_goal_raw=cells(q), goal_raw=cells(q),
        requested_goal_position=dict(zip(ORDER, q.tolist()))) for i, q in enumerate(commands)]
    return dict(completed=True, policy_commands_attempted=5, policy_commands_returned=5,
        policy_goal_reads_attempted=5, policy_goal_reads_returned=5, policy_goal_registers_verified=5,
        commands_position=commands.tolist(), measured_q_path=measured.tolist(), readback_times_s=times.tolist(),
        original_target_arrival=dict(original_target=target.tolist()), policy_goal_readbacks=receipts,
        normal_follow=dict(budget_s=2., fixed_observation_window=True, reason='fixed_window_complete',
            positions=np.repeat(tail[None], 40, axis=0).tolist(), times_s=(times[-1]+.05*np.arange(1, 41)).tolist()))


class Clock:
    def __init__(self):
        self.now = 20.
        self.on_wait = None

    def __call__(self):
        return self.now

    def wait(self, seconds):
        assert 0 < seconds <= .01000001
        self.now += seconds
        if self.on_wait:
            self.on_wait()


class Bus:
    def __init__(self, owner):
        self.owner = owner
        self.motors = {name: SimpleNamespace(id=i+1, model='sts3215') for i, name in enumerate(ORDER)}
        self.model_ctrl_table = {'sts3215': {'I_Coefficient': (23, 1), 'D_Coefficient': (22, 1)}}
        self.calibration = {name: SimpleNamespace(range_min=0, range_max=20000) for name in ORDER}
        self.values = {name: dict(IDENTITY, ID=i+1, P_Coefficient=16, I_Coefficient=0,
            D_Coefficient=0, Torque_Limit=1000, Max_Torque_Limit=1000,
            Present_Position=0, Goal_Position=0, Present_Load=112, Moving=0) for i, name in enumerate(ORDER)}
        self.set_q([0., 0., 2., 0., 0., 30.])
        self.set_goal(source_execution()['commands_position'][-1])
        self.events = []
        self.read_retries = []
        self.on_read = None
        self.on_write = None
        self.motion_step = 50
        self.candidate_path = None
        self.quiet_read_count = 0
        self.after_halt = False
        self.quiet_corruption = None

    def set_q(self, q):
        for name, raw in cells(q).items():
            self.values[name]['Present_Position'] = raw

    def set_goal(self, q):
        for name, raw in cells(q).items():
            self.values[name]['Goal_Position'] = raw

    def q(self):
        return np.array([(self.values[name]['Present_Position']-10000)/100. for name in ORDER])

    def record(self, kind, register, motor, value=None):
        assert self.owner.bus_lock._is_owned()
        self.events.append(dict(kind=kind, register=register, motor=motor, value=value,
            time=self.owner.clock.now, stopped=self.owner.stop_event.is_set()))
        self.owner.clock.now += .0002

    def _normalize(self, values):
        assert self.owner.bus_lock._is_owned()
        return {key: (value-10000)/100. for key, value in values.items()}

    def _unnormalize(self, values):
        assert self.owner.bus_lock._is_owned()
        return {key: int(round(10000+100*value)) for key, value in values.items()}

    def read(self, register, motor, *, normalize, num_retry):
        assert normalize is False and num_retry in (0, 1)
        self.read_retries.append(('read', register, motor, num_retry))
        self.record('read', register, motor)
        if self.on_read:
            self.on_read(register, motor)
        return self.values[motor][register]

    def sync_read(self, register, *, normalize, num_retry):
        assert normalize is False and num_retry in (0, 1)
        self.read_retries.append(('sync_read', register, 'all', num_retry))
        self.record('read', register, 'all')
        if register == 'Present_Position':
            if (self.values[MOTOR]['P_Coefficient'] == 32 or self.values[MOTOR]['I_Coefficient']) and not self.after_halt:
                current, target = (self.values[MOTOR][key] for key in ('Present_Position', 'Goal_Position'))
                if self.candidate_path:
                    self.values[MOTOR]['Present_Position'] = int(round(10000+100*self.candidate_path.pop(0)))
                else:
                    self.values[MOTOR]['Present_Position'] += int(np.clip(target-current, -self.motion_step, self.motion_step))
            if self.after_halt:
                self.quiet_read_count += 1
                if self.quiet_corruption:
                    self.quiet_corruption(self.quiet_read_count)
        if self.on_read:
            self.on_read(register, 'all')
        return {name: self.values[name][register] for name in ORDER}

    def write(self, register, motor, value, *, normalize, num_retry):
        assert normalize is False and num_retry == 0
        assert register in ('P_Coefficient', 'I_Coefficient', 'D_Coefficient', 'Torque_Limit') and motor == MOTOR
        self.record('write', register, motor, value)
        self.values[motor][register] = value
        if self.on_write:
            self.on_write(register, value)


class Owner:
    def __init__(self):
        self.clock = Clock()
        self.bus_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.bus = Bus(self)
        self.io = SimpleNamespace(follower=SimpleNamespace(bus=self.bus))
        self.runtime = SimpleNamespace(stopped=True, pending=None, executing=False, history=['untouched'])
        self.bridge = SimpleNamespace(runtime=self.runtime)
        self.clearance = SimpleNamespace(require_readback=self.require_readback)
        self.geometry = []
        self.paths = []
        self.on_geometry = None
        self.on_write_q = None
        self.on_lease = None
        self.halts = 0
        self.quiet_load = 24
        self._v21_hold_cache = {'old': 'must be invalidated'}

    def _lease_check(self):
        if self.on_lease:
            self.on_lease()

    def require_readback(self, q, *, label):
        assert np.asarray(q).shape == (6,) and np.isfinite(q).all()
        self.geometry.append((np.array(q), label))

    def _physical_check(self, q, commands):
        assert np.asarray(commands).shape[1] == 6
        self.paths.append((np.array(q), np.array(commands)))
        if self.on_geometry:
            self.on_geometry()

    def _write_q(self, command, previous):
        with self.bus_lock:
            assert not self.stop_event.is_set()
            assert np.all(np.abs(command-previous) <= MAX_STEP+1e-5)
            self.bus.record('position', 'Goal_Position', 'all', command.tolist())
            self._v21_hold_cache = None
            self.bus.set_goal(command)
            if self.on_write_q:
                self.on_write_q()
            return command.copy()

    def _halt(self):
        assert self.stop_event.is_set() and self._v21_hold_cache is None
        with self.bus_lock:
            raw = self.bus.sync_read('Present_Position', normalize=False, num_retry=0)
            self.bus.record('halt', 'Goal_Position', 'all', raw.copy())
            for name in ORDER:
                self.bus.values[name]['Goal_Position'] = raw[name]
                self.bus.values[name]['Moving'] = 0
            self.bus.values[MOTOR]['Present_Load'] = self.quiet_load
            goal = self.bus.sync_read('Goal_Position', normalize=False, num_retry=0)
            self.halts += 1
            self.bus.after_halt = True
            return dict(attempts=[dict(goal_register_verified=True, pose_validated=True,
                requested_goal_raw=raw, goal_readback_raw=goal, held_state=self.bus.q())])


def run(owner=None, source=None, *, mode='p32', integral_max_seconds=2):
    owner = Owner() if owner is None else owner
    return run_same_goal_integral(owner, source_execution() if source is None else source,
        enabled=True, mode=mode, integral_max_seconds=integral_max_seconds,
        clock=owner.clock, wait=owner.clock.wait)


def writes(owner):
    return [(event['register'], event['value']) for event in owner.bus.events if event['kind'] == 'write']


def rejected(owner, match=None, source=None, *, mode='p32', integral_max_seconds=2):
    with pytest.raises(SameGoalIntegralError, match=match) as caught:
        run(owner, source, mode=mode, integral_max_seconds=integral_max_seconds)
    report = caught.value.report
    json.dumps(report, allow_nan=False)
    return report


@pytest.mark.parametrize('options', [{}, {'enabled': 1}, {'mode': 'i1'}, {'enabled': True, 'mode': 'i2'}, {'enabled': True, 'mode': 'p64'}])
def test_disabled_or_unsupported_mode_rejects_before_any_owner_or_clock_access(options):
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('disabled diagnostic touched owner '+name)
    def forbidden():
        pytest.fail('disabled diagnostic touched clock')
    with pytest.raises(SameGoalIntegralError, match='no IO') as caught:
        run_same_goal_integral(Forbidden(), clock=forbidden, **options)
    assert caught.value.report['parameter_writes'] == []


@pytest.mark.parametrize('damage', ['completed', 'count', 'goal', 'command', 'original', 'nonstationary', 'cadence'])
def test_source_proof_rejection_performs_no_io_or_restoration(damage):
    owner = Owner(); source = source_execution()
    if damage == 'completed':
        source['completed'] = False
    elif damage == 'count':
        source['policy_goal_registers_verified'] = 4
    elif damage == 'goal':
        source['policy_goal_readbacks'][2]['goal_raw'][MOTOR] += 1
    elif damage == 'command':
        source['commands_position'][0][0] += .1
    elif damage == 'original':
        source['original_target_arrival']['original_target'][AXIS] += .01
    elif damage == 'nonstationary':
        for i, row in enumerate(source['normal_follow']['positions']):
            row[0] += .2*i/39
    elif damage == 'cadence':
        source['normal_follow']['times_s'][2] -= .04
    result = rejected(owner, source=source)
    assert not owner.bus.events
    assert result['cleanup']['restored'] is False and result['cleanup']['no_mutation_attempted']


@pytest.mark.parametrize('field,value', [('ID', 4), ('Model_Number', 778), ('Firmware_Minor_Version', 11),
    ('Lock', 0), ('Operating_Mode', 1), ('Torque_Enable', 0), ('P_Coefficient', 32),
    ('I_Coefficient', 1), ('D_Coefficient', 32), ('Torque_Limit', 200), ('Max_Torque_Limit', 900)])
def test_invalid_admission_configuration_causes_no_writes(field, value):
    owner = Owner(); owner.bus.values[MOTOR][field] = value
    result = rejected(owner)
    assert not writes(owner) and not owner.halts
    assert result['cleanup']['restored'] is False and result['cleanup']['no_mutation_attempted']


@pytest.mark.parametrize('damage', ['current', 'moving', 'runtime', 'foreign_runtime', 'geometry', 'changed_during_geometry'])
def test_source_runtime_and_geometry_admission_fail_without_writes(damage):
    owner = Owner()
    if damage == 'current':
        owner.bus.values[ORDER[0]]['Present_Position'] += 26
    elif damage == 'moving':
        owner.bus.values[MOTOR]['Moving'] = 1
    elif damage == 'runtime':
        owner.runtime.pending = object()
    elif damage == 'foreign_runtime':
        owner.bridge.runtime = object()
    elif damage == 'geometry':
        owner.on_geometry = lambda: (_ for _ in ()).throw(RuntimeError('fake clearance failure'))
    else:
        def change():
            if len(owner.paths) == 10:
                owner.bus.values[MOTOR]['Present_Position'] += 11
        owner.on_geometry = change
    result = rejected(owner)
    assert not writes(owner) and not owner.halts
    assert not [event for event in owner.bus.events if event['kind'] == 'position']
    assert result['cleanup']['restored'] is False


@pytest.mark.parametrize('moves,reached', [(True, True), (False, False)])
def test_exact_target_and_three_full_windows_have_actual_pre_cleanup_arrival(moves, reached):
    owner = Owner(); source = source_execution(); frozen = deepcopy(source)
    if not moves:
        owner.bus.motion_step = 0
    result = run(owner, source)
    json.dumps(result, allow_nan=False)
    assert source == frozen and owner.runtime.history == ['untouched']
    assert owner.runtime.stopped and owner.runtime.pending is None and not owner.runtime.executing
    assert result['completed'] and result['original_target_reached'] is reached
    assert result['original_target_arrival']['original_target'] == source['commands_position'][-1]
    assert result['original_target_arrival']['excludes_cleanup_target'] is True
    assert result['model_inference_calls'] == 0 and not result['model_history_committed']
    assert owner.stop_event.is_set() and result['cleanup']['restored']
    assert writes(owner) == [('Torque_Limit', 200), ('P_Coefficient', 32),
        ('Torque_Limit', 200), ('P_Coefficient', 16), ('Torque_Limit', 1000)]
    ramp = result['stages']['ramp']
    assert len(ramp['commands']) == 5 and len(ramp['readbacks']) == 10
    previous = np.asarray(result['admission']['pre_dispatch_q'])
    for command in ramp['commands']:
        actual = np.asarray(command['command'])
        assert np.all(np.abs(actual-previous) <= MAX_STEP+1e-5)
        assert command['goal_raw'] == cells(actual) and command['goal_register_verified']
        previous = actual
    assert ramp['commands'][-1]['command'] == source['commands_position'][-1]
    for name, p, cap in [('baseline', 16, 1000), ('cap_baseline', 16, 200), ('candidate', 32, 200)]:
        stage = result['stages'][name]
        assert stage['completed'] and len(stage['readbacks']) == 40
        assert 2. <= stage['readbacks'][-1]['readback_time_s']-stage['epoch_s'] <= 2.03
        for row in stage['readbacks']:
            assert row['goal_raw'] == source['policy_goal_readbacks'][-1]['goal_raw']
            assert row['parameters'] == dict(P_Coefficient=p, I_Coefficient=0, D_Coefficient=0, Torque_Limit=cap)
        assert stage['original_target_arrival']['position_tolerance'] == .2
        assert stage['original_target_arrival']['step_tolerance'] == .1
        assert stage['original_target_arrival']['quiet_samples'] == 3
    gain = next(event for event in result['parameter_writes'] if event['value'] == 32)
    assert result['stages']['candidate']['epoch_s'] <= gain['started_s'] <= gain['write_returned_s']
    assert gain['started_s'] >= result['stages']['cap_baseline']['readbacks'][-1]['returned_s']
    assert result['admission']['corridor_j3'] == [-.75, 2.75]
    assert result['admission']['single_axis_stall_checked']
    coordinates = np.unique([q[AXIS] for q, _ in owner.paths])
    corridor = coordinates[(coordinates >= -.75) & (coordinates <= 2.75)]
    assert np.diff(corridor).max() <= .100001
    assert len(result['cleanup']['pre_restore_readbacks']) == len(result['cleanup']['post_restore_readbacks']) == 3
    assert all(row['parameters']['Torque_Limit'] == 200 for row in result['cleanup']['pre_restore_readbacks'])
    assert all(row['parameters']['Torque_Limit'] == 1000 for row in result['cleanup']['post_restore_readbacks'])
    if not reached:
        assert result['cleanup']['hold']['attempts'][-1]['goal_readback_raw'] != result['original_target_raw']
        assert result['original_target_arrival']['max_abs_residual'] == 2.


@pytest.mark.parametrize('register,value', [('Torque_Limit', 200), ('P_Coefficient', 32)])
def test_lost_parameter_ack_records_may_have_applied_and_restores_after_stop(register, value):
    owner = Owner(); failed = []
    def lose_ack(name, written):
        if (name, written) == (register, value) and not failed:
            failed.append(True)
            raise ConnectionError('fake write ACK lost after applying value')
    owner.bus.on_write = lose_ack
    result = rejected(owner, 'ACK lost')
    failed_event = next(event for event in result['parameter_writes'] if event['register'] == register and event['value'] == value)
    assert failed_event['may_have_applied'] and not failed_event['verified']
    assert result['cleanup']['restored'] and owner.bus.values[MOTOR]['P_Coefficient'] == 16
    assert owner.bus.values[MOTOR]['Torque_Limit'] == 1000 and owner.stop_event.is_set()
    recovery = [event for event in owner.bus.events if event['kind'] == 'write' and event['stopped']]
    assert [(event['register'], event['value']) for event in recovery] == [
        ('Torque_Limit', 200), ('P_Coefficient', 16), ('Torque_Limit', 1000)]


@pytest.mark.parametrize('damage', ['stop', 'late_sample', 'late_gain', 'read_outage', 'goal_changed'])
def test_candidate_failure_aborts_and_runs_cleanup_without_model_or_retry(damage):
    owner = Owner(); injected = []
    def inject_read(register, motor):
        if register != 'Present_Position' or owner.bus.values[MOTOR]['P_Coefficient'] != 32 or injected:
            return
        injected.append(True)
        if damage == 'stop':
            owner.stop_event.set()
        elif damage == 'late_sample':
            owner.clock.now += .031
        elif damage == 'read_outage':
            raise ConnectionError('fake read outage')
        elif damage == 'goal_changed':
            owner.bus.values[MOTOR]['Goal_Position'] += 1
    if damage == 'late_gain':
        def inject_write(register, value):
            if (register, value) == ('P_Coefficient', 32):
                owner.clock.now += 2.1
        owner.bus.on_write = inject_write
    else:
        owner.bus.on_read = inject_read
    result = rejected(owner)
    assert not result['completed'] and result['cleanup']['restored']
    assert writes(owner).count(('P_Coefficient', 32)) == 1
    assert owner.stop_event.is_set() and owner.runtime.history == ['untouched']
    assert len(result['stages']['candidate']['readbacks']) < 40


@pytest.mark.parametrize('damage', ['ack', 'goal_mismatch'])
def test_failed_first_position_write_still_lowers_cap_before_freezing(damage):
    owner = Owner()
    def fail():
        if damage == 'ack':
            raise ConnectionError('position ACK lost')
        owner.bus.values[MOTOR]['Goal_Position'] += 1
    owner.on_write_q = fail
    result = rejected(owner)
    assert result['cleanup']['restored']
    assert writes(owner) == [('Torque_Limit', 200), ('P_Coefficient', 16), ('Torque_Limit', 1000)]
    assert len([event for event in owner.bus.events if event['kind'] == 'position']) == 1
    assert not result['stages']['ramp']['commands'][0].get('goal_register_verified', False)


@pytest.mark.parametrize('phase', ['baseline', 'cap_baseline'])
def test_still_moving_at_fixed_baseline_end_blocks_gain_change(phase):
    owner = Owner()
    def moving(register, motor):
        if register == 'Moving' and len([event for event in owner.bus.events if event['kind'] == 'position']) == 5:
            if not owner.stop_event.is_set() and owner.bus.values[MOTOR]['Torque_Limit'] == (1000 if phase == 'baseline' else 200):
                owner.bus.values[MOTOR]['Moving'] = 1
    owner.bus.on_read = moving
    result = rejected(owner, 'stationary baseline')
    assert ('P_Coefficient', 32) not in writes(owner)
    assert result['stages'][phase]['completed'] and result['cleanup']['restored']


@pytest.mark.parametrize('field,value', [('I_Coefficient', 1), ('D_Coefficient', 1), ('P_Coefficient', 33), ('Torque_Limit', 201)])
def test_live_parameter_drift_is_detected_and_unknown_i_d_forbid_torque_increase(field, value):
    owner = Owner(); injected = []
    def drift(register, motor):
        if register == 'Present_Position' and owner.bus.values[MOTOR]['P_Coefficient'] == 32 and not injected:
            injected.append(True)
            owner.bus.values[MOTOR][field] = value
    owner.bus.on_read = drift
    result = rejected(owner, 'control parameters changed')
    if field in ('I_Coefficient', 'D_Coefficient'):
        assert result['cleanup']['restored'] is False and result['cleanup']['low_cap_verified']
        assert ('Torque_Limit', 1000) not in writes(owner)
        assert owner.bus.values[MOTOR]['Torque_Limit'] == 200
    else:
        assert result['cleanup']['restored']
    assert all(register in ('P_Coefficient', 'Torque_Limit') for register, value in writes(owner))


@pytest.mark.parametrize('damage,match', [('step', 'per-sample displacement'), ('non_j3', 'non-J3 actual drift'), ('overshoot', 'overshot target')])
def test_actual_sample_envelopes_abort_with_low_cap_retained(damage, match):
    owner = Owner(); injected = []
    if damage == 'overshoot':
        owner.bus.candidate_path = [1.5, 1., .5, 0., -.26]
        owner.bus.motion_step = 0
    else:
        def move(register, motor):
            if register == 'Present_Position' and owner.bus.values[MOTOR]['P_Coefficient'] == 32 and not injected:
                injected.append(True)
                if damage == 'step':
                    owner.bus.values[MOTOR]['Present_Position'] = 10000+120
                else:
                    owner.bus.values[ORDER[0]]['Present_Position'] += 26
        owner.bus.on_read = move
    result = rejected(owner, match)
    assert not result['cleanup']['restored']
    assert owner.bus.values[MOTOR]['Torque_Limit'] == 200 and ('Torque_Limit', 1000) not in writes(owner)


@pytest.mark.parametrize('damage', ['ack', 'readback'])
def test_p16_restoration_uncertainty_never_restores_high_torque(damage):
    owner = Owner()
    def fail(register, value):
        if (register, value) == ('P_Coefficient', 16):
            if damage == 'ack':
                raise ConnectionError('P16 acknowledgement lost')
            owner.bus.values[MOTOR]['P_Coefficient'] = 32
    owner.bus.on_write = fail
    result = rejected(owner, 'restoration incomplete')
    assert result['completed'] and result['cleanup']['restored'] is False
    assert ('Torque_Limit', 1000) not in writes(owner) and result['cleanup']['low_cap_verified']
    assert result['cleanup']['hold_verified']


def test_lost_cleanup_cap_ack_still_attempts_p16_but_keeps_low_cap_and_failed_state():
    owner = Owner(); failures = []
    def fail(register, value):
        if (register, value) == ('Torque_Limit', 200) and owner.stop_event.is_set() and not failures:
            failures.append(True)
            raise ConnectionError('cleanup cap ACK lost')
    owner.bus.on_write = fail
    result = rejected(owner, 'restoration incomplete')
    assert result['cleanup']['restored'] is False and result['cleanup']['gain_restored']
    assert owner.bus.values[MOTOR]['P_Coefficient'] == 16
    assert owner.bus.values[MOTOR]['Torque_Limit'] == 200 and ('Torque_Limit', 1000) not in writes(owner)


@pytest.mark.parametrize('damage', ['moving', 'load', 'negative_load', 'invalid_load', 'goal', 'step', 'nonfinite'])
def test_missing_pre_restore_quiet_proof_never_raises_torque(damage):
    owner = Owner()
    if damage in ('load', 'negative_load', 'invalid_load'):
        owner.quiet_load = {'load': 33, 'negative_load': 1057, 'invalid_load': 2048}[damage]
    else:
        def corrupt(count):
            if damage == 'moving':
                owner.bus.values[MOTOR]['Moving'] = 1
            elif damage == 'goal':
                owner.bus.values[MOTOR]['Goal_Position'] += 1
            elif damage == 'step' and count == 2:
                owner.bus.values[MOTOR]['Present_Position'] += 11
            elif damage == 'nonfinite':
                owner.bus.values[MOTOR]['Present_Position'] = float('nan')
        owner.bus.quiet_corruption = corrupt
    result = rejected(owner, 'restoration incomplete')
    assert not result['cleanup']['restored'] and result['cleanup']['low_cap_verified']
    assert ('Torque_Limit', 1000) not in writes(owner)


@pytest.mark.parametrize('raw,decoded', [(32, 32), (1056, -32), (1024, 0)])
def test_quiet_load_uses_bit10_signed_magnitude(raw, decoded):
    owner = Owner(); owner.quiet_load = raw
    result = run(owner)
    assert result['cleanup']['restored']
    assert all(row['present_load_decoded'] == decoded for row in result['cleanup']['pre_restore_readbacks'])


def test_bad_post_restore_quiet_proof_reduces_cap_again_and_remains_failed():
    owner = Owner()
    def corrupt(count):
        if owner.bus.values[MOTOR]['Torque_Limit'] == 1000:
            owner.bus.values[MOTOR]['Present_Load'] = 33
    owner.bus.quiet_corruption = corrupt
    result = rejected(owner, 'restoration incomplete')
    assert not result['cleanup']['restored'] and result['cleanup']['pre_restore_quiet_verified']
    assert result['cleanup']['low_cap_verified'] and not result['cleanup']['torque_restored']
    assert owner.bus.values[MOTOR]['Torque_Limit'] == 200
    assert writes(owner)[-3:] == [('Torque_Limit', 1000), ('Torque_Limit', 200), ('P_Coefficient', 16)]


def test_p16_is_rechecked_inside_the_torque_increase_guard():
    owner = Owner()
    def corrupt(register, motor):
        if (register == 'P_Coefficient' and owner.bus.after_halt and owner.bus.quiet_read_count == 3
                and owner.bus.values[MOTOR]['Torque_Limit'] == 200):
            owner.bus.values[MOTOR]['P_Coefficient'] = 32
    owner.bus.on_read = corrupt
    result = rejected(owner, 'restoration incomplete')
    assert not result['cleanup']['restored'] and ('Torque_Limit', 1000) not in writes(owner)


def i1_fixture():
    # Make a P32 result with the real helper over fake actual registers. This
    # remains a P32 diagnostic; it is never disguised as a policy execution.
    previous = Owner()
    previous.bus.candidate_path = [1.5, 1.]
    previous.bus.motion_step = 0
    diagnostic = run(previous)
    source = dict(schema='v21-same-goal-service-result-v1', mode='p32', state='completed',
        completed=True, stopped=True, cleanup=deepcopy(diagnostic['cleanup']),
        original_target_arrival=deepcopy(diagnostic['original_target_arrival']), diagnostic_result=diagnostic)
    owner = Owner()
    tail = diagnostic['stages']['candidate']['readbacks'][-1]['q_position']
    owner.bus.set_q(tail)
    owner.bus.set_goal(tail)  # P32 cleanup's actual HOLD, not the original target.
    return owner, source


@pytest.mark.parametrize('damage', ['schema', 'mode', 'outer_restore', 'inner_restore', 'not_completed',
    'missing_samples', 'goal', 'original_target', 'timing', 'parameters', 'moving', 'unstable_tail'])
def test_i1_requires_unmodified_actual_p32_result_before_any_bus_io(damage):
    owner, source = i1_fixture()
    diagnostic = source['diagnostic_result']
    candidate = diagnostic['stages']['candidate']
    if damage == 'schema':
        source['schema'] = 'invented_execution'
    elif damage == 'mode':
        source['mode'] = 'i1'
    elif damage == 'outer_restore':
        source['cleanup']['restored'] = False
    elif damage == 'inner_restore':
        diagnostic['cleanup']['restored'] = False
    elif damage == 'not_completed':
        candidate['completed'] = False
    elif damage == 'missing_samples':
        candidate['readbacks'].pop()
    elif damage == 'goal':
        diagnostic['stages']['cap_baseline']['readbacks'][20]['goal_raw'][MOTOR] += 1
    elif damage == 'original_target':
        candidate['original_target'][AXIS] += .01
    elif damage == 'timing':
        candidate['readbacks'][-1]['readback_time_s'] += .04
        candidate['readbacks'][-1]['returned_s'] += .04
    elif damage == 'parameters':
        candidate['readbacks'][0]['parameters']['D_Coefficient'] = 32
    elif damage == 'moving':
        candidate['readbacks'][-1]['moving'][MOTOR] = 1
    else:
        candidate['readbacks'][-1]['q_position'][AXIS] += .11
    result = rejected(owner, source=source, mode='i1')
    assert not owner.bus.events and result['cleanup']['no_mutation_attempted']


@pytest.mark.parametrize('register,layout', [('I_Coefficient', (24, 1)), ('I_Coefficient', (23, 2)), ('D_Coefficient', (25, 1))])
def test_i1_rejects_wrong_register_address_or_width_before_writes(register, layout):
    owner, source = i1_fixture()
    owner.bus.model_ctrl_table['sts3215'][register] = layout
    result = rejected(owner, 'one byte', source, mode='i1')
    assert not writes(owner) and result['cleanup']['no_mutation_attempted']


def test_damped_i1_has_three_full_baselines_immediate_candidate_and_independent_i0_first():
    owner, source = i1_fixture(); frozen = deepcopy(source)
    original = source['diagnostic_result']['original_target']
    raw = source['diagnostic_result']['original_target_raw']
    result = run(owner, source, mode='i1')
    json.dumps(result, allow_nan=False)
    assert source == frozen and owner.runtime.history == ['untouched']
    assert result['completed'] and result['original_target_reached'] and result['cleanup']['restored']
    assert result['source_kind'] == 'completed_restored_p32_service_result'
    assert result['source_tail_phase'] == 'candidate'
    assert result['original_target'] == original and result['original_target_raw'] == raw
    assert result['source_tail_q'][AXIS] == 1.
    assert result['admission']['pre_dispatch_q'][AXIS] == 1.
    assert result['original_target'][AXIS] == 0.
    assert writes(owner) == [('Torque_Limit', 200), ('D_Coefficient', 32), ('I_Coefficient', 1),
        ('I_Coefficient', 0), ('Torque_Limit', 200), ('D_Coefficient', 0), ('Torque_Limit', 1000)]
    for event in result['parameter_writes']:
        if event['register'] in ('I_Coefficient', 'D_Coefficient'):
            assert event['address'] == (23 if event['register'] == 'I_Coefficient' else 22)
            assert event['bytes'] == 1 and event['address'] not in (24, 25)
    for name, p, i, d, cap in [('baseline', 16, 0, 0, 1000), ('cap_baseline', 16, 0, 0, 200),
            ('damping_baseline', 16, 0, 32, 200), ('candidate', 16, 1, 32, 200)]:
        stage = result['stages'][name]
        assert len(stage['readbacks']) == 40 and stage['completed']
        for row in stage['readbacks']:
            assert row['goal_raw'] == raw
            assert row['parameters'] == dict(P_Coefficient=p, I_Coefficient=i, D_Coefficient=d, Torque_Limit=cap)
        if name != 'candidate':
            assert stage['fixed_observation_window']
            assert 2. <= stage['last_sample_since_epoch_s'] <= 2.03
    candidate = result['stages']['candidate']
    assert candidate['immediate_first_sample'] and not candidate['fixed_observation_window']
    assert candidate['full_two_second_observation'] is False
    assert candidate['readbacks'][0]['readback_time_s']-candidate['epoch_s'] < .03
    assert candidate['scheduled_last_sample_s'] == 1.95
    assert 1.95 <= candidate['last_sample_since_epoch_s'] < 1.98
    assert candidate['actual_observation_span_s'] < 1.95
    exposure = result['integral_exposure']
    assert exposure['within_limit'] and exposure['maximum_s'] == 2.
    assert 1.95 < exposure['duration_s'] <= 2.
    i1_write = next(row for row in result['parameter_writes'] if row['register'] == 'I_Coefficient' and row['value'] == 1)
    assert exposure['started_s'] <= i1_write['started_s'] < candidate['readbacks'][0]['readback_time_s']
    i0_write = next(row for row in result['parameter_writes'] if row['register'] == 'I_Coefficient' and row['value'] == 0)
    assert exposure['i0_verified_s'] == i0_write['returned_s']
    assert all(owner.bus.values[MOTOR][key] == value for key, value in
               [('P_Coefficient', 16), ('I_Coefficient', 0), ('D_Coefficient', 0), ('Torque_Limit', 1000)])


@pytest.mark.parametrize('register,value', [('D_Coefficient', 32), ('I_Coefficient', 1)])
def test_i1_profile_parameter_ack_lost_still_clears_i_before_cap_and_d_restore(register, value):
    owner, source = i1_fixture(); injected = []
    def lose_ack(name, written):
        if (name, written) == (register, value) and not injected:
            injected.append(True)
            raise ConnectionError('profile write ACK lost')
    owner.bus.on_write = lose_ack
    result = rejected(owner, 'ACK lost', source, mode='i1')
    event = next(row for row in result['parameter_writes'] if (row['register'], row['value']) == (register, value))
    assert event['may_have_applied'] and not event['verified']
    assert result['cleanup']['restored'] and result['cleanup']['integral_zero_verified']
    recovery = [(event['register'], event['value']) for event in result['parameter_writes'] if event['recovery']]
    assert recovery == [('I_Coefficient', 0), ('Torque_Limit', 200), ('D_Coefficient', 0), ('Torque_Limit', 1000)]
    if register == 'D_Coefficient':
        assert ('I_Coefficient', 1) not in writes(owner)


@pytest.mark.parametrize('failure', ['cap_ack', 'p_changed', 'p_read', 'd_ack', 'd_layout'])
def test_i0_clearing_is_independent_of_cap_p_or_d_restoration_failure(failure):
    owner, source = i1_fixture(); injected = []
    def fail_write(register, value):
        if owner.stop_event.is_set() and ((failure == 'cap_ack' and (register, value) == ('Torque_Limit', 200))
                or (failure == 'd_ack' and (register, value) == ('D_Coefficient', 0))):
            raise ConnectionError('fake independent recovery failure')
    def fail_read(register, motor):
        if owner.bus.values[MOTOR]['I_Coefficient'] == 1 and register == 'Present_Position' and not injected:
            injected.append(True)
            if failure == 'p_changed':
                owner.bus.values[MOTOR]['P_Coefficient'] = 17
            elif failure == 'd_layout':
                owner.bus.model_ctrl_table['sts3215']['D_Coefficient'] = (24, 1)
        if failure == 'p_read' and injected and register == 'P_Coefficient':
            raise ConnectionError('P read failed independently')
    owner.bus.on_write = fail_write
    owner.bus.on_read = fail_read
    result = rejected(owner, source=source, mode='i1')
    assert result['cleanup']['restored'] is False
    assert result['cleanup']['integral_zero_verified'] and owner.bus.values[MOTOR]['I_Coefficient'] == 0
    assert ('I_Coefficient', 0) in writes(owner) and ('Torque_Limit', 1000) not in writes(owner)
    assert owner.bus.values[MOTOR]['Torque_Limit'] == 200 and result['cleanup']['hold_verified']
    if failure in ('cap_ack', 'p_changed', 'p_read', 'd_layout'):
        assert owner.bus.values[MOTOR]['D_Coefficient'] == 32
        assert ('D_Coefficient', 0) not in writes(owner)


@pytest.mark.parametrize('failure', ['ack', 'mismatch', 'read'])
def test_unverified_i0_never_removes_damping_or_increases_cap(failure):
    owner, source = i1_fixture()
    def fail_write(register, value):
        if (register, value) == ('I_Coefficient', 0):
            if failure == 'ack':
                raise ConnectionError('I0 ACK lost')
            if failure == 'mismatch':
                owner.bus.values[MOTOR]['I_Coefficient'] = 1
    def fail_read(register, motor):
        if failure == 'read' and owner.stop_event.is_set() and register == 'I_Coefficient':
            raise ConnectionError('I0 confirmation missing')
    owner.bus.on_write = fail_write
    owner.bus.on_read = fail_read
    result = rejected(owner, source=source, mode='i1')
    assert not result['cleanup']['restored'] and not result['cleanup']['integral_zero_verified']
    assert ('D_Coefficient', 0) not in writes(owner) and ('Torque_Limit', 1000) not in writes(owner)
    assert owner.bus.values[MOTOR]['D_Coefficient'] == 32
    assert owner.bus.values[MOTOR]['Torque_Limit'] == 200
    assert writes(owner).count(('I_Coefficient', 0)) == 2


@pytest.mark.parametrize('failure', ['stop', 'sample_late', 'i1_late', 'i0_late', 'drift', 'damping_moving'])
def test_i1_immediate_read_deadlines_stop_and_actual_guards(failure):
    owner, source = i1_fixture(); injected = []
    def inject_read(register, motor):
        if failure == 'damping_moving' and register == 'Moving' and owner.bus.values[MOTOR]['D_Coefficient'] == 32 and not owner.stop_event.is_set():
            owner.bus.values[MOTOR]['Moving'] = 1
        if (register != 'Present_Position' or owner.bus.values[MOTOR]['I_Coefficient'] != 1
                or failure not in ('stop', 'sample_late', 'drift') or injected):
            return
        injected.append(True)
        if failure == 'stop':
            owner.stop_event.set()
        elif failure == 'sample_late':
            owner.clock.now += .031
        elif failure == 'drift':
            owner.bus.values[ORDER[0]]['Present_Position'] += 26
    def inject_write(register, value):
        if (register, value) == ('I_Coefficient', 1) and failure == 'i1_late':
            owner.clock.now += 2.1
        if (register, value) == ('I_Coefficient', 0) and failure == 'i0_late' and not injected:
            injected.append(True)
            owner.clock.now += .06
    owner.bus.on_read = inject_read
    owner.bus.on_write = inject_write
    result = rejected(owner, source=source, mode='i1')
    assert owner.stop_event.is_set() and result['cleanup']['integral_zero_verified']
    if failure == 'damping_moving':
        assert ('I_Coefficient', 1) not in writes(owner)
    elif failure in ('i1_late', 'i0_late'):
        assert not result['integral_exposure']['within_limit']
        assert result['integral_exposure']['duration_s'] > 2.
    if failure != 'drift':
        assert result['cleanup']['restored']
    else:
        assert not result['cleanup']['restored'] and ('Torque_Limit', 1000) not in writes(owner)


def test_i1_cleanup_low_load_failure_retries_i0_but_keeps_cap200():
    owner, source = i1_fixture(); owner.quiet_load = 1057
    result = rejected(owner, 'restoration incomplete', source, mode='i1')
    assert result['cleanup']['integral_zero_verified'] and not result['cleanup']['restored']
    assert result['cleanup']['low_cap_verified'] and owner.bus.values[MOTOR]['Torque_Limit'] == 200
    assert ('Torque_Limit', 1000) not in writes(owner)


@pytest.mark.parametrize('mode,value', [('p32', 10), ('i1', 3), ('i1', True), ('i1', '10'),
    ('i1', None), ('i1', float('inf')), ('i1', float('nan'))])
def test_integral_duration_rejected_before_owner_or_clock_access(mode, value):
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('invalid duration touched owner '+name)
    def forbidden():
        pytest.fail('invalid duration touched clock')
    with pytest.raises(SameGoalIntegralError, match='integral_max_seconds.*no IO') as caught:
        run_same_goal_integral(Forbidden(), enabled=True, mode=mode,
            integral_max_seconds=value, clock=forbidden)
    json.dumps(caught.value.report, allow_nan=False)


def test_i1_ten_second_unreached_window_has_200_samples_with_unchanged_two_second_baselines():
    owner, source = i1_fixture(); owner.bus.motion_step = 0
    result = run(owner, source, mode='i1', integral_max_seconds=10.)
    candidate = result['stages']['candidate']
    assert result['integral_max_seconds'] == 10.
    assert result['completed'] and not result['original_target_reached'] and result['cleanup']['restored']
    assert candidate['sample_count'] == candidate['maximum_samples'] == 200
    assert len(candidate['readbacks']) == 200 and not candidate['early_arrival']
    assert candidate['termination_reason'] == 'sample_window_complete'
    assert candidate['scheduled_last_sample_s'] == 9.95
    assert candidate['max_integral_exposure_s'] == candidate['budget_s'] == 10.
    assert 9.95 < candidate['last_sample_since_epoch_s'] < 9.98
    assert 9.9 < candidate['actual_observation_span_s'] < 9.95
    assert not candidate['fixed_observation_window'] and not candidate['full_exposure_window_observed']
    for name in ('baseline', 'cap_baseline', 'damping_baseline'):
        stage = result['stages'][name]
        assert stage['sample_count'] == 40 and stage['budget_s'] == 2.
        assert 2. <= stage['last_sample_since_epoch_s'] < 2.03
    exposure = result['integral_exposure']
    assert 9.95 < exposure['duration_s'] < 10. and exposure['maximum_s'] == 10. and exposure['within_limit']
    assert 'post_integral_retention' not in result['stages']
    assert 'original_goal_retained_arrival' not in result
    assert writes(owner) == [('Torque_Limit', 200), ('D_Coefficient', 32), ('I_Coefficient', 1),
        ('I_Coefficient', 0), ('Torque_Limit', 200), ('D_Coefficient', 0), ('Torque_Limit', 1000)]


def test_ten_second_early_arrival_clears_i_immediately_then_measures_original_goal_before_hold():
    owner, source = i1_fixture()
    original = deepcopy(source['diagnostic_result']['original_target_raw'])
    # P16 can need nonzero output to maintain position; retention does not use
    # the later HOLD's <=32 PWM gate as a condition for remaining at original q.
    owner.bus.values[MOTOR]['Present_Load'] = 103
    result = run(owner, source, mode='i1', integral_max_seconds=10)
    candidate = result['stages']['candidate']
    assert candidate['early_arrival'] and candidate['sample_count'] == 4
    assert candidate['termination_reason'] == 'original_target_arrival'
    assert candidate['original_target_reached'] and candidate['maximum_samples'] == 200
    assert candidate['original_target_arrival']['sample_count'] == 4
    clear = next(row for row in result['parameter_writes'] if (row['register'], row['value']) == ('I_Coefficient', 0))
    assert 0 <= clear['started_s']-candidate['readbacks'][-1]['returned_s'] < .001
    assert result['integral_exposure']['duration_s'] < .2
    retained = result['stages']['post_integral_retention']
    assert retained['sample_count'] == 6 and retained['budget_s'] == .3
    assert .3 <= retained['last_sample_since_epoch_s'] < .33
    assert retained['epoch_s'] > clear['returned_s']
    for row in retained['readbacks']:
        assert row['goal_raw'] == original
        assert row['parameters'] == dict(P_Coefficient=16, I_Coefficient=0, D_Coefficient=0, Torque_Limit=200)
        assert row['present_load_decoded'] == 103
    arrival = result['original_goal_retained_arrival']
    assert arrival['retained'] and arrival['reached'] and arrival['sample_count'] == 6
    assert arrival['no_goal_write'] and result['original_goal_retained']
    first_hold = next(event for event in owner.bus.events if event['kind'] == 'halt')
    assert first_hold['time'] > retained['readbacks'][-1]['returned_s']
    assert len([event for event in owner.bus.events if event['kind'] == 'position']) == 5
    assert owner.stop_event.is_set() and result['cleanup']['restored']
    assert not result['physical_task_success_verified']


def test_three_actual_quiet_samples_are_required_even_when_i1_first_sample_is_near_target():
    owner, source = i1_fixture()
    # Move safely to q=.5 during the I0/D32 baseline, then the first I1 sample
    # can reach q=0 with a .5 step. Its first two good readings never suffice.
    shifted = []
    def baseline_shift(register, motor):
        if (register == 'Present_Position' and owner.bus.values[MOTOR]['D_Coefficient'] == 32
                and owner.bus.values[MOTOR]['I_Coefficient'] == 0 and not shifted):
            shifted.append(True)
            owner.bus.values[MOTOR]['Present_Position'] = 10050
    owner.bus.on_read = baseline_shift
    result = run(owner, source, mode='i1', integral_max_seconds=10)
    candidate = result['stages']['candidate']
    assert candidate['sample_count'] == 3 and candidate['early_arrival']
    assert all(row['q_position'][AXIS] == 0. for row in candidate['readbacks'])
    assert result['integral_exposure']['duration_s'] >= .1


def test_two_second_profile_does_not_early_exit_or_add_retention():
    owner, source = i1_fixture()
    result = run(owner, source, mode='i1', integral_max_seconds=2.)
    candidate = result['stages']['candidate']
    assert candidate['original_target_reached'] and candidate['sample_count'] == 40
    assert not candidate['early_arrival_enabled'] and not candidate['early_arrival']
    assert result['integral_exposure']['duration_s'] > 1.95
    assert 'post_integral_retention' not in result['stages']


def test_loss_of_arrival_after_i0_is_reported_without_replacing_original_goal():
    owner, source = i1_fixture(); clearing = []; samples = []
    def written(register, value):
        if (register, value) == ('I_Coefficient', 0):
            clearing.append(True)
    def drift(register, motor):
        if (register == 'Present_Position' and clearing and not owner.bus.after_halt
                and owner.bus.values[MOTOR]['D_Coefficient'] == 0 and len(samples) < 6):
            samples.append(True)
            owner.bus.values[MOTOR]['Present_Position'] = 10000+min(len(samples)*6, 30)
    owner.bus.on_write = written
    owner.bus.on_read = drift
    result = run(owner, source, mode='i1', integral_max_seconds=10)
    assert result['original_target_reached'] and not result['original_goal_retained']
    assert not result['original_goal_retained_arrival']['retained']
    assert result['original_goal_retained_arrival']['max_abs_residual'] == .3
    assert result['cleanup']['restored'] and not result['physical_task_success_verified']
    assert all(row['goal_raw'] == result['original_target_raw']
               for row in result['stages']['post_integral_retention']['readbacks'])


@pytest.mark.parametrize('failure', ['step', 'non_j3', 'overshoot', 'goal', 'late', 'geometry'])
def test_retention_keeps_actual_motion_goal_timing_and_geometry_guards(failure):
    owner, source = i1_fixture(); clearing = []; injected = []
    def written(register, value):
        if (register, value) == ('I_Coefficient', 0):
            clearing.append(True)
    def corrupt(register, motor):
        if (register == 'Present_Position' and clearing and not owner.bus.after_halt
                and owner.bus.values[MOTOR]['D_Coefficient'] == 0 and not injected):
            injected.append(True)
            if failure == 'step':
                owner.bus.values[MOTOR]['Present_Position'] += 80
            elif failure == 'non_j3':
                owner.bus.values[ORDER[0]]['Present_Position'] += 26
            elif failure == 'overshoot':
                owner.bus.values[MOTOR]['Present_Position'] -= 26
            elif failure == 'goal':
                owner.bus.values[MOTOR]['Goal_Position'] += 1
            elif failure == 'late':
                owner.clock.now += .031
    def geometry():
        if failure == 'geometry' and clearing:
            raise RuntimeError('retention physical path rejected')
    owner.bus.on_write = written
    owner.bus.on_read = corrupt
    owner.on_geometry = geometry
    result = rejected(owner, source=source, mode='i1', integral_max_seconds=10)
    assert result['original_target_reached'] and not result['cleanup']['restored']
    assert result['cleanup']['integral_zero_verified'] and result['cleanup']['low_cap_verified']
    assert ('Torque_Limit', 1000) not in writes(owner) and result['cleanup']['hold_verified']
    assert 'original_goal_retained_arrival' not in result


@pytest.mark.parametrize('failure', ['stop', 'drift', 'late', 'clear_late'])
def test_ten_second_exposure_keeps_late_guards_and_clears_i_against_configured_limit(failure):
    owner, source = i1_fixture(); owner.bus.motion_step = 0; count = []; injected = []
    def corrupt(register, motor):
        if register == 'Present_Position' and owner.bus.values[MOTOR]['I_Coefficient'] == 1:
            count.append(True)
            if len(count) == 100:
                if failure == 'stop':
                    owner.stop_event.set()
                elif failure == 'drift':
                    owner.bus.values[ORDER[0]]['Present_Position'] += 26
                elif failure == 'late':
                    owner.clock.now += .031
    def written(register, value):
        if (register, value) == ('I_Coefficient', 0) and failure == 'clear_late' and not injected:
            injected.append(True)
            owner.clock.now += .06
    owner.bus.on_read = corrupt
    owner.bus.on_write = written
    result = rejected(owner, source=source, mode='i1', integral_max_seconds=10)
    assert result['cleanup']['integral_zero_verified']
    assert 'post_integral_retention' not in result['stages']
    if failure == 'clear_late':
        assert result['integral_exposure']['duration_s'] > 10.
        assert not result['integral_exposure']['within_limit']
        assert '10 seconds' in result['error']
    else:
        assert result['integral_exposure']['duration_s'] < 10. and result['integral_exposure']['within_limit']
    if failure == 'drift':
        assert not result['cleanup']['restored'] and ('Torque_Limit', 1000) not in writes(owner)
    else:
        assert result['cleanup']['restored']


def test_returning_to_goal_at_retention_end_does_not_claim_six_sample_continuous_retention():
    owner, source = i1_fixture(); clearing = []; deviations = [.21, .21, .19, .19, .19, .19]
    def written(register, value):
        if (register, value) == ('I_Coefficient', 0):
            clearing.append(True)
    def leave_then_return(register, motor):
        if (register == 'Present_Position' and clearing and not owner.bus.after_halt
                and owner.bus.values[MOTOR]['D_Coefficient'] == 0 and deviations):
            owner.bus.values[MOTOR]['Present_Position'] = int(round(10000+100*deviations.pop(0)))
    owner.bus.on_write = written
    owner.bus.on_read = leave_then_return
    result = run(owner, source, mode='i1', integral_max_seconds=10)
    arrival = result['original_goal_retained_arrival']
    assert arrival['reached'] and arrival['max_abs_residual'] == .19
    assert not arrival['retained'] and not result['original_goal_retained']
    assert arrival['retention_samples'] == 6 and arrival['retention_max_abs_residual'] == .21
    assert arrival['retention_max_abs_step'] <= .1
    assert result['original_target_reached'] and result['cleanup']['restored']
