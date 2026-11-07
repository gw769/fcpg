"""Training overhaul doc section 17.92: shoulder lift and elbow flex at the Feetech factory P 32 after the verified
connect (fcpg/v21_arm_gain.py), and warm restarts that accept that gain only when the service asks for it. Fake
registers only; no hardware."""
from types import SimpleNamespace

import pytest

from fcpg.v21_arm_gain import ARM_GAIN_JOINTS, apply_arm_p_coefficient, check_arm_p_coefficient
from fcpg.v21_hardware_state import (ORDER, connect_preserving_warm_hold, existing_owner_connection_initializer,
                                     verify_existing_configuration)


class GainBus:
    protocol_version = 0

    def __init__(self, p=None, lock=1, ignore_writes=False):
        self.p = {name: 16 for name in ORDER}
        self.p.update(p or {})
        self.lock = {name: lock for name in ORDER}
        self.ignore_writes, self.writes = ignore_writes, []

    def sync_read(self, register, normalize=True, num_retry=0):
        if register == 'P_Coefficient':
            return dict(self.p)
        value = {'Operating_Mode': 0, 'I_Coefficient': 0, 'D_Coefficient': 0, 'Return_Delay_Time': 0,
                 'Acceleration': 254, 'Maximum_Acceleration': 254}[register]
        return {name: value for name in ORDER}

    def read(self, register, motor, normalize=True, num_retry=0):
        if register == 'P_Coefficient':
            return self.p[motor]
        if register == 'Lock':
            return self.lock[motor]
        if register == 'Torque_Limit' and motor == 'elbow_flex':
            return 1000
        assert motor == 'gripper'
        return {'Max_Torque_Limit': 500, 'Protection_Current': 250, 'Overload_Torque': 25}[register]

    def write(self, register, motor, value, normalize=True, num_retry=0):
        assert register == 'P_Coefficient' and normalize is False      # no torque, lock or other register writes
        self.writes.append((motor, value))
        if not self.ignore_writes:
            self.p[motor] = value


def test_only_the_factory_value_or_none_is_accepted():
    assert check_arm_p_coefficient(None) is None and check_arm_p_coefficient(32) == 32
    for value in (16, 24, 64, True, '32', 32.5):
        with pytest.raises(ValueError):
            check_arm_p_coefficient(value)


def test_writes_only_shoulder_lift_and_elbow_flex_and_reads_them_back():
    bus = GainBus()
    record = apply_arm_p_coefficient(bus, 32)
    assert bus.writes == [(name, 32) for name in ARM_GAIN_JOINTS] == [('shoulder_lift', 32), ('elbow_flex', 32)]
    assert record['before'] == {'shoulder_lift': 16, 'elbow_flex': 16}
    assert record['after'] == {'shoulder_lift': 32, 'elbow_flex': 32} and record['motor_register_writes'] == 2
    assert {name: bus.p[name] for name in ORDER if name not in ARM_GAIN_JOINTS} == {
        name: 16 for name in ORDER if name not in ARM_GAIN_JOINTS}
    assert record['torque_changed'] is False


def test_a_motor_already_at_the_gain_is_not_written_again():
    bus = GainBus(p={'shoulder_lift': 32, 'elbow_flex': 32})
    record = apply_arm_p_coefficient(bus, 32)
    assert bus.writes == [] and record['motor_register_writes'] == 0 and record['after']['elbow_flex'] == 32


def test_refuses_without_lock_one_or_with_an_unknown_gain_and_never_writes():
    for bus in (GainBus(lock=0), GainBus(p={'elbow_flex': 20})):
        with pytest.raises(RuntimeError):
            apply_arm_p_coefficient(bus, 32)
        assert bus.writes == []


def test_a_write_that_does_not_read_back_fails():
    bus = GainBus(ignore_writes=True)
    with pytest.raises(RuntimeError, match='read-back'):
        apply_arm_p_coefficient(bus, 32)


def test_warm_check_without_the_gain_still_demands_lerobot_16_everywhere():
    verify_existing_configuration(GainBus())
    with pytest.raises(RuntimeError, match='P_Coefficient'):
        verify_existing_configuration(GainBus(p={'shoulder_lift': 32, 'elbow_flex': 32}))


def test_warm_check_with_the_gain_accepts_it_on_the_two_arm_joints_only():
    verify_existing_configuration(GainBus(), arm_p_coefficient=32)
    verify_existing_configuration(GainBus(p={'shoulder_lift': 32, 'elbow_flex': 32}), arm_p_coefficient=32)
    verify_existing_configuration(GainBus(p={'elbow_flex': 32}), arm_p_coefficient=32)
    for name in ('shoulder_pan', 'wrist_flex', 'wrist_roll', 'gripper'):
        with pytest.raises(RuntimeError, match='P_Coefficient'):
            verify_existing_configuration(GainBus(p={name: 32}), arm_p_coefficient=32)


def test_the_owner_initializer_carries_the_gain_into_the_warm_connect(tmp_path, monkeypatch):
    owner = SimpleNamespace(output=tmp_path)                 # no incomplete execution: the ordinary connect
    assert existing_owner_connection_initializer(owner) is connect_preserving_warm_hold
    seen = []
    import fcpg.v21_hardware_state as module
    monkeypatch.setattr(module, 'connect_preserving_warm_hold', lambda io, **kwargs: seen.append(kwargs) or 'ok')
    initialize = existing_owner_connection_initializer(owner, arm_p_coefficient=32)
    assert initialize('io') == 'ok' and seen == [{'arm_p_coefficient': 32}]
    with pytest.raises(ValueError):
        existing_owner_connection_initializer(owner, arm_p_coefficient=64)
