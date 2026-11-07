"""Gripper-only register profile: fake bus, no hardware."""
from contextlib import contextmanager

import pytest

from fcpg.v21_gripper_force import GRIPPER_FORCE_PROFILES, apply_gripper_force_profile


class FakeBus:
    def __init__(self):
        self.values = {("gripper", "P_Coefficient"): 16, ("gripper", "Max_Torque_Limit"): 500,
                       ("gripper", "Protection_Current"): 250, ("gripper", "Overload_Torque"): 25,
                       ("shoulder_lift", "P_Coefficient"): 16}
        self.torque_off = []
        self.writes = []

    def read(self, name, motor):
        return self.values[(motor, name)]

    def write(self, name, motor, value):
        assert motor in self.torque_off, "EEPROM write with torque enabled"
        self.writes.append((motor, name, value)); self.values[(motor, name)] = value

    @contextmanager
    def torque_disabled(self, motors):
        self.torque_off.append(motors)
        try: yield
        finally: self.torque_off.remove(motors)


def test_lerobot_profile_writes_nothing():
    bus = FakeBus()
    result = apply_gripper_force_profile(bus, "lerobot")
    assert result["written"] is False and bus.writes == [] and result["before"] == result["after"]


def test_firm_profile_touches_only_gripper_and_reads_back():
    bus = FakeBus()
    result = apply_gripper_force_profile(bus, "firm")
    assert result["written"] and {m for m, _, _ in bus.writes} == {"gripper"}
    assert result["after"] == GRIPPER_FORCE_PROFILES["firm"]
    assert bus.values[("shoulder_lift", "P_Coefficient")] == 16
    assert bus.torque_off == []


def test_unknown_profile_and_readback_mismatch_raise():
    with pytest.raises(ValueError, match="unknown gripper force profile"):
        apply_gripper_force_profile(FakeBus(), "max")
    bus = FakeBus()
    bus.write = lambda name, motor, value: None  # silently ignores writes
    with pytest.raises(RuntimeError, match="readback differs"):
        apply_gripper_force_profile(bus, "firm")

