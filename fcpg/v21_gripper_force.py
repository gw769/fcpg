"""Gripper-only servo force profile applied after LeRobot's connect().

``SO101Follower.configure()`` writes half-torque, half-current and 25 %
overload torque to the gripper motor to avoid burnout. With the recorded
closing depth (~10 calibrated units above contact) that leaves too little
holding force to lift the fish. This module raises ONLY the gripper motor's
registers; the five arm joints keep LeRobot's values, which the preflight
treats as the strike-hazard bound. ``lerobot`` is the unchanged default.
"""
from __future__ import annotations

GRIPPER_MOTOR = "gripper"

# STS3215 registers. Values are register LSB, not Newtons.
GRIPPER_FORCE_PROFILES = {
    "lerobot": None,  # leave SO101Follower.configure() values untouched
    "firm": {"P_Coefficient": 32, "Max_Torque_Limit": 800, "Protection_Current": 400, "Overload_Torque": 40},
    "strong": {"P_Coefficient": 32, "Max_Torque_Limit": 1000, "Protection_Current": 500, "Overload_Torque": 60},
    # Double the position gain: same stall error, twice the holding force.
    "grip": {"P_Coefficient": 64, "Max_Torque_Limit": 1000, "Protection_Current": 500, "Overload_Torque": 60},
}
_LIMITS = {"P_Coefficient": (0, 254), "Max_Torque_Limit": (0, 1000), "Protection_Current": (0, 500), "Overload_Torque": (0, 100)}


def apply_gripper_force_profile(bus, profile: str, *, motor: str = GRIPPER_MOTOR) -> dict:
    """Write the profile to the gripper motor and read every register back."""
    if profile not in GRIPPER_FORCE_PROFILES:
        raise ValueError(f"unknown gripper force profile {profile!r}; choose from {sorted(GRIPPER_FORCE_PROFILES)}")
    values = GRIPPER_FORCE_PROFILES[profile]
    registers = list(_LIMITS)
    before = {name: int(bus.read(name, motor)) for name in registers}
    if values is None:
        return dict(profile=profile, motor=motor, written=False, before=before, after=before)
    for name, value in values.items():
        low, high = _LIMITS[name]
        if not low <= int(value) <= high:
            raise ValueError(f"{name}={value} outside register range [{low},{high}]")
    # EEPROM-backed registers need torque off on this motor only; the arm holds.
    with bus.torque_disabled(motor):
        for name, value in values.items():
            bus.write(name, motor, int(value))
    after = {name: int(bus.read(name, motor)) for name in registers}
    mismatch = {name: (values[name], after[name]) for name in values if after[name] != int(values[name])}
    if mismatch:
        raise RuntimeError(f"gripper register readback differs from written values: {mismatch}")
    return dict(profile=profile, motor=motor, written=True, before=before, after=after)


__all__ = ["GRIPPER_FORCE_PROFILES", "apply_gripper_force_profile"]

