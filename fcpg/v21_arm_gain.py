"""Shoulder-lift and elbow-flex position gain applied after the service's verified connect (training overhaul doc
section 17.92, James 10/24 「ok」).

``SO101Follower.configure()`` writes P_Coefficient 16 (half the Feetech factory 32, "to avoid shakiness"), I 0 and
D 0 to every motor. Under the arm's own weight the loaded shoulder lift and elbow then stop about 2 units short of
their goal (10/24 robot sessions; docs/next_build_backlog.md 2026-08-18: the gravity one-sided sag of lift/elbow), so
near the table the jaws end 5-8 mm under the commands. This module sets ONLY those two motors' P to the factory value.
Unlike fcpg/v21_gripper_force.py it never disables torque (the arm would fall): the write happens with torque on and
Lock 1, which the STS memory table defines as "EPROM values not saved after power-off", so a power cycle or LeRobot's
configure() on a cold connect returns the motors to 16. Every other register and motor is left as it is.
"""
from __future__ import annotations

import time

ARM_GAIN_JOINTS = ('shoulder_lift', 'elbow_flex')
ARM_P_COEFFICIENTS = (32,)          # the Feetech factory P; LeRobot's 16 stays the baseline everywhere else
BASELINE_P = 16


def check_arm_p_coefficient(value):
    """None (LeRobot's 16 everywhere) or one of ARM_P_COEFFICIENTS, as an int."""
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value or int(value) not in ARM_P_COEFFICIENTS:
        raise ValueError(f'arm P coefficient must be one of {ARM_P_COEFFICIENTS}, not {value!r}')
    return int(value)


def lewm_arm_setting(value, *, device=None):
    """17.153: the control-setting inputs of a LeWM that reads them (fcpg/v21_full_patch_lewm.py CONTROLLER_INPUTS)
    for a service whose arm P is value (None = LeRobot's 16): [shoulder_lift/elbow_flex at P 32, 1 = the service's
    planner sends the commands]."""
    import torch
    return torch.tensor([float(check_arm_p_coefficient(value) == 32), 1.], device=device)


def apply_arm_p_coefficient(bus, value, *, clock=time.monotonic) -> dict:
    """Set P_Coefficient of ARM_GAIN_JOINTS to value with torque on; -> record. Refuses (no write) unless both
    motors read Lock 1 and a P of 16 or value; writes only a motor not already at value; reads every write back."""
    value = check_arm_p_coefficient(value)
    if value is None:
        raise ValueError('no arm P coefficient to apply')
    record = dict(schema='v21-arm-p-coefficient-v1', section='training overhaul doc 17.92', joints=list(ARM_GAIN_JOINTS),
                  value=value, baseline=BASELINE_P, started_s=float(clock()), lock={}, before={}, after={},
                  motor_register_writes=0, torque_changed=False)
    for name in ARM_GAIN_JOINTS:
        record['lock'][name] = int(bus.read('Lock', name, normalize=False, num_retry=4))
        record['before'][name] = int(bus.read('P_Coefficient', name, normalize=False, num_retry=4))
    if any(lock != 1 for lock in record['lock'].values()):
        raise RuntimeError(f'arm P coefficient needs Lock 1 (not saved after power-off): {record["lock"]}')
    if any(p not in (BASELINE_P, value) for p in record['before'].values()):
        raise RuntimeError(f'arm P coefficient found an unexpected gain: {record["before"]}')
    for name in ARM_GAIN_JOINTS:
        if record['before'][name] != value:
            bus.write('P_Coefficient', name, value, normalize=False, num_retry=0)
            record['motor_register_writes'] += 1
        record['after'][name] = int(bus.read('P_Coefficient', name, normalize=False, num_retry=4))
    record['finished_s'] = float(clock())
    if any(p != value for p in record['after'].values()):
        raise RuntimeError(f'arm P coefficient read-back differs from {value}: {record["after"]}')
    return record


__all__ = ['ARM_GAIN_JOINTS', 'ARM_P_COEFFICIENTS', 'BASELINE_P', 'apply_arm_p_coefficient', 'check_arm_p_coefficient']
