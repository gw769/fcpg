"""docs/V21_SYSTEM_LEWM_CEM_AUDIT_20261013.md section 14.12: the table gate's floor is lowered only explicitly (James
10/16: -8 mm, because the jaws resting on the table read -6.4 to -6.7 mm and every firm contact deadlocked the arm)."""
import numpy as np
import pytest

import scripts.serve_so101_prod_v17_7 as established
from scripts.canary_so101_v21_research import clearance_gate


class Flat:
    """Three anchors at height q[0] mm, so with table_z_mm 0 the clearance is q[0]."""
    def forward_anchors(self, state):
        return np.asarray([[0., 0., float(np.asarray(state)[0]) / 1000.]] * 3)


def gate(**kwargs):
    return established.V177ClearanceGate(kinematics=Flat(), table_z_mm=0., **kwargs)


def at(height_mm):
    return np.asarray([height_mm, 0., 0., 0., 0., 0.])


def test_the_default_floor_still_refuses_anything_below_minus_six():
    with pytest.raises(ValueError):
        gate(planned_clearance_mm=-8., readback_clearance_mm=-8.)
    default = gate()
    assert default.require_readback(at(-5.9), label='test') == pytest.approx(-5.9)
    with pytest.raises(RuntimeError):
        default.require_readback(at(-6.5), label='test')


def test_an_explicit_floor_of_minus_eight_admits_jaws_resting_on_the_table():
    lowered = gate(planned_clearance_mm=-8., readback_clearance_mm=-8., physical_floor_mm=-8.)
    assert lowered.require_readback(at(-6.5), label='test') == pytest.approx(-6.5)
    with pytest.raises(RuntimeError):
        lowered.require_readback(at(-8.5), label='test')


@pytest.mark.parametrize('floor', [-10.5, -5.])
def test_a_floor_outside_minus_ten_to_minus_six_is_refused(floor):
    with pytest.raises(ValueError):
        gate(planned_clearance_mm=max(floor, -6.), readback_clearance_mm=max(floor, -6.), physical_floor_mm=floor)


def test_the_trial_gate_uses_a_floor_only_when_given():
    default = clearance_gate(Flat())
    assert (default.planned_clearance_mm, default.readback_clearance_mm, default.physical_floor_mm) == (-6., -6., -6.)
    assert default.table_z_mm == established.TABLE_Z_MM
    lowered = clearance_gate(Flat(), -8)
    assert (lowered.planned_clearance_mm, lowered.readback_clearance_mm, lowered.physical_floor_mm) == (-8., -8., -8.)
    with pytest.raises(ValueError):
        clearance_gate(Flat(), -12)


def test_a_planning_clearance_keeps_the_floor_for_read_backs():
    """Training overhaul doc section 17.91 (James 10/24): plans stay 8 mm up, read-backs may sink to the -10 mm floor."""
    gate = clearance_gate(Flat(), -10, 8)
    assert (gate.planned_clearance_mm, gate.readback_clearance_mm, gate.physical_floor_mm) == (8., -10., -10.)

    def high(mm):                                    # the clearance of this gate (table at TABLE_Z_MM) is mm
        return at(mm + gate.table_z_mm)

    def block(mm):
        return np.repeat(high(mm)[None], 5, 0)
    assert gate.require_readback(high(-9.5), label='test') == pytest.approx(-9.5)
    with pytest.raises(RuntimeError):
        gate.require_readback(high(-10.5), label='test')
    assert gate.require_policy_path(high(9.), block(9.))['mode'] == 'planned_envelope'
    with pytest.raises(RuntimeError):
        gate.require_policy_path(high(9.), block(2.))           # a plan under 8 mm is refused
    # sunk under its commands to 1 mm: back up to 8 mm is admitted, staying low or going lower is not
    assert gate.require_policy_path(high(1.), block(9.))['mode'] == 'monotonic_recovery'
    for low in (1., 0.):
        with pytest.raises(RuntimeError):
            gate.require_policy_path(high(1.), block(low))
    with pytest.raises(ValueError):
        clearance_gate(Flat(), -10, -12)                         # a plan under the floor
    with pytest.raises(ValueError):
        clearance_gate(Flat(), None, 8)                          # only with an explicit floor
    same = clearance_gate(Flat(), -10, None)
    assert (same.planned_clearance_mm, same.readback_clearance_mm) == (-10., -10.)
