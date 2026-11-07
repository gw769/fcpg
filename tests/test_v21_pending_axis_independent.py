"""Independent sequential-arrival regressions; fake clocks/registers only."""
from copy import deepcopy

import numpy as np
import pytest

from test_v21_arm_position_servo import (
    J2, J3, J4, change_pose, rejected, run, three_axis_fixture,
)


@pytest.mark.parametrize('drift', [0., .08, .24, .26])
def test_future_wrist_pitch_in_band_is_checked_at_its_turn(drift):
    owner, execution = three_axis_fixture()
    q = owner.bus.q()
    q[1], q[3] = -.03, .23
    change_pose(owner, execution, q)
    before = deepcopy(execution)
    fired = []

    def move_on_first_elbow_i0(register, motor, value):
        if ((register, motor, value) == ('I_Coefficient', J3, 0)
                and not fired and not owner.stop_event.is_set()):
            fired.append(True)
            owner.bus.values[J4]['Present_Position'] += round(100 * drift)

    owner.bus.on_write = move_on_first_elbow_i0
    result = rejected(owner, execution) if drift == .26 else run(owner, execution)
    assert fired and execution == before
    roles = result['preflight_classification']['final']
    assert roles['needs_finishing_axes'] == [2]
    assert roles['pending_axes'] == [1, 2, 3]
    first_corridor = result['axes'][J3]['corridor_precheck']
    assert first_corridor['other_axis_centers'][J4] == pytest.approx(.23)
    assert first_corridor['other_axis_offsets'][J4] == [-.25, 0., .25]
    if drift == .26:
        assert not result['selected_axes_completed']
        assert not result['axes'][J4]['applied']
        assert result['cleanup']['hold_performed']
        return
    assert all(row['goal_raw'] == result['original_target_raw'] for row in result['readbacks'])
    axis = result['axes'][J4]
    assert axis['completed'] and axis['applied'] is (drift != 0.)
    stage = axis['stages']['naturally_reached' if drift == 0. else 'restored_original_goal']
    assert stage['sample_count'] == 3 and stage['axis_arrival']['reached']
    assert stage['readbacks'][0]['readback_time_s'] > result['axes'][J3]['stages']['restored_original_goal']['readbacks'][-1]['readback_time_s']
    assert result['axes'][J2]['stages']['naturally_reached']['sample_count'] == 3
    hold = result['original_goal_hold']
    assert hold['completed'] and hold['sample_count'] == 22
    error = np.asarray([row['q_position'] for row in hold['readbacks']]) - result['original_target']
    assert np.abs(error).max() <= .2


def test_completed_elbow_drift_remains_an_error_while_later_axis_is_finishing():
    owner, execution = three_axis_fixture()
    before = deepcopy(execution)
    fired = []

    def move_completed_elbow(register, motor, value):
        if ((register, motor, value) == ('I_Coefficient', J4, 1)
                and not fired and not owner.stop_event.is_set()):
            fired.append(True)
            # J3 previously settled at +.06; +.15 puts it outside +.2.
            owner.bus.values[J3]['Present_Position'] += 15

    owner.bus.on_write = move_completed_elbow
    result = rejected(owner, execution, 'completed or untouched axis')
    assert fired and execution == before
    assert result['axes'][J3]['completed']
    assert not result['selected_axes_completed'] and not result['original_goal_held']
    assert result['cleanup']['hold_performed']
    assert 'completion_follow' not in result


def preflight_settling_fixture(changes):
    owner, execution = three_axis_fixture()
    q = owner.bus.q()
    q[1] = -.8
    change_pose(owner, execution, q)
    execution['measured_q_path'][1:] = np.repeat(q[None], 5, axis=0).tolist()
    reads = []

    def settle(register, motor):
        if register != 'Present_Position' or owner.stop_event.is_set():
            return
        reads.append(owner.clock())
        change = changes.get(len(reads))
        if change is not None:
            owner.bus.values[J2]['Present_Position'] += round(100 * change)

    owner.bus.on_read = settle
    return owner, execution, reads


def test_stationary_settling_after_geometry_requires_rebuilt_corridor_and_bridge_acceptance():
    from fcpg.v21_hardware_bridge import validate_arm_original_goal_hold

    owner, execution, reads = preflight_settling_fixture({2: .33})
    before = deepcopy(execution)
    result = run(owner, execution)
    receipt = result['preflight_classification']
    assert receipt['committed'] and len(receipt['rebuilds']) == 1
    assert len(result['admission_geometry_passes']) == 2
    assert len(receipt['verification_windows']) == 2
    for window in receipt['verification_windows']:
        assert window['completed'] and window['sample_count'] == 3
    assert receipt['committed_q'][1] == pytest.approx(-.47)
    assert result['initial_admission']['current_q'][1] == pytest.approx(-.8)
    assert result['admission']['current_q'] == result['deferred_anchor_q']
    anchor_row = receipt['verification_windows'][0]['readbacks'][-1]
    assert result['admission']['readback_time_s'] == anchor_row['readback_time_s']
    second = result['admission_geometry_passes'][1]
    assert second['corridor_prechecks'][J3]['other_axis_centers'][J2] == pytest.approx(-.47)
    assert all(write['started_s'] > receipt['committed_s'] for write in result['parameter_writes'])
    assert len(reads) >= 7 and result['original_goal_held'] and execution == before
    validate_arm_original_goal_hold(result, result['completion_follow'],
                                    np.asarray(execution['commands_position'][-1]))


@pytest.mark.parametrize('changes', [
    {2: .33, 5: .11},  # A second geometric anchor change must not ratchet.
    {2: .76},  # Existing cross-snapshot .75 bound still applies.
    {2: .33, 3: .08, 4: .08},  # Small individual steps are not a stable window.
])
def test_preflight_anchor_rebuild_never_admits_repeated_or_unsettled_motion(changes):
    owner, execution, _ = preflight_settling_fixture(changes)
    before = deepcopy(execution)
    result = rejected(owner, execution)
    assert not result['preflight_classification']['committed']
    assert result['parameter_writes'] == [] and result['touched_motors'] == []
    assert not result['selected_axes_completed'] and not result['original_goal_held']
    assert result['cleanup']['hold_performed'] and execution == before


def test_changed_anchor_geometry_corner_must_pass_before_the_first_parameter_write():
    owner, execution, reads = preflight_settling_fixture({2: .33})
    physical_check = owner._physical_check
    rejected_corner = []

    def check(q, commands):
        # The original J2 anchor was -.8. Only the rebuilt box around -.47
        # contains this exact -.25 cross-axis corner.
        if len(reads) >= 4 and np.isclose(q[1], -.72):
            rejected_corner.append(np.asarray(q).tolist())
            raise RuntimeError('rebuilt pending-axis corner rejected')
        return physical_check(q, commands)

    owner._physical_check = check
    result = rejected(owner, execution, 'rebuilt pending-axis corner rejected')
    assert rejected_corner and len(result['preflight_classification']['rebuilds']) == 1
    assert not result['preflight_classification']['committed']
    assert result['parameter_writes'] == [] and result['touched_motors'] == []
