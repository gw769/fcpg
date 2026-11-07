"""Offline tests only: no camera, serial, torque, service, or real-robot commands."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.v21_vertical_approach import VerticalApproachSolver, approach_geometry


MODEL = Path(__file__).resolve().parents[1] / 'third_party/so_arm100/Simulation/SO101/so101_new_calib.xml'
CURRENT = np.array([-20.01445, 4.89914, 28.66016, 48.15436, -71.58442, 5.19054])
LOWER = np.array([-100.] * 5 + [0.])
UPPER = np.array([100.] * 6)


@pytest.fixture(scope='module')
def kinematics():
    pytest.importorskip('mujoco')
    if not MODEL.is_file():
        pytest.skip('official SO-101 model unavailable')
    from fcpg.so101_kinematics import SO101MuJoCoKinematics
    return SO101MuJoCoKinematics(MODEL)


def real_gate(kinematics):
    return SimpleNamespace(readback_clearance_mm=-3., planned_clearance_mm=-3.,
        clearance_mm=lambda q: float(1000 * kinematics.forward_anchors(q)[:, 2].min() + 17.118124))


def test_nonlinear_view_chunk_reaches_vertical_without_shrinking_away_height_allowance(kinematics):
    q=np.array([0.,-56.525318,65.927238,68.288589,-51.584415,31.143232])
    solver=VerticalApproachSolver(kinematics,real_gate(kinematics),LOWER,UPPER)
    result=solver.propose_view_chunk(q)
    assert result.accepted,result.diagnostic
    commands=result.commands_position
    assert commands.shape==(5,6)
    np.testing.assert_array_equal(commands[:,[0,4,5]],np.repeat(q[None,[0,4,5]],5,0))
    assert np.abs(np.diff(np.vstack((q,commands)),axis=0)).max()<=6.+1e-6
    assert result.diagnostic['endpoint']['tilt_deg']<5.
    previous=q;start=approach_geometry(kinematics,q)
    for index,command in enumerate(commands):
        g=approach_geometry(kinematics,command)
        assert g['radial_mm']==pytest.approx(start['radial_mm'],abs=.02)
        assert g['centre_z_mm']==pytest.approx(start['centre_z_mm']+index+1,abs=.02)
        segment=previous+np.linspace(0,1,101)[:,None]*(command-previous)
        before=approach_geometry(kinematics,previous)['centre_z_mm']
        assert min(approach_geometry(kinematics,x)['centre_z_mm'] for x in segment)>=before-.02
        assert min(solver.clearance.clearance_mm(x) for x in segment)>=20.
        previous=command


def test_nonlinear_radial_view_probe_stays_high_and_does_not_roll_or_close(kinematics):
    q=np.array([0.,-28.7439774,43.1243436,79.8371826,-51.584415,31.143232])
    solver=VerticalApproachSolver(kinematics,real_gate(kinematics),LOWER,UPPER)
    result=solver.propose_view_chunk(q,radial_probe_mm=5.)
    assert result.accepted,result.diagnostic
    assert result.diagnostic['stage']=='radial_view_probe'
    assert result.diagnostic['endpoint']['radial_mm']-result.diagnostic['source']['radial_mm']==pytest.approx(5.,abs=.02)
    np.testing.assert_array_equal(result.commands_position[:,[0,4,5]],np.repeat(q[None,[0,4,5]],5,0))


def test_nonlinear_view_chunk_does_not_waive_low_clearance(kinematics):
    solver=VerticalApproachSolver(kinematics,real_gate(kinematics),LOWER,UPPER)
    result=solver.propose_view_chunk(CURRENT)
    assert not result.accepted and 'rotation_clearance' in result.diagnostic['reason']


def test_larger_high_view_chunk_keeps_same_waypoint_caps_and_clearance(kinematics):
    q=np.array([0.,-23.754631,40.372669,79.530205,-51.584415,31.143232])
    solver=VerticalApproachSolver(kinematics,real_gate(kinematics),LOWER,UPPER)
    result=solver.propose_view_chunk(q,radial_probe_mm=30.)
    assert result.accepted,result.diagnostic
    assert result.diagnostic['endpoint']['radial_mm']-result.diagnostic['source']['radial_mm']==pytest.approx(30.,abs=.02)
    assert result.diagnostic['minimum_path_clearance_mm']>=20.
    assert np.abs(np.diff(np.vstack((q,result.commands_position)),axis=0)).max()<=6.+1e-6
    np.testing.assert_array_equal(result.commands_position[:,[0,4,5]],np.repeat(q[None,[0,4,5]],5,0))
    assert not solver.propose_view_chunk(q,radial_probe_mm=31.).accepted


def test_view_acquisition_does_not_repeatedly_verticalize_small_tracking_tilt(kinematics):
    q=np.array([-18.352602,20.461096,-7.187222,83.389259,-51.584415,31.143232])
    solver=VerticalApproachSolver(kinematics,real_gate(kinematics),LOWER,UPPER,vertical_tolerance_deg=10.)
    result=solver.propose_view_chunk(q,radial_probe_mm=30.)
    assert result.accepted,result.diagnostic
    assert result.diagnostic['stage']=='radial_view_probe'
    assert result.diagnostic['minimum_path_clearance_mm']>=20.
    assert result.diagnostic['endpoint']['radial_mm']-result.diagnostic['source']['radial_mm']>20.
    assert result.diagnostic['endpoint']['tilt_deg']<10.
    # The precision/contact solver's default was NOT changed.
    assert VerticalApproachSolver(kinematics,real_gate(kinematics),LOWER,UPPER).tolerance==5.


def test_official_axis_not_an_assumed_rotation_column(kinematics):
    from fcpg.so101_kinematics import normalized_positions_to_mujoco_qpos
    with kinematics._forward_lock:
        kinematics._set_state(normalized_positions_to_mujoco_qpos(CURRENT, kinematics.all_joint_ranges))
        axis = (kinematics.data.site_xpos[kinematics.gripperframe_site_id]
                - kinematics.data.xpos[kinematics.gripper_body_id]).copy()
    axis /= np.linalg.norm(axis)
    actual = approach_geometry(kinematics, CURRENT)
    np.testing.assert_allclose(actual['approach_axis_world'], axis)
    assert actual['tilt_deg'] == pytest.approx(14.0444, abs=0.01)
    np.testing.assert_allclose(actual['anchor_xyz_mm'], 1000 * kinematics.forward_anchors(CURRENT))


def test_low_tilted_pose_only_lifts_and_preserves_other_joints(kinematics):
    solver = VerticalApproachSolver(kinematics, real_gate(kinematics), LOWER, UPPER)
    result = solver.propose(CURRENT, dz_mm=-5., dr_mm=10.)
    assert result.accepted
    assert result.diagnostic['stage'] == 'lift_before_verticalize'
    assert result.diagnostic['requires_physical_audit']
    commands = result.commands_position
    assert commands.shape == (5, 6)
    np.testing.assert_allclose(commands[:, [0, 4, 5]], np.tile(CURRENT[[0, 4, 5]], (5, 1)))
    assert np.abs(commands[-1] - CURRENT).max() <= 2.0 + 1e-12
    sweep = CURRENT + np.linspace(0., 1., 101)[:, None] * (commands[-1] - CURRENT)
    anchors = kinematics.forward_anchors(sweep)
    assert (anchors[:, :, 2] >= anchors[0, :, 2] - 2e-5).all()
    assert result.diagnostic['endpoint']['centre_z_mm'] > result.diagnostic['source']['centre_z_mm']


def test_real_model_verticalizes_before_descending(kinematics):
    solver = VerticalApproachSolver(kinematics, real_gate(kinematics), LOWER, UPPER)
    q = CURRENT.copy()
    stages = []
    for _ in range(20):
        result = solver.propose(q, -3., 0.)
        assert result.accepted, result.diagnostic
        diag = result.diagnostic
        stages.append(diag['stage'])
        if diag['stage'] == 'verticalize':
            assert diag['endpoint']['tilt_deg'] < diag['source']['tilt_deg']
            assert diag['endpoint']['centre_z_mm'] >= diag['source']['centre_z_mm'] - .02
        if diag['stage'] == 'approach':
            assert diag['source']['tilt_deg'] <= 5.
            assert diag['endpoint']['tilt_deg'] <= 5.
            assert diag['endpoint']['centre_z_mm'] < diag['source']['centre_z_mm']
            break
        q = result.commands_position[-1]
    assert set(stages) == {'lift_before_verticalize', 'verticalize', 'approach'}


class LinearSolver(VerticalApproachSolver):
    def _geometry(self, q):
        return dict(radial_mm=200. + q[1], centre_z_mm=30. + q[2],
                    pitch_rad=.01 * q[3], tilt_deg=abs(np.rad2deg(.01 * q[3])),
                    anchor_xyz_mm=[[200. + q[1], 0., 30. + q[2]],
                                   [200. + q[1], -2., 30. + q[2]],
                                   [200. + q[1], 2., 30. + q[2]]])


def linear_solver(**kwargs):
    gate = SimpleNamespace(readback_clearance_mm=-3., planned_clearance_mm=-3.,
                           clearance_mm=lambda q: 30. + q[2])
    return LinearSolver(None, gate, LOWER, UPPER, **kwargs)


@pytest.mark.parametrize('q,dz,dr', [([0.] * 5, -2., 0.),
                                   ([0., 0., np.nan, 0., 0., 5.], -2., 0.),
                                   ([0.] * 5 + [5.], np.inf, 0.),
                                   ([0.] * 5 + [5.], -2., np.nan)])
def test_nonfinite_or_wrong_shape_input_rejected(q, dz, dr):
    proposal = linear_solver().propose(q, dz, dr)
    assert not proposal.accepted
    assert proposal.commands_position is None


def test_bounds_not_silently_clipped():
    q = np.array([0., 100.01, 0., 0., 0., 5.])
    proposal = linear_solver().propose(q, -2.)
    assert proposal.diagnostic['reason'] == 'source_outside_joint_bounds'


def test_unknown_floor_and_lowered_floor_fail_closed():
    solver = linear_solver()
    solver.clearance.readback_clearance_mm = -4.
    assert not solver.propose(np.array([0.] * 5 + [5.]), -2.).accepted
    solver.clearance.readback_clearance_mm = np.nan
    assert not solver.propose(np.array([0.] * 5 + [5.]), -2.).accepted


def test_singular_jacobian_rejected_without_pseudomotion():
    solver = linear_solver()
    solver._geometry = lambda q: dict(radial_mm=200., centre_z_mm=30., pitch_rad=0.,
        tilt_deg=0., anchor_xyz_mm=[[200., 0., 30.]] * 3)
    proposal = solver.propose(np.array([0.] * 5 + [5.]), -2.)
    assert proposal.diagnostic['reason'] == 'singular_or_ill_conditioned_jacobian'
    assert proposal.commands_position is None


def test_jaw_sweep_rejects_deep_intermediate_dip():
    solver = linear_solver()
    # Same endpoint clearance does not prove the intermediate path is above the table.
    solver.clearance.clearance_mm = lambda q: 30. - 40. * np.exp(-((q[1] - .5) / .2)**2)
    proposal = solver.propose(np.array([0.] * 5 + [5.]), 0., 1.)
    assert proposal.accepted
    assert proposal.commands_position[-1, 1] < .4
    assert proposal.diagnostic['minimum_path_clearance_mm'] >= 2.


def test_descent_cannot_go_below_tracking_reserve():
    solver = linear_solver()
    q = np.array([0., 0., -27.99, 0., 0., 5.])
    proposal = solver.propose(q, -5.)
    assert not proposal.accepted
    assert proposal.commands_position is None


def test_lateral_unreachable_orientation_is_not_radial_pitch_success():
    solver = linear_solver()
    original = solver._geometry
    def geometry(q):
        result = original(q)
        result['tilt_deg'] = 12.
        return result
    solver._geometry = geometry
    proposal = solver.propose(np.array([0.] * 5 + [5.]), -2.)
    assert proposal.diagnostic['reason'] == 'vertical_unreachable_with_fixed_J5'


def test_parameters_cannot_remove_tracking_reserve():
    with pytest.raises(ValueError):
        linear_solver(tracking_reserve_mm=0.1)


def test_noop_is_not_fake_success():
    proposal = linear_solver().propose(np.array([0.] * 5 + [5.]), 0., 0.)
    assert not proposal.accepted
    assert proposal.diagnostic['reason'] == 'already_vertical_no_translation_requested'


def test_below_reserve_but_above_hard_floor_only_recovers_upward():
    solver = linear_solver()
    q = np.array([0., 0., -28.5, 0., 0., 5.])  # 1.5 mm, above -3 but below 2 mm reserve.
    proposal = solver.propose(q, -5., 4.)
    assert proposal.accepted
    assert proposal.diagnostic['stage'] == 'lift_to_clearance_reserve'
    assert proposal.diagnostic['monotonic_reserve_recovery']
    assert proposal.diagnostic['target_clearance_reserve_mm'] == 2.
    path_clearance = [solver.clearance.clearance_mm(row) for row in proposal.commands_position]
    assert np.all(np.diff([1.5, *path_clearance]) >= 0.)
    assert path_clearance[-1] > 2.
    np.testing.assert_allclose(proposal.commands_position[:, [0, 1, 3, 4, 5]],
                               np.tile(q[[0, 1, 3, 4, 5]], (5, 1)))
    assert solver.clearance.readback_clearance_mm == -3.


def test_below_hard_floor_cannot_use_reserve_recovery():
    q = np.array([0., 0., -33.001, 0., 0., 5.])
    proposal = linear_solver().propose(q, 5., 0.)
    assert not proposal.accepted
    assert proposal.diagnostic['reason'] == 'source_below_readback_floor'
