"""Synthetic geometry tests; no RGB accuracy, camera or hardware claims."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.so101_camera_calibration import ProjectionFit
from fcpg.v21_column_pose_prior import ColumnCameraUncertainty, ColumnPosePrior


MATRIX = np.array([[500., 0., 0., 112.], [0., 500., 0., 112.]])
WORLD = np.array([[.03, .04, .2], [-.03, .04, .2]])
EXACT = ColumnCameraUncertainty(rotation_std_deg=(0., 0., 0.), log_scale_std=0.)


class SyntheticSixJointFK:
    """A minimal six-DOF manipulator: XYZ, tilt, wrist roll, jaw opening."""

    def forward_anchor_frames(self, q):
        self.last_q = q.copy()
        rows = []
        for state in q:
            center = state[:3] * .001
            tilt, roll = np.deg2rad(state[3:5])
            separation = (.03 + state[5] * .0005) * np.array([
                np.cos(roll) * np.cos(tilt), np.cos(roll) * np.sin(tilt), np.sin(roll)])
            rows.append(np.stack((center, center - separation / 2., center + separation / 2.)))
        result = np.array(rows)
        return result, result.copy()


def fit(matrix=MATRIX):
    return ProjectionFit("column", ("x_base", "y_base", "z_base", "bias"), matrix.T, 0.)


def test_full_exposure_q_is_forwarded_and_every_joint_changes_geometry():
    kin = SyntheticSixJointFK()
    q = np.array([10., 20., 30., 35., 40., 50.])
    baseline = ColumnPosePrior.from_kinematics(kin, q, fit(), uncertainty=EXACT)
    np.testing.assert_array_equal(kin.last_q, q[None])
    for joint in range(6):
        changed = q.copy()
        changed[joint] += 5.
        prior = ColumnPosePrior.from_kinematics(kin, changed, fit(), uncertainty=EXACT)
        assert not np.allclose(prior.world_pair_xyz, baseline.world_pair_xyz)
        np.testing.assert_array_equal(prior.q_at_image, changed)
    assert baseline.as_dict()["full_joint_fk"]


def test_camera_translation_changes_location_not_relative_geometry_or_order():
    first = ColumnPosePrior.from_world_pair(WORLD, MATRIX)
    matrix = MATRIX.copy()
    translation = np.array([60., -30.])
    matrix[:, 3] += translation
    second = ColumnPosePrior.from_world_pair(WORLD, matrix)
    np.testing.assert_allclose(second.mean_pair_px, first.mean_pair_px + translation)
    np.testing.assert_allclose(second.relative_cov_px2, first.relative_cov_px2, atol=1e-12)
    np.testing.assert_allclose(second.relative_mean_px, first.relative_mean_px)
    assert second.score_pair(first.mean_pair_px + translation)["log_likelihood"] == pytest.approx(
        first.score_pair(first.mean_pair_px)["log_likelihood"])


def test_j5_sideview_naturally_overlaps_without_minimum_span():
    kin = SyntheticSixJointFK()
    front = ColumnPosePrior.from_kinematics(kin, [0., 0., 0., 0., 0., 50.], fit(), uncertainty=EXACT)
    side = ColumnPosePrior.from_kinematics(kin, [0., 0., 0., 0., 90., 50.], fit(), uncertainty=EXACT)
    assert np.linalg.norm(front.relative_mean_px) > 20.
    np.testing.assert_allclose(side.relative_mean_px, 0., atol=1e-12)
    score = side.score_pair(np.tile(side.center_mean_px, (2, 1)))
    assert score["likelihood"] == pytest.approx(1.)
    assert score["identity_probability"] == pytest.approx(.5)
    report = side.as_dict()
    assert report["overlap_affinity"] == pytest.approx(1.)
    assert report["foreshortening_fraction"] < 1e-12
    assert not report["observed"] and report["inferred"]


def test_swapped_rgb_pair_association_and_coincident_identity_ambiguity():
    prior = ColumnPosePrior.from_world_pair(WORLD, MATRIX, uncertainty=EXACT)
    ordered = prior.score_pair(prior.nominal_pair_px)
    swapped = prior.score_pair(prior.nominal_pair_px[::-1])
    assert not ordered["swapped"] and swapped["swapped"]
    np.testing.assert_allclose(ordered["ordered_pair"], prior.nominal_pair_px)
    assert swapped["identity_probability"] > .99
    np.testing.assert_allclose(swapped["ordered_pair"], prior.nominal_pair_px)
    assert ordered["log_likelihood"] == pytest.approx(swapped["log_likelihood"])
    forced = prior.score_pair(prior.nominal_pair_px[::-1], allow_swap=False)
    assert forced["log_likelihood"] < swapped["log_likelihood"]
    assert forced["identity_probability"] < .01
    assert prior.score_pair(np.tile(prior.center_mean_px, (2, 1)))["identity_probability"] == pytest.approx(.5)


def test_camera_ambiguity_increases_relative_and_absolute_uncertainty():
    narrow = ColumnPosePrior.from_world_pair(WORLD, MATRIX, uncertainty=replace(EXACT,
        translation_std_fraction=(.02, .02)))
    wide = ColumnPosePrior.from_world_pair(WORLD, MATRIX)
    assert np.trace(wide.relative_cov_px2) > np.trace(narrow.relative_cov_px2)
    assert np.trace(wide.pair_cov_px2) > np.trace(narrow.pair_cov_px2)
    assert len(wide.hypothesis_weights) == 81
    assert wide.hypothesis_weights.sum() == pytest.approx(1.)


def test_proposal_updates_common_center_only_and_preserves_camera_transform():
    proposal = np.array([[30., 80.], [70., 80.]])
    prior = ColumnPosePrior.from_world_pair(WORLD, MATRIX, proposal_pair=proposal)
    changed_shape = ColumnPosePrior.from_world_pair(WORLD, MATRIX,
        proposal_pair=proposal.mean(0) + np.array([[-80., 40.], [80., -40.]]))
    untouched = ColumnPosePrior.from_world_pair(WORLD, MATRIX)
    np.testing.assert_allclose(prior.mean_pair_px, changed_shape.mean_pair_px)
    np.testing.assert_allclose(prior.relative_cov_px2, untouched.relative_cov_px2, atol=1e-12)
    np.testing.assert_allclose(prior.relative_mean_px, untouched.relative_mean_px)
    assert np.linalg.norm(prior.center_mean_px - proposal.mean(0)) < np.linalg.norm(
        untouched.center_mean_px - proposal.mean(0))
    homogeneous = np.column_stack((WORLD, np.ones(2)))
    reconstructed = np.einsum("pj,nkj->npk", homogeneous, prior.camera_matrices)
    np.testing.assert_allclose(reconstructed, prior.hypothesis_pairs_px, atol=1e-12)
    report = prior.as_dict(include_hypotheses=True)
    assert not report["proposal_used_as_endpoint"]
    assert not report["accuracy_verified"]
    json.dumps(report, allow_nan=False)


def test_correlated_center_uncertainty_moves_both_tips_together():
    prior = ColumnPosePrior.from_world_pair(WORLD, MATRIX, uncertainty=EXACT)
    np.testing.assert_allclose(prior.pair_cov_px2[:2, 2:], prior.center_cov_px2)
    relative_transform = np.column_stack((-np.eye(2), np.eye(2)))
    np.testing.assert_allclose(relative_transform @ prior.pair_cov_px2 @ relative_transform.T, 0.)


def test_invalid_data_fails_closed_but_zero_physical_separation_is_valid():
    kin = SyntheticSixJointFK()
    for q in ([0., 0., 0., 0., 0., -1.], [101., 0., 0., 0., 0., 10.]):
        with pytest.raises(ValueError, match="no silent clipping"):
            ColumnPosePrior.from_kinematics(kin, q, fit())
    with pytest.raises(ValueError, match="two image axes"):
        ColumnPosePrior.from_world_pair(WORLD, np.zeros((2, 4)))
    with pytest.raises(ValueError, match="positive semidefinite"):
        ColumnPosePrior.from_world_pair(WORLD, MATRIX, proposal_pair=np.ones((2, 2)),
                                        proposal_center_cov_px2=[[1., 2.], [2., 1.]])
    same = ColumnPosePrior.from_world_pair(np.tile(WORLD[0], (2, 1)), MATRIX)
    assert same.score_pair(same.mean_pair_px)["identity_probability"] == pytest.approx(.5)
    assert same.as_dict()["physical_tip_separation_m"] == 0.


def test_real_mujoco_full_fk_all_joints_affect_prior():
    pytest.importorskip("mujoco")
    from fcpg.so101_kinematics import SO101MuJoCoKinematics
    model = Path(__file__).resolve().parents[1] / "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
    if not model.exists():
        pytest.skip("official model unavailable")
    kin = SO101MuJoCoKinematics(model)
    q = np.array([-5.78, -89.87, 92.01, 67.70, -52.52, 34.49])
    baseline = ColumnPosePrior.from_kinematics(kin, q, fit())
    for joint in range(6):
        changed = q.copy()
        changed[joint] += 2.
        other = ColumnPosePrior.from_kinematics(kin, changed, fit())
        assert np.linalg.norm(other.world_pair_xyz - baseline.world_pair_xyz) > 1e-6
        assert np.linalg.norm(other.nominal_pair_px - baseline.nominal_pair_px) > 1e-5
