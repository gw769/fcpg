from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from fcpg.so101_camera_calibration import ProjectionFit
from fcpg.v21_column_pose_prior import ColumnCameraUncertainty, ColumnPosePrior
from fcpg.v21_column_temporal_prior import (
    METHOD_LABEL, ColumnTemporalPrior, ColumnTemporalTracker, MAX_SCORE_BONUS,
)
from fcpg.v21_column_tip_visual import locate_column_jaw_tips


def state(index, session='one'):
    return SimpleNamespace(progress_session_id=session, q=torch.zeros(1, 6),
        memory=SimpleNamespace(current_readback_index=torch.tensor([index])),
        feedback_readback_index=torch.tensor([index]), feedback_available=torch.tensor([index > 0]))


def fixtures():
    source, current = state(0), state(5)
    predicted = torch.zeros(1, 6, 6)
    predicted[0, -1, 0] = 1.
    commands = torch.zeros(1, 5, 6)
    forecast = SimpleNamespace(source_q=source.q.clone(),
        response=SimpleNamespace(predicted_q_path=predicted),
        trace=SimpleNamespace(positions=commands))
    bridge = SimpleNamespace(runtime=SimpleNamespace(requires_home_after_incomplete_execution=False),
        last_source_state=source, last_forecast=forecast,
        last_attempt=dict(completed=True, policy_commands_attempted=5, policy_commands_returned=5,
            measured_q_path=np.zeros((6, 6)).tolist(), commands_position=commands[0].tolist()))
    def world(q):
        result = np.array([[[0., 0., 0.], [-.02, 0., 0.], [.02, 0., 0.]]])
        result += np.array([.01 * q[0, 0], 0., 0.])
        return result, result.copy()
    kin = SimpleNamespace(forward_anchor_frames=world)
    fit = ProjectionFit('column', ('x_base', 'y_base', 'z_base', 'bias'),
        np.array([[1000., 0.], [0., 1000.], [0., 0.], [107., 100.]]), 0.)
    tip = dict(available=True, source='column-rgb-jaw-contour',
        fm_px=[[90., 143.], [124., 143.]], q_at_image=np.zeros(6).tolist(), fk_based=False)
    tracker = ColumnTemporalTracker()
    tracker.record_observation(source, tip)
    return tracker, source, current, bridge, kin, fit, tip


def test_only_completed_readback_index_transports_previous_rgb_center():
    tracker, source, current, bridge, kin, fit, tip = fixtures()
    prior = tracker.for_current(current, bridge, kin, fit)
    assert prior is not None
    record = prior.as_dict()
    assert record['source_readback_index'] == 0 and record['target_readback_index'] == 5
    assert record['progress_session_id'] == 'one'
    # The old observed center is 107,143; q forecast moves FK to the right.
    assert 115. < record['predicted_midpoint_px'][0] < 118.
    assert record['predicted_midpoint_px'][1] == pytest.approx(143.)
    assert record['full_joint_fk'] and record['shared_camera_transform']
    assert record['method_label'] == METHOD_LABEL
    assert not record['fcpg_pixel_prediction_available']
    assert not record['actual_visual_memory_write']
    assert not record['display_only'] and not record['direct_control_input']
    assert record['proposal_ranking_input']
    assert record['may_influence_control_via_current_rgb_observation']
    assert not record['predicted_endpoints_published_as_observation']
    assert set(vars(current.memory)) == {'current_readback_index'}
    np.testing.assert_array_equal(tip['fm_px'], [[90., 143.], [124., 143.]])


@pytest.mark.parametrize('kind', ['cross_session', 'wrong_target_index', 'wrong_feedback_index',
    'missing_readbacks', 'wrong_commands', 'wrong_forecast_source', 'missing_source_rgb', 'planned_only'])
def test_misaligned_or_unexecuted_evidence_never_becomes_temporal_prior(kind):
    tracker, source, current, bridge, kin, fit, tip = fixtures()
    if kind == 'cross_session':
        current.progress_session_id = 'other'
    elif kind == 'wrong_target_index':
        current.memory.current_readback_index.fill_(10)
    elif kind == 'wrong_feedback_index':
        current.feedback_readback_index.fill_(4)
    elif kind == 'missing_readbacks':
        bridge.last_attempt['measured_q_path'] = [[0.] * 6] * 5
    elif kind == 'wrong_commands':
        bridge.last_attempt['commands_position'][0][0] = 42.
    elif kind == 'wrong_forecast_source':
        bridge.last_forecast.source_q[0, 0] = 42.
    elif kind == 'missing_source_rgb':
        tracker.observations.clear()
    else:
        bridge.runtime.pending = bridge.last_forecast
        bridge.last_forecast = None
        bridge.last_attempt = None
    assert tracker.for_current(current, bridge, kin, fit) is None


def test_initial_i_zero_without_prior_is_normal():
    tracker, source, _, bridge, kin, fit, _ = fixtures()
    assert tracker.for_current(source, bridge, kin, fit) is None
    assert tracker.observations
    assert tracker.diagnostic['reason'] == 'no_completed_transition'


def test_temporal_status_distinguishes_missing_visual_seed_and_failed_response():
    tracker, _, current, bridge, kin, fit, _ = fixtures()
    tracker.observations.clear()
    assert tracker.for_current(current, bridge, kin, fit) is None
    assert tracker.diagnostic['reason'] == 'source_visual_pair_unavailable'
    assert tracker.diagnostic['source_readback_index'] == 0

    bridge.last_attempt['response_verified'] = False
    bridge.last_attempt['response_error'] = 'J5 actual motion below threshold'
    assert tracker.for_current(current, bridge, kin, fit) is None
    assert tracker.diagnostic['reason'] == 'completed_response_not_verified'
    assert tracker.diagnostic['response_error'] == bridge.last_attempt['response_error']
    assert not tracker.diagnostic['fcpg_pixel_prediction_available']


def test_successful_and_cached_temporal_status_reports_actual_prediction_source():
    tracker, _, current, bridge, kin, fit, _ = fixtures()
    first = tracker.for_current(current, bridge, kin, fit)
    assert first is not None
    assert tracker.diagnostic['available'] and tracker.diagnostic['reason'] is None
    assert tracker.diagnostic['method_label'] == METHOD_LABEL
    assert not tracker.diagnostic['fcpg_pixel_prediction_available']
    assert tracker.for_current(current, bridge, kin, fit) is first
    assert tracker.diagnostic['available']


@pytest.mark.parametrize('source_index', [0, 5, 10, 95])
def test_next_camera_observation_is_next_chunk_but_fifth_readback_sample(source_index):
    tracker, source, current, bridge, kin, fit, tip = fixtures()
    source.memory.current_readback_index.fill_(source_index)
    source.feedback_readback_index.fill_(source_index)
    current.memory.current_readback_index.fill_(source_index + 5)
    current.feedback_readback_index.fill_(source_index + 5)
    tracker.record_observation(source, tip)
    record = tracker.for_current(current, bridge, kin, fit).as_dict()
    assert record['target_readback_index'] - record['source_readback_index'] == 5
    assert record['target_chunk_index'] - record['source_chunk_index'] == 1
    assert record['forecast_q_path_row'] == 5
    assert record['readbacks_per_completed_chunk'] == 5
    # Intermediate q1 is predicted, but the service has no q1 camera endpoint
    # and has not committed a completed policy chunk at that index.
    current.memory.current_readback_index.fill_(source_index + 1)
    current.feedback_readback_index.fill_(source_index + 1)
    assert tracker.for_current(current, bridge, kin, fit) is None


def test_replan_and_duplicate_publication_do_not_retarget_executed_forecast():
    tracker, source, current, bridge, kin, fit, tip = fixtures()
    original = tracker.for_current(current, bridge, kin, fit)
    old = original.as_dict()
    # Pending plans and UI sequences are irrelevant to completed action i.
    bridge.runtime.pending = deepcopy(bridge.last_forecast)
    bridge.runtime.pending.response.predicted_q_path.add_(70.)
    bridge.runtime.pending.source_time = torch.tensor([1e12])
    current.snapshot = SimpleNamespace(sequence=99999)
    current.time = torch.tensor([1e12])
    new_tip = dict(tip, fm_px=[[110., 143.], [144., 143.]])
    tracker.record_observation(current, new_tip)
    again = tracker.for_current(current, bridge, kin, fit)
    assert again is original and again.as_dict() == old


@pytest.mark.parametrize('kind', ['partial', 'failed_complete', 'failed_response', 'requires_home'])
def test_failure_clears_prior_and_observation_seed(kind):
    tracker, _, current, bridge, kin, fit, _ = fixtures()
    assert tracker.for_current(current, bridge, kin, fit) is not None
    if kind == 'partial':
        bridge.last_attempt = dict(completed=False, policy_commands_attempted=2)
    elif kind == 'failed_complete':
        bridge.last_attempt['error'] = 'capture failure'
    elif kind == 'failed_response':
        bridge.last_attempt['response_verified'] = False
    else:
        bridge.runtime.requires_home_after_incomplete_execution = True
    assert tracker.for_current(current, bridge, kin, fit) is None
    assert not tracker.observations and tracker._prior is None


def temporal(center):
    pair = (np.asarray(center) + [[-17., 0.], [17., 0.]]).tolist()
    return ColumnTemporalPrior(dict(source='test-completed-temporal',
        predicted_midpoint_px=center, midpoint_cov_px2=[[64., 0.], [0., 64.]],
        predicted_pair_px=pair, pair_cov_px2=(np.eye(4) * 64.).tolist(),
        source_pair_associations=[dict(probability=1., predicted_pair_px=pair)]))


def scene():
    image = np.full((224, 224, 3), 40, np.uint8)
    image[:28] = 0; image[196:] = 0
    for x in (90, 124):
        cv2.rectangle(image, (x - 8, 89), (x + 8, 104), (180, 210, 225), -1)
        cv2.fillConvexPoly(image, np.array([[x - 5, 102], [x + 5, 102], [x, 143]]), (60, 90, 130))
    return image


def test_temporal_bonus_is_bounded_and_never_changes_contour_coordinates():
    image, proposal = scene(), [[94., 141.], [120., 141.]]
    plain = locate_column_jaw_tips(image, proposal, image_space='letterbox224')
    for center in ([107., 143.], [107., 70.]):
        prior = temporal(center)
        actual = locate_column_jaw_tips(image, proposal, image_space='letterbox224', temporal_prior=prior)
        assert actual['available'] and actual['temporal_prior_used']
        np.testing.assert_array_equal(actual['fm_px'], plain['fm_px'])
        for row in actual['pair_candidates']:
            assert 0. <= row['temporal_match']['score_bonus'] <= MAX_SCORE_BONUS
    absent = locate_column_jaw_tips(np.zeros_like(image), proposal,
        image_space='letterbox224', temporal_prior=temporal([107., 143.]))
    assert not absent['available'] and absent['fm_px'] is None


def test_perfect_temporal_match_cannot_override_full_q_relative_gate():
    proposal = [[94., 141.], [120., 141.]]
    vertical_prior = ColumnPosePrior.from_world_pair(
        [[0., -.05, 0.], [0., .05, 0.]],
        [[1000., 0., 0., 107.], [0., 1000., 0., 143.]], proposal_pair=proposal,
        uncertainty=ColumnCameraUncertainty(rotation_std_deg=(0., 0., 0.), log_scale_std=0.))
    result = locate_column_jaw_tips(scene(), proposal, image_space='letterbox224',
        pose_prior=vertical_prior, temporal_prior=temporal([107., 143.]))
    assert not result['available'] and result['fm_px'] is None
    assert result['pose_relative_consistency_rejections']


@pytest.mark.parametrize('rotation,opening', [(90., 0.), (0., 50.), (65., 35.)])
def test_full_fk_pair_prediction_distinguishes_rotation_and_span_at_same_midpoint(rotation, opening):
    tracker, source, current, bridge, _, fit, tip = fixtures()
    def frames(q):
        angle = np.deg2rad(q[0, 4])
        half = .017 * (1. + q[0, 5] / 50.)
        axis = half * np.array([np.cos(angle), np.sin(angle), 0.])
        world = np.array([[np.zeros(3), -axis, axis]])
        return world, world.copy()
    kin = SimpleNamespace(forward_anchor_frames=frames)
    predicted = bridge.last_forecast.response.predicted_q_path
    predicted.zero_()
    predicted[0, -1, 4] = rotation
    predicted[0, -1, 5] = opening
    prior = tracker.for_current(current, bridge, kin, fit)
    record = prior.as_dict()
    expected = np.asarray(record['predicted_pair_px'])
    np.testing.assert_allclose(expected.mean(0), np.asarray(tip['fm_px']).mean(0), atol=1e-9)
    actual_score = prior.score_pair(expected)
    previous_score = prior.score_pair(tip['fm_px'])
    # Midpoint-only ranking assigned both candidates the same score. A
    # transported pair correctly distinguishes jaw rotation/opening.
    assert actual_score['midpoint_mahalanobis_sq'] == pytest.approx(0., abs=1e-20)
    assert previous_score['midpoint_mahalanobis_sq'] == pytest.approx(0., abs=1e-20)
    assert actual_score['score_bonus'] > previous_score['score_bonus'] + .1
    assert actual_score['score_bonus'] == pytest.approx(MAX_SCORE_BONUS)
    assert prior.score_pair(expected[::-1])['score_bonus'] == pytest.approx(actual_score['score_bonus'])
    assert prior.score_pair(expected[::-1])['candidate_pair_swapped']
    covariance = np.asarray(record['pair_cov_px2'])
    assert covariance.shape == (4, 4) and np.linalg.eigvalsh(covariance).min() > 0.
    assert np.linalg.norm(covariance[:2, 2:]) > 0.  # common camera/center covariance
    assert not record['fcpg_pixel_prediction_available']


def test_ambiguous_previous_identity_keeps_two_pair_modes_without_collapsing_tips():
    tracker, source, current, bridge, _, fit, tip = fixtures()
    def frames(q):
        world = np.array([[[0., 0., 0.], [0., -.017, 0.], [0., .017, 0.]]])
        world += np.array([.01 * q[0, 0], 0., 0.])
        return world, world.copy()
    prior = tracker.for_current(current, bridge, SimpleNamespace(forward_anchor_frames=frames), fit)
    record = prior.as_dict()
    assert record['source_identity_probability'] == pytest.approx(.5)
    assert len(record['source_pair_associations']) == 2
    for mode in record['source_pair_associations']:
        pair = np.asarray(mode['predicted_pair_px'])
        assert np.linalg.norm(pair[1] - pair[0]) == pytest.approx(34.)
        assert prior.score_pair(pair)['score_bonus'] == pytest.approx(MAX_SCORE_BONUS)


def test_display_suggestion_cannot_seed_the_following_prediction_as_an_observation():
    tracker, _, current, bridge, kin, fit, _ = fixtures()
    prior = tracker.for_current(current, bridge, kin, fit)
    suggestion = prior.as_suggestion(reason='no new Column image')
    missing_visual = dict(available=False, observed=False, source='column-rgb-jaw-contour',
        fm_px=None, suggested_fm_px=suggestion['suggested_fm_px'], q_at_image=[0.] * 6,
        prediction_suggestion=suggestion)
    tracker.record_observation(current, missing_visual)
    assert 5 not in tracker.observations
    bridge.last_source_state = current
    assert tracker.for_current(state(10), bridge, kin, fit) is None


@pytest.mark.parametrize('case,expected', [('close_weights', 1), ('low_weight', 0),
    ('low_visibility', 0), ('bad_pose', 0), ('no_prior', 0)])
def test_current_dino_proposal_temporal_rerank_preserves_visual_weight_and_pose_evidence(case, expected):
    from fcpg.v21_tip_display import select_column_proposal_component
    proposals = np.asarray([[[90., 143.], [124., 143.]], [[102., 143.], [136., 143.]]])
    weights = np.asarray([.51, .49])
    visible = np.ones((2, 2)) * .99
    prior = temporal([119., 143.])
    pose = ColumnPosePrior.from_world_pair(
        [[-.017, 0., 0.], [.017, 0., 0.]],
        [[1000., 0., 0., 107.], [0., 1000., 0., 143.]], proposal_pair=proposals[0])
    if case == 'low_weight':
        weights = np.asarray([.99, .01])
    elif case == 'low_visibility':
        visible[1, 0] = .01
    elif case == 'bad_pose':
        proposals[1] = [[119., 126.], [119., 160.]]
        prior.record['predicted_pair_px'] = proposals[1].tolist()
        prior.record['source_pair_associations'][0]['predicted_pair_px'] = proposals[1].tolist()
    elif case == 'no_prior':
        prior = None
    before = proposals.copy()
    selected, diagnostic = select_column_proposal_component(weights, visible, proposals,
        temporal_prior=prior, pose_prior=pose, visibility_threshold=.5)
    assert selected == expected
    assert diagnostic['changed'] is bool(expected)
    np.testing.assert_array_equal(proposals, before)
    if case == 'close_weights':
        assert 0. < diagnostic['candidates'][1]['temporal_bonus'] <= MAX_SCORE_BONUS
    if case in ('low_visibility', 'bad_pose'):
        assert not diagnostic['candidates'][1]['eligible']
        assert diagnostic['candidates'][1]['temporal_bonus'] == 0.
