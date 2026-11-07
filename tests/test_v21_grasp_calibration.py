"""Synthetic serialized calibration receipts; no hardware or physical evidence.

AlignmentSession runs only against the in-memory callbacks below. Its complete
receipts are serialized to pytest temporary files and read by the real loader.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.prod_v20_policy import empty_history
from fcpg.v21_alignment_session import AlignmentSession
from fcpg.v21_grasp_cost import load_measured_wrist_response
from fcpg.v21_visual_alignment import alignment_observation


SOURCE_Q = np.array([-5., -25., 35., 10., -65., 30.])
IMAGE_MAP = np.array([[-6., 0., -1., .5, 0.],
                      [0., -5., .5, -1., 0.],
                      [0., 0., 0., 0., np.radians(8.)]])


@pytest.fixture
def receipt_data(request):
    q, clock = SOURCE_Q.copy(), [10.]
    body_truncated = getattr(request, 'param', False)

    def measure():
        features = np.array([132., 112., 0.]) + IMAGE_MAP @ (q - SOURCE_Q)[:5]
        target = dict(role='pickup', view='wrist', visible=True,
                      view_source='current_observation', source='operator_colour_body',
                      image_time_s=clock[0], cog_px=features[:2].tolist(),
                      axis_px=[np.cos(features[2]), np.sin(features[2])],
                      width_px=20., length_px=40., body_truncated=body_truncated)
        return alignment_observation(target, q, [[92., 112.], [132., 112.]],
                                     image_time_s=clock[0], q_time_s=clock[0] + .01,
                                     now_s=clock[0] + .02, max_age_s=.2)

    def simulated_motion(builder, source):
        state = SimpleNamespace(q=torch.tensor(q)[None], history=empty_history(1))
        commands = builder(state)[0].detach().numpy().copy()
        origin = q.copy()
        # Achieved motion deliberately differs from the requested motion and
        # model prediction. The loader must fit the saved achieved pairs.
        measured = np.vstack((origin, origin + .8 * (commands - origin)))
        predicted = np.vstack((origin, commands))
        source_time = clock[0] + .02
        command_times = source_time + np.arange(5) * .1
        readback_times = source_time + np.arange(1, 6) * .1
        q[:] = measured[-1]
        clock[0] = readback_times[-1] + .02
        external = dict(available=False, reason='synthetic fixture has no External camera')
        return dict(completed=True, action_source=source, source_time_s=source_time,
                    command_times_s=command_times.tolist(), readback_times_s=readback_times.tolist(),
                    commands_position=commands.tolist(), measured_q_path=measured.tolist(),
                    retimed_predicted_q_path=predicted.tolist(),
                    q_path_residual=(measured - predicted).tolist(),
                    external_task_before=external, external_task_after=deepcopy(external))

    session = AlignmentSession(measure, simulated_motion, lower=[-100.] * 5 + [0.],
                               upper=[100.] * 6, gripper_tolerance=.25)
    records = []
    for joint in range(5):
        records.append(session.probe(joint, 1))
        records.append(session.return_from_probe())
    return dict(schema='v21-wrist-alignment-actual-pairs-v1', records=records,
                evidence='synthetic', physical_grasp_verified=False)


@pytest.fixture
def load_serialized(tmp_path):
    def load(data):
        path = tmp_path / 'synthetic-completed-alignment-pairs.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        return load_measured_wrist_response(path)

    return load


def test_loads_full_completed_session_receipts_and_fits_achieved_motion(receipt_data, load_serialized):
    records = receipt_data['records']
    assert len(records) == 10 and receipt_data['evidence'] == 'synthetic'
    assert {record['action_source'] for record in records} == {
        'visual_alignment_probe', 'visual_alignment_return'}
    for record in records:
        assert record['motion_completed'] is True
        assert record['physical_grasp_verified'] is False
        assert record['before']['body_truncated'] is False
        assert record['after']['body_truncated'] is False
        assert all(value is not None for value in record['model_response'].values())
        motion = record['model_response']
        np.testing.assert_allclose(motion['measured_q_path'][0], record['before']['q_at_image'])
        np.testing.assert_allclose(motion['measured_q_path'][-1], record['after']['q_at_image'])
        assert not np.allclose(motion['measured_q_path'][1:], motion['commands_position'])
    response = load_serialized(receipt_data)
    assert response.sample_count == 10
    np.testing.assert_allclose(response.jacobian, IMAGE_MAP, atol=1e-12)
    np.testing.assert_allclose(response.reference_q, SOURCE_Q)
    np.testing.assert_allclose(response.measured_support, np.full(5, 1.6), atol=1e-12)
    np.testing.assert_allclose(response.residual_rms, np.zeros(3), atol=1e-12)
    assert response.design_condition == pytest.approx(1.)


@pytest.mark.parametrize('completed', [False, None, 1])
def test_non_completed_or_non_boolean_motion_is_rejected(receipt_data, load_serialized, completed):
    receipt_data['records'][3]['motion_completed'] = completed
    with pytest.raises(ValueError, match='completed probes/returns'):
        load_serialized(receipt_data)


@pytest.mark.parametrize('source', ['visual_alignment_servo', 'raw_actor', None])
def test_other_action_sources_cannot_be_calibration_probes(receipt_data, load_serialized, source):
    receipt_data['records'][3]['action_source'] = source
    with pytest.raises(ValueError, match='completed probes/returns'):
        load_serialized(receipt_data)


@pytest.mark.parametrize('side,clock_case', [
    ('before', 'readback_before_image'), ('after', 'late_readback'),
    ('before', 'nan_image'), ('after', 'infinite_readback'),
])
def test_bad_image_readback_clock_pair_is_rejected(receipt_data, load_serialized, side, clock_case):
    row = receipt_data['records'][2][side]
    if clock_case == 'readback_before_image':
        row['q_time_s'] = row['image_time_s'] - .001
    elif clock_case == 'late_readback':
        row['q_time_s'] = row['image_time_s'] + .051
    elif clock_case == 'nan_image':
        row['image_time_s'] = float('nan')
    else:
        row['q_time_s'] = float('inf')
    with pytest.raises(ValueError, match='clocks'):
        load_serialized(receipt_data)


@pytest.mark.parametrize('offset', [0., -.1])
def test_observation_clocks_must_advance_across_motion(receipt_data, load_serialized, offset):
    record = receipt_data['records'][4]
    for key in ('image_time_s', 'q_time_s'):
        record['after'][key] = record['before'][key] + offset
    with pytest.raises(ValueError, match='advance actual time'):
        load_serialized(receipt_data)


@pytest.mark.parametrize('bad_path', ['source_mismatch', 'endpoint_mismatch', 'missing_q0', 'nan_interior'])
def test_measured_path_must_match_observations_and_have_six_finite_rows(receipt_data, load_serialized, bad_path):
    motion = receipt_data['records'][1]['model_response']
    path = motion['measured_q_path']
    if bad_path == 'source_mismatch':
        path[0][0] += .251
    elif bad_path == 'endpoint_mismatch':
        path[-1][0] += .501
    elif bad_path == 'missing_q0':
        motion['measured_q_path'] = path[1:]
    else:
        path[3][2] = float('nan')
    with pytest.raises(ValueError, match='actual motion readbacks'):
        load_serialized(receipt_data)


@pytest.mark.parametrize('field,value', [
    ('target_long_axis_px', [1., 0., 0.]),
    ('target_long_axis_px', [float('nan'), 0.]),
    ('jaw_fm_reference_px', [[92., 112.]]),
    ('jaw_fm_reference_px', [[92., 112.], [float('inf'), 112.]]),
])
def test_malformed_axis_or_jaw_array_is_rejected(receipt_data, load_serialized, field, value):
    receipt_data['records'][0]['before'][field] = value
    with pytest.raises(ValueError, match='invalid actual Wrist calibration observation'):
        load_serialized(receipt_data)


@pytest.mark.parametrize('geometry', ['zero_axis', 'collapsed_jaw', 'jaw_padding',
                                     'jaw_right_edge', 'target_padding', 'target_right_edge'])
def test_degenerate_or_out_of_frame_geometry_is_rejected(receipt_data, load_serialized, geometry):
    records = receipt_data['records']
    if geometry == 'zero_axis':
        # atan2(0, 0) happens to equal this sample's valid zero angle, so fit
        # residuals cannot substitute for checking a nondegenerate direction.
        records[0]['before']['target_long_axis_px'] = [0., 0.]
    elif geometry == 'collapsed_jaw':
        records[-1]['after']['jaw_fm_reference_px'] = [[112., 112.], [112., 112.]]
    elif geometry == 'jaw_padding':
        records[0]['before']['jaw_fm_reference_px'] = [[92., 27.], [132., 27.]]
    elif geometry == 'jaw_right_edge':
        records[-1]['after']['jaw_fm_reference_px'][1][0] = 224.
    else:
        # A constant offset leaves the otherwise valid fitted image response
        # unchanged; every observation must still lie in real image content.
        axis, shift = (1, -100.) if geometry == 'target_padding' else (0, 224.)
        for record in records:
            for side in ('before', 'after'):
                record[side]['target_center_px'][axis] += shift
    with pytest.raises(ValueError):
        load_serialized(receipt_data)


@pytest.mark.parametrize('side', ['before', 'after'])
def test_explicitly_recorded_truncated_wrist_target_is_rejected(receipt_data, load_serialized, side):
    # Check both saved Wrist observations, independently of External metadata.
    receipt_data['records'][3][side]['body_truncated'] = True
    with pytest.raises(ValueError, match='explicitly complete target body'):
        load_serialized(receipt_data)


@pytest.mark.parametrize('side', ['before', 'after'])
def test_missing_completeness_flag_cannot_establish_a_complete_target(receipt_data, load_serialized, side):
    assert receipt_data['records'][3][side].pop('body_truncated') is False
    with pytest.raises(ValueError, match='explicitly complete target body'):
        load_serialized(receipt_data)


@pytest.mark.parametrize('receipt_data', [True], indirect=True, ids=['synthetic-truncated-target'])
def test_truncated_target_survives_actual_session_serialization_and_is_rejected(receipt_data, load_serialized):
    # The True flag originated in measure()'s target record and passed through
    # alignment_observation and AlignmentSession's real observation_payload.
    # No saved row is patched to manufacture the serialization assertion.
    for record in receipt_data['records']:
        assert record['before']['body_truncated'] is True
        assert record['after']['body_truncated'] is True
    with pytest.raises(ValueError, match='explicitly complete target body'):
        load_serialized(receipt_data)


def test_replayed_records_cannot_turn_five_unique_probes_into_ten_samples(receipt_data, load_serialized):
    unique_probes = receipt_data['records'][::2]
    receipt_data['records'] = deepcopy(unique_probes + unique_probes)
    assert len(receipt_data['records']) == 10
    with pytest.raises(ValueError):
        load_serialized(receipt_data)


@pytest.mark.parametrize('problem', ['wrong_schema', 'too_few_pairs'])
def test_requires_actual_pair_schema_and_ten_completed_records(receipt_data, load_serialized, problem):
    if problem == 'wrong_schema':
        receipt_data['schema'] = 'predicted-wrist-pairs-v1'
    else:
        receipt_data['records'] = receipt_data['records'][:9]
    with pytest.raises(ValueError, match='actual completed observation pairs|at least ten'):
        load_serialized(receipt_data)
