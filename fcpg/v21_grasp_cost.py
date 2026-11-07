"""Two image-space costs on predicted actual states; LeWM remains goal-free.

The Wrist camera moves with the arm: the fitted response transports the fish
centre AND its long axis in the moving image. The existing same-view jaw
reference is held at the measured opening. This local model does not establish
depth/contact or permit closing, and unsupported forecasts must be rejected.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .v21_visual_alignment import AlignmentObservation, MeasuredImageResponse, fit_measured_response
from .v21_wrist_guard import require_directed_fg_forecast, require_wrist_path


def load_measured_wrist_response(path):
    """Fit saved completed AlignmentSession probe/return records, never goals.

    The explicit file binds the camera setup chosen by the operator. The live
    consumer also checks current pose, displacement, source time and view.
    """
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema') != 'v21-wrist-alignment-actual-pairs-v1':
        raise ValueError('Wrist calibration must contain actual completed observation pairs')
    pairs = []
    seen = set()
    for record in data.get('records', []):
        if (record.get('motion_completed') is not True or record.get('action_source') not in
                ('visual_alignment_probe', 'visual_alignment_return')):
            raise ValueError('Wrist response accepts completed probes/returns only')
        observations = []
        for name in ('before', 'after'):
            row = record[name]
            values = [np.array(row[key], np.float64) for key in
                ('q_at_image', 'target_center_px', 'target_long_axis_px', 'jaw_fm_reference_px')]
            for value, shape in zip(values, ((6,), (2,), (2,), (2, 2))):
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('invalid actual Wrist calibration observation')
            _, center, axis, jaw = values
            if row.get('body_truncated') is not False:
                raise ValueError('Wrist calibration requires an explicitly complete target body')
            if np.linalg.norm(axis) < 1e-8 or np.linalg.norm(jaw[1]-jaw[0]) < 1e-6:
                raise ValueError('Wrist calibration axis and jaw must be nondegenerate')
            pixels = np.vstack((center, jaw))
            if ((pixels < [0., 28.]) | (pixels >= [224., 196.])).any():
                raise ValueError('Wrist calibration geometry is outside real image content')
            image, readback = float(row['image_time_s']), float(row['q_time_s'])
            if not np.isfinite([image, readback]).all() or not 0 <= readback-image <= .05:
                raise ValueError('Wrist calibration clocks are not exposure-paired')
            observations.append(AlignmentObservation(*values, image, readback, False))
        identity = tuple(t for o in observations for t in (o.image_time_s, o.q_time_s))
        if identity in seen:
            raise ValueError('replayed Wrist calibration pair does not add an actual sample')
        seen.add(identity)
        measured = np.asarray(record['model_response']['measured_q_path'], np.float64)
        if (measured.shape != (6, 6) or not np.isfinite(measured).all()
                or np.max(np.abs(measured[0]-observations[0].q_position)) > .25
                or np.max(np.abs(measured[-1]-observations[1].q_position)) > .5):
            raise ValueError('Wrist calibration observations do not match actual motion readbacks')
        pairs.append(tuple(observations))
    if len(pairs) < 10:
        raise ValueError('Wrist calibration needs at least ten completed probe/return pairs')
    response = fit_measured_response(pairs, unmodelled_gripper_tolerance=.25)
    # Apply exactly the same quality checks as live scoring before installation.
    WristGraspCost(pairs[-1][1], response)
    return response


def line_error_rad(first, second):
    """Signed shortest angle between UNORIENTED lines, in [-pi/2, pi/2]."""
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    if (a.shape[-1:] != (2,) or b.shape[-1:] != (2,)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or (np.linalg.norm(a, axis=-1) < 1e-8).any()
            or (np.linalg.norm(b, axis=-1) < 1e-8).any()):
        raise ValueError('line directions must be finite and nondegenerate')
    angle = np.arctan2(b[..., 1], b[..., 0]) - np.arctan2(a[..., 1], a[..., 0])
    return .5 * np.arctan2(np.sin(2 * angle), np.cos(2 * angle))


def distance_alignment_scores(center, jaw_pair, body_long_axis):
    """Blue is perpendicular to body-long-axis; reversing F/G changes no cost."""
    center = np.asarray(center, np.float64)
    pair = np.asarray(jaw_pair, np.float64)
    long = np.asarray(body_long_axis, np.float64)
    if (center.shape[-1:] != (2,) or pair.shape[-2:] != (2, 2)
            or long.shape[-1:] != (2,) or not np.isfinite(center).all()
            or not np.isfinite(pair).all()):
        raise ValueError('finite same-view centre, jaw pair and body axis required')
    white = pair[..., 1, :] - pair[..., 0, :]
    blue = np.stack((-long[..., 1], long[..., 0]), axis=-1)
    distance = np.linalg.norm(center - pair.mean(axis=-2), axis=-1)
    signed = line_error_rad(white, blue)
    return dict(distance_px=distance, angle_error_deg=np.abs(np.degrees(signed)),
                signed_angle_error_deg=np.degrees(signed),
                distance_score=distance / 224., alignment_score=np.abs(signed) / (np.pi / 2.))


@dataclass(frozen=True)
class WristGraspCost:
    observation: AlignmentObservation
    response: MeasuredImageResponse
    distance_weight: float = 1.
    alignment_weight: float = 1.
    alignment_tolerance_deg: float = 15.

    def __post_init__(self):
        if not isinstance(self.observation, AlignmentObservation) or not isinstance(self.response, MeasuredImageResponse):
            raise TypeError('grasp cost needs exposure-paired observation and measured image response')
        r = self.response
        for value, shape in ((r.jacobian, (3, 5)), (r.reference_q, (6,)),
                             (r.measured_support, (5,)), (r.observed_radius, (5,)),
                             (r.residual_rms, (3,))):
            if np.shape(value) != shape or not np.isfinite(value).all():
                raise ValueError('invalid measured Wrist response')
        if (r.sample_count < 10 or not np.isfinite(r.design_condition)
                or not 0 < r.design_condition <= 100 or (r.residual_rms < 0).any()
                or (r.residual_rms[:2] > 4.).any() or r.residual_rms[2] > np.radians(5)
                or (r.measured_support <= 0).any() or (r.observed_radius <= 0).any()):
            raise ValueError('measured Wrist response lacks support or exceeds fit quality limits')
        if (np.linalg.norm(r.jacobian[:2]) < 1e-6 or np.linalg.norm(r.jacobian[2]) < 1e-6):
            raise ValueError('measured response must identify both distance and orientation')
        if (not np.isfinite([self.distance_weight, self.alignment_weight, self.alignment_tolerance_deg]).all()
                or min(self.distance_weight, self.alignment_weight) <= 0
                or not 0 < self.alignment_tolerance_deg < 90):
            raise ValueError('positive costs and an acute alignment tolerance required')
        self._check_support(self.observation.q_position[None])

    def _check_support(self, path):
        r, o = self.response, self.observation
        if (np.abs(path[:, :5] - r.reference_q[:5]) > r.observed_radius + 1e-6).any():
            raise ValueError('grasp forecast leaves measured pose support')
        if (np.abs(path[:, :5] - o.q_position[:5]) > r.measured_support + 1e-6).any():
            raise ValueError('grasp forecast exceeds local measured displacement')
        if (np.abs(path[:, 5] - r.reference_q[5]) > .25).any() or (np.abs(path[:, 5] - o.q_position[5]) > .25).any():
            raise ValueError('open-jaw response does not support gripper motion')

    def evaluate(self, predicted_q):
        path = np.asarray(predicted_q, np.float64)
        if path.shape not in ((6, 6), (41, 6)) or not np.isfinite(path).all():
            raise ValueError('grasp cost requires LeWM q0..q5 or q0..q40')
        self._check_support(path)
        require_wrist_path(self.observation.q_position, path,
                           max_delta=None, max_step=None)
        o = self.observation
        features = o.scene_features + (path[:, :5] - o.q_position[:5]) @ self.response.jacobian.T
        centres = features[:, :2]
        if ((centres < [0., 28.]) | (centres >= [224., 196.])).any():
            raise ValueError('predicted target leaves the real image content')
        axes = np.stack((np.cos(features[:, 2]), np.sin(features[:, 2])), axis=-1)
        # F/G are physical endpoints, not an unordered line. Select the blue
        # short-axis sign once at the source exposure and carry it through all
        # predicted target axes. This is the real scoring guard: line_error_rad
        # remains available for standalone unoriented-line diagnostics, but it
        # cannot turn a half-turn candidate into a low alignment cost here.
        jaw = np.asarray(o.jaw_fm_px, np.float64)
        directed_guard = require_directed_fg_forecast(
            jaw[0], jaw[1], o.target_axis_px,
            np.repeat(jaw[0][None], len(path), axis=0),
            np.repeat(jaw[1][None], len(path), axis=0), axes)
        forecast_fg = np.asarray(directed_guard['forecast_directed_f_to_g'], np.float64)
        forecast_blue = np.asarray(directed_guard['forecast_blue_short_axis'], np.float64)
        signed = np.arctan2(
            forecast_fg[:, 0] * forecast_blue[:, 1]
            - forecast_fg[:, 1] * forecast_blue[:, 0],
            np.sum(forecast_fg * forecast_blue, axis=-1))
        distance = np.linalg.norm(centres - jaw.mean(axis=0), axis=-1)
        scores = dict(
            distance_px=distance,
            angle_error_deg=np.abs(np.degrees(signed)),
            signed_angle_error_deg=np.degrees(signed),
            distance_score=distance / 224.,
            alignment_score=np.abs(signed) / (np.pi / 2.))
        scores.update(angle_error_deg=np.abs(np.degrees(signed)),
                      signed_angle_error_deg=np.degrees(signed),
                      alignment_score=np.abs(signed) / (np.pi / 2))
        # First executed block and long lookahead have equal weight. A delayed
        # improvement cannot conceal a bad first five responses (planner gate).
        weight = np.ones(len(path) - 1) / (len(path) - 1)
        if len(path) == 41:
            weight[:5] = .5 / 5
            weight[5:] = .5 / 35
        distance = float(weight @ scores['distance_score'][1:]) * self.distance_weight
        alignment = float(weight @ scores['alignment_score'][1:]) * self.alignment_weight
        return dict(cost=distance + alignment, distance_cost=distance, alignment_cost=alignment,
                    **{key: np.asarray(value).tolist() for key, value in scores.items()},
                    predicted_centres_px=centres.tolist(),
                    predicted_target_axes_px=axes.tolist(),
                    directed_fg_guard=directed_guard,
                    directed_fg_protection=True,
                    prediction_source='lewm_actual_q_via_measured_moving_wrist_response')

    def receipt(self):
        o, r = self.observation, self.response
        jaw = np.asarray(o.jaw_fm_px, np.float64)
        directed_guard = require_directed_fg_forecast(
            jaw[0], jaw[1], o.target_axis_px,
            jaw[0][None], jaw[1][None], o.target_axis_px[None])
        source_blue = np.asarray(directed_guard['source_blue_short_axis'], np.float64)
        source_fg = np.asarray(directed_guard['source_directed_f_to_g'], np.float64)
        source_signed = np.arctan2(
            source_fg[0] * source_blue[1] - source_fg[1] * source_blue[0],
            np.dot(source_fg, source_blue))
        return dict(available=True, source='measured_wrist_distance_alignment',
                    score_terms=['distance', 'blue_white_alignment'],
                    distance_weight=self.distance_weight, alignment_weight=self.alignment_weight,
                    angle_period_deg=180, angle_error_range_deg=[0, 90],
                    jaw_endpoint_order=['fixed_finger_tip', 'moving_finger_tip'],
                    jaw_endpoint_semantics=dict(F='fixed_finger_tip', G='moving_finger_tip'),
                    angle_branch='nearest_source_F_to_G', half_turn_allowed=False,
                    directed_fg_guard_applied=True, signed_angle_continuity=True,
                    modulo_pi_candidate_scoring=False, wrist_angle_boundary_deg=90,
                    blue_axis='perpendicular_to_fish_long_axis',
                    alignment_tolerance_deg=self.alignment_tolerance_deg,
                    image_time_s=o.image_time_s, capture_q=o.q_position.tolist(),
                    source_distance_px=float(np.linalg.norm(o.target_center_px - jaw.mean(axis=0))),
                    source_angle_error_deg=float(abs(np.degrees(source_signed))),
                    target_center_px=o.target_center_px.tolist(),
                    target_long_axis_px=o.target_axis_px.tolist(), jaw_reference_px=o.jaw_fm_px.tolist(),
                    jaw_reference_source='existing_wrist_fk_reference',
                    jacobian=r.jacobian.tolist(), reference_q=r.reference_q.tolist(),
                    measured_support=r.measured_support.tolist(), observed_radius=r.observed_radius.tolist(),
                    residual_rms=r.residual_rms.tolist(), sample_count=r.sample_count,
                    design_condition=r.design_condition, gripper_held=True,
                    learned_pixel_prediction=False, physical_alignment_verified=False,
                    physical_grasp_verified=False)
