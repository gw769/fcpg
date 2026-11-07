"""Measured local image task cost; never a learned pixel prediction or FK label."""
from dataclasses import dataclass
import numpy as np
import torch

from .v21_column_dino_transport import ColumnDinoFGSample, ColumnDinoFGTransport, bind_column_dino_fg


@dataclass(frozen=True)
class MeasuredVisualTaskCost:
    transport: ColumnDinoFGTransport
    target_px: tuple
    image_time_s: float
    weight: float = 1.0

    def __post_init__(self):
        if not isinstance(self.transport, ColumnDinoFGTransport):
            raise TypeError('visual task cost requires measured Column response transport')
        target = np.asarray(self.target_px, dtype=np.float64)
        if (target.shape != (2,) or not np.isfinite(target).all()
                or (target < 0).any() or (target >= 224).any()):
            raise ValueError('visual task target must be finite image coordinates')
        if not np.isfinite([self.image_time_s, self.weight]).all() or self.weight <= 0:
            raise ValueError('visual task clock and weight must be finite and weight positive')
        if self.transport.residual_rms_px > 4 or self.transport.design_condition > 100:
            raise ValueError('visual task transport exceeds measured fit quality limits')
        if np.linalg.norm(self.transport.jacobian_px_per_joint[:, :5]) < 1e-6:
            raise ValueError('visual task transport has no candidate-sensitive arm response')
        object.__setattr__(self, 'target_px', tuple(float(x) for x in target))

    def evaluate(self, path):
        """Score all five LeWM responses; forbid extrapolation/clamped FK claims."""
        q = np.asarray(path, dtype=np.float64)
        if q.shape != (6, 6) or not np.isfinite(q).all():
            raise ValueError('visual task requires a finite predicted q0..q5 path')
        tr = self.transport
        if ((q[:, :5] < tr.lower_q-1e-9) | (q[:, :5] > tr.upper_q+1e-9)).any():
            raise ValueError('visual task prediction leaves measured response support')
        if tr.jacobian_px_per_joint.shape[1] == 6:
            if ((q[:, 5] < tr.lower_q6-1e-9) | (q[:, 5] > tr.upper_q6+1e-9)).any():
                raise ValueError('visual task gripper leaves measured response support')
        elif np.max(np.abs(q[:, 5]-tr.capture_q[5])) > .25:
            raise ValueError('visual task has no measured gripper response support')
        pixels = np.stack([tr.project(row) for row in q])
        distances = np.linalg.norm(pixels-np.asarray(self.target_px), axis=1)
        # Keep the existing response term intact. Penalize distance and retreat,
        # using every response sample rather than a commanded endpoint proxy.
        retreat = np.maximum(distances[1:]-distances[0], 0.)
        cost = float(self.weight*np.mean((distances[1:]+retreat)/224.))
        return dict(cost=cost, predicted_midpoints_px=pixels.tolist(),
                    distances_px=distances.tolist(), prediction_source='lewm_q_via_measured_local_image_response')

    def receipt(self):
        tr = self.transport
        return dict(available=True, source='measured_column_response', target_px=list(self.target_px),
            image_time_s=self.image_time_s, weight=self.weight, sample_count=tr.sample_count,
            residual_rms_px=tr.residual_rms_px, design_condition=tr.design_condition,
            capture_q=tr.capture_q.tolist(), observed_midpoint_px=tr.observed_midpoint_px.tolist(),
            jacobian_px_per_joint=tr.jacobian_px_per_joint.tolist(), lower_q=tr.lower_q.tolist(),
            upper_q=tr.upper_q.tolist(), lower_q6=tr.lower_q6, upper_q6=tr.upper_q6,
            physical_task_success_verified=False, learned_pixel_prediction=False, fk_used=False)

    @classmethod
    def from_receipt(cls, record):
        if record.get('source') != 'measured_column_response' or record.get('fk_used') is not False:
            raise ValueError('visual task replay requires measured response provenance')
        transport = ColumnDinoFGTransport(**{key: record[key] for key in (
            'jacobian_px_per_joint', 'capture_q', 'observed_midpoint_px', 'lower_q', 'upper_q',
            'residual_rms_px', 'design_condition', 'sample_count', 'lower_q6', 'upper_q6')})
        return cls(transport, tuple(record['target_px']), record['image_time_s'], record['weight'])


class ImageAnchoredVisualCost:
    """Fixed-camera cost: transport F/G while the observed fish stays put.

    This is deliberately NOT an eye-in-hand cost. A moving Wrist must
    transport the observed scene, not keep its target pixel/axis constant.
    """

    def __init__(self, project_fg, image_fg, q0, target_px, *, view, blue_axis_px=None):
        if view != 'column':
            raise ValueError('a moving Wrist requires MeasuredWristVisualCost; fixed target pixels are invalid')
        self.project_fg = project_fg
        self.image_fg = np.asarray(image_fg, np.float64)
        self.q0 = np.asarray(q0, np.float64).reshape(6)
        self.target_px = np.asarray(target_px, np.float64).reshape(2)
        self.view = view
        blue = None
        if blue_axis_px is not None:
            blue = np.asarray(blue_axis_px, np.float64).reshape(-1)
            if blue.shape == (2,) and np.isfinite(blue).all() and float(np.linalg.norm(blue)) > 1e-8:
                blue = blue / float(np.linalg.norm(blue))
            else:
                blue = None
        self.blue_axis = blue
        if self.image_fg.shape != (2, 2) or not np.isfinite(self.image_fg).all():
            raise ValueError('image F/G required')
        if not np.isfinite(self.target_px).all() or not np.isfinite(self.q0).all():
            raise ValueError('fish pixel and exposure q required')
        start = np.asarray(project_fg(self.q0), np.float64)
        if start.shape != (2, 2) or not np.isfinite(start).all():
            raise ValueError('cannot project the image F/G pose')
        self.shift = self.image_fg - start

    def evaluate(self, path):
        q = np.asarray(path, np.float64)
        if q.shape != (6, 6) or not np.isfinite(q).all():
            raise ValueError('visual task requires a finite predicted q0..q5 path')
        pairs = []
        for row in q:
            pair = np.asarray(self.project_fg(row), np.float64) + self.shift
            pairs.append(pair)
        pairs = np.stack(pairs)
        mids = pairs.mean(1)
        distances = np.linalg.norm(mids - self.target_px, axis=1)
        jaw = pairs[:, 1] - pairs[:, 0]
        angles = np.zeros(len(q))
        length = np.linalg.norm(jaw, axis=1)
        ok = length > 1e-6
        if self.blue_axis is not None and ok.any():
            # White jaw line versus the blue grasp line. Distance 0 does not
            # cancel this penalty: a jaw sitting on the fish but crossed to
            # the blue line still loses.
            unit = jaw[ok] / length[ok, None]
            blue = self.blue_axis
            cross = unit[:, 0] * blue[1] - unit[:, 1] * blue[0]
            dot = unit @ blue
            turned = np.abs(np.arctan2(cross, dot))
            angles[ok] = np.minimum(turned, np.pi - turned)
        else:
            toward = self.target_px - mids
            usable = ok & (np.linalg.norm(toward, axis=1) > 1.)
            if usable.any():
                delta = (np.arctan2(toward[usable, 1], toward[usable, 0])
                         - np.arctan2(jaw[usable, 1], jaw[usable, 0]))
                angles[usable] = np.abs(0.5 * np.arctan2(np.sin(2 * delta), np.cos(2 * delta)))
        # CEM minimizes this. 32 px of miss equals a jaw crossed 90° to the
        # blue line, so a far fish outranks spinning in place. Distance 0 is
        # still the bottom of the distance penalty, and the angle penalty
        # stays on there.
        cost = float(np.mean(distances[1:] / 32. + angles[1:] / (np.pi / 2.)))
        return dict(cost=cost, predicted_midpoints_px=mids.tolist(),
                    distances_px=distances.tolist(), angles_rad=angles.tolist(),
                    prediction_source='lewm_q_anchored_to_image_fg')

    def receipt(self):
        return dict(available=True, source='image_fg_anchored', view=self.view,
                    target_px=self.target_px.tolist(), image_fg_px=self.image_fg.tolist(),
                    learned_pixel_prediction=False, physical_task_success_verified=False)


class ColumnDropVisualCost(ImageAnchoredVisualCost):
    """DINO box pixel goal, with a nominal vertical drop onto its opening plane.

    Reuses the existing forward projection and current image F/G anchor. No
    inverse camera mapping or taught box XY is needed. Plane height is the
    operator's approximate box height, not measured metric depth.
    """
    def __init__(self, project_fg, image_fg, q0, target_px, *, midpoint,
                 z_pixel_per_m, opening_z_m, image_time_s, target_memory=None):
        super().__init__(project_fg, image_fg, q0, target_px, view='column')
        self.midpoint = midpoint
        self.z_pixel_per_m = np.asarray(z_pixel_per_m, np.float64).reshape(2)
        self.opening_z_m, self.image_time_s = float(opening_z_m), float(image_time_s)
        self.target_memory = None if target_memory is None else dict(target_memory)
        if not np.isfinite(np.r_[self.z_pixel_per_m, self.opening_z_m, self.image_time_s]).all():
            raise ValueError('finite nominal box opening projection required')

    def evaluate(self, path):
        item = super().evaluate(path)
        heights = np.asarray([self.midpoint(q)[2] for q in path])
        jaw = np.asarray(item['predicted_midpoints_px'])
        landing = jaw + (self.opening_z_m-heights)[:, None] * self.z_pixel_per_m
        distances = np.linalg.norm(landing-self.target_px, axis=1)
        return dict(cost=float(np.mean(distances[1:]/32.)), distances_px=distances.tolist(),
                    angles_rad=np.zeros(len(path)).tolist(),
                    predicted_midpoints_px=jaw.tolist(), predicted_drop_pixels=landing.tolist(),
                    prediction_source='lewm_q_image_anchored_nominal_vertical_drop')

    def receipt(self):
        return dict(super().receipt(), source=('current_dino_column_drop_zone' if self.target_memory is None
                                              else self.target_memory['source']),
                    target_role='drop_zone', image_time_s=self.image_time_s,
                    target_memory=self.target_memory,
                    box_xy_taught_required=False, nominal_opening_z_m=self.opening_z_m,
                    vertical_projection_is_metric_calibration=False)


class MeasuredWristVisualCost:
    """Adapt the existing measured moving-camera cost to five-step task CEM.

    LeWM predicts actual joints. The measured image response then moves the
    fish centre AND blue axis in the Wrist image. No motor commands, IK,
    fixed-world jaw-anchor extrapolation, or hand-written left/right signs.
    """
    view = 'wrist'

    def __init__(self, observation, response):
        from .v21_grasp_cost import WristGraspCost
        self.task = WristGraspCost(observation, response)
        self.image_fg = np.asarray(observation.jaw_fm_px, np.float64)
        self.target_px = np.asarray(observation.target_center_px, np.float64)

    def evaluate(self, path):
        item = self.task.evaluate(path)
        distances = np.asarray(item['distance_px'], np.float64)
        angles = np.radians(item['angle_error_deg'])
        return dict(cost=float(np.mean(distances[1:] / 32. + angles[1:] / (np.pi / 2.))),
                    distances_px=distances.tolist(), angles_rad=angles.tolist(),
                    predicted_midpoints_px=np.repeat(self.image_fg.mean(0)[None], len(distances), axis=0).tolist(),
                    predicted_target_px=item['predicted_centres_px'],
                    predicted_target_axes_px=item['predicted_target_axes_px'],
                    prediction_source=item['prediction_source'])

    def receipt(self):
        return dict(self.task.receipt(), view=self.view, source='measured_moving_wrist',
                    target_px=self.target_px.tolist(), image_fg_px=self.image_fg.tolist(),
                    target_motion_model='actual_joint_to_observed_scene_response',
                    target_pixel_fixed=False, target_axis_fixed=False)


class MovingWristPlaneCost:
    """Re-anchor the observed fish, then move the CAMERA with LeWM future q.

    Uses the existing hand-eye/FOV estimate and nominal table plane. This is
    approximate geometric scoring, not new calibration or a learned pixel head.
    The jaw reference rides with the Wrist; the fish centre AND axis move.
    """
    view = 'wrist'

    def __init__(self, kinematics, camera, q_at_image, image_fg, target_px,
                 axis_px, *, plane_z_m, image_time_s):
        if camera.get('reference_frame') != 'wrist-link':
            raise ValueError('Wrist camera must be expressed in its moving link')
        self.kinematics = kinematics
        self.K = np.asarray(camera['intrinsic'], np.float64)
        self.E = np.asarray(camera['camera_from_reference'], np.float64)
        self.q0 = np.asarray(q_at_image, np.float64).reshape(6)
        self.image_fg = np.asarray(image_fg, np.float64).reshape(2, 2)
        self.target_px = np.asarray(target_px, np.float64).reshape(2)
        axis = np.asarray(axis_px, np.float64).reshape(2)
        self.plane_z_m, self.image_time_s = float(plane_z_m), float(image_time_s)
        self.intrinsic_status = camera.get('intrinsic_status', 'existing estimate')
        if (self.K.shape != (3, 3) or self.E.shape != (4, 4)
                or not np.isfinite(np.r_[self.K.ravel(), self.E.ravel(), self.q0,
                    self.image_fg.ravel(), self.target_px, axis, plane_z_m, image_time_s]).all()
                or np.linalg.norm(axis) < 1e-6 or np.linalg.norm(self.image_fg[1]-self.image_fg[0]) < 1e-6):
            raise ValueError('finite current Wrist fish, jaw and camera geometry required')
        axis = axis / np.linalg.norm(axis)
        pixels = np.array([self.target_px, self.target_px-8.*axis, self.target_px+8.*axis])
        pose = kinematics.forward_gripper_poses(self.q0[None])[0]
        world_from_camera = pose @ np.linalg.inv(self.E)
        rays = np.linalg.solve(self.K, np.c_[pixels, np.ones(3)].T).T
        rays = rays @ world_from_camera[:3, :3].T
        depth = (self.plane_z_m-world_from_camera[2, 3]) / rays[:, 2]
        if not np.isfinite(depth).all() or (depth <= 0.).any():
            raise ValueError('current Wrist fish does not intersect the nominal table in front of camera')
        self.world_points = world_from_camera[:3, 3] + depth[:, None]*rays

    def evaluate(self, path):
        path = np.asarray(path, np.float64)
        if path.shape != (6, 6) or not np.isfinite(path).all():
            raise ValueError('moving Wrist score needs six LeWM states')
        poses = self.kinematics.forward_gripper_poses(path)
        local = np.matmul(self.world_points[None]-poses[:, None, :3, 3], poses[:, :3, :3])
        camera = local @ self.E[:3, :3].T + self.E[:3, 3]
        if (camera[:, :, 2] <= 0.).any():
            raise ValueError('predicted Wrist fish lies behind the camera')
        homogeneous = camera @ self.K.T
        pixels = homogeneous[:, :, :2] / homogeneous[:, :, 2:3]
        centres = pixels[:, 0]
        axes = pixels[:, 2]-pixels[:, 1]
        angles = np.radians([abs(signed_blue_white_angle_deg(self.image_fg, axis)) for axis in axes])
        mids = np.repeat(self.image_fg.mean(0)[None], 6, axis=0)
        distances = np.linalg.norm(centres-mids, axis=1)
        return dict(cost=float(np.mean(distances[1:]/32.)), distances_px=distances.tolist(),
            angles_rad=angles.tolist(), predicted_midpoints_px=mids.tolist(),
            predicted_target_px=centres.tolist(), predicted_target_axes_px=axes.tolist(),
            prediction_source='lewm_q_moving_wrist_existing_pinhole_nominal_plane')

    def receipt(self):
        return dict(available=True, view=self.view, source='current_image_moving_wrist_nominal_plane',
            image_time_s=self.image_time_s, target_px=self.target_px.tolist(),
            image_fg_px=self.image_fg.tolist(), q_at_image=self.q0.tolist(),
            nominal_plane_z_m=self.plane_z_m, intrinsic_status=self.intrinsic_status,
            target_pixel_fixed=False, target_axis_fixed=False,
            approximate_geometry=True, new_calibration_performed=False,
            physical_task_success_verified=False)


def signed_blue_white_angle_deg(pair, axis):
    """Signed angle from the white jaw line to the blue grasp line, in [-90, 90]."""
    if pair is None or axis is None:
        return None
    pair = np.asarray(pair, np.float64).reshape(-1)
    axis = np.asarray(axis, np.float64).reshape(-1)
    if pair.shape != (4,) or axis.shape != (2,) or not np.isfinite(np.r_[pair, axis]).all():
        return None
    jaw = pair.reshape(2, 2)
    jaw = jaw[1] - jaw[0]
    if float(np.linalg.norm(jaw)) < 1e-6 or float(np.linalg.norm(axis)) < 1e-8:
        return None
    jaw = jaw / float(np.linalg.norm(jaw))
    axis = axis / float(np.linalg.norm(axis))
    blue = np.array((-axis[1], axis[0]), np.float64)
    angle = float(np.degrees(np.arctan2(jaw[0] * blue[1] - jaw[1] * blue[0], float(np.dot(jaw, blue)))))
    if angle > 90.:
        angle -= 180.
    elif angle < -90.:
        angle += 180.
    return angle


class BothViewsVisualCost:
    """Column distance plus wrist angle. No other visual penalty.

    A view whose target is the fingertip itself is not included: that residual
    would be zero without the fish being there.
    """

    def __init__(self, parts, wrist_dx_px=None, wrist_angle_deg=None, reach_dj23=None, wrist_dist_px=None):
        if not parts or any(not isinstance(part, (ImageAnchoredVisualCost, MeasuredWristVisualCost, MovingWristPlaneCost)) for part in parts):
            raise TypeError('both-view score needs image-anchored view costs')
        self.parts = tuple(parts)
        self.wrist_dx_px = None if wrist_dx_px is None else float(wrist_dx_px)
        self.wrist_angle_deg = None if wrist_angle_deg is None else float(wrist_angle_deg)
        self.wrist_dist_px = None if wrist_dist_px is None else float(wrist_dist_px)
        reach = None if reach_dj23 is None else np.asarray(reach_dj23, np.float64).reshape(-1)
        if reach is not None and (reach.shape != (2,) or not np.isfinite(reach).all()):
            reach = None
        self.reach_dj23 = reach

    def evaluate(self, path):
        results = [(part, part.evaluate(path)) for part in self.parts]
        wrist_available = any(part.view == 'wrist' for part in self.parts)
        cost = 0.
        column_distance = None
        wrist_angle = None
        mids = []
        q = np.asarray(path, np.float64)
        # +J1 is to the right. Centroid right of M (dx>0) must increase J1.
        # J5 only rolls once the wrist is already near the centroid.
        near = self.wrist_dist_px is not None and self.wrist_dist_px <= 40.
        if (not near and self.wrist_dx_px is not None and abs(self.wrist_dx_px) > 8.
                and q.ndim == 2 and q.shape[1] == 6 and q.shape[0] >= 2):
            dj1 = q[1:, 0] - q[0, 0]
            wrong = np.maximum(-dj1, 0.) if self.wrist_dx_px > 0 else np.maximum(dj1, 0.)
            cost += float(np.mean(wrong))
        # Wrist camera rolls with J5, so the white line stays put and the blue
        # line turns. Measured gain is positive: +J5 increases this signed angle.
        if (near and self.wrist_angle_deg is not None and abs(self.wrist_angle_deg) > 8.
                and q.ndim == 2 and q.shape[1] == 6 and q.shape[0] >= 2):
            dj5 = q[1:, 4] - q[0, 4]
            wrong = np.maximum(dj5, 0.) if self.wrist_angle_deg > 0 else np.maximum(-dj5, 0.)
            cost += float(np.mean(wrong))
        # Wrist image distance does not move under an eye-in-hand projection,
        # so rank the reach joints directly. A step against the fish costs
        # as many degrees as it travels.
        if (self.reach_dj23 is not None
                and q.ndim == 2 and q.shape[1] == 6 and q.shape[0] >= 2):
            moved = q[1:, 1:3] - q[0, 1:3]
            against = np.maximum(-np.sign(self.reach_dj23) * moved, 0.)
            active = np.abs(self.reach_dj23) > 0.2
            if bool(active.any()):
                cost += float(np.mean(against[:, active]))
        for part, item in results:
            mids.extend(item['predicted_midpoints_px'])
            if part.view == 'column':
                dist = np.asarray(item['distances_px'], np.float64).reshape(-1)
                if dist.size > 1:
                    column_distance = dist
                    # Hand fine alignment to wrist only when that view actually
                    # supplies a target; losing wrist must not erase all distance.
                    if float(dist[0]) > 40. or not wrist_available:
                        cost += float(np.mean(dist[1:] / 32.))
            elif part.view == 'wrist':
                # M to the object's centre of mass, plus the jaw angle.
                # Alignment is not the only penalty.
                dist = np.asarray(item['distances_px'], np.float64).reshape(-1)
                if dist.size > 1:
                    cost += float(np.mean(dist[1:] / 32.))
                ang = np.asarray(item['angles_rad'], np.float64).reshape(-1)
                if ang.size > 1:
                    wrist_angle = ang
                    cost += float(np.mean(ang[1:] / (np.pi / 2.)))
        return dict(cost=float(cost), predicted_midpoints_px=mids,
                    distances_px=([] if column_distance is None else column_distance.tolist()),
                    angles_rad=([] if wrist_angle is None else wrist_angle.tolist()),
                    views=[part.view for part, _item in results],
                    prediction_source='column_distance_plus_wrist_angle')

    def receipt(self):
        return dict(available=True, source='both_views_distance_plus_angle',
                    views=[part.receipt() for part in self.parts],
                    learned_pixel_prediction=False, physical_task_success_verified=False)


def visual_task_precondition(response, state):
    """Check optional fit/state before accessing any display or geometry object."""
    if type(response) is not dict:
        return 'measured_six_joint_response_unavailable'
    try:
        jacobian = np.asarray(response.get('jacobian_px_per_joint'), dtype=np.float64)
        if jacobian.shape != (2, 6):
            return 'measured_six_joint_response_required'
        lower = np.asarray(response['lower_q'], dtype=np.float64)
        upper = np.asarray(response['upper_q'], dtype=np.float64)
        quality = np.asarray([response['residual_rms_px'], response['design_condition'],
                              response['lower_q6'], response['upper_q6']], dtype=np.float64)
        if (not np.isfinite(jacobian).all() or lower.shape != (5,) or upper.shape != (5,)
                or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or (upper < lower).any() or not np.isfinite(quality).all()
                or not 0 <= quality[0] <= 4 or not 0 < quality[1] <= 100
                or quality[3] < quality[2] or type(response['sample_count']) is not int
                or response['sample_count'] < 10):
            return 'measured_six_joint_response_invalid'
        cue = getattr(state, 'object_relative', None)
        if (not isinstance(cue, torch.Tensor) or cue.shape != (1, 8)
                or not bool(torch.isfinite(cue).all()) or float(cue[0, 0]) != 1.):
            return 'actual_actor_object_token_unavailable'
        memory = getattr(state, 'memory', None)
        if (memory is None or memory.image_times.shape != (1, 2, 3)
                or memory.q_at_image.shape != (1, 2, 3, 6)
                or memory.valid.shape != (1, 2, 3) or not bool(memory.valid[0, 1, -1])
                or not bool(torch.isfinite(memory.q_at_image[0, 1, -1]).all())
                or not bool(torch.isfinite(memory.image_times[0, 1, -1]))):
            return 'actual_column_source_memory_unavailable'
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return 'measured_visual_task_precondition_invalid'
    return None


def bind_visual_task_cost(response, tip, target, state, *, actor_commands=None):
    """Bind actual same-exposure measurements, never an overlay/FK proposal."""
    if (not isinstance(tip, dict) or not tip.get('available') or tip.get('observed') is not True
            or tip.get('fk_based') or tip.get('source_contract_valid') is not True):
        raise ValueError('visual task needs an observed Column jaw measurement')
    if (not isinstance(target, dict) or not target.get('visible')
            or target.get('source') != 'operator_colour_body' or target.get('body_truncated')):
        raise ValueError('visual task needs a complete measured fish body')
    stamp = float(tip['image_time_s'])
    if not np.isfinite(stamp) or stamp != float(target['image_time_s']):
        raise ValueError('visual task target and jaw need the same exposure')
    image_stamp = float(state.memory.image_times[0, 1, -1])
    capture_q = np.asarray(tip['q_at_image'], dtype=np.float64)
    actual_capture_q = state.memory.q_at_image[0, 1, -1].detach().cpu().numpy()
    if stamp != image_stamp or not np.allclose(capture_q, actual_capture_q, rtol=0, atol=1e-5):
        raise ValueError('visual task exposure differs from FCPG source memory')
    cue = state.object_relative
    if cue is None or cue.shape != (1, 8) or not bool(cue.isfinite().all()) or float(cue[0, 0]) != 1.:
        raise ValueError('visual task requires an available actual Actor object token')
    sample = ColumnDinoFGSample(capture_q, tip['m_px'], stamp)
    transport = bind_column_dino_fg(response, sample)
    if actor_commands is not None and transport.jacobian_px_per_joint.shape[1] == 5:
        commands = actor_commands.detach().cpu().numpy()
        if commands.shape != (1, 5, 6) or not np.isfinite(commands).all():
            raise ValueError('visual task requires finite actual Actor commands')
        if np.max(np.abs(commands[0, :, 5]-capture_q[5])) > .25:
            raise ValueError('visual task unavailable: Actor changes J6 outside held-gripper calibration; '
                             'a measured six-joint response or measured midpoint-invariance contract is required')
        raise ValueError('live hybrid visual task requires a measured six-joint response fit')
    return MeasuredVisualTaskCost(transport, tuple(target['cog_px']), stamp)
