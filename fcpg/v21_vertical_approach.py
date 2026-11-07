"""Offline, bounded geometry proposals for supervised vertical approach.

This is engineering assistance, not Actor output, a contact detector, or a
hardware safety certificate.  The caller must audit the returned five-command
path with the established joint/clearance checks and execute it through the
existing transaction bridge.  All joints use ``lerobot_calibrated_position``;
Cartesian distances use mm.  CAD geometry is an estimate, not physical truth.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .so101_kinematics import normalized_positions_to_mujoco_qpos


@dataclass(frozen=True)
class ApproachProposal:
    accepted: bool
    commands_position: np.ndarray | None
    diagnostic: dict[str, Any]


def approach_geometry(kinematics, q_position) -> dict[str, Any]:
    """Read the official gripper-body -> gripperframe axis in the model world.

    No arbitrary column of a rotation matrix is assumed to be the tool axis.
    The world convention matches ``forward_anchors`` and the live table gate.
    J5/gripper geometry can give a small lateral tilt; it is reported rather
    than silently treating radial-plane pitch as the full 3-D tilt.
    """
    q = np.asarray(q_position, dtype=np.float64)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("q_position must be a finite calibrated six-vector")
    with kinematics._forward_lock:
        kinematics._set_state(normalized_positions_to_mujoco_qpos(
            q, kinematics.all_joint_ranges))
        axis = (kinematics.data.site_xpos[kinematics.gripperframe_site_id]
                - kinematics.data.xpos[kinematics.gripper_body_id]).copy()
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm < 1e-9:
            raise ValueError("gripper-body to gripperframe axis is degenerate")
        axis /= norm
        anchors = np.asarray(kinematics.forward_anchors(q), dtype=np.float64)
    if anchors.shape != (3, 3) or not np.isfinite(anchors).all():
        raise ValueError("forward_anchors must return finite [3,3] metres")
    centre = anchors[0] * 1000.0
    radius = float(np.linalg.norm(centre[:2]))
    if radius < 1.0:
        raise ValueError("radial approach undefined within 1 mm of base axis")
    radial = centre[:2] / radius
    pitch = float(np.arctan2(axis[:2] @ radial, -axis[2]))
    return dict(radial_mm=radius, centre_z_mm=float(centre[2]),
                pitch_rad=pitch,
                tilt_deg=float(np.rad2deg(np.arccos(np.clip(-axis[2], -1., 1.)))),
                approach_axis_world=axis.tolist(),
                anchor_xyz_mm=(anchors * 1000.0).tolist())


class VerticalApproachSolver:
    """One small J2/J3/J4 damped-Jacobian proposal; never opens hardware.

    A tilted gripper first lifts with pitch held until there is a rotation
    reserve.  It then changes pitch while holding radial position and slightly
    raising the centre.  Descent is considered only when full 3-D tilt is small.
    These checks do not identify contact or decide when to close the gripper.
    """

    def __init__(self, kinematics, clearance, lower, upper, *,
                 max_joint_step_position=2.0, max_translation_mm=5.0,
                 vertical_tolerance_deg=5.0, rotation_clearance_mm=20.0,
                 tracking_reserve_mm=5.0, damping=0.10,
                 angular_lever_mm=50.0, interpolation_step_position=0.10):
        self.kinematics, self.clearance = kinematics, clearance
        self.lower = np.asarray(lower, dtype=np.float64).copy()
        self.upper = np.asarray(upper, dtype=np.float64).copy()
        if (self.lower.shape != (6,) or self.upper.shape != (6,)
                or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all()
                or np.any(self.lower >= self.upper)):
            raise ValueError("finite ordered calibrated-position bounds required")
        values = np.asarray([max_joint_step_position, max_translation_mm,
                             vertical_tolerance_deg, rotation_clearance_mm,
                             tracking_reserve_mm, damping, angular_lever_mm,
                             interpolation_step_position], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("approach settings must be finite and positive")
        if vertical_tolerance_deg >= 15 or tracking_reserve_mm < 5:
            raise ValueError("vertical tolerance must be <15 deg; tracking reserve >=5 mm")
        self.max_step, self.max_translation = float(values[0]), float(values[1])
        self.tolerance, self.rotation_clearance = float(values[2]), float(values[3])
        self.reserve, self.damping = float(values[4]), float(values[5])
        self.lever, self.interpolation_step = float(values[6]), float(values[7])

    def _geometry(self, q):
        return approach_geometry(self.kinematics, q)

    def _state(self, geometry):
        return np.asarray([geometry['radial_mm'], geometry['centre_z_mm'],
                           self.lever * geometry['pitch_rad']])

    def propose_view_chunk(self,q_position,*,radial_probe_mm=5.,maximum_command_delta=6.) -> ApproachProposal:
        """Re-solve five high-clearance waypoints; never close or descend.

        The old one-Jacobian chord scaled away its small height allowance when
        limiting a large rotation, forcing tiny accepted steps. Each waypoint
        here satisfies nonlinear position/orientation constraints afresh.
        This is a nominal engineering plan, not measured motion or a grasp.
        """
        from scipy.optimize import minimize
        q=np.asarray(q_position,np.float64)
        diag=dict(kind='nonlinear_high_view_chunk',accepted=False,geometry_is_estimate=True,
                  executable=False,gripper_held=True,waypoints=[])
        def reject(reason):
            diag['reason']=reason
            return ApproachProposal(False,None,diag)
        if (q.shape!=(6,) or not np.isfinite(q).all() or (q<self.lower).any() or (q>self.upper).any()
                or not np.isfinite([radial_probe_mm,maximum_command_delta]).all()
                or abs(radial_probe_mm)>30. or not 0<maximum_command_delta<=6.):
            return reject('invalid_view_chunk_input')
        base=self._geometry(q)
        floor=max(self.clearance.planned_clearance_mm,
                  self.clearance.readback_clearance_mm+self.reserve,self.rotation_clearance)
        if self.clearance.clearance_mm(q)<floor:
            return reject('view_chunk_requires_existing_rotation_clearance')
        verticalize=base['tilt_deg']>self.tolerance
        if verticalize and abs(base['pitch_rad'])<np.deg2rad(.2):
            return reject('vertical_unreachable_with_fixed_J5')
        diag.update(stage='verticalize' if verticalize else 'radial_view_probe',
            source=base,minimum_required_clearance_mm=float(floor),
            radial_probe_mm=0. if verticalize else float(radial_probe_mm),
            maximum_command_delta_position=float(maximum_command_delta))
        current=q.copy();commands=[];minimum=float('inf')
        for index in range(5):
            start_geo=self._geometry(current)
            target_z=base['centre_z_mm']+(index+1)*(1. if verticalize else .2)
            target_r=base['radial_mm']+(0. if verticalize else radial_probe_mm*(index+1)/5.)
            def geometry(values):
                candidate=current.copy();candidate[[1,2,3]]=values
                return self._geometry(candidate)
            def equality(values):
                g=geometry(values)
                return np.array([g['centre_z_mm']-target_z,
                    g['radial_mm']-target_r if verticalize else self.lever*(g['pitch_rad']-base['pitch_rad'])])
            def objective(values):
                g=geometry(values)
                return (self.lever*g['pitch_rad'])**2 if verticalize else (g['radial_mm']-target_r)**2
            lower=np.maximum(self.lower[[1,2,3]],current[[1,2,3]]-maximum_command_delta)
            upper=np.minimum(self.upper[[1,2,3]],current[[1,2,3]]+maximum_command_delta)
            result=minimize(objective,current[[1,2,3]],method='SLSQP',bounds=list(zip(lower,upper)),
                constraints={'type':'eq','fun':equality},
                options={'maxiter':60,'ftol':1e-7,'eps':1e-4})
            if not result.success or not np.isfinite(result.x).all() or np.abs(equality(result.x)).max()>.02:
                diag['solver_failure']=dict(waypoint=index+1,message=str(result.message),
                    iterations=int(result.nit),equality_residual=equality(result.x).tolist())
                return reject('nonlinear_waypoint_constraints_not_solved')
            end=current.copy();end[[1,2,3]]=result.x
            count=max(5,int(np.ceil(np.abs(end-current).max()/self.interpolation_step)))
            segment=current+np.linspace(0,1,count+1)[:,None]*(end-current)
            geometries=[self._geometry(row) for row in segment]
            clearances=np.asarray([self.clearance.clearance_mm(row) for row in segment])
            if (not np.isfinite(clearances).all() or clearances.min()<floor-1e-9
                    or min(g['centre_z_mm'] for g in geometries)<start_geo['centre_z_mm']-.02):
                return reject('nonlinear_view_sweep_violates_unchanged_clearance_or_height')
            if not verticalize and max(g['tilt_deg'] for g in geometries)>self.tolerance:
                return reject('radial_view_probe_loses_vertical_pose')
            minimum=min(minimum,float(clearances.min()))
            commands.append(end.copy());current=end
            diag['waypoints'].append(dict(q_position=end.tolist(),geometry=geometries[-1],
                equality_residual=equality(result.x).tolist(),samples=len(segment)))
        end_geo=self._geometry(current)
        if verticalize and end_geo['tilt_deg']>=base['tilt_deg']-.01:
            return reject('no_verticalization_progress')
        if not verticalize and abs(end_geo['radial_mm']-base['radial_mm'])<.1:
            return reject('no_radial_view_progress')
        diag.update(accepted=True,reason='five_nonlinear_waypoints',endpoint=end_geo,
            minimum_path_clearance_mm=minimum,physical_grasp_verified=False)
        return ApproachProposal(True,np.asarray(commands),diag)

    def propose(self, q_position, dz_mm, dr_mm=0.0) -> ApproachProposal:
        q = np.asarray(q_position, dtype=np.float64)
        diag = dict(kind='engineering_vertical_approach', units='lerobot_calibrated_position/mm',
                    requires_physical_audit=True, geometry_is_estimate=True,
                    accepted=False, requested_dz_mm=float(dz_mm), requested_dr_mm=float(dr_mm))

        def reject(reason):
            diag['reason'] = reason
            return ApproachProposal(False, None, diag)

        if (q.shape != (6,) or not np.isfinite(q).all()
                or not np.isfinite([dz_mm, dr_mm]).all()):
            return reject('invalid_nonfinite_input')
        if np.any(q < self.lower) or np.any(q > self.upper):
            return reject('source_outside_joint_bounds')
        try:
            base = self._geometry(q)
            clearance = float(self.clearance.clearance_mm(q))
            hard_floor = float(self.clearance.readback_clearance_mm)
            planned_floor = float(self.clearance.planned_clearance_mm)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            diag['detail'] = str(exc)
            return reject('geometry_unavailable')
        if not np.isfinite([clearance, hard_floor, planned_floor]).all():
            return reject('nonfinite_clearance')
        if hard_floor < -3.0 or planned_floor < hard_floor:
            return reject('invalid_existing_clearance_contract')
        if clearance < hard_floor:
            return reject('source_below_readback_floor')
        floor = max(planned_floor, hard_floor + self.reserve)
        diag.update(source=base, source_clearance_mm=clearance,
                    required_predicted_clearance_mm=floor)
        state = self._state(base)
        target = state.copy()
        tilted = base['tilt_deg'] > self.tolerance
        below_reserve = clearance < floor
        if tilted and clearance < self.rotation_clearance:
            stage = 'lift_before_verticalize'
            target[1] += min(self.max_translation, self.rotation_clearance - clearance + 0.5)
        elif below_reserve:
            stage = 'lift_to_clearance_reserve'
            target[1] += min(self.max_translation, floor - clearance + 0.5)
        elif tilted:
            stage = 'verticalize'
            # Slight rise prevents an orientation correction from being used as descent.
            target[1] += 0.5
            target[2] = 0.0
        else:
            stage = 'approach'
            target[:2] += np.clip([dr_mm, dz_mm], -self.max_translation, self.max_translation)
            target[2] = 0.0
            if abs(dz_mm) + abs(dr_mm) < 1e-8:
                return reject('already_vertical_no_translation_requested')
        diag['stage'] = stage
        if below_reserve:
            # Above the unchanged physical floor, only a monotonic upward
            # recovery is admissible. Requiring the *source* already to meet
            # the tracking reserve would make that recovery impossible.
            diag['target_clearance_reserve_mm'] = floor
            floor = clearance
            diag.update(monotonic_reserve_recovery=True,
                        required_predicted_clearance_mm=floor)
        if stage == 'verticalize':
            # Keep a separate rotation reserve throughout the jaw sweep, not
            # merely at its starting pose.  The physical floor is unchanged.
            floor = max(floor, self.rotation_clearance - self.reserve)
            diag['required_predicted_clearance_mm'] = floor
        if stage == 'verticalize' and abs(base['pitch_rad']) < np.deg2rad(0.2):
            return reject('vertical_unreachable_with_fixed_J5')
        jacobian = np.zeros((3, 3), dtype=np.float64)
        try:
            for i, joint in enumerate((1, 2, 3)):
                plus, minus = q.copy(), q.copy()
                plus[joint] = min(self.upper[joint], q[joint] + 0.10)
                minus[joint] = max(self.lower[joint], q[joint] - 0.10)
                jacobian[:, i] = ((self._state(self._geometry(plus))
                                  - self._state(self._geometry(minus)))
                                 / (plus[joint] - minus[joint]))
            singular = np.linalg.svd(jacobian, compute_uv=False)
            diag['jacobian_singular_values'] = singular.tolist()
            if singular[-1] < 1e-4 or singular[0] / singular[-1] > 1e4:
                return reject('singular_or_ill_conditioned_jacobian')
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + self.damping**2 * np.eye(3), target - state)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            diag['detail'] = str(exc)
            return reject('jacobian_unavailable')
        delta *= min(1.0, self.max_step / max(float(np.abs(delta).max()), 1e-12))
        initial_error = float(np.linalg.norm(target - state))
        # Bound before evaluation; never silently clip a joint and call the task solved.
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            end = q.copy()
            end[[1, 2, 3]] += scale * delta
            if np.any(end < self.lower) or np.any(end > self.upper):
                continue
            count = max(5, int(np.ceil(np.abs(end - q).max() / self.interpolation_step)))
            path = q + np.linspace(0.0, 1.0, count + 1)[:, None] * (end - q)
            try:
                geometry = [self._geometry(row) for row in path]
                clearances = np.asarray([self.clearance.clearance_mm(row) for row in path])
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                diag['detail'] = str(exc)
                return reject('path_geometry_unavailable')
            if not np.isfinite(clearances).all() or clearances.min() < floor - 1e-9:
                continue
            # Full jaw sweep, not just endpoint/centre height. Parent must still run
            # the established physical path and actual-readback admission checks.
            anchors = np.asarray([g['anchor_xyz_mm'] for g in geometry])
            if stage in ('lift_before_verticalize', 'lift_to_clearance_reserve'):
                if (anchors[:, :, 2] < anchors[0, :, 2] - 0.02).any():
                    continue
                if abs(geometry[-1]['pitch_rad'] - base['pitch_rad']) > np.deg2rad(0.5):
                    continue
                if below_reserve and ((np.diff(clearances) < -1e-9).any()
                                      or clearances[-1] <= clearance + 0.01):
                    continue
            elif stage == 'verticalize':
                if min(g['centre_z_mm'] for g in geometry) < base['centre_z_mm'] - 0.02:
                    continue
                if geometry[-1]['tilt_deg'] >= base['tilt_deg'] - 0.01:
                    continue
            elif dz_mm < 0 and max(g['tilt_deg'] for g in geometry) > self.tolerance:
                continue
            endpoint_error = float(np.linalg.norm(target - self._state(geometry[-1])))
            if endpoint_error >= initial_error - 1e-5:
                continue
            if stage == 'approach' and dz_mm < 0 and geometry[-1]['centre_z_mm'] >= base['centre_z_mm']:
                continue
            commands = q + np.linspace(0.2, 1.0, 5)[:, None] * (end - q)
            diag.update(accepted=True, reason='bounded_geometry_proposal', endpoint=geometry[-1],
                        endpoint_error_mm_equivalent=endpoint_error,
                        minimum_path_clearance_mm=float(clearances.min()),
                        delta_position=(end - q).tolist(), path_samples=len(path))
            return ApproachProposal(True, commands, diag)
        return reject('no_progress_within_joint_clearance_and_tilt_constraints')
