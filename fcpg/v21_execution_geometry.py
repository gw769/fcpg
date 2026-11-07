"""Execution-geometry vertical pinch — not Actor output and not assistance.

The frozen 3k Actor emits joint positions. It never outputs a grasp angle.
``cos2θ,sin2θ`` in the 8-D token is the *measured* object-axis vs jaw-line
angle. Rewriting those two numbers to a vertical frame would fight the
training distribution: the Actor learned to treat a small θ as “already
aligned.” This module leaves the token alone.

When a planned chunk is closing the gripper, wrist_flex (J4) and wrist_roll
(J5) are nudged toward a more vertical tool axis (world −Z), inside joint
limits and the wrap-safe J5 band. J6 stays the Actor’s. J1–J3 are
overwritten by M-align from current *q* plus a damped Jacobian step of
M→target — not Actor J1–J3 plus a nudge. This is not grounding-correction,
head-grasp, visual alignment, or Actor success.
"""
from __future__ import annotations

import numpy as np

from .v21_vertical_approach import approach_geometry

# Demonstrated wrist-roll band. Outside it the wrist camera / cables wrap.
J5_WRAP_MIN = -95.0
J5_WRAP_MAX = -45.0
CLOSE_Q6_DELTA = 3.0
MAX_FLEX_STEP = 4.0
MAX_ROLL_STEP = 4.0
TILT_PROBE = 1.0
TILT_WORTH_MOVING_DEG = 1.0


def _finite_six(value, name):
    array = np.asarray(value, np.float64)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite six-vector')
    return array


def _clip_joint(value, index, lower, upper):
    lo, hi = float(lower[index]), float(upper[index])
    if index == 4:
        lo, hi = max(lo, J5_WRAP_MIN), min(hi, J5_WRAP_MAX)
    return float(min(hi, max(lo, value)))


def _tilt_deg(kinematics, q):
    return float(approach_geometry(kinematics, q)['tilt_deg'])


def _descent_step(kinematics, q, index, *, probe, limit, lower, upper):
    """One-sided finite difference: move the joint the way that lowers tilt."""
    base = _tilt_deg(kinematics, q)
    best = 0.0
    best_tilt = base
    for sign in (-1.0, 1.0):
        trial = q.copy()
        trial[index] = _clip_joint(q[index] + sign * probe, index, lower, upper)
        if abs(trial[index] - q[index]) < 1e-6:
            continue
        tilt = _tilt_deg(kinematics, trial)
        if tilt + 1e-6 < best_tilt:
            best_tilt = tilt
            best = float(np.clip(trial[index] - q[index], -limit, limit))
    if base - best_tilt < TILT_WORTH_MOVING_DEG:
        return 0.0, base
    return best, best_tilt


def project_vertical_pinch(q_now, commands, kinematics, lower, upper):
    """Project J4/J5 toward vertical at close. Returns (commands, diagnostic)."""
    q = _finite_six(q_now, 'q_now')
    cmds = np.asarray(commands, np.float64)
    if cmds.shape != (5, 6) or not np.isfinite(cmds).all():
        raise ValueError('commands must be finite [5,6]')
    lower = _finite_six(lower, 'lower')
    upper = _finite_six(upper, 'upper')
    if np.any(lower >= upper):
        raise ValueError('joint limits must be strictly ordered')
    diagnostic = dict(
        kind='execution_geometry_vertical_pinch',
        learned_policy_success=False,
        assistance=False,
        token_rewritten=False,
        applied=False,
        reason=None,
        tilt_deg_before=None,
        tilt_deg_after=None,
        j4_delta=0.0,
        j5_delta=0.0,
    )
    closing = bool((cmds[:, 5] < q[5] - CLOSE_Q6_DELTA).any())
    if not closing:
        diagnostic['reason'] = 'not_closing'
        return cmds.copy(), diagnostic
    if kinematics is None:
        diagnostic['reason'] = 'kinematics_unavailable'
        return cmds.copy(), diagnostic
    try:
        tilt0 = _tilt_deg(kinematics, q)
        dj4, _ = _descent_step(kinematics, q, 3, probe=TILT_PROBE,
                               limit=MAX_FLEX_STEP, lower=lower, upper=upper)
        rolled = q.copy()
        rolled[3] = _clip_joint(q[3] + dj4, 3, lower, upper)
        dj5, _ = _descent_step(kinematics, rolled, 4, probe=TILT_PROBE,
                               limit=MAX_ROLL_STEP, lower=lower, upper=upper)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        diagnostic['reason'] = f'geometry_unavailable:{type(exc).__name__}'
        return cmds.copy(), diagnostic
    diagnostic['tilt_deg_before'] = tilt0
    if abs(dj4) < 1e-3 and abs(dj5) < 1e-3:
        diagnostic['reason'] = 'already_vertical_or_no_progress'
        diagnostic['tilt_deg_after'] = tilt0
        return cmds.copy(), diagnostic
    out = cmds.copy()
    ramp = np.linspace(0.2, 1.0, 5)
    out[:, 3] = [_clip_joint(cmds[i, 3] + dj4 * t, 3, lower, upper) for i, t in enumerate(ramp)]
    out[:, 4] = [_clip_joint(cmds[i, 4] + dj5 * t, 4, lower, upper) for i, t in enumerate(ramp)]
    end = cmds[-1].copy()
    end[3], end[4] = out[-1, 3], out[-1, 4]
    try:
        diagnostic['tilt_deg_after'] = _tilt_deg(kinematics, end)
    except (ValueError, RuntimeError, FloatingPointError):
        diagnostic['tilt_deg_after'] = None
    diagnostic.update(applied=True, reason='projected_at_close',
                      j4_delta=float(out[-1, 3] - q[3]),
                      j5_delta=float(out[-1, 4] - q[4]))
    return out, diagnostic


TABLE_Z_M = -0.017118124
# Slightly higher grasp plane so Actor / IBVS stay above readback clearance.
GRASP_LIFT_M = 0.035
M_DEADZONE_M = 0.008
M_MAX_ERR_M = 0.12
M_MAX_JOINT_STEP = 8.0
# Follow treats <~3° as within tolerance; a step must clear that when far.
M_MIN_JOINT_STEP = 4.0
M_MIN_STEP_ERR_M = 0.03
M_DAMP_REL = 1e-2
# M-align engages only at close or once the Actor has M this near the 重心.
M_ENGAGE_M = 0.06


def _finite_xyz(value, name):
    array = np.asarray(value, np.float64).reshape(-1)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite three-vector')
    return array


def fk_midpoint_m(kinematics, q):
    """FK centre M: midpoint of the two fingertip anchors, metres, base frame."""
    anchors = np.asarray(kinematics.forward_anchors(q), dtype=np.float64)
    if anchors.shape == (3, 3):
        mid = anchors[0]
    elif anchors.ndim >= 2 and anchors.shape[-2:] == (3, 3):
        mid = anchors.reshape(-1, 3, 3)[0, 0]
    else:
        raise ValueError('forward_anchors must return [3,3] centre/fixed/moving')
    return _finite_xyz(mid, 'fk_midpoint_M')


def project_fk_m_toward_xyz(q_now, commands, kinematics, target_xyz_m, lower, upper, *, lock_j1=False):
    """Drive FK centre M toward a base-frame 重心. Not fingertip image servo.

    M is ``forward_anchors(...)[0]`` = 0.5*(fixed_tip + moving_tip). J1–J3
    start from the *current* ``q``, plus the damped Jacobian of M→target.
    Actor J1–J3 are overwritten. J4–J6 stay the Actor / later pinch path.
    Tips are not steered in the wrist JPEG. ``learned_policy_success`` stays
    False: this is not Actor success and not head-grasp assist.
    """
    q = _finite_six(q_now, 'q_now')
    cmds = np.asarray(commands, np.float64)
    if cmds.shape != (5, 6) or not np.isfinite(cmds).all():
        raise ValueError('commands must be finite [5,6]')
    lower = _finite_six(lower, 'lower')
    upper = _finite_six(upper, 'upper')
    diagnostic = dict(
        kind='execution_geometry_align_M',
        learned_policy_success=False,
        assistance=False,
        token_rewritten=False,
        applied=False,
        reason=None,
        error_m=None,
        m_xyz=None,
        target_xyz=None,
        joint_delta=[0.0, 0.0, 0.0],
    )
    if kinematics is None:
        diagnostic['reason'] = 'kinematics_unavailable'
        return cmds.copy(), diagnostic
    try:
        target = _finite_xyz(target_xyz_m, 'target_xyz_m')
        mid = fk_midpoint_m(kinematics, q)
    except (ValueError, TypeError, RuntimeError) as exc:
        diagnostic['reason'] = f'geometry_unavailable:{type(exc).__name__}'
        return cmds.copy(), diagnostic
    err = target - mid
    norm = float(np.linalg.norm(err))
    diagnostic.update(error_m=norm, m_xyz=mid.tolist(), target_xyz=target.tolist())
    if norm < M_DEADZONE_M:
        diagnostic['reason'] = 'within_deadzone'
        return cmds.copy(), diagnostic
    if norm > M_MAX_ERR_M:
        err = err * (M_MAX_ERR_M / norm)
    def _clipped_step(delta3):
        stepped = q.copy()
        for joint in range(3):
            if lock_j1 and joint == 0:
                continue
            stepped[joint] = _clip_joint(q[joint] + float(delta3[joint]), joint, lower, upper)
        return stepped

    def _error_at(q_try):
        return float(np.linalg.norm(target - fk_midpoint_m(kinematics, q_try)))

    jacobian = np.zeros((3, 3), np.float64)
    try:
        for column, joint in enumerate((0, 1, 2)):
            # Probe away from the nearer limit so a folded joint still reports.
            sign = -1.0 if (upper[joint] - q[joint]) < (q[joint] - lower[joint]) else 1.0
            probe = q.copy()
            probe[joint] = _clip_joint(q[joint] + sign * 0.5, joint, lower, upper)
            h = float(probe[joint] - q[joint])
            if abs(h) < 1e-6:
                continue
            jacobian[:, column] = (fk_midpoint_m(kinematics, probe) - mid) / h
    except (ValueError, TypeError, RuntimeError) as exc:
        diagnostic['reason'] = f'jacobian_unavailable:{type(exc).__name__}'
        return cmds.copy(), diagnostic

    candidates = []
    # Damped least squares, re-solved with limit-blocked joints removed.
    blocked = np.zeros(3, dtype=bool)
    blocked[0] = bool(lock_j1)
    for _ in range(3):
        jac = jacobian.copy()
        jac[:, blocked] = 0.0
        jjt = jac @ jac.T
        trace = float(np.trace(jjt))
        if trace <= 1e-14:
            break
        # Relative damping: J is ~1e-3 m/deg so JJᵀ ~1e-6..1e-5; a fixed 1e-4
        # swamped it at the fold and left a ~2° step inside follow tolerance.
        gram = jjt + (M_DAMP_REL * trace / 3.0) * np.eye(3)
        try:
            delta = jac.T @ np.linalg.solve(gram, err)
        except np.linalg.LinAlgError:
            break
        peak = max(float(np.max(np.abs(delta))), 1e-9)
        if peak > M_MAX_JOINT_STEP:
            delta = delta * (M_MAX_JOINT_STEP / peak)
        elif norm > M_MIN_STEP_ERR_M and peak < M_MIN_JOINT_STEP:
            # Far from target: never emit a step the follower treats as no-op.
            delta = delta * (M_MIN_JOINT_STEP / peak)
        stepped = _clipped_step(delta)
        newly_blocked = np.array([
            abs(stepped[j] - q[j]) < 0.5 * abs(float(delta[j])) and abs(float(delta[j])) > 1e-3
            for j in range(3)
        ])
        candidates.append(('dls', stepped))
        if not newly_blocked.any() or (newly_blocked & ~blocked).sum() == 0:
            break
        blocked |= newly_blocked
    # True-FK coordinate descent: immune to the fold linearization and limits.
    step = M_MAX_JOINT_STEP if norm > M_MIN_STEP_ERR_M else M_MIN_JOINT_STEP
    for joint in range(3):
        for sign in (-1.0, 1.0):
            delta = np.zeros(3)
            delta[joint] = sign * step
            candidates.append(('cd1', _clipped_step(delta)))
    for s2 in (-1.0, 1.0):
        for s3 in (-1.0, 1.0):
            delta = np.array([0.0, s2 * step, s3 * step])
            candidates.append(('cd2', _clipped_step(delta)))

    best_q, best_err, best_kind = None, norm, None
    try:
        for kind, stepped in candidates:
            if float(np.max(np.abs(stepped[:3] - q[:3]))) < 1e-3:
                continue
            e = _error_at(stepped)
            if e < best_err - 1e-4:
                best_q, best_err, best_kind = stepped, e, kind
    except (ValueError, TypeError, RuntimeError) as exc:
        diagnostic['reason'] = f'fk_unavailable:{type(exc).__name__}'
        return cmds.copy(), diagnostic
    if best_q is None:
        diagnostic['reason'] = 'no_progress_within_limits'
        return cmds.copy(), diagnostic
    delta = best_q[:3] - q[:3]
    out = cmds.copy()
    ramp = np.linspace(0.2, 1.0, 5)
    aimed = q[None, 0:3] + ramp[:, None] * delta
    for i in range(5):
        for joint in range(3):
            out[i, joint] = _clip_joint(float(aimed[i, joint]), joint, lower, upper)
    diagnostic.update(applied=True, reason='aligned_M_to_target', solver=best_kind,
                      error_m_after_step=float(best_err),
                      joint_delta=[float(out[-1, j] - q[j]) for j in (0, 1, 2)])
    return out, diagnostic


# Wrist-image approach: shrink r = |object CoG − jaw midpoint| in letterbox224.
# Column M-align cannot do this — its target is a table-plane proxy and the
# fish can stay half a frame left of the jaws while M reports ~1 cm.
WRIST_R_DEADZONE_PX = 12.0
WRIST_R_MAX_JOINT_STEP = 6.0
WRIST_R_MIN_JOINT_STEP = 3.0
WRIST_R_MIN_STEP_ERR_PX = 30.0
WRIST_R_MAX_ERR_PX = 120.0
# IBVS XY floor. 28 mm then 8 mm still rejected every J1 pan — but the
# real blocker is scoring, not the floor (see project_wrist_m_toward_pixel).
# Warm-connect floor is −3 mm; 2 mm keeps commanded M above the table.
WRIST_R_TABLE_CLEARANCE_M = 0.002
# Lift-first escape may start below the XY floor; only refuse table penetration.
WRIST_R_LIFT_FLOOR_M = 0.002
# Eye-in-hand J1: jaws are almost fixed in the wrist image, so the jaw-mid
# Jacobian never credits a pan. Isolated trial 1789367192245946185 (J2/J3
# held): +19.6° J1 moved CoG x 89→15, so +J1 slides the scene −x (~4 px/deg).
# Fish left of jaws (err_x<0) therefore needs −J1. The older 25 px/deg and
# opposite sign came from a column-M mix that also slammed J2/J3.
WRIST_R_PAN_PX = 12.0
WRIST_R_J1_PX_PER_DEG = 5.0
WRIST_R_J1_MAX_STEP = 4.0
WRIST_R_J1_MIN_STEP = 0.8


def _scene_j1_delta(err_x):
    """J1 that slides the wrist scene toward the jaws. Jaws stay put in-image."""
    ex = float(err_x)
    if abs(ex) < WRIST_R_PAN_PX:
        return 0.0
    delta = ex / WRIST_R_J1_PX_PER_DEG
    if abs(delta) < WRIST_R_J1_MIN_STEP:
        delta = np.copysign(WRIST_R_J1_MIN_STEP, delta)
    return float(np.clip(delta, -WRIST_R_J1_MAX_STEP, WRIST_R_J1_MAX_STEP))


def project_wrist_m_toward_pixel(q_now, commands, jaw_pair_fn, target_px, lower, upper,
                                 *, kinematics=None):
    """Drive the wrist projection of the jaw midpoint toward ``target_px``.

    ``jaw_pair_fn(q) -> [2,2]`` is the existing wrist FK projection of the
    two jaw tips (letterbox224). J1–J3 start from the current ``q`` plus a
    damped Jacobian step of pixel error; Actor J1–J3 are overwritten, J4–J6
    stay the Actor's. No roll, no FG⊥. When ``kinematics`` is given, steps
    that would put FK-M below the table clearance are rejected.

    The wrist camera rides with the jaws, so J1 does not move ``jaw_pair``
    pixels. Scoring only that Jacobian makes every servo ``lift_clearance``
    with unchanged error (trial 1789365918973703885). After the jaw-mid
    loop, a small eye-in-hand J1 pan is applied from image ``err_x``.
    """
    q = _finite_six(q_now, 'q_now')
    cmds = np.asarray(commands, np.float64)
    if cmds.shape != (5, 6) or not np.isfinite(cmds).all():
        raise ValueError('commands must be finite [5,6]')
    lower = _finite_six(lower, 'lower')
    upper = _finite_six(upper, 'upper')
    diagnostic = dict(
        kind='execution_geometry_wrist_r',
        learned_policy_success=False,
        assistance=False,
        token_rewritten=False,
        applied=False,
        reason=None,
        error_px=None,
        mid_px=None,
        target_px=None,
        joint_delta=[0.0, 0.0, 0.0],
    )
    if jaw_pair_fn is None:
        diagnostic['reason'] = 'jaw_projection_unavailable'
        return cmds.copy(), diagnostic

    def _mid(q_try):
        pair = np.asarray(jaw_pair_fn(q_try), np.float64).reshape(-1, 2)
        if pair.shape != (2, 2) or not np.isfinite(pair).all():
            raise ValueError('jaw pair must be finite [2,2]')
        return pair.mean(0)

    try:
        target = np.asarray(target_px, np.float64).reshape(-1)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError('target_px must be a finite pixel')
        mid = _mid(q)
    except (ValueError, TypeError, RuntimeError) as exc:
        diagnostic['reason'] = f'projection_unavailable:{type(exc).__name__}'
        return cmds.copy(), diagnostic
    raw_err = target - mid
    norm = float(np.linalg.norm(raw_err))
    diagnostic.update(error_px=norm, mid_px=mid.tolist(), target_px=target.tolist(),
                      err_x_px=float(raw_err[0]))
    if norm < WRIST_R_DEADZONE_PX:
        diagnostic['reason'] = 'within_deadzone'
        return cmds.copy(), diagnostic
    err = raw_err.copy()
    if norm > WRIST_R_MAX_ERR_PX:
        err = err * (WRIST_R_MAX_ERR_PX / norm)

    def _clipped_step(delta3):
        stepped = q.copy()
        for joint in range(3):
            stepped[joint] = _clip_joint(q[joint] + float(delta3[joint]), joint, lower, upper)
        return stepped

    def _z(q_try):
        if kinematics is None:
            return None
        try:
            return float(fk_midpoint_m(kinematics, q_try)[2])
        except (ValueError, TypeError, RuntimeError, AttributeError):
            return None

    def _above(q_try, floor_m):
        z1 = _z(q_try)
        return True if z1 is None else z1 >= TABLE_Z_M + float(floor_m)

    def _safe(q_try):
        return _above(q_try, WRIST_R_TABLE_CLEARANCE_M)

    def _no_dive(q_try, *, allow_below_floor=False):
        z1 = _z(q_try)
        if z1 is None:
            return True
        z0 = _z(q)
        if z0 is None:
            return True
        if z1 < z0 - 1e-4:
            return False
        if allow_below_floor:
            return True
        return z1 >= TABLE_Z_M + WRIST_R_LIFT_FLOOR_M

    jacobian = np.zeros((2, 3), np.float64)
    try:
        for column, joint in enumerate((0, 1, 2)):
            sign = -1.0 if (upper[joint] - q[joint]) < (q[joint] - lower[joint]) else 1.0
            probe = q.copy()
            probe[joint] = _clip_joint(q[joint] + sign * 0.5, joint, lower, upper)
            h = float(probe[joint] - q[joint])
            if abs(h) < 1e-6:
                continue
            jacobian[:, column] = (_mid(probe) - mid) / h
    except (ValueError, TypeError, RuntimeError) as exc:
        diagnostic['reason'] = f'jacobian_unavailable:{type(exc).__name__}'
        return cmds.copy(), diagnostic

    candidates = []
    jjt = jacobian @ jacobian.T
    trace = float(np.trace(jjt))
    if trace > 1e-14:
        gram = jjt + (M_DAMP_REL * trace / 2.0) * np.eye(2)
        try:
            delta = jacobian.T @ np.linalg.solve(gram, err)
            peak = max(float(np.max(np.abs(delta))), 1e-9)
            if peak > WRIST_R_MAX_JOINT_STEP:
                delta = delta * (WRIST_R_MAX_JOINT_STEP / peak)
            elif norm > WRIST_R_MIN_STEP_ERR_PX and peak < WRIST_R_MIN_JOINT_STEP:
                delta = delta * (WRIST_R_MIN_JOINT_STEP / peak)
            candidates.append(('dls', _clipped_step(delta)))
        except np.linalg.LinAlgError:
            pass
    step = WRIST_R_MAX_JOINT_STEP if norm > WRIST_R_MIN_STEP_ERR_PX else WRIST_R_MIN_JOINT_STEP
    for joint in range(3):
        for sign in (-1.0, 1.0):
            delta = np.zeros(3)
            delta[joint] = sign * step
            candidates.append(('cd1', _clipped_step(delta)))

    best_q, best_err, best_kind = None, norm, None
    try:
        for kind, stepped in candidates:
            if float(np.max(np.abs(stepped[:3] - q[:3]))) < 1e-3:
                continue
            if not _safe(stepped):
                continue
            e = float(np.linalg.norm(target - _mid(stepped)))
            if e < best_err - 0.5:
                best_q, best_err, best_kind = stepped, e, kind
    except (ValueError, TypeError, RuntimeError) as exc:
        diagnostic['reason'] = f'projection_unavailable:{type(exc).__name__}'
        return cmds.copy(), diagnostic
    if best_q is not None and abs(best_q[0] - q[0]) < 0.5:
        # J2/J3 won on jaw-mid pixels; still pan if the fish is off to the side.
        dj1 = _scene_j1_delta(raw_err[0])
        if abs(dj1) >= 1e-3:
            combo = best_q.copy()
            combo[0] = _clip_joint(q[0] + dj1, 0, lower, upper)
            if (abs(combo[0] - q[0]) >= 1e-3
                    and _no_dive(combo, allow_below_floor=True)):
                best_q = combo
                best_kind = 'lift_and_pan'
                diagnostic['j1_pan_deg'] = float(combo[0] - q[0])
    lift_best = None
    if best_q is None and kinematics is not None:
        # Raise FK-M when every jaw-mid candidate is blocked or scores nothing.
        z0 = _z(q)
        if z0 is not None:
            lift_z = z0
            for joint in (1, 2):
                for sign in (-1.0, 1.0):
                    delta = np.zeros(3)
                    delta[joint] = sign * WRIST_R_MAX_JOINT_STEP
                    stepped = _clipped_step(delta)
                    if float(np.max(np.abs(stepped[:3] - q[:3]))) < 1e-3:
                        continue
                    if not _above(stepped, WRIST_R_LIFT_FLOOR_M):
                        continue
                    z1 = _z(stepped)
                    if z1 is not None and z1 > lift_z + 1e-4:
                        lift_best, lift_z = stepped, z1
    if best_q is None:
        # Jaw-mid Jacobian is ~0 on J1 (eye-in-hand). Command a scene pan
        # from image err_x; do not require mid_px to move.
        dj1 = _scene_j1_delta(raw_err[0])
        z0 = _z(q)
        need_lift = z0 is not None and z0 < TABLE_Z_M + WRIST_R_TABLE_CLEARANCE_M
        if abs(dj1) >= 1e-3:
            combo = q.copy()
            kind = 'j1_pan'
            if need_lift and lift_best is not None:
                combo = lift_best.copy()
                kind = 'lift_and_pan'
            combo[0] = _clip_joint(q[0] + dj1, 0, lower, upper)
            allow_floor = not (need_lift and lift_best is not None)
            if (abs(combo[0] - q[0]) >= 1e-3
                    and _no_dive(combo, allow_below_floor=allow_floor)):
                best_q, best_err, best_kind = combo, norm, kind
                diagnostic['j1_pan_deg'] = float(combo[0] - q[0])
        if best_q is None and lift_best is not None:
            best_q, best_err, best_kind = lift_best, norm, 'lift_clearance'
    if best_q is None:
        diagnostic['reason'] = 'no_progress_within_limits'
        return cmds.copy(), diagnostic
    delta = best_q[:3] - q[:3]
    out = cmds.copy()
    ramp = np.linspace(0.2, 1.0, 5)
    aimed = q[None, 0:3] + ramp[:, None] * delta
    for i in range(5):
        for joint in range(3):
            out[i, joint] = _clip_joint(float(aimed[i, joint]), joint, lower, upper)
    # A pure eye-in-hand pan owns J1, not the Actor's independent reach
    # joints. Preserve J2/J3 only when the combined commands remain above
    # the existing floor at every command. Full bridge checks still apply.
    diagnostic['actor_reach_preserved'] = False
    if best_kind == 'j1_pan' and kinematics is not None:
        combined = out.copy()
        combined[:, 1:3] = cmds[:, 1:3]
        if all(_z(row) is not None and _safe(row) for row in combined):
            out = combined
            diagnostic['actor_reach_preserved'] = True
    for i in range(5):
        for joint in range(6):
            out[i, joint] = _clip_joint(float(out[i, joint]), joint, lower, upper)
    diagnostic.update(applied=True, reason='servoed_wrist_r', solver=best_kind,
                      error_px_after_step=float(best_err),
                      joint_delta=[float(out[-1, j] - q[j]) for j in (0, 1, 2)])
    return out, diagnostic


def _is_closing(q_now, commands):
    q = np.asarray(q_now, np.float64).reshape(-1)
    cmds = np.asarray(commands, np.float64)
    return bool(cmds.ndim == 2 and cmds.shape[1] == 6 and q.shape[0] == 6
                and (cmds[:, 5] < q[5] - CLOSE_Q6_DELTA).any())


def _m_within_engage(q_now, kinematics, target_xyz_m):
    try:
        mid = fk_midpoint_m(kinematics, np.asarray(q_now, np.float64))
        return float(np.linalg.norm(_finite_xyz(target_xyz_m, 'target') - mid)) <= M_ENGAGE_M
    except (ValueError, TypeError, RuntimeError, AttributeError):
        return False


def project_trial_geometry(q_now, commands, kinematics, lower, upper, *,
                           target_xyz_m=None):
    """Vertical-pinch J4/J5 at close; align FK M only when the grasp is near.

    The reach is the Actor's. M is pulled onto the 重心 only when the chunk
    is closing the gripper, or M is already within ``M_ENGAGE_M`` of it.
    """
    if target_xyz_m is None:
        aimed, aim = commands, dict(kind='execution_geometry_align_M', applied=False,
                                    reason='no_target', learned_policy_success=False)
        if isinstance(commands, np.ndarray):
            aimed = commands.copy()
    elif not (_is_closing(q_now, commands)
              or _m_within_engage(q_now, kinematics, target_xyz_m)):
        aimed, aim = commands, dict(kind='execution_geometry_align_M', applied=False,
                                    reason='not_grasping_yet', learned_policy_success=False)
        if isinstance(commands, np.ndarray):
            aimed = commands.copy()
    else:
        aimed, aim = project_fk_m_toward_xyz(
            q_now, commands, kinematics, target_xyz_m, lower, upper)
    pinched, pinch = project_vertical_pinch(q_now, aimed, kinematics, lower, upper)
    closing = _is_closing(q_now, commands)
    # Grasp-time M–重心 distance: an execution-geometry accuracy number,
    # reported at the closing chunk only. Not a grasp-success score.
    m_error_at_close = None
    if closing and target_xyz_m is not None and kinematics is not None:
        try:
            end = np.asarray(pinched, np.float64)[-1]
            m_error_at_close = float(np.linalg.norm(
                _finite_xyz(target_xyz_m, 'target') - fk_midpoint_m(kinematics, end)))
        except (ValueError, TypeError, RuntimeError, AttributeError):
            m_error_at_close = None
    return pinched, dict(kind='execution_geometry_trial',
                         learned_policy_success=False, assistance=False,
                         aim=aim, pinch=pinch, closing=closing,
                         m_error_at_close_m=m_error_at_close,
                         applied=bool(aim.get('applied') or pinch.get('applied')))


__all__ = ['project_vertical_pinch', 'project_fk_m_toward_xyz', 'project_trial_geometry',
           'project_wrist_m_toward_pixel', 'fk_midpoint_m', 'TABLE_Z_M', 'GRASP_LIFT_M',
           'J5_WRAP_MIN', 'J5_WRAP_MAX']
