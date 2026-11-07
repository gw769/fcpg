"""Vertical pinch is execution geometry, not an 8-D token rewrite."""
import numpy as np
import pytest

from fcpg.so101_frozen_vit_grounding import support_is_clipped
from fcpg.v21_execution_geometry import (
    J5_WRAP_MAX,
    J5_WRAP_MIN,
    TABLE_Z_M,
    WRIST_R_J1_MAX_STEP,
    WRIST_R_TABLE_CLEARANCE_M,
    project_fk_m_toward_xyz,
    project_trial_geometry,
    project_vertical_pinch,
    project_wrist_m_toward_pixel,
)


class _FakeKin:
    def __init__(self, tilt_from_q):
        self.tilt_from_q = tilt_from_q

    def _unused(self):
        return None


def _geometry_monkey(monkeypatch, tilt_fn):
    def fake(kinematics, q):
        return dict(tilt_deg=float(tilt_fn(q)), pitch_rad=0.0,
                    radial_mm=200.0, centre_z_mm=80.0,
                    approach_axis_world=[0.0, 0.0, -1.0], anchor_xyz_mm=[[0, 0, 0]] * 3)
    monkeypatch.setattr('fcpg.v21_execution_geometry.approach_geometry', fake)


LOWER = np.array([-100., -100., -100., -80., -100., 0.])
UPPER = np.array([100., 100., 100., 80., 100., 100.])


def test_open_chunk_is_unchanged(monkeypatch):
    _geometry_monkey(monkeypatch, lambda q: 40.0)
    q = np.array([0., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    cmds[:, 5] = 32.0
    out, diag = project_vertical_pinch(q, cmds, object(), LOWER, UPPER)
    np.testing.assert_array_equal(out, cmds)
    assert diag['applied'] is False and diag['reason'] == 'not_closing'
    assert diag['token_rewritten'] is False and diag['learned_policy_success'] is False


def test_close_nudges_flex_the_way_that_lowers_tilt(monkeypatch):
    _geometry_monkey(monkeypatch, lambda q: max(0.0, 30.0 - 2.0 * (q[3] - 30.0)))
    q = np.array([1., 2., 3., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    cmds[:, 5] = np.linspace(32., 8., 5)
    out, diag = project_vertical_pinch(q, cmds, object(), LOWER, UPPER)
    assert diag['applied'] is True
    assert out[-1, 3] > cmds[-1, 3]
    np.testing.assert_allclose(out[:, :3], cmds[:, :3])
    np.testing.assert_allclose(out[:, 5], cmds[:, 5])
    assert diag['tilt_deg_after'] < diag['tilt_deg_before']


def test_j5_stays_in_wrap_safe_band(monkeypatch):
    _geometry_monkey(monkeypatch, lambda q: max(0.0, 80.0 + q[4]))  # more negative J5 lowers tilt
    q = np.array([0., 0., 0., 0., -46., 32.])
    cmds = np.tile(q, (5, 1))
    cmds[:, 5] = 8.0
    out, diag = project_vertical_pinch(q, cmds, object(), LOWER, UPPER)
    assert np.all(out[:, 4] >= J5_WRAP_MIN) and np.all(out[:, 4] <= J5_WRAP_MAX)
    assert out[-1, 4] <= -46.0 or diag['applied'] is False


def test_border_support_is_clipped():
    mask = np.zeros((224, 224), dtype=bool)
    mask[40:180, 0:20] = True
    assert support_is_clipped(mask) is True
    mask[:] = False
    mask[80:140, 80:140] = True
    assert support_is_clipped(mask) is False


class _FkM:
    def forward_anchors(self, q):
        q = np.asarray(q, np.float64).reshape(-1)
        mid = np.array([0.20 + 0.002 * q[0], 0.10 + 0.002 * q[1], 0.02 + 0.002 * q[2]])
        return np.stack([mid, mid + [0.01, 0, 0], mid + [-0.01, 0, 0]])


def test_aligns_fk_m_not_fingertip_pixels():
    q = np.array([0., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    target = np.array([0.24, 0.12, 0.03])
    out, diag = project_fk_m_toward_xyz(q, cmds, _FkM(), target, LOWER, UPPER)
    assert diag['applied'] is True and diag['kind'] == 'execution_geometry_align_M'
    assert diag['learned_policy_success'] is False
    assert out[-1, 0] > cmds[-1, 0]
    np.testing.assert_allclose(out[:, 3:], cmds[:, 3:])


def test_deadzone_does_not_move_m():
    q = np.array([0., 0., 0., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    target = np.array([0.20, 0.10, 0.02])
    out, diag = project_fk_m_toward_xyz(q, cmds, _FkM(), target, LOWER, UPPER)
    np.testing.assert_array_equal(out, cmds)
    assert diag['applied'] is False and diag['reason'] == 'within_deadzone'


def test_align_m_overwrites_actor_j1j3_from_current_q():
    """J1–J3 start at current q, not Actor J1–J3 plus a nudge."""
    q = np.array([0., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    cmds[:, :3] = [-40., -35., -30.]
    target = np.array([0.24, 0.14, 0.08])
    out, diag = project_fk_m_toward_xyz(q, cmds, _FkM(), target, LOWER, UPPER)
    assert diag['applied'] is True and diag['learned_policy_success'] is False
    assert abs(out[-1, 0] - q[0]) < 10.0
    assert abs(out[-1, 0] - cmds[-1, 0]) > 30.0
    assert out[-1, 0] > q[0]
    assert out[-1, 1] > q[1]
    assert out[-1, 2] > q[2]
    np.testing.assert_allclose(out[:, 3:], cmds[:, 3:])


def _jaw_pair_linear(q):
    """Wrist projection stand-in: J1 pans x, J2/J3 move y. Jaw span 40 px."""
    q = np.asarray(q, np.float64).reshape(-1)
    mid = np.array([112. + 3. * q[0], 140. - 2. * q[1] + 1. * q[2]])
    return np.stack([mid - [20., 0.], mid + [20., 0.]])


def test_wrist_r_servo_moves_jaw_midpoint_toward_cog_with_j1j3_only():
    q = np.array([0., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    cmds[:, :3] = [-40., -35., -30.]          # Actor reach is discarded
    target = np.array([40., 150.])            # fish left of the jaws (dx≈−0.32)
    out, diag = project_wrist_m_toward_pixel(q, cmds, _jaw_pair_linear, target, LOWER, UPPER)
    assert diag['applied'] is True and diag['kind'] == 'execution_geometry_wrist_r'
    assert diag['learned_policy_success'] is False and diag['token_rewritten'] is False
    assert diag['error_px_after_step'] < diag['error_px']
    assert out[-1, 0] < q[0]                   # J1 pans toward the left target
    assert abs(out[-1, 0] - q[0]) <= 6.0 + 1e-9
    np.testing.assert_allclose(out[:, 3:], cmds[:, 3:])   # J4–J6 untouched


def test_wrist_r_servo_deadzone_and_missing_projection():
    q = np.array([0., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    mid = _jaw_pair_linear(q).mean(0)
    out, diag = project_wrist_m_toward_pixel(q, cmds, _jaw_pair_linear, mid + [5., 5.], LOWER, UPPER)
    np.testing.assert_array_equal(out, cmds)
    assert diag['applied'] is False and diag['reason'] == 'within_deadzone'
    out, diag = project_wrist_m_toward_pixel(q, cmds, None, [40., 150.], LOWER, UPPER)
    np.testing.assert_array_equal(out, cmds)
    assert diag['reason'] == 'jaw_projection_unavailable'


def test_wrist_r_servo_rejects_steps_below_table():
    class _Low:
        def forward_anchors(self, q):
            q = np.asarray(q, np.float64).reshape(-1)
            # Any J1 change dives M below the table; J2/J3 stay high.
            z = -0.05 if abs(q[0]) > 1e-6 else 0.05
            mid = np.array([0.2, 0.1, z])
            return np.stack([mid, mid, mid])
    q = np.array([0., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    out, diag = project_wrist_m_toward_pixel(q, cmds, _jaw_pair_linear, [40., 150.], LOWER, UPPER,
                                             kinematics=_Low())
    assert abs(out[-1, 0] - q[0]) < 1e-9       # J1 step rejected as unsafe
    if diag['applied']:
        assert abs(out[-1, 1] - q[1]) > 0 or abs(out[-1, 2] - q[2]) > 0


def _jaw_pair_j1_invariant(q):
    """Real wrist camera: jaws do not move in-image when J1 pans."""
    q = np.asarray(q, np.float64).reshape(-1)
    mid = np.array([112., 140.])
    return np.stack([mid - [20., 0.], mid + [20., 0.]])


def test_wrist_r_j1_pan_when_jaws_fixed_and_fish_is_left():
    q = np.array([0., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    out, diag = project_wrist_m_toward_pixel(
        q, cmds, _jaw_pair_j1_invariant, [40., 140.], LOWER, UPPER, kinematics=_FkM())
    assert diag['applied'] is True
    assert diag['solver'] in ('j1_pan', 'lift_and_pan')
    assert out[-1, 0] < q[0]
    assert abs(out[-1, 0] - q[0]) <= WRIST_R_J1_MAX_STEP + 1e-9
    np.testing.assert_allclose(out[:, 3:], cmds[:, 3:])


def test_wrist_r_j1_pan_recenters_fish_on_the_right():
    q = np.array([10., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    out, diag = project_wrist_m_toward_pixel(
        q, cmds, _jaw_pair_j1_invariant, [190., 140.], LOWER, UPPER, kinematics=_FkM())
    assert diag['applied'] is True
    assert diag['solver'] in ('j1_pan', 'lift_and_pan')
    assert out[-1, 0] > q[0]


def test_pure_pan_preserves_safe_actor_reach_joints():
    q = np.array([0., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    cmds[:, 1] += np.linspace(.2, 1., 5)
    cmds[:, 2] -= np.linspace(.2, 1., 5)
    class Safe:
        def forward_anchors(self, state):
            mid = np.array([.2, .1, TABLE_Z_M + .05])
            return np.stack([mid, mid, mid])
    out, diag = project_wrist_m_toward_pixel(q, cmds, _jaw_pair_j1_invariant,
        [40., 100.], LOWER, UPPER, kinematics=Safe())
    assert out[-1, 0] < q[0]
    np.testing.assert_allclose(out[:, 1:3], cmds[:, 1:3])
    assert diag['actor_reach_preserved']


def test_pure_pan_does_not_preserve_actor_reach_below_floor():
    q = np.array([0., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1)); cmds[:, 1] += 1.
    class Unsafe:
        def forward_anchors(self, state):
            z = TABLE_Z_M + .05 if state[1] == q[1] else TABLE_Z_M - .01
            return np.tile([.2, .1, z], (3, 1))
    out, diag = project_wrist_m_toward_pixel(q, cmds, _jaw_pair_j1_invariant,
        [40., 100.], LOWER, UPPER, kinematics=Unsafe())
    np.testing.assert_allclose(out[:, 1], q[1])


def test_pure_pan_keeps_existing_roll_wrap_boundary():
    q = np.array([0., 10., 20., 30., -90., 32.])
    cmds = np.tile(q, (5, 1)); cmds[:, 4] = -99.
    out, _ = project_wrist_m_toward_pixel(q, cmds, _jaw_pair_j1_invariant,
        [40., 100.], LOWER, UPPER, kinematics=_FkM())
    assert (out[:, 4] >= -95.).all()


def test_wrist_r_servo_allows_xy_just_above_softened_floor():
    # 12 mm used to fail the 28 mm IBVS floor; 8 mm must admit the step.
    assert WRIST_R_TABLE_CLEARANCE_M <= 0.010

    class _NearTable:
        def forward_anchors(self, q):
            mid = np.array([0.2, 0.1, TABLE_Z_M + 0.012])
            return np.stack([mid, mid, mid])

    q = np.array([0., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    out, diag = project_wrist_m_toward_pixel(
        q, cmds, _jaw_pair_linear, [40., 150.], LOWER, UPPER, kinematics=_NearTable())
    assert diag['applied'] is True
    assert abs(out[-1, 0] - q[0]) > 1e-3


def test_trial_geometry_skips_m_align_until_grasp(monkeypatch):
    """Reach is Actor J1–J3; M-align stays off when not closing and M is far."""
    _geometry_monkey(monkeypatch, lambda q: 40.0)
    q = np.array([0., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    cmds[:, :3] = [-40., -35., -30.]
    cmds[:, 5] = 32.0
    target = np.array([0.40, 0.30, 0.20])
    out, diag = project_trial_geometry(q, cmds, _FkM(), LOWER, UPPER, target_xyz_m=target)
    assert diag['aim']['reason'] == 'not_grasping_yet'
    assert diag['aim']['applied'] is False
    assert diag['closing'] is False
    assert diag['m_error_at_close_m'] is None
    np.testing.assert_allclose(out[:, :3], cmds[:, :3])


def test_trial_geometry_records_m_error_when_closing(monkeypatch):
    _geometry_monkey(monkeypatch, lambda q: 40.0)
    q = np.array([0., 10., 20., 30., -70., 32.])
    cmds = np.tile(q, (5, 1))
    cmds[:, 5] = np.linspace(32., 8., 5)
    target = np.array([0.24, 0.12, 0.03])
    out, diag = project_trial_geometry(q, cmds, _FkM(), LOWER, UPPER, target_xyz_m=target)
    assert diag['closing'] is True
    assert isinstance(diag['m_error_at_close_m'], float)
    assert np.isfinite(diag['m_error_at_close_m'])
    assert out.shape == (5, 6)


def test_forcing_token_theta_is_not_this_path():
    from fcpg.v21_object_relative import encode_vector
    record = dict(centroid_px=[100., 100.], axis_px=[1., 0.], length_px=40., width_px=12., score=0.5)
    jaws = np.array([[80., 150.], [180., 150.]])
    vector = encode_vector(record, jaws)
    assert vector.shape == (8,) and vector[0] == 1
    # Token still reports the measured jaw-vs-axis angle; this module does not overwrite it.
    assert abs(vector[6] - 1.0) < 1e-5 and abs(vector[7]) < 1e-5


def test_descent_locked_base_cannot_steal_j1():
    q = np.array([0., 10., 20., 30., -70., 32.])
    commands = np.tile(q, (5, 1))
    class Linear:
        def forward_anchors(self, state):
            mid = np.array([.2 + .001*state[0], .1, .05 + .001*state[1]])
            return np.stack([mid, mid, mid])
    out, diagnostic = project_fk_m_toward_xyz(q, commands, Linear(),
        [.25, .1, .04], LOWER, UPPER, lock_j1=True)
    np.testing.assert_allclose(out[:, 0], q[0])
    assert diagnostic['applied'] and out[-1, 1] < q[1]
