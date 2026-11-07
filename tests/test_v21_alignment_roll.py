"""align_roll (11/03, James 「它现在不会转J6阿。全部版本阿」, 「wrist 和 external都要看吧？」): with the jaws open and aimed at
the fish, the alignment CEM also turns the wrist roll so the jaws close across the fish: the fish's axis from the column
picture and the jaw line by forward kinematics, both on the table."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from fcpg.v21_alignment_cem import (ARM, ARM_JOINT_NAMES, COLUMN_SCHEMA, MAX_ROLL_OFFSET, ORIENT_TOLERANCE_PX, ROLL,
                                    AlignmentCEM, AlignmentCEMAdapter, ColumnCamera, GeometricResponse, WristResponse,
                                    across_deg, arc_from_across, fish_on_table, load_column_camera)

JACOBIAN4 = np.asarray([[-5.5, -1.5, -2.0, -1.2],
                        [5.0, -2.3, -3.9, -3.6]])          # the 10/11 fit's four searched joints (as test_v21_cem_teacher)
RATE = .0178                                               # rad of the wrist picture per roll unit (the geometry, 11/03)
DEG = 1.69                                                 # degrees of the jaw line per roll unit (the robot's calibration)
SCALE = 1500.                                              # wrist px per metre on the table at the fish
Q = np.asarray([-24.6, 26.2, 3.9, 53.9, -46.5, 30.])


class Arm:
    """The jaw midpoint moves with lift (x), pan (y) and elbow (z); the jaw line turns DEG degrees per roll unit about
    the vertical; the fingertips are 1 mm apart per gripper unit."""
    def forward_anchors(self, states):
        s = np.asarray(states, np.float64).reshape(-1, 6)
        centre = np.stack((.25 + .001 * s[:, 1], .001 * s[:, 0], .05 + .001 * s[:, 2]), 1)
        yaw = np.radians(DEG * s[:, ROLL])
        half = .0005 * s[:, 5:6] * np.stack((np.cos(yaw), np.sin(yaw), np.zeros(len(s))), 1)
        return np.stack((centre, centre + half, centre - half), 1)


def camera():
    """Looking straight down from 1 m above the base: 4.4 mm of the table per letterbox224 pixel at the fish."""
    return ColumnCamera(np.r_[np.pi, 0., 0., 0., 0., 1., np.log(224.)], (224, 224), (224, 224), 0., Arm())


@dataclass(frozen=True)
class TurningGeometry(GeometricResponse):
    """The roll turns every point of the picture about the jaw midpoint by RATE per unit; the other joints shift every
    point alike."""
    def local(self, q, offset_px):
        x, y = np.asarray(offset_px, np.float64)
        return WristResponse(np.c_[JACOBIAN4, RATE * np.asarray((-y, x))], 0, 'turning')

    def table_scale(self, q, offset_px):
        return SCALE


def geometry():
    return TurningGeometry(np.r_[200., np.zeros(11)], None)


def follows_commands(q):
    """An ideal LeWM: every read-back is the command before it."""
    return lambda commands: np.concatenate((np.repeat(q[None, None], len(commands), 0), np.asarray(commands)), 1)


def token(dx_px=0., dy_px=0.):
    return np.asarray([1., 1., dx_px / 224., dy_px / 224., .3, .1, 1., 0.])


def fish(degrees_from_the_jaw_line, length_m=.08):
    """A fish on the table lying this far (degrees) from the jaw line at Q."""
    yaw = np.radians(DEG * Q[ROLL] + degrees_from_the_jaw_line)
    return dict(centre=np.zeros(2), direction=np.asarray((np.cos(yaw), np.sin(yaw))), length_m=length_m)


def proposal(grip_last=30.):
    block = np.repeat(Q[None], 5, 0)
    block[:, 5] = np.linspace(Q[5], grip_last, 5)
    return block


def cem(**options):
    return AlignmentCEM(geometry(), column_camera=camera(), seed=3, **options)


def plan(planner, axis, *, grip_last=30., seen=True, fish_point=None):
    return planner.plan(Q, proposal(grip_last), token() if seen else None, follows_commands(Q), lambda c: True,
                        commit=False, fish_axis=axis, fish_point=fish_point)


def outline(cog=(112., 112.), axis=(1., 0.), length=20., width=5.):
    return dict(visible=True, cog_px=list(cog), axis_px=list(axis), length_px=length, width_px=width)


def test_the_fish_lies_on_the_table_from_its_column_outline():
    on_table = fish_on_table(camera(), outline())
    assert on_table['length_m'] == pytest.approx(20 * .985 / 224)                  # 88 mm
    assert np.allclose(np.abs(on_table['direction']), [1., 0.]) and np.allclose(on_table['centre'], 0., atol=1e-9)
    upright = fish_on_table(camera(), outline(axis=(0., 1.)))                       # the picture's y is the table's -y
    assert np.allclose(np.abs(upright['direction']), [0., 1.])
    assert fish_on_table(camera(), outline(length=40.)) is None                    # 176 mm: more than the fish
    assert fish_on_table(camera(), outline(width=18.)) is None                     # too round for an axis
    assert fish_on_table(camera(), dict(visible=False)) is None and fish_on_table(camera(), None) is None


def test_the_arc_from_across_and_the_angle_across():
    lines = np.asarray([[1., 0.], [0., 1.], [np.sqrt(.5), np.sqrt(.5)], [np.cos(.05), np.sin(.05)]])
    along_x = dict(direction=np.asarray((1., 0.)), length_m=.08)
    # the ends' arc to lying across: even per degree, near a fish lying along the jaws too
    assert np.allclose(arc_from_across(along_x, lines), [.04 * np.pi / 2, 0., .04 * np.pi / 4, .04 * (np.pi / 2 - .05)])
    assert np.allclose(across_deg(along_x, lines), [0., 90., 45., np.degrees(.05)])
    line = camera().jaw_line(np.r_[Q[:5], 2.][None])                               # the jaws shut: still their line
    assert np.allclose(line, [[np.cos(np.radians(DEG * Q[ROLL])), np.sin(np.radians(DEG * Q[ROLL]))]])


def test_without_the_switch_the_roll_stays_the_actors():
    selected, record = plan(cem(), fish(0.))
    assert np.array_equal(selected[:, ROLL], proposal()[:, ROLL]) and 'roll' not in record


def test_a_fish_lying_off_the_normal_turns_the_roll_to_lay_the_jaws_across():
    selected, record = plan(cem(align_roll=True), fish(70.))      # the nearer normal is 20 degrees back
    roll = record['roll']
    assert roll['turn_rad'] == pytest.approx(np.radians(-20.))
    assert roll['rate_rad_per_unit'] == pytest.approx(np.radians(DEG))
    assert roll['across_deg_proposal'] == pytest.approx(70.)
    turned = selected[-1, ROLL] - proposal()[-1, ROLL]
    assert turned == pytest.approx(-20. / DEG, abs=2.)
    assert roll['across_deg_selected'] > 85. and roll['selected_arc'] < roll['proposal_arc'] - 5.
    assert not record['selected_is_actor'] and roll['units'] == 'px'
    # the other side of the normal turns the other way
    selected, record = plan(cem(align_roll=True), fish(110.))
    assert selected[-1, ROLL] - proposal()[-1, ROLL] == pytest.approx(20. / DEG, abs=2.)


def test_a_fish_lying_along_the_jaws_turns_as_far_as_a_block_may():
    selected, record = plan(cem(align_roll=True), fish(5.))       # the nearer normal 85 degrees back
    assert selected[-1, ROLL] - proposal()[-1, ROLL] == pytest.approx(-MAX_ROLL_OFFSET, abs=1e-6)
    assert record['roll']['across_deg_selected'] > record['roll']['across_deg_proposal'] + 25.


def test_a_fish_lying_across_the_jaws_keeps_the_roll():
    selected, record = plan(cem(align_roll=True), fish(90.))
    assert record['roll']['proposal_arc'] == pytest.approx(0., abs=1e-6)
    assert abs(selected[-1, ROLL] - proposal()[-1, ROLL]) < 2.


def test_a_close_waits_while_the_fish_lies_along_the_jaws():
    planner = cem(align_roll=True)
    selected, record = plan(planner, fish(0.), grip_last=10.)
    assert record['closing'] and record['close_check_cost_px'] <= planner.close_tolerance_px
    assert record['roll']['close_arc'] > ORIENT_TOLERANCE_PX
    assert record['close_deferred'] and np.allclose(selected[:, 5], Q[5])          # the jaws held open
    selected, record = plan(cem(align_roll=True), fish(90.), grip_last=10.)
    assert record['closing'] and not record['close_deferred']


def test_the_column_leading_the_jaws_turns_the_roll_on_the_way():
    jaw = camera().jaw(Q[None])[0]
    selected, record = plan(cem(align_roll=True), fish(60.), seen=False, fish_point=jaw + np.r_[.02, .01, 0.])
    assert record['aimed_by'] == 'column_fish' and record['roll']['units'] == 'mm'
    assert record['roll']['scale_per_m'] == 1000.
    assert selected[-1, ROLL] - proposal()[-1, ROLL] < -5.


def test_the_column_leads_while_the_jaws_are_far_from_its_fish_whatever_the_wrist_reports():
    # 11/03 rollb trials 2-5: the wrist finder reported a fish in every picture while the column fish lay far away
    jaw = camera().jaw(Q[None])[0]
    selected, record = plan(cem(align_roll=True), fish(60.), fish_point=jaw + np.r_[.15, .08, 0.])
    assert record['aimed_by'] == 'column_fish' and record['column_leads_far'] and record['units'] == 'mm'
    assert record['column_fish_from_jaws_mm'] == pytest.approx(170., abs=1.)
    # near it, the wrist aims the jaws (and turns the roll) as before
    selected, record = plan(cem(align_roll=True), fish(60.), fish_point=jaw + np.r_[.05, .03, 0.])
    assert record['aimed_by'] == 'wrist' and not record['column_leads_far'] and record['roll']['units'] == 'px'
    # without a column fish the wrist aims as before
    selected, record = plan(cem(align_roll=True), fish(60.))
    assert record['aimed_by'] == 'wrist' and record['column_fish_from_jaws_mm'] is None


def test_the_search_reaches_further_while_the_column_leads_far_jaws():
    from fcpg.v21_alignment_cem import FAR_SCALE, MAX_OFFSET
    jaw = camera().jaw(Q[None])[0]
    selected, record = plan(cem(), None, seen=False, fish_point=jaw + np.r_[.0, .3, 0.])     # 300 mm along the pan
    assert record['aimed_by'] == 'column_fish' and record['search_scale'] == FAR_SCALE
    moved = selected[-1, 0] - proposal()[-1, 0]
    assert MAX_OFFSET[0] < abs(moved) <= FAR_SCALE * MAX_OFFSET[0] + 1e-9
    selected, record = plan(cem(), None, seen=False, fish_point=jaw + np.r_[.0, .05, 0.])    # near: as before
    assert record['search_scale'] == 1. and abs(selected[-1, 0] - proposal()[-1, 0]) <= MAX_OFFSET[0] + 1e-9


def test_a_close_far_from_the_column_fish_waits():
    jaw = camera().jaw(Q[None])[0]
    selected, record = plan(cem(align_roll=True), fish(90.), grip_last=10., fish_point=jaw + np.r_[.15, .08, 0.])
    assert record['closing'] and record['aimed_by'] == 'column_fish'
    assert record['close_deferred'] and np.allclose(selected[:, 5], Q[5])          # the jaws held open


def test_the_switch_needs_the_geometry_and_the_column_camera():
    with pytest.raises(ValueError):
        AlignmentCEM(WristResponse(np.c_[JACOBIAN4, np.zeros(2)], 0, 'measured'), align_roll=True,
                     column_camera=camera())
    with pytest.raises(ValueError):
        AlignmentCEM(geometry(), align_roll=True)
    with pytest.raises(ValueError):
        AlignmentCEM(geometry(), align_roll=1, column_camera=camera())


def test_no_fish_axis_leaves_the_roll_and_says_why():
    selected, record = plan(cem(align_roll=True), None)
    assert record['roll'] == dict(reason='no_fish_axis') and np.array_equal(selected[:, ROLL], proposal()[:, ROLL])
    assert ARM == 5


class Service:
    def __init__(self):
        self.targets = None

    def show(self, column):
        self.targets = dict(targets=dict(pickup=dict(column=column)))


def test_the_axis_is_kept_from_the_last_outline_of_the_whole_fish():
    adapter = AlignmentCEMAdapter.__new__(AlignmentCEMAdapter)
    adapter.planner, adapter.column_camera, adapter.clock = cem(align_roll=True), camera(), iter(range(100)).__next__
    adapter.service, adapter.fish_axis, adapter.fish_length_m = Service(), None, None
    adapter.service.show(dict(visible=False))
    assert adapter._fish_axis() == (None, dict(outline='no_outline', length_mm=None, trial_length_mm=None,
                                                kept_from_s=None))
    adapter.service.show(outline())                                             # the trial's first: 88 mm along x
    axis, note = adapter._fish_axis()
    assert note['outline'] == 'first' and np.allclose(np.abs(axis['direction']), [1., 0.])
    adapter.service.show(outline(axis=(0., 1.), length=10.))                    # half hidden by the arm: kept
    axis, note = adapter._fish_axis()
    assert note['outline'] == 'length_disagrees' and np.allclose(np.abs(axis['direction']), [1., 0.])
    adapter.service.show(outline(axis=(0., 1.), length=22.))                    # pushed, whole: taken
    axis, note = adapter._fish_axis()
    assert note['outline'] == 'taken' and np.allclose(np.abs(axis['direction']), [0., 1.])
    assert note['trial_length_mm'] == pytest.approx(20 * .985 / 224 * 1e3)


def test_the_jaws_are_led_to_the_kept_whole_fish_not_to_a_jump():
    # 11/03 rolld trial 1: with the arm in the column picture the outline's centre jumped to the robot's own parts
    adapter = AlignmentCEMAdapter.__new__(AlignmentCEMAdapter)
    adapter.planner, adapter.column_camera, adapter.clock = cem(), camera(), iter(range(100)).__next__
    adapter.service, adapter.fish_axis, adapter.fish_length_m = Service(), None, None
    adapter.service.show(outline(cog=(112., 112.)))
    assert adapter._fish_axis()[0] is None                       # no align_roll: the roll gets nothing ...
    assert np.allclose(adapter._column_fish_point()[:2], 0., atol=1e-9)      # ... the jaws are led to the fish
    adapter.service.show(outline(cog=(40., 40.), length=8.))                 # the arm's parts: 35 mm long
    adapter._fish_axis()
    assert np.allclose(adapter._column_fish_point()[:2], 0., atol=1e-9)      # kept where the whole fish was
    adapter.service.show(outline(cog=(122., 112.)))                          # pushed 44 mm, whole: followed
    adapter._fish_axis()
    assert adapter._column_fish_point()[0] == pytest.approx(10 * .985 / 224)
    adapter.fish_axis = None                                                  # no whole fish yet: nothing leads
    adapter.service.show(outline(cog=(40., 40.), length=8.))
    assert adapter._column_fish_point() is None


MODEL = Path(__file__).resolve().parents[1] / 'third_party/so_arm100/Simulation/SO101/so101_new_calib.xml'


def unit(v):
    return v / np.linalg.norm(v)


def test_the_hinges_turn_the_jaws_past_the_model_range():
    pytest.importorskip('mujoco')
    from fcpg.so101_kinematics import SO101MuJoCoKinematics, normalized_positions_to_mujoco_qpos
    kinematics = SO101MuJoCoKinematics(MODEL)
    states = np.asarray([Q, [0., -97.6, 98.8, 62.7, -51.3, 30.]])
    qpos = normalized_positions_to_mujoco_qpos(states, kinematics.all_joint_ranges)
    assert np.allclose(kinematics.forward_anchors_qpos(qpos), kinematics.forward_anchors(states))
    edge, beyond = qpos[:1].copy(), qpos[:1].copy()
    edge[0, ROLL] = kinematics.all_joint_ranges[ROLL][0]
    beyond[0, ROLL] = edge[0, ROLL] - .2
    tips = [a[0, 1] - a[0, 2] for a in (kinematics.forward_anchors_qpos(edge), kinematics.forward_anchors_qpos(beyond))]
    assert np.degrees(np.arccos(unit(tips[0]) @ unit(tips[1]))) == pytest.approx(np.degrees(.2), abs=.5)


def test_the_column_camera_jaw_line_reads_the_robots_whole_roll(tmp_path):
    pytest.importorskip('mujoco')
    from fcpg.so101_kinematics import SO101MuJoCoKinematics
    ranges = np.asarray(SO101MuJoCoKinematics(MODEL).all_joint_ranges)[:ARM]
    # the model's ranges in motor counts, the roll as wide as the robot's (11/03: 122-3972, +-169 degrees)
    counts = 2047. + ranges * 4096. / (2 * np.pi)
    calibration = {name: dict(range_min=float(low), range_max=float(high)) for name, (low, high) in
                   zip(ARM_JOINT_NAMES, counts)}
    calibration['wrist_roll'] = dict(range_min=122, range_max=3972)
    (tmp_path / 'robot.json').write_text(json.dumps(calibration))
    (tmp_path / 'camera.json').write_text(json.dumps(dict(
        schema=COLUMN_SCHEMA, model=str(MODEL), conversion='robot_calibration',
        robot_calibration=str(tmp_path / 'robot.json'), params=[np.pi, 0., 0., 0., 0., 1., float(np.log(224.))],
        working_size=[224, 224], raw_size=[224, 224], table_z_m=0.)))
    camera = load_column_camera(tmp_path / 'camera.json')
    within = np.asarray([[0., -97.6, 98.8, 62.7, -51.3, 30.]])
    hinges = camera.to_hinges(within)
    assert hinges[0, ROLL] == pytest.approx(np.deg2rad((122 + (within[0, ROLL] + 100) / 200 * 3850 - 2047) * 360 / 4096))
    assert np.allclose(camera.kinematics.forward_anchors_qpos(hinges), camera.kinematics.forward_anchors(
        camera.to_model_units(within)))
    # roll -99 is 167.5 degrees on this robot, past the model's 157: the line still turns there
    rolls = np.repeat(within, 3, 0)
    rolls[:, ROLL] = (-90., -95., -99.)
    lines = camera.jaw_line(rolls)
    turns = np.degrees(np.arccos(np.clip(np.sum(lines[1:] * lines[:-1], 1), -1., 1.)))
    assert np.all(turns > 4.)


def test_the_wrist_reference_is_the_column_s_far_from_the_fish_and_its_own_near_it():
    from types import SimpleNamespace
    column_reference = object()
    encoder = SimpleNamespace(reference_of=column_reference)
    adapter = AlignmentCEMAdapter.__new__(AlignmentCEMAdapter)
    adapter.planner, adapter.column_camera, adapter.clock = cem(), camera(), iter(range(100)).__next__
    adapter.service, adapter.fish_axis, adapter.fish_length_m = Service(), None, None
    adapter.service.owner = SimpleNamespace(object_relative=encoder)
    adapter.column_reference_of = column_reference
    jaw = camera().jaw(Q[None])[0]
    adapter.fish_axis = dict(point=jaw + np.r_[.3, 0., 0.], centre=(jaw + np.r_[.3, 0., 0.])[:2],
                             direction=np.r_[1., 0.], length_m=.08, measured_s=0.)
    assert adapter._choose_wrist_reference(Q) == 'column' and encoder.reference_of is column_reference
    adapter.fish_axis = dict(adapter.fish_axis, point=jaw + np.r_[.05, 0., 0.])          # 50 mm: the wrist's own
    assert adapter._choose_wrist_reference(Q) == 'wrist' and encoder.reference_of is None
    adapter.fish_axis = dict(adapter.fish_axis, point=jaw + np.r_[.3, 0., 0.])
    shut = Q.copy()
    shut[5] = 5.                                                                         # jaws shut: the wrist's own
    assert adapter._choose_wrist_reference(shut) == 'wrist' and encoder.reference_of is None
    assert adapter._choose_wrist_reference(Q) == 'column' and encoder.reference_of is column_reference
    adapter.column_reference_of = None                                                   # no text finder: no choice
    assert adapter._choose_wrist_reference(Q) is None


def test_a_close_at_the_fish_off_the_jaw_centre_is_lowered_and_held_while_high():
    # 11/03 rollf trial 1: with the fish 35 px from the jaw centre the Actor's close went ahead 45-76 mm above it
    from fcpg.v21_alignment_cem import CLOSE_HIGH_MM
    heights = lambda states: 50. + np.asarray(states, np.float64).reshape(-1, 6)[:, 2]     # mm; the elbow lowers them
    selected, record = cem().plan(Q, proposal(10.), token(35., 0.), follows_commands(Q), lambda c: True, commit=False,
                                  jaw_height=heights)
    assert record['closing'] and isinstance(record.get('descend'), dict)
    assert record['descend']['proposal_close_height_mm'] == pytest.approx(53.9)
    assert record['descend']['selected_close_height_mm'] < record['descend']['proposal_close_height_mm']   # lowered
    assert record['descend']['selected_close_height_mm'] > CLOSE_HIGH_MM
    assert record['close_deferred'] and np.allclose(selected[:, 5], Q[5])              # the jaws held open


def test_a_slow_close_across_the_open_line_is_a_close_and_a_high_close_lowers_further():
    from fcpg.v21_alignment_cem import FAR_SCALE, OPEN_GRIP
    heights = lambda states: 50. + np.asarray(states, np.float64).reshape(-1, 6)[:, 2]
    q = Q.copy()
    q[5] = OPEN_GRIP + 1.9                                       # half shut; the Actor takes it 2 units further
    block = np.repeat(q[None], 5, 0)
    block[:, 5] = np.linspace(q[5], OPEN_GRIP - .1, 5)
    selected, record = cem().plan(q, block, token(0., 0.), follows_commands(q), lambda c: True, commit=False,
                                  jaw_height=heights)
    assert record['closing'] and record['close_deferred'] and np.allclose(selected[:, 5], q[5])
    assert record['search_scale'] == FAR_SCALE                 # 53.9 mm up: the lowering may reach further
