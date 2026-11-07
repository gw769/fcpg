"""Training overhaul doc section 17.91: HOME lifts one joint first when the arm stopped just under the read-back floor
(fcpg/v21_home.lift_below_floor), on a simulated arm whose loaded elbow stays up to 3 units under its goal (10/24 the
real elbow needed its goal moved 3.5 units before it moved at all)."""
from types import SimpleNamespace
import json
import threading

import numpy as np
import pytest

from fcpg.v21_hardware_state import ORDER
from fcpg.v21_home import LIFT_STEP, lift_below_floor
from scripts.serve_so101_prod_v17_7 import V177ClearanceGate


class ElbowHeight:
    """The jaws' height is 95 - elbow (mm) with table_z_mm 0: a smaller elbow lifts them; no other joint does."""
    def forward_anchors(self, state):
        return np.asarray([[0., 0., (95. - float(np.asarray(state)[2])) / 1000.]] * 3)


class Event:
    def __init__(self, stop_on_wait=None):
        self.stopped, self.waits, self.stop_on_wait = False, 0, stop_on_wait

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, delay):
        self.waits += 1
        if self.stop_on_wait is not None and self.waits >= self.stop_on_wait:
            self.stopped = True
        return self.stopped


class Arm:
    def __init__(self, elbow_goal, elbow_present, torque=1, drift_pan=0.):
        self.goal = np.array([0., -60., elbow_goal, 40., -50., 30.])
        self.present = self.goal.copy()
        self.present[2] = elbow_present
        self.torque, self.drift_pan = torque, drift_pan
        self.writes, self.bus_reads = [], []

    def read_q(self):
        if self.present[2] - self.goal[2] > 3.:          # the loaded elbow trails its goal by up to 3 units
            self.present[2] = self.goal[2] + 3.
        if self.writes:
            self.present[0] = self.drift_pan * len(self.writes)
        return self.present.copy()


class Bus:
    def __init__(self, arm):
        self.arm = arm
        self.motors = {name: SimpleNamespace(id=index + 1) for index, name in enumerate(ORDER)}

    def sync_read(self, register, normalize=False, num_retry=0):
        self.arm.bus_reads.append(register)
        values = dict(Present_Position=self.arm.present, Goal_Position=self.arm.goal,
                      Torque_Enable=[1, 1, self.arm.torque, 1, 1, 1], Moving=[0] * 6)[register]
        return {name: int(round(float(value))) for name, value in zip(ORDER, values)}

    def _normalize(self, raw):
        return {key: float(value) for key, value in raw.items()}


def owner_for(arm, event=None):
    event = Event() if event is None else event

    def write_q(command, previous):
        if event.is_set():
            raise RuntimeError('operator STOP is latched before motor write')
        arm.writes.append((np.asarray(command, np.float64).copy(), np.asarray(previous, np.float64).copy()))
        arm.goal = np.asarray(command, np.float64).copy()
        return command

    gate = V177ClearanceGate(kinematics=ElbowHeight(), table_z_mm=0., planned_clearance_mm=8.,
                             readback_clearance_mm=-10., physical_floor_mm=-10.)
    return SimpleNamespace(clearance=gate, _read_q=arm.read_q, _write_q=write_q, _lease_check=lambda: None,
                           bus_lock=threading.RLock(), stop_event=event,
                           io=SimpleNamespace(follower=SimpleNamespace(bus=Bus(arm))))


def test_on_or_above_the_floor_nothing_is_read_or_moved(tmp_path):
    arm = Arm(elbow_goal=100., elbow_present=102.)              # -7 mm: above the -10 mm floor
    assert lift_below_floor(owner_for(arm), tmp_path) is None
    assert not arm.writes and not arm.bus_reads and not list(tmp_path.iterdir())


def test_lifts_only_the_chosen_joint_from_its_goal_register_until_the_target(tmp_path):
    arm = Arm(elbow_goal=104., elbow_present=106.)              # -11 mm, the goal 2 units above the measured pose
    record = lift_below_floor(owner_for(arm), tmp_path)
    assert record['outcome'] == 'lifted' and record['joint'] == 'elbow_flex' and record['planned_change'] < 0
    assert record['start_clearance_mm'] == pytest.approx(-11.) and record['end_clearance_mm'] >= 5.
    first, previous = arm.writes[0]
    assert first[2] == pytest.approx(104. - LIFT_STEP)         # from the Goal register, not from the read-back
    for command, before in arm.writes:
        np.testing.assert_allclose(np.delete(command, 2), np.delete(arm.goal, 2))
        assert command[2] == pytest.approx(before[2] - LIFT_STEP)
    assert not any(step['other_joints_moved'] for step in record['steps'])
    saved = json.loads((tmp_path / record['path'].split('/')[-1]).read_text())
    assert saved['outcome'] == 'lifted' and len(saved['steps']) == len(arm.writes)


def test_too_far_under_the_floor_is_left_to_a_person(tmp_path):
    arm = Arm(elbow_goal=118., elbow_present=120.)              # -25 mm
    with pytest.raises(RuntimeError, match='a person must free the arm'):
        lift_below_floor(owner_for(arm), tmp_path)
    assert not arm.writes and not arm.bus_reads
    assert json.loads(next(tmp_path.iterdir()).read_text())['outcome'] == 'refused'


def test_no_motion_without_every_torque_on(tmp_path):
    arm = Arm(elbow_goal=104., elbow_present=106., torque=0)
    with pytest.raises(RuntimeError, match='torque'):
        lift_below_floor(owner_for(arm), tmp_path)
    assert not arm.writes


def test_another_joint_moving_stops_the_lift(tmp_path):
    arm = Arm(elbow_goal=104., elbow_present=106., drift_pan=4.)
    with pytest.raises(RuntimeError, match='another joint moved'):
        lift_below_floor(owner_for(arm), tmp_path)
    assert len(arm.writes) == 1


def test_stop_during_the_lift_ends_it(tmp_path):
    arm = Arm(elbow_goal=104., elbow_present=106.)
    with pytest.raises(RuntimeError, match='STOP'):
        lift_below_floor(owner_for(arm, Event(stop_on_wait=3)), tmp_path)
    assert len(arm.writes) == 3
    assert json.loads(next(tmp_path.iterdir()).read_text())['outcome'] == 'stopped'


def test_a_joint_that_never_follows_stops_after_the_stall_limit(tmp_path):
    arm = Arm(elbow_goal=104., elbow_present=106.)
    arm.read_q = lambda: np.r_[0., -60., 106., 40., -50., 30.]   # stuck: the elbow never moves
    with pytest.raises(RuntimeError, match='raised nothing'):
        lift_below_floor(owner_for(arm), tmp_path)
