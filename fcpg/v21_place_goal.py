"""Where the object goes, as a goal for the arm (training overhaul doc section 17.7, the B route).

The goal is a point in the robot's base frame: the jaw midpoint at the moment the object is let go. The Actor reads
it relative to where its own jaws are now, in the arm's polar frame about the pan axis (8 numbers, the token size of
the other cues):

    [available, d_azimuth, d_reach, 0, 0, 0, 0, 0]
    d_azimuth  turn about the pan axis from the jaws to the goal, radians, in (-pi, pi]
    d_reach    horizontal distance from the pan axis, goal minus jaws, per 0.1 m
The height is left out on purpose: how high to let go is the task's (the demonstrations teach it), and live the box's
height is the least certain part of where it is seen.

Training labels are hindsight (scripts/build_v21_place_goal_tables.py): in a recording that picks something up,
carries it and lets it go, every frame up to the release gets the jaw midpoint at the release (the robot model on the
recorded joints); frames after the release and recordings without such a carry get zeros. Live, the goal point comes
from where the box is seen (section 17.7). The robot model's arm is symmetric about the pan-zero plane (pan -> -pan
is fcpg/v21_mirror.py's mirror), so mirroring flips d_azimuth only (GOAL_SIGN). No robot I/O, no learned state.

Section 17.12 labels the goal only while the jaws hold the object ('while_held', tables v3): in our demonstrations the
fish always lies on the side away from the box, so a goal before the grasp points against the way the arm turns. The
hold comes from the gripper read-back alone (HoldTracker), the same class for the labels of our demonstrations and on
the robot; a model records which labels it learned (goal_availability), and the service gives the goal accordingly.

Section 17.14 ('target', tables v4) makes the token the arm's target throughout: before the hold it points at the
object to grasp, with GRASP (the fourth number) set to 1; while held it is the place goal of section 17.12, GRASP 0;
after the release, zeros. Archive labels: the jaw midpoint where the detected hold starts, for the frames before it;
live, the fish the column camera sees at the start of the trial (LivePlaceGoal.observe_fish). The Actor learned where
the fish is only from our two demonstrations (the Archive has no such input); this teaches it from every Archive
recording that picks something up.
"""
import numpy as np

GOAL_SIGN = (1., -1., 1., 1., 1., 1., 1., 1.)   # [available, d_azimuth, d_reach, grasp, 0, 0, 0, 0]
GRASP = 3            # the token's number that marks a grasp target (section 17.14); zero for every place goal
UNIT_M = .1
SCHEMA = 'v21-place-goal-v2'


class PanFrame:
    """The arm's polar frame: the pan axis located from the robot model by turning the pan joint alone."""

    def __init__(self, kinematics):
        self.kinematics = kinematics
        base = np.array([0., -20., 20., 40., 0., 10.])
        poses = np.repeat(base[None], 9, 0)
        poses[:, 0] = np.linspace(-80., 80., 9)
        xy = np.asarray(kinematics.forward_anchors(poses))[:, 0, :2]
        fit = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], (xy ** 2).sum(1), rcond=None)[0]
        self.axis_xy = np.asarray(fit[:2], np.float64)
        radius = np.linalg.norm(xy - self.axis_xy, axis=1)
        if np.ptp(radius) > 1e-4:
            raise ValueError('the jaws do not turn on a circle about one pan axis')

    def jaws(self, q):
        """Jaw midpoint [..., 3] in the base frame for joint readings [..., 6] (normalized units)."""
        return np.asarray(self.kinematics.forward_anchors(np.asarray(q, np.float64)))[..., 0, :]

    def polar(self, xyz):
        xyz = np.asarray(xyz, np.float64)
        d = xyz[..., :2] - self.axis_xy
        return np.arctan2(d[..., 1], d[..., 0]), np.hypot(d[..., 0], d[..., 1]), xyz[..., 2]

    def goal_vectors(self, jaws_xyz, goal_xyz):
        """[..., 8] tokens for jaws [..., 3] and goal [..., 3] (broadcast)."""
        jaw_az, jaw_r, _ = self.polar(jaws_xyz)
        goal_az, goal_r, _ = self.polar(goal_xyz)
        turn = np.angle(np.exp(1j * (goal_az - jaw_az)))
        shape = np.broadcast(turn, goal_r - jaw_r).shape
        out = np.zeros((*shape, 8), np.float32)
        out[..., 0] = 1.
        out[..., 1] = turn
        out[..., 2] = (goal_r - jaw_r) / UNIT_M
        return out


def hindsight_goals(frame, q, release):
    """[n, 8] labels for one recording's joint readings q [n, 6]: the jaw midpoint at frame `release` for every frame
    up to and including it, zeros after it (and everywhere when release is None)."""
    q = np.asarray(q, np.float64)
    out = np.zeros((len(q), 8), np.float32)
    if release is None:
        return out
    release = int(release)
    if not 0 <= release < len(q):
        raise ValueError('release frame outside the recording')
    goal = frame.jaws(q[release])
    out[:release + 1] = frame.goal_vectors(frame.jaws(q[:release + 1]), goal)
    return out


def goals_while_held(frame, q, start, release):
    """[n, 8] labels for one recording's joint readings q [n, 6] (section 17.12): the jaw midpoint at frame `release`
    for the held frames start..release-1, zeros everywhere else."""
    q = np.asarray(q, np.float64)
    out = np.zeros((len(q), 8), np.float32)
    start, release = int(start), int(release)
    if not 0 <= start < release < len(q):
        raise ValueError('the hold must lie inside the recording and end in a release')
    out[start:release] = frame.goal_vectors(frame.jaws(q[start:release]), frame.jaws(q[release]))
    return out


def targets_around_hold(frame, q, start, release):
    """[n, 8] labels for one recording's joint readings q [n, 6] (section 17.14): the grasp target, the jaw midpoint
    at frame `start` with GRASP set, for frames 0..start-1; the place goal of goals_while_held for start..release-1;
    zeros after."""
    out = goals_while_held(frame, q, start, release)
    q = np.asarray(q, np.float64)
    if start:
        out[:start] = frame.goal_vectors(frame.jaws(q[:start]), frame.jaws(q[start]))
        out[:start, GRASP] = 1.
    return out


def mirror_goal(goal):
    """The mirrored recording's goal token: d_azimuth changes sign (fcpg/v21_mirror.py, pan -> -pan)."""
    return np.asarray(goal, np.float32) * np.asarray(GOAL_SIGN, np.float32)


BOX_WALL_M = .07     # the box wall (10/16 column picture); the drop zone as the column camera sees it sits at half of it
OPEN_GRIP = 15.      # jaws count as shut below this read-back (fcpg/v21_alignment_cem.OPEN_GRIP)
MIN_SAMPLES = 3      # column detections in the drop goal's median before the box is taken
FISH_SAMPLES = 6     # column pictures of the fish in the grasp target's median, then it is kept for the trial
FISH_SOURCE = None   # the column fish record's source, set by the service from its object finder (10/31: the operator
                     # colour body this was is gone from the code; without a finder no record is taken)
AVAILABILITY = ('until_release', 'while_held', 'target')   # the goal tables' labels: section 17.7 (v2), 17.12 (v3), 17.14 (v4)


class HoldTracker:
    """Whether the jaws hold the object, from the gripper read-back alone (training overhaul doc section 17.12), with
    the threshold the alignment CEM and the done check use (OPEN_GRIP): the hold starts the first time the read-back
    falls below it after having been at or above it in this trial (the jaws opened for the grasp, then shut on the
    object) and ends for the rest of the trial when it is at or above it again (the release); a later shut (the jaws
    closing for the way home) starts nothing. Our demonstrations hold the fish at 9.6-12.2 and open to 21-32 to let go;
    on the robot H1b holds at 7-14 but lets go opening only to 15-19 (10/16 control sessions), so a higher release
    threshold misses its releases. A rest pose just above OPEN_GRIP that dips below it would count as a grasp: the
    robot rests at about 30, demonstration 0 at 16.0-16.2 without dipping."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.opened = self.held = self.released = False

    def update(self, grip):
        """Feed the next read-back; True while the jaws hold the object."""
        shut = float(grip) < OPEN_GRIP
        if not self.opened:
            self.opened = not shut
        elif not self.held:
            self.held = shut
        elif not self.released:
            self.released = not shut
        return self.held and not self.released


def hold_interval(grip):
    """(first held frame, release frame) of a recorded read-back sequence under HoldTracker, or None when there is no
    hold that ends in a release."""
    tracker, start = HoldTracker(), None
    for i, value in enumerate(np.asarray(grip, np.float64)):
        holding = tracker.update(value)
        if holding and start is None:
            start = i
        elif start is not None and not holding:
            return start, i
    return None


def goal_availability(model_root):
    """When the place goal was labelled in the tables a model learned from: 'until_release' (section 17.7, tables v2,
    which carry no 'availability'), 'while_held' (section 17.12) or 'target' (section 17.14: a grasp target before the
    hold), from the run's config.json protocol."""
    import json
    from pathlib import Path
    config = json.loads((Path(model_root) / 'config.json').read_text())
    manifest = ((config.get('protocol') or {}).get('place_goal') or {}).get('manifest')
    if not isinstance(manifest, dict):
        raise ValueError('the model records no place-goal tables')
    value = manifest.get('availability', 'until_release')
    if value not in AVAILABILITY:
        raise ValueError(f'unknown place-goal availability {value!r}')
    return value


class LivePlaceGoal:
    """The place goal on the robot (training overhaul doc section 17.7): the box located once per trial with the column
    camera, the goal token recomputed from the arm's pose at every capture.

    The box pixel is the service's drop goal (the median of the column DINO drop-zone detections since HOME); it is cast
    onto the plane at half the box wall with the robot-calibrated column camera (fcpg/v21_alignment_cem.ColumnCamera)
    and dropped to the table, as the 10/16 carry did (its three trials into the box: 2-4 cm from where the jaws let go,
    live_goal_accuracy_check.py), once the median has MIN_SAMPLES detections. It is frozen when the jaws first shut
    after having been open in this trial, the grasp (from then on the arm hides the box from the column camera), and
    forgotten at HOME. The jaws come from the same camera's robot-calibrated
    kinematics; the turn is converted to the robot model's pan units, in which the Actor learned it.
    With while_held (section 17.12, for a model whose goal tables say so: goal_availability) the token is zeros except
    while the jaws hold the object (HoldTracker, as the labels of our demonstrations), and the box is frozen when the
    hold starts.
    With grasp_target as well (section 17.14, availability 'target') the token points at the fish until the hold starts,
    GRASP set: the median of the trial's first FISH_SAMPLES column pictures of the fish (the service's column pick-up
    record from its object finder, observe_fish), cast onto the plane at half the fish height with the same camera (ColumnCamera
    .fish_point) and kept for the trial; zeros until then. On the 10/16 control trials that point lay 2-7 cm (median 3)
    from the jaws where they shut on the fish, about 3 cm farther from the pan axis
    (analysis/v21/fish_goal_live_check.py)."""

    def __init__(self, camera, pan_scale, while_held=False, grasp_target=False):
        if not (np.isfinite(pan_scale) and pan_scale > 0):
            raise ValueError('the pan unit conversion must be a positive number')
        if grasp_target and not while_held:
            raise ValueError('the grasp target comes with the place goal while held (section 17.14)')
        self.camera, self.frame, self.pan_scale = camera, PanFrame(camera.kinematics), float(pan_scale)
        self.while_held, self.grasp_target, self.hold = bool(while_held), bool(grasp_target), HoldTracker()
        # the one pick-up record source taken for the fish: the service sets its object finder's (the learned column
        # finder's, or fcpg/v21_sam3_finder.SOURCE; training overhaul doc sections 17.32, 17.51)
        self.fish_source = FISH_SOURCE
        self.reset()

    @classmethod
    def from_availability(cls, path, availability):
        """The live goal for a model's goal_availability."""
        if availability not in AVAILABILITY:
            raise ValueError(f'unknown place-goal availability {availability!r}')
        return cls.from_column_camera(path, while_held=availability != 'until_release',
                                      grasp_target=availability == 'target')

    @classmethod
    def from_column_camera(cls, path, while_held=False, grasp_target=False):
        """The robot-calibrated column camera file; the pan conversion from its robot calibration."""
        import json
        from pathlib import Path
        from .v21_alignment_cem import load_column_camera
        camera = load_column_camera(path)
        if camera.conversion != 'robot_calibration':
            raise ValueError('the live place goal needs the robot-calibrated column camera')
        data = json.loads(Path(path).read_text())
        pan = json.loads(Path(data['robot_calibration']).read_text())['shoulder_pan']
        robot = (pan['range_max'] - pan['range_min']) / 200. * 2. * np.pi / 4096.     # rad per position unit
        low, high = np.asarray(camera.kinematics.all_joint_ranges, np.float64)[0]
        model = (high - low) / 200.
        return cls(camera, model / robot, while_held=while_held, grasp_target=grasp_target)

    def reset(self):
        self.box_xy, self.pixel, self.samples, self.frozen, self.opened = None, None, 0, False, False
        self.holding = False
        self.hold.reset()
        self.fish_xy, self.fish_pixels, self.fish_times = None, [], []

    def observe_fish(self, record):
        """The service's column pick-up record of this exposure (targets['targets']['pickup']['column']); with
        grasp_target, its centre joins the fish median until FISH_SAMPLES distinct current pictures are in, then the
        fish is kept. Ignored otherwise, and once the hold has started."""
        if not self.grasp_target or self.fish_xy is not None or self.hold.held or not isinstance(record, dict):
            return
        if (record.get('visible') is not True or record.get('source') != self.fish_source
                or record.get('view_source') != 'current_observation'):
            return
        pixel = np.asarray(record.get('cog_px') if record.get('cog_px') is not None else record.get('centroid_px'),
                           np.float64).reshape(-1)
        stamp = record.get('image_time_s')
        if pixel.shape != (2,) or not np.isfinite(pixel).all() or stamp is None or stamp in self.fish_times:
            return
        self.fish_pixels.append(pixel.tolist())
        self.fish_times.append(stamp)
        if len(self.fish_pixels) >= FISH_SAMPLES:
            point = self.camera.fish_point(np.median(np.asarray(self.fish_pixels), axis=0))
            if point is None:
                self.fish_pixels, self.fish_times = [], []
            else:
                self.fish_xy = np.asarray(point[:2], np.float64)

    def observe(self, drop_goal):
        """The service's drop goal (dict with target_px and sample_count); ignored once frozen."""
        if self.frozen or not isinstance(drop_goal, dict) or int(drop_goal.get('sample_count') or 0) < MIN_SAMPLES:
            return
        pixel = np.asarray(drop_goal.get('target_px'), np.float64).reshape(-1)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            return
        point = self.camera.point_on_plane(pixel, BOX_WALL_M / 2.)
        if point is None:
            return
        self.box_xy, self.pixel, self.samples = np.asarray(point[:2], np.float64), pixel, int(drop_goal.get('sample_count') or 0)

    def vector(self, q):
        """[8] goal token for the measured joints q [6]; zeros while no box is known (and, while_held, outside the
        hold); with grasp_target, before the hold the grasp target (zeros while no fish is known)."""
        q = np.asarray(q, np.float64).reshape(6)
        self.holding = self.hold.update(q[5])
        if self.while_held:
            self.frozen = self.frozen or (self.holding and self.box_xy is not None)
        elif q[5] >= OPEN_GRIP:
            self.opened = True
        elif self.opened and self.box_xy is not None:
            self.frozen = True
        if self.grasp_target and not self.hold.held:
            target = self.fish_xy
        elif self.box_xy is None or (self.while_held and not self.holding):
            target = None
        else:
            target = self.box_xy
        if target is None:
            return np.zeros(8, np.float32)
        jaws = self.camera.jaw(q[None])[0]
        token = self.frame.goal_vectors(jaws, np.r_[target, jaws[2]])
        token[1] *= self.pan_scale
        token[GRASP] = float(target is self.fish_xy)
        return token

    def describe(self):
        availability = 'target' if self.grasp_target else 'while_held' if self.while_held else 'until_release'
        return dict(source=getattr(self.camera, 'source', ''), box_xy_m=None if self.box_xy is None else self.box_xy.tolist(),
                    box_pixel=None if self.pixel is None else self.pixel.tolist(), samples=self.samples,
                    frozen=self.frozen, pan_scale=self.pan_scale, box_wall_m=BOX_WALL_M,
                    availability=availability, holding=self.holding,
                    hold=dict(opened=self.hold.opened, held=self.hold.held, released=self.hold.released),
                    **(dict(fish_xy_m=None if self.fish_xy is None else self.fish_xy.tolist(),
                            fish_samples=len(self.fish_pixels),
                            fish_pixel=None if not self.fish_pixels else np.median(np.asarray(self.fish_pixels), axis=0).tolist())
                       if self.grasp_target else {}))
