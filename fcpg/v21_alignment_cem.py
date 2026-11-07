"""The paper's planner step (ICRA2027 sec. IV-D) on the system's own models.

One block: the frozen FCPG + Actor propose five commands; CEM (32 candidates, 8 elites, 3 rounds) searches
around the proposal; the system LeWM (the FCPG-conditioned joint-response model, fcpg/v21_full_patch_lewm.py)
forecasts the joint response of every candidate; candidates are ranked by the gripper-to-object alignment at
the forecast execution-prefix state (the commands run before the next picture: all five of the block since 10/26,
section 17.109; three before), with zero Actor-prior weight in the cost.

The alignment is the paper's "currently available object-gripper geometry, augmented by the measured
joint-to-image response": the policy's own wrist object token (fish centre minus jaw centre in the wrist
image, taken at the planning pose) moved by a linear response fitted on completed real blocks
(scripts/fit_v21_wrist_response.py). It is relative to the gripper, never to a demonstration picture, so it
does not care where the fish or the box lie on the table. Optionally (GeometricResponse, a
v21-wrist-geometric-response-v1 file from scripts/fit_v21_wrist_geometric_response.py) the response comes from
geometry instead of one fixed table: robot forward kinematics, a pinhole wrist camera fixed to the gripper frame and
the fish on the table plane, linearized around the planning pose and the fish's offset at every plan
(docs/V21_SYSTEM_LEWM_CEM_AUDIT_20261013.md section 13: 7.6 px against the table's 10.0 px on held-out October
blocks). When that geometry cannot place the fish (its ray misses the table) the Actor's proposal passes unchanged.

Two things here are choices of this implementation, not of the paper:
- the search is over the wanted image shift (2 numbers); a candidate is the Actor's proposal plus the
  minimum-norm joint change for that shift, ramped in over the five commands and bounded by MAX_OFFSET.
  Searching all joints freely would let the three directions the cost cannot see wander at random;
- the search refines, it does not steer the approach: while the Actor itself still moves an arm joint by
  more than TRAVEL in the block, its proposal passes unchanged. The measured response comes from the last
  stretch before the grasp and says nothing reliable about height, so lift, elbow and wrist flex may depart
  only half as far as pan. steer_approach=True (training overhaul doc section 17.101, 10/25) searches those
  blocks too while the jaws are open (the fish), as the paper does; the box keeps the skip. On the robot the
  Actor's own approach turned too little and came down beside the fish in trials 5-7, and the geometric
  response got the direction of the fish's image shift right on big moves too (median cosine 0.9);
- closing the gripper is admitted only when the forecast alignment (after CLOSE_CHECK commands) is within
  close_tolerance_px (2026-10-11: the Actor closed with the fish 30-120 px beside the jaws). After max_deferrals
  refusals in a row the close goes through, so a trial cannot stall on it.
Nothing is planned when the object (below) is not seen either.

The object follows the jaws (docs/V21_SYSTEM_LEWM_CEM_AUDIT_20261013.md section 14.14): with the jaws open
(read-back from OPEN_GRIP up) it is the fish, with them shut it is the box, matched in the same wrist picture with the
drop zone's role prototype from its column registration, or with its wrist-view registration when the service has one
(fcpg/v21_target_tracker.py; training overhaul doc section 17.6), and put in the fish token's layout against the same
jaw midpoint. The box is aligned exactly like the fish (same response, same bounds, same
refining role); nothing is searched while the Actor is still closing the jaws, so the arm is left to it until the grip
is made, and no close is deferred for the box. The gripper stays the Actor's. align_box=False (training overhaul doc
section 17.116, 10/27 trial 14) leaves the whole carry to the Actor: the geometry puts every object on the table plane,
the box's rim is about 7 cm above it, and on trial 14 each box correction moved the box about twice as far as forecast
(pan swung 8 units left and right every block for 13 blocks) while right after the lift it pulled lift, elbow and wrist
flex back by the most it may; the fish fell beside the box. Off (trial 15), the Actor carried the fish beside the box's
wall about 3 cm above the table and let go there. box_height_m (James 10/27: the box is 12 cm tall) aligns the box on
the plane of its rim instead of the table's: the same geometry, one measured number.

Admission: a selection the service's physical gate would refuse is replaced by the admissible candidate nearest the
Actor's proposal (offsets inside MAX_OFFSET, smallest first), else by holding the arm where it is (10/16 pure-system
trials 2 and 3: the Actor's own commands pressed the jaws into the table and the trial stopped on the gate).

A trial that has had the fish between shut jaws records released_over_box at the first block that starts with the
jaws open again, the fish no longer between them and the box in the wrist picture; fcpg/v21_capped_trial.py ends the
trial there (James 10/16: home once the task is done). It is the planner's report, not a verified success. On the
eleven 10/16 control trials (analysis/v21/done_check_replay.py) it fires in the three that put the fish in the box, at
the block the Actor let go, and in none of the others (the same blocks with the wrist-view registration, training
overhaul doc section 17.6). The column picture cannot tell instead: its drop-zone support
covers only part of the box, so the fish lying in the box is never inside it. With the grip latch in control the
adapter also records it at a block that starts with the jaws open after the latch let go (it lets go only on the Actor's
open with the box in the wrist picture), until the next grasp: on 10/27 trial 24 (section 17.120) the fish fell
straight down into the box and still lay between the open jaws in the wrist picture (14.5 of the 15 px), the picture
rule never fired, and the Actor shut the jaws on the fish below and lifted the arm to 25 cm until the budget ran out.

The gripper rule (GripLatch, adapter option grip_latch; training overhaul doc section 17.45; James 10/20: "j6可以hack"):
the one hand-written control here, on joint 6 only. In 4 of the 7 17.37 trials the Actor's jaw command crept open
right after the grasp while lifting and the fish slipped; in the 2 successes it stayed at the read-back until the box.
From the grasp the jaws are held a little below the read-back (never opening) until the Actor asks for open jaws with
the box in the wrist picture, or nothing is held any more.

The 10/15 column-camera approach and the 10/16 carry to the box were removed after audit 14.13 ("这是需要做的是具身
智能，不是hardcode"; section 14.14): host geometry with hand-set heights and paths that drove the arm instead of
refining the Actor. ColumnCamera and load_column_camera stayed for the analysis scripts. Since 10/27 (section 17.118;
James 「不应该column+wrist都联合判断马？」) the adapter's box_solid_camera measures the box itself from the column picture:
a cuboid on the table fitted to the learned box finder's cells (fit_box_solid: where it stands, how big, how tall; no
hand-set height), once at a trial's first block; with the jaws shut the search then ranks candidates by the forecast
jaws' horizontal distance from the solid's centre (JawTable, forward kinematics, mm), within the same MAX_OFFSET of the
Actor's proposal and with the jaws' height left to the Actor. Trial 16 showed why: the wrist, tilted up, read the box
"centred" while the jaws were at its near wall. No path or height is written here; the wrist still sees the fish, the
grasp, released_over_box and the grip latch's box. Section 17.120 (trial 21: the arm sagged 5-10 mm below every carried
block, 59 of the ~100 mm the Actor and the search asked for): carrying, the search also moves the jaws' height and each
mm the system LeWM forecasts them ending below the Actor's own last command costs SAG_WEIGHT, so the arm ends where the
Actor commanded; how high is still the Actor's.

mode 'shadow': the Actor's commands execute, the planner is only recorded. mode 'control': the planner's
selection executes. Dispatch re-bases like fcpg/v21_c4_service.py: the Actor at the measured pose plus the
arm offset already chosen for this block (refused when the plan is older than DISPATCH_MAX_AGE_S).
"""
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

MODES = ('shadow', 'control')
SOURCES = dict(shadow='v21_actor_align_shadow', control='v21_lewm_align_cem')
RECORD_KEY = 'alignment_cem'
# what forecasts the arm for every candidate: the system LeWM, or (the ablation of the review's question "what does the
# LeWM itself add", James 10/28 「全部改好」) the arm reaching every command exactly, as a planner without a world model
# assumes; everything else in the search stays the same
DYNAMICS = dict(lewm='the system LeWM (fcpg/v21_full_patch_lewm.py)',
                commands='none: the arm reaches every command exactly (ablation of the LeWM)')
POPULATION, ELITES, ITERATIONS = 32, 8, 3      # the paper's live CEM configuration
PREFIX = 5                                     # commands dispatched before the next picture; this state is scored.
                                               # Was 3 until 10/26: the service runs all five commands of a block
                                               # (--no-replan-on-execute) and takes the next picture after it, so
                                               # scoring the third command's state (60% of the ramped offset)
                                               # under-predicted the picture's shift 2.4-2.8 times and the arm swung
                                               # left and right (training overhaul doc section 17.109)
CLOSE_CHECK = 3                                # the close gate still asks the alignment after three commands (as before
                                               # 10/26): the Actor's close runs during the block, not at its end
ARM = 5
SEARCH_JOINTS = (0, 1, 2, 3)                   # pan, lift, elbow, wrist flex; wrist roll and the gripper stay the Actor's
MAX_OFFSET = np.asarray((8., 4., 4., 3.))      # largest departure from the Actor's last command of the block (units)
FAR_SCALE = 2.                                 # 11/03: while the column fish leads jaws further than NEAR_FISH_MM from
                                               # it, the search may depart this many times as far (the Actor, which does
                                               # not steer by the fish, took the arm elsewhere and the search sat at
                                               # MAX_OFFSET in 98 of 114 such blocks of the rollc/rolld trials)
# align_roll (11/03, James 「它现在不会转J6阿。全部版本阿」: the jaws shut along a fish lying across them; the two
# demonstrations never turn the wrist, so neither the Actor nor a search of four joints does): while the jaws are open
# and aimed at the fish, the search also turns the wrist roll, and a candidate costs, besides the offset, how far the
# fish's ends are from lying across the forecast jaw line, along the arc they turn on (half the fish's length times the
# angle; for a small one, how far the fish reaches along the jaw line; a cost of the reach itself hardly changes near a
# fish lying along the jaws, and the search turned either way there in a replay of 11/03's trials). Both lie on
# the table in the column camera's base frame: the fish's axis from the column picture, which shows the whole fish
# (James 「wrist 和 external都要看吧？」; the wrist picture near the fish sees it foreshortened and cut, and read a fish
# lying across the jaws as lying along them: 11/03 trials 1-5, robot-20261103/roll-angles-trials-20261103.txt), the jaw
# line by forward kinematics (at the robot's carried grasps of 10/31-11/02 it lay 75-88 degrees to that axis in 6 of 9,
# robot-20261103/check_column_axis_20261103.py)
ROLL = 4                                       # the wrist roll (1.69 degrees of the jaw line per unit, the robot's
                                               # calibration)
MAX_ROLL_OFFSET = 20.                          # largest roll departure from the Actor's last command of the block
ORIENT_WEIGHT = 3.                             # cost per unit of that arc (in the cost's units: wrist px at the fish's
                                               # depth, or mm with the column leading). Replay of 11/03's 92 plans
                                               # (robot-20261103/replay_roll_20261103.py): at 1 the search turned the
                                               # wrong way in 7 and turned 76% of the first-order turn; at 3 in none,
                                               # 91%, the fish 6 px further from the jaw centre after the block
ORIENT_TOLERANCE_PX = 20.                      # a close at the fish also waits while the arc is longer
ELONGATED = 1.4                                # the fish's axis counts only for an outline this much longer than wide
MAX_FISH_MM = 120.                             # ... and at most this long on the table (the fish measures 55-114 mm at
                                               # 16 trial starts; 178 and 250 mm outlines took in more than the fish)
FISH_LENGTH_AGREE = .25                        # a later outline replaces the axis only when its length on the table is
                                               # within this share of the trial's first: with the arm near the fish the
                                               # outlines measured 34-50 mm (part hidden) or 110-245 mm (more taken in),
                                               # their axes up to 88 degrees off; the whole fish, pushed or not, 62-91 mm
                                               # (robot-20261103/column-axis-steps-20261103.txt, trial 1)
DROPPED_ABOVE_MM = 40.                         # 11/04 (sam31hb trial 1): a hold the grip latch finds empty with the jaws
                                               # this far above the table dropped the fish off the table: the kept whole
                                               # fish (its place and its length) is forgotten and the next outline sets
                                               # them again. It slipped out 132 mm up into the box; the kept place stayed
                                               # where it had lain on the table (the outlines in the box measured 100-108
                                               # mm against the kept 72) and the arm searched the empty table to the end
TRAVEL = 8.                                    # no search while the Actor itself still moves an arm joint this far in the block
RAMP = np.asarray((.2, .4, .6, .8, 1.))
IMAGE = 224.
OPEN_GRIP = 15.                                # jaws count as open from this read-back up (was 20 until 10/16: H1b
                                               # half-shuts to 18-20 on its last approach, so the alignment and its close
                                               # check were off and it closed with the fish 30 px off-centre, audit 14.12)
CLOSE_STEP = 3.                                # the proposal closes when its last gripper command is this far below it
SQUEEZE = 3.                                   # GripLatch: held this far below the read-back (demonstrations 0 and 3
                                               # hold 4.9 and 5.2 below it; 17.37's Actor about 0; 5 that followed the
                                               # read-back down squeezed the fish out on 10/20 trial 1)
SQUEEZE_LIMIT = 4.                             # GripLatch: the hold never goes more than this below the grasp read-back
EMPTY_GRIP = 4.                                # GripLatch: a grasp needs at least this read-back (shut on each other: 1.3)
CLOSED_GRIP = 2.                               # GripLatch: a read-back below this: the jaws met, nothing held
LOST_PLANS = 2                                 # GripLatch: plans in a row without the fish between the jaws: nothing held
RELEASE_OPEN = 30.                             # GripLatch: a release opens at least this far (HOME opens to about 31)
RELEASE_HIGH_MM = 80.                          # GripLatch without a measured box solid: over the box the jaws let go
                                               # only this low (jaw clearance above the table); demonstration 0 let go
                                               # at 34 mm inside the ~70 mm box, HLwb at 125-145 mm, and on 10/26 trial
                                               # 12 the fish missed the box. With the solid (section 17.120) the jaws let
                                               # go over its footprint at any height
RELEASE_DEFERRALS = 6                          # GripLatch: after this many held-back releases in one hold it lets go
WRIST_BOX_SIDE_PX = 40.                        # section 17.148 (10/31 trial 2: let go 19 mm inside a box solid
                                               # measured from the corner the column picture shows, the fish fell
                                               # beside the box): a release is held back while the wrist picture shows
                                               # the box reaching less than this to either side of the hold point
                                               # along x (today's 12 releases into the box: >= 78 px; the miss: 11)
NEAR_FISH_MM = 100.                            # 11/03: the column camera's fish leads the open jaws until they are
                                               # within this of it on the table, even when the wrist reports a fish:
                                               # in rollb trials 2-5 the wrist finder reported one in every picture
                                               # (the table, a monitor, a red bottle cap) with the column fish 150-480 mm
                                               # from the jaws; on the real fish (trial 1) it closed from 29-39 mm and
                                               # saw it from 165 mm (robot-20261103/wrist-vs-column-rollb-20261103.txt)
NEAR_BOX_MM = 30.                              # section 17.148: within this of the measured box solid's footprint a
                                               # wrist picture that shows the box to one side leads the jaws instead
MIN_GAIN_PX = 2.                               # a candidate replaces the proposal only when it is this much better
RANKED_CHECKS = 8                              # physical admission is asked for at most this many candidates, best first.
                                               # 10/26 (section 17.109) the whole ranking was tried: in replays it found
                                               # only ~3 px more at the table, and on the robot its picks beyond the
                                               # best eight hugged the floor before both safety stops (trials 11, 13)
DISPATCH_MAX_AGE_S = 2.0
RESPONSE_SCHEMA = 'v21-wrist-response-v1'
GEOMETRIC_SCHEMA = 'v21-wrist-geometric-response-v1'
GEOMETRIC_NAMES = ('focal_px', 'principal_x_px', 'principal_y_px', 'mount_rotvec_x', 'mount_rotvec_y', 'mount_rotvec_z',
                   'mount_x_m', 'mount_y_m', 'mount_z_m', 'table_z_m', 'jaw_midpoint_x_px', 'jaw_midpoint_y_px')
FD_STEP = 1.                                   # joint units for the central differences of the geometric response
COLUMN_SCHEMA = 'v21-column-self-calibration-v1'
FISH_HALF_HEIGHT_M = .015                      # the fish centre above the table (ColumnCamera.fish_point, fish_on_table)
ARM_JOINT_NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
BOX_SOLID_LOW = np.asarray((.10, -.10, -np.pi / 2, .08, .08, .02))   # section 17.118: where the box solid is searched:
BOX_SOLID_HIGH = np.asarray((.60, .50, np.pi / 2, .45, .45, .30))    # centre x, y, turn, length, width, height (m, rad)
BOX_SOLID_POPULATION, BOX_SOLID_ELITES, BOX_SOLID_ROUNDS, BOX_SOLID_RESTARTS = 64, 8, 40, 3
HEIGHT_SLACK_MM = 2.                           # section 17.118 b: aiming over the box solid, a candidate whose jaws
                                               # end lower than the Actor's own forecast by more than this is never
                                               # chosen (trial 18: the horizontal aim alone pressed the lift down)
SAG_WEIGHT = 10.                               # section 17.120 (James 「狠狠惩罚这种」): carrying, each mm the forecast
                                               # jaws end below the Actor's own last command of the block costs this
                                               # much (mm of aim). The arm sags under the fish (trial 21: 5-10 mm a
                                               # block, 59 mm over the carry) and the system LeWM forecasts it (median
                                               # error 2.3 mm), so the search also lifts the commands until the arm
                                               # itself ends where the Actor commanded; how high stays the Actor's
SAG_SLACK_MM = 1.
CLOSE_HIGH_MM = 15.                            # section 17.140 (17.138's robot trials; James 「彻底改好」, the CEM is
                                               # the designed part: 「CEM本来就是设计好的 要人工规则吧」): a close at
                                               # the fish is admitted only while the jaws' forecast height above the
                                               # table at the close check is at most this. The closes that held the
                                               # fish began 4-11 mm up, two empty ones 22-24 mm (p16-1, p16-5)
GRASP_MM = 8.                                  # section 17.140: while the Actor closes at the fish, the search lowers
                                               # the jaws towards this height (the middle of the closes that held it)
GRASP_LOW_MM = 4.                              # section 17.140 b: ... and not below this (the lowest close that held
                                               # the fish began at 4 mm; the 17.140 replay chose 1.9 mm once)
DESCEND_WEIGHT = 5.                            # section 17.140: each mm the forecast jaws are above GRASP_MM (or below
                                               # GRASP_LOW_MM) at the close check costs this much (px of aim) while
                                               # the Actor closes at the fish
BOX_SOLID_UNSEEN = .05                         # the fit's tie-break: cells of the box's floor beyond the picture (its
                                               # edges cut the box; how far it goes past them is not seen) count this
                                               # much each, against the finder's cell count: the smallest floor the
                                               # picture allows wins; the height stays the walls' the picture shows


@dataclass(frozen=True)
class WristResponse:
    """Measured response of the wrist-camera fish offset to the five arm joints: px per unit, [2, 5]."""
    jacobian: np.ndarray
    pairs: int = 0
    source: str = ''

    def __post_init__(self):
        jacobian = np.asarray(self.jacobian, np.float64)
        if jacobian.shape != (2, ARM) or not np.isfinite(jacobian).all():
            raise ValueError('wrist response must be a finite [2, 5] matrix')
        searched = jacobian[:, list(SEARCH_JOINTS)]
        if np.linalg.matrix_rank(searched, tol=1e-3) < 2:
            raise ValueError('wrist response does not identify both image directions')
        object.__setattr__(self, 'jacobian', jacobian)
        # smallest joint change for an image shift, each joint measured against what it may depart: [4, 2]
        weight = np.diag(MAX_OFFSET ** 2)
        object.__setattr__(self, 'inverse', weight @ searched.T @ np.linalg.inv(searched @ weight @ searched.T))

    def shift(self, joint_change):
        """[..., 5] change of the arm joints -> [..., 2] change of the fish offset in px."""
        return np.asarray(joint_change, np.float64) @ self.jacobian.T

    def offset(self, shift_px):
        """[..., 2] wanted change of the fish offset -> [..., 4] smallest change of the searched joints."""
        return np.asarray(shift_px, np.float64) @ self.inverse.T


def fit_response(joint_change, offset_change_px):
    """Least squares over completed block pairs: [n, 5] joint changes, [n, 2] px -> [2, 5]."""
    joint_change = np.asarray(joint_change, np.float64)
    offset_change_px = np.asarray(offset_change_px, np.float64)
    if joint_change.ndim != 2 or joint_change.shape[1] != ARM or offset_change_px.shape != (len(joint_change), 2):
        raise ValueError('response fit needs [n, 5] joint changes and [n, 2] offset changes')
    if len(joint_change) < 20:
        raise ValueError('response fit needs at least 20 completed block pairs')
    return np.linalg.lstsq(joint_change, offset_change_px, rcond=None)[0].T


def fish_on_table(camera, record):
    """align_roll: the fish on the table from a column finder record (fcpg/v21_object_finder.LearnedColumnFinder: centre
    cog_px, axis_px, length_px and width_px in letterbox224 pixels): its outline's two ends, the centre +- half its
    length along its axis, placed at the fish's half height by the column camera -> dict(centre [2], direction [2]
    (unit), length_m), base frame; None when unseen, too round for an axis (ELONGATED), off the table or longer than
    MAX_FISH_MM."""
    if not record or not record.get('visible'):
        return None
    try:
        centre = np.asarray(record['cog_px'], np.float64).reshape(2)
        axis = np.asarray(record['axis_px'], np.float64).reshape(2)
        length, width = float(record['length_px']), float(record['width_px'])
    except (KeyError, TypeError, ValueError):
        return None
    if (not np.isfinite(centre).all() or not np.isfinite(axis).all() or not np.linalg.norm(axis) > 1e-9
            or not length >= ELONGATED * max(width, 1e-6)):
        return None
    half = .5 * length * axis / np.linalg.norm(axis)
    ends = [camera.point_on_plane(centre - half, FISH_HALF_HEIGHT_M),
            camera.point_on_plane(centre + half, FISH_HALF_HEIGHT_M)]
    if any(end is None or not np.isfinite(end).all() for end in ends):
        return None
    along = (ends[1] - ends[0])[:2]
    span = float(np.linalg.norm(along))
    if not 0. < span <= MAX_FISH_MM / 1000.:
        return None
    return dict(centre=.5 * (ends[0] + ends[1])[:2], point=.5 * (ends[0] + ends[1]), direction=along / span,
                length_m=span)


def arc_from_across(fish, jaw_lines):
    """align_roll: [n] how far the fish's ends (fish_on_table) are from lying across each jaw line [n, 2] (unit, on the
    table), along the arc they turn on, m: half its length times the angle from across (0 for a fish lying in the gap;
    for a small angle, how far it reaches along the jaw line on either side of its centre)."""
    cosine = np.abs(np.asarray(jaw_lines, np.float64) @ np.asarray(fish['direction'], np.float64))
    return .5 * fish['length_m'] * np.arcsin(np.clip(cosine, 0., 1.))


def across_deg(fish, jaw_lines):
    """[n] the angle (degrees, 0..90) between the fish's axis and each jaw line on the table; 90: the jaws close across
    the fish."""
    cosine = np.abs(np.asarray(jaw_lines, np.float64) @ np.asarray(fish['direction'], np.float64))
    return np.degrees(np.arccos(np.clip(cosine, 0., 1.)))


def _rotation(rotvec):
    rotvec = np.asarray(rotvec, np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    k = rotvec / angle
    cross = np.array([[0., -k[2], k[1]], [k[2], 0., -k[0]], [-k[1], k[0], 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * cross @ cross


def _cast(theta, pose, pixel):
    """Where the ray of each wrist pixel [n, 2] (picture coordinates) with the gripper at pose [n, 4, 4] (base frame)
    meets the table plane: the point [n, 3] and its depth along the optical axis [n] (m; not positive where the ray
    misses the table in front of the camera). theta: GEOMETRIC_NAMES."""
    f, cx, cy = theta[:3]
    mount, offset, table = _rotation(theta[3:6]), theta[6:9], theta[9]
    ray = np.stack(((pixel[:, 0] - cx) / f, (pixel[:, 1] - cy) / f, np.ones(len(pixel))), axis=1)
    rotation0 = pose[:, :3, :3] @ mount
    origin0 = pose[:, :3, 3] + np.einsum('nij,j->ni', pose[:, :3, :3], offset)
    direction = np.einsum('nij,nj->ni', rotation0, ray)
    reach = (table - origin0[:, 2]) / np.where(np.abs(direction[:, 2]) < 1e-9, 1e-9, direction[:, 2])
    return origin0 + reach[:, None] * direction, reach


def geometric_shift(theta, pose_before, pose_after, offset_px):
    """Image shift [n, 2] px of the fish whose offset from the jaw midpoint is offset_px [n, 2] at gripper pose
    pose_before [n, 4, 4] (base frame) once the gripper is at pose_after: cast the fish pixel through a pinhole camera
    fixed to the gripper frame onto the table plane, project it again. NaN where the ray misses the table in front of
    the camera or the fish ends behind it. theta: GEOMETRIC_NAMES."""
    theta = np.asarray(theta, np.float64)
    f, cx, cy = theta[:3]
    mount, offset, jaw = _rotation(theta[3:6]), theta[6:9], theta[10:12]
    pixel = np.asarray(offset_px, np.float64) + jaw
    point, reach = _cast(theta, pose_before, pixel)
    rotation1 = pose_after[:, :3, :3] @ mount
    origin1 = pose_after[:, :3, 3] + np.einsum('nij,j->ni', pose_after[:, :3, :3], offset)
    camera = np.einsum('nji,nj->ni', rotation1, point - origin1)
    valid = (reach > 0) & (camera[:, 2] > 1e-3)
    depth = np.where(valid, camera[:, 2], 1.)
    after = np.stack((f * camera[:, 0] / depth + cx, f * camera[:, 1] / depth + cy), axis=1)
    return np.where(valid[:, None], after - pixel, np.nan)


@dataclass(frozen=True)
class GeometricResponse:
    """Joint-to-image response from geometry (docs/V21_SYSTEM_LEWM_CEM_AUDIT_20261013.md section 13). local() gives the
    measured-response interface for one plan: central differences of geometric_shift around the planning pose and
    the fish's offset; None when the geometry cannot place the fish or the local table does not see both image
    directions. kinematics: anything with forward_gripper_poses(states [n, 6]) -> [n, 4, 4]. jaw: optional
    q [6] -> jaw-midpoint pixel [2] from the robot's wrist calibration; then the fitted jaw midpoint is zero and the
    fish pixel is its offset plus this calibrated midpoint."""
    theta: np.ndarray
    kinematics: object
    pairs: int = 0
    source: str = ''
    jaw: object = None

    def __post_init__(self):
        theta = np.asarray(self.theta, np.float64)
        if theta.shape != (len(GEOMETRIC_NAMES),) or not np.isfinite(theta).all() or theta[0] <= 0:
            raise ValueError('geometric response needs 12 finite numbers with a positive focal length')
        object.__setattr__(self, 'theta', theta)

    def local(self, q, offset_px):
        q = np.asarray(q, np.float64)
        states = np.repeat(q[None], 1 + 2 * ARM, 0)
        for joint in range(ARM):
            states[1 + 2 * joint, joint] += FD_STEP
            states[2 + 2 * joint, joint] -= FD_STEP
        poses = self.kinematics.forward_gripper_poses(states)
        pixel = np.asarray(offset_px, np.float64) + (0. if self.jaw is None else np.asarray(self.jaw(q), np.float64))
        shift = geometric_shift(self.theta, np.repeat(poses[:1], 2 * ARM, 0), poses[1:], np.repeat(pixel[None], 2 * ARM, 0))
        if not np.isfinite(shift).all():
            return None
        try:
            return WristResponse(((shift[0::2] - shift[1::2]) / (2 * FD_STEP)).T, self.pairs, self.source)
        except ValueError:
            return None

    def table_scale(self, q, offset_px):
        """align_roll: wrist pixels per metre on the table at the fish offset_px from the jaw midpoint at q (as local()
        places it): the focal length over its depth in the camera; None where its ray misses the table in front."""
        q = np.asarray(q, np.float64)
        pixel = (np.asarray(offset_px, np.float64) + self.theta[10:12]
                 + (0. if self.jaw is None else np.asarray(self.jaw(q), np.float64)))
        _, depth = _cast(self.theta, self.kinematics.forward_gripper_poses(q[None]), pixel.reshape(1, 2))
        return float(self.theta[0] / depth[0]) if np.isfinite(depth[0]) and depth[0] > 1e-3 else None


def calibrated_jaw_midpoint(wrist_reference, kinematics):
    """q [6] -> jaw-midpoint pixel [2] in the letterbox224 wrist image, from the robot's wrist calibration (the same
    projection the object finder's encoder (jaw_pair) uses for the policy's object token)."""
    from .so101_camera_calibration import ProjectionFit, predict_projection
    fits = json.loads(Path(wrist_reference).read_text())['fits']
    row = fits['wrist'] if isinstance(fits, dict) else next(r for r in fits if r.get('view') == 'wrist')
    fit = ProjectionFit('wrist', tuple(row['feature_names']), np.asarray(row['coefficients'], np.float64), float(row['ridge']))

    def jaw(q):
        q = np.asarray(q, np.float64).reshape(-1, 6)
        world, local = kinematics.forward_anchor_frames(q)
        pixels = predict_projection(fit, world, local, q[:, 5])
        midpoint = pixels[:, 1:3].mean(1)
        return midpoint[0] if len(midpoint) == 1 else midpoint
    return jaw


@dataclass(frozen=True)
class ColumnCamera:
    """The column camera located in the robot base frame from the robot's own motion
    (analysis/v21/column_self_calibration.py): a pinhole camera (params: rotation vector, translation, log focal length
    at working_size; principal point at the picture centre) fitted so that the robot's mesh, placed by forward
    kinematics, covers what moves in the column pictures. kinematics: anything with forward_anchors(states [n, 6]) ->
    [n, 3, 3] (jaw midpoint first); to_model_units: [n, 6] measured positions -> the positions forward_anchors expects
    (the joint conversion the camera was fitted with). to_hinges (align_roll's jaw line, with a kinematics that has
    forward_anchors_qpos): [n, 6] measured positions -> MuJoCo hinge positions by the same conversion, not clipped to
    the model's joint ranges."""
    params: np.ndarray
    working_size: tuple
    raw_size: tuple
    table_z: float
    kinematics: object
    to_model_units: object = None
    source: str = ''
    conversion: str = ''
    to_hinges: object = None

    def __post_init__(self):
        params = np.asarray(self.params, np.float64)
        if params.shape != (7,) or not np.isfinite(params).all() or not np.isfinite(self.table_z):
            raise ValueError('column camera needs 7 finite numbers and a finite table height')
        object.__setattr__(self, 'params', params)
        rotation = _rotation(params[:3])
        object.__setattr__(self, 'rotation', rotation)
        object.__setattr__(self, 'center', -rotation.T @ params[3:6])

    def fish_point(self, pixel_lb224):
        """Letterbox224 column pixel of the fish -> its centre on the table, base frame [3]; None when the ray misses."""
        return self.point_on_plane(pixel_lb224, FISH_HALF_HEIGHT_M)

    def point_on_plane(self, pixel_lb224, height_m):
        """Letterbox224 column pixel -> where its ray meets the plane height_m above the table, base frame [3]; None
        when the ray misses it."""
        pixel = np.asarray(pixel_lb224, np.float64).reshape(-1)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            return None
        (raw_w, raw_h), (w, h) = self.raw_size, self.working_size
        scale = IMAGE / raw_w
        raw = np.array((pixel[0] / scale, (pixel[1] - (IMAGE - raw_h * scale) / 2.) / scale))
        work = raw * w / raw_w
        focal = float(np.exp(self.params[6]))
        direction = self.rotation.T @ np.array(((work[0] - w / 2.) / focal, (work[1] - h / 2.) / focal, 1.))
        if abs(direction[2]) < 1e-9:
            return None
        reach = (self.table_z + height_m - self.center[2]) / direction[2]
        return self.center + reach * direction if reach > 0 else None

    def project(self, points):
        """Base-frame points [n, 3] -> letterbox224 column pixels [n, 2], NaN behind the camera (point_on_plane's
        inverse)."""
        points = np.asarray(points, np.float64).reshape(-1, 3)
        camera = points @ self.rotation.T + self.params[3:6]
        (raw_w, raw_h), (w, h) = self.raw_size, self.working_size
        focal = float(np.exp(self.params[6]))
        depth = np.where(camera[:, 2] > 1e-6, camera[:, 2], np.nan)
        raw = np.stack((focal * camera[:, 0] / depth + w / 2., focal * camera[:, 1] / depth + h / 2.), 1) * raw_w / w
        scale = IMAGE / raw_w
        return np.stack((raw[:, 0] * scale, raw[:, 1] * scale + (IMAGE - raw_h * scale) / 2.), 1)

    def jaw(self, states):
        """[n, 6] measured positions -> jaw midpoints (between the fingertips) [n, 3], base frame."""
        states = np.asarray(states, np.float64).reshape(-1, 6)
        if self.to_model_units is not None:
            states = self.to_model_units(states)
        return np.asarray(self.kinematics.forward_anchors(states), np.float64)[:, 0]

    def jaw_line(self, states):
        """align_roll: [n, 6] measured positions -> the jaw line on the table [n, 2] (unit, base frame; NaN where it
        stands upright): the fixed fingertip minus the moving one with the jaws taken open (RELEASE_OPEN), the line they
        close along."""
        states = np.array(states, np.float64).reshape(-1, 6)
        states[:, 5] = RELEASE_OPEN
        if self.to_hinges is not None:          # the robot's roll beyond the model's range still turns the line
            anchors = np.asarray(self.kinematics.forward_anchors_qpos(self.to_hinges(states)), np.float64)
        else:
            if self.to_model_units is not None:
                states = self.to_model_units(states)
            anchors = np.asarray(self.kinematics.forward_anchors(states), np.float64)
        line = anchors[:, 1, :2] - anchors[:, 2, :2]
        size = np.linalg.norm(line, axis=1, keepdims=True)
        return np.where(size > 1e-3, line / np.maximum(size, 1e-12), np.nan)


def load_column_camera(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != COLUMN_SCHEMA:
        raise ValueError(f'{path} is not a {COLUMN_SCHEMA} file')
    from .so101_kinematics import (SO101MuJoCoKinematics, normalized_positions_to_mujoco_qpos,
                                   radians_to_normalized_arm_positions)
    kinematics = SO101MuJoCoKinematics(data['model'])
    to_model_units = None

    def to_hinges(states):
        """[n, 6] measured positions -> MuJoCo hinge positions [n, 6], the arm's not clipped to the model's ranges
        (align_roll: the robot's wrist roll turns up to 12 degrees further than the model's; 11/03 trial 1 went there)"""
        states = np.asarray(states, np.float64).reshape(-1, 6)
        hinges = normalized_positions_to_mujoco_qpos(states, kinematics.all_joint_ranges)
        if to_model_units is not None:
            raw = low + (np.clip(states[:, :ARM], -100, 100) + 100.) / 200. * (high - low)
            hinges[:, :ARM] = np.deg2rad((raw - 2047.) * 360. / 4096.)
        return hinges
    if data['conversion'] == 'robot_calibration':
        calibration = json.loads(Path(data['robot_calibration']).read_text())
        low = np.array([calibration[n]['range_min'] for n in ARM_JOINT_NAMES], np.float64)
        high = np.array([calibration[n]['range_max'] for n in ARM_JOINT_NAMES], np.float64)
        ranges = np.asarray(kinematics.all_joint_ranges, np.float64)[:ARM]

        def to_model_units(states):
            states = np.array(states, np.float64)
            raw = low + (np.clip(states[:, :ARM], -100, 100) + 100.) / 200. * (high - low)
            states[:, :ARM] = radians_to_normalized_arm_positions(np.deg2rad((raw - 2047.) * 360. / 4096.), ranges)
            return states
    elif data['conversion'] != 'model_ranges':
        raise ValueError(f'{path}: unknown joint conversion {data["conversion"]!r}')
    return ColumnCamera(np.asarray(data['params'], np.float64), tuple(data['working_size']), tuple(data['raw_size']),
                        float(data['table_z_m']), kinematics, to_model_units, str(path), data['conversion'], to_hinges)


@dataclass(frozen=True)
class BoxSolid:
    """The box as a cuboid standing on the table, measured from the column picture (training overhaul doc section
    17.118; James 10/27 「为什么不能自己识别呢？」 about its height): centre (x, y), length, width and height in metres in
    the column camera's base frame, turn in radians; overlap is the intersection over union of its outline with the
    learned box finder's cells inside the picture, cells how many cells the finder gave."""
    centre: tuple
    turn: float
    length: float
    width: float
    height: float
    overlap: float
    cells: int

    def describe(self):
        return dict(centre_mm=[round(float(v) * 1000., 1) for v in self.centre],
                    turn_deg=round(float(np.degrees(self.turn)), 1), length_mm=round(self.length * 1000., 1),
                    width_mm=round(self.width * 1000., 1), height_mm=round(self.height * 1000., 1),
                    overlap=round(self.overlap, 3), cells=self.cells)


def footprint_margin_mm(solid, point):
    """How far a point on the table (base frame, m; x, y) is inside the box solid's footprint: mm to the nearer of its
    two sides, negative outside."""
    dx, dy = np.asarray(point, np.float64).reshape(-1)[:2] - np.asarray(solid.centre, np.float64)
    c, s = np.cos(solid.turn), np.sin(solid.turn)
    return float(min(solid.length / 2. - abs(dx * c + dy * s), solid.width / 2. - abs(dy * c - dx * s)) * 1000.)


def cuboid_corners(params, table_z):
    """Centre x, y, turn, length, width, height (m, rad) of a cuboid on the table -> its eight corners [8, 3] (the four
    on the table first)."""
    x, y, turn, length, width, height = (float(v) for v in params)
    c, s = np.cos(turn), np.sin(turn)
    local = np.asarray([[sx * length / 2., sy * width / 2.] for sx in (-1., 1.) for sy in (-1., 1.)])
    xy = local @ np.asarray([[c, s], [-s, c]]) + (x, y)
    return np.r_[np.c_[xy, np.full(4, table_z)], np.c_[xy, np.full(4, table_z + height)]]


def _hull(points):
    """Convex hull of a few 2-D points (Andrew's monotone chain), in order around it."""
    points = sorted(map(tuple, np.asarray(points, np.float64)))

    def chain(sequence):
        out = []
        for p in sequence:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out
    return chain(points)[:-1] + chain(reversed(points))[:-1]


def box_outline(camera, params, grid, margin=0, floor=False):
    """[grid + 2 margin, grid + 2 margin] bool: the cells of the letterbox224 column grid, extended by margin cells on
    every side, inside the outline of the cuboid params (floor: of its floor only); None when a corner is behind the
    camera."""
    from PIL import Image, ImageDraw
    pixels = camera.project(cuboid_corners(params, camera.table_z)[:4 if floor else 8])
    if not np.isfinite(pixels).all():
        return None
    cell = IMAGE / grid
    image = Image.new('L', (grid + 2 * margin, grid + 2 * margin), 0)
    ImageDraw.Draw(image).polygon([(u / cell + margin, v / cell + margin) for u, v in _hull(pixels)], fill=1)
    return np.asarray(image) > 0


def fit_box_solid(camera, prob, *, found=.5, min_cells=2, seed=0):
    """The cuboid standing on the table whose outline in the column picture best covers the learned box finder's cells
    (prob [grid, grid], probability >= found; fcpg/v21_object_finder.LearnedBoxFinder): the cross-entropy method over
    centre, turn, sizes and height inside BOX_SOLID_LOW..HIGH, scored by intersection over union inside the picture
    (the letterbox's empty bands are not picture). The picture's edges cut our box: how far it goes past them is not
    seen, and boxes of any such size fit equally, so the floor's cells beyond the picture cost BOX_SOLID_UNSEEN each
    (against the finder's cell count): the smallest floor the picture allows, whose centre errs towards the side seen,
    inside the box; the height is left to the walls the picture shows. Nothing about the box is set by hand; the table plane and the camera are the 10/15 self-calibration. None when
    the finder found no box."""
    prob = np.asarray(prob, np.float64)
    grid = prob.shape[0]
    if prob.shape != (grid, grid):
        raise ValueError('the box finder map must be square')
    raw_w, raw_h = camera.raw_size
    band = (IMAGE - raw_h * IMAGE / raw_w) / 2.
    rows = (np.arange(grid) + .5) * IMAGE / grid
    inside = np.broadcast_to(((rows > band) & (rows < IMAGE - band))[:, None], prob.shape)
    target = (prob >= found) & inside
    if target.sum() < min_cells:
        return None
    rng = np.random.default_rng(seed)

    def score(params):
        full = box_outline(camera, params, grid, margin=grid)
        floor = box_outline(camera, params, grid, margin=grid, floor=True)
        if full is None or floor is None:
            return -1.
        outline = full[grid:2 * grid, grid:2 * grid] & inside
        unseen = int(floor.sum() - (floor[grid:2 * grid, grid:2 * grid] & inside).sum())
        union = (outline | target).sum()
        return float((outline & target).sum() / union if union else 0.) - BOX_SOLID_UNSEEN * unseen / target.sum()

    best, best_score = None, -1.
    for _ in range(BOX_SOLID_RESTARTS):
        mean, spread = (BOX_SOLID_LOW + BOX_SOLID_HIGH) / 2., (BOX_SOLID_HIGH - BOX_SOLID_LOW) / 3.
        for _ in range(BOX_SOLID_ROUNDS):
            samples = np.clip(rng.normal(mean, spread, (BOX_SOLID_POPULATION, 6)), BOX_SOLID_LOW, BOX_SOLID_HIGH)
            if best is not None:
                samples[0] = best
            scores = np.asarray([score(s) for s in samples])
            elite = samples[np.argsort(scores, kind='stable')[-BOX_SOLID_ELITES:]]
            mean, spread = elite.mean(0), elite.std(0) + 1e-3 * (BOX_SOLID_HIGH - BOX_SOLID_LOW)
            if scores.max() > best_score:
                best_score, best = float(scores.max()), samples[int(np.argmax(scores))].copy()
    x, y, turn, length, width, height = (float(v) for v in best)
    return BoxSolid((x, y), turn, length, width, height, best_score, int(target.sum()))


class JawTable:
    """The jaws' response to the arm joints at one pose (section 17.118), for aiming them over the box solid: forward
    kinematics of the jaw midpoint in the column camera's base frame (ColumnCamera.jaw, the conversion the camera was
    calibrated with) by central differences, in mm per joint unit. The search's interface (as WristResponse):
    shift(joint change [..., 5]) -> the change of the box's horizontal offset from the jaws (minus the jaws' move);
    offset(wanted [..., 2]) -> the smallest change of the searched joints, each measured against what it may depart
    (MAX_OFFSET), that moves the jaws so and keeps their height (the height stays the Actor's; the search's clip to
    MAX_OFFSET can move it a little); offset_rise(wanted [..., 3]) the same with the jaws' rise (mm) as the third
    column (section 17.120: carrying, the search makes up the forecast sag)."""

    def __init__(self, camera, q):
        q = np.asarray(q, np.float64).reshape(6)
        states = np.repeat(q[None], 1 + 2 * ARM, 0)
        for joint in range(ARM):
            states[1 + 2 * joint, joint] += FD_STEP
            states[2 + 2 * joint, joint] -= FD_STEP
        jaw = np.asarray(camera.jaw(states), np.float64) * 1000.
        derivative = ((jaw[1::2] - jaw[2::2]) / (2 * FD_STEP)).T          # [3, 5] mm per unit
        if not np.isfinite(jaw).all() or not np.isfinite(derivative).all():
            raise ValueError('the jaws cannot be placed at this pose')
        self.q, self.jaw_mm = q, jaw[0]
        self.jacobian = -derivative[:2]
        self.height_row = derivative[2]
        searched = np.vstack((self.jacobian, derivative[2:3]))[:, list(SEARCH_JOINTS)]
        if np.linalg.matrix_rank(searched, tol=1e-6) < 3:
            raise ValueError('the searched joints cannot move the jaws over the table and keep their height here')
        weight = np.diag(MAX_OFFSET ** 2)
        self.rise_inverse = weight @ searched.T @ np.linalg.inv(searched @ weight @ searched.T)
        self.inverse = self.rise_inverse[:, :2]
        self.source, self.pairs = 'forward kinematics in the column camera frame (section 17.118)', 0

    def shift(self, joint_change):
        return np.asarray(joint_change, np.float64)[..., :ARM] @ self.jacobian.T

    def offset(self, shift_mm):
        return np.asarray(shift_mm, np.float64) @ self.inverse.T

    def offset_rise(self, wanted):
        return np.asarray(wanted, np.float64) @ self.rise_inverse.T

    def height(self, states):
        """Jaw height (mm, column camera frame) at joint states [..., >= 5], linearised around this pose."""
        return self.jaw_mm[2] + (np.asarray(states, np.float64)[..., :ARM] - self.q[:ARM]) @ self.height_row


def load_response(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema') == GEOMETRIC_SCHEMA:
        from .so101_kinematics import SO101MuJoCoKinematics
        if list(data.get('theta_names', [])) != list(GEOMETRIC_NAMES):
            raise ValueError(f'{path}: unexpected geometric response parameters')
        kinematics = SO101MuJoCoKinematics(data['kinematics_model'])
        jaw = None
        if data.get('wrist_reference'):
            jaw = calibrated_jaw_midpoint(data['wrist_reference'], kinematics)
            if not np.allclose(np.asarray(data['theta'])[10:12], 0.):
                raise ValueError(f'{path}: a calibrated jaw midpoint needs the fitted one at zero')
        return GeometricResponse(np.asarray(data['theta'], np.float64), kinematics, int(data['pairs']), str(path), jaw)
    if data.get('schema') != RESPONSE_SCHEMA:
        raise ValueError(f'{path} is not a {RESPONSE_SCHEMA} or {GEOMETRIC_SCHEMA} file')
    return WristResponse(np.asarray(data['jacobian_px_per_unit'], np.float64), int(data['pairs']), str(path))


class AlignmentCEM:
    """The search itself; knows nothing of the service. forecast(commands [n, 5, 6]) -> joints [n, 6, 6]."""

    def __init__(self, response, *, target_px=(0., 0.), close_tolerance_px=15., max_deferrals=6, seed=0,
                 lower=None, upper=None, clock=time.monotonic, steer_approach=False, align_box=True,
                 box_height_m=None, column_camera=None, align_roll=False):
        if not isinstance(response, (WristResponse, GeometricResponse)):
            raise TypeError('alignment CEM needs a measured WristResponse or a GeometricResponse')
        if column_camera is not None and not isinstance(column_camera, ColumnCamera):
            raise TypeError('the box solid needs a ColumnCamera')
        if not isinstance(align_roll, bool) or (align_roll and not (isinstance(response, GeometricResponse)
                                                                     and column_camera is not None)):
            raise ValueError('align_roll is a bool and needs a GeometricResponse (the wrist\'s scale at the fish) and '
                             'the column camera (the fish\'s axis and the jaw line on the table)')
        self.align_roll = align_roll            # 11/03: also turn the wrist roll so the jaws close across the fish
        # section 17.118: with a column camera, the box measured as a solid from the column picture (fit_box_solid, set
        # by the adapter at a trial's first block) is what the shut jaws are aimed over, by forward kinematics
        self.camera = column_camera
        self.box_solid = None
        if not isinstance(steer_approach, bool):
            raise ValueError('steer_approach is a bool')
        if not isinstance(align_box, bool):
            raise ValueError('align_box is a bool')
        if box_height_m is not None and not (isinstance(response, GeometricResponse) and np.isfinite(box_height_m)
                                             and box_height_m >= 0):
            raise ValueError('a box height needs a GeometricResponse and a finite height of at least 0')
        self.response = response
        self.steer_approach = steer_approach    # section 17.101: search the fish approach while the Actor travels
        self.align_box = align_box              # section 17.116: False leaves the carry to the box to the Actor
        self.box_height_m = None if box_height_m is None else float(box_height_m)
        self.box_response = None                # section 17.116: the geometry with the table plane at the box's rim
        if self.box_height_m is not None:
            theta = response.theta.copy()
            theta[9] += self.box_height_m
            self.box_response = replace(response, theta=theta)
        self.target = np.asarray(target_px, np.float64)
        self.close_tolerance_px = float(close_tolerance_px)
        self.max_deferrals = int(max_deferrals)
        if self.target.shape != (2,) or not self.close_tolerance_px > 0 or self.max_deferrals < 0:
            raise ValueError('invalid alignment target, close tolerance or deferral limit')
        self.lower = None if lower is None else np.asarray(lower, np.float64)
        self.upper = None if upper is None else np.asarray(upper, np.float64)
        self.rng = np.random.default_rng(seed)
        self.clock = clock
        self.deferrals = 0
        self.height_deferrals = 0           # section 17.140 b: closes held back for height, counted apart
        self.grasped = False

    def reset(self):
        self.deferrals = 0
        self.height_deferrals = 0
        self.grasped = False
        self.box_solid = None

    def candidates(self, proposal, offsets):
        """The proposal plus each searched-joint offset ramped in over the five commands: [n, 4] -> [n, 5, 6]; a fifth
        column (align_roll) is the wrist roll's."""
        out = np.repeat(proposal[None], len(offsets), 0)
        out[:, :, list(SEARCH_JOINTS)] += RAMP[None, :, None] * offsets[:, None, :len(SEARCH_JOINTS)]
        if offsets.shape[1] > len(SEARCH_JOINTS):
            out[:, :, ROLL] += RAMP[None, :] * offsets[:, None, len(SEARCH_JOINTS)]
        if self.lower is not None:
            out = np.clip(out, self.lower, self.upper)
        return out

    def predicted_offset(self, q, token_px, forecast_q, response=None, state=PREFIX):
        """Fish offset (px) at forecast state `state` (PREFIX, the block's end; the close gate asks CLOSE_CHECK) of
        each candidate: [n, 6, 6] -> [n, 2]. response: the table of this plan (a GeometricResponse's local table);
        defaults to the measured WristResponse."""
        response = self.response if response is None else response
        if isinstance(response, GeometricResponse):
            raise ValueError('a geometric response is linearized per plan: pass its local table')
        forecast_q = np.asarray(forecast_q, np.float64)
        if forecast_q.ndim != 3 or forecast_q.shape[1:] != (6, 6) or not np.isfinite(forecast_q).all():
            raise ValueError('LeWM forecast must be finite [n, 6, 6] (source and five read-backs)')
        return token_px + response.shift(forecast_q[:, state, :ARM] - q[:ARM])

    def plan(self, q, proposal, token, forecast, admissible, *, commit=True, box_token=None, fish_point=None,
             jaw_height=None, box_from_wrist=False, fish_axis=None):
        """-> (commands [5, 6], record). token: the fish's wrist token, box_token the box's in the same layout
        (fcpg/v21_object_relative.py), None when unseen. fish_point: the fish on the table from the column picture (base
        frame, m; section 17.118 d), None when the column finder does not see it. commit=False (shadow) leaves the
        deferral count alone. jaw_height (section 17.140): joint states [n, 6] -> the jaws' heights above the table
        (mm); with it a close at the fish waits until the jaws are low enough, and the search lowers them meanwhile.
        fish_axis (align_roll): the fish on the table (fish_on_table, from the column picture); with it the search
        also turns the wrist roll while the jaws are open and aimed at the fish."""
        started = self.clock()
        q = np.asarray(q, np.float64)
        proposal = np.asarray(proposal, np.float64)
        if q.shape != (6,) or proposal.shape != (5, 6) or not np.isfinite(q).all() or not np.isfinite(proposal).all():
            raise ValueError('alignment CEM needs finite q [6] and a proposal [5, 6]')
        # 11/03 (rollg trial 3): the Actor also shuts the jaws slowly, less than CLOSE_STEP a block; the block that takes
        # open jaws below OPEN_GRIP is a close too (after it the planner sees shut jaws and aims at the box)
        closing = bool(proposal[-1, 5] < q[5] - CLOSE_STEP or (q[5] >= OPEN_GRIP > proposal[-1, 5]))
        record = dict(population=POPULATION, elites=ELITES, iterations=ITERATIONS, prefix=PREFIX, searched=False,
                      closing=closing, close_deferred=False, selected_is_actor=True, lewm_calls=0,
                      response_source=self.response.source, target_px=self.target.tolist(),
                      close_tolerance_px=self.close_tolerance_px)
        fish, box = _seen(token), _seen(box_token)
        opened = bool(q[5] >= OPEN_GRIP)
        between = bool(fish is not None and np.linalg.norm(fish[2:4] * IMAGE - self.target) <= self.close_tolerance_px)
        if self.grasped and opened and not between and box is not None:
            record.update(released_over_box=True)
        if not opened and between:
            self.grasped = True
        name, item = ('fish', fish) if opened else ('box', box)
        # section 17.118: the measured box solid stands for the box whether or not the wrist sees it (the wrist only
        # says the box is in its picture; tilted up it reads "centred" while the jaws are at the near wall)
        # section 17.148: near the measured box with the wrist showing it to one side (box_from_wrist), the wrist
        # picture leads the jaws over it instead of the solid measured from the part the column picture shows
        solid = (name == 'box' and self.grasped and self.box_solid is not None and self.camera is not None
                 and not box_from_wrist)
        # (trial 19: jaws shut on nothing are no carry; the solid is aimed at only after a grasp)
        # section 17.118 d (trial 20, James 「不应该column+wrist都联合判断马？」): jaws open and the fish not in the wrist
        # picture, the column camera's fish leads the jaws towards it until the wrist sees it and takes over; 11/03
        # (James 「这个模型在干嘛呢」): also while the jaws are further than NEAR_FISH_MM from it on the table, whatever
        # the wrist reports (the column picture shows the whole table; the wrist far from the fish reported other things)
        apart_mm = None
        if name == 'fish' and fish_point is not None and self.camera is not None:
            try:
                jaws = np.asarray(self.camera.jaw(q[None]), np.float64).reshape(-1)[:2]
                apart_mm = float(np.linalg.norm(np.asarray(fish_point, np.float64).reshape(-1)[:2] - jaws) * 1000.)
            except Exception:
                apart_mm = None
            apart_mm = apart_mm if apart_mm is not None and np.isfinite(apart_mm) else None
        far = apart_mm is not None and apart_mm > NEAR_FISH_MM
        guided = (name == 'fish' and fish_point is not None and self.camera is not None and (fish is None or far))
        column_target = (np.asarray(self.box_solid.centre, np.float64) if solid else
                         np.asarray(fish_point, np.float64).reshape(-1)[:2] if guided else None)
        table = None
        if column_target is not None:
            try:
                table = JawTable(self.camera, q)
            except ValueError:
                table = None
        travelling = bool(np.abs(proposal[:, :ARM] - q[:ARM]).max() > TRAVEL)
        record.update(object=name, box_seen=box is not None, grasped=self.grasped,
                      grasp_seen=bool(not opened and between),          # this block's grasp condition (GripLatch)
                      travelling=travelling, align_box=self.align_box, box_height_m=self.box_height_m,
                      box_solid=None if self.box_solid is None else self.box_solid.describe(),
                      column_fish_mm=None if fish_point is None else
                      [round(float(v) * 1000., 1) for v in np.asarray(fish_point, np.float64).reshape(-1)],
                      aimed_by='box_solid' if solid else 'column_fish' if guided else 'wrist',
                      column_fish_from_jaws_mm=apart_mm, column_leads_far=bool(guided and fish is not None),
                      box_from_wrist=bool(box_from_wrist))
        if item is None and column_target is None:
            reason = name + '_not_seen'
        elif name == 'box' and not self.align_box:
            reason = 'box_alignment_off'
        elif name == 'box' and closing:
            reason = 'actor_closing'
        elif travelling and not (self.steer_approach and (name == 'fish' or solid)):
            reason = 'actor_travelling'
        elif column_target is not None and table is None:
            reason = 'jaws_not_placed'
        else:
            reason = None
        if reason is None and column_target is not None:
            offset_mm = column_target * 1000. - table.jaw_mm[:2]
            record.update(jaw_mm=table.jaw_mm.round(1).tolist())
            # a close while the column leads (the fish not between the jaws in the wrist) waits like a close beside it
            selected = self._align(q, proposal, offset_mm, forecast, admissible, record, closing and guided, commit, table,
                                   target=np.zeros(2), units='mm', floor=table, lift=solid,
                                   fish_axis=self._roll_axis(fish_axis, record) if guided else None,
                                   reach=FAR_SCALE if guided and far else 1.)
        elif reason is None:
            selected = self._align(q, proposal, item[2:4] * IMAGE, forecast, admissible, record,
                                   closing and name == 'fish', commit, self.box_response if name == 'box' else None,
                                   jaw_height=jaw_height if name == 'fish' else None,
                                   fish_axis=self._roll_axis(fish_axis, record) if name == 'fish' else None)
        else:
            selected = proposal.copy()
            record.update(reason=reason)
        selected = self._admit(q, proposal, selected, admissible, record)
        record.update(planning_s=self.clock() - started)
        return selected, record

    def _roll_axis(self, fish_axis, record):
        """align_roll: the fish on the table this plan turns the roll by, or None (with the reason recorded)."""
        if not self.align_roll:
            return None
        try:
            direction = np.asarray(fish_axis['direction'], np.float64).reshape(2)
            length = float(fish_axis['length_m'])
        except (KeyError, TypeError, ValueError):
            record.update(roll=dict(reason='no_fish_axis'))
            return None
        if not np.isfinite(direction).all() or abs(np.linalg.norm(direction) - 1.) > 1e-6 or not length > 0.:
            record.update(roll=dict(reason='no_fish_axis'))
            return None
        return dict(direction=direction, length_m=length)

    def _descent(self, q, response, jaw_height):
        """Section 17.140: [4, 3] smallest change of the searched joints for a wanted (fish shift px [2], jaw rise mm),
        each joint measured against what it may depart (as WristResponse.offset), from the response and the jaws'
        height row at q by central differences; None where the heights cannot be had or the searched joints cannot
        move the fish in the picture and the jaws' height apart."""
        states = np.repeat(np.asarray(q, np.float64)[None], 2 * ARM, 0)
        for joint in range(ARM):
            states[2 * joint, joint] += FD_STEP
            states[2 * joint + 1, joint] -= FD_STEP
        try:
            heights = np.asarray(jaw_height(states), np.float64).reshape(-1)
        except Exception:
            return None
        if heights.shape != (2 * ARM,) or not np.isfinite(heights).all():
            return None
        row = (heights[0::2] - heights[1::2]) / (2 * FD_STEP)
        searched = np.vstack((np.asarray(response.jacobian, np.float64), row[None]))[:, list(SEARCH_JOINTS)]
        if np.linalg.matrix_rank(searched, tol=1e-6) < 3:
            return None
        weight = np.diag(MAX_OFFSET ** 2)
        return weight @ searched.T @ np.linalg.inv(searched @ weight @ searched.T)

    def _align(self, q, proposal, token_px, forecast, admissible, record, gate_close, commit, response=None,
               target=None, units='px', floor=None, lift=False, jaw_height=None, fish_axis=None, reach=1.):
        """The paper's step on one object: candidates ranked by the forecast offset of the object from the jaw centre
        (token_px: its offset now, in the response's units); a close (gate_close) is deferred while that offset exceeds
        close_tolerance_px. response: another geometry for this object (the box's rim plane; section 17.118 a JawTable,
        in mm against the box solid's centre), default the planner's; target in its units, default target_px.
        floor (a JawTable, section 17.118 b): no candidate whose forecast jaws end more than HEIGHT_SLACK_MM lower than
        the Actor's own forecast is chosen; the height, and lifting over what is in the way, stay the Actor's.
        lift (with floor; section 17.120, the carry): the search also moves the jaws' height, and each mm the forecast
        jaws end below the Actor's own last command of the block costs SAG_WEIGHT, so the arm ends where the Actor
        commanded although it sags under the fish.
        jaw_height (section 17.140; aiming at the fish in the wrist picture, no floor): while the Actor closes, the
        search also moves the jaws' height, each mm the forecast jaws are above GRASP_MM at the close check costs
        DESCEND_WEIGHT, and the close is also held back while they are above CLOSE_HIGH_MM there.
        fish_axis (align_roll; the fish on the table, fish_on_table): the search also turns the wrist roll from a
        first-order guess of the turn that lays the jaw line across the fish (both on the table; the jaw line by forward
        kinematics in the column camera's frame, its turn per roll unit by central differences), each unit of the arc
        the fish's ends are from lying across the forecast jaw line (arc_from_across; in the cost's units: wrist px at
        the fish's depth, or mm) costs ORIENT_WEIGHT, and a close is also held back while that arc is longer than
        ORIENT_TOLERANCE_PX at the close check. reach: the search box's scale (FAR_SCALE while the column fish leads
        far jaws)."""
        token_px = np.asarray(token_px, np.float64)
        target = self.target if target is None else np.asarray(target, np.float64)
        record.update(units=units, search_scale=float(reach))
        response = self.response if response is None else response
        geometry = response if isinstance(response, GeometricResponse) else None
        if isinstance(response, GeometricResponse):
            response = response.local(q, token_px)
            if response is None:
                record.update(reason='geometry_unavailable')
                return proposal.copy()
            record.update(local_jacobian_px_per_unit=response.jacobian.tolist())
        roll = None
        if fish_axis is not None:
            # the cost's units per metre on the table: mm, or the wrist's px at the fish's depth
            scale = 1000. if units == 'mm' else None if geometry is None else geometry.table_scale(q, token_px)
            around = np.repeat(q[None], 3, 0)
            around[1, ROLL] += FD_STEP
            around[2, ROLL] -= FD_STEP
            lines = self.camera.jaw_line(around)
            if scale is None or not np.isfinite(lines).all():
                record.update(roll=dict(reason='geometry_unavailable' if scale is None else 'jaw_line_upright'))
            else:
                axis = fish_axis['direction']
                normal = np.asarray((-axis[1], axis[0]))           # the jaw line lies across the fish along a normal
                normal = normal if normal @ lines[0] >= 0 else -normal
                turn = float(np.arctan2(lines[0, 0] * normal[1] - lines[0, 1] * normal[0], lines[0] @ normal))
                yaw = np.arctan2(lines[1:, 1], lines[1:, 0])
                rate = float(np.angle(np.exp(1j * (yaw[0] - yaw[1])))) / (2 * FD_STEP)   # jaw line turn, rad per unit
                guess = float(np.clip(turn / rate, -MAX_ROLL_OFFSET, MAX_ROLL_OFFSET)) if abs(rate) > 1e-4 else 0.
                roll = dict(scale=scale, rate=rate, turn=turn, guess=guess, line=lines[0])

        def arc(path, state=PREFIX):
            if roll is None:
                return np.zeros(len(path))
            distance = arc_from_across(fish_axis, self.camera.jaw_line(np.asarray(path)[:, state]))
            # an upright forecast jaw line holds nothing across: the longest arc
            return roll['scale'] * np.where(np.isfinite(distance), distance, .25 * np.pi * fish_axis['length_m'])
        commands = [proposal[None].copy()]
        paths = [np.asarray(forecast(commands[0]), np.float64)]
        predicted = [self.predicted_offset(q, token_px, paths[0], response)]
        offsets = [np.zeros((1, len(SEARCH_JOINTS) + (roll is not None)))]
        lowest = None if floor is None else float(floor.height(paths[0][:, PREFIX])[0]) - HEIGHT_SLACK_MM
        commanded = float(floor.height(proposal[PREFIX - 1])) if lift and floor is not None else None
        descend = None
        # section 17.140 b: only on the last approach -- the fish in the wrist picture with the jaws near it and nothing
        # grasped yet in this trial (those after a release are none of the descent's business). 11/03: near it is where
        # the wrist aims at all (the column fish leads jaws further than NEAR_FISH_MM); the fish within the close
        # tolerance of the jaw centre is no longer asked for (rollf trial 1: with it 35 px off the Actor's close went
        # ahead with the jaws 45-76 mm above it, no height asked)
        if gate_close and jaw_height is not None and floor is None and not self.grasped:
            descend = self._descent(q, response, jaw_height)
            if descend is None:
                record.update(descend='height_unavailable')

        def close_heights(path):
            return np.asarray(jaw_height(np.asarray(path)[:, CLOSE_CHECK]), np.float64).reshape(-1)

        def descended(path):
            if descend is None:
                return np.zeros(len(path))
            height = close_heights(path)
            return DESCEND_WEIGHT * (np.maximum(height - GRASP_MM, 0.) + np.maximum(GRASP_LOW_MM - height, 0.))

        def lowered(path):
            if lowest is None:
                return np.zeros(len(path))
            return np.where(floor.height(path[:, PREFIX]) < lowest, np.inf, 0.)

        def sagged(path):
            if commanded is None:
                return np.zeros(len(path))
            return SAG_WEIGHT * np.maximum(commanded - SAG_SLACK_MM - floor.height(path[:, PREFIX]), 0.)
        need = target - predicted[0][0]
        if roll is not None:        # the roll's own shift of the fish at its guess, made up by the other joints
            need = need - response.jacobian[:, ROLL] * roll['guess']
        mean, spread = need.copy(), np.maximum(np.abs(need) / 2., 4.)
        if commanded is not None:
            rise = commanded - float(floor.height(paths[0][:, PREFIX])[0])      # the sag the LeWM forecasts
            mean, spread = np.r_[mean, rise], np.r_[spread, max(abs(rise) / 2., 2.)]
        if descend is not None:
            rise = GRASP_MM - float(close_heights(paths[0])[0])                 # how far the jaws are to go down
            mean, spread = np.r_[mean, rise], np.r_[spread, max(abs(rise) / 2., 2.)]
            # 11/03 (rollg trial 2): the Actor kept the jaws 29-48 mm up through six held closes while the search,
            # within MAX_OFFSET, lowered them a few mm a block; above CLOSE_HIGH_MM it may depart as far as for far jaws
            if GRASP_MM - rise > CLOSE_HIGH_MM:
                reach = max(reach, FAR_SCALE)
                record.update(search_scale=float(reach))
        if roll is not None:                                                    # the roll itself, last
            mean, spread = np.r_[mean, roll['guess']], np.r_[spread, max(abs(roll['guess']) / 2., 2.)]
        rounds = []
        for _ in range(ITERATIONS):
            wanted = np.concatenate((mean[None], mean + spread * self.rng.standard_normal((POPULATION - 1, len(mean)))))
            main = wanted[:, :-1] if roll is not None else wanted
            offset = np.clip(floor.offset_rise(main) if commanded is not None else
                             main @ descend.T if descend is not None else response.offset(main),
                             -reach * MAX_OFFSET, reach * MAX_OFFSET)
            if roll is not None:
                offset = np.c_[offset, np.clip(wanted[:, -1], -MAX_ROLL_OFFSET, MAX_ROLL_OFFSET)]
            batch = self.candidates(proposal, offset)
            path = np.asarray(forecast(batch), np.float64)
            guess = self.predicted_offset(q, token_px, path, response)
            cost = (np.linalg.norm(guess - target, axis=1) + lowered(path) + sagged(path) + descended(path)
                    + ORIENT_WEIGHT * arc(path))
            keep = np.argsort(cost, kind='stable')[:ELITES]
            mean, spread = wanted[keep].mean(0), np.maximum(wanted[keep].std(0), 1.)
            commands.append(batch)
            paths.append(path)
            predicted.append(guess)
            offsets.append(offset)
            finite = np.isfinite(cost)      # section 17.118 b: candidates below the Actor's height cost inf; never recorded
            rounds.append(dict(best_px=float(cost[finite].min()) if finite.any() else None,
                               elite_mean_px=float(cost[keep][finite[keep]].mean()) if finite[keep].any() else None,
                               lowered=int((~finite).sum())))
        commands, predicted, offsets = np.concatenate(commands), np.concatenate(predicted), np.concatenate(offsets)
        paths = np.concatenate(paths)
        costs = (np.linalg.norm(predicted - target, axis=1) + lowered(paths) + sagged(paths) + descended(paths)
                 + ORIENT_WEIGHT * arc(paths))
        chosen, checked = 0, 0
        for index in np.argsort(costs, kind='stable')[:RANKED_CHECKS]:
            if costs[index] > costs[0] - MIN_GAIN_PX:
                break
            checked += 1
            if admissible(commands[index]):
                chosen = int(index)
                break
        selected = commands[chosen].copy()
        close_px = self.predicted_offset(q, token_px, paths[chosen:chosen + 1], response, state=CLOSE_CHECK)[0]
        close_cost = float(np.linalg.norm(close_px - target))
        close_height = None if descend is None else float(close_heights(paths[chosen:chosen + 1])[0])
        close_arc = float(arc(paths[chosen:chosen + 1], CLOSE_CHECK)[0])
        aimed = close_cost <= self.close_tolerance_px and close_arc <= ORIENT_TOLERANCE_PX
        # section 17.140 b: a close aimed at the fish but too high is held back on its own count
        high = bool(aimed and close_height is not None and close_height > CLOSE_HIGH_MM
                    and self.height_deferrals < self.max_deferrals)
        deferred = bool(gate_close and ((not aimed and self.deferrals < self.max_deferrals) or high))
        if deferred:
            held = selected.copy()
            held[:, 5] = q[5]
            if admissible(held):
                selected = held
            else:
                deferred = False
        if commit:
            self.deferrals = self.deferrals + 1 if deferred and not high else 0 if gate_close and aimed else (
                0 if gate_close and not deferred else self.deferrals)
            self.height_deferrals = self.height_deferrals + 1 if deferred and high else (
                0 if gate_close and not deferred else self.height_deferrals)
        record.update(
            searched=True, reason=None if chosen else ('no_gain' if checked == 0 else 'not_admitted'),
            selected_is_actor=chosen == 0, close_deferred=deferred, deferrals=self.deferrals,
            lewm_calls=1 + ITERATIONS, candidates=int(len(costs)), admission_checks=checked,
            token_px=token_px.tolist(), proposal_predicted_px=predicted[0].tolist(), proposal_cost_px=float(costs[0]),
            selected_predicted_px=predicted[chosen].tolist(), selected_cost_px=float(costs[chosen]),
            close_check_px=close_px.tolist(), close_check_cost_px=close_cost,
            best_cost_px=float(costs.min()), offset=offsets[chosen].tolist(), rounds=rounds)
        if descend is not None:
            record.update(height_deferrals=self.height_deferrals)
            record.update(descend=dict(section='17.140', grasp_mm=GRASP_MM, grasp_low_mm=GRASP_LOW_MM,
                                       close_high_mm=CLOSE_HIGH_MM,
                                       weight=DESCEND_WEIGHT, held_back_high=bool(high and deferred),
                                       proposal_close_height_mm=float(close_heights(paths[:1])[0]),
                                       selected_close_height_mm=close_height))
        if roll is not None:
            ahead = self.camera.jaw_line(paths[[0, chosen], PREFIX])
            record.update(roll=dict(
                source='the column picture\'s fish and the forward-kinematics jaw line, on the table',
                fish_direction=np.asarray(fish_axis['direction']).tolist(), fish_length_mm=fish_axis['length_m'] * 1e3,
                jaw_line=roll['line'].tolist(), scale_per_m=roll['scale'], units=units,
                rate_rad_per_unit=roll['rate'], turn_rad=roll['turn'], guess_units=roll['guess'],
                across_deg_now=float(across_deg(fish_axis, roll['line'][None])[0]),
                across_deg_proposal=float(across_deg(fish_axis, ahead[:1])[0]),
                across_deg_selected=float(across_deg(fish_axis, ahead[1:])[0]),
                proposal_arc=float(arc(paths[:1])[0]), selected_arc=float(arc(paths[chosen:chosen + 1])[0]),
                close_arc=close_arc, offset_units=float(offsets[chosen][-1]), weight=ORIENT_WEIGHT,
                tolerance=ORIENT_TOLERANCE_PX, max_offset_units=MAX_ROLL_OFFSET))
        if floor is not None:
            record.update(height_floor='the Actor\'s own forecast (section 17.118 b)',
                          proposal_height_mm=float(floor.height(paths[0][PREFIX])),
                          selected_height_mm=float(floor.height(paths[chosen][PREFIX])),
                          lowered_candidates=int(np.isinf(costs).sum()))
        if commanded is not None:
            record.update(commanded_height_mm=commanded, sag_weight=SAG_WEIGHT,
                          proposal_shortfall_mm=max(commanded - float(floor.height(paths[0][PREFIX])), 0.),
                          selected_shortfall_mm=max(commanded - float(floor.height(paths[chosen][PREFIX])), 0.))
        return selected

    def _admit(self, q, proposal, selected, admissible, record):
        """The selection when the physical gate admits it, else the admissible candidate nearest the Actor's
        proposal (offsets inside MAX_OFFSET, smallest first), else the arm held where it is (gripper the proposal's)."""
        if admissible(selected):
            record.update(admission='selected')
            return selected
        offsets = self.rng.uniform(-1., 1., (POPULATION, len(SEARCH_JOINTS))) * MAX_OFFSET
        offsets = offsets[np.argsort(np.linalg.norm(offsets / MAX_OFFSET, axis=1), kind='stable')]
        for index, candidate in enumerate(self.candidates(proposal, offsets)):
            if admissible(candidate):
                record.update(admission='nearest_admissible', admission_offset=offsets[index].tolist(),
                              admission_tries=index + 1, selected_is_actor=False, close_deferred=False)
                return candidate
        hold = np.repeat(q[None], 5, 0)
        hold[:, 5] = proposal[:, 5]
        if admissible(hold):
            record.update(admission='hold', admission_tries=len(offsets) + 1, selected_is_actor=False,
                          close_deferred=False)
            return hold
        record.update(admission='none_admissible', admission_tries=len(offsets) + 1)
        return selected


class GripLatch:
    """The gripper rule of training overhaul doc section 17.45 (James 10/20: "j6可以hack"); joint 6 only.

    A hold starts at the block whose plan sees the fish between shut jaws (the planner's grasp condition) with
    something between them (read-back at least EMPTY_GRIP). The hold level starts SQUEEZE below the read-back, follows
    the read-back down while the jaws settle, but never goes more than SQUEEZE_LIMIT below the grasp read-back (10/20
    trial 1: a level that chased the read-back squeezed the fish out like soap). The jaw commands are the Actor's,
    capped at the hold level and, from 11/03, held at most SQUEEZE_LIMIT below the grasp read-back as well (the
    Actor's own squeeze pushed the fish out); the jaws never open on the way. The hold ends when the
    Actor asks for open jaws (its last gripper command at least OPEN_GRIP) with the box in the wrist picture -- that
    block opens to at least RELEASE_OPEN, ramped in -- or when nothing is held: the jaws met (read-back below
    CLOSED_GRIP) or LOST_PLANS plans in a row do not see the fish between them; that block opens the jaws again in
    the same way, for another grasp (section 17.140). Outside a hold the gripper is the Actor's (None). A release asked for where the adapter holds it back is held back, at most RELEASE_DEFERRALS times
    in one hold; the arm stays the Actor's and CEM's. The adapter holds back while the jaws are not over the box solid
    it measured, without one while they are more than RELEASE_HIGH_MM above the table (section 17.109: demonstration 0
    let go inside the box at 34 mm, HLwb at 125-145 mm and on 10/26 trial 12 the fish fell outside the box; section
    17.120: over the measured box the height no longer counts -- in trials 22-26 the Actor asked to open within 24 mm of
    the box's centre at 80-110 mm, no model lowers the jaws into the box, so the height hold-back only waited out its
    six deferrals, 7-16 s, and every fish fell into the box from 90-145 mm)."""

    def __init__(self, lower=0.):
        self.lower = float(lower)
        self.reset()

    def reset(self):
        self.holding, self.command, self.grasp_reading, self.unseen, self.held_back = False, None, None, 0, 0

    def step(self, reading, actor5, grasp_seen, box_seen, hold_back=False):
        """reading: the jaw read-back; actor5: the Actor's five gripper commands of this block; hold_back: a release asked
        for now would miss the box (the adapter's test; False when it cannot tell) and is held back, at most
        RELEASE_DEFERRALS times in one hold (training overhaul doc sections 17.109, 17.120).
        -> (five gripper commands, or None for the Actor's; event: free / grasp / hold / release_held / release / empty)"""
        reading, actor5 = float(reading), np.asarray(actor5, np.float64).reshape(-1)
        if actor5.shape != (5,) or not np.isfinite(reading) or not np.isfinite(actor5).all():
            raise ValueError('gripper rule needs a finite read-back and five finite Actor commands')
        if not self.holding:
            if grasp_seen and reading >= EMPTY_GRIP:
                self.holding, self.grasp_reading, self.unseen, self.held_back = True, reading, 0, 0
                self.command = max(self.lower, reading - SQUEEZE)
                return np.clip(actor5, self._squeeze_floor(), self.command), 'grasp'
            return None, 'free'
        self.unseen = 0 if grasp_seen else self.unseen + 1
        if reading < CLOSED_GRIP or self.unseen >= LOST_PLANS:
            # section 17.140 (17.138 p16-1, p16-5, p16-6): nothing held, so the jaws open again as at a release and
            # the adapter sends the planner back to the fish (it had kept the grasp and taken the empty jaws to the box)
            self.reset()
            return reading + (max(RELEASE_OPEN, reading) - reading) * RAMP, 'empty'
        event = 'hold'
        if box_seen and actor5[-1] >= OPEN_GRIP:
            if not (hold_back and self.held_back < RELEASE_DEFERRALS):
                self.reset()
                return reading + (max(float(actor5[-1]), RELEASE_OPEN) - reading) * RAMP, 'release'
            self.held_back += 1
            event = 'release_held'
        self.command = max(self.lower, self.grasp_reading - SQUEEZE_LIMIT, min(self.command, reading - SQUEEZE))
        return np.clip(actor5, self._squeeze_floor(), self.command), event

    def _squeeze_floor(self):
        """11/03: the Actor's own commands are held at most SQUEEZE_LIMIT below the grasp read-back too (the release
        Actor closes to 0: in 3 of the day's 4 holds the read-back sank 6-9 below the grasp's and the fish slipped out,
        as the hold level that chased the read-back on 10/20)."""
        return min(self.command, max(self.lower, self.grasp_reading - SQUEEZE_LIMIT))


def wrist_box_sides_px(box_token):
    """Section 17.148: how far the box in the wrist picture (a token in the fish token's layout, against the hold
    point) reaches to the left and to the right of the hold point along the picture's x, in letterbox224 pixels; None
    when the box is not seen or the token carries no extent. Only the sideways reach: tilted up, the wrist reads
    the box centred while the jaws are at its near wall (section 17.118), so forwards and back stay the column's."""
    box = _seen(box_token)
    if box is None or len(box) < 8 or not np.isfinite(box[4:8]).all() or max(box[4], box[5]) <= 0:
        return None
    theta = .5 * np.arctan2(box[7], box[6])
    half = .5 * (abs(box[4] * np.cos(theta)) + abs(box[5] * np.sin(theta))) * IMAGE
    return float(half - box[2] * IMAGE), float(half + box[2] * IMAGE)


def _seen(token):
    """A wrist object token [8] (fcpg/v21_object_relative.py layout) when it marks a visible object, else None."""
    if token is None:
        return None
    token = np.asarray(token, np.float64).reshape(-1)
    return token if len(token) >= 4 and np.isfinite(token[:4]).all() and token[0] > .5 else None


class AlignmentCEMAdapter:
    """Service wiring; the same small interface as fcpg/v21_c4_service.C4ServiceAdapter."""
    record_key = RECORD_KEY

    def __init__(self, mode, response, *, bf16=True, save_inputs=True, grip_latch=False, box_solid_camera=None,
                 dynamics='lewm', **planner_options):
        if mode not in MODES:
            raise ValueError('alignment CEM mode must be shadow or control')
        if not isinstance(grip_latch, bool):
            raise ValueError('grip_latch is a bool')
        if dynamics not in DYNAMICS:
            raise ValueError(f'alignment CEM dynamics must be one of {sorted(DYNAMICS)}')
        self.dynamics = dynamics
        self.mode, self.source = mode, SOURCES[mode]
        self.response, self.bf16, self.save_inputs = response, bool(bf16), bool(save_inputs)
        self.planner_options = planner_options
        # section 17.118: the column camera that measures the box solid (a v21-column-self-calibration-v1 file)
        self.box_solid_camera = None if box_solid_camera is None else Path(box_solid_camera)
        self.column_camera = None if box_solid_camera is None else load_column_camera(box_solid_camera)
        self.box_solid_fit = None
        self.grip_latch = grip_latch       # section 17.45: control executes it, shadow only records it
        self.latch = None
        self.let_go = False                # section 17.120 trial 24: the latch let go over the box, no grasp since
        self.planner = None
        self.record = None
        self.offset5 = None
        self.grip_hold = None
        self.grip_override = None
        self.planned_at_s = None
        self.input_errors = 0
        self.fish_axis = None              # the whole fish on the table this trial (_fish_axis)
        self.fish_length_m = None
        self.grasp_s = None                # when the grip latch last took a grasp this trial (_fish_in_box)
        self.column_reference_of = None    # the wrist text finder's column reference (_choose_wrist_reference)

    def attach(self, service):
        self.service, self.clock = service, service.clock
        self.column_reference_of = getattr(getattr(service.owner, 'object_relative', None), 'reference_of', None)
        actor = service.owner.runtime.core.actor
        bounds = [None if getattr(actor, name, None) is None else getattr(actor, name).detach().float().cpu().numpy()
                  for name in ('position_lower', 'position_upper')]
        if (bounds[0] is None) != (bounds[1] is None):
            raise ValueError('alignment CEM needs both joint limits or neither')
        self.planner = AlignmentCEM(self.response, lower=bounds[0], upper=bounds[1], column_camera=self.column_camera,
                                    **self.planner_options)
        if self.column_camera is not None and (getattr(service, 'object_finder', None) or {}).get('box') is None:
            raise ValueError('the box solid is measured from the learned box finder\'s column map (section 17.57)')
        if self.grip_latch:
            self.latch = GripLatch(lower=0. if bounds[0] is None else float(bounds[0][5]))
        # A model that reads where the object goes (V21Config.place_relative, training overhaul doc section 17) carries
        # that token in its state; the CEM then aligns to the very token the Actor sees. Earlier models: the service's
        # DINO targets, as before.
        reads_place = bool(getattr(service.owner.runtime.core.config, 'place_relative', False))
        # section 17.117: a model of the 'hold' convention (the outline cues) reads its fish and place cues against where
        # the demonstrations' jaws held the fish; the CEM keeps the jaw midpoint its grasp test (15 px), the grip latch
        # and released_over_box were made with: the fish cue moved back to it, the box from the service's own targets
        encoder = getattr(service.owner, 'object_relative', None)
        self.hold_px = (np.asarray(encoder.hold_px, np.float64).reshape(2)
                        if getattr(encoder, 'convention', None) == 'hold' else None)
        self.place_source = 'actor_state' if reads_place and self.hold_px is None else 'service_targets'
        registration = getattr(service, 'wrist_drop_zone', None)
        learned_box = (getattr(service, 'object_finder', None) or {}).get('box')
        box_finder = ('the learned box finder on the wrist tokens of the exposure (training overhaul doc section 17.57, '
                      'fcpg/v21_object_finder.LearnedBoxFinder)' if learned_box is not None else
                      'the drop zone matched in the wrist with its wrist-view registration (fcpg/v21_target_tracker.py)'
                      if registration else 'the drop zone matched in the wrist with the column registration role '
                      'prototype (fcpg/v21_target_tracker.py)')
        self.identity = dict(planner='system_lewm_alignment_cem', response=self.response.source,
                             response_pairs=self.response.pairs,
                             **(dict(jacobian_px_per_unit=self.response.jacobian.tolist())
                                if isinstance(self.response, WristResponse) else
                                dict(geometric_theta=dict(zip(GEOMETRIC_NAMES, self.response.theta.tolist())))),
                             lewm=type(service.owner.runtime.core.lewm).__name__, bf16=self.bf16,
                             dynamics=dict(kind=self.dynamics, forecast=DYNAMICS[self.dynamics]),
                             population=POPULATION, elites=ELITES, iterations=ITERATIONS, prefix=PREFIX,
                             close_check=CLOSE_CHECK)
        self.identity.update(
            objects=dict(open='fish: the policy wrist token' + (
                             ' (its hold-convention centre moved back to the jaw midpoint, section 17.117)'
                             if self.hold_px is not None else ''),
                         shut=('place: the policy wrist place token (the drop zone matched in the wrist with the column '
                               'registration role prototype, fish-token layout)' if self.place_source == 'actor_state'
                               else f'box: {box_finder}, fish-token layout')),
            place_source=self.place_source,
            hold_px=None if self.hold_px is None else self.hold_px.tolist(),
            wrist_drop_zone=None if registration is None else {key: registration[key] for key in
                                                               ('path', 'sha256', 'source', 'centre_px', 'radius_px')},
            open_from=OPEN_GRIP, max_offset=MAX_OFFSET.tolist(), close_tolerance_px=self.planner.close_tolerance_px,
            travel=TRAVEL, steer_approach=self.planner.steer_approach, align_box=self.planner.align_box,
            box_height_m=self.planner.box_height_m,
            align_roll=dict(on=self.planner.align_roll, max_offset=MAX_ROLL_OFFSET, weight=ORIENT_WEIGHT,
                            tolerance=ORIENT_TOLERANCE_PX, elongated=ELONGATED, max_fish_mm=MAX_FISH_MM,
                            length_agree=FISH_LENGTH_AGREE,
                            fish='the learned column finder\'s outline, its ends placed on the table at the fish\'s '
                                 'half height (fish_on_table); the trial\'s first sets the fish\'s length, a later one '
                                 'replaces the axis when its length agrees',
                            jaw_line='forward kinematics in the column camera\'s frame (ColumnCamera.jaw_line)'),
            box_solid=None if self.column_camera is None else dict(
                camera=str(self.box_solid_camera), section='17.118',
                measured='a cuboid on the table fitted to the learned box finder\'s column cells at the trial\'s first '
                         'block (fit_box_solid); the shut jaws aimed over its centre by forward kinematics (JawTable), '
                         'their height left to the Actor; the wrist box only for released_over_box and the grip latch; '
                         'jaws open with the fish not in the wrist picture, the column finder\'s fish (ColumnCamera.'
                         'fish_point) leads the jaws the same way until the wrist sees it (section 17.118 d), and '
                         f'until they are within {NEAR_FISH_MM:g} mm of it on the table whatever the wrist reports (11/03)',
                search=dict(low=BOX_SOLID_LOW.tolist(), high=BOX_SOLID_HIGH.tolist(), population=BOX_SOLID_POPULATION,
                            elites=BOX_SOLID_ELITES, rounds=BOX_SOLID_ROUNDS, restarts=BOX_SOLID_RESTARTS),
                carry_sag=dict(section='17.120', weight=SAG_WEIGHT, slack_mm=SAG_SLACK_MM,
                               rule='carrying, each mm the LeWM forecasts the jaws below the Actor\'s own last command '
                                    'costs the weight; the search also moves their height')),
            admission='selection, else the admissible candidate nearest the proposal, else hold the arm',
            released_over_box='grasped (fish between shut jaws), then jaws open, fish not between them, box seen in '
                              'the wrist picture; or, the grip latch in control, jaws open after the latch let go '
                              '(released_over_box_by grip_latch; section 17.120 trial 24)',
            gripper=('the Actor\'s' if self.latch is None else
                     dict(rule='GripLatch (training overhaul doc section 17.45)', executed=self.mode == 'control',
                          squeeze=SQUEEZE, squeeze_limit=SQUEEZE_LIMIT, empty_grip=EMPTY_GRIP, closed_grip=CLOSED_GRIP,
                          lost_plans=LOST_PLANS, release_open=RELEASE_OPEN, open_from=OPEN_GRIP,
                          release_held_back=('while the jaws are not over the measured box solid (section 17.120)'
                                             if self.column_camera is not None else
                                             f'while the jaws are more than {RELEASE_HIGH_MM:g} mm above the table '
                                             '(section 17.109)'),
                          release_deferrals=RELEASE_DEFERRALS,
                          dropped=f'a hold found empty more than {DROPPED_ABOVE_MM:g} mm above the table forgets the kept '
                                  'whole fish; the next column outline sets it again (11/04)',
                          fish_in_box='open jaws holding nothing with the whole fish an outline showed after the last '
                                      'grasp inside the measured box solid\'s footprint: released_over_box by '
                                      'column_fish_in_box (11/04)')))

    def close(self):
        pass

    def _forecast(self, state, command_offsets):
        """-> forecast(commands [n, 5, 6] numpy) -> [n, 6, 6] numpy: the system LeWM, as V21ResearchCore.predict
        calls it, for n candidates at once; with dynamics 'commands', the source state followed by the five commands
        themselves (the arm reaches each exactly: no sag, no lag)."""
        if self.dynamics == 'commands':
            source = state.q.detach().double().cpu().numpy().reshape(-1)

            def follows(commands):
                commands = np.asarray(commands, np.float64)
                if commands.ndim != 3 or commands.shape[1:] != (5, source.size):
                    raise ValueError(f'commands [n, 5, {source.size}], got {commands.shape}')
                return np.concatenate([np.broadcast_to(source, (len(commands), 1, source.size)), commands], 1)
            return follows
        from .v21_lewm_cem import _repeat_visual_memory, repeat_history
        from .v21_models import CommandTrace
        from .v21_picture_memory import expand_pictures
        lewm = self.service.owner.runtime.core.lewm
        pictures = (self.service.owner.runtime.core.lewm_pictures(state) if hasattr(lewm, 'memory_pictures')
                    else None)
        device, dtype = state.q.device, state.q.dtype
        offsets = command_offsets.to(device=device, dtype=torch.float64)
        autocast = self.bf16 and device.type == 'cuda'

        def forecast(commands):
            n = len(commands)
            trace = CommandTrace(torch.as_tensor(np.asarray(commands), dtype=dtype, device=device), offsets.expand(n, -1),
                                 torch.ones(n, 5, dtype=torch.bool, device=device), 'planned', True)
            if pictures is not None:     # 17.167: the state's last N per camera, one memory for every candidate
                shared = expand_pictures(pictures, n)
                visual = dict(patches=shared.features, patch_valid=shared.valid, memory=shared,
                              now_time=state.time.expand(n))
            else:
                visual = dict(patches=state.memory.features.expand(n, -1, -1, -1, -1),
                              patch_valid=state.memory.valid.expand(n, -1, -1))
                if getattr(lewm, 'reads_visual_memory', False):
                    visual.update(memory=_repeat_visual_memory(state.memory, n), now_time=state.time.expand(n))
            if getattr(lewm, 'controller_input', False):     # 17.153: the arm's control setting of this runtime
                visual['controller'] = self.service.owner.runtime.core.lewm_controller(n, state.q)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                out = lewm(state.q.expand(n, -1), trace, repeat_history(state.history, n), (offsets + .1).expand(n, -1), **visual)
            return out.predicted_q_path.float().cpu().numpy()
        return forecast

    def _token(self, state):
        token = getattr(state, 'object_relative', None)
        return None if token is None else token.detach().float().cpu().numpy().reshape(-1)

    def _jaw_layout(self, value, q):
        """The fish token the CEM plans with, in the jaw-midpoint layout: a 'hold' cue (section 17.117) moved back to the
        jaw midpoint at q, the same centre; size and axis fields as given (the CEM reads only the centre)."""
        hold = getattr(self, 'hold_px', None)
        if hold is None or value is None or len(value) < 4 or not value[0] > .5:
            return value
        pair = np.asarray(self.service.owner.object_relative.jaw_pair(q), np.float64).reshape(2, 2)
        out = np.asarray(value, np.float64).copy()
        out[2:4] = (out[2:4] * IMAGE + hold - pair.mean(0)) / IMAGE
        return out.astype(np.asarray(value).dtype)

    def _fish_axis(self):
        """-> (the fish on the table for align_roll, or None without it; what this observation's column outline did).
        The fish (fish_on_table: its centre, axis and length) is kept from the last column outline that showed the whole
        fish: the trial's first outline sets the fish's length; a later one replaces it only when its length agrees with
        that within FISH_LENGTH_AGREE (the arm near the fish hides part of it or adds to the outline; a fish pushed by a
        missed grasp moves and turns and keeps its length). The kept centre is also the column fish the jaws are led to
        (_column_fish_point; 11/03 rolld trial 1: with the arm in the column picture the outline's centre jumped to the
        robot's own orange parts, 10-25 cm from the fish). (None, None) without a column camera."""
        if self.column_camera is None:
            return None, None
        targets = getattr(self.service, 'targets', None) or {}
        record = ((targets.get('targets') or {}).get('pickup') or {}).get('column') or {}
        try:
            fish = fish_on_table(self.column_camera, record)
        except Exception:
            fish = None
        if fish is None:
            seen = 'no_outline'
        elif self.fish_length_m is None:
            self.fish_length_m, seen = fish['length_m'], 'first'
        elif abs(fish['length_m'] - self.fish_length_m) <= FISH_LENGTH_AGREE * self.fish_length_m:
            seen = 'taken'
        else:
            seen = 'length_disagrees'
        if seen in ('first', 'taken'):
            self.fish_axis = dict(fish, measured_s=float(self.clock()))
        note = dict(outline=seen, length_mm=None if fish is None else fish['length_m'] * 1e3,
                    trial_length_mm=None if self.fish_length_m is None else self.fish_length_m * 1e3,
                    kept_from_s=None if self.fish_axis is None else self.fish_axis['measured_s'])
        return (self.fish_axis if self.planner.align_roll else None), note

    def _place_token(self, state, q):
        """Where the object goes, in the fish token's layout, or None when unseen: the state's own place token for a
        model that reads it (self.place_source == 'actor_state'), else the service's targets (_box_token)."""
        if self.place_source != 'actor_state':
            return self._box_token(q)
        token = getattr(state, 'place_relative', None)
        if token is None:
            return None
        value = token.detach().float().cpu().numpy().reshape(-1)
        return value if len(value) == 8 and np.isfinite(value).all() and value[0] > .5 else None

    def _box_token(self, q):
        """The box in the current wrist picture against the jaw midpoint, in the fish token's layout
        [available, score, dx, dy, length, width, cos 2theta, sin 2theta] (fcpg/v21_object_relative.py), or None.
        The drop zone record comes from the service's DINO targets of this observation (fcpg/v21_target_tracker.py
        matches it in the wrist with the column registration's role prototype, or with the wrist-view registration
        when the service has one); the jaw pair is the one the fish token is taken against."""
        targets = getattr(self.service, 'targets', None) or {}
        record = ((targets.get('targets') or {}).get('drop_zone') or {}).get('wrist') or {}
        encoder = getattr(getattr(self.service, 'owner', None), 'object_relative', None)
        if not record.get('visible') or record.get('view_source') != 'current_observation' or encoder is None:
            return None
        try:
            centre = np.asarray(record.get('cog_px') or record.get('centroid_px'), np.float64).reshape(-1)
            axis = np.asarray(record.get('axis_px'), np.float64).reshape(-1)
            pair = np.asarray(encoder.jaw_pair(q), np.float64).reshape(2, 2)
        except (TypeError, ValueError):
            return None
        middle = pair.mean(0)
        if centre.shape != (2,) or axis.shape != (2,) or not np.isfinite(np.r_[centre, middle, axis]).all():
            return None
        jaw = pair[1] - pair[0]
        theta = np.arctan2(axis[1], axis[0]) - np.arctan2(jaw[1], jaw[0])
        return np.array((1., float(record.get('score') or 0.), *(centre - middle) / IMAGE,
                         float(record.get('length_px') or 0.) / IMAGE, float(record.get('width_px') or 0.) / IMAGE,
                         np.cos(2 * theta), np.sin(2 * theta)))

    def _save_input(self, state, record):
        """What an offline replay of this block needs (the exact model state); never stops a block."""
        try:
            from .v21_live_service import cpu_actual_state
            stamp = time.time_ns()
            path = Path(self.service.output) / f'alignment-input-{stamp}.pt'
            with path.open('xb') as stream:
                torch.save(dict(kind='nonexecutable-alignment-input', schema=1, executable=False,
                                state=cpu_actual_state(state), source_time_s=float(state.time[0])), stream)
            record['input_snapshot'] = path.name
        except Exception as error:
            self.input_errors += 1
            record['input_snapshot_error'] = f'{type(error).__name__}: {error}'

    def builder(self, state, command_offsets):
        """Runtime command builder: -> [1, 5, 6]. Sets self.record."""
        rt = self.service.owner.runtime
        actor5 = rt.core.act(state, rt.text_ids, command_offsets).commands_position
        if getattr(rt, 'recomputing_at_dispatch', False):
            return self._at_dispatch(actor5, state)
        self.offset5, self.grip_hold, self.grip_override, self.planned_at_s = None, None, None, float(self.clock())
        fresh = not bool(state.history.valid.any())       # after HOME: no completed block of this run yet
        if fresh:
            self.planner.reset()
            self.let_go = False
            self.fish_axis, self.fish_length_m, self.grasp_s = None, None, None
            if self.latch is not None:
                self.latch.reset()
            if self.column_camera is not None:
                self.box_solid_fit = self._measure_box_solid()
                self.planner.box_solid = self.box_solid_fit.get('solid_object')
        record = dict(mode=self.mode, action_source=self.source, identity=self.identity,
                      source_time_s=float(state.time[0]), timeline_cleared=fresh, input_errors=self.input_errors)
        if self.column_camera is not None and fresh:
            record['box_solid_fit'] = {k: v for k, v in self.box_solid_fit.items() if k != 'solid_object'}

        def admissible(commands5):
            candidate = torch.as_tensor(np.asarray(commands5), dtype=actor5.dtype, device=actor5.device)[None]
            try:
                rt._physical(state.q, candidate)
                return True
            except Exception:
                return False
        q = state.q[0].detach().float().cpu().numpy()
        box_token = self._place_token(state, q)
        # section 17.148: the wrist's word on the box's sides (None: not seen, or no extent)
        sides = wrist_box_sides_px(box_token)
        beside = sides is not None and min(sides) < WRIST_BOX_SIDE_PX
        margin = self._box_solid_margin_mm(q)
        fish_axis, outline = self._fish_axis()
        selected, plan = self.planner.plan(
            q, actor5[0].detach().float().cpu().numpy(), self._jaw_layout(self._token(state), q),
            self._forecast(state, command_offsets), admissible, commit=self.mode == 'control', box_token=box_token,
            fish_point=self._column_fish_point(), jaw_height=self._jaw_heights(),
            box_from_wrist=bool(beside and margin is not None and margin > -NEAR_BOX_MM), fish_axis=fish_axis)
        record.update(plan, available=True, box_token=None if box_token is None else box_token.tolist())
        if outline is not None:
            record.update(fish_axis_outline=outline)
        reference = self._choose_wrist_reference(q)
        if reference is not None:
            record.update(wrist_reference_next=reference)
        if self.let_go and q[5] >= OPEN_GRIP and not plan.get('released_over_box'):
            # section 17.120 trial 24: the fish dropped straight below the open jaws still looks between them
            record.update(released_over_box=True, released_over_box_by='grip_latch')
        # 11/04 (sam31hb trial 1): the fish slipped out of the jaws over the box and lay in it; the trial searched on
        in_box = self._fish_in_box()
        if in_box is not None:
            record.update(fish_in_box_margin_mm=in_box)
            if (in_box >= 0. and q[5] >= OPEN_GRIP and not record.get('released_over_box')
                    and not (self.latch is not None and self.latch.holding)):
                record.update(released_over_box=True, released_over_box_by='column_fish_in_box')
        clearance = self._jaw_clearance_mm(q)
        over_box = self._over_box_solid(q)
        hold_back = (not over_box) if over_box is not None else clearance is not None and clearance > RELEASE_HIGH_MM
        hold_back = bool(hold_back or beside)          # section 17.148: the wrist shows the box to one side
        jaw, event = (None, None) if self.latch is None else self.latch.step(
            q[5], actor5[0, :, 5].detach().float().cpu().numpy(), plan['grasp_seen'], plan['box_seen'],
            hold_back=hold_back)
        if event == 'grasp':
            self.grasp_s = float(self.clock())
        if self._forget_grasp(event, clearance):
            record.update(fish_forgotten='dropped above the table (11/04)')
        if self.save_inputs:
            self._save_input(state, record)
        self.record = record
        if self.mode == 'shadow':
            if self.latch is not None:
                record['grip_latch'] = dict(event=event, holding=self.latch.holding, reading=float(q[5]),
                                            would_command=None if jaw is None else jaw.tolist(), used='recorded_only',
                                            jaw_clearance_mm=clearance, over_box=over_box, box_margin_mm=margin,
                                            wrist_box_sides_px=sides, wrist_beside=beside,
                                            held_back=self.latch.held_back)
            record['executed'] = 'actor'
            return actor5
        record['executed'] = 'alignment_cem'
        if self.latch is not None:
            used = 'actor'
            if jaw is not None:
                # the rule's jaw; refused by the physical gate (a squeeze still at the table moves the tip down):
                # the jaws stay at the read-back this block, never opening; refused again: the selection as it was
                tries = [('rule', jaw)] + ([('reading', np.full(5, float(q[5])))]
                                           if event in ('grasp', 'hold', 'release_held') else [])
                used = 'refused'
                for label, value in tries:
                    candidate = selected.copy()
                    candidate[:, 5] = value
                    if admissible(candidate):
                        selected, used = candidate, label
                        self.grip_override = candidate[:, 5].copy()
                        break
            record['grip_latch'] = dict(event=event, holding=self.latch.holding, reading=float(q[5]),
                                        actor_last=float(actor5[0, -1, 5]), command=None if jaw is None else jaw.tolist(),
                                        used=used, jaw_clearance_mm=clearance, over_box=over_box, box_margin_mm=margin,
                                        wrist_box_sides_px=sides, wrist_beside=beside,
                                        held_back=self.latch.held_back)
            self.let_go = event == 'release' or (self.let_go and event != 'grasp')
        chosen5 = torch.as_tensor(selected, dtype=actor5.dtype, device=actor5.device)[None]
        arm_only = torch.tensor([1.] * ARM + [0.], dtype=actor5.dtype, device=actor5.device)
        self.offset5 = (chosen5 - actor5) * arm_only          # the gripper is the Actor's, held (below) or the rule's
        self.grip_hold = float(selected[0, 5]) if plan['close_deferred'] and self.grip_override is None else None
        return chosen5
    builder.wants_command_offsets = True

    def _choose_wrist_reference(self, q):
        """11/03: where the wrist text finder takes its reference for the next exposure
        (fcpg/v21_text_finder.TextWristEncoder.reference_of): the column picture's (the object named, wherever it lies)
        while the jaws are open and further than NEAR_FISH_MM from the column fish; the wrist picture's own while they
        are nearer or shut, where the object fills the wrist picture (made from the far column picture the reference
        found it there in few cells or not at all: rolle trial 2 held the jaws 5-28 mm from the fish for 10 s without the
        wrist seeing it, and the rolld/rolle trials saw no grasp). -> 'column' / 'wrist', None when there is no choice."""
        encoder = getattr(getattr(self.service, 'owner', None), 'object_relative', None)
        if getattr(self, 'column_reference_of', None) is None or encoder is None or self.column_camera is None:
            return None
        near = bool(q[5] < OPEN_GRIP)
        point = self._column_fish_point()
        if not near and point is not None:
            try:
                jaws = np.asarray(self.column_camera.jaw(np.asarray(q, np.float64)[None]), np.float64)[0, :2]
                near = bool(np.linalg.norm(np.asarray(point, np.float64)[:2] - jaws) * 1000. <= NEAR_FISH_MM)
            except Exception:
                near = False
        encoder.reference_of = None if near else self.column_reference_of
        return 'wrist' if near else 'column'

    def _column_fish_point(self):
        """The fish on the table (base frame, m; section 17.118 d): the centre of the whole fish kept by _fish_axis
        (11/03), None without a column camera or before an outline of the trial showed a whole fish. (Until 11/03 the
        learned column finder's record of every observation, placed at the fish's half height: with no fish on the table
        it outlined the robot's own parts behind its base, and the column led the arm out to them until the table gate
        stopped it, rollh trial 4.)"""
        if self.column_camera is None or getattr(self, 'fish_axis', None) is None:
            return None
        return np.asarray(self.fish_axis['point'], np.float64)

    def _measure_box_solid(self):
        """The box solid from this exposure's column picture (section 17.118): the learned box finder's column map of
        the observation just made (LearnedBoxFinder.last_prob) fitted by fit_box_solid; at a trial's first block the arm
        is at HOME and hides none of the box. Never stops a block: a failure leaves the wrist alignment as before."""
        started = self.clock()
        finder = (getattr(self.service, 'object_finder', None) or {}).get('box')
        prob = (getattr(finder, 'last_prob', None) or {}).get('column')
        if prob is None:
            return dict(solid=None, reason='no column box map of this exposure', seconds=float(self.clock() - started))
        try:
            solid = fit_box_solid(self.column_camera, prob)
        except Exception as error:
            return dict(solid=None, reason=f'{type(error).__name__}: {error}', seconds=float(self.clock() - started))
        return dict(solid=None if solid is None else solid.describe(), solid_object=solid,
                    reason=None if solid is not None else 'the box finder found no box in the column picture',
                    seconds=float(self.clock() - started))

    def _over_box_solid(self, q):
        """Whether the jaws are over the footprint of the box solid measured at the trial's first block (forward
        kinematics in the column camera's base frame, as JawTable), None when no solid was measured (section 17.120)."""
        margin = self._box_solid_margin_mm(q)
        return None if margin is None else bool(margin >= 0.)

    def _box_solid_margin_mm(self, q):
        """Section 17.148: how far the jaws are inside the measured box solid's footprint (mm; the nearer of its
        two sides, negative outside), None when no solid was measured or the jaws cannot be placed."""
        solid, camera = getattr(self.planner, 'box_solid', None), getattr(self.planner, 'camera', None)
        if solid is None or camera is None:
            return None
        try:
            jaw = np.asarray(camera.jaw(np.asarray(q, np.float64).reshape(1, 6)), np.float64).reshape(-1)[:2]
        except Exception:
            return None
        if not np.isfinite(jaw).all():
            return None
        return footprint_margin_mm(solid, jaw)

    def _jaw_heights(self):
        """Section 17.140: joint states [n, 6] -> the jaws' heights above the table (mm) by the service's own
        clearance gate (the jaw clearance the records show), or None when the service has none."""
        gate = getattr(getattr(self.service, 'owner', None), 'clearance', None)
        if gate is None:
            return None
        return lambda states: np.asarray([gate.clearance_mm(s) for s in np.asarray(states, np.float64).reshape(-1, 6)])

    def _forget_grasp(self, event, clearance=None):
        """Section 17.140 (17.138 p16-1, p16-5, p16-6): when the grip latch finds nothing held ('empty') the planner
        no longer counts the grasp: it aims the reopened jaws at the fish again instead of taking them to the box, and
        a later opening near the box is no released_over_box. 11/04: with the jaws more than DROPPED_ABOVE_MM above the
        table (clearance, mm) the fish fell somewhere else: the kept whole fish is forgotten too (-> True)."""
        if event == 'empty' and self.planner is not None:
            self.planner.grasped = False
        if event == 'empty' and clearance is not None and clearance > DROPPED_ABOVE_MM:
            self.fish_axis, self.fish_length_m = None, None
            return True
        return False

    def _fish_in_box(self):
        """11/04: how far the whole fish an outline showed after the trial's last grasp lies inside the measured box
        solid's footprint (mm, negative outside), None without a box solid, before a grasp or without such an outline."""
        solid, kept = getattr(self.planner, 'box_solid', None), getattr(self, 'fish_axis', None)
        if solid is None or kept is None or getattr(self, 'grasp_s', None) is None or not kept['measured_s'] > self.grasp_s:
            return None
        return footprint_margin_mm(solid, kept['centre'])

    def _jaw_clearance_mm(self, q):
        """The jaws' height above the table (mm) by the service's own clearance gate, or None when it has none."""
        gate = getattr(getattr(self.service, 'owner', None), 'clearance', None)
        if gate is None:
            return None
        try:
            value = float(gate.clearance_mm(np.asarray(q, np.float64)))
        except Exception:
            return None
        return value if np.isfinite(value) else None

    def _at_dispatch(self, actor5, state):
        """The pending block re-derived at the pose measured just before the first write; no second search. A block
        the physical gate refuses there falls back to the Actor's own block, else to the arm held at the measured
        pose, with the same jaws (10/26 robot trial 11, section 17.109: the selection, admitted at the planning pose,
        crossed the table floor once re-based at the measured one and the trial stopped on the gate)."""
        record = self.record if isinstance(self.record, dict) else {}
        note = record.setdefault('dispatch', dict(recomputed=0))
        note['recomputed'] += 1
        if self.mode == 'shadow':
            note['rule'] = 'actor_at_measured_pose'
            return actor5
        age = None if self.planned_at_s is None else float(self.clock()) - self.planned_at_s
        if self.offset5 is None or age is None or not 0. <= age <= DISPATCH_MAX_AGE_S:
            raise RuntimeError(f'alignment CEM selection cannot be re-based at dispatch (plan age {age})')
        note.update(rule='actor_at_measured_pose_plus_planned_offset', plan_age_s=age)
        out = self._jaws(actor5 + self.offset5.to(actor5.device, actor5.dtype))
        rt = self.service.owner.runtime

        def admissible(commands):
            try:
                rt._physical(state.q, commands)
                return True
            except Exception:
                return False
        if admissible(out):
            return out
        held = actor5.clone()
        held[..., :ARM] = state.q[:, None, :ARM].to(actor5.device, actor5.dtype)
        for label, candidate in (('actor', self._jaws(actor5.clone())), ('hold', self._jaws(held))):
            if admissible(candidate):
                note.update(refused=True, fallback=label)
                return candidate
        note.update(refused=True, fallback='none')
        return out

    def _jaws(self, block):
        """The block with this plan's jaws: the gripper rule's five commands (section 17.45), the deferred close's
        hold, else its own."""
        if self.grip_override is not None:
            jaw = torch.as_tensor(self.grip_override, dtype=block.dtype, device=block.device).reshape(1, 5, 1)
            return torch.cat((block[..., :ARM], jaw), -1)
        if self.grip_hold is not None:
            return torch.cat((block[..., :ARM], torch.full_like(block[..., ARM:], self.grip_hold)), -1)
        return block
