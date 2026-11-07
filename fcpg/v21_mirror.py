"""Left-right mirror of recorded rows: manufactured recordings with the scene on the other side
(docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md, section 9).

The SO-101 is symmetric about the vertical plane through its base and forward axis. Mirroring a recording across
that plane gives another physically valid recording:

    shoulder pan     -> -pan
    wrist roll       -> roll_centre - roll   (roll_centre = twice the roll at which the wrist camera is level)
    lift, elbow, wrist flex, jaw: unchanged
    wrist picture    -> flipped left-right
    wrist object cue -> dx and the sine of the doubled axis angle change sign (the place cue alike, section 17)
    place goal       -> the turn about the pan axis changes sign (fcpg/v21_place_goal.py, section 17.7)

The wrist picture of the mirrored arm is the flipped picture because the camera sits between the two jaws, which
look alike from the camera (checked on the fish robot's live wrist pictures). An external camera's mirrored
picture would show the robot on the other side of the scene, so this is meant for a model that reads the wrist
camera only (V21Config.wrist_only). The frozen DINO features are flipped on their 16 x 16 grid instead of encoding
the flipped picture again (per-patch cosine 0.87-0.89 against the re-encoded flipped picture, 0.56-0.58 for the
unflipped grid; analysis/dino_flip_check.py). A mirrored row stays recognisable (section 17.53): a small probe tells a
flipped grid from a recorded one every time, and a picture flipped before DINO 92-100% of the time for our wrist camera
(also with the jaws hidden) and every time for the column camera, so the mirror adds recordings with the scene on the
other side but cannot by itself make a model symmetric. A wrong roll centre shifts the mirrored roll by a constant, like a
different roll calibration. Rows whose mirrored pan or roll would leave the calibrated range stay as recorded.
No robot I/O, no learned state; mirroring twice returns the original values.
"""
from dataclasses import replace

import torch

from .v21_checks import deferring
from .v21_compiled import region
from .v21_encoder import DIM, GRID, PATCHES
from .v21_place_goal import GOAL_SIGN

LIMIT = 100.
CUE_SIGN = (1., 1., -1., 1., 1., 1., 1., -1.)   # [available, score, dx, dy, length, width, cos 2theta, sin 2theta]


_SIGNS = {}


def _signs(values, like):
    """``like.new_tensor(values)``, made once per device and dtype (a new tensor from Python numbers is a copy to the
    GPU that waits for it, section 17.81)."""
    key = (tuple(values), like.device, like.dtype)
    if key not in _SIGNS:
        _SIGNS[key] = like.new_tensor(values)
    return _SIGNS[key]


def _per_row(value, like):
    return value.to(like.device, like.dtype).reshape(len(like), *([1] * (like.ndim - 1)))


def mirrored(values, roll_centre):
    """[B,...,6] joint positions in the mirror; roll_centre [B]."""
    if values.shape[-1] != 6 or roll_centre.shape != (len(values),):
        raise ValueError('joint values [B,...,6] and one roll centre per row are required')
    out = values.clone()
    out[..., 0] = -values[..., 0]
    out[..., 4] = _per_row(roll_centre, values[..., 4]) - values[..., 4]
    return out


def fits(values, roll_centre, valid=None):
    """[B]: every (valid) mirrored pan and roll stays inside the calibrated range."""
    moved = mirrored(values, roll_centre)
    moved = torch.stack((moved[..., 0], moved[..., 4]), -1)    # pan and roll (a tuple index is a copy that waits, 17.82)
    inside = (moved.abs() <= LIMIT).all(-1)
    if valid is not None:
        if valid.shape != values.shape[:-1] or valid.dtype != torch.bool:
            raise ValueError('validity must match the joint values')
        inside = inside | ~valid
    return inside.reshape(len(values), -1).all(-1)


def _apply(values, rows, roll_centre, valid=None):
    select = rows.to(values.device).reshape(len(values), *([1] * (values.ndim - 2)))
    if valid is not None:
        select = select & valid
    return torch.where(select[..., None], mirrored(values, roll_centre), values)


def flipped_features(features, rows):
    """[B,T,2,PATCHES,DIM] frozen DINO patch tokens, row-major GRID x GRID: the selected rows' pictures flipped left-right."""
    if features.ndim != 5 or features.shape[2:] != (2, PATCHES, DIM):
        raise ValueError(f'features must be [B,T,2,{PATCHES},{DIM}]')
    return _flipped(features, rows.to(features.device))


@region         # section 17.84: one compiled program in a training update (it only moves values)
def _flipped(features, rows):
    b, t = features.shape[:2]
    flip = features.reshape(b, t, 2, GRID, GRID, DIM).flip(4).reshape(features.shape)
    return torch.where(rows.reshape(b, 1, 1, 1, 1), flip, features)


def mirror_rows(chain, targets, rows, roll_centre):
    """Mirror the selected rows of a TrainingChain together with their HorizonTargets.

    Poses, recorded paths and commands, capture poses of valid exposures, pictures and the wrist object cue
    change together; padding and invalid exposures stay as they are. Returns (chain, targets, mirrored [B])."""
    if chain.camera_types is not None:
        raise ValueError('typed camera streams are not mirrored')
    if rows.shape != (len(chain.q),) or rows.dtype != torch.bool:
        raise ValueError('mirror rows must be bool [B]')
    groups = [(chain.q, None), (chain.commands, None), (chain.paths, None),
              (targets.commands, targets.valid), (targets.readbacks, targets.valid),
              (targets.trace.positions, targets.trace.valid), *[(t.positions, t.valid) for t in chain.traces]]
    if chain.capture_q is not None:
        groups.append((chain.capture_q, chain.view_valid.to(chain.capture_q.device)))
    rows = rows.to(chain.q.device).clone()
    for values, valid in groups:
        rows &= fits(values, roll_centre, valid)
    if not deferring() and not rows.any():      # a shortcut only: mirroring no row changes nothing (section 17.81)
        return chain, targets, rows
    joints = lambda values, valid=None: _apply(values, rows, roll_centre, valid)
    trace = lambda t: replace(t, positions=joints(t.positions, t.valid))
    cue = chain.object_relative
    if cue is not None:
        cue = torch.where(rows.reshape(-1, *([1] * (cue.ndim - 1))), cue * _signs(CUE_SIGN, cue), cue)
    place = getattr(chain, 'place_relative', None)
    if place is not None:
        place = torch.where(rows.reshape(-1, *([1] * (place.ndim - 1))), place * _signs(CUE_SIGN, place), place)
    goal = getattr(chain, 'place_goal', None)
    if goal is not None:
        goal = torch.where(rows.reshape(-1, *([1] * (goal.ndim - 1))), goal * _signs(GOAL_SIGN, goal), goal)
    chain = replace(chain, q=joints(chain.q), commands=joints(chain.commands), paths=joints(chain.paths),
                    traces=tuple(trace(t) for t in chain.traces),
                    features=None if chain.features is None else flipped_features(chain.features, rows),
                    object_relative=cue, **({} if place is None else dict(place_relative=place)),
                    **({} if goal is None else dict(place_goal=goal)),
                    capture_q=None if chain.capture_q is None else joints(chain.capture_q, groups[-1][1]))
    targets = replace(targets, trace=trace(targets.trace), commands=joints(targets.commands, targets.valid),
                      readbacks=joints(targets.readbacks, targets.valid))
    return chain, targets, rows
