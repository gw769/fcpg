"""The Actor's wrist object cue from an outline (training overhaul doc section 17.30): no colour anywhere.

The same eight numbers as fcpg/v21_object_relative.py, so the Actor's existing object token reads them and the mirror
(fcpg/v21_mirror.py: dx and sin 2theta change sign) applies unchanged:

    [available, score, dx, dy, length, width, cos 2theta, sin 2theta]

computed from a binary mask of the thing to grasp in the wrist picture:
- dx, dy: the mask's centre minus a reference point, in units of the picture's long side (= letterbox224 pixels / 224,
  the object token's unit). The reference is where the jaws hold the thing: in the Archive labels (hindsight) the mean
  centre of the same thing's outline over its held pictures, so (0, 0) means "where it will sit in the jaws".
- length, width: the extents along and across the mask's main axis (sqrt(12 x variance), the sides of a rectangle with
  the mask's second moments), same unit.
- theta: the main axis against the picture's x axis (the jaws of an SO-101 wrist camera open along it), modulo pi.
- score: 1 for an accepted outline.
Pure functions of the mask and the numbers given; no learned state, no robot I/O.
"""
import numpy as np

DIM = 8
MIN_PIXELS = 4


def mask_geometry(mask):
    """(centre_x, centre_y, length, width, theta) of a 2-D boolean mask in its own pixel units (pixel centres at +0.5),
    or None for fewer than MIN_PIXELS pixels."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2:
        raise ValueError('a 2-D mask is required')
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_PIXELS:
        return None
    x, y = xs + .5, ys + .5
    cx, cy = float(x.mean()), float(y.mean())
    cov = np.cov(np.stack((x - cx, y - cy)), bias=True)
    values, vectors = np.linalg.eigh(cov)                         # ascending
    major = vectors[:, 1]
    length = float(np.sqrt(12. * max(values[1], 0.)))
    width = float(np.sqrt(12. * max(values[0], 0.)))
    theta = float(np.arctan2(major[1], major[0]))
    return cx, cy, length, width, theta


def picture_geometry(mask, picture_hw):
    """mask_geometry in the pixels of the picture the mask was cut from ([h, w]); the mask may be smaller (a quarter-size
    outline) with the same aspect."""
    found = mask_geometry(mask)
    if found is None:
        return None
    h, w = (float(v) for v in picture_hw)
    sy, sx = h / mask.shape[0], w / mask.shape[1]
    if abs(sx - sy) > .02 * max(sx, sy):
        raise ValueError('mask and picture differ in aspect')
    cx, cy, length, width, theta = found
    # the extents scale with the (equal) pixel size; the axis keeps its angle when sx == sy
    return cx * sx, cy * sy, length * sx, width * sx, theta


def cue_vector(geometry, reference_xy, picture_hw, score=1.):
    """The eight numbers for one picture's geometry (picture_geometry) against the reference point (picture pixels)."""
    out = np.zeros(DIM, np.float32)
    if geometry is None or reference_xy is None:
        return out
    cx, cy, length, width, theta = geometry
    long_side = float(max(picture_hw))
    rx, ry = (float(v) for v in reference_xy)
    if not np.isfinite([cx, cy, length, width, theta, rx, ry, long_side]).all() or long_side <= 0:
        return out
    out[:] = (1., float(np.clip(score, 0., 1.)), (cx - rx) / long_side, (cy - ry) / long_side,
              length / long_side, width / long_side, np.cos(2 * theta), np.sin(2 * theta))
    return out


def held_reference(geometries):
    """The reference point: the mean centre of the held pictures' geometries (None entries skipped), or None."""
    centres = [(g[0], g[1]) for g in geometries if g is not None]
    if not centres:
        return None
    return tuple(np.mean(np.asarray(centres, np.float64), axis=0))


SCHEMA = 'v21-outline-cue-v1'


def read_manifest(directory):
    """The manifest of a scripts/build_v21_outline_cue_tables.py directory, checked for its schema."""
    import json
    from pathlib import Path
    manifest = json.loads((Path(directory) / 'object_cue.manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('complete') is not True or manifest.get('colour_used') is not False:
        raise ValueError('not a complete colour-free outline cue directory')
    return manifest


def attach_archive_cues(source, layout, directory):
    """Give a V21Sources the Archive outline cues, after its camera layout is installed (the layout refuses Archive
    cues it cannot remap). The cues outline the camera this same layout calls the wrist, so they stay valid when the
    layout reorders an episode's views; any other layout is refused. Returns the manifest."""
    from pathlib import Path
    manifest = read_manifest(directory)
    if getattr(source, 'archive_camera_roles', None) is not layout:
        raise ValueError('install the camera layout first')
    if manifest['camera_layout_sha256'] != layout.identity['table_sha256']:
        raise ValueError('the outline cues were built on another camera layout')
    table = np.load(Path(directory) / 'archive_object_cue.npy')
    if table.shape != (source.archive.endpoint_count, DIM) or not np.isfinite(table).all():
        raise ValueError('the Archive cue table must be finite [Archive endpoints, 8]')
    source.archive_object_relative_table = np.asarray(table, np.float32)
    source.archive_object_cues_follow_layout = True
    place = Path(directory) / 'archive_place_cue.npy'
    if place.exists():
        # section 17.117: where the held thing is put, on the same camera and against the same reference
        table = np.load(place)
        if table.shape != (source.archive.endpoint_count, DIM) or not np.isfinite(table).all():
            raise ValueError('the Archive place cue table must be finite [Archive endpoints, 8]')
        source.archive_place_relative_table = np.asarray(table, np.float32)
    return manifest
