"""Versioned, raw-content placeholder checks for offline camera supervision.

This detects the observed decoder neutral-fill failure, not jaw visibility or
task difficulty. Apply before letterboxing: black padding hides uniform content
from a whole-canvas range test. No threshold here changes robot execution.
"""
import numpy as np

CAMERA_QUALITY_VERSION = 'raw-neutral-fill-v1'
REVIEWED_CAMERA_QUALITY_VERSION = 'raw-neutral-fill+reviewed-exposures-v2'
REVIEW_SCHEMA = 'v21-reviewed-camera-exclusions-v1'


def camera_quality_job(policy):
    """Carry the same explicit review into every raw-image consumer."""
    policy = policy or {}
    result = dict(camera_quality_version=policy.get('fish'))
    if policy.get('fish') == REVIEWED_CAMERA_QUALITY_VERSION:
        result['camera_quality_review'] = policy['fish_review']
    return result


def reviewed_exposure_mask(review, job, view):
    """Only withdraw reviewed identities, never classify images by colour."""
    from pathlib import Path
    if (not isinstance(review, dict) or review.get('schema') != REVIEW_SCHEMA
            or Path(review['dataset_root']).resolve() != Path(job['row']['dataset_root']).resolve()):
        raise ValueError('same-dataset reviewed exposure manifest required')
    ids = np.asarray(job['endpoint_ids'])
    local = np.asarray(job['local'])
    if ids.shape != local.shape:
        raise ValueError('review endpoint/local identities differ')
    bad = np.zeros(len(ids), bool)
    seen = set()
    for row in review['exposures']:
        key = (row['episode'], row['endpoint'], row['view'])
        if (row['episode'] not in (0, 1, 3) or row['view'] not in ('wrist', 'column')
                or key in seen or not row.get('reason')):
            raise ValueError('unique admitted reviewed camera identities required')
        seen.add(key)
        if row['episode'] != job['identity'] or row['view'] != view.rsplit('.', 1)[-1]:
            continue
        selected = ids == row['endpoint']
        if selected.any() and not np.all(local[selected] == row['local_frame']):
            raise ValueError('reviewed exposure local frame differs')
        bad |= selected
    return bad


def raw_camera_quality(rgb):
    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8 or not frame.size:
        raise ValueError('camera quality requires nonempty raw uint8 RGB')
    # Exact full-width neutral rows are the observed codec fill pattern. A
    # gray object, a small missing band, or black letterbox padding is not it.
    neutral_rows = np.all(frame == 128, axis=(1, 2))
    edges = np.flatnonzero(np.diff(np.r_[False, neutral_rows, False]))
    longest = int(np.max(edges[1::2]-edges[::2], initial=0))
    uniform = int(np.ptp(frame)) <= 2
    filled = longest >= (len(frame)+1)//2
    return dict(version=CAMERA_QUALITY_VERSION, valid=not (uniform or filled),
        reason='uniform-raw-content' if uniform else ('neutral-fill-row-band' if filled else None),
        raw_shape=list(frame.shape), raw_range=int(np.ptp(frame)),
        neutral_full_width_rows=int(neutral_rows.sum()), longest_neutral_run_rows=longest)
