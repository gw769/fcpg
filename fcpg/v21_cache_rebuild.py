"""Bounded CPU decode jobs shared by the cache preflight and full rebuild."""
from pathlib import Path
import time

import numpy as np

from .v21_recorded_clock import (TIME_EPS, read_recording, resolve_media,
                                decode_at_recorded_times, decode_at_recorded_times_skipping)
from .v21_camera_quality import (CAMERA_QUALITY_VERSION, REVIEWED_CAMERA_QUALITY_VERSION,
                                 raw_camera_quality, reviewed_exposure_mask)

# by name, looked up when a job runs (tests replace decode_at_recorded_times on this module)
DECODERS = {'in-order': 'decode_at_recorded_times', 'keyframe-skip': 'decode_at_recorded_times_skipping'}


def episode_job(job):
    started = time.monotonic()
    quality_version = job.get('camera_quality_version')
    if quality_version not in (None, CAMERA_QUALITY_VERSION, REVIEWED_CAMERA_QUALITY_VERSION):
        raise ValueError('unknown camera quality recipe')
    review = job.get('camera_quality_review')
    if quality_version == REVIEWED_CAMERA_QUALITY_VERSION and review is None:
        raise ValueError('reviewed camera quality requires its exposure manifest')
    row, views, local = job['row'], job['views'], np.asarray(job['local'], np.int64)
    decode = globals()[DECODERS[job.get('decode', 'in-order')]]   # the same pictures either way (tests/test_v21_skip_decode.py)
    record = read_recording(row, views)
    if (local < 0).any() or (local >= len(record['time'])).any() or (np.diff(local) <= 0).any():
        raise ValueError('episode endpoint indices must be sorted unique')
    rgb = np.zeros((len(local), 2, 224, 224, 3), np.uint8)
    valid = np.zeros((len(local), 2), bool)
    capture = np.zeros((len(local), 2), np.float64)
    capture_q = np.zeros((len(local), 2, 6), np.float32)
    video_pts = np.full((len(local), 2), np.nan)
    reasons = np.zeros((len(local), 2), np.uint8)
    detail = []
    raw = {} if job.get('retain_raw', False) else None
    quality_details = {} if quality_version is not None else None
    for c, view in enumerate(views):
        media = resolve_media(row, view, record['time'])
        extra = {}
        quality_ok = np.zeros(len(local), bool)
        if raw is not None:
            raw[view] = {}
        if quality_details is not None:
            quality_details[view] = {}
        if raw is not None or quality_version is not None:
            def on_raw(i, frame):
                if raw is not None:
                    raw[view][i] = frame
                if quality_details is not None:
                    result = raw_camera_quality(frame)
                    quality_details[view][i] = result
                    quality_ok[i] = result['valid']
            extra['raw_rgb_callback'] = on_raw
        frames, pts, found, info = decode(
            media['path'], media['requested_pts'][local], media['start'], media['stop'], **extra)
        actual = pts + media['clock_origin']
        # Equal within float32 serialization tolerance means the same sample;
        # retain the original decoded PTS separately for reproducibility.
        actual = np.minimum(actual, record['time'][local])
        state_ids = np.searchsorted(record['time'], np.nan_to_num(actual, nan=-np.inf)+TIME_EPS, side='right')-1
        safe = np.maximum(state_ids, 0)
        mask = record['masks'][local, c] & record['masks'][safe, c]
        placeholder = (frames.max(axis=(1, 2, 3)) <= 2) | (np.ptp(frames, axis=(1, 2, 3)) <= 2)
        admitted = found & (state_ids >= 0) & mask & ~placeholder
        reasons[:, c] = np.select([~found, state_ids < 0, ~mask, placeholder], [1, 2, 3, 4], default=0)
        if 'decode_error' in info:          # 7: left unserved by a corrupt packet part-way (fcpg/v21_recorded_clock.py)
            reasons[(np.arange(len(local)) >= info['first_unserved']) & ~found, c] = 7
        if quality_version is not None:
            reasons[admitted & ~quality_ok, c] = 5
            admitted &= quality_ok
        if quality_version == REVIEWED_CAMERA_QUALITY_VERSION:
            reviewed_bad = reviewed_exposure_mask(review, job, view)
            reasons[admitted & reviewed_bad, c] = 6
            admitted &= ~reviewed_bad
        valid[:, c] = admitted
        rgb[admitted, c] = frames[admitted]
        # Missing slots keep finite timestamps/q and are explicitly masked.
        capture[:, c] = record['time'][local]
        capture[admitted, c] = actual[admitted]
        capture_q[:, c] = record['q'][local]
        capture_q[admitted, c] = record['q'][state_ids[admitted]]
        video_pts[:, c] = pts
        info.update(view=view, clock_origin=media['clock_origin'], interval=[media['start'], media['stop']],
                    provenance=media['provenance'], declared_video_fps=media['declared_video_fps'],
                    mask_present=int(record['mask_present'][local, c].sum()),
                    valid=int(admitted.sum()), reasons={str(r): int((reasons[:, c] == r).sum())
                        for r in [*range(7 if review is not None else (6 if quality_version is not None else 5)),
                                  *([7] if 'decode_error' in info else [])]})
        if quality_version is not None:
            info['camera_quality_version'] = quality_version
        detail.append(info)
    return dict(identity=job['identity'], endpoint_ids=np.asarray(job['endpoint_ids'], np.int64),
                local=local, rgb=rgb, valid=valid, capture_times=capture, capture_q=capture_q,
                video_pts=video_pts, reasons=reasons, times=record['time'], q=record['q'],
                actions=record['action'], details=detail, raw_rgb=raw, raw_quality=quality_details,
                elapsed_s=time.monotonic()-started)


def bounded_results(pool, jobs, depth):
    """Do not enqueue hundreds of decoded episodes while the GPU is busy."""
    from collections import deque
    stream, pending = iter(jobs), deque()
    for _ in range(depth):
        job = next(stream, None)
        if job is not None:
            pending.append(pool.submit(episode_job, job))
    while pending:
        result = pending.popleft().result(timeout=600)
        job = next(stream, None)
        if job is not None:
            pending.append(pool.submit(episode_job, job))
        yield result


def fish_jobs(raw_root, fish):
    """Use raw v3 videos; episode 2 payloads are never requested/decoded."""
    import pyarrow.parquet as pq
    raw_root = Path(raw_root)
    # Read only identity metadata for the shared parquet file, then slice the
    # admitted episodes in read_recording (no episode-2 q/action reduction).
    segments = {e: [] for e in (0, 1, 3)}
    for path in sorted(raw_root.glob('data/chunk-*/*.parquet')):
        ids = np.asarray(pq.read_table(path, columns=['episode_index'])['episode_index'])
        for ep in segments:
            selected = np.flatnonzero(ids == ep)
            if len(selected):
                if not (np.diff(selected) == 1).all():
                    raise ValueError('noncontiguous fish episode in parquet')
                segments[ep].append(dict(data_path=str(path.relative_to(raw_root)),
                    row_start=int(selected[0]), row_stop=int(selected[-1])+1))
    for ep in (0, 1, 3):
        ids = np.flatnonzero(fish.episodes == ep)
        fish.validate_frames(ids)
        if sum(s['row_stop']-s['row_start'] for s in segments[ep]) != len(ids):
            raise ValueError('raw fish frame count differs from frozen motion identities')
        yield dict(identity=ep, row=dict(dataset_root=str(raw_root), episode_index=ep,
            length=len(ids), segments=segments[ep]), views=['observation.images.wrist', 'observation.images.column'],
            local=np.asarray(fish.frame_indices[ids]).tolist(), endpoint_ids=ids.tolist())
