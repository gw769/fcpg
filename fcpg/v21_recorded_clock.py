"""Recorded-time/PTS mapping for the explicit v21 DINO cache migration.

No torch, robot I/O, nominal video-fps conversion, or automatic view swaps.
The legacy caches/readers remain readable only through their old configuration.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .lerobot_archive import (_episode_video_metadata, _format_indices,
                             _read_parquet_slice, letterbox_rgb)
from .so101_archive_units import canonicalize_joint_values, row_joint_transform

CLOCK_VERSION = 'recorded-time-video-pts-v1'
EXTRACTOR_VERSION = 'dinov2-s14-rgb-letterbox224-imagenet-fp16-v1'
TIME_EPS = 1e-4  # serialized float32 clocks; not a camera-latency tolerance


@lru_cache(maxsize=256)
def dataset_info(root):
    return json.loads((Path(root)/'meta/info.json').read_text())


def read_recording(row, views):
    """Read ONLY the requested episode, including original camera masks."""
    import pyarrow.parquet as pq
    pieces, masks, has_masks = [], [], []
    for segment in row['segments']:
        path = Path(row['dataset_root'])/segment['data_path']
        names = pq.ParquetFile(path).schema_arrow.names
        mask_names = [view + '_mask' for view in views]
        columns = ['episode_index', 'timestamp', 'observation.state', 'action']
        table = _read_parquet_slice(path, int(segment['row_start']), int(segment['row_stop']),
                                    columns + [m for m in mask_names if m in names])
        if any(e != int(row['episode_index']) for e in table['episode_index'].to_pylist()):
            raise ValueError('recording segment crosses episode')
        pieces.append({key: np.asarray(table[key].to_pylist()) for key in columns[1:]})
        present = np.asarray([m in names for m in mask_names], dtype=bool)
        values = []
        for name, exists in zip(mask_names, present):
            value = np.asarray(table[name].to_pylist()) if exists else np.ones(len(table), bool)
            if value.shape != (len(table),) or not np.isin(value, [0, 1]).all():
                raise ValueError(f'non-boolean raw camera mask: {path}:{name}')
            values.append(value.astype(bool))
        masks.append(np.stack(values, -1))
        has_masks.append(np.broadcast_to(present, (len(table), len(views))))
    times = np.concatenate([p['timestamp'] for p in pieces]).astype(np.float64)
    if len(times) != int(row['length']) or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('real recording timestamps must be finite and strictly increasing')
    result = dict(time=times, masks=np.concatenate(masks), mask_present=np.concatenate(has_masks))
    for key, column in [('q', 'observation.state'), ('action', 'action')]:
        values = np.concatenate([p[column] for p in pieces])
        order, scale, offset = row_joint_transform(row, 'state' if key == 'q' else 'action')
        values = canonicalize_joint_values(values, order=order, scale=scale, offset=offset).astype(np.float32)
        if values.shape != (len(times), 6) or not np.isfinite(values).all():
            raise ValueError('invalid canonical six-joint recording')
        result[key] = values
    return result


def resolve_media(row, view, record_times):
    """Resolve an episode's actual video interval, not a parquet file number.

V3's from_timestamp anchors episode-relative time in a video shard. Some
recorders keep a global data clock; subtracting the documented episode start
there is explicit clock conversion, never rebasing a video's first PTS.
V2 has an episode-local video clock; a nonzero first PTS is preserved.
"""
    root = Path(row['dataset_root'])
    info = dataset_info(str(root))
    pattern = info['video_path']
    values = _format_indices(row['segments'][0]['data_path'], int(row['episode_index']))
    values['video_key'] = view
    times = np.asarray(record_times, dtype=np.float64)
    if '{episode_index' in pattern:
        origin, start = 0., 0.
        stop = float(times[-1] + np.median(np.diff(times)))
        provenance = 'per-episode-recorded-clock'
    else:
        metadata = _episode_video_metadata(str(root)).get(int(row['episode_index']), {})
        prefix = f'videos/{view}/'
        required = ['chunk_index', 'file_index', 'from_timestamp', 'to_timestamp']
        if any(prefix + key not in metadata for key in required):
            raise ValueError(f'missing authoritative video interval: {root}, {row["episode_index"]}, {view}')
        values['chunk_index'] = values['episode_chunk'] = int(metadata[prefix+'chunk_index'])
        values['file_index'] = int(metadata[prefix+'file_index'])
        start, stop = float(metadata[prefix+'from_timestamp']), float(metadata[prefix+'to_timestamp'])
        if not np.isfinite([start, stop]).all() or stop <= start:
            raise ValueError('invalid episode video interval')
        origin = float(times[0] - start)
        provenance = 'episode-metadata-from-to-timestamp'
    return dict(path=str(root/pattern.format(**values)), start=start, stop=stop,
                requested_pts=times-origin, clock_origin=origin, provenance=provenance,
                declared_video_fps=info.get('features', {}).get(view, {}).get('info', {}).get('video.fps'))


def decode_at_recorded_times(path, requested_pts, start, stop, *, raw_rgb_callback=None):
    """Causal latest-frame lookup by decoded PTS, including VFR/nonzero PTS.

Only a frame within one actual video period is admissible. Decode failure is
an error (not silently masked); missing files are explicit unavailable views.
No first-frame substitution or episode-tail repetition is performed.
One exception, explicit (17.97 pass 2): a corrupt packet part-way through a
file (InvalidDataError once frames have decoded; the hand video of
NeilKim/so100-teleop-record-7 episode 0, the hub's own bytes, fails at 22.4 of
23.2 s) ends the decoding of that view. The requests not served by then stay
unavailable (never the last good frame) and the detail names the error, the
last good PTS and the first unserved request; a file that decodes no frame
still raises. On such a file this decoder and the skipping one can differ.
"""
    wanted = np.asarray(requested_pts, dtype=np.float64)
    if wanted.ndim != 1 or not len(wanted) or not np.isfinite(wanted).all() or (np.diff(wanted) < 0).any():
        raise ValueError('video requests must be finite and sorted')
    rgb = np.zeros((len(wanted), 224, 224, 3), np.uint8)
    selected = np.full(len(wanted), np.nan)
    valid = np.zeros(len(wanted), bool)
    if not Path(path).is_file():
        return rgb, selected, valid, dict(reason='missing-video', path=str(path))
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        if rate is None or not 1 <= float(rate) <= 240:
            raise ValueError(f'cannot establish actual video period: {path}')
        period = 1./float(rate)
        target = max(start, float(wanted[0]) - 2*period)
        container.seek(max(0, int(np.floor(target/float(stream.time_base)))), stream=stream,
                       backward=True, any_frame=False)
        previous = None
        previous_pts = None
        i, decoded = 0, 0

        def emit(index):
            if (previous is not None and previous_pts >= start-TIME_EPS and previous_pts < stop
                    and wanted[index] >= start-TIME_EPS and wanted[index] < stop
                    and -TIME_EPS <= wanted[index]-previous_pts <= period*1.01+TIME_EPS):
                raw_bgr = previous.to_ndarray(format='bgr24')
                rgb[index] = letterbox_rgb(raw_bgr, image_size=224)
                if raw_rgb_callback is not None:
                    # Optional offline review: the EXACT decoded exposure,
                    # never an upsample of the224px feature input.
                    raw_rgb_callback(index, np.ascontiguousarray(raw_bgr[..., ::-1]))
                selected[index], valid[index] = previous_pts, True

        error = None
        try:
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError(f'decoded frame without PTS: {path}')
                pts = float(frame.pts * frame.time_base)
                decoded += 1
                if previous_pts is not None and pts <= previous_pts:
                    raise ValueError(f'non-increasing presentation times: {path}')
                while i < len(wanted) and wanted[i] < pts-TIME_EPS:
                    emit(i)
                    i += 1
                if i == len(wanted):
                    break
                previous, previous_pts = frame, pts
        except av.error.InvalidDataError as caught:      # a corrupt packet part-way (see the docstring)
            if not decoded:
                raise
            error = caught
        if error is None:
            while i < len(wanted):
                emit(i)
                i += 1
        detail = dict(path=str(path), actual_fps=float(rate), decoded_frames=decoded,
                      stream_start_pts=stream.start_time, time_base=str(stream.time_base),
                      requested=len(wanted), matched=int(valid.sum()))
        if error is not None:
            detail.update(decode_error=str(error), decode_error_after_pts=previous_pts, first_unserved=i)
    return rgb, selected, valid, detail


def keyframe_times(container, stream, start, stop):
    """Sorted presentation times of the keyframe packets of [start, stop] (demuxed only, nothing decoded)."""
    container.seek(max(0, int(np.floor(start/float(stream.time_base)))), stream=stream, backward=True, any_frame=False)
    times = []
    for packet in container.demux(stream):
        if packet.pts is None:
            continue
        pts = float(packet.pts * packet.time_base)
        if pts > stop + 1.:                      # decode order may run a few frames ahead of display order
            break
        if packet.is_keyframe:
            times.append(pts)
    return np.unique(np.asarray(times, np.float64))


def decode_at_recorded_times_skipping(path, requested_pts, start, stop, *, raw_rgb_callback=None):
    """decode_at_recorded_times, jumping over the frames between requests where a keyframe allows it.

Before each request it seeks to the last keyframe at or before (request - 2 periods) when that keyframe lies after
the frame decoded last; otherwise it keeps decoding in order. Every frame from that keyframe on is decoded, so the
frame chosen for a request (the latest frame up to it, within one period) and its PTS are the ones the in-order
decoder chooses; recordings with sparse keyframes simply never seek. LeRobot's AV1 recordings (a keyframe every 2
frames) then decode about 2 frames per half-second request instead of 15 (James 10/21 「搞个最好的版本就行吧 0.5秒一个」).
Equality with decode_at_recorded_times: tests/test_v21_skip_decode.py and scripts/check_v21_skip_decode.py.
A corrupt packet part-way ends the decoding of the view as in decode_at_recorded_times (17.97 pass 2); the frames
this decoder jumps over are never decoded, so on a corrupt file it can serve requests the in-order decoder cannot.
"""
    wanted = np.asarray(requested_pts, dtype=np.float64)
    if wanted.ndim != 1 or not len(wanted) or not np.isfinite(wanted).all() or (np.diff(wanted) < 0).any():
        raise ValueError('video requests must be finite and sorted')
    rgb = np.zeros((len(wanted), 224, 224, 3), np.uint8)
    selected = np.full(len(wanted), np.nan)
    valid = np.zeros(len(wanted), bool)
    if not Path(path).is_file():
        return rgb, selected, valid, dict(reason='missing-video', path=str(path))
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        if rate is None or not 1 <= float(rate) <= 240:
            raise ValueError(f'cannot establish actual video period: {path}')
        period = 1./float(rate)
        keys = keyframe_times(container, stream, start, stop)
        state = dict(previous=None, previous_pts=None, last_decoded_pts=None)
        i, decoded, seeks = 0, 0, 0

        def emit(index):
            previous, previous_pts = state['previous'], state['previous_pts']
            if (previous is not None and previous_pts >= start-TIME_EPS and previous_pts < stop
                    and wanted[index] >= start-TIME_EPS and wanted[index] < stop
                    and -TIME_EPS <= wanted[index]-previous_pts <= period*1.01+TIME_EPS):
                raw_bgr = previous.to_ndarray(format='bgr24')
                rgb[index] = letterbox_rgb(raw_bgr, image_size=224)
                if raw_rgb_callback is not None:
                    raw_rgb_callback(index, np.ascontiguousarray(raw_bgr[..., ::-1]))
                selected[index], valid[index] = previous_pts, True

        def jump_for(index):
            """The keyframe to seek to for this request, or None to keep decoding in order."""
            target = max(start, float(wanted[index]) - 2*period)
            k = int(np.searchsorted(keys, target + TIME_EPS, side='right')) - 1
            last = state['previous_pts']
            if last is None or (k >= 0 and keys[k] > last + TIME_EPS):
                return target
            return None

        frames, error = None, None
        try:
            while i < len(wanted):
                target = jump_for(i) if frames is not None else max(start, float(wanted[i]) - 2*period)
                if target is not None:
                    container.seek(max(0, int(np.floor(target/float(stream.time_base)))), stream=stream,
                                   backward=True, any_frame=False)
                    frames = container.decode(stream)
                    state.update(previous=None, previous_pts=None)
                    seeks += 1
                ended = True
                for frame in frames:
                    if frame.pts is None:
                        raise ValueError(f'decoded frame without PTS: {path}')
                    pts = float(frame.pts * frame.time_base)
                    decoded += 1
                    if state['previous_pts'] is not None and pts <= state['previous_pts']:
                        raise ValueError(f'non-increasing presentation times: {path}')
                    state['last_decoded_pts'] = pts                 # kept across seeks (they only go forward)
                    advanced = False
                    while i < len(wanted) and wanted[i] < pts-TIME_EPS:
                        emit(i)
                        i += 1
                        advanced = True
                    state.update(previous=frame, previous_pts=pts)
                    if i == len(wanted):
                        ended = False
                        break
                    if advanced and jump_for(i) is not None:
                        ended = False
                        break
                if ended:                               # end of the stream: the last frame serves what is left
                    while i < len(wanted):
                        emit(i)
                        i += 1
        except av.error.InvalidDataError as caught:      # a corrupt packet part-way (decode_at_recorded_times)
            if not decoded:
                raise
            error = caught
        detail = dict(path=str(path), actual_fps=float(rate), decoded_frames=decoded, seeks=seeks,
                      keyframes=int(len(keys)), stream_start_pts=stream.start_time, time_base=str(stream.time_base),
                      requested=len(wanted), matched=int(valid.sum()))
        if error is not None:
            detail.update(decode_error=str(error), decode_error_after_pts=state['last_decoded_pts'], first_unserved=i)
    return rgb, selected, valid, detail


def future_grid(timestamps, begin, end):
    """Nearest real readbacks to 100ms queries, preserving every command later.

Groups only within the same episode; the last record is not stretched into
future labels. Padded offsets remain increasing but carry valid=False.
"""
    begin, end = np.asarray(begin, np.int64), np.asarray(end, np.int64)
    grid = np.zeros((len(begin), 41), np.int64)
    offsets = np.broadcast_to(np.arange(41)/10, grid.shape).copy()
    valid = np.zeros((len(begin), 40), bool)
    # 11/02 (17.171): each episode's rows found by one stable sort, in the same order np.flatnonzero(end == stop) gave
    # them, instead of one pass over every row per episode (4 minutes of every run's start on 1.48M Archive rows)
    order = np.argsort(end, kind='stable')
    stops, firsts = np.unique(end[order], return_index=True)
    for stop, rows in zip(stops, np.split(order, firsts[1:])):
        base = int(begin[rows].min())
        clock = np.asarray(timestamps[base:stop], np.float64)
        if not np.isfinite(clock).all() or not (np.diff(clock) > 0).all():
            raise ValueError('missing or nonmonotonic real episode clock')
        starts = np.asarray(timestamps[begin[rows]])
        query = starts[:, None] + np.arange(41)[None]/10
        right = np.searchsorted(clock, query)
        lo, hi = np.maximum(right-1, 0), np.minimum(right, len(clock)-1)
        idx = np.where(abs(clock[lo]-query) <= abs(clock[hi]-query), lo, hi)
        usable = query[:, 1:] <= clock[-1]+TIME_EPS
        chosen = base+idx-begin[rows, None]
        actual_offsets = clock[idx]-starts[:, None]
        grid[rows] = chosen
        offsets[rows] = np.where(np.concatenate((np.ones((len(rows), 1), bool), usable), 1),
                                 actual_offsets, offsets[rows])
        valid[rows] = usable
        if ((np.diff(chosen, axis=1) <= 0) & usable).any():
            raise ValueError('recording too slow for distinct 10Hz readbacks')
    return grid, offsets, valid
