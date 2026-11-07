"""Regression for the September 25 wrong-FPS cache incident."""
from fractions import Fraction
import json

import numpy as np
import pytest

from fcpg.v21_recorded_clock import (
    decode_at_recorded_times, read_recording, resolve_media, future_grid,
)


def video(path, fps, start=0., count=16):
    av = pytest.importorskip('av')
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('mpeg4', rate=fps)
        stream.width, stream.height, stream.pix_fmt = 32, 24, 'yuv420p'
        stream.time_base = Fraction(1, fps)
        for i in range(count):
            rgb = np.full((24, 32, 3), 30 + i*5, np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts, frame.time_base = int(round(start*fps)) + i, Fraction(1, fps)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


@pytest.mark.parametrize('fps', [10, 30])
def test_actual_pts_not_nominal_fps(tmp_path, fps):
    path = tmp_path/'video.mp4'
    video(path, fps)
    requested = np.array([0., 3/fps, 9/fps])
    rgb, pts, valid, detail = decode_at_recorded_times(path, requested, 0., 16/fps)
    np.testing.assert_allclose(pts, requested, atol=1e-6)
    assert valid.all() and detail['actual_fps'] == fps
    assert rgb[2].mean() > rgb[1].mean() > rgb[0].mean()


def test_nonzero_pts_not_zero_rebased_and_tail_not_repeated(tmp_path):
    path = tmp_path/'offset.mp4'
    video(path, 10, start=.1, count=6)
    _, pts, valid, _ = decode_at_recorded_times(path, np.array([0., .1, .35, .6, .9]), 0., 1.)
    assert valid.tolist() == [False, True, True, True, False]
    np.testing.assert_allclose(pts[valid], [.1, .3, .6])
    assert pts[2] <= .35  # Never the future .4 frame.


def test_missing_camera_is_explicit(tmp_path, monkeypatch):
    import builtins
    original_import = builtins.__import__
    def without_decoder(name, *args, **kwargs):
        if name == 'av':
            pytest.fail('missing video must not require an installed decoder')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_decoder)
    _, pts, valid, detail = decode_at_recorded_times(tmp_path/'absent.mp4', np.array([0., .1]), 0., 1.)
    assert not valid.any() and detail['reason'] == 'missing-video'
    assert np.isnan(pts).all()


def test_optional_raw_review_preserves_exact_default_cache_output(tmp_path):
    path = tmp_path/'raw.mp4'
    video(path, 30)
    wanted = np.array([0., .1, .3])
    base = decode_at_recorded_times(path, wanted, 0., 1.)
    raw = {}
    actual = decode_at_recorded_times(path, wanted, 0., 1., raw_rgb_callback=lambda i, v: raw.__setitem__(i, v))
    for left, right in zip(base[:3], actual[:3]):
        np.testing.assert_array_equal(left, right)
    assert set(raw) == {0, 1, 2}
    assert raw[0].shape == (24, 32, 3)  # Original decoded resolution, not resized224.
    assert raw[2].mean() > raw[0].mean()


def test_sharded_media_uses_metadata_origin_and_real_record_clock(tmp_path):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    (tmp_path/'meta/episodes/chunk-000').mkdir(parents=True)
    (tmp_path/'data').mkdir()
    view = 'observation.images.wrist'
    (tmp_path/'meta/info.json').write_text(json.dumps({
        'fps': 10, 'features': {view: {'info': {'video.fps': 10}}},
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}))
    pq.write_table(pa.table({'episode_index': [7], f'videos/{view}/chunk_index': [2],
        f'videos/{view}/file_index': [3], f'videos/{view}/from_timestamp': [2.5],
        f'videos/{view}/to_timestamp': [3.5]}), tmp_path/'meta/episodes/chunk-000/file-000.parquet')
    for file, times in [('a', [10., 10.1]), ('b', [10.2, 10.3])]:
        pq.write_table(pa.table({'episode_index': [7, 7], 'timestamp': times,
            'observation.state': [[1.]*6]*2, 'action': [[2.]*6]*2,
            f'{view}_mask': [1, 0]}), tmp_path/f'data/{file}.parquet')
    row = dict(dataset_root=str(tmp_path), episode_index=7, length=4,
        segments=[dict(data_path=f'data/{f}.parquet', row_start=0, row_stop=2) for f in ('a', 'b')])
    record = read_recording(row, [view])
    np.testing.assert_allclose(record['time'], [10, 10.1, 10.2, 10.3])
    assert record['masks'][:, 0].tolist() == [True, False, True, False]
    media = resolve_media(row, view, record['time'])
    assert media['path'].endswith('/chunk-002/file-003.mp4')
    np.testing.assert_allclose(media['requested_pts'], [2.5, 2.6, 2.7, 2.8])
    assert media['clock_origin'] == 7.5


def test_irregular_future_grid_uses_real_times_and_masks_tail():
    times = np.array([0., .033, .069, .101, .14, .19, .21, .24])
    grid, offsets, valid = future_grid(times, np.array([0, 3]), np.array([8, 8]))
    assert grid[0, 1:3].tolist() == [3, 6]
    np.testing.assert_allclose(offsets[0, 1:3], [.101, .21])
    assert valid[0, :2].all() and not valid[0, 2:].any()
    assert valid[1, 0] and not valid[1, 1:].any()
    assert (grid >= 0).all()
