"""The keyframe-skipping decoder returns exactly what the in-order decoder returns (pictures, chosen PTS, validity) on
synthetic recordings with dense and sparse keyframes and with B-frames, and decodes fewer frames when keyframes are
dense (fcpg/v21_recorded_clock.decode_at_recorded_times_skipping; James 10/21 「搞个最好的版本就行吧 0.5秒一个」).
17.97 pass 2: a corrupt packet part-way ends the decoding of the view for both decoders (the requests not served by
then are unavailable, reason 7 in the cache rebuild); a file that decodes no frame still raises."""
import numpy as np
import pytest

av = pytest.importorskip('av')
pytest.importorskip('cv2')

from fcpg import v21_cache_rebuild as rebuild
from fcpg.v21_recorded_clock import decode_at_recorded_times, decode_at_recorded_times_skipping


def write_video(path, *, gop, b_frames, frames=180, rate=30):
    container = av.open(str(path), mode='w')
    stream = container.add_stream('mpeg4', rate=rate)
    stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
    stream.codec_context.gop_size = gop
    stream.codec_context.max_b_frames = b_frames
    for i in range(frames):
        picture = np.zeros((48, 64, 3), np.uint8)
        picture[..., 0] = (i * 3) % 256
        picture[:, :i % 64, 1] = 255
        picture[(i // 4) % 48, :, 2] = 200
        for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format='rgb24')):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.mark.parametrize('gop,b_frames', [(2, 0), (25, 0), (12, 2), (250, 2)])
def test_skipping_decoder_matches_the_in_order_decoder(tmp_path, gop, b_frames):
    path = tmp_path / f'gop{gop}-b{b_frames}.mp4'
    write_video(path, gop=gop, b_frames=b_frames)
    half_seconds = np.arange(0., 5.6, .5)
    for requested in (half_seconds, half_seconds + .041, np.r_[half_seconds, 5.98, 6.02, 6.5],
                      np.r_[0.2, 0.25, 2.9, 3.0, 3.033, 4.4]):
        a = decode_at_recorded_times(path, requested, 0., 6.)
        b = decode_at_recorded_times_skipping(path, requested, 0., 6.)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1], equal_nan=True) and np.array_equal(a[2], b[2])
        assert a[2].any()
        if gop == 2:
            assert b[3]['decoded_frames'] < a[3]['decoded_frames']
    a = decode_at_recorded_times(path, half_seconds + 1., 1., 4.)          # an episode's interval inside the video
    b = decode_at_recorded_times_skipping(path, half_seconds + 1., 1., 4.)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1], equal_nan=True) and np.array_equal(a[2], b[2])


def test_missing_video_is_unavailable_for_both(tmp_path):
    a = decode_at_recorded_times(tmp_path / 'none.mp4', np.array([0., .5]), 0., 1.)
    b = decode_at_recorded_times_skipping(tmp_path / 'none.mp4', np.array([0., .5]), 0., 1.)
    assert not a[2].any() and not b[2].any() and a[3]['reason'] == b[3]['reason'] == 'missing-video'


def write_av1(path, *, bad=None, frames=60, rate=30):
    """An AV1 recording as LeRobot's (a keyframe every 2 frames); with bad=n, packet n (in file order) is overwritten
    with 0xFF bytes, and libdav1d raises InvalidDataError from it on, as on the hand video of
    NeilKim/so100-teleop-record-7 episode 0 (the hub's own bytes; fails at 22.4 of 23.2 s)."""
    if 'libaom-av1' not in av.codecs_available:
        pytest.skip('no AV1 encoder in this PyAV')
    container = av.open(str(path), mode='w')
    stream = container.add_stream('libaom-av1', rate=rate, options={'cpu-used': '8'})
    stream.width, stream.height, stream.pix_fmt = 64, 64, 'yuv420p'
    stream.codec_context.gop_size = 2
    for i in range(frames):
        picture = np.zeros((64, 64, 3), np.uint8)
        picture[..., 0] = (i * 3) % 256
        picture[:, :i % 64, 1] = 255
        picture[(i // 4) % 64, :, 2] = 200
        for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format='rgb24')):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    if bad is not None:
        with av.open(str(path)) as container:
            pos, size = [(p.pos, p.size) for p in container.demux(container.streams.video[0]) if p.size][bad]
        data = bytearray(path.read_bytes())
        data[pos:pos + size] = b'\xff' * size
        path.write_bytes(bytes(data))


def test_corrupt_packet_part_way_ends_the_view_for_both(tmp_path):
    good, path = tmp_path / 'good.mp4', tmp_path / 'corrupt.mp4'
    write_av1(good)
    write_av1(path, bad=45)                                   # 1.5 s of 2 s
    requested = np.arange(0., 2., .1)
    for decoder in (decode_at_recorded_times, decode_at_recorded_times_skipping):
        assert 'decode_error' not in decoder(good, requested, 0., 2.)[3] and decoder(good, requested, 0., 2.)[2].all()
        rgb, chosen, valid, detail = decoder(path, requested, 0., 2.)
        cut = detail['first_unserved']
        assert 'Invalid data' in detail['decode_error'] and 5 <= cut < len(requested)
        assert valid[:cut].all() and not valid[cut:].any()    # never the last good frame for what is left
        assert np.isnan(chosen[cut:]).all() and not rgb[cut:].any()
        assert chosen[cut - 1] <= detail['decode_error_after_pts'] + 1e-6
        reference = decode_at_recorded_times(good, requested, 0., 2.)
        assert np.array_equal(rgb[:cut], reference[0][:cut]) and np.array_equal(chosen[:cut], reference[1][:cut])


def test_a_file_that_decodes_no_frame_still_raises(tmp_path):
    path = tmp_path / 'corrupt-first.mp4'
    write_av1(path, bad=0)
    for decoder in (decode_at_recorded_times, decode_at_recorded_times_skipping):
        with pytest.raises(av.error.InvalidDataError):
            decoder(path, np.arange(0., 2., .1), 0., 2.)


def test_episode_job_marks_the_unserved_views_reason_7(tmp_path, monkeypatch):
    good, bad = tmp_path / 'good.mp4', tmp_path / 'corrupt.mp4'
    write_av1(good)
    write_av1(bad, bad=45)
    times = np.arange(0., 2., .1)
    record = dict(time=times, q=np.zeros((len(times), 6)), action=np.zeros((len(times), 6)),
                  masks=np.ones((len(times), 2), bool), mask_present=np.ones((len(times), 2), bool))
    monkeypatch.setattr(rebuild, 'read_recording', lambda *a: record)
    monkeypatch.setattr(rebuild, 'resolve_media', lambda row, view, t: dict(
        path=str(bad if view == 'wrist' else good), requested_pts=times, start=0., stop=2., clock_origin=0.,
        provenance='test', declared_video_fps=30))
    job = dict(identity=0, row={}, views=['wrist', 'column'], local=list(range(len(times))),
               endpoint_ids=list(range(len(times))), decode='keyframe-skip')
    result = rebuild.episode_job(job)
    cut = result['details'][0]['first_unserved']
    assert result['reasons'][cut:, 0].tolist() == [7] * (len(times) - cut) and not result['valid'][cut:, 0].any()
    assert result['valid'][:cut, 0].all() and (result['reasons'][:cut, 0] == 0).all()
    assert result['valid'][:, 1].all() and 'decode_error' not in result['details'][1]
    assert result['details'][0]['reasons']['7'] == len(times) - cut
    assert result['endpoint_ids'].tolist() == list(range(len(times)))     # no row silently removed
