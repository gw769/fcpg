"""Section 17.64: the recording look (fcpg/v21_recording_look.py). The pipe path gives the pictures the file path of the
compression gap measurement gives, it is deterministic, and the policy's pictures follow the valid views. The cases with
the recording encoder run only where it is installed (the lerobot environment's ffmpeg on .102)."""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fcpg.v21_recording_look import FF_RECORD, VF, X264, RecordingLook, policy_pictures, recording_look_enabled

ffmpeg = pytest.mark.skipif(not Path(FF_RECORD).exists(), reason='the recording encoder is not installed here')


def frames(n, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, 640, dtype=np.float32)[None, :, None] * np.ones((480, 1, 3), np.float32)
    return [np.clip(base + rng.normal(0, 20, (480, 640, 3)), 0, 255).astype(np.uint8) for _ in range(n)]


@ffmpeg
def test_the_pipes_give_the_pictures_of_the_measured_file_path(tmp_path):
    # per frame: the frame twice as numbered PNG files -> mp4 -> decoded (the compression gap measurement's way)
    import imageio_ffmpeg
    from PIL import Image
    pictures = frames(3)
    expected = []
    for i, frame in enumerate(pictures):
        folder = tmp_path / f'png{i}'
        folder.mkdir()
        for copy in range(2):
            Image.fromarray(frame).save(folder / f'{copy:06d}.png')
        subprocess.run([FF_RECORD, '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '30', '-i',
                        str(folder / '%06d.png'), *X264, '-threads', '1', str(tmp_path / f'key{i}.mp4')], check=True)
        raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-threads', '1', '-i',
                              str(tmp_path / f'key{i}.mp4'), '-frames:v', '2', '-vf', VF, '-pix_fmt', 'rgb24', '-f',
                              'rawvideo', 'pipe:1'], check=True, stdout=subprocess.PIPE).stdout
        expected.append(np.frombuffer(raw, np.uint8).reshape(2, 224, 224, 3)[0])
    look = RecordingLook()
    got = look.pictures(pictures)
    assert got.shape == (3, 224, 224, 3) and got.dtype == np.uint8 and np.array_equal(got, np.stack(expected))
    assert np.array_equal(look.pictures(pictures), got)                          # deterministic
    assert np.array_equal(look.pictures(pictures[1:2])[0], got[1])               # a frame's look does not depend on others
    assert not got[:, :28].any() and not got[:, -28:].any()                      # 640x480 -> 224x168, padded as in training


@ffmpeg
def test_the_look_refuses_odd_sizes_and_takes_frames_of_any_even_size():
    look = RecordingLook()
    with pytest.raises(ValueError, match='even size'):
        look.pictures([np.zeros((479, 640, 3), np.uint8)])
    got = look.pictures([frames(1)[0], np.zeros((240, 320, 3), np.uint8)])
    assert got.shape == (2, 224, 224, 3)


def test_the_policy_pictures_follow_the_valid_views():
    class Look:
        calls = []

        def pictures(self, frames):
            self.calls.append(len(frames))
            return np.stack([np.full((224, 224, 3), int(np.asarray(f)[0, 0, 0]), np.uint8) for f in frames])

    raw = dict(wrist=np.full((480, 640, 3), 7, np.uint8), column=np.full((480, 640, 3), 9, np.uint8))
    pictures, looks = policy_pictures(Look(), raw, np.array([True, True]))
    assert pictures[:, 0, 0, 0].tolist() == [7, 9] and looks == dict(wrist='recording', column='recording') and Look.calls == [2]
    pictures, looks = policy_pictures(Look(), raw, np.array([False, True]))
    assert pictures[:, 0, 0, 0].tolist() == [0, 9] and looks == dict(wrist=None, column='recording')
    raw['wrist'] = np.full((224, 224, 3), 5, np.uint8)                           # no raw frame larger than the letterbox
    pictures, looks = policy_pictures(Look(), raw, np.array([True, True]))
    assert pictures[:, 0, 0, 0].tolist() == [5, 9] and looks == dict(wrist='recording-of-letterbox', column='recording')
    with pytest.raises(ValueError, match='no raw frame'):
        policy_pictures(Look(), dict(column=raw['column']), np.array([True, True]))


def test_the_switch(monkeypatch):
    monkeypatch.delenv('V21_RECORDING_LOOK', raising=False)
    assert recording_look_enabled()
    monkeypatch.setenv('V21_RECORDING_LOOK', 'off')
    assert not recording_look_enabled()
