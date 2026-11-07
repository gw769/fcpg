"""Training overhaul doc section 17.64: the pictures the policy was trained on, made from the robot's camera frames.

The Actor and the LeWM learned from recordings: the fish demonstrations are libx264 video (crf 30, a keyframe every 2
frames, yuv420p, 640x480 at 30 fps; recorded with the lerobot environment's ffmpeg, x264 core 165) decoded for training
by imageio-ffmpeg's ffmpeg with scripts/prepare_so101_lewm.py's scale and pad to 224 (rgb24). The service letterboxed
the raw camera frame with PIL instead, and on 1,002 raw robot frames per camera the DINOv3-L features of the two
differed by about 30% (all-l-20261022/compression-gap/report.json; the Actor's proposal moved 0.24-0.36 units).
RecordingLook.pictures(frames) gives a frame the recordings' look: the frame twice in a row through the recording
encoder with the demonstrations' x264 settings (a keyframe, then a static P frame), the keyframe decoded and scaled as
training decodes (same binaries, same options, one thread each). That is the compression gap measurement's "key"
variant, except that each frame is encoded on its own: x264's look-ahead lets a keyframe depend on the frames after it,
so a frame's look would depend on the other camera's frame in the same call (all-l-20261022/recording-look-debug).
A DINOv3-L service gives the policy (Actor, LeWM: the endpoint observation) the tokens of these pictures and keeps the
live letterbox's tokens for the learned finders and judges (TrialOwner.last_model_tokens), part of whose training
pictures were raw robot photos; V21_RECORDING_LOOK=off gives the policy the live letterbox as before.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import subprocess

import numpy as np

FF_RECORD = '/home/james/miniconda3/envs/lerobot/bin/ffmpeg'      # the build that recorded the demos (x264 core 165)
VF = 'scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:(ow-iw)/2:(oh-ih)/2'   # prepare_so101_lewm.py
X264 = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-g', '2', '-crf', '30',
        '-x264-params', 'keyint=2:keyint_min=1:bframes=1:rc-lookahead=2:slices=7']      # the demos' SEI
SIZE = 224


def _run(command, data):
    done = subprocess.run(command, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if done.returncode:
        raise RuntimeError(f'{command[0]} failed: {done.stderr.decode(errors="replace")[-800:]}')
    return done.stdout


class RecordingLook:
    """pictures(frames) -> uint8 [n, 224, 224, 3]: camera frames (RGB uint8 [h, w, 3], even sizes) as the recordings
    show them, each frame on its own (the frames of one call are made side by side on ``workers`` threads)."""

    def __init__(self, encoder=FF_RECORD, decoder=None, workers=2):
        if decoder is None:
            import imageio_ffmpeg
            decoder = imageio_ffmpeg.get_ffmpeg_exe()
        self.encoder, self.decoder = str(encoder), str(decoder)
        self.pool = ThreadPoolExecutor(int(workers))

    def picture(self, frame):
        frame = np.ascontiguousarray(frame)
        h, w = frame.shape[:2]
        if frame.shape != (h, w, 3) or frame.dtype != np.uint8 or h % 2 or w % 2:
            raise ValueError('recording look: an RGB uint8 frame of even size')
        stream = _run([self.encoder, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', f'{w}x{h}', '-framerate', '30', '-i', 'pipe:0', *X264, '-threads', '1', '-f', 'h264',
                       'pipe:1'], frame.tobytes() * 2)
        raw = _run([self.decoder, '-hide_banner', '-loglevel', 'error', '-threads', '1', '-f', 'h264', '-i', 'pipe:0',
                    '-frames:v', '2', '-vf', VF, '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1'], stream)
        if len(raw) != 2 * SIZE * SIZE * 3:
            raise RuntimeError(f'recording look: decoded {len(raw)} bytes, expected 2 pictures')
        return np.frombuffer(raw, np.uint8).reshape(2, SIZE, SIZE, 3)[0].copy()

    def pictures(self, frames):
        frames = list(frames)
        if not frames:
            return np.zeros((0, SIZE, SIZE, 3), np.uint8)
        return np.stack(list(self.pool.map(self.picture, frames)))

    def describe(self):
        return dict(section='17.64', encoder=self.encoder, decoder=self.decoder, x264=X264, scale=VF,
                    keyframe='each frame on its own, twice (I, static P), the I frame decoded')


def recording_look_enabled():
    """The policy reads the recording look unless V21_RECORDING_LOOK=off."""
    return os.environ.get('V21_RECORDING_LOOK', 'on') != 'off'


def policy_pictures(look, raw_frames, valid, views=('wrist', 'column')):
    """The recording look of every valid view's raw camera frame: uint8 [len(views), 224, 224, 3] (zeros where the
    view is not valid) and per view 'recording', or 'recording-of-letterbox' when the camera gave no larger raw frame
    than the letterbox, or None."""
    pictures = np.zeros((len(views), SIZE, SIZE, 3), np.uint8)
    looks = {view: None for view in views}
    members = []
    for i, view in enumerate(views):
        if valid[i]:
            if view not in raw_frames:
                raise ValueError(f'recording look: no raw frame of the {view} camera')
            members.append((i, view, np.asarray(raw_frames[view])))
    if members:
        for (i, view, frame), picture in zip(members, look.pictures([frame for _, _, frame in members])):
            pictures[i] = picture
            looks[view] = 'recording-of-letterbox' if frame.shape[:2] == (SIZE, SIZE) else 'recording'
    return pictures, looks
