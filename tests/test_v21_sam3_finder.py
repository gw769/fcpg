"""fcpg/v21_sam3_finder.py: the wire, the geometry, both token conventions against the encoders they replace or follow,
the raw-picture match, the column record and the colour guard (training overhaul doc section 17.32)."""
import json
import socket
import subprocess
import sys
import threading

import numpy as np
import pytest

from fcpg import v21_sam3_finder as finder
from fcpg.letterbox_geometry import letterbox_geometry, letterbox_rgb_image
from fcpg.v21_object_relative import encode_vector
from fcpg.v21_outline_cue import cue_vector, picture_geometry


def ellipse_mask(h, w, cx, cy, a, b, angle):
    ys, xs = np.mgrid[:h, :w] + .5
    c, s = np.cos(angle), np.sin(angle)
    u, v = (xs - cx) * c + (ys - cy) * s, -(xs - cx) * s + (ys - cy) * c
    return (u / a) ** 2 + (v / b) ** 2 <= 1.


def test_wire_round_trip_and_masks():
    left, right = socket.socketpair()
    masks = np.random.default_rng(0).random((3, 17, 23)) > .5
    finder.send_message(left, dict(a=1, b=[2, 3]), finder.pack_masks(masks))
    header, payload = finder.receive_message(right)
    assert header == dict(a=1, b=[2, 3])
    assert np.array_equal(finder.unpack_masks(payload, 3, 17, 23), masks)
    assert finder.unpack_masks(b'', 0, 4, 5).shape == (0, 4, 5)
    left.close(), right.close()


def test_client_against_a_fake_finder(tmp_path):
    path = tmp_path / 'f.sock'
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    mask = ellipse_mask(48, 64, 30, 20, 10, 4, .3)

    def run():
        conn, _ = server.accept()
        with conn:
            while True:
                try:
                    header, payload = finder.receive_message(conn)
                except ConnectionError:
                    return
                h, w, _ = header['shape']
                assert len(payload) == h * w * 3 and header['prompt'] == 'fish'
                finder.send_message(conn, dict(ok=True, mask_shape=[1, h, w], scores=[.8], model_s=.01),
                                    finder.pack_masks(mask[None]))
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    client = finder.FinderClient(path)
    found, info = client.find(np.zeros((48, 64, 3), np.uint8))
    assert len(found) == 1 and np.array_equal(found[0][0], mask) and found[0][1] == pytest.approx(.8)
    assert info['found'] == 1 and info['model_s'] == pytest.approx(.01)
    client.close()
    server.close()


def test_jaw_token_is_the_colour_token_layout():
    lb = np.zeros((224, 224, 3), np.uint8)
    mask = ellipse_mask(224, 224, 120.3, 101.7, 30, 9, .4)
    record = finder.record_from_mask(mask, .7, finder.PictureTransform((224, 224), raw=False))
    pair = np.array([[90., 140.], [130., 138.]])
    ours = finder.token(record, 'jaw', jaw_pair_px=pair)
    theirs = encode_vector(dict(cog_px=record['cog_px'], axis_px=record['axis_px'], length_px=record['length_px'],
                                width_px=record['width_px'], score=.7), pair)
    np.testing.assert_allclose(ours, theirs, atol=1e-6)
    assert lb.shape == (224, 224, 3)


def test_hold_token_is_the_outline_cue_on_a_raw_picture():
    h, w = 480, 640
    mask = ellipse_mask(h, w, 371.2, 260.9, 80, 22, -.7)
    transform = finder.PictureTransform((h, w), raw=True)
    record = finder.record_from_mask(mask, .9, transform)
    reference_raw = (330.5, 355.25)
    hold = transform([reference_raw])[0]
    ours = finder.token(record, 'hold', hold_px=hold)
    theirs = cue_vector(picture_geometry(mask, (h, w)), reference_raw, (h, w), .9)
    np.testing.assert_allclose(ours, theirs, atol=1e-5)


def test_picture_transform_is_letterbox_v1():
    transform = finder.PictureTransform((480, 640), raw=True)
    points = np.array([[0., 0.], [640., 480.], [320., 240.]])
    np.testing.assert_allclose(transform(points), letterbox_geometry(640, 480, 224).raw_to_planning(points))
    np.testing.assert_allclose(transform([[320., 240.]])[0], [112., 112.], atol=1e-9)
    with pytest.raises(ValueError):
        finder.PictureTransform((200, 224), raw=False)


def test_matching_raw():
    rng = np.random.default_rng(1)
    raw = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    raw = np.ascontiguousarray(np.repeat(np.repeat(raw[::8, ::8], 8, 0), 8, 1))     # smooth enough to resample
    lb = letterbox_rgb_image(raw)
    assert finder.matching_raw(raw, lb) is raw or np.array_equal(finder.matching_raw(raw, lb), raw)
    assert np.array_equal(finder.matching_raw(np.ascontiguousarray(raw[..., ::-1]), lb), raw)
    other = np.ascontiguousarray(raw[::-1])
    assert finder.matching_raw(other, lb) is None
    assert finder.matching_raw(None, lb) is None


class FakeClient:
    def __init__(self, masks, scores, prompts=None):
        self.masks, self.scores = masks, scores
        self.prompts = prompts or ['fish'] * len(masks)

    def find(self, picture, prompt, threshold):
        return list(zip(self.masks, self.scores)), dict(round_trip_s=.0, model_s=.0, found=len(self.masks))

    def find_many(self, picture, prompts, threshold):
        found = [(m, s, p) for m, s, p in zip(self.masks, self.scores, self.prompts) if p in prompts]
        return found, dict(round_trip_s=.0, model_s=.0, found=len(found))

    def track(self, picture, stream, prompts, threshold, *, anchor_word, anchor_score):
        self.tracked = getattr(self, 'tracked', 0) + 1
        mask = self.masks[0] if self.masks else None
        return mask, dict(round_trip_s=.0, model_s=.0, found=int(mask is not None),
                          track=dict(source='tracked'), word_scores={anchor_word: None})

    def track_reset(self, stream):
        self.resets = getattr(self, 'resets', 0) + 1


def test_column_record_and_best_choice():
    h, w = 480, 640
    small, big = ellipse_mask(h, w, 100, 100, 8, 3, 0.), ellipse_mask(h, w, 400, 300, 40, 12, .2)
    raw = np.zeros((h, w, 3), np.uint8)
    record = finder.column_pickup_record(FakeClient([small, big], [.4, .8]), letterbox_rgb_image(raw), raw,
                                         image_time_s=12.5)
    expected = letterbox_geometry(640, 480, 224).raw_to_planning(np.array([[400., 300.]]))[0]
    assert record['visible'] and record['source'] == finder.SOURCE and record['image_time_s'] == 12.5
    np.testing.assert_allclose(record['cog_px'], expected, atol=.05)
    assert record['picture_source'] == 'raw_same_exposure' and record['colour_used'] is False
    nothing = finder.column_pickup_record(FakeClient([], []), letterbox_rgb_image(raw), None)
    assert nothing['visible'] is False and nothing['picture_source'] == 'letterbox224'


def test_hold_reference_refuses_the_exam_and_sealed_demonstrations(tmp_path):
    report = {str(d): dict(reference_px=[300., 330.], picture_hw=[448, 597]) for d in (0, 1, 3)}
    path = tmp_path / 'cues.npz'
    np.savez(path, report=json.dumps(report))
    point = finder.demo_hold_reference(path)
    np.testing.assert_allclose(point, [300. * 224 / 597, 330. * 168 / 448 + 28])
    for demos in ((0, 3), (2,)):
        with pytest.raises(ValueError):
            finder.demo_hold_reference(path, demos)


def test_object_prompts_and_rules():
    assert finder.object_prompts('Pick up the red fish toy and place it in the cardboard box') == ['fish toy', 'fish', 'toy']
    assert finder.object_prompts('put the green block into the bowl') == ['block']
    with pytest.raises(ValueError):
        finder.object_prompts('pick up the red one')
    a, b = ellipse_mask(64, 64, 20, 20, 6, 3, 0.), ellipse_mask(64, 64, 44, 40, 9, 4, 1.)
    found = [(a, .5, 'fish'), (b, .9, 'toy')]
    assert finder.choose(found, ['fish toy', 'fish', 'toy'], 'best')[2] == 'toy'
    assert finder.choose(found, ['fish toy', 'fish', 'toy'], 'first')[2] == 'fish'
    assert finder.choose([], ['fish'], 'first') is None
    assert finder.word_scores(found, ['fish toy', 'fish', 'toy']) == {'fish toy': None, 'fish': .5, 'toy': .9}


def test_word_lists_match_the_labeller():
    labeller = pytest.importorskip('analysis.v21.archive_outline_sam3')
    assert finder.FUNCTION_WORDS == labeller.FUNCTION_WORDS and finder.COLOUR_WORDS == labeller.COLOUR_WORDS


class FakeDino:
    """The colour-era encoder's DINO tracker: only locate_by_frozen_features."""
    def __init__(self, visible, score):
        self.visible, self.score, self.calls = visible, score, []

    def locate_by_frozen_features(self, role, view, frame, **kwargs):
        self.calls.append((role, view, frame.shape, np.asarray(kwargs.get('jaw_pair_px')).tolist()))
        return dict(visible=self.visible, score=self.score if self.visible else None)


