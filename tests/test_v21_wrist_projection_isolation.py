"""Wrist geometry isolation using real offline FK; no camera or motor access."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.so101_camera_calibration import predict_projection
from fcpg.v21_memory import VisualObservation
from fcpg.v21_runtime import EncodedEndpoint
from fcpg.v21_tip_display import TipDisplay
import fcpg.v21_tip_display as tip_module


@pytest.fixture(scope='module')
def kinematics():
    pytest.importorskip('mujoco')
    from fcpg.so101_kinematics import SO101MuJoCoKinematics
    model = (Path(__file__).resolve().parents[1]
             / 'third_party/so_arm100/Simulation/SO101/so101_new_calib.xml')
    if not model.is_file():
        pytest.skip('official SO-101 model unavailable')
    return SO101MuJoCoKinematics(model)


def _calibration(tmp_path, column='valid'):
    # Portable fixture coefficients, not a change to the production calibration.
    data = dict(
        geometry_protocol='letterbox_v1', content_rect=[0, 28, 224, 196],
        source_height=480, source_width=640,
        fits=dict(wrist=dict(
            feature_names=['x_gripper', 'y_gripper', 'z_gripper', 'opening', 'bias'],
            coefficients=[[20., 10.], [0., 0.], [1200., 20.], [2., 4.], [100., 110.]],
            ridge=1e-8)))
    if column == 'valid':
        data['fits']['column'] = dict(
            feature_names=['x_base', 'y_base', 'z_base', 'bias'],
            coefficients=[[200., 0.], [0., 200.], [0., -200.], [80., 110.]],
            ridge=1e-8)
    elif column == 'null':
        data['fits']['column'] = None
    elif column == 'wrong_shape':
        data['fits']['column'] = dict(
            feature_names=['x_base', 'y_base', 'z_base', 'bias'],
            coefficients=[[1., 2.]], ridge=1e-8)
    elif column == 'invalid_number':
        data['fits']['column'] = dict(
            feature_names=['x_base', 'y_base', 'z_base', 'bias'],
            coefficients=[['bad', 0.], [0., 0.], [0., 0.], [0., 0.]], ridge=1e-8)
    elif column != 'missing':
        raise ValueError(column)
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(data))
    return path, data


def _capture(*, column_available=True):
    wrist_q = [0., -50., 50., 60., -50., 30.]
    column_q = [8., -20., 60., 25., 30., 60.]
    q_at_image = torch.tensor([[wrist_q, column_q]], dtype=torch.float32)
    obs = VisualObservation(
        torch.zeros(1, 2, 256, 384), q_at_image,
        torch.tensor([[5., 5.02]], dtype=torch.float64),
        torch.tensor([[True, column_available]]))
    endpoint = EncodedEndpoint(
        torch.tensor([[9., -70., 20., 25., 20., 3.]]),
        torch.tensor([6.], dtype=torch.float64), obs)
    pixels = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    raw = {v: np.zeros((480, 640, 3), dtype=np.uint8) for v in ('wrist', 'column')}
    return pixels, endpoint, raw


def _assert_wrist(display, result, endpoint):
    wrist = result['wrist']
    q = endpoint.observation.q_at_image[0, 0].numpy().astype(np.float64)
    world, local = display.wrist_kinematics.forward_anchor_frames(q[None])
    expected = predict_projection(display.wrist_fit, world, local, q[None, 5])[0, 1:3]
    assert wrist['available']
    assert wrist['source'] == 'wrist-fk-projection'
    assert wrist['fk_based'] and wrist['display_only'] and not wrist['control_input']
    assert not wrist['accuracy_verified']
    assert wrist['image_time_s'] == 5.
    np.testing.assert_array_equal(wrist['q_at_image'], q)
    np.testing.assert_allclose(wrist['fm_px'], expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(wrist['m_px'], expected.mean(0), rtol=0, atol=1e-9)
    return wrist


@pytest.mark.parametrize('column', ['missing', 'null', 'wrong_shape', 'invalid_number'])
def test_missing_or_broken_column_calibration_keeps_wrist(tmp_path, kinematics, column):
    path, data = _calibration(tmp_path, column)
    original = path.read_bytes()
    display = TipDisplay(path, kinematics, legacy_root=tmp_path / 'missing-legacy')
    pixels, endpoint, raw = _capture()
    result = display.observe(pixels, endpoint, raw, None)
    _assert_wrist(display, result, endpoint)
    assert not result['column']['available']
    np.testing.assert_array_equal(display.wrist_fit.coefficients, data['fits']['wrist']['coefficients'])
    assert path.read_bytes() == original


@pytest.mark.parametrize('legacy_state', ['missing', 'malformed_config', 'incomplete_config'])
def test_missing_or_corrupt_legacy_assets_keep_wrist(tmp_path, kinematics, legacy_state):
    path, _ = _calibration(tmp_path)
    legacy = tmp_path / 'legacy'
    if legacy_state != 'missing':
        legacy.mkdir()
        (legacy / 'config.json').write_text('{' if legacy_state == 'malformed_config' else '{}')
    display = TipDisplay(path, kinematics, legacy_root=legacy)
    pixels, endpoint, raw = _capture()
    result = display.observe(pixels, endpoint, raw, None)
    _assert_wrist(display, result, endpoint)
    assert not result['column']['available']


def test_wrist_uses_its_exposure_q_and_preserves_inputs(tmp_path, kinematics):
    path, _ = _calibration(tmp_path)
    display = TipDisplay(path, kinematics, legacy_root=tmp_path / 'missing-legacy')
    pixels, endpoint, raw = _capture(column_available=False)
    saved_q = endpoint.observation.q_at_image.clone()
    saved_features = endpoint.observation.features.clone()
    saved_times = endpoint.observation.image_times.clone()
    before = display.observe(pixels, endpoint, raw, None)
    _assert_wrist(display, before, endpoint)
    # Changing the later readback and the other exposure cannot move Wrist F/G.
    endpoint.q[0] = torch.tensor([-20., 10., 0., 0., -20., 90.])
    endpoint.time.fill_(9.)
    endpoint.observation.q_at_image[0, 1] = torch.tensor([20., 30., 40., 50., 60., 80.])
    changed = display.observe(pixels, endpoint, raw, None)
    assert changed['wrist'] == before['wrist']
    torch.testing.assert_close(endpoint.observation.features, saved_features, rtol=0, atol=0)
    torch.testing.assert_close(endpoint.observation.image_times, saved_times, rtol=0, atol=0)
    torch.testing.assert_close(endpoint.observation.q_at_image[:, 0], saved_q[:, 0], rtol=0, atol=0)
    # Same opening at another arm pose is invariant in the moving Wrist frame.
    endpoint.observation.q_at_image[0, 0, :5] += torch.tensor([1., 2., 3., -4., 5.])
    arm_changed = display.observe(pixels, endpoint, raw, None)
    _assert_wrist(display, arm_changed, endpoint)
    np.testing.assert_allclose(arm_changed['wrist']['fm_px'], before['wrist']['fm_px'], rtol=0, atol=1e-9)
    # J6 is the opening axis. It must still move the jaw pair in the same frame.
    endpoint.observation.q_at_image[0, 0, 5] += 5.
    opened = display.observe(pixels, endpoint, raw, None)
    _assert_wrist(display, opened, endpoint)
    assert np.linalg.norm(np.asarray(opened['wrist']['fm_px']) - before['wrist']['fm_px']) > 1.


def test_column_loading_failure_does_not_change_wrist(tmp_path, kinematics, monkeypatch):
    path, _ = _calibration(tmp_path)
    display = TipDisplay(path, kinematics, legacy_root=tmp_path / 'missing-legacy')
    pixels, endpoint, raw = _capture(column_available=False)
    before = deepcopy(display.observe(pixels, endpoint, raw, None)['wrist'])

    def broken_column():
        raise RuntimeError('deliberately broken Column model')

    monkeypatch.setattr(display, '_load_column', broken_column)
    endpoint.observation.measurement_available[0, 1] = True
    result = display.observe(pixels, endpoint, raw, None)
    assert result['wrist'] == before
    _assert_wrist(display, result, endpoint)
    assert not result['column']['available']


@pytest.mark.parametrize('failure_stage', ['observer', 'visual_detector'])
def test_column_recognition_failure_keeps_same_wrist(
        tmp_path, kinematics, monkeypatch, failure_stage):
    path, _ = _calibration(tmp_path)
    display = TipDisplay(path, kinematics, legacy_root=tmp_path / 'missing-legacy')
    pixels, endpoint, raw = _capture(column_available=False)
    before = deepcopy(display.observe(pixels, endpoint, raw, None)['wrist'])
    reached = []

    class Observer:
        evidence_threshold = .5

        def __call__(self, *args, **kwargs):
            reached.append('observer')
            if failure_stage == 'observer':
                raise RuntimeError('deliberately broken Column recognition')
            return SimpleNamespace(
                candidate_belief=SimpleNamespace(
                    weights=torch.ones(1, 2, 4),
                    visible_probability=torch.ones(1, 2, 4, 2)),
                raw_column_fm_px=torch.tensor([[[[90., 120.], [160., 120.]]] * 4]))

    def broken_visual_detector(*args, **kwargs):
        reached.append('visual_detector')
        raise RuntimeError('deliberately broken Column visual refinement')

    monkeypatch.setattr(display, '_load_column', lambda: None)
    display.geometry = SimpleNamespace(make_prior=lambda q: object())
    display.observer = Observer()
    if failure_stage == 'visual_detector':
        monkeypatch.setattr(tip_module, 'locate_column_jaw_tips', broken_visual_detector)
    endpoint.observation.measurement_available[0, 1] = True
    result = display.observe(pixels, endpoint, raw, None)
    assert failure_stage in reached
    assert result['wrist'] == before
    _assert_wrist(display, result, endpoint)
    assert not result['column']['available']
