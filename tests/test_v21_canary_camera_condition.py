"""CPU-only TrialOwner.capture camera-condition hook; no hardware or checkpoint."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

import scripts.canary_so101_v21_research as entry
from fcpg.v21_camera_condition import V21CameraCondition
from fcpg.v21_capture import capture_actual_views


@pytest.fixture
def owner_case(tmp_path, monkeypatch):
    expected = {view: tmp_path / view for view in ("wrist", "column")}
    for path in expected.values():
        path.mkdir()
    monkeypatch.setattr(entry.established, "DEFAULT_WRIST_DEVICE", str(expected["wrist"]))
    monkeypatch.setattr(entry.established, "DEFAULT_COLUMN_DEVICE", str(expected["column"]))
    now = [10.]

    def clock():
        now[0] += .001
        return now[0]

    monkeypatch.setattr(entry.time, "monotonic", clock)
    monkeypatch.setattr(entry, "capture_actual_views",
                        lambda *a, **kwargs: capture_actual_views(*a, clock=clock, **kwargs))
    owner = entry.TrialOwner.__new__(entry.TrialOwner)
    owner.device = torch.device("cpu")
    owner.dino = object()
    owner.active_views = ("wrist", "column")
    owner.acquisitions = []
    owner.runtime = SimpleNamespace(state=None)
    owner._lease_check = lambda: None
    raw = {view: np.full((480, 640, 3), 222, np.uint8) for view in expected}

    def camera(view):
        value = 40 if view == "wrist" else 80
        raw[view] = np.full((480, 640, 3), value, np.uint8)
        return np.full((224, 224, 3), value, np.uint8)

    owner.io = SimpleNamespace(camera_lock=threading.RLock(), _last_raw_frames=raw, _read_camera=camera)
    q_count = [0]

    def read_q():
        q_count[0] += 1
        return np.full(6, float(q_count[0]) * .05, np.float32)

    owner.read_checked = read_q

    def extract(_backbone, rgb, **kwargs):
        return np.ones((len(rgb), 256, 384), np.float16)

    monkeypatch.setattr(entry, "extract_dinov2_tokens", extract)
    owner.object_relative = SimpleNamespace(
        encode=lambda rgb, q: np.array([1., .8, .1, .2, .3, .4, .5, .6], np.float32))
    return owner


def test_capture_filters_without_committing_until_empty_memory_accepts_seed(owner_case):
    owner = owner_case
    owner.camera_condition = V21CameraCondition("wrist_h0")
    owner.runtime.state = None
    first = owner.capture()
    assert first.observation.measurement_available.tolist() == [[True, True]]
    assert first.observation.features[0, 0].count_nonzero()
    assert not owner.camera_condition.h0_committed
    assert owner.last_proposed_camera_condition.h0_committed
    owner.discard_proposed_camera_condition()
    assert owner.last_proposed_camera_condition is None

    owner.runtime.state = None
    seed = owner.capture()
    assert seed.observation.measurement_available.tolist() == [[True, True]]
    with pytest.raises(ValueError, match="empty actual memory"):
        owner.commit_camera_condition_h0()
    owner.runtime.state = SimpleNamespace(
        memory=SimpleNamespace(valid=torch.ones(1, 2, 1, dtype=torch.bool)))
    committed = owner.commit_camera_condition_h0()
    assert committed.h0_committed and owner.camera_condition is committed
    later = owner.capture()
    assert later.observation.measurement_available.tolist() == [[False, True]]
    assert not later.observation.features[0, 0].count_nonzero()
    assert not later.observation.object_relative.count_nonzero()
    assert owner.camera_condition.h0_committed

    owner.camera_condition = V21CameraCondition("wrist_cold")
    owner.capture()
    with pytest.raises(ValueError, match="empty actual memory"):
        owner.commit_camera_condition_h0()
