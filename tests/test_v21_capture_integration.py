"""TrialOwner.capture integration, no real device access or model loading."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

import scripts.canary_so101_v21_research as entry
from fcpg.v21_capture import capture_actual_views
from test_v21_live_service import service  # noqa: F401 -- fake-IO dashboard fixture


@pytest.fixture
def owner_case(tmp_path, monkeypatch):
    expected={view:tmp_path/view for view in ("wrist","column")}
    for path in expected.values():
        path.mkdir()
    monkeypatch.setattr(entry.established,"DEFAULT_WRIST_DEVICE",str(expected["wrist"]))
    monkeypatch.setattr(entry.established,"DEFAULT_COLUMN_DEVICE",str(expected["column"]))
    events=[]; now=[10.]; failures={}; camera_calls=[]; q_count=[0]; dino_calls=[]; encode_calls=[]
    def clock():
        now[0]+=.001
        return now[0]
    monkeypatch.setattr(entry.time,"monotonic",clock)
    monkeypatch.setattr(entry,"capture_actual_views",
                        lambda *a,**kwargs:capture_actual_views(*a,clock=clock,**kwargs))
    owner=entry.TrialOwner.__new__(entry.TrialOwner)
    owner.device=torch.device("cpu")
    owner.dino=object();owner.active_views=("wrist","column")
    owner.acquisitions=[];owner.saved_column=None
    owner.lens_cover=entry.PaperCoverHold()  # the owner keeps one since the paper cover hold (it was missing here)
    owner.frames={view:np.full((480,640,3),222,np.uint8) for view in expected}
    owner.runtime=SimpleNamespace(state=SimpleNamespace(memory=object(),history=object()))
    owner._lease_check=lambda:events.append("lease")
    raw={view:np.full((480,640,3),222,np.uint8) for view in expected}
    def camera(view):
        events.append("camera:"+view);camera_calls.append(view)
        if view in failures:
            error=failures[view]
            if callable(error):
                return error()
            raise error
        value=40 if view=="wrist" else 80
        raw[view]=np.full((480,640,3),value,np.uint8)
        picture=np.full((224,224,3),value,np.uint8)
        picture[:,1::2]=value+100  # a real workspace is not flat (fcpg/v21_lens_cover.frame_is_covered); [0,0] keeps value
        return picture
    owner.io=SimpleNamespace(camera_lock=threading.RLock(),_last_raw_frames=raw,_read_camera=camera)
    def read_q():
        q_count[0]+=1
        events.append("q")
        return np.full(6,q_count[0]*.05+100*bool(dino_calls),np.float32)
    owner.read_checked=read_q
    def extract(backbone,rgb,**kwargs):
        assert len(rgb)>0,"DINO must never see an empty batch"
        events.append("dino");dino_calls.append(rgb.copy())
        now[0]+=.070  # encoding latency must not retimestamp an earlier q
        return np.ones((len(rgb),256,384),np.float16)
    monkeypatch.setattr(entry,"extract_dinov2_tokens",extract)
    def encode(rgb,q):
        events.append("cue");encode_calls.append((rgb.copy(),np.array(q).copy()))
        return np.array([1.,.8,.1,.2,.3,.4,.5,.6],np.float32)
    owner.object_relative=SimpleNamespace(encode=encode)
    return SimpleNamespace(owner=owner,expected=expected,events=events,failures=failures,
                           camera_calls=camera_calls,dino_calls=dino_calls,encode_calls=encode_calls,
                           q_count=q_count,raw=raw)


@pytest.mark.parametrize("missing", [(),("wrist",),("column",),("wrist","column")])
def test_actual_capture_masks_independent_views_cues_and_dino_without_history_change(owner_case,missing):
    c=owner_case;o=c.owner
    for view in missing:
        c.expected[view].rmdir()  # isolated test directory, not a real device
    before=o.runtime.state;memory=before.memory;history=before.history
    endpoint=o.capture()
    expected=[view not in missing for view in ("wrist","column")]
    assert endpoint.observation.measurement_available.tolist()==[expected]
    assert set(o.frames)=={view for view,available in zip(("wrist","column"),expected) if available}
    assert o.runtime.state is before and before.memory is memory and before.history is history
    assert c.camera_calls==[view for view in ("wrist","column") if view not in missing]
    assert len(c.dino_calls)==int(any(expected))
    if any(expected):
        assert len(c.dino_calls[0])==sum(expected)
        assert c.events.index("dino") < len(c.events)-1 and c.events[-1] == "q"
        if expected[0]:
            assert c.events.index("cue") < len(c.events)-1  # ALL visual encoding precedes current q
        assert endpoint.q[0,0]>100.
        assert endpoint.time.item()>max(endpoint.observation.image_times[0]).item()+.070
    else:
        assert c.q_count[0]==2  # helper final actual q + post-encoder actual q
        assert endpoint.q[0,0]==pytest.approx(.10) and not endpoint.observation.features.any()
    for index,view in enumerate(("wrist","column")):
        if view in missing:
            assert not o.last_rgb[index].any()
            assert not endpoint.observation.q_at_image[0,index].any()
            assert endpoint.observation.image_times[0,index].item()==0.
            assert o.acquisitions[-1]["views"][index]["reason"]=="known_camera_device_missing"
        else:
            assert o.frames[view][0,0,0]==(40 if view=="wrist" else 80)
    if "wrist" in missing:
        assert not c.encode_calls and not endpoint.observation.object_relative.any()
        assert o.last_object_relative==[0.]*8
    else:
        assert len(c.encode_calls)==1 and endpoint.observation.object_relative[0,0]==1.
        np.testing.assert_array_equal(c.encode_calls[0][1],endpoint.observation.q_at_image[0,0].numpy())


def test_slow_object_encoding_cannot_leave_the_final_readback_stale(owner_case):
    c=owner_case
    original=c.owner.object_relative.encode
    read=c.owner.read_checked
    def encode(rgb,q):
        result=original(rgb,q)
        c.owner.read_checked=lambda:read()+50.
        return result
    c.owner.object_relative.encode=encode
    endpoint=c.owner.capture()
    assert endpoint.q[0,0] > 150.
    assert (endpoint.observation.q_at_image < 50.).all()
    assert c.owner.acquisitions[-1]['current_q_after_all_visual_encoding']


@pytest.mark.parametrize("view",["wrist","column"])
def test_wrong_identity_runtime_error_never_becomes_missing_data(owner_case,view):
    c=owner_case;c.failures[view]=RuntimeError("wrong camera identity")
    assert c.expected[view].exists()
    with pytest.raises(RuntimeError,match="wrong camera identity"):
        c.owner.capture()
    assert not c.dino_calls


def test_verified_device_disappearance_during_read_becomes_explicit_missing(owner_case):
    c=owner_case
    def disconnected():
        c.expected["wrist"].rmdir()
        raise RuntimeError("expected stable camera alias disappeared")
    c.failures["wrist"]=disconnected
    endpoint=c.owner.capture()
    assert endpoint.observation.measurement_available.tolist()==[[False,True]]
    assert c.owner.acquisitions[-1]["views"][0]["reason"]=="known_camera_device_missing"
    assert "wrist" not in c.owner.frames


def test_readback_timeout_not_swallowed_by_known_camera_failure_adapter(owner_case):
    c=owner_case
    def failed_q():
        raise TimeoutError("serial actual readback failed")
    c.owner.read_checked=failed_q
    with pytest.raises(TimeoutError,match="serial actual readback failed"):
        c.owner.capture()
    assert not c.dino_calls


def test_active_timeout_after_previous_frame_drops_stale_frame_not_history(owner_case):
    c=owner_case;c.owner.capture()
    old_observation=c.owner.last_endpoint.observation
    saved_times=old_observation.image_times.clone()
    c.failures["column"]=TimeoutError("new column delivery failed")
    c.owner.capture()
    assert "column" not in c.owner.frames
    assert not c.owner.last_rgb[1].any()
    torch.testing.assert_close(old_observation.image_times,saved_times)
    assert c.owner.acquisitions[-1]["measurement_available"]==[True,False]


@pytest.mark.parametrize("cached_live",[False,True])
def test_missing_view_busy_live_fallback_cannot_serve_stale_or_crash_with_keyerror(service,cached_live):
    # Mimic new capture contract: lost raw frame is removed, current view invalid.
    service.owner.last_endpoint.observation.measurement_available[0,1]=False
    service.owner.frames.pop("column")
    service._publish()
    if cached_live:
        service.live_frames["column"]=np.full((480,640,3),222,np.uint8)
        service.live_times["column"]=service.clock()-2.
    with pytest.raises(ValueError,match="no current|unavailable|capture is busy"):
        service.frame("column","endpoint",service.sequence)
    service.owner.io.camera_lock.acquire()
    try:
        with pytest.raises(ValueError,match="no current|unavailable|capture is busy"):
            service.frame("column","live")
    finally:
        service.owner.io.camera_lock.release()


@pytest.mark.parametrize("on", [True, False])
def test_a_dinov3_owner_gives_the_policy_the_recording_look_and_the_finders_the_live_letterbox(owner_case, monkeypatch, on):
    # training overhaul doc section 17.64: the endpoint observation (Actor, LeWM) reads the recording look of the raw
    # frames; last_model_tokens (the learned finders and judges) keep the delivered letterbox; V21_RECORDING_LOOK=off
    # gives the policy the letterbox as before
    c = owner_case
    o = c.owner
    monkeypatch.setattr(entry, "ENCODER", "dinov3_vitl16")
    monkeypatch.setenv("V21_RECORDING_LOOK", "on" if on else "off")

    class Tokens:
        def encode(self, rgb):
            return np.repeat(rgb[:, :1, 0, 0].astype(np.float16), entry.PATCHES * entry.DIM, 1).reshape(
                len(rgb), entry.PATCHES, entry.DIM)

    class Look:                       # the raw frame's value plus one, so the two looks can be told apart
        def pictures(self, frames):
            return np.stack([np.full((224, 224, 3), int(f[0, 0, 0]) + 1, np.uint8) for f in frames])

    o.model_tokens, o.recording_look = Tokens(), Look()
    endpoint = o.capture()
    assert o.last_model_tokens[0, 0, 0] == 40 and o.last_model_tokens[1, 0, 0] == 80
    features = endpoint.observation.features[0]
    assert (features[0, 0, 0].item(), features[1, 0, 0].item()) == ((41, 81) if on else (40, 80))
    assert o.acquisitions[-1]["policy_look"] == ({"wrist": "recording", "column": "recording"} if on else "live letterbox")
