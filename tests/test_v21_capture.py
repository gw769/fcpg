"""Acquisition contracts with injected callbacks only, no cameras or serial IO."""
import numpy as np
import pytest

from fcpg.v21_capture import CameraDeliveryUnavailable, capture_actual_views


class Fixture:
    def __init__(self):
        self.time = 10.
        self.camera_calls = []
        self.q_calls = 0
        self.failures = {}
        self.frames = {"wrist":np.full((224,224,3),40,np.uint8),
                       "column":np.full((224,224,3),80,np.uint8)}

    def clock(self):
        self.time += .001
        return self.time

    def camera(self, view):
        self.camera_calls.append(view)
        if view in self.failures:
            raise self.failures[view]
        return self.frames[view]

    def read_q(self):
        self.q_calls += 1
        return np.full(6,self.q_calls * .05,np.float32)

    def capture(self, **kwargs):
        return capture_actual_views(self.camera,self.read_q,clock=self.clock,**kwargs)


@pytest.mark.parametrize("missing", ["wrist","column"])
@pytest.mark.parametrize("exception,reason", [(TimeoutError("late"),"camera_timeout"),
                                               (ConnectionError("gone"),"camera_disconnected"),
                                               (CameraDeliveryUnavailable("known_camera_device_missing","expected alias absent"),"known_camera_device_missing")])
def test_missing_view_does_not_hide_other_actual_view(missing, exception, reason):
    f=Fixture();f.failures[missing]=exception
    out=f.capture()
    index=("wrist","column").index(missing)
    assert out.measurement_available.tolist()==([False,True] if missing=="wrist" else [True,False])
    assert not out.rgb[index].any() and not out.q_at_image[index].any()
    assert out.image_times_s[index]==0
    assert out.views[index]["reason"]==reason
    assert out.views[index]["q_sample_s"] is None and out.views[index]["image_time_s"] is None
    assert np.all(out.q_at_image[1-index] > 0.) and np.all(out.q_position > 0.)
    assert out.image_times_s[1-index]>10.
    assert f.q_calls==4 and f.camera_calls==["wrist","column"]


def test_both_missing_still_reads_real_current_q_without_claiming_fresh_images():
    f=Fixture();f.failures={"wrist":TimeoutError(),"column":ConnectionError()}
    out=f.capture()
    assert not out.measurement_available.any() and not out.rgb.any()
    assert not out.q_at_image.any() and not out.image_times_s.any()
    assert np.allclose(out.q_position,.15) and out.q_time_s>10.
    assert f.q_calls==3 and out.raw_frames=={}
    assert all(not row["available"] for row in out.views)


def test_q_and_exposure_times_are_per_view_and_owned_copies():
    f=Fixture();out=f.capture()
    np.testing.assert_allclose(out.q_at_image[:,0],[.10,.20])
    assert np.allclose(out.q_position,.25) and f.q_calls==5
    assert out.q_sequence_id==5
    assert [row['readback_sequence_order'] for row in out.views]==[[1,2],[3,4]]
    assert all(row['readback_fresh'] and row['pose_stable']
               and row['image_enclosed_by_readbacks'] for row in out.views)
    for row in out.views:
        assert (row['q_before_image']['read_finished_s'] <= row['read_started_s']
                <= row['read_finished_s'] <= row['q_after_image']['read_started_s'])
    assert 10 < out.image_times_s[0] < out.image_times_s[1] < out.q_time_s
    assert out.image_times_s.dtype==np.float64 and out.measurement_available.dtype==np.bool_
    f.frames["wrist"][:]=0
    assert np.all(out.rgb[0]==40)


@pytest.mark.parametrize("when", [1,3])
@pytest.mark.parametrize("exception", [TimeoutError("actual q timeout"),ConnectionError("actual bus lost"),RuntimeError("clearance unsafe")])
def test_readback_failures_never_become_missing_camera(when,exception):
    f=Fixture();normal=f.read_q
    def read_q():
        if f.q_calls+1==when:
            raise exception
        return normal()
    with pytest.raises(type(exception),match=str(exception)):
        capture_actual_views(f.camera,read_q,clock=f.clock)


@pytest.mark.parametrize("frame", [np.zeros((223,224,3),np.uint8),np.zeros((224,224,3),np.float32),
                                   np.zeros((224,224),np.uint8),None])
def test_malformed_frame_fails_closed(frame):
    f=Fixture();f.frames["wrist"]=frame
    with pytest.raises(RuntimeError,match="RGB geometry/dtype"):
        f.capture()
    assert f.q_calls==1


def test_unknown_runtime_failure_not_reclassified_by_message():
    f=Fixture();f.failures["wrist"]=RuntimeError("wrist camera alias is missing or disconnected")
    with pytest.raises(RuntimeError,match="camera alias"):
        f.capture()  # adapter must identify known expected device and use typed error


def test_wrong_camera_identity_is_fatal_before_image_delivery():
    f=Fixture()
    def identity(view):
        raise ValueError("wrong camera identity")
    with pytest.raises(ValueError,match="wrong camera identity"):
        f.capture(camera_identity_check=identity)
    assert not f.camera_calls and f.q_calls==0


def test_known_missing_expected_device_identity_check_can_be_explicitly_unavailable():
    f=Fixture()
    def identity(view):
        if view=="wrist":
            raise CameraDeliveryUnavailable("known_camera_device_missing")
    out=f.capture(camera_identity_check=identity)
    assert out.measurement_available.tolist()==[False,True]
    assert f.camera_calls==["column"]


def test_negative_camera_identity_result_cannot_be_ignored():
    f=Fixture()
    with pytest.raises(RuntimeError,match="identity was not verified"):
        f.capture(camera_identity_check=lambda view:False)
    assert not f.camera_calls


@pytest.mark.parametrize("q", [np.zeros(5),np.full(6,np.nan),np.full(6,1e300)])
def test_invalid_actual_q_never_succeeds(q):
    f=Fixture()
    with pytest.raises(RuntimeError,match="actual readback"):
        capture_actual_views(f.camera,lambda:q,clock=f.clock)


def test_slow_stable_camera_is_accepted_with_original_host_timing_preserved():
    f=Fixture();normal=f.camera
    def slow(view):
        f.time+=.051
        return normal(view)
    out=capture_actual_views(slow,f.read_q,clock=f.clock)
    assert out.measurement_available.tolist()==[True,True]
    assert all(row['acquisition_bracket_elapsed_s']>.05 for row in out.views)
    assert all(row['within_timing_reference'] is False for row in out.views)
    assert all(row['maximum_pose_delta']==pytest.approx(.05) for row in out.views)


def test_fast_moving_camera_bracket_is_rejected_by_actual_pose():
    f=Fixture()
    values=iter((np.zeros(6),np.array([0,0,.251,0,0,0]),np.zeros(6)))
    with pytest.raises(RuntimeError,match="pose bracket is not stable"):
        capture_actual_views(f.camera,lambda:next(values),clock=f.clock)


@pytest.mark.parametrize("clock_values", [[10.,9.],[10.,np.nan],[10.,np.inf]])
def test_bad_clock_is_fatal_even_when_camera_delivery_fails(clock_values):
    f=Fixture();f.failures["wrist"]=TimeoutError()
    values=iter(clock_values)
    with pytest.raises(RuntimeError,match="clock"):
        capture_actual_views(f.camera,f.read_q,clock=lambda:next(values))


def test_inactive_views_no_camera_reads_and_no_stale_metadata():
    f=Fixture()
    out=f.capture(active_views=())
    assert not f.camera_calls and f.q_calls==1
    assert not out.measurement_available.any()
    assert all(row["reason"]=="view_not_active" for row in out.views)


def test_loss_after_success_does_not_relabel_previous_pixels_q_or_time_as_fresh():
    f=Fixture();old=f.capture()
    old_times=old.image_times_s.copy();old_q=old.q_at_image.copy();old_rgb=old.rgb.copy()
    f.failures["wrist"]=TimeoutError()
    new=f.capture()
    np.testing.assert_array_equal(old.image_times_s,old_times)
    np.testing.assert_array_equal(old.q_at_image,old_q)
    np.testing.assert_array_equal(old.rgb,old_rgb)
    assert new.image_times_s[0]==0 and not new.rgb[0].any() and not new.q_at_image[0].any()
    assert new.image_times_s[1]>old_times[1]


def test_raw_diagnostic_failures_are_not_swallowed_as_camera_unavailability():
    f=Fixture()
    with pytest.raises(RuntimeError,match="raw diagnostic"):
        f.capture(read_raw_frame=lambda view:np.zeros((480,640),np.uint8))


def test_cannot_relax_bracket_or_invent_view_role():
    f=Fixture()
    with pytest.raises(ValueError,match="50ms"):
        f.capture(maximum_bracket_s=.1)
    with pytest.raises(ValueError,match="0.25"):
        f.capture(maximum_pose_delta=.251)
    with pytest.raises(ValueError,match="active_views"):
        f.capture(active_views=("wrist","wrist"))


def test_camera_identity_is_role_bound_and_stable_during_each_frame():
    f=Fixture();resolved={"wrist":"/dev/video0","column":"/dev/video2"}
    def identity(view):
        return dict(schema='v21-camera-identity-v1',view=view,
                    stable_device='/dev/v4l/by-id/'+view,resolved_device=resolved[view])
    first=f.capture(camera_identity_check=identity)
    assert [row['camera_identity']['view'] for row in first.views]==['wrist','column']
    resolved['column']='/dev/video4'  # reconnect between observations is valid via stable alias
    second=f.capture(camera_identity_check=identity)
    assert second.views[1]['camera_identity']['resolved_device']=='/dev/video4'

    calls=[]
    def changes_during_frame(view):
        calls.append(view)
        node='/dev/video2' if len(calls)==1 else '/dev/video4'
        return dict(view=view,stable_device='/dev/v4l/by-id/'+view,resolved_device=node)
    with pytest.raises(RuntimeError,match="identity changed"):
        f.capture(active_views=('wrist',),camera_identity_check=changes_during_frame)


def test_camera_identity_mapping_cannot_swap_wrist_and_column_roles():
    f=Fixture()
    with pytest.raises(RuntimeError,match="identity was not verified"):
        f.capture(camera_identity_check=lambda view:dict(
            view='column' if view=='wrist' else 'wrist',stable_device='/stable',resolved_device='/resolved'))
    assert not f.camera_calls
