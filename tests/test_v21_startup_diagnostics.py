"""CPU-only startup diagnostics; never opens a robot, camera, or ROS node."""
import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

from scripts.canary_so101_v21_research import camera_startup_diagnostic
from scripts.serve_so101_v21_research import make_server, startup_failure_payload


def test_camera_timeout_preserves_failed_view_and_transport_evidence():
    error = TimeoutError(
        "camera 'column' did not recover after 3 reconnect attempts; "
        "last error: Timed out waiting for frame (VIDIOC_DQBUF)"
    )
    io = SimpleNamespace(
        camera_last_errors={"wrist": None, "column": "VIDIOC_DQBUF timeout"},
        camera_reconnect_counts={"wrist": 0, "column": 3},
    )
    diagnostic = camera_startup_diagnostic(
        error, io=io, active_views=("wrist", "column"), stage="initial_observation"
    )
    assert diagnostic["schema"] == "v21-camera-startup-diagnostic-v1"
    assert diagnostic["failure_reason"] == "v4l2_dqbuf_error"
    assert diagnostic["failed_views"] == ["column"]
    column = next(row for row in diagnostic["cameras"] if row["view"] == "column")
    assert column["reconnect_attempts"] == 3
    assert column["last_error"] == "VIDIOC_DQBUF timeout"
    json.dumps(diagnostic, allow_nan=False)


def test_non_camera_failure_does_not_get_a_camera_diagnosis():
    assert camera_startup_diagnostic(RuntimeError("checkpoint is missing")) is None


def test_failed_startup_status_exposes_structured_diagnostic_without_control():
    error = TimeoutError("camera 'column' timed out")
    error.camera_startup_diagnostic = camera_startup_diagnostic(
        error, active_views=("wrist", "column"), stage="hardware_open"
    )
    failure = startup_failure_payload(error)
    assert failure["schema"] == "v21-startup-failure-v1"
    assert failure["motion_authorized"] is False
    assert failure["camera"]["failed_views"] == ["column"]

    server = make_server(("127.0.0.1", 0), {
        "startup_error": failure["error"],
        "startup_diagnostic": failure,
    })
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        try:
            urlopen(f"http://127.0.0.1:{server.server_port}/api/status")
        except HTTPError as response:
            assert response.code == 503
            payload = json.loads(response.read())
        else:
            raise AssertionError("failed startup unexpectedly returned HTTP 200")
        assert payload["connected"] is False
        assert payload["startup_diagnostic"]["camera"]["failure_reason"] == "camera_timeout"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
