"""View-local overlay contour — no live_service / Unix serve imports."""
from fcpg.v21_overlay_geometry import same_view_object_contour


def test_column_keeps_its_own_contour():
    rec = {
        "visible": True,
        "view": "column",
        "contour_px": [[140, 80], [180, 100], [140, 120], [120, 100]],
    }
    out = same_view_object_contour(rec, "column", (224, 224))
    assert out == [[140.0, 80.0], [180.0, 100.0], [140.0, 120.0], [120.0, 100.0]]


def test_column_refuses_wrist_tagged_record():
    rec = {
        "visible": True,
        "view": "wrist",
        "contour_px": [[140, 80], [180, 100], [140, 120], [120, 100]],
    }
    assert same_view_object_contour(rec, "column", (224, 224)) == []


def test_out_of_frame_points_are_dropped():
    rec = {
        "visible": True,
        "view": "column",
        "contour_px": [[10, 10], [300, 10], [10, 20]],  # one off-frame
    }
    # After drop, only two points remain → not enough for a polygon.
    assert same_view_object_contour(rec, "column", (224, 224)) == []
