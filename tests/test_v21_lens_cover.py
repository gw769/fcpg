"""Paper-over-lens hold. No camera and no robot."""
import numpy as np

from fcpg.v21_lens_cover import PAPER_RETENTION_S, PaperCoverHold, frame_is_covered


def test_flat_paper_is_covered_and_a_textured_frame_is_not():
    paper = np.full((224, 224, 3), 210, np.uint8)
    scene = np.zeros((224, 224, 3), np.uint8)
    scene[::8] = 255
    scene[:, ::8] = 40
    assert frame_is_covered(paper)
    assert not frame_is_covered(scene)


def test_cover_keeps_the_last_real_frame_for_two_seconds():
    hold = PaperCoverHold()
    real = np.full((224, 224, 3), 30, np.uint8)
    real[0, 0] = 255
    first = hold.update("column", real, False, 10.0)
    assert first["covered"] is False
    inside = hold.update("column", np.full_like(real, 200), True, 11.5)
    assert inside["retained_real_frame"] and inside["within_paper_window"]
    assert inside["age_s"] == 1.5
    assert np.array_equal(hold.retained_rgb("column"), real)
    outside = hold.update("column", np.full_like(real, 200), True, 10.0 + PAPER_RETENTION_S + 0.01)
    assert outside["retained_real_frame"] and not outside["within_paper_window"]


def test_cover_without_a_prior_real_frame_does_not_invent_one():
    hold = PaperCoverHold()
    decision = hold.update("wrist", np.full((224, 224, 3), 180, np.uint8), True, 1.0)
    assert decision["covered"] and not decision["retained_real_frame"]
    assert hold.retained_rgb("wrist") is None
