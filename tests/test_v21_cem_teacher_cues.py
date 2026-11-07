"""The CEM teacher on the outline cue tables (scripts/build_v21_cem_teacher_labels.py, 11/03): the state's 'hold' fish cue
is moved back to the jaw midpoint for the CEM, as the robot's AlignmentCEMAdapter._jaw_layout does."""
import numpy as np

from fcpg.v21_alignment_cem import IMAGE
from scripts.build_v21_cem_teacher_labels import jaw_layout


def test_the_hold_cue_is_moved_back_to_the_jaw_midpoint():
    cue = np.asarray([1., 1., .1, -.05, .3, .1, 1., 0.])
    hold, midpoint = np.asarray([118., 120.]), np.asarray([112., 150.])
    out = jaw_layout(cue, np.zeros(6), hold, lambda q: midpoint)
    # the same fish centre in the picture: hold point + cue offset == jaw midpoint + its offset
    assert np.allclose(hold + cue[2:4] * IMAGE, midpoint + out[2:4] * IMAGE)
    assert np.array_equal(out[[0, 1, 4, 5, 6, 7]], cue[[0, 1, 4, 5, 6, 7]])
    assert np.array_equal(cue[2:4], [.1, -.05])                     # the state's cue is left as it is
