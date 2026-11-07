"""The gripper rule (fcpg/v21_alignment_cem.GripLatch, training overhaul doc section 17.45): a hold squeezes at most
SQUEEZE_LIMIT below the grasp read-back, the Actor's own commands included (11/03: the release Actor closes to 0, and in
3 of the day's 4 holds the fish slipped out as the read-back sank 6-9 below the grasp's)."""
import numpy as np

from fcpg.v21_alignment_cem import CLOSED_GRIP, SQUEEZE, SQUEEZE_LIMIT, GripLatch


def test_the_actor_cannot_squeeze_the_held_fish_below_the_limit():
    latch = GripLatch()
    shut = np.zeros(5)                                          # the Actor closes all the way
    commands, event = latch.step(10.4, shut, True, False)
    assert event == 'grasp' and np.allclose(commands, 10.4 - SQUEEZE_LIMIT)
    commands, event = latch.step(8., shut, True, False)
    assert event == 'hold' and np.allclose(commands, 10.4 - SQUEEZE_LIMIT)
    # between the floor and the hold level (SQUEEZE below the read-back at the grasp) the Actor's own commands stand
    latch = GripLatch()
    commands, event = latch.step(10.4, np.full(5, 6.9), True, False)
    assert event == 'grasp' and 10.4 - SQUEEZE_LIMIT < 6.9 < 10.4 - SQUEEZE and np.allclose(commands, 6.9)


def test_a_hold_still_caps_an_opening_actor_and_lets_go_over_the_box():
    latch = GripLatch()
    latch.step(12., np.full(5, 11.), True, False)
    commands, event = latch.step(10., np.full(5, 30.), True, False)          # opening, no box: held at the level
    assert event == 'hold' and np.all(commands <= 12. - SQUEEZE + 1e-9) and np.all(commands >= 12. - SQUEEZE_LIMIT)
    commands, event = latch.step(9., np.full(5, 30.), True, True)            # over the box: released
    assert event == 'release' and commands[-1] >= 29.


def test_jaws_that_met_hold_nothing():
    latch = GripLatch()
    latch.step(10., np.zeros(5), True, False)
    commands, event = latch.step(CLOSED_GRIP - .5, np.zeros(5), True, False)
    assert event == 'empty' and commands[-1] > 20.
