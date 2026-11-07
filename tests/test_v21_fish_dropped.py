"""A fish dropped during the carry (fcpg/v21_alignment_cem.AlignmentCEMAdapter, 11/04 sam31hb trial 1: it slipped out of
the jaws 132 mm up and fell into the box; the kept whole fish stayed where it had lain on the table, its outlines in the
box measured 100-108 mm against the kept 72 and were not taken, and the arm searched the empty table to the end). A hold
found empty more than DROPPED_ABOVE_MM above the table forgets the kept fish, and the next outline sets it again; open
jaws holding nothing with the fish an outline showed after the last grasp inside the box solid's footprint end the
trial (released_over_box by column_fish_in_box)."""
import numpy as np
import pytest

from fcpg.v21_alignment_cem import DROPPED_ABOVE_MM, AlignmentCEMAdapter, BoxSolid
from test_v21_alignment_roll import Service, camera, cem, outline

MM = .985 / 224                                            # m of the table per letterbox224 pixel at the fish (camera())


def adapter():
    a = AlignmentCEMAdapter.__new__(AlignmentCEMAdapter)
    a.planner, a.column_camera, a.clock = cem(), camera(), iter(range(100)).__next__
    a.service, a.fish_axis, a.fish_length_m, a.grasp_s = Service(), None, None, None
    return a


def test_a_hold_lost_above_the_table_forgets_the_kept_fish():
    a = adapter()
    a.service.show(outline(cog=(112., 112.), length=16.))                    # on the table: 70 mm
    a._fish_axis()
    a.service.show(outline(cog=(160., 112.), length=24.))                    # in the box, partly hidden: 105 mm
    assert a._fish_axis()[1]['outline'] == 'length_disagrees'
    assert np.allclose(a._column_fish_point()[:2], 0., atol=1e-9)           # still led to the table
    assert a._forget_grasp('empty', DROPPED_ABOVE_MM - 10.) is False and a.fish_axis is not None   # at the table: kept
    assert a._forget_grasp('empty', 166.) is True and a.fish_axis is None and a.fish_length_m is None
    assert a._fish_axis()[1]['outline'] == 'first'                            # the next outline sets it again
    assert a._column_fish_point()[0] == pytest.approx(48 * MM)
    assert a._forget_grasp('hold', 166.) is False and a.fish_axis is not None


def test_the_fish_in_the_box_after_a_grasp():
    a = adapter()
    a.planner.box_solid = BoxSolid(centre=(48 * MM, 0.), turn=0., length=.10, width=.10, height=.07, overlap=1., cells=9)
    a.service.show(outline(cog=(160., 112.), length=16.))                    # where the box is, before any grasp
    a._fish_axis()
    assert a._fish_in_box() is None                                          # no grasp yet: nothing to say
    a.grasp_s = float(a.clock())
    assert a._fish_in_box() is None                                          # the outline is older than the grasp
    a._fish_axis()
    assert a._fish_in_box() == pytest.approx(50.)                            # an outline since: at the box's centre
    a.service.show(outline(cog=(112., 112.), length=16.))                    # back on the table, 211 mm away
    a._fish_axis()
    assert a._fish_in_box() < 0.
