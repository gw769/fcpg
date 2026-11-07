"""The drop zone's wrist-view registration (fcpg/v21_target_tracker.py, training overhaul doc section 17.6)."""
from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.prod_v17_7_dino_targets import locate_dino_targets
from fcpg.so101_visual_targets import VisualTargetTracker
from fcpg.v21_target_tracker import (V21_TARGET_SITES, WRIST_DROP_ZONE_SCHEMA, WRIST_DROP_ZONE_SOURCE,
                                     V21TargetTracker, load_wrist_drop_zone)


def write_registration(path, *, schema=WRIST_DROP_ZONE_SCHEMA, side=56):
    template = np.full((side, side, 3), 7, np.uint8)
    np.savez(path, schema=np.asarray(schema), template_rgb=template, centre_px=np.asarray((60., 100.)),
             radius_px=np.asarray(28.), source=np.asarray('demo 0 frame 449'))
    return path


class Grounder:
    """Records which prototype each locate used; a visible hit for the wrist registration, a miss otherwise."""

    def __init__(self):
        self.crops, self.located = [], []

    def prototype_from_crop(self, crop):
        self.crops.append(np.asarray(crop).copy())
        return ('crop', int(np.asarray(crop)[0, 0, 0]))

    def locate(self, frame, prototype, **kwargs):
        self.located.append(prototype)
        if prototype == ('crop', 7):
            mask = np.zeros(frame.shape[:2], bool)
            mask[90:110, 50:70] = True
            return SimpleNamespace(peak=.5, separation=3., centroid_px=(60., 100.), support_mask=mask,
                                   symmetry_iou=.8, shape=SimpleNamespace(axis_px=(1., 0.), grasp_axis_px=(0., 1.),
                                                                          width_px=20., length_px=20., elongation=1.))
        return SimpleNamespace(peak=.1, separation=0.)


def tracker_with(grounder, column_crop_value=3, **kwargs):
    tracker = V21TargetTracker(None, **kwargs)
    tracker._frozen_grounder = grounder
    for role, view in (('pickup', 'wrist'), ('pickup', 'column'), ('drop_zone', 'column')):
        tracker.references[(role, view)] = SimpleNamespace(
            template_rgb=np.full((20, 20, 3), column_crop_value if role == 'drop_zone' else 1, np.uint8))
    return tracker


def test_the_registration_file_loads_and_refuses_anything_else(tmp_path):
    loaded = load_wrist_drop_zone(write_registration(tmp_path / 'box.npz'))
    assert loaded['template_rgb'].shape == (56, 56, 3) and loaded['radius_px'] == 28.
    assert loaded['centre_px'] == [60., 100.] and len(loaded['sha256']) == 64
    with pytest.raises(ValueError, match='is not a'):
        load_wrist_drop_zone(write_registration(tmp_path / 'other.npz', schema='something-else'))
    with pytest.raises(ValueError, match='at least 12 px'):
        load_wrist_drop_zone(write_registration(tmp_path / 'small.npz', side=8))


def test_the_wrist_registration_replaces_the_role_prototype_in_the_wrist_only(tmp_path):
    grounder = Grounder()
    tracker = tracker_with(grounder, wrist_drop_zone=write_registration(tmp_path / 'box.npz'))
    frame = np.zeros((224, 224, 3), np.uint8)
    found = tracker.locate_by_frozen_features('drop_zone', 'wrist', frame, jaw_pair_px=None)
    assert found['visible'] is True and found['source'] == WRIST_DROP_ZONE_SOURCE
    assert found['prototype_source'] == 'wrist_registration'
    assert found['registration_sha256'] == tracker.wrist_drop_zone['sha256']
    tracker.locate_by_frozen_features('drop_zone', 'wrist', frame)
    tracker.locate_by_frozen_features('drop_zone', 'column', frame)
    assert grounder.located == [('crop', 7), ('crop', 7), ('crop', 3)]
    assert [int(crop[0, 0, 0]) for crop in grounder.crops] == [7, 3]   # the wrist prototype is built once


def test_without_a_registration_the_wrist_keeps_the_column_role_prototype():
    grounder = Grounder()
    tracker = tracker_with(grounder)
    assert tracker.wrist_drop_zone is None
    found = tracker.locate_by_frozen_features('drop_zone', 'wrist', np.zeros((224, 224, 3), np.uint8))
    assert found['visible'] is False and found['reason'] == 'peak_below_threshold'
    assert grounder.located == [('crop', 3)]


def test_the_service_targets_carry_the_wrist_registration_source(tmp_path):
    tracker = tracker_with(Grounder(), wrist_drop_zone=write_registration(tmp_path / 'box.npz'))
    frame = np.zeros((224, 224, 3), np.uint8)
    payload = locate_dino_targets({'wrist': frame, 'column': frame}, tracker=tracker, sites=V21_TARGET_SITES)
    wrist = payload['targets']['drop_zone']['wrist']
    assert wrist['visible'] is True and wrist['source'] == WRIST_DROP_ZONE_SOURCE
    assert wrist['cog_px'] == [60., 100.]
    assert payload['targets']['drop_zone']['column']['visible'] is False


def test_an_explicit_prototype_excludes_the_view_specific_one_and_the_profile_stays_column_only():
    tracker = VisualTargetTracker.__new__(VisualTargetTracker)
    tracker.references, tracker.frozen_prototypes, tracker.frozen_prototype_source = {}, {}, {}
    frame = np.zeros((32, 32, 3), np.uint8)
    with pytest.raises(ValueError, match='excludes'):
        tracker.locate_by_frozen_features('pickup', 'wrist', frame, view_specific_prototype=True, prototype='p')
    with pytest.raises(ValueError, match='column view'):
        VisualTargetTracker(None).register('drop_zone', 'wrist', np.zeros((224, 224, 3), np.uint8),
                                           np.asarray((60., 100.)), radius_px=28., persist=False)
