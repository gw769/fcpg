from types import SimpleNamespace

import numpy as np

from fcpg.so101_visual_targets import VisualTargetTracker, FROZEN_FEATURE_MIN_PEAK
from fcpg.prod_v17_7_dino_targets import locate_dino_targets
from fcpg.v21_target_tracker import V21TargetTracker


def test_v21_uses_only_matching_view_and_caches_reference():
    calls = []
    class Tracker(V21TargetTracker):
        def __init__(self):
            self.references = {('pickup', v): SimpleNamespace(template_rgb=v)
                               for v in ('wrist', 'column')}
        @property
        def frozen_grounder(self):
            def encode(crop):
                calls.append(crop)
                return crop
            return SimpleNamespace(prototype_from_crop=encode,
                locate=lambda frame, prototype, **kw: SimpleNamespace(
                    peak=0.1 if prototype == 'wrist' else 0.2, separation=1.))
    t=Tracker(); frame=np.zeros((224,224,3),np.uint8)
    assert t.locate_by_frozen_features('pickup','wrist',frame)['score'] == 0.1
    assert t.locate_by_frozen_features('pickup','column',frame)['score'] == 0.2
    t.locate_by_frozen_features('pickup','wrist',frame)
    assert calls == ['wrist','column']
    del t.references[('pickup','wrist')]
    assert t.locate_by_frozen_features('pickup','wrist',frame)['reason'] == 'target_reference_not_configured'


def test_rejected_score_survives_tracker_and_display_without_coordinates():
    peak = FROZEN_FEATURE_MIN_PEAK - 0.01

    class Tracker(VisualTargetTracker):
        def __init__(self):
            pass

        def frozen_prototype(self, role):
            return object()

        @property
        def frozen_grounder(self):
            return SimpleNamespace(locate=lambda *a, **kw: SimpleNamespace(
                peak=peak, separation=2.1))

    frame = np.zeros((224, 224, 3), np.uint8)
    payload = locate_dino_targets({'wrist': frame, 'column': frame}, tracker=Tracker())
    for views in payload['targets'].values():
        for record in views.values():
            assert record['visible'] is False
            assert record['centroid_px'] is None
            assert record['axis_px'] is None
            assert record['score'] == peak
            assert record['separation'] == 2.1
            assert record['reason'] == 'peak_below_threshold'
