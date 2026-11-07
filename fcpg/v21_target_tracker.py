"""View-specific registered DINO references; no online template replacement.

The drop zone is registered in the column view only. In the wrist it is matched with its role prototype from that
column registration (audit doc V21_SYSTEM_LEWM_CEM_AUDIT_20261013.md section 14.14); V21_TARGET_SITES asks the service's
target localization for it. 14.14 found every wrist picture with the box at 0.39-0.59 against the 0.35 floor and every
one without it at 0.30-0.32 on the 10/16 recordings it checked; over all 171 jaws-shut blocks after the grasp in the
10/16 control sessions that prototype put 42 of its 109 reports off the box, most on the jaws (training overhaul doc
V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md section 17.6).

A wrist-view registration of the drop zone replaces that role prototype in the wrist when one is given
(wrist_drop_zone=, a file from scripts/register_v21_wrist_drop_zone.py: one crop of the box from one wrist picture, as
an operator would click it; on the same 171 blocks 1 of its 63 reports was off the box). It is matched with the same
frozen grounder, jaw pixels and visibility floor, and stays out of the persisted profile, whose drop zone remains
column-only for every other service and for fcpg/v21_object_relative.py's place token."""
import hashlib
import io
from pathlib import Path

import numpy as np

from .prod_v17_7_dino_targets import TARGET_SITES
from .so101_visual_targets import VisualTargetTracker

V21_TARGET_SITES = TARGET_SITES + (('drop_zone', 'wrist'),)
WRIST_DROP_ZONE_SCHEMA = 'v21-wrist-drop-zone-registration-v1'
WRIST_DROP_ZONE_SOURCE = 'frozen_vit_features_wrist_registration'


def load_wrist_drop_zone(path):
    """A wrist-view drop-zone registration: dict(template_rgb [h, w, 3] uint8, centre_px, radius_px, source, path,
    sha256). Raises on anything that is not one."""
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        if 'schema' not in data.files or str(data['schema']) != WRIST_DROP_ZONE_SCHEMA:
            raise ValueError(f'{path} is not a {WRIST_DROP_ZONE_SCHEMA} file')
        template = np.asarray(data['template_rgb'])
        centre = np.asarray(data['centre_px'], np.float64)
        radius = float(data['radius_px'])
        source = str(data['source'])
    if template.dtype != np.uint8 or template.ndim != 3 or template.shape[2] != 3 or min(template.shape[:2]) < 12:
        raise ValueError('the wrist drop-zone template must be an RGB uint8 crop at least 12 px on a side')
    if centre.shape != (2,) or not np.isfinite(centre).all() or not np.isfinite(radius) or radius < 6:
        raise ValueError('the wrist drop-zone registration needs a finite centre and a radius of at least 6 px')
    return dict(template_rgb=template, centre_px=centre.tolist(), radius_px=radius, source=source, path=str(path),
                sha256=hashlib.sha256(raw).hexdigest())


class V21TargetTracker(VisualTargetTracker):
    def __init__(self, *args, wrist_drop_zone=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrist_drop_zone = None if wrist_drop_zone is None else load_wrist_drop_zone(wrist_drop_zone)
        self._wrist_drop_zone_prototype = None

    def locate_by_frozen_features(self, role, view, frame, **kwargs):
        registration = getattr(self, 'wrist_drop_zone', None)
        if role == 'drop_zone' and view == 'wrist' and registration is not None:
            if getattr(self, '_wrist_drop_zone_prototype', None) is None:
                self._wrist_drop_zone_prototype = self.frozen_grounder.prototype_from_crop(registration['template_rgb'])
            result = super().locate_by_frozen_features(role, view, frame, prototype=self._wrist_drop_zone_prototype,
                                                       **kwargs)
            result.update(source=WRIST_DROP_ZONE_SOURCE, prototype_source='wrist_registration',
                          registration_sha256=registration['sha256'])
            return result
        kwargs['view_specific_prototype'] = not (role == 'drop_zone' and (role, view) not in self.references)
        return super().locate_by_frozen_features(role, view, frame, **kwargs)
