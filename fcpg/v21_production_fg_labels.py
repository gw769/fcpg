"""Offline production F/G pseudo-label contract, separate from fish targets.

Wrist uses the deployed FK projection; Column uses current-image fused tips.
Neither is human-reviewed ground truth. Retained predictions/proposals keep
their own status, and never become a confirmed current-image label.
"""
import numpy as np

SCHEMA = 'v21-production-fg-pseudolabels-v1'
VIEWS = ('wrist', 'column')


def label_json_default(value):
    """Keep numerical production diagnostics numerical in the audit record."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError('unsupported production label value: '+type(value).__name__)


def pair(value):
    try:
        value = np.asarray(value, np.float32)
    except (TypeError, ValueError):
        return None
    return value if value.shape == (2, 2) and np.isfinite(value).all() else None


def label_record(tips, camera_valid):
    """Preserve raw coordinates; distinguish estimate, visual support and order."""
    result = {}
    for c, view in enumerate(VIEWS):
        rec = tips.get(view, {})
        current = bool(camera_valid[c])
        xy = pair(rec.get('fk_fm_px')) if view == 'wrist' else pair(rec.get('fm_px'))
        if view == 'wrist':
            source = 'wrist-fk-projection'
            available = xy is not None and rec.get('source') == source
            ordered, visual = available, False
        else:
            if rec.get('source') == 'v2032-current-image-actual-readback-posterior':
                source = rec['source']
                available = bool(current and rec.get('available') and xy is not None
                    and rec.get('observed') is True and rec.get('current_image_valid') is True
                    and rec.get('spatial_update_valid') is True and rec.get('nominal_conflict') is False)
            else:
                available = bool(current and rec.get('available') and xy is not None
                                 and rec.get('source') == 'column-rgb-jaw-contour')
                dino = bool((rec.get('visual_diagnostics') or {}).get('dino_fg_trusted'))
                source = 'column-dino-rgb-q-history-fusion' if dino else 'column-rgb-q-history-fusion'
            ordered = bool(available and rec.get('ordered_identity_available'))
            visual = available
        if not available:
            xy = None
        in_view = (False if xy is None else bool((xy >= [0, 28]).all() and (xy < [224, 196]).all()))
        result[view] = dict(fg_px=None if xy is None else xy.tolist(), pair_available=available,
            ordered_identity_available=ordered, current_visual_supported=visual,
            current_image_valid=current, in_view=in_view, source=source,
            supervision_candidate=bool(available and current), human_verified=False,
            fk_based=view == 'wrist', estimated=True,
            retained_or_display_pair=rec.get('display_fm_px') or rec.get('suggested_fm_px'),
            retained_used_as_current_label=False, reason=rec.get('reason'), production_record=rec)
    return result


def history_samples(times, q, actions, now):
    """Causal10Hz summaries of up to3 past recorded0.5s windows.

These are recording summaries, not fabricated live execution receipts. The
production observer only consumes each window's start/end, last command and dt.
No future readback, action or other episode is requested.
"""
    times, q, actions = np.asarray(times), np.asarray(q), np.asarray(actions)
    if (times.ndim != 1 or len(times) == 0 or (np.diff(times) <= 0).any()
            or q.shape != (len(times), 6) or actions.shape != q.shape or not np.isfinite(now)):
        raise ValueError('one ordered actual episode with pairedq/actions required')
    command = np.zeros((1, 3, 5, 6), np.float32)
    readback = np.zeros((1, 3, 6, 6), np.float32)
    dt = np.zeros((1, 3, 5), np.float32)
    valid = np.zeros((1, 3), bool)
    ids = []
    for slot in range(3):
        requested = now - (3-slot)*.5 + np.arange(6)*.1
        indices = np.searchsorted(times, requested+2e-6, side='right')-1
        if (indices < 0).any() or (np.diff(indices) <= 0).any():
            ids.append(None)
            continue
        if (times[indices] > now+2e-6).any():
            raise ValueError('future sample in observer history')
        command[0, slot] = actions[indices[:-1]]
        readback[0, slot] = q[indices]
        dt[0, slot] = np.diff(times[indices])
        valid[0, slot] = True
        ids.append(indices.tolist())
    if (valid[:, :-1] & ~valid[:, 1:]).any():
        raise ValueError('history must be right aligned')
    return command, readback, dt, valid, ids


class ProductionFGTable:
    """Explicit reader; absent/incomplete/wrong-clock tables never fall back."""
    def __init__(self, root, *, clock_version, extraction_id):
        import json
        from pathlib import Path
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        m = self.manifest
        if (m.get('schema') != SCHEMA or m.get('complete') is not True
                or m.get('clock_version') != clock_version or m.get('extraction_id') != extraction_id
                or m.get('episodes') != [0, 1, 3] or m.get('episode_2_payload_read') is not False):
            raise ValueError('complete same-clock production F/G table required')
        self.ids = np.load(self.root/'endpoint_ids.npy', mmap_mode='r')
        if self.ids.ndim != 1 or (np.diff(self.ids) <= 0).any() or len(self.ids) != m['rows']:
            raise ValueError('unordered or incomplete F/G identity table')
        self.arrays = {k: np.load(self.root/(k+'.npy'), mmap_mode='r') for k in
            ('fg_px', 'pair_available', 'ordered_identity_available', 'current_visual_supported', 'current_image_valid')}
        for key, values in self.arrays.items():
            expected = (len(self.ids), 2, 2, 2) if key == 'fg_px' else (len(self.ids), 2)
            if values.shape != expected or (key != 'fg_px' and values.dtype != np.bool_):
                raise ValueError('F/G array contract differs')

    def gather(self, endpoint_ids):
        requested = np.asarray(endpoint_ids)
        if requested.dtype.kind not in 'iu' or not requested.size:
            raise ValueError('explicit integer endpoint identities required')
        indices = np.searchsorted(self.ids, requested)
        if (indices >= len(self.ids)).any() or not np.array_equal(self.ids[indices], requested):
            raise ValueError('unprocessed/sealed F/G endpoint')
        return {k: np.array(v[indices], copy=True) for k, v in self.arrays.items()}

    def require_training_admission(self):
        """A finished export is not evidence that its production teacher is right."""
        import json
        path = self.root/'quality-review.json'
        if not path.exists():
            raise ValueError('F/G label quality not reviewed; completed export permits canary only')
        review = json.loads(path.read_text())
        if (review.get('accepted_for_training') is not True
                or review.get('rows') != len(self.ids)
                or review.get('source_snapshot') != self.manifest['source_snapshot']):
            raise ValueError('F/G production labels failed quality review; formal training blocked')
