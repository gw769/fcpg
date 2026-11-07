"""Archive camera layout for FCPG/Actor training (docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md, step A).

The model reads slot 0 as the jaw-mounted wrist camera and slot 1 as an external camera, as on the fish robot.
Many Archive recordings do not: some have the two cameras the other way round, some have two external cameras and
no wrist camera at all. A per-episode table, measured from the frozen DINO features and checked against every
human-reviewed episode (scripts/audit_v21_archive_camera_layout.py, scripts/build_v21_archive_camera_layout.py), says for each
canonical episode: the slot order, whether a wrist camera exists, and whether the layout could be decided.

This adapter presents every Archive exposure packet (DINO tokens, validity, capture time, capture q) in the
model's order. An episode without a wrist camera keeps its external view in slot 1 and has no slot-0 view
(invalid, zero features, as for any missing camera). An episode with a wrist camera only (tables from 17.68 on:
``external_present``; the other slot of the Pi0.5 collection is a black video there) keeps its wrist view in slot 0
and has no slot-1 view, the same way. Rows of undecided episodes are not admitted. Raw caches,
recordings and fish data are untouched. It plugs into V21Sources through the same two hooks as the reviewed role
adapter (fcpg/v21_archive_camera_roles.py): ``archive_camera_roles`` and a role-aware recorded clock.
"""
import hashlib
from pathlib import Path

from .v21_encoder import DIM, PATCHES
import numpy as np
import torch

SCHEMA = 'v21-archive-camera-layout-v1'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class ArchiveCameraLayout:
    def __init__(self, path, archive):
        path = Path(path)
        with np.load(path) as data:
            self.order = np.asarray(data['order'], np.int64)
            self.wrist_present = np.asarray(data['wrist_present'], bool)
            self.decided = np.asarray(data['decided'], bool)
            self.external_present = (np.asarray(data['external_present'], bool) if 'external_present' in data.files
                                     else np.ones(self.decided.shape, bool))
        episodes = len(archive.episodes)
        if (self.order.shape != (episodes, 2) or self.wrist_present.shape != (episodes,)
                or self.decided.shape != (episodes,) or self.external_present.shape != (episodes,)
                or not np.isin(self.order, [0, 1]).all() or (self.order.sum(1) != 1).any()):
            raise ValueError('camera layout table does not match the Archive episodes')
        if ((self.order[~self.decided] != [0, 1]).any() or self.wrist_present[~self.decided].any()
                or not self.external_present[~self.decided].all()):
            raise ValueError('undecided episodes must keep the historical order and claim no wrist camera')
        if (self.decided & ~self.wrist_present & ~self.external_present).any():
            raise ValueError('a decided episode needs at least one camera')
        self.endpoint_episode = np.asarray(archive.endpoint_episode_index)
        self.identity = dict(schema=SCHEMA, table=str(path), table_sha256=_sha256(path))

    def _episodes(self, ids):
        ids = np.asarray(ids)
        if ids.dtype.kind not in 'iu' or (ids < 0).any() or (ids >= len(self.endpoint_episode)).any():
            raise ValueError('invalid original endpoint identity')
        return self.endpoint_episode[ids]

    def orders(self, ids):
        return self.order[self._episodes(ids)]

    def admitted_rows(self, archive, rows, *, wrist=False):
        """Archive chain rows whose episode layout is decided; wrist=True: and that have a wrist camera."""
        rows = np.asarray(rows, np.int64)
        episode = self.endpoint_episode[np.asarray(archive.row_endpoint_index)[rows, 0]]
        return rows[self.decided[episode] & (self.wrist_present[episode] | (not wrist))]

    def row_collections(self, archive, rows):
        """(names, index per row): the collection a row was recorded in, i.e. the folder directly under the
        Archive root of its dataset; the many IRA_* session folders of one laboratory count as one collection."""
        def collection(root):
            parts = [part for part in str(root).split('/') if part]
            name = parts[parts.index('raw_expand') + 1] if 'raw_expand' in parts[:-1] else (parts[-1] if parts else '')
            return 'IRA' if name.startswith('IRA_') else name
        names = [collection(episode['manifest_row'].get('dataset_root')) for episode in archive.episodes]
        ordered = sorted(set(names))
        index = np.asarray([ordered.index(name) for name in names], np.int64)
        rows = np.asarray(rows, np.int64)
        return ordered, index[self.endpoint_episode[np.asarray(archive.row_endpoint_index)[rows, 0]]]

    def remap(self, value, ids):
        """Camera axis follows the endpoint-id dimensions: [..., 2, ...]. Tokens and validity of a missing wrist
        or external camera become zero / False; capture times and q keep their (unused) values."""
        ids = np.asarray(ids)
        episode = self._episodes(ids)
        orders = self.order[episode]
        missing = ~self.wrist_present[episode] & self.decided[episode]
        no_external = ~self.external_present[episode] & self.decided[episode]
        axis = ids.ndim
        if tuple(value.shape[:axis+1]) != (*ids.shape, 2):
            raise ValueError('camera packet and endpoint identities differ')
        tokens = tuple(value.shape[axis+1:]) == (PATCHES, DIM)
        boolean = value.dtype in (torch.bool, np.bool_) if isinstance(value, torch.Tensor) else value.dtype == np.bool_
        shape = (*orders.shape, *([1] * (value.ndim - axis - 1)))
        if isinstance(value, torch.Tensor):
            out = value
            if not np.array_equal(orders, np.broadcast_to([0, 1], orders.shape)):
                index = torch.as_tensor(orders, device=value.device).reshape(shape).expand_as(value)
                out = value.gather(axis, index)
            if (missing.any() or no_external.any()) and (tokens or boolean):
                keep = torch.ones(*ids.shape, 2, device=value.device, dtype=torch.bool)
                keep[..., 0] = ~torch.as_tensor(missing, device=value.device)
                keep[..., 1] = ~torch.as_tensor(no_external, device=value.device)
                keep = keep.reshape(*ids.shape, 2, *([1] * (value.ndim - axis - 1)))
                out = torch.where(keep, out, torch.zeros_like(out))
            return out
        out = value
        if not np.array_equal(orders, np.broadcast_to([0, 1], orders.shape)):
            out = np.take_along_axis(value, np.broadcast_to(orders.reshape(shape), value.shape), axis=axis)
        if (missing.any() or no_external.any()) and (tokens or boolean):
            out = np.array(out, copy=True)
            out[(*np.nonzero(missing), 0)] = 0
            out[(*np.nonzero(no_external), 1)] = 0
        return out

    def coverage(self, archive, rows):
        rows = np.asarray(rows, np.int64)
        episode = self.endpoint_episode[np.asarray(archive.row_endpoint_index)[rows, 0]]
        decided = self.decided[episode]
        return dict(rows=int(len(rows)), decided=int(decided.sum()),
                    swapped=int((decided & (self.order[episode, 0] == 1)).sum()),
                    no_wrist=int((decided & ~self.wrist_present[episode]).sum()),
                    undecided=int((~decided).sum()),
                    **({} if self.external_present.all() else
                       dict(no_external=int((decided & ~self.external_present[episode]).sum()))))


class LayoutClock:
    """Delegate raw clock tables unchanged; Archive exposure packets in the model's camera order."""
    def __init__(self, raw, layout):
        self.raw, self.layout = raw, layout

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def observations(self, ids, kind):
        values = self.raw.observations(ids, kind)
        return values if kind == 'fish' else tuple(self.layout.remap(v, ids) for v in values)


def install(source, path):
    """Install on a V21Sources before any feature bank or staged copy is made; returns the layout."""
    if getattr(source, 'archive_camera_roles', None) is not None or getattr(source, 'camera_types', None) is not None:
        raise ValueError('one camera contract per source')
    if getattr(source, 'recorded_clock', None) is None:
        raise ValueError('the camera layout needs the recorded capture clocks')
    if getattr(source, 'archive_object_relative_table', None) is not None:
        raise ValueError('Archive object cues are wrist-only and cannot be remapped')
    layout = ArchiveCameraLayout(path, source.archive)
    source.archive_camera_roles = layout
    source.recorded_clock = LayoutClock(source.recorded_clock, layout)
    return layout
