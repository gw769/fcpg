"""Research-only physical camera types, independent of immutable image slots.

These are reviewed sensor metadata, not model targets. Unknown is a fixed zero
condition; no episode/source/time identifier is supplied to the network.
"""
from copy import deepcopy

import numpy as np
import torch

LEGACY_SLOTS = 'legacy-slots'
REVIEWED_TYPES = 'reviewed-types-v1'
SCHEMA = 'v21-reviewed-camera-types-v1'
TYPE_IDS = {'unknown': 0, 'wrist': 1, 'external': 2}


def validate_camera_types(value, batch):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.long
            or value.shape != (batch, 2) or ((value < 0) | (value > 2)).any()):
        raise ValueError('camera types must be explicit int64 [B,2]: unknown=0,wrist=1,external=2')
    return value


def camera_mode_from_checkpoint(payload):
    mode = payload.get('lewm_camera_conditioning', LEGACY_SLOTS)
    if mode not in (LEGACY_SLOTS, REVIEWED_TYPES):
        raise ValueError('unknown camera conditioning checkpoint contract')
    has_weights = 'lewm.camera_type_embedding' in payload['model']
    if has_weights != (mode == REVIEWED_TYPES):
        raise ValueError('camera conditioning metadata and checkpoint weights disagree')
    if payload.get('native40_protocol', {}).get('lewm_camera_conditioning', mode) != mode:
        raise ValueError('camera conditioning metadata and saved protocol disagree')
    return mode


class ReviewedCameraTypes:
    """Exact episode binding only; never reorder, hide or invent a camera."""

    def __init__(self, contract, archive, paths):
        if (contract.get('schema') != SCHEMA or contract.get('purpose') != 'research-only'
                or contract.get('production_enabled') is not False
                or contract.get('unknown_policy') != 'explicit-zero-type-retain-all-data'
                or contract.get('slot_policy') != 'original-order-no-permutation'
                or contract.get('source_wide_mapping_verified') is not False):
            raise ValueError('explicit episode-scoped camera type contract required')
        self.contract = deepcopy(contract)
        self.endpoint_episode = np.asarray(archive.endpoint_episode_index)
        self.types_by_episode = np.zeros((len(archive.episodes), 2), np.int64)
        self.reviewed = np.zeros(len(archive.episodes), bool)
        for entry in contract['episodes']:
            ep = entry['canonical_episode']
            if type(ep) is not int or not 0 <= ep < len(archive.episodes) or self.reviewed[ep]:
                raise ValueError('duplicate or invalid reviewed camera episode')
            row = archive.episodes[ep]['manifest_row']
            roles = entry['physical_roles']
            if (row['dataset_root'] != entry['dataset_root']
                    or row['episode_index'] != entry['source_episode']
                    or list(reversed(row['lewm_v12_slot_views'])) != entry['historical_views']
                    or len(roles) != 2 or any(role not in TYPE_IDS for role in roles)
                    or not entry.get('reviewed_frames') or not entry.get('evidence')):
                raise ValueError('camera review identity, slot order or evidence differs')
            self.types_by_episode[ep] = [TYPE_IDS[role] for role in roles]
            self.reviewed[ep] = True
        fish = contract['fish']
        if (any(fish[key] != paths[key] for key in ('fish_wrist_root', 'fish_column_root'))
                or fish.get('physical_roles') != ['wrist', 'external']
                or fish.get('allowed_episodes') != [0, 1, 3] or not fish.get('evidence')):
            raise ValueError('fish camera type review must bind the actual paired source and sealed split')

    def for_endpoints(self, ids):
        ids = np.asarray(ids)
        if ids.dtype.kind not in 'iu' or (ids < 0).any() or (ids >= len(self.endpoint_episode)).any():
            raise ValueError('invalid camera type endpoint identity')
        return self.types_by_episode[self.endpoint_episode[ids]].copy()

    def for_chain(self, ids, kind, fish):
        ids = np.asarray(ids)
        if kind == 'fish':
            fish.validate_frames(ids.reshape(-1))
            if not np.isin(fish.episodes[ids], [0, 1, 3]).all():
                raise ValueError('sealed fish episode cannot receive training metadata')
            return np.tile([1, 2], (len(ids), 1)).astype(np.int64)
        ep = self.endpoint_episode[ids]
        if not (ep == ep[:, :1]).all():
            raise ValueError('camera type chain crosses an episode')
        return self.for_endpoints(ids[:, 0])

    def coverage(self, ids):
        types = self.for_endpoints(ids)
        ep = self.endpoint_episode[np.asarray(ids)]
        return dict(windows=int(ep.size), reviewed_windows=int(self.reviewed[ep].sum()),
                    episodes=int(np.unique(ep).size),
                    unknown_slots=int((types == 0).sum()), wrist_slots=int((types == 1).sum()),
                    external_slots=int((types == 2).sum()),
                    two_external_windows=int((types == 2).all(-1).sum()))
