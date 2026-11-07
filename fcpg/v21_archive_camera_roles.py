"""Opt-in, reviewed-episode camera packets. Raw caches stay in historical order.

Only research loaders use this adapter; unknown episodes retain their original
order and remain UNVERIFIED. There is no source-wide extrapolation or robot I/O.
"""
from copy import deepcopy

import numpy as np
import torch

SCHEMA = 'v21-reviewed-archive-role-adapter-v1'


def research_adapter(review):
    if (review.get('schema') != 'v21-archive-camera-role-review-v1'
            or review.get('purpose') != 'diagnostic-only'
            or review.get('scope') != 'reviewed-canonical-episodes-only'
            or review.get('production_enabled') is not False
            or review.get('source_wide_mapping_verified') is not False):
        raise ValueError('explicit episode-scoped visual review required')
    return dict(schema=SCHEMA, purpose='research-only', production_enabled=False,
                unknown_policy='retain-historical-unverified', review=deepcopy(review))


class ReviewedArchiveRoles:
    def __init__(self, contract, archive):
        if (contract.get('schema') != SCHEMA or contract.get('purpose') != 'research-only'
                or contract.get('production_enabled') is not False
                or contract.get('unknown_policy') != 'retain-historical-unverified'):
            raise ValueError('explicit research camera role contract required')
        review = contract['review']
        if research_adapter(review) != contract:
            raise ValueError('camera role contract contains unrecognized changes')
        self.contract = deepcopy(contract)
        self.endpoint_episode = np.asarray(archive.endpoint_episode_index)
        self.orders_by_episode = np.tile([0, 1], (len(archive.episodes), 1))
        self.reviewed = np.zeros(len(archive.episodes), bool)
        for entry in review['episodes']:
            ep = entry['canonical_episode']
            if type(ep) is not int or not 0 <= ep < len(archive.episodes) or self.reviewed[ep]:
                raise ValueError('duplicate or invalid reviewed episode')
            row = archive.episodes[ep]['manifest_row']
            permutation = entry['permutation']
            if (row['dataset_root'] != review['dataset_root']
                    or row['episode_index'] != entry['source_episode']
                    or list(reversed(row['lewm_v12_slot_views'])) != review['historical_views']
                    or sorted(permutation) != [0, 1]
                    or [entry['physical_roles'][i] for i in permutation] != ['wrist','external']
                    or not entry.get('reviewed_frames') or not entry.get('upstream')):
                raise ValueError('review identity, original slots or physical roles disagree')
            self.orders_by_episode[ep] = permutation
            self.reviewed[ep] = True
        if not self.reviewed.any():
            raise ValueError('empty reviewed role registry')

    def orders(self, ids):
        ids = np.asarray(ids)
        if ids.dtype.kind not in 'iu' or (ids < 0).any() or (ids >= len(self.endpoint_episode)).any():
            raise ValueError('invalid original endpoint identity')
        return self.orders_by_episode[self.endpoint_episode[ids]]

    def remap(self, value, ids):
        """Camera axis follows the endpoint-id dimensions: [...,2,features]."""
        ids = np.asarray(ids)
        orders = self.orders(ids)
        axis = ids.ndim
        if tuple(value.shape[:axis+1]) != (*ids.shape, 2):
            raise ValueError('camera packet and endpoint identities differ')
        # Returning identity data also preserves old cache dtype and exact bits.
        if np.array_equal(orders, np.broadcast_to([0,1], orders.shape)):
            return value
        shape = (*orders.shape, *([1]*(value.ndim-axis-1)))
        if isinstance(value, torch.Tensor):
            index = torch.as_tensor(orders, device=value.device).reshape(shape).expand_as(value)
            return value.gather(axis, index)
        index = np.broadcast_to(orders.reshape(shape), value.shape)
        return np.take_along_axis(value, index, axis=axis)

    def coverage(self, endpoint_ids):
        ep = self.endpoint_episode[np.asarray(endpoint_ids)]
        known = self.reviewed[ep]
        swap = (self.orders_by_episode[ep] == [1,0]).all(-1)
        return dict(examples=int(ep.size), reviewed=int(known.sum()), swapped=int(swap.sum()),
                    unverified=int((~known).sum()), unknown_policy='retain-historical-unverified')


class RoleAwareClock:
    """Delegate raw tables/times unchanged; public exposure packets are remapped."""
    def __init__(self, raw, roles):
        self.raw, self.roles = raw, roles

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def observations(self, ids, kind):
        values = self.raw.observations(ids, kind)
        return values if kind == 'fish' else tuple(self.roles.remap(v, ids) for v in values)
