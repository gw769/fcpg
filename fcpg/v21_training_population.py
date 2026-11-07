"""Opt-in, reviewed recording isolation; never edit the source split or pixels.

Canonical episode IDs are not recording identities. This filter excludes only
explicitly reviewed TRAIN copies of DEV recordings. It does not deduplicate all
motion matches, erase a warm parent's exposure, or certify unseen validation.
"""
from copy import deepcopy

import numpy as np

SCHEMA = 'v21-reviewed-recording-isolation-v1'
CONFIG_KEY = 'training_population_contract'


def validate_population_resume(previous, contract, *, report=None):
    """Check before opening data or writing anything into a resumed run."""
    if previous.get(CONFIG_KEY) != contract:
        raise ValueError('training population changed; start a new experiment, not a resume')
    if report is not None and previous.get('training_population') != report:
        raise ValueError('effective training population changed since the saved run')


def _rows(value, upper):
    rows = np.asarray(value)
    if (rows.ndim != 1 or rows.dtype.kind not in 'iu'
            or (rows < 0).any() or (rows >= upper).any()
            or len(np.unique(rows)) != len(rows)):
        raise ValueError('population rows must be unique in-range integer identities')
    return rows


def isolate_reviewed_recordings(splits, source, contract=None):
    """Filter all TRAIN consumers once, preserving order and every DEV/fish row.

    Input must be the original population from derive_splits, not another
    filtered index. A missing contract is the exact legacy no-op.
    """
    if contract is None:
        return splits, None
    if (contract.get('schema') != SCHEMA or contract.get('purpose') != 'research-only'
            or contract.get('train_policy') != 'exclude-reviewed-cross-split-families'
            or contract.get('development_policy') != 'keep-original'
            or contract.get('parent_exposure') != 'not-reconstructed'
            or not contract.get('review_id') or not contract.get('evidence')):
        raise ValueError('explicit reviewed recording isolation contract required')

    archive = source.archive
    split = np.asarray(archive.archive.split)
    endpoint = np.asarray(archive.row_endpoint_index)
    endpoint_episode = np.asarray(archive.endpoint_episode_index)
    offsets = np.asarray(source.offsets)
    if (split.ndim != 1 or not np.isin(split, [0, 1]).all()
            or endpoint.shape != (len(split), 5) or endpoint.dtype.kind not in 'iu'
            or (endpoint < 0).any() or (endpoint >= len(endpoint_episode)).any()
            or endpoint_episode.ndim != 1 or endpoint_episode.dtype.kind not in 'iu'
            or (endpoint_episode < 0).any() or (endpoint_episode >= len(archive.episodes)).any()
            or offsets.shape != (len(archive.episodes)+1,) or offsets.dtype.kind not in 'iu'
            or (np.diff(offsets) < 0).any()):
        raise ValueError('archive episode/endpoint identity or recorded lengths differ')
    episodes = endpoint_episode[endpoint]
    if not (episodes == episodes[:, :1]).all():
        raise ValueError('population chain crosses an episode')
    row_episode = episodes[:, 0]
    dense = _rows(source.dense_rows, len(split))
    counts = {}
    for role, code in (('train', 0), ('development', 1)):
        rows = _rows(splits[role]['archive'], len(split))
        dense_rows = _rows(splits[role]['dense'], len(split))
        if (not np.array_equal(rows, np.flatnonzero(split == code))
                or not np.array_equal(dense_rows, dense[split[dense] == code])):
            raise ValueError('recording isolation requires the complete original split and dense order')
        if contract['original_archive_rows'][role] != len(rows):
            raise ValueError('original archive population count differs from reviewed evidence')
        counts[role] = np.bincount(row_episode[rows], minlength=len(archive.episodes))
    if np.any((counts['train'] > 0) & (counts['development'] > 0)):
        raise ValueError('original canonical TRAIN/DEV episodes overlap')

    seen, excluded, retained_dev, family_ids = set(), [], [], set()
    groups = contract.get('groups', [])
    if not groups:
        raise ValueError('reviewed cross-split recording groups required')
    for group in groups:
        family = group['family_id']
        if (not family or family in family_ids or not group.get('evidence')
                or type(group['frames']) is not int or group['frames'] <= 0):
            raise ValueError('duplicate family or missing recording evidence')
        family_ids.add(family)
        sides = set()
        for member in group['episodes']:
            ep = member['canonical_episode']
            if type(ep) is not int or not 0 <= ep < len(archive.episodes) or ep in seen:
                raise ValueError('duplicate or invalid reviewed recording episode')
            seen.add(ep)
            row = archive.episodes[ep]['manifest_row']
            if (row['dataset_root'] != member['dataset_root']
                    or row['episode_index'] != member['source_episode']
                    or row['tasks'] != member['tasks']
                    or list(reversed(row['lewm_v12_slot_views'])) != group['historical_views']
                    or int(offsets[ep+1]-offsets[ep]) != group['frames']):
                raise ValueError('reviewed recording identity, views or length differs')
            for role in ('train', 'development'):
                if counts[role][ep] != member[role+'_windows']:
                    raise ValueError('reviewed recording split counts differ')
            train, dev = int(counts['train'][ep]), int(counts['development'][ep])
            if (train > 0) == (dev > 0):
                raise ValueError('reviewed member must belong to exactly one active split')
            sides.add('train' if train else 'development')
            (excluded if train else retained_dev).append(ep)
        if sides != {'train', 'development'}:
            raise ValueError('only reviewed cross-split families may be isolated')

    result = deepcopy(splits)
    removed = {}
    for kind in ('archive', 'dense'):
        rows = np.asarray(splits['train'][kind])
        keep = ~np.isin(row_episode[rows], excluded)
        result['train'][kind] = rows[keep].copy()
        removed[kind] = rows[~keep].tolist()
        if not len(result['train'][kind]):
            raise ValueError('recording isolation leaves an empty TRAIN population')
        if kind == 'dense' and 'dense_train_ordinals' in splits:
            ordinals = np.asarray(splits['dense_train_ordinals'])
            if not np.array_equal(ordinals, np.flatnonzero(split[dense] == 0)):
                raise ValueError('original dense TRAIN ordinal mapping differs')
            result['dense_train_ordinals'] = ordinals[keep].copy()
    if len(removed['archive']) != contract['expected_removed_archive_rows']:
        raise ValueError('reviewed TRAIN removal count differs')
    report = dict(schema=SCHEMA, review_id=contract['review_id'],
        original_rows={role: {kind: len(rows) for kind, rows in splits[role].items()}
                       for role in ('train', 'development')},
        effective_rows={role: {kind: len(rows) for kind, rows in result[role].items()}
                        for role in ('train', 'development')},
        excluded_train_episodes=sorted(excluded), retained_development_episodes=sorted(retained_dev),
        removed_train_rows=removed, source_split_modified=False, source_files_deleted=False,
        development_unchanged=True, fish_unchanged=True,
        parent_exposure='not-reconstructed', unseen_validation_claimed=False)
    return result, report


def effective_data_contract(original, report):
    """Expose actual row counts without erasing the historical source contract."""
    if report is None:
        return original
    result = deepcopy(original)
    result.update(schema=SCHEMA,
                  original_source_contract=deepcopy(original.get('original_source_contract', original)),
                  review_id=report['review_id'], parent_exposure='not-reconstructed')
    for role in ('train', 'development'):
        for kind in ('archive', 'dense'):
            result[kind+'_'+role+'_rows'] = report['effective_rows'][role][kind]
    return result
