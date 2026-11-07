#!/usr/bin/env python3
"""Explicit relocation of existing V21 sources with ordinary identity evidence.

The reference stores complete manifest bytes and only identity/index arrays.
Motion, RGB and DINO tensor files are checked by header/size here; the original
V21Sources chain path remains responsible for their actual-value contracts.
No tensor-bank hashing, preprocessing or source-file modification is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

SCHEMA = 'v21-explicit-source-map-v1'
REFERENCE = 'source-reference.npz'
IDENTITY_SCOPE = ('complete manifests; complete endpoint/fish/DINO indices; '
                  'temporal indices for every original archive chain')
FILES = {
    'endpoint_root': ('manifest.json', 'chain_id.npy', 'episode_index.npy', 'local_frame_index.npy',
        'row_endpoint_index.npy', 'sidecar_row.npy', 'split.npy', 'task_index.npy', 'view_valid.npy'),
    'companion_root': ('manifest.json', 'chain_id.npy', 'completed_commands.npy', 'measured_state_path.npy'),
    'temporal_root': ('manifest.json', 'actions.npy', 'episode_offsets.npy', 'states.npy',
        'window_episode.npy', 'window_future_action_indices.npy', 'window_task_index.npy'),
    'rgb_root': ('manifest.json', 'sidecar_row.npy'),
    'fish_wrist_root': ('manifest.json', 'actions.npy', 'episodes.npy', 'frame_indices.npy', 'frames.npy', 'states.npy'),
    'fish_column_root': ('manifest.json', 'actions.npy', 'episodes.npy', 'frame_indices.npy', 'frames.npy', 'states.npy'),
    'dino_cache_root': tuple(f'{kind}/{name}' for kind in ('fish', 'temporal-dev', 'temporal-train')
        for name in ('config.json', 'done.npy', 'endpoint_index.npy', 'tokens.npy', 'view_valid.npy')),
}


def _paths(paths):
    if (not isinstance(paths, dict) or set(paths) != set(FILES)
            or any(not isinstance(v, str) or not Path(v).is_absolute() for v in paths.values())):
        raise ValueError('source map must explicitly name the original seven absolute source roots')
    if len(set(paths.values())) != len(paths):
        raise ValueError('source map roots must remain distinct')
    return dict(paths)


def _mode(key, filename):
    if filename.endswith('.json'): return 'manifest-bytes'
    if (key in ('endpoint_root', 'rgb_root') or filename == 'episode_offsets.npy'
            or (key == 'companion_root' and filename == 'chain_id.npy')
            or (key.startswith('fish_') and filename in ('episodes.npy', 'frame_indices.npy'))
            or (key == 'dino_cache_root' and not filename.endswith('/tokens.npy'))):
        return 'identity-array'
    if key == 'temporal_root' and filename.startswith('window_'):
        return 'archive-chain-identity'
    return 'header-only'


def _entries():
    return [(key, name, f'file_{i:03d}', _mode(key, name))
        for i, (key, name) in enumerate((key, name) for key, names in FILES.items() for name in names)]


def _header(path):
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    if value.dtype.hasobject:
        raise ValueError('object arrays cannot enter a recorded source map')
    return value, dict(shape=list(value.shape), dtype=str(value.dtype))


def _reference_path(map_path, document):
    if document.get('reference_file') != REFERENCE:
        raise ValueError('source map requires its adjacent identity reference')
    path = (map_path.parent/REFERENCE).resolve()
    if path.parent != map_path.parent:
        raise ValueError('source map reference must remain adjacent to the map')
    return path


def capture_source_map(original_paths, effective_paths, output):
    """Run beside the original sources, before relocating any dataset roots."""
    original, effective = _paths(original_paths), _paths(effective_paths)
    output = Path(output).resolve()
    if output.exists(): raise ValueError('source-map output must be a new directory')
    chain_ids = np.asarray(np.load(Path(original['endpoint_root'])/'chain_id.npy', mmap_mode='r', allow_pickle=False))
    if chain_ids.ndim != 1 or not len(chain_ids) or chain_ids.dtype.kind not in 'iu' or (chain_ids < 0).any():
        raise ValueError('original archive chain identities are invalid')
    references, files = {}, []
    for key, name, member, mode in _entries():
        path = Path(original[key])/name
        row = dict(root=key, file=name, bytes=path.stat().st_size, verification=mode)
        if mode == 'manifest-bytes':
            raw = path.read_bytes()
            json.loads(raw)  # A complete valid original manifest, preserved byte-for-byte.
            references[member] = np.frombuffer(raw, np.uint8).copy()
        else:
            value, row['array'] = _header(path)
            if mode == 'archive-chain-identity':
                if chain_ids.max() >= len(value): raise ValueError('archive chain identity leaves temporal index')
                references[member] = np.array(value[chain_ids], copy=True)
            elif mode == 'identity-array':
                references[member] = np.array(value, copy=True)
        if mode != 'header-only': row['reference_member'] = member
        files.append(row)
    output.mkdir(parents=True)
    np.savez_compressed(output/REFERENCE, **references)
    document = dict(schema=SCHEMA, complete=True, source_paths_original=original,
        source_paths_effective=effective, reference_file=REFERENCE, files=files,
        identity_scope=IDENTITY_SCOPE,
        payload_values_copied=False, source_files_modified=False, tensor_bank_hashing=False,
        fish_episode_2_payload_read=False)
    (output/'source-map.json').write_text(json.dumps(document, indent=2, allow_nan=False)+'\n')
    return output/'source-map.json'


def resolve_source_paths(original_paths, source_map=None):
    """Default calls preserve the historical paths without new checks or IO."""
    return _resolve_source_paths(original_paths, source_map, motion_only=False)


def resolve_motion_source_paths(original_paths, source_map):
    """Verify the recorded motion sources without claiming DINO is complete.

    The full Actor resolver above still requires every visual file. Only the
    explicit LeWM motion-only loader should consume these effective paths.
    """
    if source_map is None:
        raise ValueError('motion source relocation requires an explicit source map')
    return _resolve_source_paths(original_paths, source_map, motion_only=True)


def _resolve_source_paths(original_paths, source_map, *, motion_only):
    if source_map is None: return dict(original_paths), None
    original = _paths(original_paths)
    path = Path(source_map).resolve()
    doc = json.loads(path.read_text())
    if (doc.get('schema') != SCHEMA or doc.get('complete') is not True
            or doc.get('source_paths_original') != original
            or doc.get('payload_values_copied') is not False
            or doc.get('source_files_modified') is not False
            or doc.get('fish_episode_2_payload_read') is not False
            or doc.get('tensor_bank_hashing') is not False
            or doc.get('identity_scope') != IDENTITY_SCOPE):
        raise ValueError('source map must match the preserved base source_paths exactly')
    effective = _paths(doc['source_paths_effective'])
    entries = _entries()
    files = doc.get('files')
    if not isinstance(files, list) or len(files) != len(entries):
        raise ValueError('source map file population differs from the V21 loader contract')
    reference_path = _reference_path(path, doc)
    total_bytes, manifest_count, identity_count = 0, 0, 0
    with np.load(reference_path, allow_pickle=False) as reference:
        expected_members = {member for _, _, member, mode in entries if mode != 'header-only'}
        if set(reference.files) != expected_members:
            raise ValueError('source map identity reference members differ')
        chain_member = next(member for key, name, member, _ in entries
            if key == 'endpoint_root' and name == 'chain_id.npy')
        chain_ids = reference[chain_member]
        if chain_ids.ndim != 1 or not len(chain_ids) or chain_ids.dtype.kind not in 'iu' or (chain_ids < 0).any():
            raise ValueError('source map archive identities are invalid')
        for row, (key, name, member, mode) in zip(files, entries):
            if (not isinstance(row, dict) or row.get('root') != key or row.get('file') != name or row.get('verification') != mode
                    or (mode != 'header-only' and row.get('reference_member') != member)):
                raise ValueError('source map attempts to change an expected file or verification scope')
            if motion_only and key == 'dino_cache_root':
                continue
            local = Path(effective[key])/name
            if local.stat().st_size != row['bytes']:
                raise ValueError(f'relocated source file size differs: {key}/{name}')
            total_bytes += row['bytes']
            if mode == 'manifest-bytes':
                expected = reference[member]
                if expected.dtype != np.uint8 or expected.ndim != 1 or local.read_bytes() != expected.tobytes():
                    raise ValueError(f'relocated manifest differs: {key}/{name}')
                manifest_count += 1
            else:
                value, header = _header(local)
                if header != row['array']:
                    raise ValueError(f'relocated array shape/dtype differs: {key}/{name}')
                if mode != 'header-only':
                    expected = reference[member]
                    if mode == 'archive-chain-identity':
                        if chain_ids.max() >= len(value): raise ValueError('archive identity leaves relocated temporal index')
                        value = value[chain_ids]
                    if value.dtype != expected.dtype or not np.array_equal(value, expected):
                        raise ValueError(f'relocated episode/frame/chain identity differs: {key}/{name}')
                    identity_count += 1
    report = dict(path=str(path), reference_file=str(reference_path), schema=SCHEMA,
        source_paths_original=original, source_paths_effective=effective,
        complete_manifest_contents_equal=True, manifest_files=manifest_count,
        active_identity_arrays_equal=True, identity_arrays=identity_count,
        file_headers_and_sizes_equal=True,
        verified_files=sum(not motion_only or key != 'dino_cache_root' for key, _, _, _ in entries),
        verified_file_bytes=total_bytes,
        actual_motion_values_checked_by='unchanged V21Sources and TrainingChain construction',
        tensor_bank_hashing=False, fish_episode_2_payload_read=False)
    if motion_only:
        report.update(scope='recorded-motion-only', visual_features_verified=False,
            full_source_verification_complete=False,
            verified_roots=[key for key in FILES if key != 'dino_cache_root'])
    return effective, report


def pin_source_map(output, source_map):
    """Keep an ordinary copy so a resume cannot silently replace the reference."""
    if source_map is None: return
    source_map = Path(source_map).resolve()
    doc = json.loads(source_map.read_text())
    reference = _reference_path(source_map, doc)
    target = Path(output)/'source-map'
    pairs = ((source_map, 'source-map.json'), (reference, REFERENCE))
    if target.exists():
        for source, name in pairs:
            destination = target/name
            if not destination.is_file() or destination.read_bytes() != source.read_bytes():
                raise ValueError('resume source mapping/reference changed from the recorded run')
        return
    target.mkdir(parents=True)
    for source, name in pairs:
        shutil.copyfile(source, target/name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--mapping-proposal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    base = json.loads((args.base_run/'config.json').read_text())
    proposal = json.loads(args.mapping_proposal.read_text())
    if proposal['source_paths_original'] != base['source_paths']:
        raise ValueError('proposal source paths differ from the original base')
    result = capture_source_map(base['source_paths'], proposal['source_paths_proposed'], args.output)
    print(json.dumps(dict(source_map=str(result), reference_bytes=(result.parent/REFERENCE).stat().st_size,
        source_files_modified=False, payload_values_copied=False)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
