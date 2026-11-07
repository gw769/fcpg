#!/usr/bin/env python3
"""Build a body-centre Actor cue table from explicitly selected real episodes.

No actions are opened. Episode 2 RGB/q is never indexed. Unprocessed rows are
declared storage placeholders, distinguished from a processed missing cue by
the required manifest. The encoder is reset at every episode boundary.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA = 'v21-body-object-relative-table-v1'
TARGET = 'registered-fish-body-cog'
ALLOWED_EPISODES = (0, 1, 3)
TABLE_NAME = 'object_relative.npy'
WRIST_FEATURES = ('x_gripper', 'y_gripper', 'z_gripper', 'opening', 'bias')


def episode_ids(values):
    values = list(values)
    if (not values or any(type(v) not in (int, np.int32, np.int64) for v in values)
            or len(set(values)) != len(values) or not set(values) <= set(ALLOWED_EPISODES)):
        raise ValueError('explicit unique episodes from 0, 1, 3 required; episode 2 is sealed')
    return tuple(sorted(int(v) for v in values))


def manifest_path_for(table):
    return Path(table).with_suffix('.manifest.json')


def metadata(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest.get('camera') != 'observation.images.wrist'
            or manifest.get('fps') != 30 or not manifest.get('dataset_id')
            or not isinstance(manifest.get('source_root'), str) or not manifest['source_root']):
        raise ValueError('identified 30 Hz wrist source manifest required')
    episodes = np.load(root/'episodes.npy', mmap_mode='r', allow_pickle=False)
    frames = np.load(root/'frame_indices.npy', mmap_mode='r', allow_pickle=False)
    if (episodes.ndim != 1 or episodes.dtype.kind not in 'iu'
            or frames.shape != episodes.shape or frames.dtype.kind not in 'iu'
            or len(episodes) == 0 or manifest.get('frames') != len(episodes)):
        raise ValueError('source frame/episode identity metadata is inconsistent')
    return root, manifest, episodes, frames


def selected_rows(episodes, frames, selected):
    result = {}
    for episode in episode_ids(selected):
        rows = np.flatnonzero(episodes == episode)
        if (not len(rows) or np.any(np.diff(rows) != 1)
                or not np.array_equal(frames[rows], np.arange(len(rows)))):
            raise ValueError(f'episode {episode} has missing, duplicate or unordered frame identities')
        result[episode] = rows
    return result


def _json_safe(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    return value


def read_wrist_reference(path):
    """Same-sized squash/Column calibrations are not usable wrist evidence."""
    data = json.loads(Path(path).read_text())
    if (data.get('geometry_protocol') != 'letterbox_v1' or data.get('image_size') != [224, 224]
            or data.get('content_rect') != [0, 28, 224, 196]
            or (data.get('source_width'), data.get('source_height')) != (640, 480)):
        raise ValueError('wrist reference requires declared 640x480 -> letterbox224 geometry')
    fits = data.get('fits')
    if isinstance(fits, dict):
        row = fits.get('wrist', {})
    elif isinstance(fits, list):
        rows = [row for row in fits if row.get('view') == 'wrist']
        row = rows[0] if len(rows) == 1 else {}
    else:
        row = {}
    coefficients = np.asarray(row.get('coefficients', []), np.float64)
    ridge = row.get('ridge')
    if (row.get('view') != 'wrist' or tuple(row.get('feature_names', ())) != WRIST_FEATURES
            or coefficients.shape != (5, 2) or not np.isfinite(coefficients).all()
            or not isinstance(ridge, (float, int)) or not np.isfinite(ridge) or ridge < 0):
        raise ValueError('wrist fit requires gripper-frame features and finite coefficients')
    return dict(path=str(Path(path).resolve()), geometry_protocol='letterbox_v1',
        image_size=[224, 224], content_rect=[0, 28, 224, 196], source_size=[640, 480],
        fit=dict(view='wrist', feature_names=list(WRIST_FEATURES), coefficients=coefficients.tolist(), ridge=ridge))


def dependency_description(profile, wrist_reference, kinematics_model):
    """Readable asset/configuration evidence, without adding file hashes."""
    projection = read_wrist_reference(wrist_reference)
    profile = Path(profile).resolve(); model = Path(kinematics_model).resolve()
    payload = json.loads((profile/'targets.json').read_text())
    if payload.get('version') != 1 or not isinstance(payload.get('targets'), list):
        raise ValueError('registered target profile required')
    targets = payload['targets']
    if not any(row.get('role') == 'pickup' and row.get('view') == 'wrist' for row in targets):
        raise ValueError('registered wrist pickup template required')
    templates = []
    for row in targets:
        path = (profile/row['template_file']).resolve()
        if profile not in path.parents or not path.is_file():
            raise ValueError('target template must exist within the registered profile')
        templates.append(dict(file=str(path), size_bytes=path.stat().st_size))
    return dict(profile=str(profile), target_profile=payload, templates=templates,
        wrist_reference=str(Path(wrist_reference).resolve()), wrist_projection=projection,
        kinematics_model=str(model), kinematics_model_xml=model.read_text(), device='cpu')


def build_table(wrist_root, episodes, output, *, encoder_factory, source_geometry=None,
                dependencies=None, progress=None):
    """The injected factory permits CPU-only guarded-reader boundary tests."""
    selected = episode_ids(episodes)  # Reject sealed requests before any input IO.
    if source_geometry != 'letterbox_v1':
        raise ValueError('explicit verified --source-geometry letterbox_v1 is required')
    root, source_manifest, identities, frame_indices = metadata(wrist_root)
    declared_geometry = source_manifest.get('geometry_protocol')
    if (declared_geometry not in (None, source_geometry)
            or source_manifest.get('image_size') not in (224, [224, 224])
            or source_manifest.get('content_rect', [0, 28, 224, 196]) != [0, 28, 224, 196]):
        raise ValueError('source geometry metadata contradicts letterbox224 wrist input')
    rows_by_episode = selected_rows(identities, frame_indices, selected)
    output = Path(output).resolve()
    if output == root or root in output.parents or output.exists():
        raise ValueError('use a new output directory separate from the source')
    encoder = encoder_factory()
    description = encoder.describe()
    if description.get('policy_source') != TARGET or not callable(getattr(encoder, 'clear_observation', None)):
        raise ValueError('current registered body encoder with episode reset is required')
    # Opening a memmap reads only its header. Every value access below is a
    # scalar row that was admitted by episode identity before opening RGB/q.
    rgb = np.load(root/'frames.npy', mmap_mode='r', allow_pickle=False)
    q = np.load(root/'states.npy', mmap_mode='r', allow_pickle=False)
    n = len(identities)
    if rgb.shape != (n, 224, 224, 3) or rgb.dtype != np.uint8 or q.shape != (n, 6) or q.dtype.kind != 'f':
        raise ValueError('source requires paired RGB[N,224,224,3] and actual q[N,6]')
    table = np.zeros((n, 8), np.float32)
    report = dict(schema=SCHEMA, complete=False, object_target_source=TARGET,
        table_file=TABLE_NAME, table_shape=[n, 8], table_dtype='float32',
        source=dict(wrist_root=str(root), manifest=source_manifest),
        source_geometry=dict(protocol=source_geometry, configured_explicitly=True,
            original_manifest_declaration=declared_geometry, image_size=[224, 224],
            content_rect=[0, 28, 224, 196], padding_check='exact black rows 0:28 and 196:224 on every admitted frame',
            padding_checked_frames=0, original_manifest_modified=False),
        episodes=list(selected), rows=[], encoder=description,
        dependencies={} if dependencies is None else dependencies,
        episode_reset=True, source_frames_used_in_recorded_order=True,
        encoder_cadence='all selected actual frames in 30 Hz episode order; stateful body support, not independent endpoint encoding',
        unprocessed_rows_are_training_samples=False,
        unprocessed_storage='zero placeholder; forbidden unless row appears in completed manifest',
        computed_unavailable_semantics='actual encoder observed no usable body cue; existing available=0 contract',
        rgb_and_q_are_actual=True, action_files_opened=False, episode_2_values_read=False,
        hardware_used=False, model_training=False, new_demonstrations=False)
    output.mkdir(parents=True)
    started = time.monotonic()
    available_count = 0
    try:
        for episode, rows in rows_by_episode.items():
            encoder.clear_observation()
            for index in rows:
                index = int(index)
                if int(identities[index]) != episode or episode not in selected or episode == 2:
                    raise ValueError('row identity changed before RGB/q access')
                frame = np.ascontiguousarray(rgb[index])
                if np.any(frame[:28] != 0) or np.any(frame[196:] != 0):
                    raise ValueError(f'selected row {index} fails declared letterbox padding geometry')
                report['source_geometry']['padding_checked_frames'] += 1
                position = np.asarray(q[index], dtype=np.float32).copy()
                if not np.isfinite(position).all():
                    raise ValueError(f'nonfinite actual q at selected row {index}')
                vector = np.asarray(encoder.encode(frame, position), dtype=np.float32)
                if (vector.shape != (8,) or not np.isfinite(vector).all() or vector[0] not in (0., 1.)
                        or (vector[0] == 0. and np.any(vector != 0.))):
                    raise ValueError(f'invalid actual body cue at selected row {index}')
                table[index] = vector
                record = getattr(encoder, 'last_policy_record', None)
                record = record if isinstance(record, dict) else {}
                report['rows'].append(dict(row=index, episode=episode,
                    frame_index=int(frame_indices[index]), relative_time_s=int(frame_indices[index])/30.,
                    computed=True, available=bool(vector[0]), vector=vector.tolist(),
                    body_truncated=record.get('body_truncated'),
                    cog_px=record.get('cog_px'),
                    unavailable_reason=None if vector[0] else record.get('reason', 'no_usable_registered_body_at_this_frame')))
                available_count += int(vector[0])
                if progress is not None and len(report['rows']) % 100 == 0:
                    progress(dict(event='body_table_progress', episode=episode, computed=len(report['rows']),
                        available=available_count, elapsed_s=time.monotonic()-started))
            if progress is not None:
                progress(dict(event='body_table_episode_complete', episode=episode, computed=len(report['rows']),
                    available=available_count, elapsed_s=time.monotonic()-started))
        report['complete'] = True
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['computed_rows'] = len(report['rows'])
        report['available_rows'] = sum(row['available'] for row in report['rows'])
        report['skipped_rows'] = n - len(report['rows'])
        np.save(output/TABLE_NAME, table, allow_pickle=False)
        manifest_path_for(output/TABLE_NAME).write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False)+'\n')
    return report


def validate_manifest(table_path, manifest_path, fish, expected_wrist_root, required_rows, *, source_wrist_root=None):
    """Reject missing identity/computation evidence before a row reaches training."""
    table_path, manifest_path = Path(table_path).resolve(), Path(manifest_path).resolve()
    doc = json.loads(manifest_path.read_text())
    if (doc.get('schema') != SCHEMA or doc.get('complete') is not True
            or doc.get('object_target_source') != TARGET or doc.get('episode_reset') is not True
            or doc.get('episode_2_values_read') is not False
            or doc.get('action_files_opened') is not False
            or doc.get('source_frames_used_in_recorded_order') is not True
            or doc.get('unprocessed_rows_are_training_samples') is not False
            or manifest_path.parent/doc.get('table_file', '') != table_path):
        raise ValueError('complete identity-bound body table manifest required')
    root, actual_manifest, episodes, frames = metadata(expected_wrist_root)
    provenance_root = root if source_wrist_root is None else Path(source_wrist_root).resolve()
    if doc.get('source') != dict(wrist_root=str(provenance_root), manifest=actual_manifest):
        raise ValueError('body table source manifest differs from training source')
    if not np.array_equal(episodes, fish.episodes) or not np.array_equal(frames, fish.frame_indices):
        raise ValueError('body table frame identities differ from training source')
    selected = episode_ids(doc['episodes'])
    expected = np.concatenate(list(selected_rows(episodes, frames, selected).values()))
    geometry = doc.get('source_geometry', {})
    if (geometry.get('protocol') != 'letterbox_v1' or geometry.get('configured_explicitly') is not True
            or geometry.get('content_rect') != [0, 28, 224, 196]
            or geometry.get('image_size') != [224, 224]
            or geometry.get('padding_checked_frames') != len(expected)
            or geometry.get('original_manifest_declaration') != actual_manifest.get('geometry_protocol')
            or geometry.get('original_manifest_modified') is not False):
        raise ValueError('body table lacks consistent source geometry verification')
    entries = doc.get('rows')
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError('body table has missing computation identities')
    actual_rows = []
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError('body table has a forged or missing frame identity')
        index = row.get('row')
        if (type(index) is not int or not 0 <= index < len(episodes) or row.get('computed') is not True
                or row.get('episode') != int(episodes[index]) or row.get('frame_index') != int(frames[index])
                or row.get('relative_time_s') != int(frames[index])/30.
                or type(row.get('available')) is not bool or row.get('episode') == 2):
            raise ValueError('body table has a forged or missing frame identity')
        actual_rows.append(index)
    if not np.array_equal(actual_rows, expected):
        raise ValueError('body table rows are missing, duplicated or reordered')
    required = np.asarray(required_rows)
    if required.ndim != 1 or required.dtype.kind not in 'iu' or not np.isin(required, expected).all():
        raise ValueError('training/development asks for an uncomputed body table row')
    table = np.load(table_path, mmap_mode='r', allow_pickle=False)
    if table.shape != (len(episodes), 8) or table.dtype != np.float32 or not np.isfinite(table).all():
        raise ValueError('body table shape, dtype or values differ from the manifest')
    availability = np.asarray([row['available'] for row in entries])
    vectors = np.asarray([row.get('vector', []) for row in entries], np.float32)
    if vectors.shape != (len(entries), 8) or not np.array_equal(table[expected], vectors):
        raise ValueError('body cue values differ from recorded computation')
    if (not np.array_equal(table[expected, 0], availability.astype(np.float32))
            or np.any(table[expected[~availability]] != 0.)):
        raise ValueError('body availability differs from recorded computation')
    computed = np.zeros(len(episodes), bool); computed[expected] = True
    if np.any(table[~computed] != 0.):
        raise ValueError('uncomputed rows contain undeclared cue values')
    if (doc.get('table_shape') != list(table.shape) or doc.get('table_dtype') != str(table.dtype)
            or doc.get('computed_rows') != len(expected) or doc.get('available_rows') != int(availability.sum())
            or doc.get('skipped_rows') != len(table)-len(expected)):
        raise ValueError('body table summary differs from recorded computation')
    return dict(path=str(manifest_path), object_target_source=TARGET,
        episodes=list(selected), computed_rows=len(expected), available_rows=int(availability.sum()),
        source=doc['source'], source_geometry=geometry, required_rows_verified=len(np.unique(required)),
        dependencies=doc.get('dependencies'), encoder=doc.get('encoder'),
        computed_cues=[dict(row=int(index), vector=row['vector']) for index, row in zip(expected, entries)],
        unprocessed_rows_used=False, episode_2_values_read=False)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wrist-root', type=Path, required=True)
    p.add_argument('--episodes', nargs='+', type=int, required=True)
    p.add_argument('--source-geometry', choices=('letterbox_v1',), required=True,
                   help='declare only after verifying the actual source producer; every selected frame padding is checked')
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--wrist-reference', type=Path, required=True)
    p.add_argument('--kinematics-model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    episode_ids(args.episodes)
    dependencies = dependency_description(args.profile, args.wrist_reference, args.kinematics_model)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    def encoder_factory():
        import torch
        # 10/31: this table was made with the operator colour encoder, which is gone from the code; the cue
        # tables come from scripts/build_v21_halfsecond_cue_tables.py (no colour)
        raise SystemExit('the colour encoder of this table is gone (10/31): use '
                         'scripts/build_v21_halfsecond_cue_tables.py')
    result = build_table(args.wrist_root, args.episodes, args.output,
        encoder_factory=encoder_factory, source_geometry=args.source_geometry, dependencies=dependencies,
        progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({key: result[key] for key in ('complete', 'episodes', 'computed_rows', 'available_rows', 'skipped_rows')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
