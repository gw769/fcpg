"""Strict opt-in consumption of one completed clock-correct DINO generation."""
import json
from pathlib import Path

import numpy as np

from .v21_encoder import ENCODER, EXTRACTOR_VERSIONS
from .v21_recorded_clock import CLOCK_VERSION


class RecordedClockCache:
    def __init__(self, paths, archive, fish, caches):
        root = Path(paths['dino_cache_root'])
        metadata = json.loads((root/'manifest.json').read_text())
        expected = dict(clock_version=CLOCK_VERSION, extractor_version=EXTRACTOR_VERSIONS[ENCODER],
                        extraction_id=paths['dino_extraction_id'])
        if paths.get('dino_clock_version') != CLOCK_VERSION or metadata.get('complete') is not True:
            raise ValueError('incomplete or unspecified recorded-clock cache generation')
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError('recorded-clock extraction contract differs')
        quality = metadata.get('camera_quality_policy')
        if quality != paths.get('dino_camera_quality_policy'):
            raise ValueError('camera quality generation must be explicitly selected')
        if quality is not None:
            expected['camera_quality_policy'] = quality
        # A run-local frozen weight/code path differentiates identically named
        # cache directories across experiments without per-batch hashing.
        for key in ('repository', 'weights', 'input_rgb', 'image_size', 'resize', 'normalization',
                    'compute_precision', 'storage_precision', 'camera_role_policy'):
            if key not in metadata:
                raise ValueError(f'missing extraction setting: {key}')
            expected[key] = metadata[key]
        self.metadata = metadata
        self.archive_times = np.load(root/'archive_timestamps.npy', mmap_mode='r')
        self.fish_times = np.load(root/'fish_timestamps.npy', mmap_mode='r')
        if self.archive_times.shape != (archive.episodes[-1]['frame_stop'],) or self.fish_times.shape != (len(fish.q),):
            raise ValueError('recorded clock length differs from canonical motion')
        self.tables = {}
        for kind, count, values in [('archive', archive.endpoint_count, caches[:2]),
                                     ('fish', len(fish.q), caches[2:])]:
            table = dict(valid=np.zeros((count, 2), bool), capture_times=np.zeros((count, 2), np.float64),
                         capture_q=np.zeros((count, 2, 6), np.float32), present=np.zeros(count, bool))
            for cache in values:
                config = json.loads((cache.root/'config.json').read_text())
                if config.get('complete') is not True or any(config.get(k) != v for k, v in expected.items()):
                    raise ValueError('mixed or incomplete DINO extraction generations')
                ids = np.asarray(cache.endpoint_index)
                if kind == 'fish':
                    fish.validate_frames(ids)
                if table['present'][ids].any():
                    raise ValueError('overlapping cache partition identities')
                table['valid'][ids] = cache.view_valid
                for key in ('capture_times', 'capture_q'):
                    value = np.load(cache.root/f'{key}.npy', mmap_mode='r')
                    if value.shape != (len(ids), *table[key].shape[1:]) or not np.isfinite(value).all():
                        raise ValueError(f'invalid capture sidecar: {key}')
                    table[key][ids] = value
                table['present'][ids] = True
            admitted = np.arange(count) if kind == 'archive' else fish.training_frames
            if not table['present'][admitted].all():
                raise ValueError('cache removed original endpoint identities')
            self.tables[kind] = table

    def times(self, kind):
        return self.fish_times if kind == 'fish' else self.archive_times

    def observations(self, ids, kind):
        table = self.tables['fish' if kind == 'fish' else 'archive']
        if not table['present'][ids].all():
            raise ValueError('uncached or sealed exposure requested')
        return table['valid'][ids], table['capture_times'][ids], table['capture_q'][ids]
