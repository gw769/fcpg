"""The compressed DINO store reads and decodes several blocks at once on its decode threads; values unchanged.
Reading one block's files at a time was slower in training (section 17.61). Synthetic store, not real data."""
import json
from pathlib import Path
import threading
import time

import numpy as np

import fcpg.v21_p0_features as p0
from fcpg.v21_encoder import ENCODER
from fcpg.v21_lossless_features import LosslessFeatures


def store(root, blocks=3, frames=4):
    int8 = ENCODER == p0.STORAGE_KINDS[p0.INT8_KIND]
    kind = p0.INT8_KIND if int8 else p0.STORAGE_KIND
    rows, codec, rng = blocks * frames, LosslessFeatures(level=1, shuffle=True), np.random.default_rng(3)
    (root / 'fresh').mkdir()
    expected = []
    for start in range(0, rows, frames):
        if int8:
            block = rng.integers(-127, 128, (frames, *p0.ROW_SHAPE)).astype(np.int8)
            scale = rng.uniform(.01, .1, (frames, *p0.ROW_SHAPE[:-1])).astype(np.float16)
            (root / 'fresh' / f'scale-{start:09d}.v21z').write_bytes(codec.encode(scale))
            expected.append((block.astype(np.float32) * scale.astype(np.float32)[..., None]).astype(np.float16))
        else:
            block = rng.standard_normal((frames, *p0.ROW_SHAPE)).astype(np.float16)
            expected.append(block)
        (root / 'fresh' / f'block-{start:09d}.v21z').write_bytes(codec.encode(block))
    (root / 'config.json').write_text(json.dumps(dict(token_storage=dict(
        kind=kind, old_blocks=str(root / 'old'), old_rows=0, fresh_blocks=str(root / 'fresh'),
        block_frames=frames, fresh_rows=rows))))
    np.savez(root / p0.TOKEN_SOURCES, kind=np.full(rows, p0.FRESH, np.uint8), source_row=np.arange(rows))
    return np.concatenate(expected)


def test_several_blocks_are_read_at_once_and_the_values_are_unchanged(tmp_path, monkeypatch):
    expected = store(tmp_path)
    reading, most, guard, read_bytes = [0], [0], threading.Lock(), Path.read_bytes

    def slow(path):
        with guard:
            reading[0] += 1
            most[0] = max(most[0], reading[0])
        time.sleep(.02)
        try:
            return read_bytes(path)
        finally:
            with guard:
                reading[0] -= 1

    monkeypatch.setattr(Path, 'read_bytes', slow)
    tokens = p0.CombinedP0Tokens(tmp_path, decode_threads=4)
    rows = np.arange(len(expected))[::-1]
    assert np.array_equal(tokens[rows], expected[rows])
    assert most[0] > 1 and tokens.stats['block_decodes'] == 3
