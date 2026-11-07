"""Frozen DINO rows of the 1.2M archive index (P0: block endpoints H0-H4, both camera slots).

The old 320k pictures come from their original uncompressed ``tokens.npy`` or, in the merged store, from Codex's
``reused`` lossless blocks of the same rows; the 1,219,598 fresh pictures live in lossless byte-shuffle zstd1 blocks of
64 (scripts/build_v21_fresh_endpoint_features.py). A partition folder written by
scripts/build_v21_1p2m_training_roots.py holds the usual metadata arrays plus ``token-sources.npz``; this module
serves its ``tokens`` as one read-only [rows, 2, 256, 384] float16 array, so V21Sources, the feature bank and the
recorded clock read it like any V18.1 cache. Values are returned bit-for-bit as stored; nothing is re-encoded.
DINOv3-L stores (kind p0-dinov3-l-int8-v1, scripts/build_v21_dinov3_features.py; the process must run V21_ENCODER=
dinov3_vitl16) keep int8 block-*.v21z tokens with fp16 scale-*.v21z per-token scales; they are served as the float16
values int8 x scale, [rows, 2, 196, 1024], the same numbers every DINOv3-L consumer (training and robot) uses.
Blocks can be decoded on several threads (``decode_threads``, default from V21_P0_DECODE_THREADS, else 1: libzstd
runs outside the GIL); the block cache is guarded by a lock, so a prefetch thread and the main thread may share it.
Their files are read at the same time on purpose: reading one block's files at a time was about 12% slower in
training (section 17.61, HL3b against HL3), probably because the block files lie all over the hard disk and several
reads in flight let it take near ones first.
A partition packed by scripts/pack_v21_p0_store.py (token_storage.fresh_layout 'packed-v1', section 17.65) keeps the
same block and scale bytes one after the other in ascending block order in one file; it is read by one reader in
ascending order (one stream on the disk) while the decode threads decode, and every piece is checked against its
recorded crc32. The values are the same as from the block files. A copy made by scripts/split_v21_packed_store.py keeps
that file in parts on several disks (manifest ``parts``, each a run of whole blocks); a block is read from its part.
Section 17.90: an int8 partition may name further block-file sets in token_storage.more_blocks ([{blocks, rows}],
token kinds 2, 3, ...: the pictures of the recordings each later extension of the Archive added); they are read like
the old block set. Partitions without more_blocks read exactly as before.
``quantized(rows)`` (int8 stores, section 17.65) returns the stored int8 values and fp16 scales without the float16
conversion, which is the slow part on the CPU (about 0.1 s per block, and it hardly runs in parallel on .102); the
caller multiplies them on the GPU (fcpg/v21_dequantize.py). The float32 product of an int8 and a float16 is exact, so
that product rounded once to float16 is bit for bit what __getitem__ serves. Its blocks are kept in the same block
cache as int8 values and scales (half the size of a float16 block), apart from the float16 blocks __getitem__ keeps.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import zlib

import numpy as np

from .prod_v18_1_data import V181DataContractError, V181FeatureCache
from .v21_encoder import DIM, ENCODER, PATCHES
from .v21_lossless_features import LosslessFeatures

TOKEN_SOURCES = 'token-sources.npz'
STORAGE_KIND = 'p0-old-tokens-plus-fresh-lossless-v1'
INT8_KIND = 'p0-dinov3-l-int8-v1'
STORAGE_KINDS = {STORAGE_KIND: 'dinov2_vits14', INT8_KIND: 'dinov3_vitl16'}
OLD, FRESH = 0, 1
ROW_SHAPE = (2, PATCHES, DIM)
PACKED_LAYOUT = 'packed-v1'


class PackedBlocks:
    """A store partition rewritten by scripts/pack_v21_p0_store.py: each block's bytes, then its scale's bytes,
    unchanged, in ascending block order in packed.v21p (section 17.65), or that file in parts on several disks
    (scripts/split_v21_packed_store.py). A read returns both pieces after checking their recorded crc32, and leaves
    nothing in the page cache."""

    def __init__(self, root, rows, block_frames):
        root = Path(root)
        manifest = json.loads((root / 'manifest.json').read_text())
        if (manifest.get('layout') != PACKED_LAYOUT or manifest.get('complete') is not True
                or manifest.get('rows') != rows or manifest.get('block_frames') != block_frames):
            raise V181DataContractError(f'{root} is not a complete packed partition of {rows} rows in blocks of {block_frames}')
        with np.load(root / manifest['index'], allow_pickle=False) as index:
            self.offset = np.asarray(index['block_offset'], np.int64)
            self.block_length = np.asarray(index['block_length'], np.int64)
            self.scale_length = np.asarray(index['scale_length'], np.int64)
            self.crc = np.stack((index['block_crc32'], index['scale_crc32']), 1).astype(np.uint32)
            starts = np.asarray(index['start'], np.int64)
            scale_offset = np.asarray(index['scale_offset'], np.int64)
        self.block_frames = block_frames
        parts = manifest.get('parts') or [dict(path=str(root / manifest['data']), byte_start=0, bytes=manifest['bytes'],
                                               file_offset=0)]
        self.part_start = np.asarray([int(p['byte_start']) for p in parts], np.int64)
        self.file_offset = np.asarray([int(p.get('file_offset', 0)) for p in parts], np.int64)
        sizes = np.asarray([int(p['bytes']) for p in parts], np.int64)
        self.fds = [os.open(p['path'], os.O_RDONLY) for p in parts]
        self.fd = self.fds[0]
        end = self.offset + self.block_length + self.scale_length
        self.part = np.searchsorted(self.part_start, self.offset, side='right') - 1
        if (not np.array_equal(starts, np.arange(0, rows, block_frames)) or not np.array_equal(scale_offset, self.offset + self.block_length)
                or int((self.block_length + self.scale_length).sum()) != manifest['bytes']
                or self.part_start[0] != 0 or not np.array_equal(self.part_start[1:], (self.part_start + sizes)[:-1])
                or int(self.part_start[-1] + sizes[-1]) != manifest['bytes']
                or any(os.fstat(fd).st_size != off + size for fd, off, size in zip(self.fds, self.file_offset, sizes))
                or not np.array_equal(self.part, np.searchsorted(self.part_start, end - 1, side='right') - 1)):
            for fd in self.fds:
                os.close(fd)
            raise V181DataContractError(f'{root}: the packed index does not match its file')
        for fd in self.fds:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)

    def read(self, start, *, check=True):
        """The block's stored bytes and its scale bytes. check=False leaves the crc32 check to the caller (check(); the
        trainer's window reader checks on its decode threads, section 17.79)."""
        i = start // self.block_frames
        p = int(self.part[i])
        fd = self.fds[p]
        offset = int(self.offset[i] - self.part_start[p] + self.file_offset[p])
        length = int(self.block_length[i] + self.scale_length[i])

        def pread(at, n):
            parts, end = [], at + n
            while at < end:
                part = os.pread(fd, end - at, at)
                if not part:
                    raise V181DataContractError(f'packed block {start}: the file ends early')
                parts.append(part)
                at += len(part)
            return b''.join(parts) if len(parts) > 1 else parts[0]

        # section 17.79: the block and its scales are read separately, so that neither is copied out of a joint buffer
        block = pread(offset, int(self.block_length[i]))
        scale = pread(offset + int(self.block_length[i]), int(self.scale_length[i]))
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        if check:
            self.check(start, block, scale)
        return block, scale

    def check(self, start, block, scale):
        i = start // self.block_frames
        if zlib.crc32(block) != self.crc[i, 0] or zlib.crc32(scale) != self.crc[i, 1]:
            raise V181DataContractError(f'packed block {start}: the bytes differ from their recorded crc32')


class CombinedP0Tokens:
    """Row-indexable union of the old token array and the fresh lossless blocks (read-only; thread-safe cache)."""

    def __init__(self, root, *, cached_blocks=48, decode_threads=None):
        root = Path(root)
        config = json.loads((root / 'config.json').read_text())
        storage = config.get('token_storage', {})
        if STORAGE_KINDS.get(storage.get('kind')) != ENCODER:
            raise V181DataContractError(f'not a combined P0 cache of this process\'s encoder ({ENCODER}): {root}')
        self.int8 = storage['kind'] == INT8_KIND
        with np.load(root / TOKEN_SOURCES, allow_pickle=False) as sources:
            self.kind = np.asarray(sources['kind'], np.uint8)
            self.source_row = np.asarray(sources['source_row'], np.int64)
        # Old pictures: the original uncompressed tokens.npy, or (merged store) Codex's `reused` lossless blocks of the
        # same rows in the same parent order (scripts/build_v21_reused_lossless_features.py).
        self.old_root = Path(storage['old_blocks']) if storage.get('old_blocks') or self.int8 else None
        self.old = None if self.old_root else np.load(storage['old_tokens'], mmap_mode='r')
        self.old_rows = int(storage['old_rows']) if self.old_root else len(self.old)
        self.fresh_root = Path(storage['fresh_blocks'])
        self.block_frames = int(storage['block_frames'])
        self.fresh_rows = int(storage['fresh_rows'])
        layout = storage.get('fresh_layout', 'files')
        if layout not in ('files', PACKED_LAYOUT) or (layout == PACKED_LAYOUT and not self.int8):
            raise V181DataContractError(f'unknown fresh block layout {layout!r} for {storage["kind"]}')
        self.packed = PackedBlocks(self.fresh_root, self.fresh_rows, self.block_frames) if layout == PACKED_LAYOUT else None
        # section 17.90: further block-file sets, token kinds 2, 3, ...
        self.more = [(Path(more['blocks']), int(more['rows'])) for more in storage.get('more_blocks', [])]
        if self.more and not self.int8:
            raise V181DataContractError('further block sets are only for int8 stores')
        self.kinds = (OLD, FRESH, *range(2, 2 + len(self.more)))
        if ((self.old is not None and (self.old.dtype != np.float16 or self.old.shape[1:] != ROW_SHAPE))
                or len(self.kind) != len(self.source_row) or not np.isin(self.kind, self.kinds).all()):
            raise V181DataContractError('combined P0 token sources are malformed')
        for k in self.kinds:
            rows = self.source_row[self.kind == k]
            if len(rows) and (rows.min() < 0 or rows.max() >= self._set(k)[1]):
                raise V181DataContractError('combined P0 token source rows leave their arrays')
        self.shape = (len(self.kind), *ROW_SHAPE)
        self.dtype = np.dtype(np.float16)
        self.ndim = 4
        self.codec = LosslessFeatures(level=1, shuffle=True)
        self.cached_blocks = int(cached_blocks)
        self.decode_threads = int(os.environ.get('V21_P0_DECODE_THREADS', '1') if decode_threads is None else decode_threads)
        if self.decode_threads < 1:
            raise V181DataContractError('decode_threads must be at least 1')
        self._blocks = OrderedDict()
        self._lock = threading.Lock()
        self._pool = None
        self.stats = dict(old_rows=0, fresh_rows=0, block_decodes=0, block_hits=0, quantized_rows=0, quantized_decodes=0,
                          quantized_hits=0)
        if self.more:
            self.stats['more_rows'] = 0

    def _set(self, kind):
        """(folder, rows) of a token kind's block set."""
        if kind == FRESH:
            return self.fresh_root, self.fresh_rows
        if kind == OLD:
            return self.old_root, self.old_rows
        return self.more[kind - 2]

    def __len__(self):
        return self.shape[0]

    def _cached(self, key, counter='block_hits'):
        with self._lock:
            block = self._blocks.get(key)
            if block is not None:
                self._blocks.move_to_end(key)
                self.stats[counter] += 1
            return block

    def _decode_quantized(self, start, kind, raw=None, checked=True):
        """An int8 block's stored values and fp16 scales, checked. checked=False: raw are packed bytes whose crc32 is
        checked here (on a decode thread)."""
        root, rows = self._set(kind)
        if raw is None and kind == FRESH and self.packed is not None:
            raw = self.packed.read(start)
        elif raw is not None and not checked:
            self.packed.check(start, *raw)
        block = self.codec.decode(raw[0] if raw is not None else (root / f'block-{start:09d}.v21z').read_bytes())
        scale = self.codec.decode(raw[1] if raw is not None else (root / f'scale-{start:09d}.v21z').read_bytes())
        expected = (min(self.block_frames, rows - start), *ROW_SHAPE)
        if block.dtype != np.int8 or block.shape != expected or scale.dtype != np.float16 or scale.shape != expected[:-1]:
            raise V181DataContractError(f'{root.name} int8 block {start}: {block.shape} / {scale.shape}, expected {expected}')
        return block, scale

    def _decode(self, start, kind, raw=None):
        if self.int8:
            block, scale = self._decode_quantized(start, kind, raw)
            return (block.astype(np.float32) * scale.astype(np.float32)[..., None]).astype(np.float16)
        root, rows = self._set(kind)
        if raw is None and kind == FRESH and self.packed is not None:
            raw = self.packed.read(start)
        block = self.codec.decode(raw[0] if raw is not None else (root / f'block-{start:09d}.v21z').read_bytes())
        expected = (min(self.block_frames, rows - start), *ROW_SHAPE)
        if block.dtype != np.float16 or block.shape != expected:
            raise V181DataContractError(f'{root.name} block {start} has shape {block.shape}, expected {expected}')
        return block

    def _remember(self, key, block, counter='block_decodes'):
        with self._lock:
            self._blocks[key] = block
            self._blocks.move_to_end(key)
            self.stats[counter] += 1
            while len(self._blocks) > self.cached_blocks:
                self._blocks.popitem(last=False)

    def _block(self, start, kind=FRESH):
        block = self._cached((kind, start))
        if block is None:
            block = self._decode(start, kind)
            self._remember((kind, start), block)
        return block

    def __getitem__(self, index):
        rows = np.arange(self.shape[0])[index] if isinstance(index, slice) else np.asarray(index)
        scalar = rows.ndim == 0
        rows = np.atleast_1d(rows)
        if rows.dtype.kind not in 'iu' or rows.ndim != 1:
            raise V181DataContractError('combined P0 tokens take integer rows or a slice')
        rows = rows.astype(np.int64, copy=False)
        if len(rows) and (rows.min() < -self.shape[0] or rows.max() >= self.shape[0]):
            raise IndexError('combined P0 token row out of range')
        rows = np.where(rows < 0, rows + self.shape[0], rows)
        out = np.empty((len(rows), *ROW_SHAPE), np.float16)
        kind, source = self.kind[rows], self.source_row[rows]
        old = np.flatnonzero(kind == OLD)
        if len(old) and self.old is not None:            # sorted reads from the old memmap
            order = np.argsort(source[old], kind='stable')
            out[old[order]] = self.old[source[old][order]]
        elif len(old):
            self._gather(out, old, source[old], OLD)
        fresh = np.flatnonzero(kind == FRESH)
        if len(fresh):
            self._gather(out, fresh, source[fresh], FRESH)
        more = 0
        for k in self.kinds[2:]:                         # section 17.90: further block sets
            where = np.flatnonzero(kind == k)
            if len(where):
                self._gather(out, where, source[where], k)
                more += len(where)
        with self._lock:
            self.stats['old_rows'] += len(old)
            self.stats['fresh_rows'] += len(fresh)
            if self.more:
                self.stats['more_rows'] += more
        return out[0] if scalar else out

    def quantized(self, rows, out=None):
        """The rows' stored int8 values [n, 2, P, D] and fp16 scales [n, 2, P] (int8 stores only; see the module
        docstring). Blocks are read and decoded as by __getitem__ (packed: one ascending reader, the decode threads
        check and decode) and kept in the block cache as int8 values and scales. out: (values, scales) arrays to fill,
        e.g. page-locked ones (section 17.79)."""
        if not self.int8:
            raise V181DataContractError('only an int8 store keeps quantized rows')
        rows = np.asarray(rows)
        if rows.dtype.kind not in 'iu' or rows.ndim != 1:
            raise V181DataContractError('quantized rows take a one-dimensional integer array')
        rows = rows.astype(np.int64, copy=False)
        if len(rows) and (rows.min() < 0 or rows.max() >= self.shape[0]):
            raise IndexError('combined P0 token row out of range')
        if out is None:
            values = np.empty((len(rows), *ROW_SHAPE), np.int8)
            scales = np.empty((len(rows), *ROW_SHAPE[:-1]), np.float16)
        else:
            values, scales = out
            if (values.shape != (len(rows), *ROW_SHAPE) or values.dtype != np.int8
                    or scales.shape != (len(rows), *ROW_SHAPE[:-1]) or scales.dtype != np.float16):
                raise V181DataContractError('quantized output arrays have the wrong shape or type')
        kind, source = self.kind[rows], self.source_row[rows]
        for k in self.kinds:
            where = np.flatnonzero(kind == k)
            if len(where):
                self._gather_quantized(values, scales, where, source[where], k)
        with self._lock:
            self.stats['quantized_rows'] += len(rows)
        return values, scales

    def _gather_quantized(self, values, scales, where, position, kind):
        # section 17.79: the rows of a block are found by bisection and, when they are consecutive both in the block
        # and in the output (a window of consecutive moments), copied as one slice; the same assignments as before
        order = np.argsort(position // self.block_frames, kind='stable')
        where, position = where[order], position[order]
        start = position // self.block_frames * self.block_frames

        def place(value, pair):
            lo, hi = np.searchsorted(start, value), np.searchsorted(start, value, side='right')
            out, at = where[lo:hi], position[lo:hi] - value
            cut = np.flatnonzero((np.diff(out) != 1) | (np.diff(at) != 1)) + 1
            for a, b in zip(np.concatenate(([0], cut)), np.concatenate((cut, [len(out)]))):   # runs of consecutive rows
                o, t = slice(int(out[a]), int(out[b - 1]) + 1), slice(int(at[a]), int(at[b - 1]) + 1)
                values[o] = pair[0][t]
                scales[o] = pair[1][t]

        def finish(value, pair):
            self._remember(('int8', kind, value), pair, 'quantized_decodes')
            place(value, pair)

        reader = kind == FRESH and self.packed is not None    # packed: this thread reads in ascending order
        pending = OrderedDict()
        for value in (int(v) for v in np.unique(start)):
            pair = self._cached(('int8', kind, value), 'quantized_hits')
            if pair is not None:
                place(value, pair)
                continue
            raw = self.packed.read(value, check=False) if reader else None
            if self.decode_threads == 1:
                finish(value, self._decode_quantized(value, kind, raw, checked=raw is None))
                continue
            pending[value] = self._decoder().submit(self._decode_quantized, value, kind, raw, raw is None)
            while len(pending) >= 2 * self.decode_threads:
                done, future = pending.popitem(last=False)
                finish(done, future.result())
        while pending:
            done, future = pending.popitem(last=False)
            finish(done, future.result())

    def _decoder(self):
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(self.decode_threads)
            return self._pool

    def _gather(self, out, where, position, kind):
        start = position // self.block_frames * self.block_frames
        starts = [int(value) for value in np.unique(start)]
        if kind == FRESH and self.packed is not None:
            self._gather_packed(out, where, position, start, starts)
            return
        group = 2 * self.decode_threads                  # bounded: at most this many decoded blocks wait in memory
        for begin in range(0, len(starts), group):
            values = starts[begin:begin + group]
            decoded = {}
            if self.decode_threads > 1:
                missing = [value for value in values if self._cached((kind, value)) is None]
                if len(missing) > 1:
                    decoded = dict(zip(missing, self._decoder().map(self._decode, missing, [kind] * len(missing))))
                    for value in missing:
                        self._remember((kind, value), decoded[value])
            for value in values:
                block = decoded.get(value)
                if block is None:
                    block = self._block(value, kind)
                pick = start == value
                out[where[pick]] = block[position[pick] - value]

    def _gather_packed(self, out, where, position, start, starts):
        """Packed partition: this thread reads the missing blocks in ascending order (one stream on the disk) while
        the decode threads decode the blocks already read; at most 2 x decode_threads blocks wait in memory."""
        def place(value, block):
            pick = start == value
            out[where[pick]] = block[position[pick] - value]

        def finish(value, future):
            block = future.result()
            self._remember((FRESH, value), block)
            place(value, block)

        pending = OrderedDict()
        for value in starts:
            block = self._cached((FRESH, value))
            if block is not None:
                place(value, block)
                continue
            raw = self.packed.read(value)
            if self.decode_threads == 1:
                block = self._decode(value, FRESH, raw)
                self._remember((FRESH, value), block)
                place(value, block)
                continue
            pending[value] = self._decoder().submit(self._decode, value, FRESH, raw)
            while len(pending) >= 2 * self.decode_threads:
                finish(*pending.popitem(last=False))
        while pending:
            finish(*pending.popitem(last=False))

    def __array__(self, dtype=None, copy=None):
        raise V181DataContractError('refusing to materialize the whole combined P0 token array')


class P0FeatureCache(V181FeatureCache):
    """V181FeatureCache whose ``tokens`` are CombinedP0Tokens; metadata arrays are ordinary .npy files."""

    def __init__(self, root, *, expected_kind=None, cached_blocks=48, decode_threads=None):
        root = Path(root)
        config = json.loads((root / 'config.json').read_text())
        if config.get('view_order') != ['wrist', 'column'] or config.get('backbone') != ENCODER:
            raise V181DataContractError(f'combined P0 cache must keep the V18.1 view order and the process encoder {ENCODER}')
        self.root = root
        self.kind = str(config.get('kind', ''))
        if expected_kind is not None and self.kind != expected_kind:
            raise V181DataContractError(f'feature cache kind is {self.kind}, expected {expected_kind}')
        self.endpoint_index = np.load(root / 'endpoint_index.npy', mmap_mode='r')
        self.view_valid = np.load(root / 'view_valid.npy', mmap_mode='r')
        self.tokens = CombinedP0Tokens(root, cached_blocks=cached_blocks, decode_threads=decode_threads)
        rows = len(self.endpoint_index)
        if (self.endpoint_index.dtype != np.int64 or self.view_valid.shape != (rows, 2)
                or self.view_valid.dtype != np.bool_ or self.tokens.shape[0] != rows):
            raise V181DataContractError('combined P0 cache metadata shapes differ')
        if rows == 0 or bool((np.diff(self.endpoint_index) <= 0).any()):
            raise V181DataContractError('feature endpoint identities must be sorted unique')


def open_archive_cache(root, **kwargs):
    """The archive partition cache in ``root``: combined P0 when it has token sources, else the plain V18.1 cache."""
    root = Path(root)
    return P0FeatureCache(root, **kwargs) if (root / TOKEN_SOURCES).is_file() else V181FeatureCache(root)
