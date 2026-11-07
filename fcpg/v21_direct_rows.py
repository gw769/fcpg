"""Row reads of a large memory-mapped .npy without page-cache build-up.

Training reads random rows of the frozen DINO cache (194 GB, far more than RAM). Through the memory map every
row arrives by page faults, one request at a time, and fills the page cache; once memory is full the kernel has to
reclaim mapped pages for every new one and the read rate halves (measured 2026-10-12: 0.36 -> 0.7 s per update).
DirectRows reads the same bytes with parallel ``pread`` calls into the result array and tells the kernel the
pages are not needed again, so the read rate stays flat. Values are bit-identical to ``memmap[rows]``.
Only 1-D integer row arrays take this path; any other index goes to the memory map unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np


class DirectRows:
    PIECE_ROWS = 32

    def __init__(self, memmap, *, threads=8):
        if not isinstance(memmap, np.memmap) or memmap.ndim < 2 or not memmap.flags['C_CONTIGUOUS']:
            raise ValueError('DirectRows wraps a C-ordered np.memmap of rows')
        self.memmap = memmap
        self.shape, self.dtype, self.ndim = memmap.shape, memmap.dtype, memmap.ndim
        self._row_bytes = int(np.prod(memmap.shape[1:])) * memmap.dtype.itemsize
        self._offset = int(memmap.offset)
        self._fd = os.open(memmap.filename, os.O_RDONLY)
        if os.fstat(self._fd).st_size < self._offset + self.shape[0] * self._row_bytes:
            raise ValueError('file is shorter than the mapped array')
        self._pool = ThreadPoolExecutor(threads)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        if not (isinstance(rows, np.ndarray) and rows.ndim == 1 and rows.dtype.kind in 'iu'):
            return self.memmap[rows]
        rows = rows.astype(np.int64)
        if len(rows) and (rows.min() < -self.shape[0] or rows.max() >= self.shape[0]):
            raise IndexError('row out of range')
        rows = np.where(rows < 0, rows + self.shape[0], rows)
        out = np.empty((len(rows), *self.shape[1:]), self.dtype)
        if not len(rows):
            return out
        buffer = memoryview(out).cast('B')
        size = self._row_bytes
        # consecutive rows are one request (sequential on disk), cut into pieces the threads share
        breaks = np.flatnonzero(np.diff(rows) != 1) + 1
        pieces = [(c, min(c + self.PIECE_ROWS, b))
                  for a, b in zip(np.concatenate(([0], breaks)), np.concatenate((breaks, [len(rows)])))
                  for c in range(int(a), int(b), self.PIECE_ROWS)]

        def read(piece):
            a, b = piece
            position, length, done = self._offset + int(rows[a]) * size, (b - a) * size, 0
            while done < length:
                n = os.preadv(self._fd, [buffer[a * size + done:b * size]], position + done)
                if n <= 0:
                    raise IOError('short read of cached feature rows')
                done += n
            os.posix_fadvise(self._fd, position, length, os.POSIX_FADV_DONTNEED)

        list(self._pool.map(read, pieces))
        return out

    def close(self):
        self._pool.shutdown(wait=True)
        os.close(self._fd)
