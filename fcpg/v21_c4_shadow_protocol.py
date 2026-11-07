"""Wire format between the C4 shadow tap and its worker (docs/V21_LEWM_C4_STEP2_20261010.md section 3).

Stdlib + numpy only. One message = a 4-byte big-endian header length, a UTF-8 JSON header, then the raw
bytes of every array the header lists (C order, in that order). No pickle: the tap and the worker run in
different interpreters (lerobot's environment and the LeWM venv, with different numpy versions), and
nothing executable crosses the pipe.
"""
import json
import struct

import numpy as np

MAGIC = 'v21-c4-shadow-v1'
DTYPES = ('uint8', 'bool', 'int64', 'float32', 'float64')
MAX_HEADER_BYTES = 1 << 20
MAX_ARRAY_BYTES = 64 << 20


def encode(header, arrays=(), *, copy=True):
    """header (JSON-safe dict) + [(name, array)] -> byte chunks to write in order.

    copy=True: every array is copied into an immutable bytes object, so later changes to the caller's arrays
    cannot reach the message. copy=False: the chunks are views of the arrays; the caller must not change them
    until the chunks are written (the tap's writer thread owns its frame buffers until then).
    NaN or infinity in the header raises (ValueError)."""
    specs, payload = [], []
    for name, array in arrays:
        a = np.ascontiguousarray(array)
        if a.dtype.name not in DTYPES or a.nbytes > MAX_ARRAY_BYTES:
            raise ValueError(f'array {name!r}: dtype {a.dtype.name} with {a.nbytes} bytes is not allowed')
        specs.append(dict(name=str(name), dtype=a.dtype.name, shape=[int(s) for s in a.shape]))
        payload.append(a.tobytes() if copy or not a.nbytes else memoryview(a).cast('B'))
    head = json.dumps(dict(header, magic=MAGIC, arrays=specs), allow_nan=False,
                      separators=(',', ':')).encode('utf-8')
    if len(head) > MAX_HEADER_BYTES:
        raise ValueError('header too large')
    return [struct.pack('>I', len(head)), head, *payload]


def _read_exact(stream, n, *, eof_ok=False):
    chunks, need = [], n
    while need:
        chunk = stream.read(need)
        if not chunk:
            if eof_ok and need == n:
                return None
            raise EOFError('truncated C4 shadow message')
        chunks.append(chunk)
        need -= len(chunk)
    return b''.join(chunks)


def decode(stream):
    """-> (header, {name: read-only array}), or None at a clean end of the stream."""
    size = _read_exact(stream, 4, eof_ok=True)
    if size is None:
        return None
    (n,) = struct.unpack('>I', size)
    if not 0 < n <= MAX_HEADER_BYTES:
        raise ValueError('bad C4 shadow header length')
    header = json.loads(_read_exact(stream, n).decode('utf-8'))
    if header.pop('magic', None) != MAGIC:
        raise ValueError('not a C4 shadow message')
    arrays = {}
    for spec in header.pop('arrays'):
        if spec['dtype'] not in DTYPES:
            raise ValueError(f'dtype {spec["dtype"]} is not allowed')
        dtype, shape = np.dtype(spec['dtype']), tuple(int(s) for s in spec['shape'])
        if any(s < 0 for s in shape):
            raise ValueError('negative array dimension')
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes > MAX_ARRAY_BYTES:
            raise ValueError('array too large')
        arrays[spec['name']] = np.frombuffer(_read_exact(stream, nbytes), dtype=dtype).reshape(shape)
    return header, arrays
