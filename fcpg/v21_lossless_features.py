"""Independent lossless feature blocks; no quantization or frame selection.

Use the system libzstd on both hosts, without installing a training dependency.
Byte shuffle only changes byte order and is undone before returning an array.
"""
import ctypes
import ctypes.util
import json
import struct

import numpy as np


MAGIC = b'V21ZST01'
MAX_BLOCK_BYTES = 256 * 1024**2


class LosslessFeatures:
    def __init__(self, *, level=1, shuffle=True):
        library = ctypes.util.find_library('zstd')
        if not library:
            raise RuntimeError('system libzstd is unavailable')
        self.lib = ctypes.CDLL(library)
        size, pointer = ctypes.c_size_t, ctypes.c_void_p
        for name, args, result in (
            ('ZSTD_compressBound', [size], size),
            ('ZSTD_compress', [pointer, size, pointer, size, ctypes.c_int], size),
            ('ZSTD_decompress', [pointer, size, pointer, size], size),
            ('ZSTD_isError', [size], ctypes.c_uint),
            ('ZSTD_getErrorName', [size], ctypes.c_char_p),
        ):
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        if level not in (1, 3):
            raise ValueError('bounded compression levels are 1 and 3')
        self.level, self.shuffle = level, bool(shuffle)

    def checked(self, value):
        if self.lib.ZSTD_isError(value):
            raise ValueError(self.lib.ZSTD_getErrorName(value).decode())
        return value

    def encode(self, array):
        array = np.ascontiguousarray(array)
        if array.dtype not in (np.dtype('<f2'), np.dtype('i1')) or not 0 < array.nbytes <= MAX_BLOCK_BYTES:
            raise ValueError('bounded FP16 or INT8 storage block required')
        shuffled = self.shuffle and array.dtype.itemsize > 1
        raw = (array.view(np.uint8).reshape(-1, array.dtype.itemsize).T.copy().tobytes()
               if shuffled else array.tobytes())
        header = json.dumps(dict(shape=array.shape, dtype=array.dtype.str, raw_bytes=array.nbytes,
                                 byte_shuffle=shuffled, level=self.level), separators=(',', ':')).encode()
        capacity = self.checked(self.lib.ZSTD_compressBound(len(raw)))
        buffer = ctypes.create_string_buffer(capacity)
        length = self.checked(self.lib.ZSTD_compress(buffer, capacity, raw, len(raw), self.level))
        return MAGIC + struct.pack('<I', len(header)) + header + buffer.raw[:length]

    def decode(self, payload):
        if len(payload) < 12 or payload[:8] != MAGIC:
            raise ValueError('not a V21 lossless block')
        header_size = struct.unpack('<I', payload[8:12])[0]
        if not 0 < header_size <= 4096 or 12 + header_size >= len(payload):
            raise ValueError('invalid feature block header')
        header = json.loads(payload[12:12 + header_size])
        dtype, shape = np.dtype(header['dtype']), header['shape']
        if (dtype not in (np.dtype('<f2'), np.dtype('i1')) or not isinstance(shape, list)
                or not 1 <= len(shape) <= 6 or any(type(v) is not int or v <= 0 for v in shape)):
            raise ValueError('invalid feature dtype or shape')
        count = 1
        for value in shape:
            count *= value
        size = count * dtype.itemsize
        if size != header['raw_bytes'] or not 0 < size <= MAX_BLOCK_BYTES or type(header['byte_shuffle']) is not bool:
            raise ValueError('invalid decoded size or shuffle flag')
        compressed = payload[12 + header_size:]
        buffer = ctypes.create_string_buffer(size)
        actual = self.checked(self.lib.ZSTD_decompress(buffer, size, compressed, len(compressed)))
        if actual != size:
            raise ValueError('feature block decoded to the wrong length')
        raw = np.frombuffer(buffer.raw, np.uint8)
        if header['byte_shuffle']:
            raw = raw.reshape(dtype.itemsize, -1).T.copy().reshape(-1)
        return raw.view(dtype).reshape(shape).copy()
