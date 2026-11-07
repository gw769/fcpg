"""Section 17.65: an int8 store's rows multiplied on the GPU.

The DINOv3-L store (fcpg/v21_p0_features.py) keeps int8 values with fp16 per-token scales and serves them as the
float16 values int8 x scale. On .102's CPU that multiplication is the slow part of a read (about 0.1 s per 64-row block,
and it hardly runs in parallel), so a GPU reader takes the int8 values and the scales (``CombinedP0Tokens.quantized``,
half the bytes of float16) and multiplies them on the GPU. The float32 product of an int8 and a float16 is exact, so
rounding it once to float16 gives bit for bit what the store serves on the CPU.
"""
from __future__ import annotations

import numpy as np
import torch

CHUNK_ROWS = 256        # rows multiplied at a time: a float32 row is 1.6 MB, so at most about 0.4 GB of scratch


def dequantize(values, scales):
    """int8 values [..., D] times fp16 per-token scales [...] as float16, on the tensors' device."""
    return (values.to(torch.float32) * scales.to(torch.float32).unsqueeze(-1)).to(torch.float16)


def dequantizes_on_device(device_cache, tokens):
    """True when the rows of ``tokens`` that a row cache (fcpg/prod_v20_1_acceleration.TensorRowBlockCache) holds are
    multiplied on its GPU: the cache is on a GPU and the store keeps int8 rows."""
    return (device_cache is not None and torch.device(device_cache.device).type == 'cuda'
            and bool(getattr(tokens, 'int8', False)))


def quantized_rows_on_device(tokens, rows, device, view_valid, *, error=ValueError):
    """``tokens[rows]`` of an int8 store (CombinedP0Tokens) as a float16 tensor on ``device``, multiplied there, with the
    row cache's integrity contract: finite values, and the views ``view_valid`` marks absent are all zero."""
    rows = np.asarray(rows, np.int64)
    values, scales = tokens.quantized(rows)
    out = torch.empty(values.shape, dtype=torch.float16, device=device)
    for begin in range(0, len(rows), CHUNK_ROWS):
        part = slice(begin, begin + CHUNK_ROWS)
        out[part] = dequantize(torch.from_numpy(values[part]).to(device), torch.from_numpy(scales[part]).to(device))
    absent = torch.as_tensor(~np.asarray(view_valid[rows], bool), device=device)
    if not torch.isfinite(out).all() or (out[absent] != 0).any():
        raise error('frozen DINO data are nonfinite or invalid views are nonzero')
    return out
