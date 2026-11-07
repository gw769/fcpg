"""AMP contexts for native40; master parameters, clocks and losses stay FP32/64."""
from contextlib import nullcontext

import torch


def compute_context(device, precision):
    if precision not in ('fp32', 'bf16'):
        raise ValueError('native40 precision must be fp32 or bf16')
    device = torch.device(device)
    if precision == 'bf16':
        if device.type != 'cuda':
            raise ValueError('native40 BF16 requires CUDA')
        # Historical LeWM forecasts run under no_grad before the supervised
        # forward. Cached casts from that path must not detach trainable weights.
        return torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False)
    return nullcontext()
