"""The frozen picture encoder the v21 models read: one per process, chosen before any model module is imported.

V21_ENCODER=dinov2_vits14 (default: every model before 10/21; 16 x 16 patches of 384 from the 224 letterbox) or
dinov3_vitl16 (James 10/21 「DINOv3 L，8 位 全部替换成这个」: 14 x 14 patches of 1024 from the same 224 letterbox,
fp16 autocast, stored as int8 with a per-token fp16 scale, scripts/build_v21_dinov3_features.py). The model modules
take GRID / PATCHES / DIM from here at import time and keep their internal widths: the FCPG reader projects DIM -> 192,
both LeWMs project DIM -> 384 before their 384-wide vision layers (an identity for DINOv2, so earlier checkpoints load
unchanged; the NERO sim port does the same). V21Config.encoder records the encoder and the core refuses another one.
V21_DINOV2=off (training overhaul doc section 17.57, James 10/22 「都用阿」; only with a DINOv3-L encoder) is a process
that never computes DINOv2: a live service whose finders, box finder and fish token all read DINOv3-L. The service
makes every DINOv2 entry point raise (fcpg/v21_live_encoder.forbid_dinov2), so nothing falls back to it unnoticed.
"""
import os

import torch

ENCODERS = dict(dinov2_vits14=(16, 384), dinov3_vitl16=(14, 1024))
ENCODER = os.environ.get('V21_ENCODER', 'dinov2_vits14')
if ENCODER not in ENCODERS:
    raise ValueError(f'V21_ENCODER must be one of {sorted(ENCODERS)}, not {ENCODER!r}')
GRID, DIM = ENCODERS[ENCODER]
PATCHES = GRID * GRID
DINOV2 = os.environ.get('V21_DINOV2', 'on')
if DINOV2 not in ('on', 'off'):
    raise ValueError(f"V21_DINOV2 must be 'on' or 'off', not {DINOV2!r}")
if DINOV2 == 'off' and ENCODER == 'dinov2_vits14':
    raise ValueError('V21_DINOV2=off needs another encoder (V21_ENCODER=dinov3_vitl16)')
DINOV2_OFF = DINOV2 == 'off'
# The extraction recipe a DINO cache of this encoder declares (fcpg/v21_clock_cache.py checks it on every partition).
EXTRACTOR_VERSIONS = dict(dinov2_vits14='dinov2-s14-rgb-letterbox224-imagenet-fp16-v1',
                          dinov3_vitl16='dinov3-l16-rgb-letterbox224-imagenet-fp16-int8-v1')


def encoder_of(patches, dim):
    """The encoder whose tokens are [patches, dim] (a saved model or a token file says which by its shape)."""
    for name, (grid, width) in ENCODERS.items():
        if grid * grid == int(patches) and width == int(dim):
            return name
    raise ValueError(f'no known encoder makes tokens [{patches}, {dim}]')


def patch_xy():
    """Each patch centre in the picture, x and y in [-1, 1], row-major: [PATCHES, 2]."""
    y, x = torch.meshgrid(torch.linspace(-1, 1, GRID), torch.linspace(-1, 1, GRID), indexing='ij')
    return torch.stack((x.flatten(), y.flatten()), -1)
