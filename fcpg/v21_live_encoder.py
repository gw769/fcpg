"""The model's picture tokens in a live service and its offline replays, as the model was trained on them.

fcpg/v21_encoder.ENCODER is the encoder of the process. DINOv2-S/14 (the default) is the service's own backbone
(TrialOwner.dino, fcpg.prod_v18_1_training.extract_dinov2_tokens), as before 10/21. DINOv3-L/16 (V21_ENCODER=dinov3_vitl16;
training overhaul doc section 17.55) is the recipe of the training features (scripts/build_v21_dinov3_features.py: timm
from the local weights, fp32 weights with fp16 autocast, the prefix tokens dropped, every token int8 with its own fp16
scale), read back as fcpg/v21_p0_features.py serves them (float32 product -> float16): the model sees the numbers it
was trained on. analysis/v21/robot_states_dinov3l.py makes our stored robot states with the same functions.
The learned finders and trackers keep reading DINOv2 tokens (TrialOwner.last_dino_tokens) until they are retrained on
DINOv3-L: an L service runs both encoders. With V21_DINOV2=off (section 17.57) it runs only DINOv3-L, and
forbid_dinov2() makes the DINOv2 entry points raise.
"""
from pathlib import Path

import numpy as np

from .v21_encoder import DIM, DINOV2_OFF, ENCODER, PATCHES

L_WEIGHTS = Path('/home/james/so101-c4-step2/models/dinov3/vit_large_patch16_dinov3.lvd1689m/model.safetensors')


def served(tokens, scales):
    """int8 tokens [..., DIM] and their fp16 scales [...] -> float16 values, as fcpg/v21_p0_features.py serves them."""
    return (np.asarray(tokens).astype(np.float32) * np.asarray(scales).astype(np.float32)[..., None]).astype(np.float16)


def dinov3_int8_with_prefix(model, pictures, device, batch_size):
    """scripts/build_v21_dinov3_features.dinov3_int8 step for step, also returning the prefix tokens (class + 4 registers)
    of the same forward pass as float16 [m, 5, DIM], which the task-text finder's vision head reads
    (fcpg/v21_text_finder.py); the int8 tokens and scales are dinov3_int8's bit for bit (tests/test_v21_text_finder.py)."""
    import torch
    from scripts.build_v21_dinov3_features import MEAN, PREFIX, STD
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    tokens, scales, prefix = [], [], []
    with torch.inference_mode():
        for start in range(0, len(pictures), batch_size):
            x = torch.from_numpy(pictures[start:start + batch_size]).to(device).permute(0, 3, 1, 2).float().div_(255.)
            with torch.autocast('cuda', dtype=torch.float16):
                features = model.forward_features((x - mean) / std)
            t = features[:, PREFIX:].float()
            if t.shape[1:] != (PATCHES, DIM) or not bool(torch.isfinite(t).all()):
                raise ValueError(f'DINOv3-L patch tokens {tuple(t.shape)} or not finite')
            scale = (t.abs().amax(-1, keepdim=True) / 127.).clamp_min(1e-6).half()
            tokens.append(torch.round(t / scale.float()).clamp_(-127, 127).to(torch.int8).cpu())
            scales.append(scale.squeeze(-1).cpu())
            prefix.append(features[:, :PREFIX].float().half().cpu())
    return torch.cat(tokens).numpy(), torch.cat(scales).numpy(), torch.cat(prefix).numpy()


class ModelTokens:
    """encode(rgb uint8 [n, 224, 224, 3] letterbox) -> float16 [n, PATCHES, DIM], the tokens this process's model reads.
    With DINOv3-L, last_prefix keeps the same pictures' prefix tokens [n, 5, DIM] (float16) for the task-text finder."""

    def __init__(self, device, *, dinov2=None, weights=L_WEIGHTS, batch_size=2):
        self.device, self.batch_size, self.encoder = device, int(batch_size), ENCODER
        self.last_prefix = None
        if ENCODER == 'dinov2_vits14':
            if dinov2 is None:
                raise ValueError('a DINOv2 process encodes with the service backbone (dinov2=)')
            self.dinov2 = dinov2
        else:
            from scripts.build_v21_dinov3_features import load_dinov3
            self.weights = Path(weights)
            self.model = load_dinov3(self.weights, device)

    def encode(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.ndim != 4 or rgb.shape[1:] != (224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError('the model tokens are made from uint8 letterbox pictures [n, 224, 224, 3]')
        if self.encoder == 'dinov2_vits14':
            from .prod_v18_1_training import extract_dinov2_tokens
            out = np.asarray(extract_dinov2_tokens(self.dinov2, rgb, device=self.device, batch_size=self.batch_size))
        else:
            tokens, scales, self.last_prefix = dinov3_int8_with_prefix(self.model, rgb, self.device, self.batch_size)
            out = served(tokens, scales)
        if out.shape != (len(rgb), PATCHES, DIM) or out.dtype != np.float16:
            raise ValueError(f'model tokens {out.shape} {out.dtype}, expected [n, {PATCHES}, {DIM}] float16')
        return out

    def describe(self):
        return dict(encoder=self.encoder, patches=PATCHES, dim=DIM,
                    **({} if self.encoder == 'dinov2_vits14' else dict(weights=str(self.weights),
                       recipe='scripts/build_v21_dinov3_features.py dinov3_int8, served as fcpg/v21_p0_features.py')))


class DinoV2Off(RuntimeError):
    """A DINOv2 entry point was reached in a process run with V21_DINOV2=off (training overhaul doc section 17.57)."""


def forbid_dinov2():
    """V21_DINOV2=off: make every DINOv2 entry point of the live path raise DinoV2Off (the backbone loader and token
    call of fcpg.prod_v18_1_training, and the frozen-feature grounder of the old DINO target tracker,
    fcpg.so101_frozen_vit_grounding.FrozenViTGrounder), so no part of the service uses DINOv2 unnoticed. Names imported
    before this call keep the original object: the owner branches on DINOV2_OFF itself. Returns the names forbidden."""
    if not DINOV2_OFF:
        raise RuntimeError('forbid_dinov2() is for a V21_DINOV2=off process')
    import fcpg.prod_v18_1_training as backbone
    import fcpg.so101_frozen_vit_grounding as grounding

    def refuse(name):
        def raising(*args, **kwargs):
            raise DinoV2Off(f'{name}: this process runs without DINOv2 (V21_DINOV2=off, section 17.57)')
        raising.dinov2_forbidden = True
        return raising
    forbidden = []
    for name in ('load_frozen_dinov2', 'extract_dinov2_tokens'):
        if not getattr(getattr(backbone, name), 'dinov2_forbidden', False):
            setattr(backbone, name, refuse(f'fcpg.prod_v18_1_training.{name}'))
        forbidden.append(f'fcpg.prod_v18_1_training.{name}')
    if not getattr(grounding.FrozenViTGrounder.__init__, 'dinov2_forbidden', False):
        grounding.FrozenViTGrounder.__init__ = refuse('fcpg.so101_frozen_vit_grounding.FrozenViTGrounder')
    forbidden.append('fcpg.so101_frozen_vit_grounding.FrozenViTGrounder')
    return forbidden
