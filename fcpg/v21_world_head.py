"""The LeWM half of one model: Actor and world model trained together on the FCPG reader they share (training overhaul
doc section 17.113; James 10/27 「训练的时候可以actor+lewn合二为一吗？」, then 「做吧」).

From the FCPG reader's tokens of the wrist view now (the very tokens the Actor reads), the pose now and one block of
five commands, the head predicts what the block will do, and the five joint read-backs (the system LeWM's job). It
reads no picture of its own: whatever it needs about the scene has to come through the reader, so its loss reaches the
reader the Actor shares. Two targets (V21Config.world_target):
  'tokens' (second version, the one trained): the reader's own wrist tokens of the next state (frame h + 1, read
      without gradient) -- the LeWM / JEPA objective in the Actor's own representation. The reader's wrist tokens
      carry SIGReg, as a LeWM's embeddings do, so predictability cannot come from collapsing them.
  'patches' (first version): the change of the wrist picture per patch in fixed principal components of the frozen
      DINOv3-L patches (17.112 step 1). Its smoke showed the flaw: eight summary tokens do not say what each patch
      holds, so per-patch changes cannot be predicted from them (1% explained after 300 updates against 22% for the
      step-1 head that read the patches).
"""
from __future__ import annotations

import torch
from torch import nn

from .v21_encoder import DIM, PATCHES

MOTION = 36            # the pose (6) and the five commands minus the pose (30)
READER = 192           # the FCPG reader's token width
TARGETS = ('patches', 'tokens')


class WorldHead(nn.Module):
    def __init__(self, components=64, width=256, layers=2, heads=8, target='patches', queries=8):
        super().__init__()
        if target not in TARGETS:
            raise ValueError(f'world target is one of {TARGETS}')
        self.target = target
        self.components = int(components)
        self.memory_in = nn.Linear(READER, width)
        self.motion = nn.Sequential(nn.Linear(MOTION, width), nn.GELU(), nn.Linear(width, width))
        self.joint_query = nn.Parameter(torch.randn(1, 1, width) * .02)
        layer = lambda: nn.TransformerDecoderLayer(width, heads, 4 * width, dropout=0., batch_first=True,
                                                   norm_first=True)
        self.layers = nn.ModuleList(layer() for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.joints = nn.Linear(width, 30)
        if target == 'tokens':
            self.token_in = nn.Linear(READER, width)
            self.token_index = nn.Parameter(torch.randn(1, int(queries), width) * .02)
            self.token_out = nn.Linear(width, READER)
            outputs = (self.joints, self.token_out)
        else:
            self.patch_query = nn.Parameter(torch.randn(1, PATCHES, width) * .02)
            self.change = nn.Linear(width, self.components)
            outputs = (self.joints, self.change)
        for module in outputs:                       # every prediction starts at "nothing changes"
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        # fixed before training (fit_basis: 'patches' only; set_motion_scale), saved with the weights
        self.register_buffer('basis_mean', torch.zeros(DIM))
        self.register_buffer('basis', torch.zeros(DIM, self.components))
        self.register_buffer('motion_mean', torch.zeros(MOTION))
        self.register_buffer('motion_scale', torch.ones(MOTION))
        self.register_buffer('joint_scale', torch.ones(6))

    @torch.no_grad()
    def fit_basis(self, patches):
        """patches [n, DIM]: the wrist patches the basis is fitted on (training frames only). -> share of the patch
        variance the kept components hold."""
        patches = patches.float()
        if patches.ndim != 2 or patches.shape[1] != DIM or len(patches) < self.components:
            raise ValueError('the basis is fitted on [n, DIM] patches, at least one per component')
        mean = patches.mean(0)
        values, vectors = torch.linalg.eigh(torch.cov((patches - mean).T))
        self.basis_mean.copy_(mean)
        self.basis.copy_(vectors[:, -self.components:].flip(-1))
        return float(values[-self.components:].sum() / values.sum())

    @torch.no_grad()
    def set_motion_scale(self, motion, joint_moves):
        """motion [n, 36] (motion_input of training blocks), joint_moves [n, 5, 6] (read-backs minus the pose)."""
        self.motion_mean.copy_(motion.float().mean(0))
        self.motion_scale.copy_(motion.float().std(0).clamp_min(1e-3))
        self.joint_scale.copy_(joint_moves.float().abs().flatten(0, 1).mean(0).clamp_min(1e-2))

    def project(self, patches):
        """[..., PATCHES, DIM] frozen picture patches -> [..., PATCHES, components] ('patches' target)."""
        return (patches.float() - self.basis_mean) @ self.basis

    @staticmethod
    def motion_input(q, commands):
        """q [B, 6], commands [B, 5, 6] (absolute) -> [B, 36]."""
        return torch.cat((q, (commands - q[:, None]).flatten(1)), 1)

    def forward(self, wrist_tokens, q, commands):
        """wrist_tokens [B, Q, 192] (the reader's wrist view now), q [B, 6], commands [B, 5, 6] -> (prediction,
        joint read-backs [B, 5, 6] (absolute), hidden: the query states before the last LayerNorm). The prediction is
        the next wrist tokens [B, Q, 192] ('tokens': the tokens now plus a change) or the per-patch change
        [B, PATCHES, components] ('patches')."""
        b = len(q)
        motion = (self.motion_input(q.float(), commands.float()) - self.motion_mean) / self.motion_scale
        tokens = wrist_tokens.float()
        memory = torch.cat((self.motion(motion)[:, None], self.memory_in(tokens)), 1)
        own = (self.token_in(tokens) + self.token_index if self.target == 'tokens'
               else self.patch_query.expand(b, -1, -1))
        query = torch.cat((self.joint_query.expand(b, -1, -1), own), 1)
        for layer in self.layers:
            query = layer(query, memory)
        out = self.norm(query)
        joints = q.float()[:, None] + self.joints(out[:, 0]).view(b, 5, 6) * self.joint_scale
        if self.target == 'tokens':
            return tokens + self.token_out(out[:, 1:]), joints, query
        return self.change(out[:, 1:]), joints, query


def change_loss(predicted, target, valid, current=None):
    """Squared error of the prediction over the squared change itself (the error of "nothing changes": the target
    for a change, the target minus current for next tokens), on the valid rows [B] bool: 1 = no better than nothing
    changes, 0 = perfect. -> (loss, share explained); both tensors, nothing waits for the GPU; no valid row gives a
    zero loss that keeps the graph and a NaN share."""
    weight = valid.to(predicted.device, torch.float32).reshape(-1, *([1] * (predicted.ndim - 1)))
    error = ((predicted.float() - target.float()).square() * weight).sum()
    change = target.float() if current is None else target.float() - current.detach().float()
    energy = (change.square() * weight).sum()
    loss = torch.where(energy > 0, error / energy.clamp_min(1e-6), error * 0.)
    explained = torch.where(energy > 0, 1. - loss.detach(), torch.full_like(energy, float('nan')))
    return loss, explained
