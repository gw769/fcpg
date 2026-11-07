"""Causal access to complete recorded command streams, separate from vision."""
import torch
from torch import nn

LEGACY_COMMANDS = 'sampled-active-v1'
FULL_COMMANDS = 'causal-command-tokens-v1'
COMMAND_MODES = (LEGACY_COMMANDS, FULL_COMMANDS)


def command_mode_from_checkpoint(payload):
    """Never silently load trained complete-command weights as the old model."""
    mode = payload.get('lewm_command_conditioning', LEGACY_COMMANDS)
    if mode not in COMMAND_MODES:
        raise ValueError('unsupported LeWM command conditioning')
    if payload.get('native40_protocol', {}).get('lewm_command_conditioning', mode) != mode:
        raise ValueError('command conditioning metadata disagrees with checkpoint')
    has_weights = any(k.startswith(('lewm.command_projection.', 'lewm.command_readers.'))
                      for k in payload['model'])
    if has_weights != (mode == FULL_COMMANDS):
        raise ValueError('command conditioning metadata disagrees with command weights')
    return mode


class CausalCommandRead(nn.Module):
    def __init__(self, width=384, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=.1, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(width))

    def forward(self, query, commands, forbidden):
        update = self.attention(self.norm(query), commands, commands,
                                attn_mask=forbidden, need_weights=False)[0]
        return query + self.gate.tanh() * update
