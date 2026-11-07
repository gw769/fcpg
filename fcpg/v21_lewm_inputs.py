"""The LeWM's inputs after h factual blocks, without the work only the Actor reads.

factual_state runs the FCPG reader twice per observation and one LeWM forecast per completed block (plus a second one
when a block is retimed). Their results (grounding, visual_error, q_error) are Actor inputs; the LeWM reads none of them:
it reads q, the completed history, the visual memory and the time, which the state machine builds from the chain alone
(v21_core.observe / complete). For LeWM training and LeWM evaluation this runs the same state machine on a shallow copy
of the core whose reader and LeWM are zero-output stand-ins of the right shapes, and hands over only those four.
tests/test_v21_lewm_inputs.py checks them element for element against factual_state, and the LeWM output with them.
Idea from James's NERO sim session (10/21, ~1.9x per LeWM training step there); James 10/21 「加速照他的」.
"""
import copy
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from .v21_horizon_training import factual_state


@dataclass(frozen=True)
class LeWMInputs:
    q: torch.Tensor
    history: object
    memory: object
    time: torch.Tensor


class _ZeroReader(nn.Module):
    def __init__(self, queries):
        super().__init__()
        self.queries_per_view = queries

    def forward(self, memory, actual_q, now, history, *, address_mode=None, task_ids=None):
        return actual_q.new_zeros(len(actual_q), 2, self.queries_per_view, 192)


class _ZeroLeWM(nn.Module):
    reads_visual_memory = False
    retime_invariant = True     # 17.173: zeros at any clock, so complete() keeps them without asking the GPU (17.80)

    def forward(self, q, trace, history, query_offsets, **_):
        return SimpleNamespace(predicted_q_path=q.new_zeros(len(q), query_offsets.shape[1] + 1, q.shape[-1]))


def lewm_inputs(core, chain, h):
    """q / history / memory / time exactly as factual_state(core, chain, h, images=True) leaves them."""
    shadow = copy.copy(core)                   # shares config, temporal contract and buffers; own module table
    shadow._modules = dict(core._modules, reader=_ZeroReader(core.reader.queries_per_view), lewm=_ZeroLeWM())
    with torch.no_grad():
        state = factual_state(shadow, chain, h, images=True)
    return LeWMInputs(state.q, state.history, state.memory, state.time)
