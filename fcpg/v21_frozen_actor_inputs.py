"""Cache factual Actor inputs while both FCPG and LeWM are frozen.

The cache stores the original tensors before all trainable Actor layers. It
does not store actions, losses, model projections, or generated future labels.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ActorTrainingInputs:
    inputs: dict
    commands: torch.Tensor
    valid: torch.Tensor


class FrozenActorInputBank:
    def __init__(self, core):
        self.core = core
        self.banks = {}
        self.resident_bytes = 0
        self.rows = 0
        self.hits = 0
        self._signature = self._versions()

    def _versions(self):
        if getattr(self.core, '_temporal_collect_training_coverage', False):
            raise ValueError('frozen input cache cannot replace active per-call coverage collection')
        modules = (self.core.reader, self.core.lewm)
        if any(module.training for module in modules) or any(
                p.requires_grad for module in modules for p in module.parameters()):
            raise ValueError('frozen Actor input cache requires eval and frozen FCPG/LeWM')
        tensors = [value for module in modules for value in (*module.parameters(), *module.buffers())]
        return tuple((value.data_ptr(), value._version) for value in tensors)

    def _check(self):
        if self._versions() != self._signature:
            raise ValueError('FCPG/LeWM changed after frozen Actor inputs were cached')

    @torch.no_grad()
    def build(self, kind, h, rows, prepare, *, batch_size=64, progress=None):
        self._check()
        if kind not in ('fish', 'archive') or h not in range(4) or (kind, h) in self.banks:
            raise ValueError('invalid or duplicate frozen Actor input bank')
        rows = np.unique(np.asarray(rows, dtype=np.int64))
        if not len(rows) or (rows < 0).any() or batch_size < 1:
            raise ValueError('frozen Actor input cache requires actual source rows')
        blocks = []
        for begin in range(0, len(rows), batch_size):
            selected = rows[begin:begin + batch_size]
            prepared = prepare(kind, selected, h, True)
            if prepared is None or len(prepared[0].q) != len(selected):
                raise ValueError('cached source row lacks actual supervised future')
            state, ids, target = prepared
            inputs = dict(self.core.actor_inputs(state), text_ids=ids, command_offsets=target.command_offsets)
            # Text encodings are trimmed to each source batch's longest task.
            # Padding remains masked and lets different batches share a bank.
            # an Actor without task-word tokens (V21Config.task_text=False) ignores them: any fixed width
            width = self.core.actor.text_position.shape[1] if getattr(self.core.actor, 'text_tokens', True) else 64
            inputs['text_ids'] = torch.nn.functional.pad(ids, (0, width - ids.shape[1]))
            if any(not isinstance(value, torch.Tensor) or value.requires_grad for value in inputs.values()):
                raise ValueError('cached Actor inputs must be frozen tensors')
            blocks.append(ActorTrainingInputs(inputs, target.commands, target.valid))
            if progress is not None and (begin == 0 or (begin // batch_size + 1) % 50 == 0):
                progress(min(begin + batch_size, len(rows)), len(rows))
        values = ActorTrainingInputs(
            {key: torch.cat([block.inputs[key] for block in blocks]) for key in blocks[0].inputs},
            torch.cat([block.commands for block in blocks]), torch.cat([block.valid for block in blocks]))
        index = np.full(int(rows[-1]) + 1, -1, dtype=np.int64)
        index[rows] = np.arange(len(rows))
        self.banks[(kind, h)] = (index, values)
        self.rows += len(rows)
        self.resident_bytes += sum(value.numel() * value.element_size()
            for value in (*values.inputs.values(), values.commands, values.valid))
        self.core.reader._projection_cache = None
        self._check()

    def get(self, kind, h, rows):
        self._check()
        rows = np.asarray(rows)
        if rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows):
            raise ValueError('cached Actor rows must retain their original integer identities')
        index, values = self.banks[(kind, h)]
        if (rows < 0).any() or (rows >= len(index)).any() or (index[rows] < 0).any():
            raise ValueError('Actor row was not included in the factual cache schedule')
        take = torch.as_tensor(index[rows], device=values.commands.device)
        self.hits += len(rows)
        return ActorTrainingInputs({key: value.index_select(0, take) for key, value in values.inputs.items()},
            values.commands.index_select(0, take), values.valid.index_select(0, take))

    def stats(self):
        return dict(rows=self.rows, resident_bytes=self.resident_bytes, banks=len(self.banks), hits=self.hits)
