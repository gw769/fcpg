"""Contiguous resident *all-row* motion data, not a sampling/LRU policy.

This startup cache reads each source row exactly once in bounded CPU blocks.
It stores only real q, sent commands, measured paths, times and source IDs.
Text, images, inferred pixels and fabricated scene summaries are not present.
The original Actor visual path is deliberately unaffected by this module.
"""
from __future__ import annotations

from collections.abc import Mapping
import numbers

import numpy as np
import torch


GIB = 2**30
MOTION_SHAPES = {
    "q_position": (5, 6),
    "commands_position": (4, 5, 6),
    "measured_q_paths_position": (4, 6, 6),
    "time_deltas": (4, 5),
    "source_index": (),
}
OPTIONAL_SHAPES = {"initial_time_s": ()}


def estimate_motion_bank_bytes(count, *, time_dtype=torch.float64,
                               include_initial_time=True):
    """Conservative known-schema estimate; 320K rows need <0.41 GiB.

    Public positions are fp32. Actual input time dtype is retained, not forced
    to this estimate's default. The constructor enforces the *actual* bytes.
    """
    if not isinstance(count, numbers.Integral) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if time_dtype not in (torch.float32, torch.float64):
        raise ValueError("time dtype must be float32 or float64")
    time_bytes = 4 if time_dtype == torch.float32 else 8
    return int(count) * ((5*6 + 4*5*6 + 4*6*6)*4 + 4*5*time_bytes + 8 +
                         (8 if include_initial_time else 0))


class ResidentMotionBank:
    """Read-only contiguous tensors for the complete declared row population.

    ``loader(ids)`` must return CPU tensors/NumPy arrays from the unchanged
    source chain contract. Only whitelisted motion fields are retained.
    Constructors do not change global RNG, CUDA device selection, sampling,
    source data, optimizers, other caches or any process. CUDA callers must
    name a device explicitly (e.g. cuda:0 after CUDA_VISIBLE_DEVICES=1).

    One full allocation per key is filled in bounded blocks, avoiding a list
    of full-data chunks followed by a second ``cat`` allocation. A CUDA bank
    performs one final selected-stream wait before advertising readiness.
    ``max_bytes`` limits resident tensor storage; one loader block is extra
    CPU staging, and one requested batch is extra gather output.
    """
    def __init__(self, loader, count, device, *, max_bytes=GIB, block_rows=8192,
                 pin_memory=True):
        if not callable(loader):
            raise TypeError("loader must be callable")
        if not isinstance(count, numbers.Integral) or isinstance(count, bool) or count < 1:
            raise ValueError("count must be a positive integer; no partial population inferred")
        if not isinstance(block_rows, numbers.Integral) or isinstance(block_rows, bool) or block_rows < 1:
            raise ValueError("block_rows must be a positive integer")
        if not isinstance(max_bytes, numbers.Integral) or isinstance(max_bytes, bool) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda") or (self.device.type == "cuda" and self.device.index is None):
            raise ValueError("bank requires cpu or an explicit indexed CUDA device")
        self.count, self.max_bytes, self.block_rows = int(count), int(max_bytes), int(block_rows)
        self.pin_memory = bool(pin_memory)
        self.tensors = {}
        self.resident_bytes = 0
        self.blocks_loaded = self.requests = self.requested_rows = 0
        self.ready = False
        # Fail a definitely undersized budget before calling the source.
        minimum = estimate_motion_bank_bytes(count, time_dtype=torch.float32, include_initial_time=False)
        if minimum > self.max_bytes:
            raise MemoryError(f"all {count} motion rows need at least {minimum} bytes; budget {self.max_bytes}")
        schema = None
        try:
            for start in range(0, self.count, self.block_rows):
                stop = min(start+self.block_rows, self.count)
                payload = loader(np.arange(start, stop, dtype=np.int64))
                block = self._validate_block(payload, stop-start, schema)
                if schema is None:
                    schema = {key: (tuple(value.shape[1:]), value.dtype) for key, value in block.items()}
                    row_bytes = sum(value[0].numel()*value.element_size() for value in block.values())
                    self.resident_bytes = self.count*row_bytes
                    if self.resident_bytes > self.max_bytes:
                        raise MemoryError(f"all motion rows need {self.resident_bytes} bytes; budget {self.max_bytes}")
                    self.tensors = {key: torch.empty((self.count, *shape), dtype=dtype, device=self.device)
                                    for key, (shape, dtype) in schema.items()}
                for key, value in block.items():
                    # CPU copy is synchronous; CUDA H2D can overlap transfers
                    # of the next field. PyTorch owns pinned source lifetime.
                    if self.device.type == "cuda" and self.pin_memory and not value.is_pinned():
                        value = value.pin_memory()
                    self.tensors[key][start:stop].copy_(value, non_blocking=self.device.type == "cuda")
                self.blocks_loaded += 1
                del block, payload
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            self.ready = True
        except Exception:
            # Drop only this object's references; never empty another cache or
            # retry a truncated row set as if it were the complete 320K data.
            self.clear()
            raise

    @staticmethod
    def _validate_block(payload, rows, schema):
        if not isinstance(payload, Mapping):
            raise TypeError("source loader must return a mapping")
        required = set(MOTION_SHAPES)
        if not required.issubset(payload):
            raise ValueError(f"missing actual motion fields: {sorted(required-set(payload))}")
        names = (list(MOTION_SHAPES) + [key for key in OPTIONAL_SHAPES if key in payload]
                 if schema is None else list(schema))
        result = {}
        for key in names:
            if key not in payload:
                raise ValueError(f"motion field disappeared between blocks: {key}")
            value = payload[key]
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None:
                    raise ValueError("source motion must be CPU data without training gradients")
                value = value.contiguous()
            elif isinstance(value, np.ndarray):
                # No cast: preserves absolute seconds and calibrated positions.
                value = torch.from_numpy(np.array(value, copy=True, order="C"))
            else:
                raise TypeError(f"motion field {key} must be a CPU Tensor or NumPy array")
            shape = (MOTION_SHAPES | OPTIONAL_SHAPES)[key]
            if tuple(value.shape) != (rows, *shape):
                raise ValueError(f"motion field {key} expected {(rows, *shape)}, got {tuple(value.shape)}")
            allowed = ((torch.int64,) if key == "source_index" else
                       (torch.float32, torch.float64) if key in ("time_deltas", "initial_time_s") else
                       (torch.float32,))
            if value.dtype not in allowed:
                raise ValueError(f"motion field {key} has unsupported dtype {value.dtype}; no silent conversion")
            if schema is not None and value.dtype != schema[key][1]:
                raise ValueError(f"motion field {key} changed dtype between blocks")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"motion field {key} contains nonfinite data")
            if key == "time_deltas" and (value <= 0).any():
                raise ValueError("actual motion time_deltas must be positive")
            if key == "source_index" and (value < -1).any():
                raise ValueError("source_index uses -1 only for missing provenance")
            result[key] = value
        return result

    def get(self, rows):
        """Gather exact requested order/duplicates; never shuffle or resample."""
        if not self.ready:
            raise RuntimeError("motion bank is not ready or was cleared")
        if isinstance(rows, torch.Tensor):
            if rows.device.type != "cpu" or rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64):
                raise ValueError("row IDs must be one-dimensional CPU integer indices")
            indices = rows.to(dtype=torch.int64)
        else:
            array = np.asarray(rows)
            if array.ndim != 1 or (array.size and array.dtype.kind not in ("i", "u")):
                raise ValueError("row IDs must be one-dimensional integers")
            if array.size and ((array < 0).any() or (array >= self.count).any()):
                raise IndexError("motion row ID outside the full population")
            indices = torch.from_numpy(np.array(array, dtype=np.int64, copy=True))
        if ((indices < 0) | (indices >= self.count)).any():
            raise IndexError("motion row ID outside the full population")
        selected = indices.to(self.device, non_blocking=self.device.type == "cuda")
        self.requests += 1
        self.requested_rows += len(indices)
        return {key: value.index_select(0, selected) for key, value in self.tensors.items()}

    get_rows = get

    def clear(self):
        """Release only this bank; never call CUDA empty_cache or alter rows."""
        self.tensors.clear()
        self.resident_bytes = 0
        self.ready = False

    def stats(self):
        return dict(ready=self.ready, rows=self.count, resident_bytes=self.resident_bytes,
                    max_bytes=self.max_bytes, block_rows=self.block_rows,
                    blocks_loaded=self.blocks_loaded, requests=self.requests,
                    requested_rows=self.requested_rows, device=str(self.device),
                    fields={key: dict(shape=list(value.shape), dtype=str(value.dtype))
                            for key, value in self.tensors.items()})
