"""Optional immutable CEM diagnostics; does not sample or re-evaluate anything."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from copy import deepcopy

import torch


def _frozen(value, *, cpu=False):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        return (value.cpu() if cpu else value).clone()
    if is_dataclass(value):
        return {field.name: _frozen(getattr(value, field.name), cpu=cpu) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _frozen(child, cpu=cpu) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_frozen(child, cpu=cpu) for child in value)
    return deepcopy(value)


def capture_rng_state(device, generator=None):
    """Snapshot the actual device stream; training seeds are not planner seeds."""
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('CEM RNG trace supports the CPU/CUDA planner devices')
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    result = dict(source='global' if generator is None else 'explicit_generator',
        device=str(device), cpu_global_state=torch.get_rng_state().clone(),
        device_global_state=None, generator_device=None, generator_state=None)
    if device.type == 'cuda':
        result['device_global_state'] = torch.cuda.get_rng_state(device).clone()
    if generator is not None:
        result['generator_device'] = str(generator.device)
        result['generator_state'] = generator.get_state().clone()
    return result


class CEMTrace:
    """One-plan recorder with detached snapshots and a torch.save-ready payload.

    Clones remain on their original device while planning. ``payload`` copies
    them to CPU after the decision; neither method performs inference or draws
    random numbers. A recorder is deliberately not reusable across plans.
    """
    def __init__(self):
        self.events = []
        self.rng_before = None
        self.rng_after = None
        self.environment = None
        self.error = None

    def begin(self, reference, generator=None, **inputs):
        if self.rng_before is not None:
            raise ValueError('CEMTrace records exactly one planning call')
        self.rng_before = capture_rng_state(reference.device, generator)
        self.environment = dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
            device=str(reference.device), dtype=str(reference.dtype), threads=torch.get_num_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            mha_fastpath_enabled=torch.backends.mha.get_fastpath_enabled(),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))
        if reference.device.type == 'cuda':
            self.environment.update(cuda_device_name=torch.cuda.get_device_name(reference.device),
                cuda_device_capability=list(torch.cuda.get_device_capability(reference.device)))
        self.record('planning_input', **inputs)

    def record(self, event, **values):
        self.events.append(dict(event=str(event), values=_frozen(values)))

    def finish(self, reference, generator=None, *, error=None):
        self.rng_after = capture_rng_state(reference.device, generator)
        self.error = None if error is None else str(error)

    def payload(self):
        if self.rng_before is None or self.rng_after is None:
            raise RuntimeError('CEM trace requires both before and after RNG snapshots')
        return _frozen(dict(schema='v21-cem-trace-v1', executable=False,
            historical_missing_rng_recovered=False, environment=self.environment,
            rng_before=self.rng_before, rng_after=self.rng_after,
            complete=self.error is None, error=self.error, events=self.events), cpu=True)


def record_cem_event(trace, event, **values):
    if trace is not None:
        if not isinstance(trace, CEMTrace):
            raise TypeError('CEM diagnostics require CEMTrace, not an arbitrary callback')
        trace.record(event, **values)
