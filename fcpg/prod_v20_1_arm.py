"""ARM-only training representation regularization for V20.1.

Runtime ARM parameters and physical outputs are unchanged from V20LeWM.
The projector is a separate training-side module, never a runtime submodule.
SIGReg uses the existing LeWM-derived 17-knot/256-direction convention,
including its multiplication by the number of independent batch rows.

For each source, each horizon and each command offset is its own temporal
slice; only independent chain rows estimate a characteristic function.
Collect the complete effective batch before this function: averaging separate
microbatch regularizers is NOT mathematically equivalent.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .prod_v20_policy import (
    V20LeWM, ActualHistory, LeWMOutput, PolicyContractError,
    JOINTS, HISTORY, STEPS, MODEL_DIM, _tensor, _validate_history,
)
from .regularization import SIGReg

SIGREG_KNOTS = 17
SIGREG_DIRECTIONS = 256
ARM_SIGREG_WEIGHT = .02


class V201LeWM(V20LeWM):
    """Expose actual post-LayerNorm tokens without hooks or extra state keys.

    ``forward`` and ``forward_with_hidden`` share this exact response path.
    Physical q-path/uncertainty are not themselves Gaussian-regularized.
    """

    def forward_with_hidden(self, q_position: torch.Tensor,
                            commands_position: torch.Tensor,
                            actual_history: ActualHistory,
                            time_deltas: torch.Tensor) -> tuple[LeWMOutput, torch.Tensor]:
        # Same V20LeWM response algebra; no alternate training-only predictor.
        batch = len(q_position)
        q = _tensor("q_position", q_position, (batch, JOINTS)).to(self.joint_projection[0].weight)
        commands = _tensor("commands_position", commands_position, (batch, STEPS, JOINTS)).to(q)
        dt = _tensor("time_deltas", time_deltas, (batch, STEPS)).to(q)
        if torch.any(dt <= 0):
            raise PolicyContractError("planned command intervals must be positive seconds")
        _validate_history(actual_history, batch)
        history = self.history_projection(self._motion(actual_history, q)) + self.history_embedding
        joints = self.joint_projection(self.normalize_position(q)[..., None]) + self.joint_embedding
        increments = torch.diff(torch.cat((q[:, None], commands), dim=1), dim=1)
        relative = commands - q[:, None]
        cmd_input = torch.cat((increments / self.increment_scale,
                               relative / self.position_half_range,
                               torch.log1p(dt)[..., None]), dim=-1)
        future = self.command_projection(cmd_input) + self.command_embedding
        tokens = torch.cat((joints, history, future), dim=1)
        context_count = JOINTS + HISTORY
        length = context_count + STEPS
        causal = torch.zeros(length, length, device=q.device, dtype=torch.bool)
        causal[:context_count, context_count:] = True
        causal[context_count:, context_count:] = torch.triu(
            torch.ones(STEPS, STEPS, device=q.device, dtype=torch.bool), diagonal=1)
        padding = torch.cat((torch.zeros(batch, JOINTS, device=q.device, dtype=torch.bool),
                             ~actual_history.valid.to(q.device),
                             torch.zeros(batch, STEPS, device=q.device, dtype=torch.bool)), dim=1)
        for layer in self.layers:
            tokens = layer(tokens, padding=padding, causal=causal)
        hidden = self.output_norm(tokens[:, -STEPS:])
        predicted = commands + self.response_head(hidden) * self.increment_scale
        uncertainty = (F.softplus(self.uncertainty_head(hidden).float()) + 1e-4).to(q) * self.increment_scale
        predicted = predicted.to(q_position)
        output = LeWMOutput(torch.cat((q_position[:, None], predicted), dim=1), uncertainty.to(q_position))
        return output, hidden

    def forward(self, q_position: torch.Tensor, commands_position: torch.Tensor,
                actual_history: ActualHistory, time_deltas: torch.Tensor) -> LeWMOutput:
        return self.forward_with_hidden(q_position, commands_position, actual_history, time_deltas)[0]

    predict = forward


class ARMProjector(nn.Sequential):
    """Training-only 192->192->192 MLP; deliberately no output normalization."""

    def __init__(self):
        super().__init__(nn.Linear(MODEL_DIM, MODEL_DIM), nn.GELU(),
                         nn.Linear(MODEL_DIM, MODEL_DIM))


def sample_sigreg_directions(dimension=MODEL_DIM, *, device=None, generator=None):
    """Draw one shared fp32 sketch, independently of source count or lambda.

    Use the same call for matched lambda=0 and lambda=.02 experiments (or a
    dedicated identically seeded generator). Sampling fewer times in the zero
    arm would otherwise perturb an unrelated global training random stream.
    """
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("SIGReg representation dimension must be positive")
    directions = torch.randn(dimension, SIGREG_DIRECTIONS, device=device,
                             dtype=torch.float32, generator=generator)
    return directions / directions.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6)


def _fixed_directions(directions, reference):
    if not isinstance(directions, torch.Tensor) or directions.shape != (reference.shape[-1], SIGREG_DIRECTIONS):
        raise ValueError("SIGReg fixed directions must be [D,256]")
    if directions.requires_grad or directions.grad_fn is not None:
        raise ValueError("SIGReg sketch directions cannot be learned parameters")
    directions = directions.to(device=reference.device, dtype=torch.float32)
    norm = directions.norm(p=2, dim=0, keepdim=True)
    if not torch.isfinite(directions).all() or (norm <= 0).any():
        raise ValueError("SIGReg directions must be finite and nonzero")
    return directions / norm


def source_local_sigreg(projected: torch.Tensor, source_index: torch.Tensor, *,
                        directions: torch.Tensor | None = None,
                        generator: torch.Generator | None = None,
                        return_stats=False):
    """Source-macro SIGReg scalar for ``[H,B,5,D]`` projected ARM tokens.

    Missing source is exactly -1. Sources with fewer than two independent rows
    are skipped; H and command offsets never inflate that row count. Source
    losses have equal weight regardless of group size; each source loss is
    averaged across H, five offsets and 256 projections, as in existing SIGReg.

    By default a single random sketch is drawn even if every group is skipped.
    A zero-weight arm must call this function too, then multiply by zero.
    Explicit fixed directions consume no RNG. ``return_stats=True`` returns
    ``(loss, coverage_dict)``; otherwise only the differentiable scalar.
    """
    if not isinstance(projected, torch.Tensor) or projected.ndim != 4 or projected.shape[0] < 1 or projected.shape[2] != STEPS or projected.shape[-1] < 1:
        raise ValueError("projected ARM representation must be [H,B,5,D]")
    if not projected.is_floating_point() or not torch.isfinite(projected).all():
        raise ValueError("projected ARM representation must be finite floating point")
    sources = torch.as_tensor(source_index, device=projected.device)
    if sources.shape != (projected.shape[1],) or sources.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("source_index must be integer [B], missing=-1")
    if (sources < -1).any():
        raise ValueError("missing source must use -1, not another negative ID")
    if directions is not None and generator is not None:
        raise ValueError("choose fixed directions or a sampling generator")
    with torch.autocast(device_type=projected.device.type, enabled=False):
        values = projected.float()
        sketch = (sample_sigreg_directions(values.shape[-1], device=values.device, generator=generator)
                  if directions is None else _fixed_directions(directions, values))
        # Reuse the exact t/phi/quadrature convention already vendored here.
        reference = SIGReg(knots=SIGREG_KNOTS, num_proj=SIGREG_DIRECTIONS).to(values.device)
        losses = []
        eligible_rows = singleton_rows = 0
        group_sizes = {}
        for source in sources.unique(sorted=True).tolist():
            if source == -1:
                continue
            rows = sources == source
            count = int(rows.sum().item())
            group_sizes[int(source)] = count
            if count < 2:
                singleton_rows += count
                continue
            eligible_rows += count
            # [H,5,Bsource,D] -> [H*5,Bsource,D]. Only H/offset are merged.
            group = values[:, rows].permute(0, 2, 1, 3).reshape(-1, count, values.shape[-1])
            x_t = (group @ sketch).unsqueeze(-1) * reference.t
            error = ((x_t.cos().mean(dim=-3) - reference.phi).square()
                     + x_t.sin().mean(dim=-3).square())
            losses.append(((error @ reference.weights) * count).mean())
        loss = torch.stack(losses).mean() if losses else values.sum() * 0.
    stats = dict(eligible_sources=len(losses), eligible_rows=eligible_rows,
                 missing_rows=int((sources == -1).sum().item()), singleton_rows=singleton_rows,
                 total_rows=int(projected.shape[1]), group_sizes=group_sizes,
                 coverage=eligible_rows / max(1, projected.shape[1]),
                 horizons=int(projected.shape[0]), command_offsets=STEPS,
                 knots=SIGREG_KNOTS, directions=SIGREG_DIRECTIONS,
                 statistic_scaling="source_batch_size_times_ecf_discrepancy")
    return (loss, stats) if return_stats else loss


V201ARMProjector = ARMProjector
