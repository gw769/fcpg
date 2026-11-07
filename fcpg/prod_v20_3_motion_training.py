"""LeWM's exact causal physical-response training path without unused images.

V201LeWM._motion reads only completed commands/readbacks/times/valid. It does
not read ActualHistory.scene_pooled, current RGB, a belief or task text. Thus
Stage B can omit frozen observation/cache/control-encoding work while using
the identical response network, actual history, losses and SIGReg projector.
This is NOT a reduced Actor input path and must never be used for Actor BC.
"""
from __future__ import annotations

from typing import Mapping

import torch

from .prod_v20_policy import empty_history, append_completed, future_q_path_loss
from .prod_v20_3_training import LeWMLossParts, combine_lewm_loss_parts


def motion_chain_loss(core, chain: Mapping, *, schedule=None, return_parts=True):
    """H0..H3 factual LeWM unroll; no visual cache or predicted actual state.

    Required fields are q_position [B,5,6], commands_position [B,4,5,6],
    measured_q_paths_position [B,4,6,6] and time_deltas [B,4,5]. Optional
    source_index [B] is provenance for source-local SIGReg, never a model input.
    Return structured parts for whole-effective-batch SIGReg; averaging scalar
    microbatch regularizers would change the training objective.
    """
    q = chain["q_position"]
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or len(q) < 1:
        raise ValueError("V20.3 motion chain needs q_position [B,5,6]")
    batch = len(q)
    commands = chain["commands_position"]
    paths = chain["measured_q_paths_position"]
    dt = chain["time_deltas"]
    for name, value, shape in (
        ("q_position", q, (batch, 5, 6)),
        ("commands_position", commands, (batch, 4, 5, 6)),
        ("measured_q_paths_position", paths, (batch, 4, 6, 6)),
        ("time_deltas", dt, (batch, 4, 5)),
    ):
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite floating {shape}")
    if (not torch.allclose(paths[:, :, 0], q[:, :-1], atol=1e-4, rtol=0)
            or not torch.allclose(paths[:, :, -1], q[:, 1:], atol=1e-4, rtol=0)):
        raise ValueError("same-chain measured endpoint clock mismatch")
    if not bool((dt > 0).all()):
        raise ValueError("five real elapsed time intervals required")
    source = chain.get("source_index")
    source = (torch.full((batch,), -1, device=q.device, dtype=torch.long) if source is None
              else torch.as_tensor(source, device=q.device))
    if (source.shape != (batch,) or source.dtype not in
            (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8) or (source < -1).any()):
        raise ValueError("source_index must be integer [B], missing=-1")

    history = empty_history(batch, device=q.device, dtype=q.dtype)
    # Shape-only field required by ActualHistory validation. LeWM never reads
    # its values; these zeros are NOT cached images or Actor training examples.
    unused_scene = q.new_zeros(batch, 2, 384)
    regression_terms, nll_terms, projected_terms = [], [], []
    for horizon in range(4):
        prediction, hidden = core.lewm.forward_with_hidden(
            q[:, horizon], commands[:, horizon], history, dt[:, horizon])
        regression_terms.append(future_q_path_loss(
            prediction.predicted_q_path, paths[:, horizon], core.q_loss_scale))
        error = (prediction.predicted_q_path[:, 1:] - paths[:, horizon, 1:]) / core.q_loss_scale
        std = (prediction.q_uncertainty / core.q_loss_scale).clamp_min(1e-3)
        nll_terms.append((0.5 * (error.detach() / std).square() + std.log()).mean())
        if hidden.shape != (batch, 5, 192):
            raise ValueError("ARM training requires all five true response-head hidden states")
        projected_terms.append(core.arm_sigreg_projector(hidden))
        if horizon < 3:
            history = append_completed(history, commands[:, horizon], paths[:, horizon],
                                       unused_scene, dt[:, horizon])
    parts = LeWMLossParts(torch.stack(regression_terms).mean(), torch.stack(nll_terms).mean(),
                         torch.stack(projected_terms), source)
    return parts if return_parts else combine_lewm_loss_parts([parts], schedule=schedule)[0]


__all__ = ["motion_chain_loss"]
