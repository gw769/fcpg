"""Inference-only Prod-V14 dual-view candidate ranking.

The adapter deliberately stops at ranking.  It does not load an actor, create
pixel anchors, or execute a command.  A caller supplies a bank of already-safe
five-command joint/CAD increments and three current-image persistent queries:
tip-A, tip-B and the pickup target.  Prod-V14 forecasts all three in both views
and ranks the candidates with one joint column+wrist observation-space energy.

Training used two anonymous generic queries.  The model is query-equivariant,
so the three-query runtime is structurally valid, but it remains a zero-shot
engineering hypothesis until checked on real tip/target tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from fcpg.prod_v13_multirate import MultiRatePartTokenLeWM
from fcpg.prod_v14 import (
    ACTION_EXCITATION_PX,
    CHECKPOINT_FORMAT,
    ENGINEERING_BEST_STEP,
    TRAINING_SEED,
    DualViewControlDecision,
    DualViewQueryForecast,
    V14MeasurementUpdate,
    V14SeparableResponseLeWM,
    dual_view_query_control_energy,
    v14_measurement_update,
)


TOKEN_DIM = 192
VIEWS = 2
TIP_QUERY_INDICES = (0, 1)
TARGET_QUERY_INDEX = 2
RUNTIME_QUERY_COUNT = 3


@dataclass(frozen=True)
class ProdV14CandidateRanking:
    """Forecast and joint dual-view decision for one candidate bank."""

    decision: DualViewControlDecision
    delta_x_logits: torch.Tensor
    delta_y_logits: torch.Tensor
    expected_delta_px: torch.Tensor
    expected_px: torch.Tensor
    query_latent: torch.Tensor
    view_valid: torch.Tensor
    action_gate_active: torch.Tensor


def _state_tensor(
    state: Mapping[str, Any], name: str, shape: tuple[int, ...]
) -> torch.Tensor:
    value = state.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"V14 checkpoint {name} is missing or malformed")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"V14 checkpoint {name} is non-finite")
    return value


class ProdV14PlannerAdapter(nn.Module):
    """Frozen candidate ranker around the selected V14 engineering endpoint."""

    def __init__(
        self,
        *,
        control_model: MultiRatePartTokenLeWM,
        forecast_model: V14SeparableResponseLeWM,
        checkpoint_step: int,
    ) -> None:
        super().__init__()
        self.control_model = control_model
        self.forecast_model = forecast_model
        self.checkpoint_step = int(checkpoint_step)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device,
        expected_step: int = ENGINEERING_BEST_STEP,
    ) -> "ProdV14PlannerAdapter":
        """Load the fixed V14 forecast without loading either actor."""

        checkpoint = torch.load(
            Path(path).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("V14 checkpoint must contain a mapping")
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("V14 checkpoint format is not the excitation-gated v4")
        if int(checkpoint.get("step", -1)) != int(expected_step):
            raise ValueError("V14 checkpoint is not the selected engineering step")
        if int(checkpoint.get("seed", -1)) != TRAINING_SEED:
            raise ValueError("V14 checkpoint seed is not 14101")
        if checkpoint.get("architecture") != (
            "causal_arx_plus_6px_identifiable_control_response"
        ):
            raise ValueError("V14 checkpoint architecture is incompatible")
        if any("actor" in str(key).lower() for key in checkpoint):
            raise ValueError("V14 planning checkpoint must not contain actor state")

        control_state = checkpoint.get("control_model_state")
        forecast_state = checkpoint.get("forecast_model_state")
        if not isinstance(control_state, Mapping) or not isinstance(
            forecast_state, Mapping
        ):
            raise ValueError("V14 checkpoint model states are missing")
        joint_rms = _state_tensor(control_state, "joint_rms", (6,))
        cad_rms = _state_tensor(control_state, "cad_rms", (6,))
        control_model = MultiRatePartTokenLeWM(
            joint_rms=joint_rms,
            cad_rms=cad_rms,
        )
        forecast_model = V14SeparableResponseLeWM()
        control_model.load_state_dict(control_state, strict=True)
        forecast_model.load_state_dict(forecast_state, strict=True)
        target_device = torch.device(device)
        adapter = cls(
            control_model=control_model.to(target_device),
            forecast_model=forecast_model.to(target_device),
            checkpoint_step=int(expected_step),
        )
        return adapter

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def rank_candidates(
        self,
        *,
        global_history: torch.Tensor,
        query_tokens: torch.Tensor,
        query_px: torch.Tensor,
        previous_delta_px: torch.Tensor,
        candidate_joint_increments: torch.Tensor,
        candidate_cad_tip_displacement: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> ProdV14CandidateRanking:
        """Rank `[B,K]` safe chunks using both views and all three queries.

        Joint inputs are measured/commanded *increments*, not absolute servo
        positions.  CAD inputs are the two unordered tip displacements over the
        same completed five-command interval.
        """

        device = self.device
        history = torch.as_tensor(global_history, device=device).float()
        tokens = torch.as_tensor(query_tokens, device=device).float()
        points = torch.as_tensor(query_px, device=device).float()
        previous = torch.as_tensor(previous_delta_px, device=device).float()
        joint = torch.as_tensor(candidate_joint_increments, device=device).float()
        cad = torch.as_tensor(
            candidate_cad_tip_displacement, device=device
        ).float()
        valid = torch.as_tensor(view_valid, device=device)
        if history.ndim != 4 or history.shape[1:] != (2, VIEWS, TOKEN_DIM):
            raise ValueError("global_history must be [B,2,2,192]")
        batch = len(history)
        if tokens.shape != (batch, VIEWS, RUNTIME_QUERY_COUNT, TOKEN_DIM):
            raise ValueError("query_tokens must be [B,2,3,192]")
        if points.shape != (batch, VIEWS, RUNTIME_QUERY_COUNT, 2):
            raise ValueError("query_px must be [B,2,3,2]")
        if previous.shape != points.shape:
            raise ValueError("previous_delta_px must be [B,2,3,2]")
        if joint.ndim != 4 or joint.shape[0] != batch or joint.shape[2:] != (5, 6):
            raise ValueError("candidate_joint_increments must be [B,K,5,6]")
        candidates = int(joint.shape[1])
        if candidates < 1:
            raise ValueError("V14 candidate bank cannot be empty")
        if cad.shape != (batch, candidates, 2, 3):
            raise ValueError(
                "candidate_cad_tip_displacement must be [B,K,2,3]"
            )
        if valid.shape != (batch, VIEWS) or valid.dtype != torch.bool:
            raise ValueError("view_valid must be bool [B,2]")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("each V14 sample needs at least one valid view")
        values = (history, tokens, points, previous, joint, cad)
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("V14 runtime tensors must be finite")

        def expand(value: torch.Tensor) -> torch.Tensor:
            return value[:, None].expand(-1, candidates, *value.shape[1:]).reshape(
                batch * candidates, *value.shape[1:]
            )

        flat_joint = joint.reshape(batch * candidates, 5, 6)
        flat_cad = cad.reshape(batch * candidates, 2, 3)
        control = self.control_model.encode_control_chunk(
            joint_increment_chunk=flat_joint,
            cad_tip_displacement=flat_cad,
        )
        flat_forecast = self.forecast_model(
            global_history=expand(history),
            query_tokens=expand(tokens),
            query_px=expand(points),
            previous_delta_px=expand(previous),
            control_features=control,
            view_valid=expand(valid),
        )

        def unflatten(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, candidates, *value.shape[1:])

        expected = unflatten(flat_forecast.expected_px)
        forecast_valid = unflatten(flat_forecast.view_valid)
        tips = expected[..., list(TIP_QUERY_INDICES), :]
        target = expected[..., TARGET_QUERY_INDEX, :]
        decision = dual_view_query_control_energy(tips, target, forecast_valid)
        action_gate = (
            torch.linalg.vector_norm(previous, dim=-1) >= ACTION_EXCITATION_PX
        )
        action_gate = action_gate[:, None].expand(-1, candidates, -1, -1)
        return ProdV14CandidateRanking(
            decision=decision,
            delta_x_logits=unflatten(flat_forecast.delta_x_logits),
            delta_y_logits=unflatten(flat_forecast.delta_y_logits),
            expected_delta_px=unflatten(flat_forecast.expected_delta_px),
            expected_px=expected,
            query_latent=unflatten(flat_forecast.query_latent),
            view_valid=forecast_valid,
            action_gate_active=action_gate,
        )

    @torch.inference_mode()
    def update_selected_measurement(
        self,
        ranking: ProdV14CandidateRanking,
        *,
        current_query_px: torch.Tensor,
        candidate_px: torch.Tensor,
        measurement_logits: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> V14MeasurementUpdate:
        """Apply only the selected forecast to current-image candidates."""

        selected = ranking.decision.selected_index
        batch = len(selected)
        rows = torch.arange(batch, device=selected.device)
        forecast = DualViewQueryForecast(
            delta_x_logits=ranking.delta_x_logits[rows, selected],
            delta_y_logits=ranking.delta_y_logits[rows, selected],
            expected_delta_px=ranking.expected_delta_px[rows, selected],
            expected_px=ranking.expected_px[rows, selected],
            query_latent=ranking.query_latent[rows, selected],
            view_valid=ranking.view_valid[rows, selected],
        )
        return v14_measurement_update(
            forecast=forecast,
            current_query_px=torch.as_tensor(
                current_query_px, device=self.device
            ),
            candidate_px=torch.as_tensor(candidate_px, device=self.device),
            measurement_logits=torch.as_tensor(
                measurement_logits, device=self.device
            ),
            candidate_valid=torch.as_tensor(candidate_valid, device=self.device),
        )


__all__ = [
    "ProdV14CandidateRanking",
    "ProdV14PlannerAdapter",
    "RUNTIME_QUERY_COUNT",
    "TARGET_QUERY_INDEX",
    "TIP_QUERY_INDICES",
]
