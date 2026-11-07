"""Prod-V13 recursive-v3: action-innovation feedback over frozen measurement.

The protected-memory FCPG-H observer already owns visual persistence.  This
revision therefore prevents LeWM from learning a second action-independent
visual tracker.  The same frozen V12.4 carrier is evaluated with the factual
completed/planned action and with a hold-state reference.  Only their
difference reaches the recurrent feedback head, and the previous-token
similarity is subtracted from the posterior score.  Consequently factual ==
hold is a structural measurement-only null, not a separately learned arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from fcpg.offline_pipeline_manifest import sha256_file
from fcpg.prod_v13 import (
    FROZEN_PRODUCTION_ACTOR_SHA256,
    prod_v13_state_dict_sha256,
)
from fcpg.prod_v13_closed_loop import (
    EXPECTED_PROTECTED_MEMORY_SHA256,
    EXPECTED_V12_SHA256,
    ProdV13ClosedLoopOutput,
)
from fcpg.prod_v13_recursive_v2 import ProdV13RecursiveV2


FORMAT = "so101-prod-v13-action-innovation-feedback-v3"
CHECKPOINT_FORMAT = "so101-prod-v13-action-innovation-checkpoint-v3"
METHOD_REVISION = "prod-v13-recursive-v3-action-innovation-top32"
IIR_FORMAT = "so101-prod-v13-action-innovation-iir025-feedback-v3-1"
IIR_CHECKPOINT_FORMAT = "so101-prod-v13-action-innovation-iir025-checkpoint-v3-1"
IIR_METHOD_REVISION = "prod-v13-recursive-v3-1-action-innovation-iir025"
IIR_WRITEBACK_GAIN = 0.25
SOFT_IIR_FORMAT = "so101-prod-v13-action-innovation-soft-iir025-feedback-v3-2"
SOFT_IIR_CHECKPOINT_FORMAT = (
    "so101-prod-v13-action-innovation-soft-iir025-checkpoint-v3-2"
)
SOFT_IIR_METHOD_REVISION = (
    "prod-v13-recursive-v3-2-action-innovation-soft-belief-iir025"
)


@dataclass(frozen=True)
class ProdV13ActionInnovationOutput(ProdV13ClosedLoopOutput):
    reference_dynamics_latent: torch.Tensor
    action_innovation_latent: torch.Tensor


class ProdV13ActionInnovation(ProdV13RecursiveV2):
    """Head-only action innovation on exact frozen V12.4/FCPG-H endpoints."""

    def _configure_trainability(self) -> None:
        super()._configure_trainability()
        # This first mechanism screen cannot let the carrier adapt around the
        # common-mode subtraction.  Only the existing feedback head trains.
        self.v12.requires_grad_(False)
        self.fcpg.requires_grad_(False)
        self.feedback.requires_grad_(True)

    def parameter_contract(self) -> dict[str, Any]:
        contract = super().parameter_contract()
        contract.update(
            {
                "format": FORMAT,
                "method_revision": METHOD_REVISION,
                "carrier_trainability": "fully_frozen",
                "feedback_input": "v12_dynamics_factual_minus_hold_state",
                "common_mode_rejection": "subtract_before_feedback_head",
                "head_bias_rejection": "head(previous,delta)-head(previous,zero)",
                "posterior_score": (
                    "measurement+0.25*(cos(action_prior)-cos(previous))/0.1"
                ),
                "zero_innovation_semantics": "measurement_only_inside_top32",
                "separately_trained_hold_arm": False,
            }
        )
        return contract

    def forward_cached(
        self,
        *,
        vit_cls_history: torch.Tensor,
        future_vit_patch_tokens: torch.Tensor,
        initial_posterior_tokens: torch.Tensor,
        measurement_logits: torch.Tensor,
        history_view_valid: torch.Tensor,
        future_view_valid: torch.Tensor,
        measurement_topk_support: torch.Tensor | None = None,
        recursive_writeback: bool = True,
        anchor_xyz_history: torch.Tensor,
        state_history: torch.Tensor,
        past_action_chunks: torch.Tensor,
        action: torch.Tensor,
        detach_carrier: bool = True,
        writeback_gain: float = 1.0,
        soft_writeback_forward: bool = False,
    ) -> ProdV13ActionInnovationOutput:
        del measurement_topk_support, detach_carrier
        if action.ndim != 3 or action.shape[-1] != state_history.shape[-1]:
            raise ValueError("action must be [B,A,D] and match state dimension")
        if len(action) != len(state_history):
            raise ValueError("action and state batch sizes differ")
        hold_action = state_history[:, -1, None, :].expand_as(action).contiguous()

        # Both carrier calls are immutable signal transforms.  Keeping them
        # out of autograd makes the causal screen cheaper and prevents the
        # carrier from learning around its own hold reference.
        with torch.no_grad():
            factual = self.predict_dynamics_cached(
                vit_cls_history=vit_cls_history,
                history_view_valid=history_view_valid,
                anchor_xyz_history=anchor_xyz_history,
                state_history=state_history,
                past_action_chunks=past_action_chunks,
                action=action,
            )
            reference = self.predict_dynamics_cached(
                vit_cls_history=vit_cls_history,
                history_view_valid=history_view_valid,
                anchor_xyz_history=anchor_xyz_history,
                state_history=state_history,
                past_action_chunks=past_action_chunks,
                action=hold_action,
            )
            projected_future = self.v12.patch_projector(future_vit_patch_tokens)
            projected_initial = self.v12.patch_projector(initial_posterior_tokens)

        feedback = self.feedback(
            factual,
            projected_initial,
            projected_future,
            measurement_logits,
            future_view_valid.bool(),
            recursive_writeback=recursive_writeback,
            reference_dynamics_latent=reference,
            center_prior_similarity=True,
            writeback_gain=writeback_gain,
            soft_writeback_forward=soft_writeback_forward,
        )
        return ProdV13ActionInnovationOutput(
            dynamics_latent=factual,
            reference_dynamics_latent=reference,
            action_innovation_latent=factual - reference,
            feedback=feedback,
        )


class ProdV13ActionInnovationIIR025(ProdV13ActionInnovation):
    """Action innovation with a fixed first-order posterior update gain."""

    def parameter_contract(self) -> dict[str, Any]:
        contract = super().parameter_contract()
        contract.update(
            {
                "format": IIR_FORMAT,
                "method_revision": IIR_METHOD_REVISION,
                "writeback_filter": "first_order_iir",
                "writeback_gain": IIR_WRITEBACK_GAIN,
                "writeback_equation": (
                    "normalize(0.75*previous+0.25*current_measurement_token)"
                ),
            }
        )
        return contract

    def forward_cached(
        self,
        *,
        writeback_gain: float = IIR_WRITEBACK_GAIN,
        **kwargs: Any,
    ) -> ProdV13ActionInnovationOutput:
        if float(writeback_gain) != IIR_WRITEBACK_GAIN:
            raise ValueError("recursive-v3.1 fixes writeback gain at 0.25")
        return super().forward_cached(
            writeback_gain=IIR_WRITEBACK_GAIN,
            **kwargs,
        )


class ProdV13ActionInnovationSoftIIR025(ProdV13ActionInnovationIIR025):
    """IIR feedback whose recurrent state is the supported token belief."""

    def parameter_contract(self) -> dict[str, Any]:
        contract = super().parameter_contract()
        contract.update(
            {
                "format": SOFT_IIR_FORMAT,
                "method_revision": SOFT_IIR_METHOD_REVISION,
                "writeback_value": "top32_posterior_expected_current_patch_token",
                "hard_spatial_output_unchanged": True,
                "soft_support_outside_top32": 0.0,
            }
        )
        return contract

    def forward_cached(
        self,
        *,
        soft_writeback_forward: bool = True,
        **kwargs: Any,
    ) -> ProdV13ActionInnovationOutput:
        if not bool(soft_writeback_forward):
            raise ValueError("recursive-v3.2 fixes soft belief-state writeback")
        return super().forward_cached(
            soft_writeback_forward=True,
            **kwargs,
        )


def checkpoint_payload(
    model: (
        ProdV13ActionInnovation
        | ProdV13ActionInnovationIIR025
        | ProdV13ActionInnovationSoftIIR025
    ),
    *,
    step: int,
    fixed_endpoint_step: int,
    dependencies: Mapping[str, str],
    training_contract: Mapping[str, Any],
    sample_order_sha256: str,
) -> dict[str, Any]:
    required = {
        "v12_checkpoint_sha256": EXPECTED_V12_SHA256,
        "fcpg_checkpoint_sha256": EXPECTED_PROTECTED_MEMORY_SHA256,
        "production_actor_reference_sha256": FROZEN_PRODUCTION_ACTOR_SHA256,
    }
    for name, expected in required.items():
        if dependencies.get(name) != expected:
            raise ValueError(f"action-innovation dependency {name} mismatch")
    state = model.trainable_state()
    soft_iir = isinstance(model, ProdV13ActionInnovationSoftIIR025)
    iir = isinstance(model, ProdV13ActionInnovationIIR025)
    checkpoint_format = (
        SOFT_IIR_CHECKPOINT_FORMAT
        if soft_iir
        else (IIR_CHECKPOINT_FORMAT if iir else CHECKPOINT_FORMAT)
    )
    prod_v13_format = (
        SOFT_IIR_FORMAT if soft_iir else (IIR_FORMAT if iir else FORMAT)
    )
    method_revision = (
        SOFT_IIR_METHOD_REVISION
        if soft_iir
        else (IIR_METHOD_REVISION if iir else METHOD_REVISION)
    )
    return {
        "format": checkpoint_format,
        "prod_v13_format": prod_v13_format,
        "method_revision": method_revision,
        "step": int(step),
        "fixed_endpoint_step": int(fixed_endpoint_step),
        "trainable_state": state,
        "trainable_state_sha256": prod_v13_state_dict_sha256(state),
        "parameter_contract": model.parameter_contract(),
        "training_contract": dict(training_contract),
        "sample_order_sha256": str(sample_order_sha256),
        **dict(dependencies),
        "actor_checkpoint": False,
        "actor_loss": False,
        "sealed_test_read": False,
    }


def _portable_parameter_contract(contract: Any) -> dict[str, Any]:
    """Remove only the machine-local actor path from an immutable contract.

    The actor bytes are already bound by the top-level SHA and the remaining
    actor-reference fields.  Requiring the pro6000D absolute path on the V100
    made an otherwise byte-identical artifact impossible to load locally.
    """

    if not isinstance(contract, Mapping):
        raise ValueError("action-innovation checkpoint lacks parameter contract")
    result = dict(contract)
    actor = result.get("production_actor_reference")
    if not isinstance(actor, Mapping):
        raise ValueError("parameter contract lacks production actor reference")
    actor_result = dict(actor)
    path = actor_result.pop("path", None)
    if not isinstance(path, str) or not path:
        raise ValueError("production actor reference path is invalid")
    result["production_actor_reference"] = actor_result
    return result


def _load_typed_checkpoint(
    checkpoint: str | Path,
    *,
    model_type: type[ProdV13ActionInnovation],
    checkpoint_format: str,
    prod_v13_format: str,
    method_revision: str,
    label: str,
    production_actor_checkpoint: str | Path,
    v12_checkpoint: str | Path,
    fcpg_checkpoint: str | Path,
    device: str | torch.device,
    require_fixed_endpoint: bool = True,
) -> tuple[ProdV13ActionInnovation, dict[str, Any]]:
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format") != checkpoint_format
        or payload.get("prod_v13_format") != prod_v13_format
        or payload.get("method_revision") != method_revision
        or bool(payload.get("actor_checkpoint"))
        or bool(payload.get("actor_loss"))
        or bool(payload.get("sealed_test_read"))
    ):
        raise ValueError(f"{label} checkpoint identity/safety drifted")
    step = int(payload.get("step", -1))
    fixed = int(payload.get("fixed_endpoint_step", -1))
    if step <= 0 or fixed <= 0 or (require_fixed_endpoint and step != fixed):
        raise ValueError(f"{label} checkpoint is not its fixed endpoint")
    bindings = {
        "v12_checkpoint_sha256": sha256_file(v12_checkpoint),
        "fcpg_checkpoint_sha256": sha256_file(fcpg_checkpoint),
        "production_actor_reference_sha256": sha256_file(
            production_actor_checkpoint
        ),
    }
    for name, expected in bindings.items():
        if payload.get(name) != expected:
            raise ValueError(f"{label} checkpoint {name} binding mismatch")
    state = payload.get("trainable_state")
    if not isinstance(state, Mapping):
        raise ValueError(f"{label} checkpoint lacks trainable state")
    if payload.get("trainable_state_sha256") != prod_v13_state_dict_sha256(state):
        raise ValueError(f"{label} trainable-state SHA drifted")
    model = model_type.from_frozen_endpoints(
        production_actor_checkpoint=production_actor_checkpoint,
        v12_checkpoint=v12_checkpoint,
        fcpg_checkpoint=fcpg_checkpoint,
        device=device,
    )
    if _portable_parameter_contract(
        payload.get("parameter_contract")
    ) != _portable_parameter_contract(model.parameter_contract()):
        raise ValueError(f"{label} parameter contract drifted")
    model.load_trainable_state(state)
    return model.eval(), payload


def load_checkpoint(
    checkpoint: str | Path,
    *,
    production_actor_checkpoint: str | Path,
    v12_checkpoint: str | Path,
    fcpg_checkpoint: str | Path,
    device: str | torch.device,
    require_fixed_endpoint: bool = True,
) -> tuple[ProdV13ActionInnovation, dict[str, Any]]:
    model, payload = _load_typed_checkpoint(
        checkpoint,
        model_type=ProdV13ActionInnovation,
        checkpoint_format=CHECKPOINT_FORMAT,
        prod_v13_format=FORMAT,
        method_revision=METHOD_REVISION,
        label="action-innovation",
        production_actor_checkpoint=production_actor_checkpoint,
        v12_checkpoint=v12_checkpoint,
        fcpg_checkpoint=fcpg_checkpoint,
        device=device,
        require_fixed_endpoint=require_fixed_endpoint,
    )
    return model.eval(), payload


def load_iir_checkpoint(
    checkpoint: str | Path,
    *,
    production_actor_checkpoint: str | Path,
    v12_checkpoint: str | Path,
    fcpg_checkpoint: str | Path,
    device: str | torch.device,
    require_fixed_endpoint: bool = True,
) -> tuple[ProdV13ActionInnovationIIR025, dict[str, Any]]:
    model, payload = _load_typed_checkpoint(
        checkpoint,
        model_type=ProdV13ActionInnovationIIR025,
        checkpoint_format=IIR_CHECKPOINT_FORMAT,
        prod_v13_format=IIR_FORMAT,
        method_revision=IIR_METHOD_REVISION,
        label="action-innovation IIR",
        production_actor_checkpoint=production_actor_checkpoint,
        v12_checkpoint=v12_checkpoint,
        fcpg_checkpoint=fcpg_checkpoint,
        device=device,
        require_fixed_endpoint=require_fixed_endpoint,
    )
    return model.eval(), payload


def load_soft_iir_checkpoint(
    checkpoint: str | Path,
    *,
    production_actor_checkpoint: str | Path,
    v12_checkpoint: str | Path,
    fcpg_checkpoint: str | Path,
    device: str | torch.device,
    require_fixed_endpoint: bool = True,
) -> tuple[ProdV13ActionInnovationSoftIIR025, dict[str, Any]]:
    model, payload = _load_typed_checkpoint(
        checkpoint,
        model_type=ProdV13ActionInnovationSoftIIR025,
        checkpoint_format=SOFT_IIR_CHECKPOINT_FORMAT,
        prod_v13_format=SOFT_IIR_FORMAT,
        method_revision=SOFT_IIR_METHOD_REVISION,
        label="action-innovation soft IIR",
        production_actor_checkpoint=production_actor_checkpoint,
        v12_checkpoint=v12_checkpoint,
        fcpg_checkpoint=fcpg_checkpoint,
        device=device,
        require_fixed_endpoint=require_fixed_endpoint,
    )
    return model.eval(), payload


__all__ = [
    "CHECKPOINT_FORMAT",
    "FORMAT",
    "IIR_CHECKPOINT_FORMAT",
    "IIR_FORMAT",
    "IIR_METHOD_REVISION",
    "IIR_WRITEBACK_GAIN",
    "METHOD_REVISION",
    "ProdV13ActionInnovation",
    "ProdV13ActionInnovationIIR025",
    "ProdV13ActionInnovationSoftIIR025",
    "ProdV13ActionInnovationOutput",
    "checkpoint_payload",
    "load_checkpoint",
    "load_iir_checkpoint",
    "load_soft_iir_checkpoint",
    "SOFT_IIR_CHECKPOINT_FORMAT",
    "SOFT_IIR_FORMAT",
    "SOFT_IIR_METHOD_REVISION",
]
