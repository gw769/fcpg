"""Prod-V13 recursive-v2: wider measurement support with trainable recurrence.

This is a deliberately minimal successor to recursive-v1.  It keeps the
frozen V12.4 and protected-memory v3.1 endpoints, the carrier, feedback head,
losses and hard runtime assignment unchanged.  It changes only:

* frozen FCPG measurement support: top-8 -> top-32;
* recurrent writeback backward path: detached hard token -> straight-through
  hard-forward/soft-backward token.

The forward recurrent value is still exactly one token sampled from the
current V12.4 patch map.  The soft value exists only in autograd, so the prior
cannot select outside current visual support or synthesize a pixel/token.
"""

from __future__ import annotations

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
    PATCHES,
    ProdV13ClosedLoop,
    RecurrentTopKFeedbackHead,
)


FORMAT = "so101-prod-v13-recursive-feedback-v2"
CHECKPOINT_FORMAT = "so101-prod-v13-recursive-feedback-checkpoint-v2"
METHOD_REVISION = "prod-v13-recursive-v2-top32-straight-through"
SUPPORT_SIZE = 32


class ProdV13RecursiveV2(ProdV13ClosedLoop):
    """Same carrier/head parameters as recursive-v1 with a corrected update."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            **kwargs,
            feedback=RecurrentTopKFeedbackHead(
                support_size=SUPPORT_SIZE,
                straight_through_writeback=True,
            ),
        )

    def parameter_contract(self) -> dict[str, Any]:
        contract = super().parameter_contract()
        contract.update(
            {
                "format": FORMAT,
                "method_revision": METHOD_REVISION,
                "measurement_support_size": SUPPORT_SIZE,
                "measurement_support_source": (
                    "runtime_top32_of_frozen_full_16x16_fcpg_logits"
                ),
                "posterior_forward": "hard_unique_current_patch_token",
                "posterior_writeback": (
                    "hard_forward_soft_backward_supported_posterior_token"
                ),
                "straight_through_soft_scope": "autograd_only_inside_top32",
            }
        )
        return contract


def target_valid_for_top32(
    *,
    topk_support: torch.Tensor,
    target_patch_index: torch.Tensor,
    target_visible: torch.Tensor,
    target_distinct: torch.Tensor,
    view_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive v2 targets from stored teacher truth and runtime top-32.

    The sidecar's ``target_valid`` is intentionally ignored because it was
    frozen under the historical top-8 contract.
    """

    if topk_support.ndim != 6 or topk_support.shape[-2:] != (16, 16):
        raise ValueError("top-32 support must be [B,4,2,2,16,16]")
    batch = len(topk_support)
    expected_tip = (batch, 4, 2, 2)
    expected_view = (batch, 4, 2)
    if tuple(target_patch_index.shape) != expected_tip:
        raise ValueError("target patch index must be [B,4,2,2]")
    if tuple(target_visible.shape) != expected_tip or target_visible.dtype != torch.bool:
        raise ValueError("target visibility must be bool [B,4,2,2]")
    if tuple(target_distinct.shape) != expected_view or target_distinct.dtype != torch.bool:
        raise ValueError("target distinctness must be bool [B,4,2]")
    if tuple(view_valid.shape) != expected_view or view_valid.dtype != torch.bool:
        raise ValueError("view validity must be bool [B,4,2]")
    target = target_patch_index.long()
    if bool(((target < 0) | (target >= PATCHES)).any()):
        raise ValueError("teacher target lies outside the 16x16 patch grid")
    support = topk_support.flatten(-2)
    counts = support.sum(dim=-1)
    expected_count = torch.where(
        view_valid[..., None],
        torch.full_like(counts, SUPPORT_SIZE),
        torch.zeros_like(counts),
    )
    if not torch.equal(counts, expected_count):
        raise ValueError("runtime measurement support is not top-32")
    in_support = support.gather(-1, target[..., None])[..., 0]
    valid = (
        target_visible
        & in_support
        & target_distinct[..., None]
        & view_valid[..., None]
    )
    return valid, in_support & target_visible


def checkpoint_payload(
    model: ProdV13RecursiveV2,
    *,
    step: int,
    fixed_endpoint_step: int,
    dependencies: Mapping[str, str],
    training_contract: Mapping[str, Any],
    sample_order_sha256: str,
) -> dict[str, Any]:
    state = model.trainable_state()
    required = {
        "v12_checkpoint_sha256": EXPECTED_V12_SHA256,
        "fcpg_checkpoint_sha256": EXPECTED_PROTECTED_MEMORY_SHA256,
        "production_actor_reference_sha256": FROZEN_PRODUCTION_ACTOR_SHA256,
    }
    for name, expected in required.items():
        if dependencies.get(name) != expected:
            raise ValueError(f"recursive-v2 dependency {name} mismatch")
    return {
        "format": CHECKPOINT_FORMAT,
        "prod_v13_format": FORMAT,
        "method_revision": METHOD_REVISION,
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


def load_checkpoint(
    checkpoint: str | Path,
    *,
    production_actor_checkpoint: str | Path,
    v12_checkpoint: str | Path,
    fcpg_checkpoint: str | Path,
    device: str | torch.device,
    require_fixed_endpoint: bool = True,
) -> tuple[ProdV13RecursiveV2, dict[str, Any]]:
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or payload.get("prod_v13_format") != FORMAT
        or payload.get("method_revision") != METHOD_REVISION
        or bool(payload.get("actor_checkpoint"))
        or bool(payload.get("actor_loss"))
        or bool(payload.get("sealed_test_read"))
    ):
        raise ValueError("recursive-v2 checkpoint identity/safety drifted")
    step = int(payload.get("step", -1))
    fixed = int(payload.get("fixed_endpoint_step", -1))
    if step <= 0 or fixed <= 0 or (require_fixed_endpoint and step != fixed):
        raise ValueError("recursive-v2 checkpoint is not its fixed endpoint")
    bindings = {
        "v12_checkpoint_sha256": sha256_file(v12_checkpoint),
        "fcpg_checkpoint_sha256": sha256_file(fcpg_checkpoint),
        "production_actor_reference_sha256": sha256_file(
            production_actor_checkpoint
        ),
    }
    for name, expected in bindings.items():
        if payload.get(name) != expected:
            raise ValueError(f"recursive-v2 checkpoint {name} binding mismatch")
    state = payload.get("trainable_state")
    if not isinstance(state, Mapping):
        raise ValueError("recursive-v2 checkpoint lacks trainable state")
    if payload.get("trainable_state_sha256") != prod_v13_state_dict_sha256(state):
        raise ValueError("recursive-v2 checkpoint trainable-state SHA drifted")
    model = ProdV13RecursiveV2.from_frozen_endpoints(
        production_actor_checkpoint=production_actor_checkpoint,
        v12_checkpoint=v12_checkpoint,
        fcpg_checkpoint=fcpg_checkpoint,
        device=device,
    )
    if payload.get("parameter_contract") != model.parameter_contract():
        raise ValueError("recursive-v2 checkpoint parameter contract drifted")
    model.load_trainable_state(state)
    return model.eval(), payload


__all__ = [
    "CHECKPOINT_FORMAT",
    "FORMAT",
    "METHOD_REVISION",
    "ProdV13RecursiveV2",
    "SUPPORT_SIZE",
    "checkpoint_payload",
    "load_checkpoint",
    "target_valid_for_top32",
]
