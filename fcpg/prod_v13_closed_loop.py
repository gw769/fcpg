"""Prod-V13 recursive V12.4 x protected-memory FCPG-H feedback.

The visual measurement is produced by the frozen protected-memory v3.1
observer.  V12.4 receives only global image context plus geometry, state and
physical actions: generic query tokens are deliberately *not* inserted into
its CAD visual-anchor memory.  A small recurrent head may re-rank only the
eight patches already supported by the current FCPG-H measurement.

The selected current-image token is detached before it becomes the recurrent
state.  This keeps action-conditioned posterior state out of the pure visual
FCPG-H memory while retaining a real numerical H1 -> H2 -> H4 feedback path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from fcpg.offline_pipeline_manifest import sha256_file
from fcpg.prod_v13 import (
    FROZEN_PRODUCTION_ACTOR_SHA256,
    ProdV13LeWMOutput,
    _load_v12_4,
    prod_v13_state_dict_sha256,
    validate_production_actor_checkpoint,
    v12_policy_role_inventory,
)
from fcpg.protected_visual_memory import (
    CHECKPOINT_FORMAT as PROTECTED_MEMORY_CHECKPOINT_FORMAT,
    METHOD_REVISION as PROTECTED_MEMORY_METHOD_REVISION,
    ProtectedVisualEvidenceMemory,
)
from v10.lewm_canonical_forward import lewm_canonical_output


FORMAT = "so101-prod-v13-v12-4-protected-memory-recursive-feedback-v1"
CHECKPOINT_FORMAT = "so101-prod-v13-recursive-feedback-checkpoint-v1"
METHOD_REVISION = "prod-v13-top8-recurrent-measurement-feedback-v1"

TOKEN_DIM = 192
HORIZONS = 4
VIEWS = 2
TIPS = 2
ANCHORS = 3
GRID = 16
PATCHES = GRID * GRID
TOPK = 8
IMAGE_SIZE = 224
PATCH_SIZE = IMAGE_SIZE / GRID
PATCH_CENTER_OFFSET = PATCH_SIZE / 2.0 - 0.5
FEEDBACK_BETA = 0.25
PRIOR_TEMPERATURE = 0.1
PRIOR_RESIDUAL_SCALE = 0.25
FEEDBACK_PARAMETER_LIMIT = 120_000
EXPECTED_V12_SHA256 = (
    "aca718b0a7335fd3ed3afce4bf70548ede1e194b6f9f0463ba4de0a36379c339"
)
EXPECTED_PROTECTED_MEMORY_SHA256 = (
    "31e7e513bda234672f488f2ac21e8c6dedc4103679793a99670d6dd95e4a17e1"
)


@dataclass(frozen=True)
class RecursiveFeedbackOutput:
    """Four-step measurement-dominant posterior rollout."""

    measurement_logits: torch.Tensor
    posterior_logits: torch.Tensor
    topk_support: torch.Tensor
    prior_tokens: torch.Tensor
    posterior_tip_tokens: torch.Tensor
    recurrent_tip_tokens: torch.Tensor
    posterior_anchor_tokens: torch.Tensor
    posterior_tip_patch_index: torch.Tensor
    posterior_tip_px: torch.Tensor
    posterior_anchor_px: torch.Tensor
    view_valid: torch.Tensor


@dataclass(frozen=True)
class ProdV13ClosedLoopOutput:
    dynamics_latent: torch.Tensor
    feedback: RecursiveFeedbackOutput


def _patch_centres(index: torch.Tensor) -> torch.Tensor:
    if index.dtype != torch.long:
        index = index.long()
    if bool(((index < 0) | (index >= PATCHES)).any()):
        raise ValueError("patch index lies outside the 16x16 grid")
    x = index.remainder(GRID).float() * PATCH_SIZE + PATCH_CENTER_OFFSET
    y = torch.div(index, GRID, rounding_mode="floor").float() * PATCH_SIZE
    y = y + PATCH_CENTER_OFFSET
    return torch.stack((x, y), dim=-1)


def _unique_tip_indices(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Parameter-free maximum joint assignment with distinct patches."""

    if logits.ndim != 4 or logits.shape[2:] != (TIPS, PATCHES):
        raise ValueError("tip logits must be [B,V,2,256]")
    if valid.shape != logits.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("view valid mask must be bool [B,V]")
    joint = logits[:, :, 0, :, None] + logits[:, :, 1, None, :]
    diagonal = torch.eye(PATCHES, dtype=torch.bool, device=logits.device)
    joint = joint.masked_fill(diagonal, float("-inf"))
    selected = joint.flatten(-2).argmax(dim=-1)
    first = torch.div(selected, PATCHES, rounding_mode="floor")
    second = selected.remainder(PATCHES)
    result = torch.stack((first, second), dim=-1)
    return torch.where(valid[..., None], result, torch.zeros_like(result))


def _gather_tokens(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 4 or tokens.shape[2:] != (PATCHES, TOKEN_DIM):
        raise ValueError("patch tokens must be [B,V,256,192]")
    if index.shape != (*tokens.shape[:2], TIPS):
        raise ValueError("tip index must be [B,V,2]")
    return torch.gather(
        tokens,
        2,
        index[..., None].expand(-1, -1, -1, TOKEN_DIM),
    )


class RecurrentTopKFeedbackHead(nn.Module):
    """One shared recurrent token prior restricted to FCPG support.

    The default arguments preserve the registered recursive-v1 endpoint.  A
    successor may widen the frozen measurement support and opt into a
    hard-forward/soft-backward recurrent token without changing the head
    parameters or the visible hard assignment.
    """

    def __init__(
        self,
        *,
        token_dim: int = TOKEN_DIM,
        support_size: int = TOPK,
        straight_through_writeback: bool = False,
    ) -> None:
        super().__init__()
        if int(token_dim) != TOKEN_DIM:
            raise ValueError("Prod-V13 feedback is locked to 192-D V12.4 tokens")
        if not 2 <= int(support_size) < PATCHES:
            raise ValueError("feedback support must contain 2..255 patches")
        self.support_size = int(support_size)
        self.straight_through_writeback = bool(straight_through_writeback)
        self.input_norm = nn.LayerNorm(2 * TOKEN_DIM)
        self.hidden = nn.Linear(2 * TOKEN_DIM, TOKEN_DIM)
        self.output = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if self.parameter_count > FEEDBACK_PARAMETER_LIMIT:
            raise RuntimeError("recursive feedback head exceeds its parameter limit")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _validate(
        dynamics_latent: torch.Tensor,
        previous_tokens: torch.Tensor,
        future_patch_tokens: torch.Tensor,
        measurement_logits: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> int:
        if dynamics_latent.ndim != 3:
            raise ValueError("dynamics latent must be [B,4,192]")
        batch = len(dynamics_latent)
        if tuple(dynamics_latent.shape) != (batch, HORIZONS, TOKEN_DIM):
            raise ValueError("dynamics latent must be [B,4,192]")
        if tuple(previous_tokens.shape) != (batch, VIEWS, TIPS, TOKEN_DIM):
            raise ValueError("initial posterior tokens must be [B,2,2,192]")
        if tuple(future_patch_tokens.shape) != (
            batch,
            HORIZONS,
            VIEWS,
            PATCHES,
            TOKEN_DIM,
        ):
            raise ValueError("future patch tokens must be [B,4,2,256,192]")
        if tuple(measurement_logits.shape) != (
            batch,
            HORIZONS,
            VIEWS,
            TIPS,
            GRID,
            GRID,
        ):
            raise ValueError("measurement logits must be [B,4,2,2,16,16]")
        if tuple(view_valid.shape) != (batch, HORIZONS, VIEWS):
            raise ValueError("future view_valid must be [B,4,2]")
        if view_valid.dtype != torch.bool:
            raise TypeError("future view_valid must be boolean")
        floating = (
            dynamics_latent,
            previous_tokens,
            future_patch_tokens,
        )
        if not all(value.is_floating_point() for value in floating):
            raise TypeError("recursive feedback inputs must be floating point")
        if not all(bool(torch.isfinite(value).all()) for value in floating):
            raise ValueError("recursive feedback inputs contain non-finite values")
        flat_measurement = measurement_logits.flatten(-2)
        finite_or_invalid = torch.isfinite(flat_measurement) | ~view_valid[
            :, :, :, None, None
        ]
        if not bool(finite_or_invalid.all()):
            raise ValueError("valid FCPG measurement logits must be finite")
        return batch

    def forward(
        self,
        dynamics_latent: torch.Tensor,
        previous_tokens: torch.Tensor,
        future_patch_tokens: torch.Tensor,
        measurement_logits: torch.Tensor,
        view_valid: torch.Tensor,
        *,
        topk_support: torch.Tensor | None = None,
        recursive_writeback: bool = True,
        reference_dynamics_latent: torch.Tensor | None = None,
        center_prior_similarity: bool = False,
        writeback_gain: float = 1.0,
        soft_writeback_forward: bool = False,
    ) -> RecursiveFeedbackOutput:
        batch = self._validate(
            dynamics_latent,
            previous_tokens,
            future_patch_tokens,
            measurement_logits,
            view_valid,
        )
        action_innovation: torch.Tensor | None = None
        if reference_dynamics_latent is not None:
            if reference_dynamics_latent.shape != dynamics_latent.shape:
                raise ValueError(
                    "reference dynamics latent must match factual dynamics latent"
                )
            if not reference_dynamics_latent.is_floating_point():
                raise TypeError("reference dynamics latent must be floating point")
            if not bool(torch.isfinite(reference_dynamics_latent).all()):
                raise ValueError("reference dynamics latent contains non-finite values")
            # Common-mode rejection: only the action-conditioned innovation is
            # visible to the feedback head.  The matched zero branch removes
            # the previous-token/head bias exactly, so action==hold cannot
            # manufacture a visual correction through a learned shortcut.
            action_innovation = dynamics_latent - reference_dynamics_latent
        gain = float(writeback_gain)
        if not 0.0 <= gain <= 1.0:
            raise ValueError("writeback gain must lie in [0,1]")
        measurement = measurement_logits.detach().float().flatten(-2)
        patches = future_patch_tokens.float()
        previous = F.normalize(previous_tokens.float(), dim=-1, eps=1e-8)

        posterior_logits: list[torch.Tensor] = []
        support_masks: list[torch.Tensor] = []
        priors: list[torch.Tensor] = []
        posterior_parts: list[torch.Tensor] = []
        recurrent_parts: list[torch.Tensor] = []
        posterior_anchors: list[torch.Tensor] = []
        selected_indices: list[torch.Tensor] = []
        tip_pixels: list[torch.Tensor] = []
        anchor_pixels: list[torch.Tensor] = []

        for horizon in range(HORIZONS):
            dynamics_source = (
                action_innovation
                if action_innovation is not None
                else dynamics_latent
            )
            dynamics = dynamics_source[:, horizon, None, None].expand(
                -1, VIEWS, TIPS, -1
            )
            hidden = F.gelu(
                self.hidden(self.input_norm(torch.cat((previous, dynamics), dim=-1)))
            )
            raw_residual = self.output(hidden)
            if action_innovation is not None:
                zero_dynamics = torch.zeros_like(dynamics)
                zero_hidden = F.gelu(
                    self.hidden(
                        self.input_norm(
                            torch.cat((previous, zero_dynamics), dim=-1)
                        )
                    )
                )
                raw_residual = raw_residual - self.output(zero_hidden)
            residual = PRIOR_RESIDUAL_SCALE * torch.tanh(raw_residual)
            prior = F.normalize(previous + residual, dim=-1, eps=1e-8)
            current = patches[:, horizon]
            normalized_patches = F.normalize(current, dim=-1, eps=1e-8)
            similarity = torch.einsum(
                "bvtd,bvpd->bvtp", prior, normalized_patches
            ) / PRIOR_TEMPERATURE
            if center_prior_similarity:
                if action_innovation is None:
                    raise ValueError(
                        "centered prior similarity requires reference dynamics"
                    )
                baseline_prior = F.normalize(previous, dim=-1, eps=1e-8)
                baseline_similarity = torch.einsum(
                    "bvtd,bvpd->bvtp", baseline_prior, normalized_patches
                ) / PRIOR_TEMPERATURE
                similarity = similarity - baseline_similarity

            measured = measurement[:, horizon]
            valid = view_valid[:, horizon]
            if topk_support is None:
                support_index = measured.topk(self.support_size, dim=-1).indices
                support = torch.zeros_like(measured, dtype=torch.bool)
                support.scatter_(-1, support_index, True)
                support &= valid[:, :, None, None]
            else:
                if topk_support.shape != measurement_logits.shape:
                    raise ValueError("stored topk support must match measurement logits")
                if topk_support.dtype != torch.bool:
                    raise TypeError("stored topk support must be boolean")
                support = topk_support[:, horizon].detach().flatten(-2)
                expected_count = torch.where(
                    valid[:, :, None],
                    torch.full_like(
                        support.sum(dim=-1), self.support_size
                    ),
                    torch.zeros_like(support.sum(dim=-1)),
                )
                if not torch.equal(support.sum(dim=-1), expected_count):
                    raise ValueError("stored topk support count/view mask drifted")
            # FCPG/current-patch evidence is frozen.  The prior path stays
            # differentiable because recursive CE is its only task loss.
            combined = measured + FEEDBACK_BETA * similarity
            combined = combined.masked_fill(~support, float("-inf"))
            index = _unique_tip_indices(combined, valid)
            part = _gather_tokens(current, index)
            recurrent_part = part.detach()
            if self.straight_through_writeback:
                # Forward semantics remain one hard token sampled from the
                # current image.  Backward semantics use the same supported
                # posterior distribution so H2--H4 losses can train the H1
                # choice instead of stopping at argmax + detach.
                safe_combined = torch.where(
                    valid[:, :, None, None],
                    combined,
                    torch.zeros_like(combined),
                )
                probability = torch.softmax(safe_combined, dim=-1)
                probability = probability * support.to(probability.dtype)
                soft_part = torch.einsum(
                    "bvtp,bvpd->bvtd", probability, current
                )
                if soft_writeback_forward:
                    recurrent_part = soft_part
                else:
                    recurrent_part = soft_part + (part - soft_part).detach()
            elif soft_writeback_forward:
                raise ValueError(
                    "soft writeback requires the supported posterior path"
                )
            px = _patch_centres(index).to(current.dtype)
            center_px = px.mean(dim=2, keepdim=True)
            anchor_px = torch.cat((center_px, px), dim=2)
            center_index = (
                torch.floor(anchor_px[:, :, 0, 1].float() / PATCH_SIZE).long()
                .clamp(0, GRID - 1)
                * GRID
                + torch.floor(anchor_px[:, :, 0, 0].float() / PATCH_SIZE)
                .long()
                .clamp(0, GRID - 1)
            )
            center_token = torch.gather(
                current,
                2,
                center_index[:, :, None, None].expand(-1, -1, 1, TOKEN_DIM),
            )
            anchor = torch.cat((center_token, part), dim=2)
            token_mask = valid[:, :, None, None]
            part = part * token_mask
            recurrent_part = recurrent_part * token_mask
            anchor = anchor * token_mask
            px = px * valid[:, :, None, None]
            anchor_px = anchor_px * valid[:, :, None, None]

            posterior_logits.append(combined.reshape(batch, VIEWS, TIPS, GRID, GRID))
            support_masks.append(support.reshape(batch, VIEWS, TIPS, GRID, GRID))
            priors.append(prior)
            posterior_parts.append(part)
            posterior_anchors.append(anchor)
            selected_indices.append(index)
            tip_pixels.append(px)
            anchor_pixels.append(anchor_px)
            # Hard current-image measurement becomes the next recurrent state;
            # no action-conditioned token is ever written into FCPG memory.
            if recursive_writeback and gain > 0.0:
                previous = F.normalize(
                    (1.0 - gain) * previous + gain * recurrent_part.float(),
                    dim=-1,
                    eps=1e-8,
                )
            recurrent_parts.append(previous)

        return RecursiveFeedbackOutput(
            measurement_logits=measurement_logits.detach(),
            posterior_logits=torch.stack(posterior_logits, dim=1),
            topk_support=torch.stack(support_masks, dim=1),
            prior_tokens=torch.stack(priors, dim=1),
            posterior_tip_tokens=torch.stack(posterior_parts, dim=1),
            recurrent_tip_tokens=torch.stack(recurrent_parts, dim=1),
            posterior_anchor_tokens=torch.stack(posterior_anchors, dim=1),
            posterior_tip_patch_index=torch.stack(selected_indices, dim=1),
            posterior_tip_px=torch.stack(tip_pixels, dim=1),
            posterior_anchor_px=torch.stack(anchor_pixels, dim=1),
            view_valid=view_valid,
        )


def masked_recursive_spatial_ce(
    logits: torch.Tensor,
    target_patch_index: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """Per-row recursive CE over only valid view/tip targets."""

    if logits.ndim != 6 or logits.shape[1:] != (
        HORIZONS,
        VIEWS,
        TIPS,
        GRID,
        GRID,
    ):
        raise ValueError("recursive logits must be [B,4,2,2,16,16]")
    target = torch.as_tensor(target_patch_index, dtype=torch.long, device=logits.device)
    valid = torch.as_tensor(target_valid, dtype=torch.bool, device=logits.device)
    if target.shape != logits.shape[:4] or valid.shape != target.shape:
        raise ValueError("recursive targets and validity must be [B,4,2,2]")
    if bool(((target[valid] < 0) | (target[valid] >= PATCHES)).any()):
        raise ValueError("valid recursive target lies outside the patch grid")
    flat = logits.flatten(-2)
    # A target outside current FCPG top-8 support intentionally has infinite
    # CE and cannot be hidden by clamping or silently widening the support.
    selected = flat.gather(-1, target.clamp(0, PATCHES - 1)[..., None])[..., 0]
    if not bool(torch.isfinite(selected[valid]).all()):
        raise ValueError("recursive target lies outside current FCPG top-8 support")
    counts = valid.sum(dim=(1, 2, 3))
    if bool((counts == 0).any()):
        raise ValueError("recursive batch contains a row with no valid target")
    # Evaluate CE only at valid targets.  Computing it for an invalid target
    # outside the frozen top-8 produces +inf; multiplying that value by a
    # false mask afterwards is still NaN (0 * inf), which poisoned step 1 of
    # the first formal run.
    error = torch.zeros_like(target, dtype=flat.dtype)
    error[valid] = F.cross_entropy(
        flat[valid],
        target[valid],
        reduction="none",
    )
    return error.sum(dim=(1, 2, 3)) / counts


def load_frozen_protected_memory(
    checkpoint: str | Path,
    *,
    device: str | torch.device,
) -> tuple[ProtectedVisualEvidenceMemory, dict[str, Any]]:
    path = Path(checkpoint).expanduser().resolve()
    if sha256_file(path) != EXPECTED_PROTECTED_MEMORY_SHA256:
        raise ValueError("Prod-V13 requires the frozen protected-memory v3.1 endpoint")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format") != PROTECTED_MEMORY_CHECKPOINT_FORMAT
        or payload.get("method_revision") != PROTECTED_MEMORY_METHOD_REVISION
        or payload.get("arm") != "FCPGH"
    ):
        raise ValueError("protected-memory checkpoint identity drifted")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("protected-memory checkpoint lacks model state")
    model = ProtectedVisualEvidenceMemory().to(device)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.eval(), payload


class ProdV13ClosedLoop(nn.Module):
    """Trainable V12.4 carrier plus recurrent head over frozen FCPG outputs."""

    def __init__(
        self,
        *,
        v12: nn.Module,
        fcpg: ProtectedVisualEvidenceMemory,
        production_actor_reference: Mapping[str, Any],
        feedback: RecurrentTopKFeedbackHead | None = None,
    ) -> None:
        super().__init__()
        self.v12 = v12
        self.fcpg = fcpg
        self.feedback = feedback or RecurrentTopKFeedbackHead()
        self.production_actor_reference = dict(production_actor_reference)
        self._configure_trainability()

    @classmethod
    def from_frozen_endpoints(
        cls,
        *,
        production_actor_checkpoint: str | Path,
        v12_checkpoint: str | Path,
        fcpg_checkpoint: str | Path,
        device: str | torch.device,
    ) -> "ProdV13ClosedLoop":
        if sha256_file(v12_checkpoint) != EXPECTED_V12_SHA256:
            raise ValueError("Prod-V13 closed loop requires V12.4 step-20000")
        actor = validate_production_actor_checkpoint(production_actor_checkpoint)
        if actor.get("sha256") not in (None, FROZEN_PRODUCTION_ACTOR_SHA256):
            raise ValueError("production actor reference SHA drifted")
        v12, _ = _load_v12_4(Path(v12_checkpoint), device=torch.device(device))
        fcpg, _ = load_frozen_protected_memory(fcpg_checkpoint, device=device)
        # ``feedback`` is constructed by ``cls`` after both frozen endpoints
        # have already been moved.  Move the assembled carrier once more so
        # the newly created recurrent head cannot remain on CPU while cached
        # V12.4 tensors live on CUDA.
        return cls(
            v12=v12,
            fcpg=fcpg,
            production_actor_reference=actor,
        ).to(device)

    def _configure_trainability(self) -> None:
        self.requires_grad_(False)
        self.v12.context_projector.requires_grad_(True)
        self.v12.patch_projector.requires_grad_(True)
        self.v12.policy.requires_grad_(True)
        roles = v12_policy_role_inventory(self.v12.policy)
        for name in roles["parameter_names"]["actor_only_frozen"]:
            self.v12.policy.get_parameter(name).requires_grad_(False)
        self.feedback.requires_grad_(True)
        self.fcpg.requires_grad_(False)
        self.v12.vit.requires_grad_(False)
        self.v12.bridge.requires_grad_(False)
        self.v12.corrector.requires_grad_(False)
        self.v12.anchor_reliability_head.requires_grad_(False)
        self.v12.view_reliability_head.requires_grad_(False)

    def train(self, mode: bool = True) -> "ProdV13ClosedLoop":
        super().train(mode)
        self.fcpg.eval()
        self.v12.vit.eval()
        self.v12.bridge.eval()
        self.v12.corrector.eval()
        self.v12.anchor_reliability_head.eval()
        self.v12.view_reliability_head.eval()
        return self

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def parameter_contract(self) -> dict[str, Any]:
        trainable = {
            name: list(parameter.shape)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        frozen = {
            name: list(parameter.shape)
            for name, parameter in self.named_parameters()
            if not parameter.requires_grad
        }
        return {
            "format": FORMAT,
            "method_revision": METHOD_REVISION,
            "trainable_parameter_names": trainable,
            "frozen_parameter_names": frozen,
            "trainable_parameters": sum(
                self.get_parameter(name).numel() for name in trainable
            ),
            "feedback_parameters": self.feedback.parameter_count,
            "v12_visual_anchor_memory": "masked_off_for_generic_query_training",
            "v12_global_context": True,
            "v12_geometry_state_action": True,
            "v12_bundled_actor_output_used": False,
            "production_actor_loaded_into_optimizer": False,
            "production_actor_reference": self.production_actor_reference,
            "prior_score_backward": (
                "differentiable_prior_frozen_measurement_and_patch_evidence"
            ),
            "posterior_writeback": "hard_current_patch_token_detached",
        }

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.named_parameters()
            if value.requires_grad
        }

    def load_trainable_state(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        if set(state) != expected:
            raise ValueError("closed-loop trainable checkpoint names drifted")
        with torch.no_grad():
            for name, value in state.items():
                parameter = self.get_parameter(name)
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"closed-loop checkpoint shape drifted for {name}")
                parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))

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
        detach_carrier: bool = False,
    ) -> ProdV13ClosedLoopOutput:
        dynamics_latent = self.predict_dynamics_cached(
            vit_cls_history=vit_cls_history,
            history_view_valid=history_view_valid,
            anchor_xyz_history=anchor_xyz_history,
            state_history=state_history,
            past_action_chunks=past_action_chunks,
            action=action,
        )
        projected_future = self.v12.patch_projector(future_vit_patch_tokens)
        projected_initial = self.v12.patch_projector(initial_posterior_tokens)
        if detach_carrier:
            # System-identification warm-up: the already trained V12.4
            # transition supplies a fixed control response while only the
            # observer/feedback head learns how to consume it.  This is a
            # training-path stop-gradient, not a second inference model.
            dynamics_latent = dynamics_latent.detach()
            projected_future = projected_future.detach()
            projected_initial = projected_initial.detach()
        feedback = self.feedback(
            dynamics_latent,
            projected_initial,
            projected_future,
            measurement_logits,
            future_view_valid.bool(),
            topk_support=measurement_topk_support,
            recursive_writeback=recursive_writeback,
        )
        return ProdV13ClosedLoopOutput(
            dynamics_latent=dynamics_latent,
            feedback=feedback,
        )

    def predict_dynamics_cached(
        self,
        *,
        vit_cls_history: torch.Tensor,
        history_view_valid: torch.Tensor,
        anchor_xyz_history: torch.Tensor,
        state_history: torch.Tensor,
        past_action_chunks: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Run the goal-free V12.4 transition without future visual evidence."""

        if vit_cls_history.ndim != 4 or vit_cls_history.shape[1:] != (
            2,
            VIEWS,
            TOKEN_DIM,
        ):
            raise ValueError("vit_cls_history must be [B,2,2,192]")
        batch = len(vit_cls_history)
        if history_view_valid.shape != (batch, 2, VIEWS):
            raise ValueError("history_view_valid must be [B,2,2]")
        if not bool(history_view_valid.any(dim=(1, 2)).all()):
            raise ValueError("each row needs one valid global history view")
        context = self.v12.context_projector(vit_cls_history)
        zero_anchor_tokens = torch.zeros(
            batch,
            2,
            VIEWS,
            ANCHORS,
            TOKEN_DIM,
            device=context.device,
            dtype=context.dtype,
        )
        anchor_valid = torch.zeros(
            batch, 2, VIEWS, ANCHORS, device=context.device, dtype=torch.bool
        )
        q_px = torch.zeros(
            batch,
            2,
            VIEWS,
            ANCHORS,
            2,
            device=context.device,
            dtype=context.dtype,
        )
        output = lewm_canonical_output(
            self.v12.policy,
            per_view_context=context,
            per_view_anchor_tokens=zero_anchor_tokens,
            anchor_valid=anchor_valid,
            view_valid=history_view_valid.bool(),
            q_corrected_px=q_px,
            anchor_xyz_history=anchor_xyz_history,
            state_history=state_history,
            past_action_chunks=past_action_chunks,
            action=action,
        )
        return output.dynamics_latent


def checkpoint_payload(
    model: ProdV13ClosedLoop,
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
            raise ValueError(f"closed-loop dependency {name} mismatch")
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


def load_prod_v13_closed_loop_checkpoint(
    checkpoint: str | Path,
    *,
    production_actor_checkpoint: str | Path,
    v12_checkpoint: str | Path,
    fcpg_checkpoint: str | Path,
    device: str | torch.device,
    require_fixed_endpoint: bool = True,
) -> tuple[ProdV13ClosedLoop, dict[str, Any]]:
    """Restore only the registered carrier/head state over exact dependencies."""

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
        raise ValueError("closed-loop checkpoint identity/safety contract drifted")
    step = int(payload.get("step", -1))
    fixed = int(payload.get("fixed_endpoint_step", -1))
    if step <= 0 or fixed <= 0 or (require_fixed_endpoint and step != fixed):
        raise ValueError("closed-loop checkpoint is not its fixed endpoint")
    bindings = {
        "v12_checkpoint_sha256": sha256_file(v12_checkpoint),
        "fcpg_checkpoint_sha256": sha256_file(fcpg_checkpoint),
        "production_actor_reference_sha256": sha256_file(
            production_actor_checkpoint
        ),
    }
    for name, expected in bindings.items():
        if payload.get(name) != expected:
            raise ValueError(f"closed-loop checkpoint {name} binding mismatch")
    state = payload.get("trainable_state")
    if not isinstance(state, Mapping):
        raise ValueError("closed-loop checkpoint lacks trainable state")
    if payload.get("trainable_state_sha256") != prod_v13_state_dict_sha256(state):
        raise ValueError("closed-loop checkpoint trainable-state SHA drifted")
    model = ProdV13ClosedLoop.from_frozen_endpoints(
        production_actor_checkpoint=production_actor_checkpoint,
        v12_checkpoint=v12_checkpoint,
        fcpg_checkpoint=fcpg_checkpoint,
        device=device,
    )
    if payload.get("parameter_contract") != model.parameter_contract():
        raise ValueError("closed-loop checkpoint parameter contract drifted")
    model.load_trainable_state(state)
    return model.eval(), payload


__all__ = [
    "CHECKPOINT_FORMAT",
    "EXPECTED_PROTECTED_MEMORY_SHA256",
    "EXPECTED_V12_SHA256",
    "FORMAT",
    "HORIZONS",
    "METHOD_REVISION",
    "PATCHES",
    "ProdV13ClosedLoop",
    "ProdV13ClosedLoopOutput",
    "RecursiveFeedbackOutput",
    "RecurrentTopKFeedbackHead",
    "TOPK",
    "checkpoint_payload",
    "load_frozen_protected_memory",
    "load_prod_v13_closed_loop_checkpoint",
    "masked_recursive_spatial_ce",
]
