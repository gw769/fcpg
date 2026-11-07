"""Small multi-rate FCPG-H -> LeWM predict/measure loop.

The visual observer and the robot transition are deliberately different
subsystems.  FCPG-H owns the current-image measurement and is the only module
that can introduce pixel support.  The transition sees only the previous
posterior token and *increments* of completed commands/CAD motion.  Its prior
may re-rank the current FCPG-H support, but it cannot emit a coordinate or add
support outside the current image.

This module does not instantiate the V12.4 carrier, ViT, or either actor.  Raw
V12.4 patch tokens and frozen FCPG-H logits are inputs, which makes the cut
explicit and keeps the trainable dynamics small enough to debug quickly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fcpg.modules import sample_feature_map_at_pixels


TOKEN_DIM = 192
HORIZONS = 4
VIEWS = 2
TIPS = 2
GRID = 16
PATCHES = GRID * GRID
ACTION_STEPS = 20
STEPS_PER_HORIZON = ACTION_STEPS // HORIZONS
COMMAND_ENDPOINTS = (4, 9, 14, 19)
SUPPORT_SIZE = 32
FEEDBACK_BETA = 0.25
PRIOR_TEMPERATURE = 0.1
PRIOR_RESIDUAL_SCALE = 0.25
WRITEBACK_GAIN = 0.25
# The token prior is a tie-breaker, not a second visual detector.  Clipping the
# temperature-scaled innovation keeps each candidate correction in
# [-FEEDBACK_BETA, +FEEDBACK_BETA].  Consequently a two-tip measurement-pair
# lead greater than 4 * FEEDBACK_BETA cannot be overturned by the prior.
MAX_NORMALIZED_INNOVATION = 1.0
PARAMETER_LIMIT = 300_000
PATCH_SIZE = 224.0 / GRID
PATCH_CENTER_OFFSET = PATCH_SIZE / 2.0 - 0.5
# Diagnostic only: projected tips closer than this are not visually separable
# at the current resolution.  It never excludes a candidate or raises.
TIP_DEGENERATE_RADIUS_PX = 1.0


@dataclass(frozen=True)
class MultiRateFeedbackOutput:
    """Four-horizon output of the separated predict/measure loop."""

    measurement_logits: torch.Tensor
    posterior_logits: torch.Tensor
    topk_support: torch.Tensor
    prior_tokens: torch.Tensor
    posterior_tokens: torch.Tensor
    recurrent_tokens: torch.Tensor
    dynamics_latent: torch.Tensor
    posterior_tip_patch_index: torch.Tensor
    posterior_tip_px: torch.Tensor
    posterior_center_px: torch.Tensor
    tip_pair_degenerate: torch.Tensor
    view_valid: torch.Tensor


@dataclass(frozen=True)
class PartTokenPriorOutput:
    """One low-rate LeWM transition, independent of visual measurement."""

    prior_tokens: torch.Tensor
    dynamics_latent: torch.Tensor


@dataclass(frozen=True)
class TrackerCandidateFeedbackOutput:
    """Measurement-dominant update over current tracker candidates only."""

    posterior_logits: torch.Tensor
    candidate_support: torch.Tensor
    prior_similarity: torch.Tensor
    measurement_candidate_index: torch.Tensor
    selected_candidate_index: torch.Tensor
    prior_changed_selection: torch.Tensor
    posterior_tip_px: torch.Tensor
    posterior_center_px: torch.Tensor
    posterior_tokens: torch.Tensor
    tip_pair_degenerate: torch.Tensor
    view_valid: torch.Tensor
    measurement_missing: torch.Tensor


@dataclass(frozen=True)
class TrackerCandidateRolloutOutput:
    """Causal token feedback over the three H1->H2->H3->H4 transitions."""

    prior_tokens: torch.Tensor
    posterior_tokens: torch.Tensor
    posterior_logits: torch.Tensor
    candidate_support: torch.Tensor
    measurement_candidate_index: torch.Tensor
    selected_candidate_index: torch.Tensor
    prior_changed_selection: torch.Tensor
    posterior_tip_px: torch.Tensor
    posterior_center_px: torch.Tensor
    dynamics_latent: torch.Tensor
    tip_pair_degenerate: torch.Tensor
    view_valid: torch.Tensor
    measurement_missing: torch.Tensor


def causal_action_increments(
    current_state: torch.Tensor,
    absolute_actions: torch.Tensor,
) -> torch.Tensor:
    """Convert absolute SO-101 commands to four causal five-step increments."""

    state = torch.as_tensor(current_state)
    action = torch.as_tensor(
        absolute_actions,
        device=state.device,
        dtype=state.dtype,
    )
    if state.ndim != 2 or state.shape[1] != 6:
        raise ValueError("current_state must be [B,6]")
    if action.shape != (len(state), ACTION_STEPS, 6):
        raise ValueError("absolute_actions must be [B,20,6]")
    if not state.is_floating_point() or not action.is_floating_point():
        raise TypeError("state/action must be floating point")
    if not bool(torch.isfinite(state).all() and torch.isfinite(action).all()):
        raise ValueError("state/action must be finite")
    sequence = torch.cat((state[:, None], action), dim=1)
    increments = sequence[:, 1:] - sequence[:, :-1]
    return increments.reshape(len(state), HORIZONS, STEPS_PER_HORIZON, 6)


def tracker_confidence_to_log_odds(
    confidence: torch.Tensor,
    *,
    eps: float = 1.0e-4,
) -> torch.Tensor:
    """Convert tracker probabilities to the feedback score contract.

    ``tracker_candidate_feedback`` consumes additive log-odds, never raw
    ``[0, 1]`` confidence.  This explicit conversion prevents the bounded
    token-prior tie-breaker from becoming the dominant numerical scale.
    """

    value = torch.as_tensor(confidence)
    if not value.is_floating_point():
        raise TypeError("tracker confidence must be floating point")
    if not 0.0 < float(eps) < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("tracker confidence must be finite")
    if not bool(((value >= 0.0) & (value <= 1.0)).all()):
        raise ValueError("tracker confidence must lie in [0,1]")
    return torch.logit(value.clamp(float(eps), 1.0 - float(eps)))


def causal_cad_tip_increments(
    current_tip_xyz: torch.Tensor,
    command_tip_xyz: torch.Tensor,
) -> torch.Tensor:
    """Convert four command-FK endpoints to per-horizon tip displacement."""

    current = torch.as_tensor(current_tip_xyz)
    command = torch.as_tensor(
        command_tip_xyz,
        device=current.device,
        dtype=current.dtype,
    )
    if current.ndim != 3 or current.shape[1:] != (TIPS, 3):
        raise ValueError("current_tip_xyz must be [B,2,3]")
    if command.shape != (len(current), HORIZONS, TIPS, 3):
        raise ValueError("command_tip_xyz must be [B,4,2,3]")
    if not current.is_floating_point() or not command.is_floating_point():
        raise TypeError("CAD positions must be floating point")
    if not bool(torch.isfinite(current).all() and torch.isfinite(command).all()):
        raise ValueError("CAD positions must be finite")
    sequence = torch.cat((current[:, None], command), dim=1)
    return sequence[:, 1:] - sequence[:, :-1]


def unordered_query_cad_signal(cad_tip_increments: torch.Tensor) -> torch.Tensor:
    """Represent gripper motion without binding query A/B to fixed/moving.

    Both visual slots receive the same rigid-centre displacement and the same
    physical jaw-separation change. Their previous visual tokens, rather than
    an arbitrary generic-query ordering, determine their token transitions.
    """

    value = torch.as_tensor(cad_tip_increments)
    if value.ndim != 4 or value.shape[1:] != (HORIZONS, TIPS, 3):
        raise ValueError("cad_tip_increments must be [B,4,2,3]")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("CAD increments must be finite floating point")
    centre = value.mean(dim=2)
    jaw_relative = value[:, :, 1] - value[:, :, 0]
    return torch.cat((centre, jaw_relative), dim=-1)


def signal_rms(value: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
    """Training-set RMS used to compare unlike physical signal channels."""

    tensor = torch.as_tensor(value).float()
    if not tensor.numel() or not bool(torch.isfinite(tensor).all()):
        raise ValueError("signal RMS requires a non-empty finite tensor")
    rms = tensor.square().mean(dim=dimensions).sqrt()
    return rms.clamp_min(1.0e-4)


def _patch_centres(index: torch.Tensor) -> torch.Tensor:
    if index.dtype != torch.long:
        index = index.long()
    if bool(((index < 0) | (index >= PATCHES)).any()):
        raise ValueError("patch index lies outside the 16x16 grid")
    x = index.remainder(GRID).float() * PATCH_SIZE + PATCH_CENTER_OFFSET
    y = torch.div(index, GRID, rounding_mode="floor").float() * PATCH_SIZE
    y = y + PATCH_CENTER_OFFSET
    return torch.stack((x, y), dim=-1)


def _tip_indices(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Select each query-owned tip slot without a false pixel-uniqueness rule."""

    if logits.ndim != 4 or logits.shape[2:] != (TIPS, PATCHES):
        raise ValueError("tip logits must be [B,V,2,256]")
    if valid.shape != logits.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("view valid mask must be bool [B,V]")
    result = logits.argmax(dim=-1)
    return torch.where(valid[..., None], result, torch.zeros_like(result))


def _select_tracker_candidates(
    *,
    logits: torch.Tensor,
    support: torch.Tensor,
    view_valid: torch.Tensor,
) -> torch.Tensor:
    """Select candidates from independent query-owned tracker banks.

    Candidate index and coordinate equality across tip A/B do not identify a
    duplicated proposal: each bank belongs to a different initialized track.
    Closed jaws may therefore legitimately select the same image coordinate.
    """

    selected = logits.masked_fill(~support, float("-inf")).argmax(dim=-1)
    return torch.where(
        view_valid[..., None], selected, torch.zeros_like(selected)
    )


def tracker_candidate_feedback(
    *,
    previous_tokens: torch.Tensor,
    prior_tokens: torch.Tensor,
    current_patch_tokens: torch.Tensor,
    candidate_px: torch.Tensor,
    measurement_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    view_valid: torch.Tensor,
    beta: float = FEEDBACK_BETA,
    temperature: float = PRIOR_TEMPERATURE,
) -> TrackerCandidateFeedbackOutput:
    """Re-rank only candidates measured in the current image.

    Candidate coordinates and additive measurement log-odds must come from
    the frozen visual tracker.  Raw ``[0,1]`` confidence must first pass
    through :func:`tracker_confidence_to_log_odds`.  The token prior contributes
    at most ``+/-0.25`` to any candidate and therefore only breaks measurement
    near-ties.  Tip A/B own separate tracker banks and may select the same
    coordinate when the jaw is closed.  Missing candidates invalidate only
    that sample/view.  No coordinate is transported from the previous frame.
    """

    previous = torch.as_tensor(previous_tokens)
    prior = torch.as_tensor(
        prior_tokens, device=previous.device, dtype=previous.dtype
    )
    patches = torch.as_tensor(
        current_patch_tokens, device=previous.device, dtype=previous.dtype
    )
    points = torch.as_tensor(candidate_px, device=previous.device)
    measured = torch.as_tensor(measurement_logits, device=previous.device)
    support = torch.as_tensor(candidate_valid, device=previous.device)
    valid_view = torch.as_tensor(view_valid, device=previous.device)
    if previous.ndim != 4 or previous.shape[1:] != (VIEWS, TIPS, TOKEN_DIM):
        raise ValueError("previous_tokens must be [B,2,2,192]")
    batch = len(previous)
    if prior.shape != previous.shape:
        raise ValueError("prior_tokens must match previous_tokens")
    if patches.shape != (batch, VIEWS, PATCHES, TOKEN_DIM):
        raise ValueError("current_patch_tokens must be [B,2,256,192]")
    if points.ndim != 5 or points.shape[:3] != (batch, VIEWS, TIPS):
        raise ValueError("candidate_px must be [B,2,2,M,2]")
    candidates = points.shape[3]
    if candidates < 1 or points.shape[4:] != (2,):
        raise ValueError("candidate_px must contain at least one xy candidate")
    expected = (batch, VIEWS, TIPS, candidates)
    if measured.shape != expected or support.shape != expected:
        raise ValueError("measurement logits/valid must be [B,2,2,M]")
    if valid_view.shape != (batch, VIEWS):
        raise ValueError("view_valid must be [B,2]")
    if support.dtype != torch.bool or valid_view.dtype != torch.bool:
        raise TypeError("candidate_valid and view_valid must be boolean")
    if beta != FEEDBACK_BETA or temperature != PRIOR_TEMPERATURE:
        raise ValueError("feedback beta=0.25 and temperature=0.10 are frozen")
    floating = (previous, prior, patches, points, measured)
    if not all(value.is_floating_point() for value in floating):
        raise TypeError("tracker feedback tensors must be floating point")
    if not bool(
        torch.isfinite(previous).all()
        and torch.isfinite(prior).all()
        and torch.isfinite(patches).all()
        and torch.isfinite(points).all()
        and torch.isfinite(measured[support]).all()
    ):
        raise ValueError("valid tracker feedback values must be finite")
    in_frame = (
        (points[..., 0] >= 0.0)
        & (points[..., 0] <= 223.0)
        & (points[..., 1] >= 0.0)
        & (points[..., 1] <= 223.0)
    )
    support = support & in_frame & valid_view[:, :, None, None]
    measurement_missing = valid_view & ~support.any(dim=-1).all(dim=-1)
    effective_view_valid = valid_view & ~measurement_missing
    support = support & effective_view_valid[:, :, None, None]

    feature = patches.reshape(batch * VIEWS, GRID, GRID, TOKEN_DIM)
    feature = feature.permute(0, 3, 1, 2)
    sampled = sample_feature_map_at_pixels(
        feature.float(),
        points.reshape(batch * VIEWS, TIPS * candidates, 2).float(),
        image_size=(224, 224),
    ).reshape(batch, VIEWS, TIPS, candidates, TOKEN_DIM)
    sampled = F.normalize(sampled, dim=-1, eps=1.0e-8)
    normalized_previous = F.normalize(previous.float(), dim=-1, eps=1.0e-8)
    normalized_prior = F.normalize(prior.detach().float(), dim=-1, eps=1.0e-8)
    prior_similarity = (sampled * normalized_prior[..., None, :]).sum(dim=-1)
    previous_similarity = (
        sampled * normalized_previous[..., None, :]
    ).sum(dim=-1)
    innovation = (prior_similarity - previous_similarity) / float(temperature)
    innovation = innovation.clamp(
        -MAX_NORMALIZED_INNOVATION, MAX_NORMALIZED_INNOVATION
    )
    posterior_logits = measured.float() + float(beta) * innovation
    posterior_logits = posterior_logits.masked_fill(~support, float("-inf"))
    measurement_only_logits = measured.float().masked_fill(
        ~support, float("-inf")
    )
    measurement_selected = _select_tracker_candidates(
        logits=measurement_only_logits,
        support=support,
        view_valid=effective_view_valid,
    )
    selected = _select_tracker_candidates(
        logits=posterior_logits,
        support=support,
        view_valid=effective_view_valid,
    )
    gather_px = selected[..., None, None].expand(-1, -1, -1, 1, 2)
    posterior_px = points.gather(3, gather_px).squeeze(3).float()
    gather_token = selected[..., None, None].expand(
        -1, -1, -1, 1, TOKEN_DIM
    )
    posterior_token = sampled.gather(3, gather_token).squeeze(3)
    posterior_px = torch.where(
        effective_view_valid[..., None, None],
        posterior_px,
        torch.zeros_like(posterior_px),
    )
    posterior_token = torch.where(
        effective_view_valid[..., None, None],
        posterior_token,
        torch.zeros_like(posterior_token),
    )
    tip_separation = torch.linalg.vector_norm(
        posterior_px[:, :, 0] - posterior_px[:, :, 1], dim=-1
    )
    tip_pair_degenerate = effective_view_valid & (
        tip_separation < TIP_DEGENERATE_RADIUS_PX
    )
    return TrackerCandidateFeedbackOutput(
        posterior_logits=posterior_logits,
        candidate_support=support,
        prior_similarity=prior_similarity,
        measurement_candidate_index=measurement_selected,
        selected_candidate_index=selected,
        prior_changed_selection=(selected != measurement_selected)
        & effective_view_valid[..., None],
        posterior_tip_px=posterior_px,
        posterior_center_px=posterior_px.mean(dim=2),
        posterior_tokens=posterior_token,
        tip_pair_degenerate=tip_pair_degenerate,
        view_valid=effective_view_valid,
        measurement_missing=measurement_missing,
    )


class MultiRatePartTokenLeWM(nn.Module):
    """Fresh control-rate transition over a frozen visual measurement.

    The action and CAD towers meet the visual state only in token space.  The
    output layer is zero initialized, so the complete system starts exactly at
    measurement-only.  Subtracting the same transition evaluated with a zero
    control makes hold/zero displacement an exact structural null even after
    training.
    """

    def __init__(
        self,
        *,
        joint_rms: torch.Tensor | None = None,
        cad_rms: torch.Tensor | None = None,
        support_size: int = SUPPORT_SIZE,
    ) -> None:
        super().__init__()
        if not 2 <= int(support_size) < PATCHES:
            raise ValueError("support_size must lie in [2,255]")
        self.support_size = int(support_size)
        joint_scale = torch.ones(6) if joint_rms is None else torch.as_tensor(joint_rms)
        cad_scale = torch.ones(6) if cad_rms is None else torch.as_tensor(cad_rms)
        if joint_scale.shape != (6,) or cad_scale.shape != (6,):
            raise ValueError("joint_rms/cad_rms must both be [6]")
        if not bool(
            torch.isfinite(joint_scale).all()
            and torch.isfinite(cad_scale).all()
            and (joint_scale > 0).all()
            and (cad_scale > 0).all()
        ):
            raise ValueError("signal RMS values must be finite and positive")
        self.register_buffer("joint_rms", joint_scale.float().clone())
        self.register_buffer("cad_rms", cad_scale.float().clone())

        self.joint_encoder = nn.Sequential(
            nn.Linear(STEPS_PER_HORIZON * 6, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.cad_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.previous_norm = nn.LayerNorm(TOKEN_DIM)
        self.transition_norm = nn.LayerNorm(2 * TOKEN_DIM)
        self.transition_hidden = nn.Linear(2 * TOKEN_DIM, TOKEN_DIM)
        self.transition_output = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        nn.init.zeros_(self.transition_output.weight)
        nn.init.zeros_(self.transition_output.bias)
        if self.parameter_count > PARAMETER_LIMIT:
            raise RuntimeError("multi-rate PartTokenLeWM exceeds parameter limit")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _control_features(
        self,
        joint_increments: torch.Tensor,
        cad_tip_increments: torch.Tensor,
    ) -> torch.Tensor:
        joint = joint_increments.float() / self.joint_rms[None, None, None]
        cad = unordered_query_cad_signal(cad_tip_increments).float()
        cad = cad / self.cad_rms[None, None]
        joint_flat = joint.flatten(-2)
        # Encoder biases must not turn a physical hold into a learned motion
        # signal.  Referencing each tower to its own zero input makes the
        # multimodal origin exact without adding a regularizer.
        joint_feature = self.joint_encoder(joint_flat)
        joint_feature = joint_feature - self.joint_encoder(
            torch.zeros_like(joint_flat)
        )
        cad_feature = self.cad_encoder(cad)
        cad_feature = cad_feature - self.cad_encoder(torch.zeros_like(cad))
        joint_feature = joint_feature[:, :, None].expand(-1, -1, TIPS, -1)
        cad_feature = cad_feature[:, :, None].expand(-1, -1, TIPS, -1)
        return torch.cat((joint_feature, cad_feature), dim=-1)

    def _residual(
        self,
        previous: torch.Tensor,
        control: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_previous = self.previous_norm(previous)
        expanded_control = control[:, None].expand(-1, VIEWS, -1, -1)
        hidden = F.gelu(
            self.transition_hidden(
                self.transition_norm(
                    torch.cat((normalized_previous, expanded_control), dim=-1)
                )
            )
        )
        zero_hidden = F.gelu(
            self.transition_hidden(
                self.transition_norm(
                    torch.cat(
                        (normalized_previous, torch.zeros_like(expanded_control)),
                        dim=-1,
                    )
                )
            )
        )
        raw = self.transition_output(hidden) - self.transition_output(zero_hidden)
        residual = PRIOR_RESIDUAL_SCALE * torch.tanh(raw)
        return residual, hidden

    def predict_token_prior(
        self,
        *,
        previous_tokens: torch.Tensor,
        joint_increment_chunk: torch.Tensor,
        cad_tip_displacement: torch.Tensor,
    ) -> PartTokenPriorOutput:
        """Predict one token prior without reading pixels or coordinates."""

        previous = torch.as_tensor(previous_tokens)
        joint = torch.as_tensor(
            joint_increment_chunk, device=previous.device, dtype=previous.dtype
        )
        cad = torch.as_tensor(
            cad_tip_displacement, device=previous.device, dtype=previous.dtype
        )
        if previous.ndim != 4 or previous.shape[1:] != (
            VIEWS,
            TIPS,
            TOKEN_DIM,
        ):
            raise ValueError("previous_tokens must be [B,2,2,192]")
        batch = len(previous)
        if joint.shape != (batch, STEPS_PER_HORIZON, 6):
            raise ValueError("joint_increment_chunk must be [B,5,6]")
        if cad.shape != (batch, TIPS, 3):
            raise ValueError("cad_tip_displacement must be [B,2,3]")
        values = (previous, joint, cad)
        if not all(value.is_floating_point() for value in values):
            raise TypeError("token-transition inputs must be floating point")
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("token-transition inputs must be finite")

        expanded_joint = joint[:, None].expand(-1, HORIZONS, -1, -1)
        expanded_cad = cad[:, None].expand(-1, HORIZONS, -1, -1)
        control = self._control_features(expanded_joint, expanded_cad)[:, 0]
        normalized = F.normalize(previous.float(), dim=-1, eps=1.0e-8)
        residual, hidden = self._residual(normalized, control)
        candidate = F.normalize(normalized + residual, dim=-1, eps=1.0e-8)
        physical_hold = (joint == 0).all(dim=(1, 2)) & (cad == 0).all(dim=(1, 2))
        prior = torch.where(
            physical_hold[:, None, None, None], normalized, candidate
        )
        return PartTokenPriorOutput(
            prior_tokens=prior,
            dynamics_latent=hidden.mean(dim=(1, 2)),
        )

    def encode_control_chunk(
        self,
        *,
        joint_increment_chunk: torch.Tensor,
        cad_tip_displacement: torch.Tensor,
    ) -> torch.Tensor:
        """Expose the frozen 192-D action/CAD carrier for a planning head.

        The seed-13161 endpoint already learned these two signal towers.  V14
        reuses their representation instead of silently creating a second
        action encoder.  Both generic visual query slots receive the same
        unordered physical control, so the duplicated tip axis is removed.
        """

        joint = torch.as_tensor(joint_increment_chunk)
        cad = torch.as_tensor(
            cad_tip_displacement, device=joint.device, dtype=joint.dtype
        )
        if joint.ndim != 3 or joint.shape[1:] != (STEPS_PER_HORIZON, 6):
            raise ValueError("joint_increment_chunk must be [B,5,6]")
        if cad.shape != (len(joint), TIPS, 3):
            raise ValueError("cad_tip_displacement must be [B,2,3]")
        if not joint.is_floating_point() or not cad.is_floating_point():
            raise TypeError("control chunk inputs must be floating point")
        if not bool(torch.isfinite(joint).all() and torch.isfinite(cad).all()):
            raise ValueError("control chunk inputs must be finite")
        expanded_joint = joint[:, None].expand(-1, HORIZONS, -1, -1)
        expanded_cad = cad[:, None].expand(-1, HORIZONS, -1, -1)
        encoded = self._control_features(expanded_joint, expanded_cad)[:, 0]
        return encoded[:, 0]

    @staticmethod
    def _validate(
        *,
        previous_tokens: torch.Tensor,
        future_patch_tokens: torch.Tensor,
        measurement_logits: torch.Tensor,
        future_view_valid: torch.Tensor,
        joint_increments: torch.Tensor,
        cad_tip_increments: torch.Tensor,
    ) -> int:
        if previous_tokens.ndim != 4 or previous_tokens.shape[1:] != (
            VIEWS,
            TIPS,
            TOKEN_DIM,
        ):
            raise ValueError("previous_tokens must be [B,2,2,192]")
        batch = len(previous_tokens)
        if future_patch_tokens.shape != (
            batch,
            HORIZONS,
            VIEWS,
            PATCHES,
            TOKEN_DIM,
        ):
            raise ValueError("future_patch_tokens must be [B,4,2,256,192]")
        if measurement_logits.shape != (
            batch,
            HORIZONS,
            VIEWS,
            TIPS,
            GRID,
            GRID,
        ):
            raise ValueError("measurement_logits must be [B,4,2,2,16,16]")
        if future_view_valid.shape != (batch, HORIZONS, VIEWS):
            raise ValueError("future_view_valid must be [B,4,2]")
        if future_view_valid.dtype != torch.bool:
            raise TypeError("future_view_valid must be boolean")
        if joint_increments.shape != (
            batch,
            HORIZONS,
            STEPS_PER_HORIZON,
            6,
        ):
            raise ValueError("joint_increments must be [B,4,5,6]")
        if cad_tip_increments.shape != (batch, HORIZONS, TIPS, 3):
            raise ValueError("cad_tip_increments must be [B,4,2,3]")
        floating = (
            previous_tokens,
            future_patch_tokens,
            joint_increments,
            cad_tip_increments,
        )
        if not all(value.is_floating_point() for value in floating):
            raise TypeError("multi-rate model inputs must be floating point")
        if not all(bool(torch.isfinite(value).all()) for value in floating):
            raise ValueError("multi-rate model inputs must be finite")
        measured = measurement_logits.flatten(-2)
        finite = torch.isfinite(measured) | ~future_view_valid[..., None, None]
        if not bool(finite.all()):
            raise ValueError("valid measurement logits must be finite")
        devices = {value.device for value in (*floating, future_view_valid)}
        if len(devices) != 1:
            raise ValueError("multi-rate model inputs must share one device")
        return batch

    def forward(
        self,
        *,
        previous_tokens: torch.Tensor,
        future_patch_tokens: torch.Tensor,
        measurement_logits: torch.Tensor,
        future_view_valid: torch.Tensor,
        joint_increments: torch.Tensor,
        cad_tip_increments: torch.Tensor,
        recursive_writeback: bool = True,
    ) -> MultiRateFeedbackOutput:
        batch = self._validate(
            previous_tokens=previous_tokens,
            future_patch_tokens=future_patch_tokens,
            measurement_logits=measurement_logits,
            future_view_valid=future_view_valid,
            joint_increments=joint_increments,
            cad_tip_increments=cad_tip_increments,
        )
        control = self._control_features(joint_increments, cad_tip_increments)
        previous = F.normalize(previous_tokens.float(), dim=-1, eps=1.0e-8)
        measured = measurement_logits.detach().float().flatten(-2)
        patches = future_patch_tokens.float()

        posterior_logits: list[torch.Tensor] = []
        supports: list[torch.Tensor] = []
        priors: list[torch.Tensor] = []
        posterior_tokens: list[torch.Tensor] = []
        recurrent_tokens: list[torch.Tensor] = []
        dynamics: list[torch.Tensor] = []
        selected_indices: list[torch.Tensor] = []
        tip_pixels: list[torch.Tensor] = []
        center_pixels: list[torch.Tensor] = []
        tip_degenerate: list[torch.Tensor] = []

        for horizon in range(HORIZONS):
            residual, hidden = self._residual(previous, control[:, horizon])
            candidate_prior = F.normalize(
                previous + residual, dim=-1, eps=1.0e-8
            )
            # Branch on the physical input, not on the predicted residual.
            # The output layer starts at zero; branching on residual==0 would
            # therefore disconnect the factual CE gradient at initialization.
            physical_hold = (
                (joint_increments[:, horizon] == 0).all(dim=(1, 2))
                & (cad_tip_increments[:, horizon] == 0).all(dim=(1, 2))
            )
            prior = torch.where(
                physical_hold[:, None, None, None],
                previous,
                candidate_prior,
            )
            current = patches[:, horizon]
            normalized_patch = F.normalize(current, dim=-1, eps=1.0e-8)
            prior_similarity = torch.einsum(
                "bvtd,bvpd->bvtp", prior, normalized_patch
            )
            previous_similarity = torch.einsum(
                "bvtd,bvpd->bvtp", previous, normalized_patch
            )
            innovation = (prior_similarity - previous_similarity) / PRIOR_TEMPERATURE

            valid = future_view_valid[:, horizon]
            measured_h = measured[:, horizon]
            support_index = measured_h.topk(self.support_size, dim=-1).indices
            support = torch.zeros_like(measured_h, dtype=torch.bool)
            support.scatter_(-1, support_index, True)
            support &= valid[:, :, None, None]
            combined = measured_h + FEEDBACK_BETA * innovation
            combined = combined.masked_fill(~support, float("-inf"))

            safe = torch.where(
                valid[:, :, None, None],
                combined,
                torch.zeros_like(combined),
            )
            probability = torch.softmax(safe, dim=-1) * support.to(safe.dtype)
            posterior = torch.einsum("bvtp,bvpd->bvtd", probability, current)
            posterior = posterior * valid[:, :, None, None]
            candidate_recurrent = F.normalize(
                (1.0 - WRITEBACK_GAIN) * previous
                + WRITEBACK_GAIN * posterior,
                dim=-1,
                eps=1.0e-8,
            )
            if recursive_writeback:
                previous = torch.where(
                    valid[:, :, None, None],
                    candidate_recurrent,
                    previous,
                )

            index = _tip_indices(combined, valid)
            px = _patch_centres(index).to(current.dtype)
            px = px * valid[:, :, None, None]
            tip_separation = torch.linalg.vector_norm(
                px[:, :, 0] - px[:, :, 1], dim=-1
            )

            posterior_logits.append(
                combined.reshape(batch, VIEWS, TIPS, GRID, GRID)
            )
            supports.append(support.reshape(batch, VIEWS, TIPS, GRID, GRID))
            priors.append(prior)
            posterior_tokens.append(posterior)
            recurrent_tokens.append(previous)
            dynamics.append(hidden.mean(dim=(1, 2)))
            selected_indices.append(index)
            tip_pixels.append(px)
            center_pixels.append(px.mean(dim=2))
            tip_degenerate.append(
                valid & (tip_separation < TIP_DEGENERATE_RADIUS_PX)
            )

        return MultiRateFeedbackOutput(
            measurement_logits=measurement_logits.detach(),
            posterior_logits=torch.stack(posterior_logits, dim=1),
            topk_support=torch.stack(supports, dim=1),
            prior_tokens=torch.stack(priors, dim=1),
            posterior_tokens=torch.stack(posterior_tokens, dim=1),
            recurrent_tokens=torch.stack(recurrent_tokens, dim=1),
            dynamics_latent=torch.stack(dynamics, dim=1),
            posterior_tip_patch_index=torch.stack(selected_indices, dim=1),
            posterior_tip_px=torch.stack(tip_pixels, dim=1),
            posterior_center_px=torch.stack(center_pixels, dim=1),
            tip_pair_degenerate=torch.stack(tip_degenerate, dim=1),
            view_valid=future_view_valid,
        )


def rollout_tracker_candidate_feedback(
    *,
    lewm: MultiRatePartTokenLeWM,
    initial_tokens: torch.Tensor,
    current_patch_tokens: torch.Tensor,
    candidate_px: torch.Tensor,
    measurement_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    view_valid: torch.Tensor,
    full_joint_increments: torch.Tensor,
    full_cad_tip_increments: torch.Tensor,
) -> TrackerCandidateRolloutOutput:
    """Roll out H2--H4 with unambiguous control alignment.

    The current-image tensors contain exactly H2, H3 and H4.  Control tensors
    contain the full H1--H4 sequence; this function intentionally ignores H1
    and consumes indices 1, 2 and 3 for H1->H2, H2->H3 and H3->H4.  Callers
    cannot silently pass an already sliced but temporally ambiguous sequence.
    """

    initial = torch.as_tensor(initial_tokens)
    if initial.ndim != 4 or initial.shape[1:] != (VIEWS, TIPS, TOKEN_DIM):
        raise ValueError("initial_tokens must be [B,2,2,192]")
    batch = len(initial)
    if current_patch_tokens.ndim != 5 or current_patch_tokens.shape[0] != batch:
        raise ValueError("current_patch_tokens must be [B,S,2,256,192]")
    steps = current_patch_tokens.shape[1]
    if steps != HORIZONS - 1 or current_patch_tokens.shape[2:] != (
        VIEWS,
        PATCHES,
        TOKEN_DIM,
    ):
        raise ValueError("tracker feedback requires exactly H2--H4")
    if candidate_px.ndim != 6 or candidate_px.shape[:4] != (
        batch,
        steps,
        VIEWS,
        TIPS,
    ):
        raise ValueError("candidate_px must be [B,S,2,2,M,2]")
    candidates = candidate_px.shape[4]
    if candidates < 1 or candidate_px.shape[5] != 2:
        raise ValueError("candidate_px must contain xy candidates")
    candidate_shape = (batch, steps, VIEWS, TIPS, candidates)
    if measurement_logits.shape != candidate_shape:
        raise ValueError("measurement_logits must be [B,S,2,2,M]")
    if candidate_valid.shape != candidate_shape:
        raise ValueError("candidate_valid must be [B,S,2,2,M]")
    if view_valid.shape != (batch, steps, VIEWS):
        raise ValueError("view_valid must be [B,S,2]")
    if full_joint_increments.shape != (
        batch,
        HORIZONS,
        STEPS_PER_HORIZON,
        6,
    ):
        raise ValueError("full_joint_increments must be [B,4,5,6]")
    if full_cad_tip_increments.shape != (batch, HORIZONS, TIPS, 3):
        raise ValueError("full_cad_tip_increments must be [B,4,2,3]")

    previous = F.normalize(initial.float(), dim=-1, eps=1.0e-8)
    priors = []
    posteriors = []
    logits = []
    supports = []
    measurement_selected = []
    selected = []
    changed_selection = []
    tip_px = []
    center_px = []
    dynamics = []
    tip_degenerate = []
    effective_views = []
    missing_measurements = []
    for step in range(steps):
        control_horizon = step + 1
        prior = lewm.predict_token_prior(
            previous_tokens=previous,
            joint_increment_chunk=full_joint_increments[:, control_horizon],
            cad_tip_displacement=full_cad_tip_increments[:, control_horizon],
        )
        posterior = tracker_candidate_feedback(
            previous_tokens=previous,
            prior_tokens=prior.prior_tokens,
            current_patch_tokens=current_patch_tokens[:, step],
            candidate_px=candidate_px[:, step],
            measurement_logits=measurement_logits[:, step],
            candidate_valid=candidate_valid[:, step],
            view_valid=view_valid[:, step],
        )
        previous = torch.where(
            posterior.view_valid[:, :, None, None],
            posterior.posterior_tokens,
            previous,
        )
        priors.append(prior.prior_tokens)
        posteriors.append(posterior.posterior_tokens)
        logits.append(posterior.posterior_logits)
        supports.append(posterior.candidate_support)
        measurement_selected.append(posterior.measurement_candidate_index)
        selected.append(posterior.selected_candidate_index)
        changed_selection.append(posterior.prior_changed_selection)
        tip_px.append(posterior.posterior_tip_px)
        center_px.append(posterior.posterior_center_px)
        dynamics.append(prior.dynamics_latent)
        tip_degenerate.append(posterior.tip_pair_degenerate)
        effective_views.append(posterior.view_valid)
        missing_measurements.append(posterior.measurement_missing)
    return TrackerCandidateRolloutOutput(
        prior_tokens=torch.stack(priors, dim=1),
        posterior_tokens=torch.stack(posteriors, dim=1),
        posterior_logits=torch.stack(logits, dim=1),
        candidate_support=torch.stack(supports, dim=1),
        measurement_candidate_index=torch.stack(measurement_selected, dim=1),
        selected_candidate_index=torch.stack(selected, dim=1),
        prior_changed_selection=torch.stack(changed_selection, dim=1),
        posterior_tip_px=torch.stack(tip_px, dim=1),
        posterior_center_px=torch.stack(center_px, dim=1),
        dynamics_latent=torch.stack(dynamics, dim=1),
        tip_pair_degenerate=torch.stack(tip_degenerate, dim=1),
        view_valid=torch.stack(effective_views, dim=1),
        measurement_missing=torch.stack(missing_measurements, dim=1),
    )


def masked_spatial_cross_entropy(
    logits: torch.Tensor,
    target_patch_index: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """Mean recursive CE per row without evaluating invalid ``-inf`` targets."""

    if logits.ndim != 6 or logits.shape[1:] != (
        HORIZONS,
        VIEWS,
        TIPS,
        GRID,
        GRID,
    ):
        raise ValueError("logits must be [B,4,2,2,16,16]")
    target = torch.as_tensor(target_patch_index, device=logits.device).long()
    valid = torch.as_tensor(target_valid, device=logits.device).bool()
    if target.shape != logits.shape[:4] or valid.shape != target.shape:
        raise ValueError("target and target_valid must be [B,4,2,2]")
    if bool(((target[valid] < 0) | (target[valid] >= PATCHES)).any()):
        raise ValueError("valid target lies outside the patch grid")
    flat = logits.flatten(-2)
    selected = flat.gather(-1, target.clamp(0, PATCHES - 1)[..., None])[..., 0]
    if not bool(torch.isfinite(selected[valid]).all()):
        raise ValueError("valid target lies outside current measurement support")
    count = valid.sum(dim=(1, 2, 3))
    if bool((count == 0).any()):
        raise ValueError("each row needs at least one valid target")
    error = torch.zeros_like(target, dtype=flat.dtype)
    error[valid] = F.cross_entropy(flat[valid], target[valid], reduction="none")
    return error.sum(dim=(1, 2, 3)) / count


__all__ = [
    "ACTION_STEPS",
    "COMMAND_ENDPOINTS",
    "FEEDBACK_BETA",
    "HORIZONS",
    "MAX_NORMALIZED_INNOVATION",
    "MultiRateFeedbackOutput",
    "MultiRatePartTokenLeWM",
    "PartTokenPriorOutput",
    "PARAMETER_LIMIT",
    "SUPPORT_SIZE",
    "TIP_DEGENERATE_RADIUS_PX",
    "TrackerCandidateFeedbackOutput",
    "TrackerCandidateRolloutOutput",
    "causal_action_increments",
    "causal_cad_tip_increments",
    "masked_spatial_cross_entropy",
    "rollout_tracker_candidate_feedback",
    "signal_rms",
    "tracker_candidate_feedback",
    "tracker_confidence_to_log_odds",
    "unordered_query_cad_signal",
]
