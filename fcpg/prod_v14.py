"""Prod-V14 dual-view action-conditioned persistent-query world model.

The frozen high-rate observer remains the authority for the *current* image.
This module is the low-rate planning forecast: given persistent current-image
query tokens/coordinates, the two anonymous view contexts and one candidate
command chunk, it predicts where those same queries will be observed after the
chunk.  The prediction is used only to rank candidate actions.  After execution
the forecast is discarded and FCPG-H measures the new images again.

No calibration, camera identifier, mask, optical flow or pixel residual is an
input.  View and query permutations are therefore structural symmetries rather
than data-augmentation conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


TOKEN_DIM = 192
VIEWS = 2
TIP_COUNT = 2
IMAGE_SIZE = 224
DISPLACEMENT_RADIUS_PX = IMAGE_SIZE - 1
DISPLACEMENT_BINS = 2 * DISPLACEMENT_RADIUS_PX + 1
STEPS_PER_CHUNK = 5
CONTROL_DIM = 192
MEASUREMENT_PRIOR_GAIN = 0.25
RESPONSE_DIM = 64
RESPONSE_RANGE_PX = 64.0
RESPONSE_SIGMA_PX = 6.0
HISTORY_DECAY = 0.30
ACTION_EXCITATION_PX = 6.0
CHECKPOINT_FORMAT = "so101-prod-v14-excitation-gated-arx-lewm-v4"
ENGINEERING_BEST_STEP = 2_000
TRAINING_SEED = 14_101


@dataclass(frozen=True)
class DualViewQueryForecast:
    """One candidate chunk's future distribution for persistent queries."""

    delta_x_logits: torch.Tensor
    delta_y_logits: torch.Tensor
    expected_delta_px: torch.Tensor
    expected_px: torch.Tensor
    query_latent: torch.Tensor
    view_valid: torch.Tensor


@dataclass(frozen=True)
class DualViewControlDecision:
    """Joint observation-space score over an action candidate bank."""

    energy: torch.Tensor
    selected_index: torch.Tensor
    per_view_distance_px: torch.Tensor
    view_valid: torch.Tensor


@dataclass(frozen=True)
class V14MeasurementUpdate:
    """Measurement-dominant correction on next-frame visual candidates."""

    posterior_logits: torch.Tensor
    forecast_candidate_score: torch.Tensor
    selected_candidate_index: torch.Tensor
    posterior_px: torch.Tensor
    measurement_missing: torch.Tensor


def _validate_control_features(control: torch.Tensor, batch: int) -> None:
    if control.shape != (batch, CONTROL_DIM):
        raise ValueError("control_features must be [B,192]")
    if not control.is_floating_point() or not bool(torch.isfinite(control).all()):
        raise ValueError("control_features must be finite floating point")


class V14DualViewQueryLeWM(nn.Module):
    """A small permutation-equivariant forecast head on a frozen LeWM control.

    ``control_features`` come from the fixed seed-13161 PartTokenLeWM action/CAD
    tower.  The head adds the information missing from that endpoint: current
    query coordinates, each view's global temporal context, and an explicit
    cross-view interaction.  It emits a categorical one-pixel displacement
    distribution rather than a calibrated coordinate residual.
    """

    def __init__(self, *, token_dim: int = TOKEN_DIM) -> None:
        super().__init__()
        if int(token_dim) != TOKEN_DIM:
            raise ValueError("Prod-V14 dual-view forecast fixes token_dim=192")
        self.token_dim = int(token_dim)
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.position_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, self.token_dim),
        )
        self.view_context = nn.Sequential(
            nn.LayerNorm(2 * self.token_dim),
            nn.Linear(2 * self.token_dim, self.token_dim),
            nn.GELU(),
        )
        self.visual_fusion = nn.Sequential(
            nn.LayerNorm(3 * self.token_dim),
            nn.Linear(3 * self.token_dim, self.token_dim),
            nn.GELU(),
        )
        self.cross_view = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=4,
            dim_feedforward=2 * self.token_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.control_fusion = nn.Sequential(
            nn.LayerNorm(self.token_dim + CONTROL_DIM),
            nn.Linear(self.token_dim + CONTROL_DIM, self.token_dim),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(self.token_dim)
        self.delta_x = nn.Linear(self.token_dim, DISPLACEMENT_BINS)
        self.delta_y = nn.Linear(self.token_dim, DISPLACEMENT_BINS)

        # The new attachment starts as a narrow copy forecast, not a random
        # 224-pixel jump.  All learned weights are zero only at the last layer;
        # CE immediately trains those layers and then the visual/control path.
        nn.init.zeros_(self.delta_x.weight)
        nn.init.zeros_(self.delta_y.weight)
        support = torch.arange(
            -DISPLACEMENT_RADIUS_PX,
            DISPLACEMENT_RADIUS_PX + 1,
            dtype=torch.float32,
        )
        # Keep the mean at exact copy while leaving nonzero gradient support
        # across the complete image. A 2px Gaussian would put a 100px target
        # thousands of logits below zero and needlessly turn initialization
        # into an optimization barrier.
        copy_bias = -0.5 * (support / 32.0).square()
        with torch.no_grad():
            self.delta_x.bias.copy_(copy_bias)
            self.delta_y.bias.copy_(copy_bias)
        self.register_buffer("displacement_support_px", support)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        *,
        global_history: torch.Tensor,
        query_tokens: torch.Tensor,
        query_px: torch.Tensor,
        control_features: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> DualViewQueryForecast:
        tokens = torch.as_tensor(query_tokens)
        batch = len(tokens)
        if tokens.ndim != 4 or tokens.shape[1] != VIEWS or tokens.shape[-1] != TOKEN_DIM:
            raise ValueError("query_tokens must be [B,2,Q,192]")
        queries = int(tokens.shape[2])
        if queries < 1:
            raise ValueError("dual-view forecast needs at least one query")
        history = torch.as_tensor(
            global_history, device=tokens.device, dtype=tokens.dtype
        )
        points = torch.as_tensor(query_px, device=tokens.device, dtype=tokens.dtype)
        valid = torch.as_tensor(view_valid, device=tokens.device)
        control = torch.as_tensor(
            control_features, device=tokens.device, dtype=tokens.dtype
        )
        if history.shape != (batch, 2, VIEWS, TOKEN_DIM):
            raise ValueError("global_history must be [B,2,2,192]")
        if points.shape != (batch, VIEWS, queries, 2):
            raise ValueError("query_px must be [B,2,Q,2]")
        if valid.shape != (batch, VIEWS) or valid.dtype != torch.bool:
            raise ValueError("view_valid must be bool [B,2]")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("each forecast sample needs at least one valid view")
        floating = (tokens, history, points)
        if not all(value.is_floating_point() for value in floating):
            raise TypeError("dual-view visual inputs must be floating point")
        if not all(bool(torch.isfinite(value).all()) for value in floating):
            raise ValueError("dual-view visual inputs must be finite")
        _validate_control_features(control, batch)

        normalized_px = 2.0 * points.float() / float(IMAGE_SIZE - 1) - 1.0
        position = self.position_encoder(normalized_px)
        temporal = torch.cat(
            (history[:, 1], history[:, 1] - history[:, 0]), dim=-1
        )
        view_context = self.view_context(temporal.float())
        view_expanded = view_context[:, :, None].expand(-1, -1, queries, -1)
        visual = self.visual_fusion(
            torch.cat(
                (self.query_norm(tokens.float()), position, view_expanded),
                dim=-1,
            )
        )

        flat = visual.reshape(batch, VIEWS * queries, TOKEN_DIM)
        padding = (~valid)[:, :, None].expand(-1, -1, queries).reshape(
            batch, VIEWS * queries
        )
        mixed = self.cross_view(flat, src_key_padding_mask=padding)
        mixed = mixed.reshape(batch, VIEWS, queries, TOKEN_DIM)
        expanded_control = control.float()[:, None, None].expand(
            -1, VIEWS, queries, -1
        )
        latent = mixed + self.control_fusion(
            torch.cat((mixed, expanded_control), dim=-1)
        )
        latent = self.output_norm(latent)
        x_logits = self.delta_x(latent)
        y_logits = self.delta_y(latent)
        valid_logits = valid[:, :, None, None]
        x_logits = torch.where(valid_logits, x_logits, torch.zeros_like(x_logits))
        y_logits = torch.where(valid_logits, y_logits, torch.zeros_like(y_logits))

        support = self.displacement_support_px.to(x_logits)
        expected_x = (x_logits.softmax(dim=-1) * support).sum(dim=-1)
        expected_y = (y_logits.softmax(dim=-1) * support).sum(dim=-1)
        delta = torch.stack((expected_x, expected_y), dim=-1)
        expected = (points.float() + delta).clamp(0.0, float(IMAGE_SIZE - 1))
        expected = expected * valid[:, :, None, None]
        return DualViewQueryForecast(
            delta_x_logits=x_logits,
            delta_y_logits=y_logits,
            expected_delta_px=delta,
            expected_px=expected,
            query_latent=latent,
            view_valid=valid,
        )


class V14SeparableResponseLeWM(nn.Module):
    """Low-order controlled response for the V14 planning forecast.

    This revision treats the two modalities as a signal/system pair. Visual
    features describe the current observation state; the zero-referenced
    action/CAD carrier is the input. Their only route to a displacement is a
    multiplicative response term. Consequently a physical hold is an exact
    copy forecast, while a shared, permutation-equivariant view state can
    still express different image responses to the same command.
    """

    def __init__(self, *, token_dim: int = TOKEN_DIM) -> None:
        super().__init__()
        if int(token_dim) != TOKEN_DIM:
            raise ValueError("Prod-V14 response model fixes token_dim=192")
        self.query_norm = nn.LayerNorm(TOKEN_DIM)
        self.query_projection = nn.Linear(TOKEN_DIM, RESPONSE_DIM)
        self.position_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, RESPONSE_DIM),
        )
        self.velocity_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, RESPONSE_DIM),
        )
        self.view_context = nn.Sequential(
            nn.LayerNorm(2 * TOKEN_DIM),
            nn.Linear(2 * TOKEN_DIM, RESPONSE_DIM),
            nn.GELU(),
        )
        self.local_norm = nn.LayerNorm(RESPONSE_DIM)
        self.cross_view_state = nn.Sequential(
            nn.LayerNorm(2 * RESPONSE_DIM),
            nn.Linear(2 * RESPONSE_DIM, RESPONSE_DIM),
            nn.GELU(),
        )
        self.control_norm = nn.LayerNorm(CONTROL_DIM)
        self.control_projection = nn.Linear(CONTROL_DIM, RESPONSE_DIM)
        self.response = nn.Sequential(
            nn.Linear(RESPONSE_DIM, RESPONSE_DIM),
            nn.GELU(),
            nn.Linear(RESPONSE_DIM, 2),
        )
        nn.init.zeros_(self.response[-1].weight)
        nn.init.zeros_(self.response[-1].bias)
        self.register_buffer(
            "displacement_support_px",
            torch.arange(
                -DISPLACEMENT_RADIUS_PX,
                DISPLACEMENT_RADIUS_PX + 1,
                dtype=torch.float32,
            ),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        *,
        global_history: torch.Tensor,
        query_tokens: torch.Tensor,
        query_px: torch.Tensor,
        previous_delta_px: torch.Tensor,
        control_features: torch.Tensor,
        view_valid: torch.Tensor,
        action_gate_active: torch.Tensor | None = None,
    ) -> DualViewQueryForecast:
        tokens = torch.as_tensor(query_tokens)
        batch = len(tokens)
        if tokens.ndim != 4 or tokens.shape[1] != VIEWS or tokens.shape[-1] != TOKEN_DIM:
            raise ValueError("query_tokens must be [B,2,Q,192]")
        queries = int(tokens.shape[2])
        if queries < 1:
            raise ValueError("dual-view response needs at least one query")
        history = torch.as_tensor(
            global_history, device=tokens.device, dtype=tokens.dtype
        )
        points = torch.as_tensor(query_px, device=tokens.device, dtype=tokens.dtype)
        velocity = torch.as_tensor(
            previous_delta_px, device=tokens.device, dtype=tokens.dtype
        )
        valid = torch.as_tensor(view_valid, device=tokens.device)
        control = torch.as_tensor(
            control_features, device=tokens.device, dtype=tokens.dtype
        )
        if history.shape != (batch, 2, VIEWS, TOKEN_DIM):
            raise ValueError("global_history must be [B,2,2,192]")
        if points.shape != (batch, VIEWS, queries, 2):
            raise ValueError("query_px must be [B,2,Q,2]")
        if velocity.shape != points.shape:
            raise ValueError("previous_delta_px must be [B,2,Q,2]")
        if valid.shape != (batch, VIEWS) or valid.dtype != torch.bool:
            raise ValueError("view_valid must be bool [B,2]")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("each response sample needs at least one valid view")
        if not all(
            value.is_floating_point() for value in (tokens, history, points, velocity)
        ):
            raise TypeError("dual-view visual inputs must be floating point")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (tokens, history, points, velocity)
        ):
            raise ValueError("dual-view visual inputs must be finite")
        _validate_control_features(control, batch)

        normalized_px = 2.0 * points.float() / float(IMAGE_SIZE - 1) - 1.0
        temporal = torch.cat(
            (history[:, 1], history[:, 1] - history[:, 0]), dim=-1
        )
        local = self.query_projection(self.query_norm(tokens.float()))
        local = local + self.position_encoder(normalized_px)
        local = local + self.velocity_encoder(velocity.float() / 32.0)
        local = local + self.view_context(temporal.float())[:, :, None]
        local = self.local_norm(local)

        view_mask = valid[:, :, None, None].to(local)
        paired = (local * view_mask).sum(dim=1, keepdim=True)
        paired = paired / view_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        paired = paired.expand(-1, VIEWS, -1, -1)
        state = self.cross_view_state(torch.cat((local, paired), dim=-1))

        normalized_control = self.control_norm(control.float())
        zero_control = self.control_norm(torch.zeros_like(control).float())
        drive = self.control_projection(normalized_control)
        drive = drive - self.control_projection(zero_control)
        driven_state = state * drive[:, None, None]
        raw = self.response(driven_state) - self.response(
            torch.zeros_like(driven_state)
        )
        controlled_delta = RESPONSE_RANGE_PX * torch.tanh(raw / RESPONSE_RANGE_PX)
        if action_gate_active is None:
            # Historical V14 contract: use lagged image motion as an
            # identifiability heuristic. Existing checkpoints and runtime keep
            # this exact behavior when no explicit physical gate is supplied.
            action_identifiable = (
                torch.linalg.vector_norm(velocity.float(), dim=-1, keepdim=True)
                >= ACTION_EXCITATION_PX
            )
        else:
            # A matched system-identification experiment may instead gate the
            # control innovation from the *current* known physical input. The
            # physical gate can be one scalar per command chunk, or its
            # parameter-free intersection with the existing per-query visual
            # observability mask. Neither form can generate a pixel direction
            # or camera-specific coordinate by itself.
            action_identifiable = torch.as_tensor(
                action_gate_active, device=tokens.device
            )
            if action_identifiable.dtype != torch.bool:
                raise ValueError("action_gate_active must be boolean")
            if action_identifiable.shape == (batch,):
                action_identifiable = action_identifiable[:, None, None, None]
            elif action_identifiable.shape == (batch, VIEWS, queries):
                action_identifiable = action_identifiable[..., None]
            else:
                raise ValueError(
                    "action_gate_active must be bool [B] or [B,2,Q]"
                )
        controlled_delta = controlled_delta * action_identifiable
        mean_delta = (HISTORY_DECAY * velocity.float() + controlled_delta).clamp(
            -float(DISPLACEMENT_RADIUS_PX), float(DISPLACEMENT_RADIUS_PX)
        )

        support = self.displacement_support_px.to(mean_delta)
        x_logits = -0.5 * (
            (support - mean_delta[..., 0, None]) / RESPONSE_SIGMA_PX
        ).square()
        y_logits = -0.5 * (
            (support - mean_delta[..., 1, None]) / RESPONSE_SIGMA_PX
        ).square()
        valid_logits = valid[:, :, None, None]
        x_logits = torch.where(valid_logits, x_logits, torch.zeros_like(x_logits))
        y_logits = torch.where(valid_logits, y_logits, torch.zeros_like(y_logits))
        expected = (points.float() + mean_delta).clamp(
            0.0, float(IMAGE_SIZE - 1)
        )
        expected = expected * valid[:, :, None, None]
        return DualViewQueryForecast(
            delta_x_logits=x_logits,
            delta_y_logits=y_logits,
            expected_delta_px=mean_delta,
            expected_px=expected,
            query_latent=state,
            view_valid=valid,
        )


def displacement_target_index(
    current_px: torch.Tensor, target_px: torch.Tensor
) -> torch.Tensor:
    current = torch.as_tensor(current_px).float()
    target = torch.as_tensor(target_px, device=current.device).float()
    if current.shape != target.shape or current.shape[-1] != 2:
        raise ValueError("current and target pixels must share [...,2]")
    if not bool(torch.isfinite(current).all() and torch.isfinite(target).all()):
        raise ValueError("displacement targets must be finite")
    delta = torch.round(target - current).long()
    if bool(
        ((delta < -DISPLACEMENT_RADIUS_PX) | (delta > DISPLACEMENT_RADIUS_PX)).any()
    ):
        raise ValueError("pixel displacement lies outside the image-supported range")
    return delta + DISPLACEMENT_RADIUS_PX


def masked_displacement_cross_entropy(
    forecast: DualViewQueryForecast,
    *,
    current_px: torch.Tensor,
    target_px: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """The sole spatial forecast objective, averaged over visible queries."""

    target = displacement_target_index(current_px, target_px)
    valid = torch.as_tensor(
        target_valid, device=forecast.delta_x_logits.device, dtype=torch.bool
    )
    expected_shape = forecast.delta_x_logits.shape[:-1]
    if target.shape != (*expected_shape, 2) or valid.shape != expected_shape:
        raise ValueError("forecast target/mask shape mismatch")
    valid = valid & forecast.view_valid[:, :, None]
    if not bool(valid.any()):
        raise ValueError("displacement CE requires at least one visible query")
    x = F.cross_entropy(
        forecast.delta_x_logits[valid], target[..., 0][valid], reduction="none"
    )
    y = F.cross_entropy(
        forecast.delta_y_logits[valid], target[..., 1][valid], reduction="none"
    )
    return (x + y).mean()


def dual_view_query_control_energy(
    tip_px: torch.Tensor,
    target_px: torch.Tensor,
    view_valid: torch.Tensor,
) -> DualViewControlDecision:
    """Select a candidate by one joint column+wrist observation-space energy.

    Inputs contain the *forecast* for every candidate.  There is no preferred
    camera and no geometric backprojection.  If both views are valid both must
    contribute; a single-view fallback is represented explicitly by the mask.
    """

    tips = torch.as_tensor(tip_px).float()
    target = torch.as_tensor(target_px, device=tips.device).float()
    valid = torch.as_tensor(view_valid, device=tips.device)
    if tips.ndim != 5 or tips.shape[2:] != (VIEWS, TIP_COUNT, 2):
        raise ValueError("tip_px must be [B,K,2,2,2]")
    batch, candidates = tips.shape[:2]
    if target.shape != (batch, candidates, VIEWS, 2):
        raise ValueError("target_px must be [B,K,2,2]")
    if valid.shape == (batch, VIEWS):
        valid = valid[:, None].expand(-1, candidates, -1)
    if valid.shape != (batch, candidates, VIEWS) or valid.dtype != torch.bool:
        raise ValueError("view_valid must be bool [B,2] or [B,K,2]")
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("each candidate needs at least one forecast view")
    if not bool(torch.isfinite(tips).all() and torch.isfinite(target).all()):
        raise ValueError("dual-view control points must be finite")
    center = tips.mean(dim=-2)
    distance = torch.linalg.vector_norm(center - target, dim=-1)
    normalized = distance / (2.0**0.5 * float(IMAGE_SIZE))
    energy = (normalized.square() * valid).sum(dim=-1) / valid.sum(dim=-1)
    selected = energy.argmin(dim=1)
    return DualViewControlDecision(
        energy=energy,
        selected_index=selected,
        per_view_distance_px=distance,
        view_valid=valid,
    )


def v14_measurement_update(
    *,
    forecast: DualViewQueryForecast,
    current_query_px: torch.Tensor,
    candidate_px: torch.Tensor,
    measurement_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> V14MeasurementUpdate:
    """Use the spatial LeWM only as a bounded current-candidate tie breaker.

    The LeWM forecast cannot create a candidate or emit the posterior point.
    Coordinates are always gathered from the next image's FCPG/tracker bank.
    A bad prior can alter each measurement score by at most ``0.25``.
    """

    points = torch.as_tensor(candidate_px).float()
    measured = torch.as_tensor(measurement_logits, device=points.device).float()
    valid = torch.as_tensor(candidate_valid, device=points.device)
    current = torch.as_tensor(current_query_px, device=points.device).float()
    if points.ndim != 5 or points.shape[1] != VIEWS or points.shape[-1] != 2:
        raise ValueError("candidate_px must be [B,2,Q,M,2]")
    batch, _, queries, candidates = points.shape[:4]
    if candidates < 1:
        raise ValueError("measurement update needs at least one candidate")
    if measured.shape != (batch, VIEWS, queries, candidates):
        raise ValueError("measurement_logits must be [B,2,Q,M]")
    if valid.shape != measured.shape or valid.dtype != torch.bool:
        raise ValueError("candidate_valid must be bool [B,2,Q,M]")
    if current.shape != (batch, VIEWS, queries, 2):
        raise ValueError("current_query_px must be [B,2,Q,2]")
    if forecast.delta_x_logits.shape[:3] != (batch, VIEWS, queries):
        raise ValueError("forecast/query candidate shape mismatch")
    if not bool(
        torch.isfinite(points).all()
        and torch.isfinite(measured).all()
        and torch.isfinite(current).all()
    ):
        raise ValueError("measurement update inputs must be finite")

    delta = torch.round(points - current[..., None, :]).long()
    delta = delta.clamp(
        -DISPLACEMENT_RADIUS_PX, DISPLACEMENT_RADIUS_PX
    ) + DISPLACEMENT_RADIUS_PX
    log_x = forecast.delta_x_logits.float().log_softmax(dim=-1)
    log_y = forecast.delta_y_logits.float().log_softmax(dim=-1)
    score = torch.gather(
        log_x[..., None, :].expand(-1, -1, -1, candidates, -1),
        -1,
        delta[..., 0, None],
    )[..., 0]
    score = score + torch.gather(
        log_y[..., None, :].expand(-1, -1, -1, candidates, -1),
        -1,
        delta[..., 1, None],
    )[..., 0]
    safe_score = score.masked_fill(~valid, 0.0)
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
    centered = safe_score - (safe_score * valid).sum(
        dim=-1, keepdim=True
    ) / count
    correction = MEASUREMENT_PRIOR_GAIN * torch.tanh(centered)
    posterior = (measured + correction).masked_fill(~valid, float("-inf"))
    missing = ~valid.any(dim=-1) | ~forecast.view_valid[:, :, None]
    safe = torch.where(missing[..., None], torch.zeros_like(posterior), posterior)
    selected = safe.argmax(dim=-1)
    gathered = torch.gather(
        points,
        3,
        selected[..., None, None].expand(-1, -1, -1, 1, 2),
    )[..., 0, :]
    posterior_px = torch.where(
        missing[..., None], current, gathered
    )
    return V14MeasurementUpdate(
        posterior_logits=posterior,
        forecast_candidate_score=score,
        selected_candidate_index=selected,
        posterior_px=posterior_px,
        measurement_missing=missing,
    )


__all__ = [
    "ACTION_EXCITATION_PX",
    "CHECKPOINT_FORMAT",
    "CONTROL_DIM",
    "DISPLACEMENT_BINS",
    "DISPLACEMENT_RADIUS_PX",
    "MEASUREMENT_PRIOR_GAIN",
    "DualViewControlDecision",
    "DualViewQueryForecast",
    "ENGINEERING_BEST_STEP",
    "TRAINING_SEED",
    "V14DualViewQueryLeWM",
    "V14SeparableResponseLeWM",
    "V14MeasurementUpdate",
    "dual_view_query_control_energy",
    "displacement_target_index",
    "masked_displacement_cross_entropy",
    "v14_measurement_update",
]
