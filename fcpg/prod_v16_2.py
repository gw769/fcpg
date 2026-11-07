"""Prod-V16.2 native-V12.4 visual measurement, recurrent LeWM, and actor.

This module deliberately separates three signal paths:

* the measurement path sees only frozen native-ViT candidates and their causal
  visual tubes;
* completed controls can score those candidates and drive the goal-free LeWM,
  but can never synthesize a pixel coordinate;
* the actor consumes detached grounded/world-model features and therefore
  cannot move the measurement or LeWM representations with imitation loss.

The two fingertip slots are an unordered set.  No camera calibration, camera
identity, optical flow, tracker, mask, manual query, or nominal pixel enters
the model.  Closed-jaw tips may share a patch or a sub-pixel location; no
collision exception exists in this implementation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from fcpg.offline_pipeline_manifest import sha256_file
from v10.general_task_space_dual_jaw import (
    GENERAL_TASK_SPACE_JOINT_ACTION_CONTRACT_FORMAT,
    general_task_space_joint_action_contract,
)


TOKEN_DIM = 192
STATE_DIM = 6
VIEWS = 2
TIPS = 2
HORIZONS = 4
ACTION_HORIZON = 20
RAW_STEPS_PER_CHUNK = 5
NATIVE_GRID = 16
NATIVE_PATCHES = NATIVE_GRID * NATIVE_GRID
IMAGE_SIZE = 224
PATCH_SIZE = 14
PATCH_CENTER = 6.5
PHASE_OFFSETS_XY = ((0, 0), (7, 0), (0, 7), (7, 7))
PHASE_COUNT = len(PHASE_OFFSETS_XY)
FULL_PHASE_CANDIDATES = PHASE_COUNT * NATIVE_PATCHES
SAVED_CANDIDATES = 64
PRIOR_SUPPORT_TOPK = 64
PRIOR_GAIN = 0.25
PRIOR_TEMPERATURE = 0.10
LEWM_RESIDUAL_GAIN = 0.25
SIGREG_WEIGHT = 0.02
V12_4_STEP = 20_000
V12_4_SHA256 = "aca718b0a7335fd3ed3afce4bf70548ede1e194b6f9f0463ba4de0a36379c339"
CHECKPOINT_FORMAT = "so101-prod-v16-2-native-vit-fcpg-lewm-actor-v1"
SIDECAR_FORMAT = "so101-prod-v16-2-native-vit-sidecar-v1"
METHOD_REVISION = "prod-v16-2-unordered-native-vit-measurement-v1"


def prod_v16_2_actor_action_contract() -> dict[str, torch.Tensor]:
    """Return the robot-wide contract for the new actor, not the V12 actor fit."""

    contract = general_task_space_joint_action_contract()
    if contract.get("format") != GENERAL_TASK_SPACE_JOINT_ACTION_CONTRACT_FORMAT:
        raise ValueError("general task-space action contract format drifted")
    return {
        "step_cap": torch.as_tensor(contract["step_cap"]).float(),
        "action_lower": torch.as_tensor(contract["lower"]).float(),
        "action_upper": torch.as_tensor(contract["upper"]).float(),
    }


def project_bounded_action_trajectory(
    current_state: torch.Tensor,
    desired_trajectory: torch.Tensor,
    *,
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> torch.Tensor:
    """Project absolute targets sequentially without snapping an overshot state.

    Archived measured positions can slightly exceed the canonical command box.
    In that case the only admissible behavior is to move toward the box by at
    most one step cap.  A projection must never jump directly to a limit or
    move farther outside it.
    """

    state = torch.as_tensor(current_state)
    desired = torch.as_tensor(
        desired_trajectory,
        device=state.device,
        dtype=state.dtype,
    )
    cap = torch.as_tensor(step_cap, device=state.device, dtype=state.dtype)
    lower = torch.as_tensor(action_lower, device=state.device, dtype=state.dtype)
    upper = torch.as_tensor(action_upper, device=state.device, dtype=state.dtype)
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError("current_state must be [B,6]")
    if desired.ndim != 3 or desired.shape[0] != len(state) or desired.shape[-1] != STATE_DIM:
        raise ValueError("desired_trajectory must be [B,K,6]")
    if any(value.shape != (STATE_DIM,) for value in (cap, lower, upper)):
        raise ValueError("action contract vectors must be [6]")
    if not all(bool(torch.isfinite(value).all()) for value in (state, desired, cap, lower, upper)):
        raise ValueError("action projection inputs must be finite")
    if bool((cap <= 0).any()) or bool((lower >= upper).any()):
        raise ValueError("action projection contract is invalid")
    current = state
    projected = []
    for step in range(desired.shape[1]):
        bounded = torch.maximum(torch.minimum(desired[:, step], upper), lower)
        delta = torch.clamp(bounded - current, min=-cap, max=cap)
        current = current + delta
        projected.append(current)
    return torch.stack(projected, dim=1)


def _assert_recovering_action_trajectory(
    trajectory: torch.Tensor,
    current_state: torch.Tensor,
    *,
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> None:
    previous = torch.cat((current_state[:, None], trajectory[:, :-1]), dim=1)
    if bool(((trajectory - previous).abs() > step_cap + 1.0e-6).any()):
        raise ValueError("actor baseline violates the joint step cap")
    previous_outside = torch.relu(action_lower - previous) + torch.relu(
        previous - action_upper
    )
    current_outside = torch.relu(action_lower - trajectory) + torch.relu(
        trajectory - action_upper
    )
    if bool((current_outside > previous_outside + 1.0e-6).any()):
        raise ValueError("actor trajectory moves farther outside joint limits")


@dataclass(frozen=True)
class FourPhaseTokens:
    """Frozen ViT support with original-image coordinates and validity."""

    tokens: torch.Tensor
    global_context: torch.Tensor
    coordinates_px: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class UnorderedMeasurement:
    """Two exchangeable fingertip measurements on current-image support."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    tokens: torch.Tensor
    tip_px: torch.Tensor
    center_px: torch.Tensor
    aperture_response: torch.Tensor
    identity_valid: torch.Tensor
    moving_slot: torch.Tensor


@dataclass(frozen=True)
class ProdV162Output:
    """Joint measurement, recurrent posterior, dynamics and actor output."""

    measurement_logits: torch.Tensor
    posterior_logits: torch.Tensor
    measurement_tokens: torch.Tensor
    posterior_tokens: torch.Tensor
    measurement_tip_px: torch.Tensor
    posterior_tip_px: torch.Tensor
    measurement_center_px: torch.Tensor
    posterior_center_px: torch.Tensor
    prior_tokens: torch.Tensor
    dynamics_latent: torch.Tensor
    actor_action: torch.Tensor
    actor_baseline: torch.Tensor
    actor_residual: torch.Tensor
    identity_valid: torch.Tensor
    moving_slot: torch.Tensor
    prior_support: torch.Tensor


def phase_shifted_crops(images: torch.Tensor) -> torch.Tensor:
    """Create the four registered 224px crops without changing ViT inputs.

    The image is extended only on its right and bottom boundary.  A patch
    centre in crop ``(dx, dy)`` maps back to original coordinate
    ``centre + (dx, dy)``.  Candidates outside the original 224px image are
    retained in the tensor but marked invalid by :func:`phase_candidate_grid`.
    """

    value = torch.as_tensor(images)
    if value.ndim != 4 or tuple(value.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("images must be [N,3,224,224]")
    padded = F.pad(value, (0, 7, 0, 7), mode="replicate")
    crops = [
        padded[:, :, dy : dy + IMAGE_SIZE, dx : dx + IMAGE_SIZE]
        for dx, dy in PHASE_OFFSETS_XY
    ]
    return torch.stack(crops, dim=1)


def phase_candidate_grid(
    *, device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return four-phase candidate coordinates and the in-image mask."""

    axis = torch.arange(NATIVE_GRID, device=device, dtype=dtype) * PATCH_SIZE
    axis = axis + PATCH_CENTER
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    native = torch.stack((x, y), dim=-1).reshape(NATIVE_PATCHES, 2)
    coordinates = []
    for dx, dy in PHASE_OFFSETS_XY:
        offset = torch.tensor((dx, dy), device=device, dtype=dtype)
        coordinates.append(native + offset)
    grid = torch.stack(coordinates, dim=0)
    valid = (
        (grid[..., 0] >= 0.0)
        & (grid[..., 0] <= IMAGE_SIZE - 1.0)
        & (grid[..., 1] >= 0.0)
        & (grid[..., 1] <= IMAGE_SIZE - 1.0)
    )
    return grid, valid


class FrozenV12FourPhaseViT(nn.Module):
    """The exact frozen V12.4 ViT evaluated at four input phases.

    ``FrozenV9ViT`` is the historical class name of the original encoder
    embedded in V12.4.  It owns the original ImageNet normalization and
    position embedding; this wrapper adds no trainable parameter.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        expected_sha256: str = V12_4_SHA256,
    ) -> None:
        super().__init__()
        from v10.model import FrozenV9ViT

        path = Path(checkpoint).expanduser().resolve()
        digest = sha256_file(path)
        if digest != expected_sha256:
            raise ValueError(f"V12.4 checkpoint SHA mismatch: {digest}")
        self.vit = FrozenV9ViT(path)
        self.vit.requires_grad_(False)
        self.vit.eval()
        self.source_checkpoint = str(path)
        self.source_sha256 = digest
        grid, valid = phase_candidate_grid()
        self.register_buffer("candidate_coordinates_px", grid, persistent=False)
        self.register_buffer("candidate_valid", valid, persistent=False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenV12FourPhaseViT":
        super().train(False)
        self.vit.eval()
        return self

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode_native(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CLS and 16x16 patches under the original V12.4 encoder."""

        value = torch.as_tensor(images)
        if value.ndim != 4 or tuple(value.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("images must be [N,3,224,224]")
        pixels = value.float()
        if value.dtype == torch.uint8:
            pixels = pixels / 255.0
        self.vit.eval()
        with torch.no_grad():
            encoded = self.vit(pixels)
        return encoded[:, 0], encoded[:, 1:]

    def encode_native_layer(
        self,
        images: torch.Tensor,
        *,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one frozen transformer block on the native 16x16 grid.

        Layers are numbered 1--12. Intermediate blocks use the encoder's
        existing final LayerNorm so every probe has a comparable scale. Layer
        12 returns the canonical post-LayerNorm endpoint and is exactly equal
        to :meth:`encode_native`.
        """

        return self.encode_native_layers(images, layers=(int(layer),))[int(layer)]

    def encode_native_layers(
        self,
        images: torch.Tensor,
        *,
        layers: tuple[int, ...],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Read several frozen blocks in one encoder pass.

        This is the efficient diagnostic path for comparing local detail at
        multiple depths.  It does not add a feature pyramid to deployment:
        each candidate reader still consumes exactly one selected ViT layer.
        """

        value = torch.as_tensor(images)
        if value.ndim != 4 or tuple(value.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("images must be [N,3,224,224]")
        selected = tuple(dict.fromkeys(int(layer) for layer in layers))
        if not selected or any(not 1 <= layer <= 12 for layer in selected):
            raise ValueError("ViT layers must be a non-empty subset of [1,12]")
        pixels = value.float()
        if value.dtype == torch.uint8:
            pixels = pixels / 255.0
        pixels = (pixels - self.vit.image_mean) / self.vit.image_std
        self.vit.eval()
        with torch.no_grad():
            output = self.vit.encoder(
                pixels,
                interpolate_pos_encoding=True,
                output_hidden_states=True,
            )
            result = {}
            for layer in selected:
                if layer == 12:
                    encoded = output.last_hidden_state
                else:
                    encoded = self.vit.encoder.layernorm(
                        output.hidden_states[layer]
                    )
                if encoded.shape[1:] != (NATIVE_PATCHES + 1, TOKEN_DIM):
                    raise RuntimeError(
                        "intermediate ViT layer violated the native token contract"
                    )
                result[layer] = (encoded[:, 0], encoded[:, 1:])
        return result

    def forward(self, images: torch.Tensor) -> FourPhaseTokens:
        value = torch.as_tensor(images)
        crops = phase_shifted_crops(value)
        batch = len(value)
        flat = crops.reshape(batch * PHASE_COUNT, 3, IMAGE_SIZE, IMAGE_SIZE)
        cls, patches = self.encode_native(flat)
        tokens = patches.reshape(batch, PHASE_COUNT, NATIVE_PATCHES, TOKEN_DIM)
        coordinates = self.candidate_coordinates_px.to(
            device=tokens.device, dtype=tokens.dtype
        )
        valid = self.candidate_valid.to(device=tokens.device)
        return FourPhaseTokens(
            tokens=tokens,
            # The carrier consumes the exact native (0,0) CLS token.  Shifted
            # passes exist only to densify the current-image spatial support;
            # averaging their CLS tokens would silently change the original
            # V12.4 global-context contract.
            global_context=cls.reshape(batch, PHASE_COUNT, TOKEN_DIM)[:, 0],
            coordinates_px=coordinates,
            valid=valid,
        )


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if logits.shape != valid.shape:
        raise ValueError("masked softmax shape mismatch")
    mask = valid.bool()
    finite = torch.where(mask, logits.float(), torch.full_like(logits.float(), -1.0e9))
    maximum = finite.amax(dim=-1, keepdim=True)
    exponent = torch.where(mask, torch.exp(finite - maximum), torch.zeros_like(finite))
    denominator = exponent.sum(dim=-1, keepdim=True)
    return torch.where(denominator > 0, exponent / denominator.clamp_min(1.0e-12), exponent)


def _weighted_tokens(probabilities: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...tc,...cd->...td", probabilities, tokens)


def _weighted_coordinates(
    probabilities: torch.Tensor, coordinates_px: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("...tc,...ce->...te", probabilities, coordinates_px)


def permutation_invariant_spatial_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """Hungarian CE for an unordered two-element set.

    The two-element Hungarian problem has only the identity and swapped
    assignment.  Targets may share a patch/index; closed-jaw supervision does
    not raise or silently invalidate the row.
    """

    value = torch.as_tensor(logits)
    label = torch.as_tensor(targets, device=value.device).long()
    valid = torch.as_tensor(target_valid, device=value.device).bool()
    if value.ndim != 5 or value.shape[-2] != TIPS:
        raise ValueError("logits must be [B,H,V,2,C]")
    if label.shape != value.shape[:-1] or valid.shape != label.shape:
        raise ValueError("unordered target shape mismatch")
    units = valid.all(dim=-1)
    if not bool(units.any()):
        raise ValueError("unordered spatial CE has no complete visible set")
    selected_logits = value[units].float()
    selected_target = label[units]
    candidates = value.shape[-1]
    if bool(((selected_target < 0) | (selected_target >= candidates)).any()):
        raise ValueError("unordered target index lies outside candidate support")
    log_prob = F.log_softmax(selected_logits, dim=-1)
    identity = -(
        log_prob[:, 0].gather(1, selected_target[:, 0:1]).squeeze(1)
        + log_prob[:, 1].gather(1, selected_target[:, 1:2]).squeeze(1)
    )
    swapped = -(
        log_prob[:, 0].gather(1, selected_target[:, 1:2]).squeeze(1)
        + log_prob[:, 1].gather(1, selected_target[:, 0:1]).squeeze(1)
    )
    return torch.minimum(identity, swapped).mean()


def unordered_set_epe(
    predicted_px: torch.Tensor,
    target_px: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mean optimal two-point assignment error for reporting."""

    prediction = torch.as_tensor(predicted_px).float()
    target = torch.as_tensor(target_px, device=prediction.device).float()
    mask = torch.as_tensor(valid, device=prediction.device).bool()
    if prediction.shape != target.shape or prediction.shape[-2:] != (TIPS, 2):
        raise ValueError("point sets must end in [2,2]")
    if mask.shape != prediction.shape[:-1]:
        raise ValueError("point validity shape mismatch")
    units = mask.all(dim=-1)
    if not bool(units.any()):
        raise ValueError("unordered EPE has no complete visible set")
    p = prediction[units]
    q = target[units]
    identity = torch.linalg.vector_norm(p - q, dim=-1).mean(dim=-1)
    swapped = torch.linalg.vector_norm(p - q.flip(1), dim=-1).mean(dim=-1)
    return torch.minimum(identity, swapped).mean()


class UnorderedActionDemodulatedMeasurement(nn.Module):
    """Shared-view measurement that ranks native-ViT support only."""

    def __init__(self, *, token_dim: int = TOKEN_DIM, hidden_dim: int = 192) -> None:
        super().__init__()
        if token_dim <= 0 or hidden_dim <= 0:
            raise ValueError("measurement dimensions must be positive")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.current_norm = nn.LayerNorm(token_dim)
        self.tube_norm = nn.LayerNorm(token_dim)
        self.visual_projection = nn.Sequential(
            nn.Linear(3 * token_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.control_projection = nn.Sequential(
            nn.Linear(RAW_STEPS_PER_CHUNK * STATE_DIM + STATE_DIM + TIPS * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.tip_slots = nn.Parameter(torch.empty(TIPS, hidden_dim))
        self.score_bias = nn.Linear(3, TIPS, bias=False)
        nn.init.normal_(self.tip_slots, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        *,
        candidate_tokens: torch.Tensor,
        tube_tokens: torch.Tensor,
        candidate_coordinates_px: torch.Tensor,
        candidate_valid: torch.Tensor,
        view_valid: torch.Tensor,
        completed_joint_increments: torch.Tensor,
        measured_state: torch.Tensor,
        cad_tip_displacement: torch.Tensor,
        demodulation_features: torch.Tensor,
    ) -> UnorderedMeasurement:
        current = torch.as_tensor(candidate_tokens)
        tube = torch.as_tensor(tube_tokens, device=current.device, dtype=current.dtype)
        coordinates = torch.as_tensor(
            candidate_coordinates_px, device=current.device, dtype=current.dtype
        )
        support_valid = torch.as_tensor(candidate_valid, device=current.device).bool()
        views = torch.as_tensor(view_valid, device=current.device).bool()
        joint = torch.as_tensor(
            completed_joint_increments, device=current.device, dtype=current.dtype
        )
        state = torch.as_tensor(measured_state, device=current.device, dtype=current.dtype)
        cad = torch.as_tensor(
            cad_tip_displacement, device=current.device, dtype=current.dtype
        )
        demod = torch.as_tensor(
            demodulation_features, device=current.device, dtype=current.dtype
        )
        if current.ndim != 5:
            raise ValueError("candidate_tokens must be [B,H,V,C,D]")
        batch, horizons, num_views, candidates, dimension = current.shape
        if (horizons, num_views, dimension) != (HORIZONS, VIEWS, self.token_dim):
            raise ValueError("candidate token contract drifted")
        expected = {
            "tube_tokens": ((batch, HORIZONS, VIEWS, candidates, dimension), tube),
            "candidate_coordinates_px": ((batch, HORIZONS, VIEWS, candidates, 2), coordinates),
            "candidate_valid": ((batch, HORIZONS, VIEWS, candidates), support_valid),
            "view_valid": ((batch, HORIZONS, VIEWS), views),
            "completed_joint_increments": ((batch, HORIZONS, RAW_STEPS_PER_CHUNK, STATE_DIM), joint),
            "measured_state": ((batch, HORIZONS, STATE_DIM), state),
            "cad_tip_displacement": ((batch, HORIZONS, TIPS, 3), cad),
            "demodulation_features": ((batch, HORIZONS, VIEWS, candidates, 3), demod),
        }
        for name, (shape, value) in expected.items():
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must be {shape}")
        has_support = support_valid.any(dim=-1)
        if not bool((has_support | ~views).all()):
            raise ValueError("every valid horizon/view needs native-ViT support")
        finite_tensors = (current, tube, coordinates, joint, state, cad, demod)
        if not all(bool(torch.isfinite(value).all()) for value in finite_tensors):
            raise ValueError("measurement inputs must be finite")

        valid = support_valid & views[..., None]
        delta = self.current_norm(current) - self.tube_norm(tube)
        weights = valid.to(delta.dtype)[..., None]
        common = (delta * weights).sum(dim=-2, keepdim=True) / weights.sum(
            dim=-2, keepdim=True
        ).clamp_min(1.0)
        relative = delta - common
        visual = self.visual_projection(
            torch.cat((self.current_norm(current), self.tube_norm(tube), relative), dim=-1)
        )
        visual = F.normalize(visual, dim=-1)
        control = torch.cat(
            (joint.flatten(start_dim=-2), state, cad.flatten(start_dim=-2)), dim=-1
        )
        control_query = self.control_projection(control)
        query = F.normalize(control_query[..., None, :] + self.tip_slots, dim=-1)
        logits = torch.einsum("bhvcd,bhtd->bhvtc", visual, query)
        logits = logits / 0.10
        logits = logits + self.score_bias(demod).movedim(-1, -2)
        valid_for_tip = valid[..., None, :].expand(-1, -1, -1, TIPS, -1)
        logits = logits.masked_fill(~valid_for_tip, float("-inf"))
        probability = _masked_softmax(logits, valid_for_tip)
        tokens = _weighted_tokens(probability, current)
        tip_px = _weighted_coordinates(probability, coordinates)
        center = tip_px.mean(dim=-2)

        selected = probability.argmax(dim=-1)
        aperture = demod[..., 2]
        selected_aperture = aperture[..., None, :].expand(
            -1, -1, -1, TIPS, -1
        ).gather(-1, selected[..., None]).squeeze(-1)
        moving_slot = selected_aperture.abs().argmax(dim=-1)
        aperture_motion = joint[..., -1].sum(dim=-1).abs()
        separation = (
            selected_aperture[..., 0] - selected_aperture[..., 1]
        ).abs()
        identity_valid = (
            views
            & (aperture_motion[..., None] >= 1.0e-3)
            & (separation >= 0.05)
        )
        return UnorderedMeasurement(
            logits=logits,
            probabilities=probability,
            tokens=tokens,
            tip_px=tip_px,
            center_px=center,
            aperture_response=selected_aperture,
            identity_valid=identity_valid,
            moving_slot=moving_slot,
        )


class NativeV12InitializedLeWM(nn.Module):
    """Goal-free recurrent part-token dynamics initialized from V12.4."""

    def __init__(self, *, token_dim: int = TOKEN_DIM) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.set_projection = nn.Sequential(
            nn.LayerNorm(3 * token_dim),
            nn.Linear(3 * token_dim, token_dim),
            nn.GELU(),
        )
        self.memory_token_norm = nn.LayerNorm(token_dim)
        memory_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=6,
            dim_feedforward=4 * token_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            memory_layer,
            num_layers=2,
            norm=nn.LayerNorm(token_dim),
            enable_nested_tensor=False,
        )
        self.control_projection = nn.Sequential(
            nn.Linear(RAW_STEPS_PER_CHUNK * STATE_DIM + STATE_DIM + TIPS * 3, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.dynamics_input = nn.Sequential(
            nn.LayerNorm(2 * token_dim), nn.Linear(2 * token_dim, token_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=6,
            dim_feedforward=4 * token_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.dynamics = nn.TransformerEncoder(
            layer,
            num_layers=2,
            norm=nn.LayerNorm(token_dim),
            enable_nested_tensor=False,
        )
        self.horizon_embedding = nn.Parameter(torch.empty(1, HORIZONS, token_dim))
        self.feedback_head = nn.Sequential(
            nn.LayerNorm(2 * token_dim),
            nn.Linear(2 * token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        nn.init.normal_(self.horizon_embedding, std=0.02)
        nn.init.zeros_(self.feedback_head[-1].weight)
        nn.init.zeros_(self.feedback_head[-1].bias)
        self.initialization = "fresh_shape_compatible"

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initialize_from_v12_policy(self, policy: nn.Module) -> None:
        """Copy only shape-identical goal-free V12.4 dynamics tensors."""

        required = (
            "memory_encoder",
            "dynamics_input",
            "dynamics",
            "horizon_embedding",
        )
        if not all(hasattr(policy, name) for name in required):
            raise ValueError("V12.4 policy does not expose goal-free dynamics")
        source_input = getattr(policy, "dynamics_input")
        source_dynamics = getattr(policy, "dynamics")
        source_memory = getattr(policy, "memory_encoder")
        source_horizon = getattr(policy, "horizon_embedding")
        self.memory_encoder.load_state_dict(
            copy.deepcopy(source_memory.state_dict()), strict=True
        )
        self.dynamics_input.load_state_dict(copy.deepcopy(source_input.state_dict()), strict=True)
        self.dynamics.load_state_dict(copy.deepcopy(source_dynamics.state_dict()), strict=True)
        if tuple(source_horizon.shape) != tuple(self.horizon_embedding.shape):
            raise ValueError("V12.4 horizon embedding shape drifted")
        with torch.no_grad():
            self.horizon_embedding.copy_(source_horizon)
        self.initialization = "v12_4_goal_free_dynamics_exact_copy"

    def transition(
        self,
        *,
        previous_posterior: torch.Tensor,
        global_context: torch.Tensor,
        view_valid: torch.Tensor,
        completed_joint_increments: torch.Tensor,
        measured_state: torch.Tensor,
        cad_tip_displacement: torch.Tensor,
        prefix_inputs: list[torch.Tensor],
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if not 0 <= horizon < HORIZONS:
            raise ValueError("horizon lies outside H1-H4")
        batch = len(previous_posterior)
        if previous_posterior.shape != (batch, VIEWS, TIPS, self.token_dim):
            raise ValueError("previous posterior must be [B,2,2,D]")
        if global_context.shape != (batch, VIEWS, self.token_dim):
            raise ValueError("global context must be [B,2,D]")
        views = torch.as_tensor(view_valid, device=global_context.device).bool()
        if views.shape != (batch, VIEWS) or not bool(views.any(dim=1).all()):
            raise ValueError("each LeWM row needs at least one valid view")
        view_weight = views.to(global_context.dtype)
        posterior_weight = view_weight[:, :, None, None]
        posterior_set = (previous_posterior * posterior_weight).sum(dim=(1, 2))
        posterior_set = posterior_set / (view_weight.sum(dim=1, keepdim=True) * TIPS)
        context_masked = global_context * view_weight[..., None]
        view_set = context_masked.sum(dim=1) / view_weight.sum(dim=1, keepdim=True)
        both = views.all(dim=1, keepdim=True).to(global_context.dtype)
        cross_view_difference = (global_context[:, 1] - global_context[:, 0]) * both
        memory_tokens = torch.stack(
            (posterior_set, context_masked[:, 0], context_masked[:, 1]), dim=1
        )
        memory_summary = self.memory_encoder(
            self.memory_token_norm(memory_tokens)
        ).mean(dim=1)
        observation = self.set_projection(
            torch.cat((posterior_set, view_set, cross_view_difference), dim=-1)
        ) + memory_summary
        control = torch.cat(
            (
                completed_joint_increments.flatten(start_dim=-2),
                measured_state,
                cad_tip_displacement.flatten(start_dim=-2),
            ),
            dim=-1,
        )
        control = self.control_projection(control)
        current_input = self.dynamics_input(torch.cat((observation, control), dim=-1))
        current_input = current_input + self.horizon_embedding[:, horizon]
        updated_prefix = [*prefix_inputs, current_input]
        sequence = torch.stack(updated_prefix, dim=1)
        causal = nn.Transformer.generate_square_subsequent_mask(
            sequence.shape[1], device=sequence.device, dtype=sequence.dtype
        )
        latent = self.dynamics(sequence, mask=causal)[:, -1]
        expanded = latent[:, None, None, :].expand(-1, VIEWS, TIPS, -1)
        residual = torch.tanh(
            self.feedback_head(torch.cat((expanded, previous_posterior), dim=-1))
        )
        # Predict in the original V12.4 token space so the Smooth-L1 target is
        # not burdened with an irreducible unit-norm scale mismatch.  Cosine
        # normalization is applied only at the spatial re-ranking interface.
        prior = previous_posterior + LEWM_RESIDUAL_GAIN * residual
        return prior, latent, updated_prefix


class DetachedResidualActor(nn.Module):
    """Task-conditioned bounded residual over a causal motion baseline."""

    def __init__(
        self,
        *,
        step_cap: torch.Tensor,
        action_lower: torch.Tensor,
        action_upper: torch.Tensor,
        token_dim: int = TOKEN_DIM,
    ) -> None:
        super().__init__()
        for name, value in (
            ("step_cap", step_cap),
            ("action_lower", action_lower),
            ("action_upper", action_upper),
        ):
            tensor = torch.as_tensor(value).float()
            if tensor.shape != (STATE_DIM,) or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be one finite six-joint vector")
        if bool((torch.as_tensor(step_cap) <= 0).any()):
            raise ValueError("actor step cap must be positive")
        if bool((torch.as_tensor(action_lower) >= torch.as_tensor(action_upper)).any()):
            raise ValueError("actor limits are invalid")
        self.register_buffer("step_cap", torch.as_tensor(step_cap).float().clone())
        self.register_buffer("action_lower", torch.as_tensor(action_lower).float().clone())
        self.register_buffer("action_upper", torch.as_tensor(action_upper).float().clone())
        input_dim = 5 * token_dim + STATE_DIM + 2 * STATE_DIM
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 2 * token_dim),
            nn.GELU(),
            nn.Linear(2 * token_dim, 2 * token_dim),
            nn.GELU(),
        )
        self.action_head = nn.Linear(2 * token_dim, ACTION_HORIZON * STATE_DIM)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        *,
        current_posterior_tokens: torch.Tensor,
        dynamics_latent: torch.Tensor,
        measured_state: torch.Tensor,
        past_actions: torch.Tensor,
        global_context: torch.Tensor,
        view_valid: torch.Tensor,
        task_token: torch.Tensor,
        baseline_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        posterior = current_posterior_tokens.detach()
        dynamics = dynamics_latent.detach()
        context = global_context.detach()
        views = torch.as_tensor(view_valid, device=posterior.device).bool()
        task = task_token.detach()
        state = measured_state.detach()
        past = past_actions.detach()
        baseline = baseline_action.detach()
        batch = len(posterior)
        expected = {
            "current_posterior_tokens": ((batch, VIEWS, TIPS, TOKEN_DIM), posterior),
            "dynamics_latent": ((batch, HORIZONS, TOKEN_DIM), dynamics),
            "measured_state": ((batch, STATE_DIM), state),
            "past_actions": ((batch, 2, STATE_DIM), past),
            "global_context": ((batch, VIEWS, TOKEN_DIM), context),
            "view_valid": ((batch, VIEWS), views),
            "task_token": ((batch, TOKEN_DIM), task),
            "baseline_action": ((batch, ACTION_HORIZON, STATE_DIM), baseline),
        }
        for name, (shape, value) in expected.items():
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must be {shape}")
        if not bool(torch.isfinite(baseline).all()):
            raise ValueError("actor baseline must be finite")
        if not bool(views.any(dim=1).all()):
            raise ValueError("each actor row needs at least one valid view")
        _assert_recovering_action_trajectory(
            baseline,
            state,
            step_cap=self.step_cap,
            action_lower=self.action_lower,
            action_upper=self.action_upper,
        )
        view_weight = views.to(posterior.dtype)
        posterior_mean = (
            posterior * view_weight[:, :, None, None]
        ).sum(dim=(1, 2)) / (view_weight.sum(dim=1, keepdim=True) * TIPS)
        context_mean = (context * view_weight[..., None]).sum(dim=1)
        context_mean = context_mean / view_weight.sum(dim=1, keepdim=True)
        features = torch.cat(
            (
                posterior_mean,
                dynamics.mean(dim=1),
                dynamics[:, -1],
                context_mean,
                task,
                state,
                past.flatten(start_dim=1),
            ),
            dim=-1,
        )
        residual = torch.tanh(self.action_head(self.encoder(features))).reshape(
            batch, ACTION_HORIZON, STATE_DIM
        )
        residual = residual * self.step_cap.view(1, 1, -1)
        desired = baseline + residual
        action = project_bounded_action_trajectory(
            state,
            desired,
            step_cap=self.step_cap,
            action_lower=self.action_lower,
            action_upper=self.action_upper,
        )
        return action, action - baseline


class ProdV162(nn.Module):
    """End-to-end cached-token V16.2 model with isolated optimizers."""

    def __init__(
        self,
        *,
        step_cap: torch.Tensor,
        action_lower: torch.Tensor,
        action_upper: torch.Tensor,
    ) -> None:
        super().__init__()
        self.measurement = UnorderedActionDemodulatedMeasurement()
        self.lewm = NativeV12InitializedLeWM()
        self.actor = DetachedResidualActor(
            step_cap=step_cap,
            action_lower=action_lower,
            action_upper=action_upper,
        )

    def measurement_parameters(self):
        yield from self.measurement.parameters()

    def lewm_parameters(self):
        yield from self.lewm.parameters()

    def actor_parameters(self):
        yield from self.actor.parameters()

    def forward(
        self,
        *,
        candidate_tokens: torch.Tensor,
        tube_tokens: torch.Tensor,
        candidate_coordinates_px: torch.Tensor,
        candidate_valid: torch.Tensor,
        view_valid: torch.Tensor,
        initial_posterior_tokens: torch.Tensor,
        global_context: torch.Tensor,
        completed_joint_increments: torch.Tensor,
        measured_state: torch.Tensor,
        cad_tip_displacement: torch.Tensor,
        demodulation_features: torch.Tensor,
        past_actions: torch.Tensor,
        task_token: torch.Tensor,
        baseline_action: torch.Tensor,
        recursive_writeback: bool = True,
    ) -> ProdV162Output:
        measurement = self.measurement(
            candidate_tokens=candidate_tokens,
            tube_tokens=tube_tokens,
            candidate_coordinates_px=candidate_coordinates_px,
            candidate_valid=candidate_valid,
            view_valid=view_valid,
            completed_joint_increments=completed_joint_increments,
            measured_state=measured_state,
            cad_tip_displacement=cad_tip_displacement,
            demodulation_features=demodulation_features,
        )
        current = torch.as_tensor(candidate_tokens)
        coordinates = torch.as_tensor(
            candidate_coordinates_px, device=current.device, dtype=current.dtype
        )
        valid = torch.as_tensor(candidate_valid, device=current.device).bool()
        views = torch.as_tensor(view_valid, device=current.device).bool()
        previous = torch.as_tensor(
            initial_posterior_tokens, device=current.device, dtype=current.dtype
        )
        batch = len(current)
        if previous.shape != (batch, VIEWS, TIPS, TOKEN_DIM):
            raise ValueError("initial posterior tokens must be [B,2,2,192]")
        if global_context.shape != (batch, HORIZONS, VIEWS, TOKEN_DIM):
            raise ValueError("global_context must be [B,4,2,192]")

        posterior_logits = []
        posterior_tokens = []
        posterior_px = []
        prior_tokens = []
        support_masks = []
        dynamics_latent = []
        prefix: list[torch.Tensor] = []
        for horizon in range(HORIZONS):
            prior, latent, prefix = self.lewm.transition(
                previous_posterior=previous,
                global_context=global_context[:, horizon],
                view_valid=views[:, horizon],
                completed_joint_increments=completed_joint_increments[:, horizon],
                measured_state=measured_state[:, horizon],
                cad_tip_displacement=cad_tip_displacement[:, horizon],
                prefix_inputs=prefix,
                horizon=horizon,
            )
            # The LeWM optimizer owns only dynamics and feedback.  Spatial CE
            # trains the measurement in its independent optimizer; future-token
            # loss must not pull that measurement toward an action shortcut.
            measure = measurement.logits[:, horizon].detach()
            measure_valid = valid[:, horizon] & views[:, horizon, :, None]
            candidates = measure.shape[-1]
            k = min(PRIOR_SUPPORT_TOPK, candidates)
            top = torch.topk(measure, k=k, dim=-1).indices
            support = torch.zeros_like(measure, dtype=torch.bool)
            support.scatter_(-1, top, True)
            support = support & measure_valid[:, :, None, :]
            normalized_candidate = F.normalize(current[:, horizon].float(), dim=-1)
            normalized_prior = F.normalize(prior.detach().float(), dim=-1)
            similarity = torch.einsum(
                "bvcd,bvtd->bvtc", normalized_candidate, normalized_prior
            )
            combined = measure.float() + (PRIOR_GAIN / PRIOR_TEMPERATURE) * similarity
            combined = combined.masked_fill(~support, float("-inf"))
            probability = _masked_softmax(combined, support)
            token = _weighted_tokens(probability, current[:, horizon].float())
            pixel = _weighted_coordinates(probability, coordinates[:, horizon].float())
            posterior_logits.append(combined)
            posterior_tokens.append(token)
            posterior_px.append(pixel)
            prior_tokens.append(prior)
            support_masks.append(support)
            dynamics_latent.append(latent)
            previous = token if recursive_writeback else prior

        posterior_logits_tensor = torch.stack(posterior_logits, dim=1)
        posterior_tokens_tensor = torch.stack(posterior_tokens, dim=1)
        posterior_px_tensor = torch.stack(posterior_px, dim=1)
        prior_tokens_tensor = torch.stack(prior_tokens, dim=1)
        dynamics_tensor = torch.stack(dynamics_latent, dim=1)
        support_tensor = torch.stack(support_masks, dim=1)
        # Actor planning is an open-loop use of the same LeWM.  It starts from
        # the H0 posterior and may read the candidate control chunks, but it
        # cannot read the future H1--H4 RGB measurements or their posteriors.
        planning_previous = torch.as_tensor(
            initial_posterior_tokens, device=current.device, dtype=current.dtype
        )
        planning_prefix: list[torch.Tensor] = []
        planning_latent = []
        initial_state = measured_state[:, 0]
        cumulative_joint = torch.zeros_like(initial_state)
        planning_context = global_context[:, 0]
        planning_view_valid = views[:, 0]
        for horizon in range(HORIZONS):
            cumulative_joint = cumulative_joint + completed_joint_increments[
                :, horizon
            ].sum(dim=1)
            planning_state = initial_state + cumulative_joint
            planning_prior, latent, planning_prefix = self.lewm.transition(
                previous_posterior=planning_previous,
                global_context=planning_context,
                view_valid=planning_view_valid,
                completed_joint_increments=completed_joint_increments[:, horizon],
                measured_state=planning_state,
                cad_tip_displacement=cad_tip_displacement[:, horizon],
                prefix_inputs=planning_prefix,
                horizon=horizon,
            )
            planning_latent.append(latent)
            planning_previous = planning_prior
        planning_dynamics_tensor = torch.stack(planning_latent, dim=1)
        actor_action, actor_residual = self.actor(
            current_posterior_tokens=initial_posterior_tokens,
            dynamics_latent=planning_dynamics_tensor,
            measured_state=measured_state[:, 0],
            past_actions=past_actions,
            global_context=planning_context,
            view_valid=planning_view_valid,
            task_token=task_token,
            baseline_action=baseline_action,
        )
        return ProdV162Output(
            measurement_logits=measurement.logits,
            posterior_logits=posterior_logits_tensor,
            measurement_tokens=measurement.tokens,
            posterior_tokens=posterior_tokens_tensor,
            measurement_tip_px=measurement.tip_px,
            posterior_tip_px=posterior_px_tensor,
            measurement_center_px=measurement.center_px,
            posterior_center_px=posterior_px_tensor.mean(dim=-2),
            prior_tokens=prior_tokens_tensor,
            dynamics_latent=dynamics_tensor,
            actor_action=actor_action,
            actor_baseline=baseline_action,
            actor_residual=actor_residual,
            identity_valid=measurement.identity_valid,
            moving_slot=measurement.moving_slot,
            prior_support=support_tensor,
        )


def load_v12_4_dynamics_and_action_contract(
    checkpoint: str | Path,
) -> tuple[Mapping[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load only the V12.4 policy state needed to initialize V16.2.

    This helper intentionally does not construct or return the old actor.  The
    caller creates the complete frozen carrier long enough to copy the
    goal-free dynamics weights, then discards it before optimization.
    """

    path = Path(checkpoint).expanduser().resolve()
    digest = sha256_file(path)
    if digest != V12_4_SHA256:
        raise ValueError(f"V12.4 checkpoint SHA mismatch: {digest}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != V12_4_STEP:
        raise ValueError("V16.2 requires V12.4 step-20000")
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("V12.4 checkpoint lacks architecture metadata")
    if architecture.get("family") != "v12":
        raise ValueError("V16.2 base does not declare family=v12")
    if architecture.get("structured_goal_enters_lewm") is not False:
        raise ValueError("V12.4 LeWM initialization is not goal-free")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("V12.4 checkpoint lacks complete model state")
    contract = payload.get("action_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("V12.4 checkpoint lacks action contract")
    tensors = {
        "step_cap": torch.as_tensor(contract["step_cap"]).float(),
        "action_lower": torch.as_tensor(contract["lower"]).float(),
        "action_upper": torch.as_tensor(contract["upper"]).float(),
    }
    return state, tensors


def actor_step_scaled_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    step_cap: torch.Tensor,
) -> torch.Tensor:
    """Actor residual error in commandable joint-step units."""

    value = torch.as_tensor(prediction).float()
    truth = torch.as_tensor(target, device=value.device).float()
    cap = torch.as_tensor(step_cap, device=value.device, dtype=value.dtype)
    if value.shape != truth.shape or value.shape[-2:] != (ACTION_HORIZON, STATE_DIM):
        raise ValueError("actor actions must be [B,20,6]")
    if cap.shape != (STATE_DIM,) or bool((cap <= 0).any()):
        raise ValueError("actor step cap must be positive [6]")
    return F.smooth_l1_loss(value / cap, truth / cap)
