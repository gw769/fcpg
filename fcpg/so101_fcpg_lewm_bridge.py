"""A small FCPG anchor interface for a frozen SO-101 LeWM checkpoint."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from fcpg.grounding_confidence_fusion import (
    COARSE_FINE_VARIANT,
    GroundingConfidenceSidecar,
    as_residual_only_lewm_output,
    fuse_from_coarse_fine_output,
    is_coarse_fine_output,
)
from fcpg.modules import sample_feature_map_at_pixels
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from fcpg.transformer_grounding import (
    LeWMAnchorResidualCorrector,
    LeWMGroundingOutput,
    TwoTipStructuredLeWMAnchorCorrector,
)


SO101_FCPG_CORRECTOR_VARIANTS = ("independent3", "two_tip_structured")
SO101_FCPG_OPTIONAL_CORRECTOR_VARIANTS = (COARSE_FINE_VARIANT,)
SO101_FCPG_KNOWN_CORRECTOR_VARIANTS = (
    SO101_FCPG_CORRECTOR_VARIANTS + SO101_FCPG_OPTIONAL_CORRECTOR_VARIANTS
)


def infer_so101_fcpg_corrector_variant(payload: dict[str, object]) -> str:
    """Infer the corrector architecture while preserving old checkpoints.

    ``coarse_fine`` is never inferred from state-dict keys. Legacy payloads
    without ``corrector_variant`` stay ``independent3`` / ``two_tip_structured``.
    """

    variant = payload.get("corrector_variant")
    args = payload.get("args")
    # Legacy dumps store ``args`` as ``vars(args)``. A leftover CLI default of
    # ``coarse_fine`` must not reclassify independent3 / two_tip weights and
    # then explode on strict load. Only an explicit top-level
    # ``corrector_variant`` may select ``coarse_fine``. Namespace args are
    # ignored entirely; Mapping args may still name the two legacy variants.
    if variant is None and isinstance(args, Mapping):
        args_variant = args.get("corrector_variant")
        if args_variant is not None and str(args_variant) != COARSE_FINE_VARIANT:
            variant = args_variant
    if variant is None:
        state = payload.get("corrector")
        if isinstance(state, dict) and any(
            str(key).startswith("tip_corrector.") for key in state
        ):
            variant = "two_tip_structured"
        else:
            variant = "independent3"
    variant = str(variant)
    if variant not in SO101_FCPG_KNOWN_CORRECTOR_VARIANTS:
        raise ValueError(f"unsupported SO-101 FCPG corrector variant: {variant}")
    return variant


def build_so101_fcpg_corrector(
    model: StateResidualLeWM,
    *,
    camera_context_dim: int,
    variant: str = "independent3",
) -> nn.Module:
    """Build one checkpoint-compatible SO-101 FCPG corrector variant."""

    if variant not in SO101_FCPG_KNOWN_CORRECTOR_VARIANTS:
        raise ValueError(f"unsupported SO-101 FCPG corrector variant: {variant}")
    heads = 6 if model.latent_dim % 6 == 0 else 1
    kwargs = {
        "visual_dim": model.latent_dim,
        "obs_dim": model.state_dim + int(camera_context_dim),
        "model_dim": model.latent_dim,
        "local_grid_size": 5,
        "local_radius_px": 16.0,
        "local_layers": 2,
        "global_layers": 1,
        "heads": heads,
        "dropout": 0.0,
        "max_residual_px": 20.0,
    }
    if variant == COARSE_FINE_VARIANT:
        return _build_coarse_fine_corrector(model, camera_context_dim, kwargs)
    if variant == "two_tip_structured":
        return TwoTipStructuredLeWMAnchorCorrector(**kwargs)
    return LeWMAnchorResidualCorrector(num_keypoints=3, **kwargs)


def _call_with_supported_kwargs(factory, kwargs: dict[str, object]):
    parameters = inspect.signature(factory).parameters
    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in parameters
        and parameters[key].kind != inspect.Parameter.VAR_KEYWORD
    }
    return factory(**filtered)


def _build_coarse_fine_corrector(
    model: StateResidualLeWM,
    camera_context_dim: int,
    kwargs: dict[str, object],
) -> nn.Module:
    del model, camera_context_dim
    try:
        import fcpg.coarse_fine_fcpg as coarse_fine_fcpg
    except ImportError as exc:
        raise ImportError(
            "coarse_fine corrector variant requires fcpg.coarse_fine_fcpg"
        ) from exc
    builder = getattr(coarse_fine_fcpg, "build_coarse_fine_fcpg", None)
    if callable(builder):
        return _call_with_supported_kwargs(builder, kwargs)
    cls = getattr(coarse_fine_fcpg, "CoarseFineFCPG", None)
    if cls is None:
        raise ImportError(
            "fcpg.coarse_fine_fcpg does not export CoarseFineFCPG"
        )
    return _call_with_supported_kwargs(cls, kwargs)


@dataclass(frozen=True)
class SO101AnchorBridgeOutput:
    context_embedding: torch.Tensor
    anchor_tokens: torch.Tensor
    sampled_nominal: torch.Tensor
    sampled_corrected: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True)
class SO101FCPGLeWMOutput:
    fcpg: LeWMGroundingOutput
    bridge: SO101AnchorBridgeOutput
    base_prediction: torch.Tensor
    prediction: torch.Tensor
    command_state: torch.Tensor
    state_prediction: torch.Tensor
    confidence: GroundingConfidenceSidecar | None = None
    raw_context: torch.Tensor | None = None


@dataclass(frozen=True)
class SO101DualCameraFusionOutput:
    context_embedding: torch.Tensor
    fused_tokens: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True)
class SO101DualCameraFCPGLeWMOutput:
    fcpg: LeWMGroundingOutput
    q_corrected_px: torch.Tensor
    per_view_context: torch.Tensor
    per_view_anchor_tokens: torch.Tensor
    fusion: SO101DualCameraFusionOutput
    base_prediction: torch.Tensor
    prediction: torch.Tensor
    command_state: torch.Tensor
    state_prediction: torch.Tensor


class SO101FCPGAnchorBridge(nn.Module):
    """Inject corrected local anchor evidence into an existing LeWM CLS state.

    The module consumes patch tokens from the *same* ViT used by LeWM. It does
    not run a second visual backbone or predict dynamics. A zero-initialized
    gate makes the initial output exactly equal to the frozen LeWM context.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        num_anchors: int = 3,
        heads: int = 6,
        ffn_dim: int = 384,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or num_anchors <= 0 or ffn_dim <= 0:
            raise ValueError("bridge dimensions must be positive")
        if latent_dim % heads:
            raise ValueError("latent_dim must be divisible by heads")
        self.latent_dim = int(latent_dim)
        self.num_anchors = int(num_anchors)
        geometry_dim = 2 + 2 + 3 + 1 + self.num_anchors
        self.anchor_projection = nn.Sequential(
            nn.LayerNorm(2 * self.latent_dim + geometry_dim),
            nn.Linear(2 * self.latent_dim + geometry_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.anchor_encoder = nn.TransformerEncoder(
            layer,
            num_layers=1,
            norm=nn.LayerNorm(self.latent_dim),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.gate = nn.Parameter(torch.zeros(self.latent_dim))

    def _validate_inputs(
        self,
        context_embedding: torch.Tensor,
        patch_tokens: torch.Tensor,
        q_nominal_px: torch.Tensor,
        q_corrected_px: torch.Tensor,
        anchors_xyz: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[int, int, int]:
        if context_embedding.ndim != 3:
            raise ValueError("context_embedding must have shape [B, T, D]")
        batch, frames, latent = context_embedding.shape
        if latent != self.latent_dim:
            raise ValueError(f"expected context latent dimension {self.latent_dim}")
        if patch_tokens.ndim != 4 or patch_tokens.shape[:2] != (batch, frames):
            raise ValueError("patch_tokens must have shape [B, T, P, D]")
        if patch_tokens.shape[-1] != self.latent_dim:
            raise ValueError(f"expected patch latent dimension {self.latent_dim}")
        point_shape = (batch, frames, self.num_anchors, 2)
        if q_nominal_px.shape != point_shape or q_corrected_px.shape != point_shape:
            raise ValueError(f"grounding points must have shape {point_shape}")
        if anchors_xyz.shape != (batch, frames, self.num_anchors, 3):
            raise ValueError("anchors_xyz has an incompatible shape")
        if valid.shape != (batch, frames, self.num_anchors):
            raise ValueError("valid has an incompatible shape")
        patch_count = int(patch_tokens.shape[2])
        grid_size = math.isqrt(patch_count)
        if grid_size * grid_size != patch_count:
            raise ValueError("patch token count must form a square grid")
        return batch, frames, grid_size

    def forward(
        self,
        context_embedding: torch.Tensor,
        patch_tokens: torch.Tensor,
        q_nominal_px: torch.Tensor,
        q_corrected_px: torch.Tensor,
        anchors_xyz: torch.Tensor,
        valid: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> SO101AnchorBridgeOutput:
        batch, frames, grid_size = self._validate_inputs(
            context_embedding,
            patch_tokens,
            q_nominal_px,
            q_corrected_px,
            anchors_xyz,
            valid,
        )
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must be positive")
        feature_map = (
            patch_tokens.reshape(
                batch * frames, grid_size, grid_size, self.latent_dim
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        flat_nominal = q_nominal_px.reshape(
            batch * frames, self.num_anchors, 2
        )
        flat_corrected = q_corrected_px.reshape(
            batch * frames, self.num_anchors, 2
        )
        sampled_nominal = sample_feature_map_at_pixels(
            feature_map,
            flat_nominal,
            image_size=image_size,
            padding_mode="border",
        ).reshape(batch, frames, self.num_anchors, self.latent_dim)
        sampled_corrected = sample_feature_map_at_pixels(
            feature_map,
            flat_corrected,
            image_size=image_size,
            padding_mode="border",
        ).reshape(batch, frames, self.num_anchors, self.latent_dim)

        scale = q_nominal_px.new_tensor(
            [max(width - 1, 1), max(height - 1, 1)]
        )
        identity = torch.eye(
            self.num_anchors,
            dtype=q_nominal_px.dtype,
            device=q_nominal_px.device,
        ).view(1, 1, self.num_anchors, self.num_anchors)
        identity = identity.expand(batch, frames, -1, -1)
        geometry = torch.cat(
            (
                q_nominal_px / scale,
                (q_corrected_px - q_nominal_px) / scale,
                anchors_xyz,
                valid.to(dtype=q_nominal_px.dtype).unsqueeze(-1),
                identity,
            ),
            dim=-1,
        )
        anchor_input = torch.cat(
            (
                sampled_corrected,
                sampled_corrected - sampled_nominal,
                geometry,
            ),
            dim=-1,
        )
        token_valid = valid.to(dtype=torch.bool)
        anchor_tokens = self.anchor_projection(anchor_input)
        anchor_tokens = anchor_tokens.masked_fill(
            ~token_valid.unsqueeze(-1), 0.0
        )
        flat_tokens = anchor_tokens.reshape(
            batch * frames, self.num_anchors, self.latent_dim
        )
        flat_valid = token_valid.reshape(batch * frames, self.num_anchors)
        has_valid = flat_valid.any(dim=1)
        if flat_valid.all():
            anchor_tokens = self.anchor_encoder(flat_tokens)
        else:
            padding_mask = ~flat_valid
            # PyTorch attention produces NaNs when every key is masked. A zero
            # dummy key keeps those rows finite; their outputs are removed below.
            padding_mask = padding_mask.clone()
            padding_mask[~has_valid, 0] = False
            anchor_tokens = self.anchor_encoder(
                flat_tokens,
                src_key_padding_mask=padding_mask,
            )
        anchor_tokens = anchor_tokens.masked_fill(
            ~flat_valid.unsqueeze(-1), 0.0
        ).reshape(batch, frames, self.num_anchors, self.latent_dim)

        weights = token_valid.to(dtype=anchor_tokens.dtype).unsqueeze(-1)
        pooled = (anchor_tokens * weights).sum(dim=2) / weights.sum(
            dim=2
        ).clamp_min(1.0)
        residual = self.output_projection(pooled)
        residual = residual.masked_fill(
            ~has_valid.reshape(batch, frames, 1), 0.0
        )
        context = context_embedding + torch.tanh(self.gate) * residual
        return SO101AnchorBridgeOutput(
            context_embedding=context,
            anchor_tokens=anchor_tokens,
            sampled_nominal=sampled_nominal,
            sampled_corrected=sampled_corrected,
            residual=residual,
        )


class SO101DualCameraAnchorFusion(nn.Module):
    """Fuse camera-specific anchor evidence while preserving a primary view.

    The primary camera context remains the exact LeWM input while ``gate`` is
    zero. Each view contributes one global token and identity-preserving anchor
    tokens; invalid anchors are masked, while a camera's global token remains
    available. A later reliability module can additionally mask a whole view.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        num_views: int = 2,
        num_anchors: int = 3,
        primary_view: int = 0,
        heads: int = 6,
        ffn_dim: int = 384,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(latent_dim, num_views, num_anchors, ffn_dim) <= 0:
            raise ValueError("fusion dimensions must be positive")
        if latent_dim % heads:
            raise ValueError("latent_dim must be divisible by heads")
        if not 0 <= primary_view < num_views:
            raise ValueError("primary_view is outside the configured views")
        self.latent_dim = int(latent_dim)
        self.num_views = int(num_views)
        self.num_anchors = int(num_anchors)
        self.primary_view = int(primary_view)
        self.view_embedding = nn.Parameter(
            torch.empty(self.num_views, self.latent_dim)
        )
        nn.init.normal_(self.view_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=1,
            norm=nn.LayerNorm(self.latent_dim),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.gate = nn.Parameter(torch.zeros(self.latent_dim))

    def forward(
        self,
        per_view_context: torch.Tensor,
        per_view_anchor_tokens: torch.Tensor,
        valid: torch.Tensor,
    ) -> SO101DualCameraFusionOutput:
        if per_view_context.ndim != 4:
            raise ValueError("per_view_context must have shape [B, T, V, D]")
        batch, frames, views, latent = per_view_context.shape
        expected_anchor_shape = (
            batch,
            frames,
            self.num_views,
            self.num_anchors,
            self.latent_dim,
        )
        if (views, latent) != (self.num_views, self.latent_dim):
            raise ValueError("per_view_context has incompatible view or latent dimensions")
        if per_view_anchor_tokens.shape != expected_anchor_shape:
            raise ValueError(
                f"per_view_anchor_tokens must have shape {expected_anchor_shape}"
            )
        if valid.shape != expected_anchor_shape[:-1]:
            raise ValueError("valid has an incompatible shape")

        view_identity = self.view_embedding.view(
            1, 1, self.num_views, 1, self.latent_dim
        )
        global_tokens = per_view_context.unsqueeze(3) + view_identity
        anchor_tokens = per_view_anchor_tokens + view_identity
        tokens = torch.cat((global_tokens, anchor_tokens), dim=3)
        tokens = tokens.reshape(
            batch * frames,
            self.num_views * (self.num_anchors + 1),
            self.latent_dim,
        )
        global_valid = torch.ones(
            batch,
            frames,
            self.num_views,
            1,
            dtype=torch.bool,
            device=valid.device,
        )
        token_valid = torch.cat((global_valid, valid), dim=3).reshape(
            batch * frames,
            self.num_views * (self.num_anchors + 1),
        )
        encoded = self.encoder(tokens, src_key_padding_mask=~token_valid)
        weights = token_valid.to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        residual = self.output_projection(pooled).reshape(
            batch, frames, self.latent_dim
        )
        primary = per_view_context[:, :, self.primary_view]
        context = primary + torch.tanh(self.gate) * residual
        return SO101DualCameraFusionOutput(
            context_embedding=context,
            fused_tokens=encoded.reshape(
                batch,
                frames,
                self.num_views * (self.num_anchors + 1),
                self.latent_dim,
            ),
            residual=residual,
        )


class SO101FCPGConditionedLeWM(nn.Module):
    """Run FCPG and a frozen SO-101 LeWM from one shared ViT forward pass.

    The corrector predicts image-space residuals from the LeWM patch map. The
    bridge then samples corrected local evidence and injects it into the CLS
    history consumed by the existing dynamics model. The bridge gate and FCPG
    output head are zero initialized, so a newly constructed module is exactly
    equivalent to the supplied checkpoint until its new components are trained.
    """

    def __init__(
        self,
        model: StateResidualLeWM,
        *,
        corrector: nn.Module | None = None,
        bridge: SO101FCPGAnchorBridge | None = None,
        num_anchors: int = 3,
        camera_context_dim: int = 0,
        freeze_model: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(model, StateResidualLeWM):
            raise TypeError("model must be a StateResidualLeWM")
        if num_anchors <= 0:
            raise ValueError("num_anchors must be positive")
        if camera_context_dim < 0:
            raise ValueError("camera_context_dim must be non-negative")
        self.model = model
        self.num_anchors = int(num_anchors)
        self.camera_context_dim = int(camera_context_dim)
        self.freeze_model = bool(freeze_model)
        heads = 6 if model.latent_dim % 6 == 0 else 1
        self.corrector = corrector or build_so101_fcpg_corrector(
            model,
            camera_context_dim=self.camera_context_dim,
        )
        self.bridge = bridge or SO101FCPGAnchorBridge(
            latent_dim=model.latent_dim,
            num_anchors=self.num_anchors,
            heads=heads,
            ffn_dim=2 * model.latent_dim,
            dropout=0.0,
        )
        if self.corrector.visual_dim != model.latent_dim:
            raise ValueError("corrector visual dimension must match LeWM latent dimension")
        expected_obs_dim = model.state_dim + self.camera_context_dim
        if self.corrector.obs_dim != expected_obs_dim:
            raise ValueError(
                "corrector observation dimension must match SO-101 state plus "
                "camera context"
            )
        if self.corrector.num_keypoints != self.num_anchors:
            raise ValueError("corrector anchor count mismatch")
        if self.bridge.latent_dim != model.latent_dim:
            raise ValueError("bridge latent dimension must match LeWM")
        if self.bridge.num_anchors != self.num_anchors:
            raise ValueError("bridge anchor count mismatch")
        if self.freeze_model:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> "SO101FCPGConditionedLeWM":
        super().train(mode)
        if self.freeze_model:
            self.model.eval()
        return self

    def _encode_shared_vision(
        self,
        pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pixels.ndim != 5 or pixels.shape[2] != 3:
            raise ValueError("pixels must have shape [B, T, 3, H, W]")
        batch, frames, channels, height, width = pixels.shape
        flat = pixels.reshape(batch * frames, channels, height, width).float()
        output = self.model.base.encoder(flat, interpolate_pos_encoding=True)
        hidden = output.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[0] != batch * frames:
            raise ValueError("LeWM vision encoder returned incompatible hidden states")
        if hidden.shape[1] <= 1 or hidden.shape[2] != self.model.latent_dim:
            raise ValueError("LeWM vision encoder must return CLS plus patch tokens")
        context = self.model.base.projector(hidden[:, 0]).reshape(
            batch, frames, self.model.latent_dim
        )
        patches = hidden[:, 1:].reshape(
            batch, frames, hidden.shape[1] - 1, self.model.latent_dim
        )
        grid_size = math.isqrt(int(patches.shape[2]))
        if grid_size * grid_size != patches.shape[2]:
            raise ValueError("LeWM patch token count must form a square grid")
        feature_map = (
            patches.reshape(
                batch * frames,
                grid_size,
                grid_size,
                self.model.latent_dim,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return context, patches, feature_map

    def predict_context(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        normalized_state: torch.Tensor,
        q_nominal_px: torch.Tensor,
        anchors_xyz: torch.Tensor,
        valid: torch.Tensor,
        camera_context: torch.Tensor | None = None,
        last_accepted_px: torch.Tensor | None = None,
        last_accepted_view: str | None = None,
        column_wrist_conflict: bool = False,
        view: str = "column",
    ) -> SO101FCPGLeWMOutput:
        """Predict one horizon sequence using deployable history only.

        Optional last-accepted / conflict arguments feed the confidence sidecar
        only. They do not enter LeWM dynamics, raw CLS, or residual-only
        ``fcpg``. Ranking is not an argument; it cannot change corrected.
        """

        if pixels.ndim != 5:
            raise ValueError("pixels must have shape [B, T, 3, H, W]")
        batch, frames, _, height, width = pixels.shape
        if actions.shape[:2] != (batch, frames):
            raise ValueError("actions must align with the visual history")
        if normalized_state.shape != (batch, frames, self.model.state_dim):
            raise ValueError("normalized_state has an incompatible shape")
        if q_nominal_px.shape != (batch, frames, self.num_anchors, 2):
            raise ValueError("q_nominal_px has an incompatible shape")
        if anchors_xyz.shape != (batch, frames, self.num_anchors, 3):
            raise ValueError("anchors_xyz has an incompatible shape")
        if valid.shape != (batch, frames, self.num_anchors):
            raise ValueError("valid has an incompatible shape")
        if self.camera_context_dim:
            expected_camera_shape = (batch, frames, self.camera_context_dim)
            if camera_context is None or camera_context.shape != expected_camera_shape:
                raise ValueError(
                    f"camera_context must have shape {expected_camera_shape}"
                )
        elif camera_context is not None:
            raise ValueError(
                "camera_context was supplied but camera_context_dim is zero"
            )

        context, patch_tokens, feature_map = self._encode_shared_vision(pixels)
        flat_nominal = q_nominal_px.reshape(batch * frames, self.num_anchors, 2)
        flat_xyz = anchors_xyz.reshape(batch * frames, self.num_anchors, 3)
        corrector_obs = normalized_state
        if camera_context is not None:
            corrector_obs = torch.cat((corrector_obs, camera_context), dim=-1)
        flat_state = corrector_obs.reshape(batch * frames, self.corrector.obs_dim)
        fcpg_native = self.corrector(
            feature_map,
            flat_nominal,
            flat_xyz,
            flat_state,
            image_size=(width, height),
        )
        fcpg = as_residual_only_lewm_output(fcpg_native)
        q_corrected = fcpg.q_corrected_px.reshape(
            batch, frames, self.num_anchors, 2
        )
        bridge = self.bridge(
            context,
            patch_tokens,
            q_nominal_px,
            q_corrected,
            anchors_xyz,
            valid,
            image_size=(width, height),
        )
        action_embedding = self.model.base.action_encoder(actions)
        base_prediction, prediction, command_state, state_prediction = (
            self.model.predict_context(
                bridge.context_embedding,
                actions,
                normalized_state,
                action_embedding=action_embedding,
            )
        )
        confidence = None
        if is_coarse_fine_output(fcpg_native):
            confidence = GroundingConfidenceSidecar(
                fusion=fuse_from_coarse_fine_output(
                    fcpg_native,
                    nominal_px=q_nominal_px.reshape(
                        batch * frames, self.num_anchors, 2
                    ),
                    view_valid=valid.reshape(batch * frames, self.num_anchors),
                    last_accepted_px=last_accepted_px,
                    last_accepted_view=last_accepted_view,
                    column_wrist_conflict=column_wrist_conflict,
                    view=view,
                )
            )
        return SO101FCPGLeWMOutput(
            fcpg=fcpg,
            bridge=bridge,
            base_prediction=base_prediction,
            prediction=prediction,
            command_state=command_state,
            state_prediction=state_prediction,
            confidence=confidence,
            raw_context=context,
        )


class SO101DualCameraFCPGConditionedLeWM(nn.Module):
    """Apply one shared FCPG to two views and fuse matching physical anchors."""

    def __init__(
        self,
        single_view: SO101FCPGConditionedLeWM,
        *,
        fusion: SO101DualCameraAnchorFusion | None = None,
        num_views: int = 2,
        primary_view: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(single_view, SO101FCPGConditionedLeWM):
            raise TypeError("single_view must be an SO101FCPGConditionedLeWM")
        if num_views <= 0:
            raise ValueError("num_views must be positive")
        self.single_view = single_view
        self.num_views = int(num_views)
        model = single_view.model
        heads = 6 if model.latent_dim % 6 == 0 else 1
        self.fusion = fusion or SO101DualCameraAnchorFusion(
            latent_dim=model.latent_dim,
            num_views=self.num_views,
            num_anchors=single_view.num_anchors,
            primary_view=primary_view,
            heads=heads,
            ffn_dim=2 * model.latent_dim,
            dropout=0.0,
        )
        if self.fusion.num_views != self.num_views:
            raise ValueError("fusion view count mismatch")
        if self.fusion.num_anchors != single_view.num_anchors:
            raise ValueError("fusion anchor count mismatch")
        if self.fusion.latent_dim != model.latent_dim:
            raise ValueError("fusion latent dimension mismatch")

    @property
    def model(self) -> StateResidualLeWM:
        return self.single_view.model

    def train(self, mode: bool = True) -> "SO101DualCameraFCPGConditionedLeWM":
        super().train(mode)
        if self.single_view.freeze_model:
            self.model.eval()
        return self

    @staticmethod
    def _flatten_views(value: torch.Tensor) -> torch.Tensor:
        order = (0, 2, 1, *range(3, value.ndim))
        transposed = value.permute(order)
        return transposed.reshape(
            value.shape[0] * value.shape[2],
            value.shape[1],
            *value.shape[3:],
        )

    def predict_context(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        normalized_state: torch.Tensor,
        q_nominal_px: torch.Tensor,
        anchors_xyz: torch.Tensor,
        valid: torch.Tensor,
        camera_context: torch.Tensor | None = None,
    ) -> SO101DualCameraFCPGLeWMOutput:
        if pixels.ndim != 6 or pixels.shape[3] != 3:
            raise ValueError("pixels must have shape [B, T, V, 3, H, W]")
        batch, frames, views, _, height, width = pixels.shape
        if views != self.num_views:
            raise ValueError(f"expected {self.num_views} camera views")
        if actions.shape[:2] != (batch, frames):
            raise ValueError("actions must align with the visual history")
        if normalized_state.shape != (batch, frames, self.model.state_dim):
            raise ValueError("normalized_state has an incompatible shape")
        point_shape = (
            batch,
            frames,
            views,
            self.single_view.num_anchors,
        )
        if q_nominal_px.shape != (*point_shape, 2):
            raise ValueError("q_nominal_px has an incompatible shape")
        if anchors_xyz.shape != (*point_shape, 3):
            raise ValueError("anchors_xyz has an incompatible shape")
        if valid.shape != point_shape:
            raise ValueError("valid has an incompatible shape")
        context_dim = self.single_view.camera_context_dim
        if context_dim:
            expected = (batch, frames, views, context_dim)
            if camera_context is None or camera_context.shape != expected:
                raise ValueError(f"camera_context must have shape {expected}")
        elif camera_context is not None:
            raise ValueError("camera_context was supplied but the corrector does not use it")

        flat_pixels = self._flatten_views(pixels)
        flat_nominal = self._flatten_views(q_nominal_px)
        flat_xyz = self._flatten_views(anchors_xyz)
        flat_valid = self._flatten_views(valid)
        flat_state = normalized_state[:, None].expand(
            -1, views, -1, -1
        ).reshape(batch * views, frames, self.model.state_dim)
        flat_camera = (
            None if camera_context is None else self._flatten_views(camera_context)
        )
        context, patch_tokens, feature_map = self.single_view._encode_shared_vision(
            flat_pixels
        )
        corrector_obs = flat_state
        if flat_camera is not None:
            corrector_obs = torch.cat((corrector_obs, flat_camera), dim=-1)
        fcpg_native = self.single_view.corrector(
            feature_map,
            flat_nominal.reshape(
                batch * views * frames, self.single_view.num_anchors, 2
            ),
            flat_xyz.reshape(
                batch * views * frames, self.single_view.num_anchors, 3
            ),
            corrector_obs.reshape(
                batch * views * frames, self.single_view.corrector.obs_dim
            ),
            image_size=(width, height),
        )
        fcpg = as_residual_only_lewm_output(fcpg_native)
        flat_corrected = fcpg.q_corrected_px.reshape(
            batch * views, frames, self.single_view.num_anchors, 2
        )
        bridge = self.single_view.bridge(
            context,
            patch_tokens,
            flat_nominal,
            flat_corrected,
            flat_xyz,
            flat_valid,
            image_size=(width, height),
        )
        per_view_context = bridge.context_embedding.reshape(
            batch, views, frames, self.model.latent_dim
        ).permute(0, 2, 1, 3)
        per_view_anchor_tokens = bridge.anchor_tokens.reshape(
            batch,
            views,
            frames,
            self.single_view.num_anchors,
            self.model.latent_dim,
        ).permute(0, 2, 1, 3, 4)
        per_view_valid = flat_valid.reshape(
            batch, views, frames, self.single_view.num_anchors
        ).permute(0, 2, 1, 3)
        fusion = self.fusion(
            per_view_context,
            per_view_anchor_tokens,
            per_view_valid,
        )
        action_embedding = self.model.base.action_encoder(actions)
        base_prediction, prediction, command_state, state_prediction = (
            self.model.predict_context(
                fusion.context_embedding,
                actions,
                normalized_state,
                action_embedding=action_embedding,
            )
        )
        q_corrected = flat_corrected.reshape(
            batch,
            views,
            frames,
            self.single_view.num_anchors,
            2,
        ).permute(0, 2, 1, 3, 4)
        return SO101DualCameraFCPGLeWMOutput(
            fcpg=fcpg,
            q_corrected_px=q_corrected,
            per_view_context=per_view_context,
            per_view_anchor_tokens=per_view_anchor_tokens,
            fusion=fusion,
            base_prediction=base_prediction,
            prediction=prediction,
            command_state=command_state,
            state_prediction=state_prediction,
        )


__all__ = [
    "SO101AnchorBridgeOutput",
    "SO101DualCameraAnchorFusion",
    "SO101DualCameraFCPGConditionedLeWM",
    "SO101DualCameraFCPGLeWMOutput",
    "SO101DualCameraFusionOutput",
    "SO101FCPGAnchorBridge",
    "SO101FCPGConditionedLeWM",
    "SO101FCPGLeWMOutput",
    "SO101_FCPG_CORRECTOR_VARIANTS",
    "SO101_FCPG_KNOWN_CORRECTOR_VARIANTS",
    "SO101_FCPG_OPTIONAL_CORRECTOR_VARIANTS",
    "as_residual_only_lewm_output",
    "build_so101_fcpg_corrector",
    "infer_so101_fcpg_corrector_variant",
]
