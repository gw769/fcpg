"""Factorized full-image fingertip grounding for Prod-V17.1.

V17.1 retains one ordered semantic F/M query per view.  The state contains no
explicit pixel, posterior, nominal-coordinate, action, or readback field.  A
query update reads current content memory without this module's explicit 2-D
position buffer; current spatial logits are then measured against a separate
current-image memory that does contain that buffer.  This architectural split
does not claim that a learned semantic latent is mathematically free of all
position information.

A fixed Gaussian energy is computed independently from the nominal F/M
projection.  The fused spatial distribution is the fixed product-of-experts
identity

``fused_logits = image_logits + nominal_prior_logits``.

The image factor has no action, readback, nominal-coordinate, gate, or router
input.  The optional history input is only the semantic query described above.
The final categorical class means only ``projection_invalid`` (behind the
camera or outside the image); it is not a claim that physical occlusion is
supervised.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


IMAGE_SIZE = 224
VIEWS = 2
TIPS = 2
SEMANTIC_GRID = 16
VISUAL_GRID = 56
VISUAL_CELLS = VISUAL_GRID * VISUAL_GRID
PROJECTION_INVALID_INDEX = VISUAL_CELLS
MODEL_DIMENSION = 192
DECODER_LAYERS = 3
DECODER_HEADS = 6
DECODER_FFN = 768
NOMINAL_PRIOR_SIGMA_PX = 32.0


class V171ContractError(ValueError):
    """Raised when the factorized V17.1 tensor contract is violated."""


@dataclass(frozen=True)
class V171Output:
    """Independent prior/image factors and their fixed spatial PoE."""

    prior_spatial_logits: torch.Tensor
    image_spatial_logits: torch.Tensor
    fused_spatial_logits: torch.Tensor
    projection_invalid_logit: torch.Tensor
    image_categorical_logits: torch.Tensor
    prior_spatial_probability: torch.Tensor
    image_spatial_probability: torch.Tensor
    fused_spatial_probability: torch.Tensor
    image_categorical_probability: torch.Tensor
    projection_invalid_probability: torch.Tensor
    prior_fm_px: torch.Tensor
    image_fm_px: torch.Tensor
    fused_fm_px: torch.Tensor
    prior_midpoint_px: torch.Tensor
    image_midpoint_px: torch.Tensor
    fused_midpoint_px: torch.Tensor
    prior_cfm_px: torch.Tensor
    image_cfm_px: torch.Tensor
    fused_cfm_px: torch.Tensor
    previous_query: torch.Tensor
    decoded_query: torch.Tensor
    next_query: torch.Tensor
    native_v12_tokens: torch.Tensor
    view_valid: torch.Tensor


@dataclass(frozen=True)
class V171ImageCategoricalLoss:
    """The sole V17.1 objective: image-only 3137-way categorical NLL."""

    total: torch.Tensor
    in_projection: torch.Tensor
    projection_invalid: torch.Tensor
    active_tips: torch.Tensor
    in_projection_tips: torch.Tensor
    projection_invalid_tips: torch.Tensor


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise V171ContractError(f"{name} must be finite")


def _boolean(
    name: str,
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device)
    if result.shape != shape or result.dtype != torch.bool:
        raise V171ContractError(f"{name} must be boolean {list(shape)}")
    return result


def _normalized_rgb_with_contrast(rgb: torch.Tensor) -> torch.Tensor:
    mean = rgb.mean(dim=(-2, -1), keepdim=True)
    scale = rgb.std(dim=(-2, -1), keepdim=True).clamp_min(0.05)
    low = F.avg_pool2d(
        F.pad(rgb, (7, 7, 7, 7), mode="reflect"),
        kernel_size=15,
        stride=1,
    )
    return torch.cat(((rgb - mean) / scale, (rgb - low) / scale), dim=1)


def _sine_position_2d(*, grid: int, dimension: int) -> torch.Tensor:
    if dimension % 4:
        raise ValueError("2-D sine position dimension must be divisible by four")
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, grid),
        torch.linspace(0.0, 1.0, grid),
        indexing="ij",
    )
    frequency = torch.exp(
        torch.arange(dimension // 4, dtype=torch.float32)
        * (-math.log(10_000.0) / max(dimension // 4 - 1, 1))
    )
    x_phase = 2.0 * math.pi * x.reshape(-1, 1) * frequency
    y_phase = 2.0 * math.pi * y.reshape(-1, 1) * frequency
    return torch.cat(
        (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()),
        dim=1,
    )


def _coordinates(
    probability: torch.Tensor, visual_xy_px: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fm = torch.einsum("bvtn,nd->bvtd", probability, visual_xy_px)
    midpoint = fm.mean(dim=2)
    cfm = torch.stack((midpoint, fm[:, :, 0], fm[:, :, 1]), dim=2)
    return fm, midpoint, cfm


class V171FactorizedQueryFCPG(nn.Module):
    """History-aware ordered-query image likelihood with fixed nominal prior.

    ``previous_query`` is a semantic latent, not an explicit spatial state.
    Views share weights but neither their pixels nor their query states mix.
    """

    def __init__(
        self,
        *,
        semantic_dim: int = MODEL_DIMENSION,
        dimension: int = MODEL_DIMENSION,
    ) -> None:
        super().__init__()
        if int(semantic_dim) <= 0:
            raise ValueError("semantic_dim must be positive")
        if int(dimension) != MODEL_DIMENSION:
            raise ValueError("V17.1 dimension is fixed at 192")
        self.semantic_dim = int(semantic_dim)
        self.dimension = int(dimension)

        # Names and shapes intentionally match the reusable V17 visual path.
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(6, 96, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(12, 96),
            nn.GELU(),
            nn.Conv2d(
                96,
                self.dimension,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(24, self.dimension),
            nn.GELU(),
            nn.Conv2d(
                self.dimension,
                self.dimension,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        self.semantic_projection = nn.Linear(self.semantic_dim, self.dimension)
        self.visual_fusion = nn.Sequential(
            nn.LayerNorm(self.dimension),
            nn.Linear(self.dimension, self.dimension),
            nn.GELU(),
            nn.Linear(self.dimension, self.dimension),
        )
        self.tip_identity = nn.Parameter(torch.empty(TIPS, self.dimension))
        self.view_identity = nn.Parameter(torch.empty(VIEWS, self.dimension))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.dimension,
            nhead=DECODER_HEADS,
            dim_feedforward=DECODER_FFN,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=DECODER_LAYERS,
            norm=nn.LayerNorm(self.dimension),
        )
        self.spatial_query_projection = nn.Linear(
            self.dimension, self.dimension, bias=False
        )
        self.spatial_key_projection = nn.Linear(
            self.dimension, self.dimension, bias=False
        )
        self.projection_invalid_head = nn.Linear(self.dimension, 1)

        self.register_buffer(
            "visual_position",
            _sine_position_2d(grid=VISUAL_GRID, dimension=self.dimension),
            persistent=True,
        )
        axis = torch.linspace(0.0, IMAGE_SIZE - 1.0, VISUAL_GRID)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "visual_xy_px",
            torch.stack((xx, yy), dim=-1).reshape(VISUAL_CELLS, 2),
            persistent=True,
        )
        nn.init.normal_(self.tip_identity, std=0.02)
        nn.init.normal_(self.view_identity, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_query(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the ordered H0 semantic query ``[B,2,2,192]``."""

        if isinstance(batch, bool) or int(batch) <= 0:
            raise V171ContractError("batch must be a positive integer")
        query = self.tip_identity[None, None] + self.view_identity[None, :, None]
        return query.to(device=device, dtype=dtype).expand(
            int(batch), VIEWS, TIPS, self.dimension
        )

    def update_semantic_query(
        self,
        previous_query: torch.Tensor,
        content_memory: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Update one H-state without accepting any explicit spatial history.

        ``content_memory`` must be the position-buffer-free branch returned by
        :meth:`encode_visual_grid`.  The method is shared by runtime forward and
        temporal training so view-local no-write semantics cannot drift.
        """

        content = torch.as_tensor(content_memory)
        if content.ndim != 4 or tuple(content.shape[1:]) != (
            VIEWS,
            VISUAL_CELLS,
            self.dimension,
        ):
            raise V171ContractError(
                "content_memory must be [B,2,3136,192]"
            )
        _finite("content_memory", content)
        batch = len(content)
        previous = torch.as_tensor(previous_query, device=content.device).to(
            content.dtype
        )
        if previous.shape != (batch, VIEWS, TIPS, self.dimension):
            raise V171ContractError("previous_query must be [B,2,2,192]")
        _finite("previous_query", previous)
        views = _boolean(
            "view_valid", view_valid, shape=(batch, VIEWS), device=content.device
        )
        decoded = self.decoder(
            previous.reshape(batch * VIEWS, TIPS, self.dimension),
            content.reshape(batch * VIEWS, VISUAL_CELLS, self.dimension),
        ).reshape(batch, VIEWS, TIPS, self.dimension)
        return torch.where(views[:, :, None, None], decoded, previous)

    def encode_visual_grid(
        self,
        rgb: torch.Tensor,
        frozen_v12_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return content memory, current spatial memory, and V12 tokens.

        ``content_memory`` does not receive this module's explicit sine-position
        buffer.  Frozen V12 features and convolutional content can nevertheless
        contain position-correlated information, so it is a semantic-state
        boundary rather than a proof of position-free latent information.
        """

        images = torch.as_tensor(rgb)
        if images.ndim != 5 or tuple(images.shape[1:]) != (
            VIEWS,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise V171ContractError("rgb must be [B,2,3,224,224]")
        batch = len(images)
        native = torch.as_tensor(frozen_v12_tokens, device=images.device)
        if tuple(native.shape) != (
            batch,
            VIEWS,
            SEMANTIC_GRID * SEMANTIC_GRID,
            self.semantic_dim,
        ):
            raise V171ContractError(
                f"frozen_v12_tokens must be [B,2,256,{self.semantic_dim}]"
            )
        images = images.float()
        native = native.detach().float()
        _finite("rgb", images)
        _finite("frozen_v12_tokens", native)

        rgb_grid = self.rgb_stem(
            _normalized_rgb_with_contrast(
                images.reshape(batch * VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE)
            )
        )
        if tuple(rgb_grid.shape) != (
            batch * VIEWS,
            self.dimension,
            VISUAL_GRID,
            VISUAL_GRID,
        ):
            raise RuntimeError("RGB stem must produce a 56x56 token grid")
        semantic = self.semantic_projection(native).reshape(
            batch * VIEWS,
            SEMANTIC_GRID,
            SEMANTIC_GRID,
            self.dimension,
        )
        semantic = F.interpolate(
            semantic.permute(0, 3, 1, 2),
            size=(VISUAL_GRID, VISUAL_GRID),
            mode="bilinear",
            align_corners=False,
        )
        fused = self.visual_fusion((rgb_grid + semantic).permute(0, 2, 3, 1))
        content_memory = fused.reshape(
            batch, VIEWS, VISUAL_CELLS, self.dimension
        )
        spatial_memory = (
            content_memory + self.visual_position.to(content_memory)[None, None]
        )
        return content_memory, spatial_memory, native

    def forward(
        self,
        rgb: torch.Tensor,
        frozen_v12_tokens: torch.Tensor,
        nominal_fm_px: torch.Tensor,
        view_valid: torch.Tensor,
        previous_query: torch.Tensor | None = None,
    ) -> V171Output:
        """Update semantic history and compute the current fixed spatial PoE.

        ``previous_query=None`` is exactly H0.  H1--H4 pass the preceding
        endpoint's ``next_query``.  The caller owns the horizon clock; it is not
        an input feature of this model.
        """

        content_memory, spatial_memory, detached_native = self.encode_visual_grid(
            rgb, frozen_v12_tokens
        )
        batch = len(content_memory)
        device, dtype = content_memory.device, content_memory.dtype
        views = _boolean(
            "view_valid", view_valid, shape=(batch, VIEWS), device=device
        )

        # H0 uses learned ordered identities.  Later endpoints carry only this
        # semantic tensor; no pixel, posterior, nominal, action, or readback is
        # accepted as historical state.
        if previous_query is None:
            previous = self.initial_query(batch, device=device, dtype=dtype)
        else:
            previous = torch.as_tensor(previous_query, device=device).to(dtype)
            if previous.shape != (batch, VIEWS, TIPS, self.dimension):
                raise V171ContractError("previous_query must be [B,2,2,192]")
            _finite("previous_query", previous)

        # The recurrent update sees no explicit spatial-position buffer.  The
        # same public method is used by temporal training; the spatial readout
        # below is the only consumer of current spatial memory.
        next_query = self.update_semantic_query(previous, content_memory, views)
        decoded = next_query
        image_query = self.spatial_query_projection(decoded)
        image_key = self.spatial_key_projection(spatial_memory)
        raw_image_logits = torch.einsum(
            "bvtd,bvnd->bvtn", image_query, image_key
        ) / math.sqrt(self.dimension)
        raw_projection_invalid = self.projection_invalid_head(decoded).squeeze(-1)

        valid_tip = views[:, :, None].expand(-1, -1, TIPS)
        # A sensor-invalid view contributes a uniform image factor.  Semantic
        # no-write has already been applied by ``update_semantic_query``.
        image_logits = torch.where(
            valid_tip[..., None], raw_image_logits, torch.zeros_like(raw_image_logits)
        )
        projection_invalid_logit = torch.where(
            valid_tip,
            raw_projection_invalid,
            torch.zeros_like(raw_projection_invalid),
        )

        # Nominal is confined to this fixed, parameter-free prior branch.
        nominal = torch.as_tensor(nominal_fm_px, device=device).detach().to(dtype)
        if nominal.shape != (batch, VIEWS, TIPS, 2):
            raise V171ContractError("nominal_fm_px must be [B,2,2,2]")
        _finite("nominal_fm_px", nominal)
        if bool(((nominal < 0.0) | (nominal > IMAGE_SIZE - 1.0)).any()):
            raise V171ContractError("nominal_fm_px must lie inside the image")
        displacement = (
            self.visual_xy_px.to(device=device, dtype=dtype)[None, None, None]
            - nominal[:, :, :, None]
        )
        prior_logits = -0.5 * displacement.square().sum(dim=-1) / (
            NOMINAL_PRIOR_SIGMA_PX**2
        )
        fused_logits = image_logits + prior_logits

        prior_probability = torch.softmax(prior_logits, dim=-1)
        image_probability = torch.softmax(image_logits, dim=-1)
        fused_probability = torch.softmax(fused_logits, dim=-1)
        image_categorical_logits = torch.cat(
            (image_logits, projection_invalid_logit[..., None]), dim=-1
        )
        image_categorical_probability = torch.softmax(
            image_categorical_logits, dim=-1
        )
        projection_invalid_probability = image_categorical_probability[
            ..., PROJECTION_INVALID_INDEX
        ]

        xy = self.visual_xy_px.to(device=device, dtype=dtype)
        prior_fm, prior_midpoint, prior_cfm = _coordinates(prior_probability, xy)
        image_fm, image_midpoint, image_cfm = _coordinates(image_probability, xy)
        fused_fm, fused_midpoint, fused_cfm = _coordinates(fused_probability, xy)

        for name, value in (
            ("image_spatial_logits", image_logits),
            ("prior_spatial_logits", prior_logits),
            ("fused_spatial_logits", fused_logits),
            ("projection_invalid_probability", projection_invalid_probability),
            ("prior_fm_px", prior_fm),
            ("image_fm_px", image_fm),
            ("fused_fm_px", fused_fm),
            ("previous_query", previous),
            ("decoded_query", decoded),
            ("next_query", next_query),
        ):
            _finite(name, value)

        return V171Output(
            prior_spatial_logits=prior_logits,
            image_spatial_logits=image_logits,
            fused_spatial_logits=fused_logits,
            projection_invalid_logit=projection_invalid_logit,
            image_categorical_logits=image_categorical_logits,
            prior_spatial_probability=prior_probability,
            image_spatial_probability=image_probability,
            fused_spatial_probability=fused_probability,
            image_categorical_probability=image_categorical_probability,
            projection_invalid_probability=projection_invalid_probability,
            prior_fm_px=prior_fm,
            image_fm_px=image_fm,
            fused_fm_px=fused_fm,
            prior_midpoint_px=prior_midpoint,
            image_midpoint_px=image_midpoint,
            fused_midpoint_px=fused_midpoint,
            prior_cfm_px=prior_cfm,
            image_cfm_px=image_cfm,
            fused_cfm_px=fused_cfm,
            previous_query=previous,
            decoded_query=decoded,
            next_query=next_query,
            native_v12_tokens=detached_native,
            view_valid=views,
        )


def v171_image_categorical_nll(
    output: V171Output,
    target_fm_px: torch.Tensor,
    target_projection_valid: torch.Tensor,
    target_valid: torch.Tensor | None = None,
) -> V171ImageCategoricalLoss:
    """Image-only bilinear NLL over 56x56 plus projection-invalid.

    The fixed nominal prior and the fused logits are intentionally absent from
    this objective.  They cannot receive optimization pressure through it.
    """

    logits = output.image_categorical_logits
    batch = len(logits)
    device, dtype = logits.device, logits.dtype
    target = torch.as_tensor(target_fm_px, device=device).detach().to(dtype)
    if target.shape != (batch, VIEWS, TIPS, 2):
        raise V171ContractError("target_fm_px must be [B,2,2,2]")
    _finite("target_fm_px", target)
    in_projection = _boolean(
        "target_projection_valid",
        target_projection_valid,
        shape=(batch, VIEWS, TIPS),
        device=device,
    )
    active = output.view_valid[:, :, None].expand(-1, -1, TIPS)
    if target_valid is not None:
        active = active & _boolean(
            "target_valid",
            target_valid,
            shape=(batch, VIEWS, TIPS),
            device=device,
        )
    outside = ((target < 0.0) | (target >= IMAGE_SIZE)).any(dim=-1)
    if bool((outside & active & in_projection).any()):
        raise V171ContractError("active in-projection target lies outside the image")

    grid_xy = target * ((VISUAL_GRID - 1.0) / (IMAGE_SIZE - 1.0))
    lower = torch.floor(grid_xy).long().clamp(0, VISUAL_GRID - 1)
    upper = (lower + 1).clamp(0, VISUAL_GRID - 1)
    fraction = (grid_xy - lower.to(dtype)).clamp(0.0, 1.0)
    x0, y0 = lower.unbind(dim=-1)
    x1, y1 = upper.unbind(dim=-1)
    fx, fy = fraction.unbind(dim=-1)
    log_probability = F.log_softmax(logits, dim=-1)

    def log_at(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        index = y * VISUAL_GRID + x
        return torch.gather(log_probability, -1, index[..., None]).squeeze(-1)

    spatial_log_likelihood = (
        (1.0 - fx) * (1.0 - fy) * log_at(x0, y0)
        + fx * (1.0 - fy) * log_at(x1, y0)
        + (1.0 - fx) * fy * log_at(x0, y1)
        + fx * fy * log_at(x1, y1)
    )
    invalid_log_likelihood = log_probability[..., PROJECTION_INVALID_INDEX]
    selected = torch.where(
        in_projection, spatial_log_likelihood, invalid_log_likelihood
    )
    in_projection_active = active & in_projection
    invalid_active = active & ~in_projection

    def masked_negative_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any()):
            return logits.sum() * 0.0
        return -value.masked_select(mask).mean()

    return V171ImageCategoricalLoss(
        total=masked_negative_mean(selected, active),
        in_projection=masked_negative_mean(
            spatial_log_likelihood, in_projection_active
        ),
        projection_invalid=masked_negative_mean(
            invalid_log_likelihood, invalid_active
        ),
        active_tips=active.sum(),
        in_projection_tips=in_projection_active.sum(),
        projection_invalid_tips=invalid_active.sum(),
    )


__all__ = [
    "DECODER_FFN",
    "DECODER_HEADS",
    "DECODER_LAYERS",
    "IMAGE_SIZE",
    "MODEL_DIMENSION",
    "NOMINAL_PRIOR_SIGMA_PX",
    "PROJECTION_INVALID_INDEX",
    "SEMANTIC_GRID",
    "TIPS",
    "VIEWS",
    "VISUAL_CELLS",
    "VISUAL_GRID",
    "V171ContractError",
    "V171FactorizedQueryFCPG",
    "V171ImageCategoricalLoss",
    "V171Output",
    "v171_image_categorical_nll",
]
