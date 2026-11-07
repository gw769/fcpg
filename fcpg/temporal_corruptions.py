"""Causal, geometry-known corruptions for temporal grounding.

No function in this module estimates motion from images.  Temporal blur only
mixes the current frame with already observed frames, while jitter uses a
sampled affine trajectory whose exact transform is returned for equivariance.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def causal_trailing_blur(
    images: torch.Tensor,
    *,
    frames: int,
    decay: float = 0.72,
) -> torch.Tensor:
    """Mix each frame with at most ``frames-1`` past frames, never the future."""

    if images.ndim != 6:
        raise ValueError("images must have shape [B,T,V,C,H,W]")
    if frames not in {3, 5}:
        raise ValueError("causal blur frames must be 3 or 5")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must lie in (0,1]")
    time = images.shape[1]
    weights = images.new_zeros((time, time))
    for target in range(time):
        start = max(0, target - frames + 1)
        source = torch.arange(start, target + 1, device=images.device)
        age = target - source
        row = decay**age
        weights[target, start : target + 1] = row / row.sum()
    return torch.einsum("ts,bsvchw->btvchw", weights, images)


def _pixel_affine_inverse_grid(
    affine: torch.Tensor,
    *,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if affine.shape[-2:] != (2, 3):
        raise ValueError("affine must end in [2,3]")
    compute_dtype = (
        torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    )
    full = torch.zeros(
        *affine.shape[:-2], 3, 3, device=affine.device, dtype=compute_dtype
    )
    full[..., :2, :] = affine.to(dtype=compute_dtype)
    full[..., 2, 2] = 1.0
    inverse = torch.linalg.inv(full)
    y, x = torch.meshgrid(
        torch.arange(height, device=affine.device, dtype=compute_dtype),
        torch.arange(width, device=affine.device, dtype=compute_dtype),
        indexing="ij",
    )
    target = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    source = torch.einsum("...ij,hwj->...hwi", inverse, target)
    x_normalized = (2.0 * source[..., 0] + 1.0) / width - 1.0
    y_normalized = (2.0 * source[..., 1] + 1.0) / height - 1.0
    return torch.stack((x_normalized, y_normalized), dim=-1)


def warp_by_pixel_affine(
    value: torch.Tensor,
    source_to_target: torch.Tensor,
    *,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Warp ``[...,C,H,W]`` by an exact source-pixel to target-pixel affine."""

    if value.ndim < 4:
        raise ValueError("value must end in [C,H,W]")
    if source_to_target.shape != (*value.shape[:-3], 2, 3):
        raise ValueError("affine leading dimensions must match value")
    if not torch.is_floating_point(value):
        raise TypeError("affine warping requires a floating-point tensor")
    height, width = value.shape[-2:]
    flat_value = value.reshape(-1, *value.shape[-3:])
    flat_affine = source_to_target.reshape(-1, 2, 3)
    original_dtype = value.dtype
    compute_dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )
    # linalg.inv and CUDA grid_sample do not support every low-precision
    # autocast combination.  Geometry is computed in fp32 and cast back while
    # retaining gradients through the corruption-equivariance objective.
    with torch.autocast(device_type=value.device.type, enabled=False):
        grid = _pixel_affine_inverse_grid(
            flat_affine.to(dtype=compute_dtype),
            height=height,
            width=width,
            dtype=compute_dtype,
        )
        warped = F.grid_sample(
            flat_value.to(dtype=compute_dtype),
            grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=False,
        )
    return warped.to(dtype=original_dtype).reshape_as(value)


def sample_smooth_jitter(
    images: torch.Tensor,
    *,
    max_translation_px: float = 12.0,
    max_rotation_deg: float = 3.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a smooth per-view camera trajectory and return exact affines."""

    if images.ndim != 6:
        raise ValueError("images must have shape [B,T,V,C,H,W]")
    if max_translation_px < 0 or max_rotation_deg < 0:
        raise ValueError("jitter limits cannot be negative")
    batch, time, views, _, height, width = images.shape
    sample_shape = (batch, views, time)
    translation_step = torch.randn(
        *sample_shape,
        2,
        device=images.device,
        dtype=images.dtype,
        generator=generator,
    )
    rotation_step = torch.randn(
        *sample_shape,
        device=images.device,
        dtype=images.dtype,
        generator=generator,
    )
    translation = translation_step.cumsum(dim=2)
    rotation = rotation_step.cumsum(dim=2)
    translation = F.avg_pool1d(
        translation.permute(0, 1, 3, 2).reshape(batch * views, 2, time),
        kernel_size=5,
        stride=1,
        padding=2,
    ).reshape(batch, views, 2, time).permute(0, 1, 3, 2)
    rotation = F.avg_pool1d(
        rotation.reshape(batch * views, 1, time),
        kernel_size=5,
        stride=1,
        padding=2,
    ).reshape(batch, views, time)
    translation_scale = translation.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    rotation_scale = rotation.abs().amax(dim=2, keepdim=True).clamp_min(1e-6)
    translation = translation / translation_scale * float(max_translation_px)
    rotation = rotation / rotation_scale * math.radians(float(max_rotation_deg))

    cosine, sine = rotation.cos(), rotation.sin()
    affine = images.new_zeros((batch, views, time, 2, 3))
    affine[..., 0, 0] = cosine
    affine[..., 0, 1] = -sine
    affine[..., 1, 0] = sine
    affine[..., 1, 1] = cosine
    center = images.new_tensor(((width - 1.0) / 2.0, (height - 1.0) / 2.0))
    rotated_center = torch.einsum("...ij,j->...i", affine[..., :2], center)
    affine[..., :, 2] = center - rotated_center + translation
    affine = affine.permute(0, 2, 1, 3, 4).contiguous()
    return warp_by_pixel_affine(images, affine), affine


def image_affine_to_grid_affine(
    source_to_target: torch.Tensor,
    *,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
) -> torch.Tensor:
    """Conjugate an image-pixel affine into patch-grid pixel coordinates."""

    if source_to_target.shape[-2:] != (2, 3):
        raise ValueError("affine must end in [2,3]")
    image_width, image_height = image_size
    grid_width, grid_height = grid_size
    if min(image_width, image_height, grid_width, grid_height) <= 0:
        raise ValueError("image and grid dimensions must be positive")
    original_dtype, device = source_to_target.dtype, source_to_target.device
    dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )
    full = torch.zeros(*source_to_target.shape[:-2], 3, 3, dtype=dtype, device=device)
    full[..., :2, :] = source_to_target.to(dtype=dtype)
    full[..., 2, 2] = 1.0
    scale_x, scale_y = image_width / grid_width, image_height / grid_height
    grid_to_image = torch.tensor(
        [[scale_x, 0.0, (scale_x - 1.0) / 2.0],
         [0.0, scale_y, (scale_y - 1.0) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    image_to_grid = torch.linalg.inv(grid_to_image)
    converted = image_to_grid @ full @ grid_to_image
    return converted[..., :2, :]


def warp_endpoint_heatmaps(
    heatmaps: torch.Tensor,
    source_to_target: torch.Tensor,
    *,
    endpoint_indices: tuple[int, int] = (10, 15),
    image_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Warp ``[B,2,V,2,G,G]`` targets by endpoint jitter transforms."""

    if heatmaps.ndim != 6 or heatmaps.shape[1] != 2 or heatmaps.shape[3] != 2:
        raise ValueError("heatmaps must have shape [B,2,V,2,G,G]")
    if source_to_target.shape[:3] != (heatmaps.shape[0], 16, heatmaps.shape[2]):
        raise ValueError("jitter affine must have shape [B,16,V,2,3]")
    selected = source_to_target[:, endpoint_indices]
    grid_affine = image_affine_to_grid_affine(
        selected,
        image_size=image_size,
        grid_size=(heatmaps.shape[-1], heatmaps.shape[-2]),
    )
    expanded = grid_affine[:, :, :, None].expand(-1, -1, -1, 2, -1, -1)
    warped = warp_by_pixel_affine(heatmaps.unsqueeze(-3), expanded, padding_mode="zeros")
    return warped.squeeze(-3)
