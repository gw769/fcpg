"""Frozen-feature full-image fingertip localization for SO-101."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fcpg.modules import sample_feature_map_at_pixels


class SO101GlobalTipHeatmapHead(nn.Module):
    """Small camera-conditioned head over a frozen full-image feature map."""

    def __init__(
        self,
        *,
        visual_dim: int = 192,
        camera_context_dim: int = 5,
        camera_embedding_dim: int = 16,
        hidden_dim: int = 128,
        num_anchors: int = 3,
    ) -> None:
        super().__init__()
        if min(
            visual_dim,
            camera_context_dim,
            camera_embedding_dim,
            hidden_dim,
            num_anchors,
        ) <= 0:
            raise ValueError("heatmap-head dimensions must be positive")
        self.visual_dim = int(visual_dim)
        self.camera_context_dim = int(camera_context_dim)
        self.camera_embedding_dim = int(camera_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_anchors = int(num_anchors)
        self.camera_embedding = nn.Sequential(
            nn.Linear(self.camera_context_dim, self.camera_embedding_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(
                self.visual_dim + self.camera_embedding_dim,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim // 2, self.num_anchors, kernel_size=1),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        camera_context: torch.Tensor,
    ) -> torch.Tensor:
        if feature_map.ndim != 4 or feature_map.shape[1] != self.visual_dim:
            raise ValueError(
                f"feature_map must have shape [B,{self.visual_dim},H,W]"
            )
        if camera_context.shape != (
            feature_map.shape[0],
            self.camera_context_dim,
        ):
            raise ValueError(
                "camera_context must have shape "
                f"[B,{self.camera_context_dim}]"
            )
        camera = self.camera_embedding(camera_context).unsqueeze(-1).unsqueeze(-1)
        camera = camera.expand(-1, -1, feature_map.shape[-2], feature_map.shape[-1])
        return self.head(torch.cat((feature_map, camera), dim=1))


def load_global_tip_heatmap_head(
    path: Path,
    device: torch.device,
) -> tuple[SO101GlobalTipHeatmapHead, dict[str, Any]]:
    """Load a markerless fingertip head and its small runtime contract."""

    source = path.expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("format") != "so101-global-tip-heatmap-v1":
        raise ValueError("unsupported global heatmap checkpoint")
    head_config = payload.get("head_config")
    if not isinstance(head_config, dict):
        raise ValueError("global heatmap checkpoint has no head configuration")
    head = SO101GlobalTipHeatmapHead(**head_config).to(device)
    head.load_state_dict(payload["head"], strict=True)
    head.requires_grad_(False)
    head.eval()
    return head, {
        "path": source,
        "step": int(payload.get("step", -1)),
        "training_metrics": payload.get("metrics", {}),
    }


def global_tip_sequence_diagnostics(
    points_px: np.ndarray,
    confidence: np.ndarray,
    nominal_px: np.ndarray,
    *,
    image_size: tuple[int, int],
    max_tip_separation_px: float,
    minimum_tip_confidence: float = 0.08,
    edge_margin_px: float = 2.0,
    max_motion_disagreement_px: float = 30.0,
    content_rect: tuple[int, int, int, int] | None = None,
    max_tip_offset_disagreement_ratio: float = 0.5,
    separation_floor_px: float = 8.0,
    max_nominal_disagreement_px: float | None = None,
) -> dict[str, Any]:
    """Decide whether a three-frame coarse localization can seed local FCPG.

    Confidence alone is not a reliable failure detector.  The gate also checks
    visibility, fingertip geometry, whether measured motion is compatible with
    the short-horizon FK motion, and how far the measurement sits from the FK
    projection it is allowed to replace.  It never alters the predicted points.

    ``content_rect`` is ``(left, top, right, bottom)`` with right/bottom
    exclusive, naming the sub-rectangle of the canvas that carries real camera
    pixels.  The visibility gate asks "is this tip about to leave the CAMERA
    FRAME", which under a letterbox canvas is the content rect, not the canvas:
    a tip one row inside the pad band is 28 rows from the canvas edge but 1 row
    from the last pixel any camera actually saw.  Omitting it keeps the
    historical canvas-edge arithmetic exactly, because the default rect derived
    from ``image_size`` reduces the expression term-for-term.
    """

    points = np.asarray(points_px, dtype=np.float32)
    scores = np.asarray(confidence, dtype=np.float32)
    nominal = np.asarray(nominal_px, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (3, 2):
        raise ValueError("points_px must have shape [T,3,2]")
    if scores.shape != points.shape[:2]:
        raise ValueError("confidence must have shape [T,3]")
    if nominal.shape != points.shape:
        raise ValueError("nominal_px must match points_px")
    if min(
        image_size[0],
        image_size[1],
        max_tip_separation_px,
        max_motion_disagreement_px,
    ) <= 0:
        raise ValueError("image and geometry limits must be positive")
    if minimum_tip_confidence < 0 or edge_margin_px < 0:
        raise ValueError("confidence and edge margin must be non-negative")
    if max_tip_offset_disagreement_ratio <= 0 or separation_floor_px <= 0:
        raise ValueError("tip geometry tolerances must be positive")
    if max_nominal_disagreement_px is not None and (
        not np.isfinite(max_nominal_disagreement_px)
        or max_nominal_disagreement_px <= 0
    ):
        raise ValueError("max_nominal_disagreement_px must be finite and positive")

    finite = bool(
        np.all(np.isfinite(points))
        and np.all(np.isfinite(scores))
        and np.all(np.isfinite(nominal))
    )
    width, height = map(float, image_size)
    # Default to the full canvas so an omitted `content_rect` reproduces the
    # historical expression term-for-term: left=0 and top=0 make the first two
    # terms bare coordinates, and right/bottom being exclusive makes the last
    # valid pixel `right-1`/`bottom-1`, i.e. exactly `width-1`/`height-1`.
    if content_rect is None:
        left, top, right, bottom = 0.0, 0.0, width, height
    else:
        left, top, right, bottom = map(float, content_rect)
        if right - left < 1.0 or bottom - top < 1.0:
            raise ValueError("content_rect must enclose at least one pixel")
    edge_distance = np.minimum.reduce(
        (
            points[..., 0] - left,
            right - 1.0 - points[..., 0],
            points[..., 1] - top,
            bottom - 1.0 - points[..., 1],
        )
    )
    minimum_edge_distance = float(np.min(edge_distance)) if finite else -np.inf
    tip_confidence = scores[:, 1:]
    minimum_observed_confidence = (
        float(np.min(tip_confidence)) if finite else -np.inf
    )
    tip_separation = np.linalg.norm(points[:, 1] - points[:, 2], axis=-1)
    maximum_observed_separation = (
        float(np.max(tip_separation)) if finite else np.inf
    )
    minimum_observed_separation = float(np.min(tip_separation)) if finite else 0.0
    # The separation bound above is one-sided, so a locator that puts both jaws
    # on one patch passes every historical check: zero separation is never
    # "too far apart", both tips sit far from the image edge, and a frozen pair
    # disagrees with no motion.  Compare the fixed-to-moving vector against the
    # one FK plus the projection fit predict instead.  One normalized ratio
    # covers both failure shapes, because a collapsed pair disagrees by a whole
    # expected separation and a mis-oriented pair by more.
    observed_offset = points[:, 2] - points[:, 1]
    expected_offset = nominal[:, 2] - nominal[:, 1]
    expected_separation = np.linalg.norm(expected_offset, axis=-1)
    offset_disagreement = np.linalg.norm(
        observed_offset - expected_offset, axis=-1
    ) / np.maximum(expected_separation, float(separation_floor_px))
    # Below the floor FK itself reports the jaws closed to within one patch, so
    # a collapsed measurement is the correct answer rather than a failure.
    resolvable = expected_separation >= float(separation_floor_px)
    maximum_offset_disagreement = (
        float(np.max(offset_disagreement[resolvable]))
        if finite and bool(np.any(resolvable))
        else 0.0
    )
    if len(points) > 1 and finite:
        measured_motion = np.diff(points[:, 1:], axis=0)
        nominal_motion = np.diff(nominal[:, 1:], axis=0)
        motion_disagreement = np.linalg.norm(
            measured_motion - nominal_motion, axis=-1
        )
        maximum_motion_disagreement = float(np.max(motion_disagreement))
    else:
        maximum_motion_disagreement = 0.0

    # Every check above is invariant to translating the whole triple: shape
    # compares tip-to-tip offsets, motion compares frame differences, and a
    # triple that is consistently wrong across all three frames agrees with FK
    # on both.  So a heatmap that locks onto three background patches and holds
    # them passes every check while placing the gripper anywhere in the image.
    # Measured over the last 120 rounds of the 2026-08-15 column session: the
    # accepted locator put the jaws 74-156 px from the FK projection on 28 of
    # them (rounds 18-31, 72-75, 92-114, anchors near (161-180, 95-120) against
    # an FK projection near (85, 160)), while every other round sat at 6-10 px.
    # The distribution is bimodal -- median 8.5 px, p90 111.7 px -- with no
    # accepted round anywhere between 10 px and 74 px.
    nominal_disagreement = np.linalg.norm(points - nominal, axis=-1)
    maximum_nominal_disagreement = (
        float(np.max(nominal_disagreement)) if finite else np.inf
    )

    reasons: list[str] = []
    if not finite:
        reasons.append("non_finite")
    if minimum_edge_distance < edge_margin_px:
        reasons.append("near_image_edge")
    if minimum_observed_confidence < minimum_tip_confidence:
        reasons.append("low_tip_confidence")
    if maximum_observed_separation > max_tip_separation_px:
        reasons.append("tip_separation")
    if maximum_offset_disagreement > max_tip_offset_disagreement_ratio:
        reasons.append("tip_offset_disagreement")
    if maximum_motion_disagreement > max_motion_disagreement_px:
        reasons.append("motion_disagreement")
    if (
        max_nominal_disagreement_px is not None
        and maximum_nominal_disagreement > max_nominal_disagreement_px
    ):
        reasons.append("nominal_disagreement")
    return {
        "accepted": not reasons,
        "rejection_reasons": tuple(reasons),
        "minimum_tip_confidence": minimum_observed_confidence,
        "minimum_edge_distance_px": minimum_edge_distance,
        "maximum_tip_separation_px": maximum_observed_separation,
        "minimum_tip_separation_px": minimum_observed_separation,
        "expected_tip_separation_px": (
            float(expected_separation[-1]) if finite else np.inf
        ),
        "maximum_tip_offset_disagreement_ratio": maximum_offset_disagreement,
        "maximum_motion_disagreement_px": maximum_motion_disagreement,
        "maximum_nominal_disagreement_px": maximum_nominal_disagreement,
    }


def feature_patch_centers(
    feature_height: int,
    feature_width: int,
    *,
    image_size: tuple[int, int],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return image-space centers for a rectangular feature grid."""

    if feature_height <= 0 or feature_width <= 0:
        raise ValueError("feature dimensions must be positive")
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    xs = (torch.arange(feature_width, device=device, dtype=dtype) + 0.5) * (
        float(image_width) / float(feature_width)
    )
    ys = (torch.arange(feature_height, device=device, dtype=dtype) + 0.5) * (
        float(image_height) / float(feature_height)
    )
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


def centered_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    """Remove per-image scene bias while preserving local patch descriptors."""

    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape [B,C,H,W]")
    return feature_map - feature_map.mean(dim=(-2, -1), keepdim=True)


def extract_anchor_prototypes(
    feature_map: torch.Tensor,
    anchor_points_px: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Sample normalized anchor descriptors from scene-centered feature maps."""

    if anchor_points_px.ndim != 3 or anchor_points_px.shape[-1] != 2:
        raise ValueError("anchor_points_px must have shape [B,A,2]")
    if feature_map.shape[0] != anchor_points_px.shape[0]:
        raise ValueError("feature and anchor batches must match")
    descriptors = sample_feature_map_at_pixels(
        centered_feature_map(feature_map),
        anchor_points_px,
        image_size=image_size,
    )
    return F.normalize(descriptors, dim=-1, eps=1.0e-6)


def prototype_similarity_maps(
    feature_map: torch.Tensor,
    prototype_bank: torch.Tensor,
) -> torch.Tensor:
    """Score every patch against a bank of per-anchor prototypes.

    Args:
        feature_map: Query features with shape ``[B,C,H,W]``.
        prototype_bank: Features with shape ``[P,A,C]``.

    Returns:
        Maximum cosine similarity with shape ``[B,A,H,W]``.
    """

    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape [B,C,H,W]")
    if prototype_bank.ndim != 3:
        raise ValueError("prototype_bank must have shape [P,A,C]")
    if feature_map.shape[1] != prototype_bank.shape[-1]:
        raise ValueError("feature and prototype channels must match")
    if prototype_bank.shape[0] == 0 or prototype_bank.shape[1] == 0:
        raise ValueError("prototype_bank must be non-empty")

    query = F.normalize(centered_feature_map(feature_map), dim=1, eps=1.0e-6)
    prototypes = F.normalize(prototype_bank, dim=-1, eps=1.0e-6)
    similarities = torch.einsum("bchw,pac->bpahw", query, prototypes)
    return similarities.amax(dim=1)


def localize_similarity_peaks(
    similarity_maps: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert each per-anchor similarity peak to an image-space coordinate."""

    if similarity_maps.ndim != 4:
        raise ValueError("similarity_maps must have shape [B,A,H,W]")
    batch, anchors, height, width = similarity_maps.shape
    flat = similarity_maps.reshape(batch, anchors, height * width)
    confidence, indices = flat.max(dim=-1)
    centers = feature_patch_centers(
        height,
        width,
        image_size=image_size,
        device=similarity_maps.device,
        dtype=similarity_maps.dtype,
    ).reshape(height * width, 2)
    return centers[indices], confidence


def derive_center_from_fingertips(points_px: torch.Tensor) -> torch.Tensor:
    """Replace anchor zero with the midpoint of fixed and moving fingertips."""

    if points_px.ndim < 2 or points_px.shape[-2:] != (3, 2):
        raise ValueError("points_px must end with shape [3,2]")
    output = points_px.clone()
    output[..., 0, :] = output[..., 1:, :].mean(dim=-2)
    return output


def gaussian_heatmap_targets(
    points_px: torch.Tensor,
    *,
    feature_size: tuple[int, int],
    image_size: tuple[int, int],
    sigma_patches: float = 0.85,
) -> torch.Tensor:
    """Build normalized Gaussian targets on the feature grid."""

    if points_px.ndim != 3 or points_px.shape[-1] != 2:
        raise ValueError("points_px must have shape [B,A,2]")
    feature_width, feature_height = feature_size
    if feature_width <= 0 or feature_height <= 0 or sigma_patches <= 0:
        raise ValueError("feature dimensions and sigma must be positive")
    centers = feature_patch_centers(
        feature_height,
        feature_width,
        image_size=image_size,
        device=points_px.device,
        dtype=points_px.dtype,
    )
    patch_width = float(image_size[0]) / float(feature_width)
    patch_height = float(image_size[1]) / float(feature_height)
    delta_x = (centers[..., 0] - points_px[..., 0, None, None]) / patch_width
    delta_y = (centers[..., 1] - points_px[..., 1, None, None]) / patch_height
    logits = -0.5 * (delta_x.square() + delta_y.square()) / (sigma_patches**2)
    flat = logits.flatten(start_dim=-2)
    return torch.softmax(flat, dim=-1).reshape_as(logits)


def heatmap_distribution_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy against soft heatmap targets with an anchor mask."""

    if logits.shape != targets.shape or logits.ndim != 4:
        raise ValueError("logits and targets must share shape [B,A,H,W]")
    if valid.shape != logits.shape[:2]:
        raise ValueError("valid must have shape [B,A]")
    log_probabilities = torch.log_softmax(logits.flatten(start_dim=-2), dim=-1)
    losses = -(targets.flatten(start_dim=-2) * log_probabilities).sum(dim=-1)
    weights = valid.to(dtype=losses.dtype)
    denominator = weights.sum()
    if float(denominator.detach().cpu()) <= 0:
        raise ValueError("heatmap batch contains no valid anchors")
    return (losses * weights).sum() / denominator


def decode_heatmap_peaks(
    logits: torch.Tensor,
    *,
    image_size: tuple[int, int],
    local_radius: int = 1,
    temperature: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a sub-patch point using a local soft argmax around each peak."""

    if logits.ndim != 4:
        raise ValueError("logits must have shape [B,A,H,W]")
    if local_radius < 0 or temperature <= 0:
        raise ValueError("local_radius must be non-negative and temperature positive")
    batch, anchors, height, width = logits.shape
    flat = logits.reshape(batch, anchors, height * width)
    probabilities = torch.softmax(flat, dim=-1)
    _, peak_indices = probabilities.max(dim=-1)
    peak_y = torch.div(peak_indices, width, rounding_mode="floor")
    peak_x = peak_indices.remainder(width)
    centers = feature_patch_centers(
        height,
        width,
        image_size=image_size,
        device=logits.device,
        dtype=logits.dtype,
    )
    decoded = torch.empty(batch, anchors, 2, device=logits.device, dtype=logits.dtype)
    confidence = torch.empty(batch, anchors, device=logits.device, dtype=logits.dtype)
    probability_maps = probabilities.reshape(batch, anchors, height, width)
    for batch_index in range(batch):
        for anchor_index in range(anchors):
            center_y = int(peak_y[batch_index, anchor_index])
            center_x = int(peak_x[batch_index, anchor_index])
            y0 = max(0, center_y - local_radius)
            y1 = min(height, center_y + local_radius + 1)
            x0 = max(0, center_x - local_radius)
            x1 = min(width, center_x + local_radius + 1)
            local_logits = logits[batch_index, anchor_index, y0:y1, x0:x1]
            confidence[batch_index, anchor_index] = probability_maps[
                batch_index, anchor_index, y0:y1, x0:x1
            ].sum()
            local_weights = torch.softmax(
                local_logits.reshape(-1) / temperature, dim=0
            )
            local_centers = centers[y0:y1, x0:x1].reshape(-1, 2)
            decoded[batch_index, anchor_index] = (
                local_centers * local_weights.unsqueeze(-1)
            ).sum(dim=0)
    return decoded, confidence


def decode_structured_tip_heatmaps(
    logits: torch.Tensor,
    *,
    image_size: tuple[int, int],
    max_tip_separation_px: float,
    candidates_per_tip: int = 24,
    center_weight: float = 1.0,
    expected_tip_offset_px: torch.Tensor | None = None,
    geometry_weight: float = 4.0,
    max_offset_disagreement_ratio: float = 0.5,
    separation_floor_px: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode two coherent fingertips and derive their exact midpoint.

    Independent heatmap maxima can land on unrelated white objects.  This
    decoder searches high-scoring fingertip pairs, rejects physically
    impossible separations, and scores the pair midpoint with the center
    heatmap before returning ``[center, fixed, moving]``.

    Heatmap evidence alone has a degenerate optimum.  A pair that puts both
    fingertips on ONE patch maximizes the fixed and moving terms together, and
    its midpoint lands on that same patch, so it collects the center term as
    well; a collapsed gripper therefore outscores the true separated pair
    whenever the three channels peak near each other.  ``expected_tip_offset_px``
    supplies the fixed-to-moving vector that FK plus the projection fit predict
    for the current joint state, which breaks that tie with pose information the
    heatmaps do not carry.  The prior constrains only the SHAPE of the pair --
    how far apart the jaws are and along which axis -- and never its absolute
    position, because relocating a stale nominal is precisely this locator's
    job.  Omit it to keep the historical evidence-only behaviour.
    """

    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("logits must have shape [B,3,H,W]")
    if max_tip_separation_px <= 0 or candidates_per_tip <= 0 or center_weight < 0:
        raise ValueError("structured decoder arguments are invalid")
    if (
        geometry_weight < 0
        or max_offset_disagreement_ratio <= 0
        or separation_floor_px <= 0
    ):
        raise ValueError("structured decoder geometry arguments are invalid")
    batch, _, height, width = logits.shape
    cells = height * width
    candidates = min(candidates_per_tip, cells)
    log_probabilities = torch.log_softmax(logits.flatten(start_dim=-2), dim=-1)
    centers = feature_patch_centers(
        height,
        width,
        image_size=image_size,
        device=logits.device,
        dtype=logits.dtype,
    ).reshape(cells, 2)
    fixed_scores, fixed_indices = log_probabilities[:, 1].topk(candidates, dim=-1)
    moving_scores, moving_indices = log_probabilities[:, 2].topk(candidates, dim=-1)
    fixed_points = centers[fixed_indices]
    moving_points = centers[moving_indices]
    pair_midpoints = (
        fixed_points.unsqueeze(2) + moving_points.unsqueeze(1)
    ) * 0.5
    separations = torch.linalg.vector_norm(
        fixed_points.unsqueeze(2) - moving_points.unsqueeze(1), dim=-1
    )

    image_width, image_height = image_size
    midpoint_x = torch.clamp(
        torch.floor(pair_midpoints[..., 0] * width / float(image_width)).long(),
        0,
        width - 1,
    )
    midpoint_y = torch.clamp(
        torch.floor(pair_midpoints[..., 1] * height / float(image_height)).long(),
        0,
        height - 1,
    )
    midpoint_indices = midpoint_y * width + midpoint_x
    center_scores = torch.gather(
        log_probabilities[:, 0].unsqueeze(1).expand(-1, candidates, -1),
        2,
        midpoint_indices,
    )
    pair_scores = (
        fixed_scores.unsqueeze(2)
        + moving_scores.unsqueeze(1)
        + center_weight * center_scores
    )
    pair_scores = pair_scores.masked_fill(
        separations > float(max_tip_separation_px),
        -torch.inf,
    )
    if expected_tip_offset_px is not None:
        expected = torch.as_tensor(
            expected_tip_offset_px, dtype=logits.dtype, device=logits.device
        )
        if expected.ndim == 1:
            expected = expected.unsqueeze(0).expand(batch, -1)
        if expected.shape != (batch, 2) or not bool(torch.all(torch.isfinite(expected))):
            raise ValueError(
                "expected_tip_offset_px must be a finite fixed-to-moving vector "
                "with shape [2] or [B,2]"
            )
        expected_separation = torch.linalg.vector_norm(expected, dim=-1)
        offsets = moving_points.unsqueeze(1) - fixed_points.unsqueeze(2)
        disagreement = torch.linalg.vector_norm(
            offsets - expected[:, None, None, :], dim=-1
        )
        # Normalizing by the expected separation makes one ratio cover both
        # failure shapes: a collapsed pair disagrees by a full separation, and a
        # pair whose jaw axis is rotated away from FK disagrees by more still.
        scale = expected_separation.clamp_min(float(separation_floor_px))
        ratio = disagreement / scale[:, None, None]
        # Growing quadratically against the rejection boundary keeps the prior
        # cheap among geometrically plausible pairs while making it cost
        # ``geometry_weight`` nats right at the boundary, so a small heatmap
        # advantage cannot buy a partly collapsed pair -- which is the failure
        # a flat penalty still admits.
        pair_scores = pair_scores - geometry_weight * (
            ratio / float(max_offset_disagreement_ratio)
        ) ** 2
        # A gripper FK reports as open cannot project both jaws onto one patch.
        # Below the floor the jaws are closed to within a patch, so the pair is
        # genuinely unresolvable and the historical behaviour is kept.
        resolvable = (expected_separation >= float(separation_floor_px))[:, None, None]
        pair_scores = pair_scores.masked_fill(
            resolvable & (ratio > float(max_offset_disagreement_ratio)),
            -torch.inf,
        )
    best_flat = pair_scores.flatten(start_dim=1).argmax(dim=-1)
    best_fixed = torch.div(best_flat, candidates, rounding_mode="floor")
    best_moving = best_flat.remainder(candidates)
    batch_indices = torch.arange(batch, device=logits.device)
    fixed = fixed_points[batch_indices, best_fixed]
    moving = moving_points[batch_indices, best_moving]
    points = torch.stack(((fixed + moving) * 0.5, fixed, moving), dim=1)

    probabilities = torch.softmax(logits.flatten(start_dim=-2), dim=-1)
    fixed_confidence = probabilities[:, 1].gather(
        1, fixed_indices[batch_indices, best_fixed].unsqueeze(-1)
    ).squeeze(-1)
    moving_confidence = probabilities[:, 2].gather(
        1, moving_indices[batch_indices, best_moving].unsqueeze(-1)
    ).squeeze(-1)
    center_confidence = probabilities[:, 0].gather(
        1,
        midpoint_indices[batch_indices, best_fixed, best_moving].unsqueeze(-1),
    ).squeeze(-1)
    confidence = torch.stack(
        (center_confidence, fixed_confidence, moving_confidence), dim=1
    )
    return points, confidence
