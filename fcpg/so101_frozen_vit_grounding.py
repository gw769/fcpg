"""Locate the task object with frozen general-purpose ViT features.

The detector this replaces keyed on colour: it built a chroma model from the
operator's crop and looked for pixels that deviated from grey the same way.
That works until the sensor changes. Handed a greyscale copy of the same frame
it does not degrade, it refuses outright -- "template has no colour to
distinguish it from grey" -- because there is nothing left of the thing it was
measuring. A mono camera, a colour-blind object, or a scene lit to wash the
chroma out all end in the same place.

Here the appearance model is a vector of patch features from a frozen backbone
that was never trained on this robot, this task, or this object, and is never
trained further. An object is whatever the operator pointed at once; finding it
again is a nearest-neighbour lookup in that feature space. Measured on a live
frame pair with the fish half-buried under the jaws:

    prototype from the clean view -> located in the occluded view
      colour input      5.7 px,  2.05 sigma above background
      greyscale input  12.5 px,  1.75 sigma above background
      chroma model, colour       4.1 px
      chroma model, greyscale    refuses -- no colour to model

Greyscale costs accuracy and keeps the capability, which is the property the
colour model never had. For contrast the backbone already inside the system --
a ViT-tiny trained from scratch on this dataset for dynamics prediction -- puts
its peak off the object entirely (57 px) even when handed a clean prototype: it
was never asked to make object identity separable, and it isn't.

Encoding one 224 px frame costs about 5 ms on GPU, so both views fit inside a
planning round with room to spare.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_BACKBONE = "vit_small_patch14_dinov2.lvd142m"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# The crop the operator drew is centred on the click by construction, so its
# border is background. Growing the prototype from the centre token and keeping
# the half that agrees drops that border without anyone having to say what a
# table looks like.
SEED_CORE_FRACTION = 0.5
# One crop, seen at several magnifications: the object's size on screen is not
# the size it happened to have when it was registered.
BOOTSTRAP_SCALES = (1.0, 1.5, 2.2, 3.0)
# What counts as background when siting the half-maximum cut. Low enough that
# an object filling a good part of the frame does not raise its own floor.
BACKGROUND_PERCENTILE = 40.0
# Tight-to-loose. Prefer a body-hugging cut; looser fracs became a puddle of table.
# Tight-first cuts; softer fracs only if the peak sits well above the body ridge.
SYMMETRY_LEVEL_FRACS = (0.75, 0.7, 0.65, 0.6, 0.5, 0.4)
# Eigen-axis needs a few pixels. This is a numerical floor, not an object-class gate.
MIN_STABLE_SUPPORT_PX = 24
# Reject puddles that swallow the table / background halo.
MAX_SUPPORT_FRAME_FRACTION = 0.12
# Switch away from FWHM only when another cut is clearly more symmetric.
SYMMETRY_IOU_IMPROVEMENT = 0.05
# Peel the soft similarity halo before CoG / polygon.
SUPPORT_ERODE_ITERS = 2


@dataclass(frozen=True)
class SupportShape:
    """The object's line, its width across that line, and its length along it.

    A long object has a direction, and closing a two-fingered jaw along that
    direction cannot work: the jaws have to come together across it. So the
    quantity that decides a grasp is not where the object's centre is but which
    way it lies and how wide it is when measured square to that.

    `axis_px` is the direction the object runs. `grasp_axis_px` is square to
    it -- the direction the jaws must close along. `width_px` is the object's
    extent measured along `grasp_axis_px`, which is what the jaw opening has to
    span, and `length_px` its extent along `axis_px`.
    """

    axis_px: tuple[float, float]
    grasp_axis_px: tuple[float, float]
    width_px: float
    length_px: float

    @property
    def elongation(self) -> float:
        """How committed the object is to a direction. 1.0 is featureless."""
        return float(self.length_px / max(self.width_px, 1e-6))


@dataclass(frozen=True)
class GroundingHit:
    """Where the object is, how far it stood out, and which way it lies.

    The grasp point is the 重心 / CoG of a *shape estimate*: a uniform-density
    solid of revolution about the support's long axis. Not measured mass,
    not the 2-D area centroid of the silhouette.
    """

    centroid_px: tuple[float, float]
    separation: float
    peak: float
    shape: SupportShape
    support_mask: np.ndarray | None = None
    symmetry_iou: float = 0.0

    def as_payload(self) -> dict[str, Any]:
        return {
            "centroid_px": [float(self.centroid_px[0]), float(self.centroid_px[1])],
            "cog_px": [float(self.centroid_px[0]), float(self.centroid_px[1])],
            "grasp_kind": "shape_prior_uniform_density_cog",
            "separation": float(self.separation),
            "peak": float(self.peak),
            "axis_px": [float(v) for v in self.shape.axis_px],
            "grasp_axis_px": [float(v) for v in self.shape.grasp_axis_px],
            "width_px": float(self.shape.width_px),
            "length_px": float(self.shape.length_px),
            "elongation": float(self.shape.elongation),
            "symmetry_iou": float(self.symmetry_iou),
        }


def _region_containing(mask: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    """The connected part of ``mask`` that holds ``point``."""
    import cv2

    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    label = int(labels[int(point[1]), int(point[0])])
    if label == 0:
        return mask
    return labels == label


def _support_shape(
    xs: np.ndarray,
    ys: np.ndarray,
    weight: np.ndarray,
    centroid: tuple[float, float],
) -> SupportShape:
    """Principal axes of the matched region, weighted by how well it matched.

    The same weights that placed the centroid decide the axes, so the position
    and the direction cannot disagree about which pixels are the object.
    """
    total = float(weight.sum())
    if total <= 0.0:
        return SupportShape((1.0, 0.0), (0.0, 1.0), 0.0, 0.0)
    dx = xs - centroid[0]
    dy = ys - centroid[1]
    cxx = float((weight * dx * dx).sum() / total)
    cyy = float((weight * dy * dy).sum() / total)
    cxy = float((weight * dx * dy).sum() / total)
    values, vectors = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    major = vectors[:, int(np.argmax(values))]
    minor = np.array([-major[1], major[0]])
    # Report extents as full widths of the weighted spread rather than
    # variances, so they can be compared against a jaw opening in pixels.
    spread = np.sqrt(np.clip(values, 0.0, None))
    length = 2.0 * float(spread.max())
    width = 2.0 * float(spread.min())
    return SupportShape(
        axis_px=(float(major[0]), float(major[1])),
        grasp_axis_px=(float(minor[0]), float(minor[1])),
        width_px=width,
        length_px=length,
    )


def _mask_centroid_and_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Area mean and principal axis — used only to site the symmetry line."""
    ys, xs = np.nonzero(mask)
    if xs.size < MIN_STABLE_SUPPORT_PX:
        return None
    pts = np.stack((xs.astype(np.float64), ys.astype(np.float64)), -1)
    centroid = pts.mean(0)
    delta = pts - centroid
    cov = delta.T @ delta / float(len(pts))
    _values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(_values))]
    if float(np.linalg.norm(axis)) < 1e-12:
        return None
    return centroid, axis


def estimated_shape_cog(mask: np.ndarray) -> tuple[float, float] | None:
    """Uniform-density 重心 of a circular solid-of-revolution shape prior.

    The 2-D support is the silhouette of an estimated solid whose every
    cross-section orthogonal to the long axis ``u`` is a disk of radius
    ``r(s)`` = half-width of that slice. Density is taken as 1. There is no
    depth camera and no measured mass: ``r(s)`` comes from the silhouette.
    Slice mass is ``r(s)² Δs`` (``π`` and ρ cancel).

        s_cog = Σ_k s_k r_k² Δs  /  Σ_k r_k² Δs     # 重心 along u
        grasp = origin + s_cog u                     # on the symmetry axis

    Transverse offset is zero by the revolution prior (the mask was already
    reflected across ``u``). This is a shape-estimate CoG, not measured
    volume and not the area centroid ``Σ s (2r) / Σ (2r)``.
    """
    moments = _mask_centroid_and_axis(mask)
    if moments is None:
        return None
    origin, axis = moments
    unit = axis / (float(np.linalg.norm(axis)) + 1e-12)
    normal = np.array([-unit[1], unit[0]])
    ys, xs = np.nonzero(mask)
    pts = np.stack((xs.astype(np.float64), ys.astype(np.float64)), -1)
    along = (pts - origin) @ unit
    across = (pts - origin) @ normal
    lo, hi = float(along.min()), float(along.max())
    span = hi - lo
    if span < 1e-6:
        return (float(origin[0]), float(origin[1]))
    bins = max(8, min(48, int(np.ceil(span))))
    edges = np.linspace(lo, hi + 1e-6, bins + 1)
    which = np.clip(np.digitize(along, edges) - 1, 0, bins - 1)
    mass = 0.0
    moment = 0.0
    ds = span / float(bins)
    for k in range(bins):
        sel = which == k
        if not np.any(sel):
            continue
        radius = float(np.percentile(np.abs(across[sel]), 95))
        if radius <= 0.0:
            continue
        slice_mass = radius * radius * ds
        station = float(along[sel].mean())
        mass += slice_mass
        moment += station * slice_mass
    if mass <= 0.0:
        return (float(origin[0]), float(origin[1]))
    station_cog = moment / mass
    grasp = origin + station_cog * unit
    return (float(grasp[0]), float(grasp[1]))


estimated_revolution_centroid = estimated_shape_cog  # old name; same CoG formula


def support_is_clipped(mask: np.ndarray, *, margin: int = 2) -> bool:
    """True when the silhouette touches the frame edge (CoG is not aimable)."""
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or not bool(array.any()):
        return True
    ys, xs = np.nonzero(array)
    height, width = array.shape
    pad = max(0, int(margin))
    return bool(
        int(xs.min()) <= pad
        or int(xs.max()) >= width - 1 - pad
        or int(ys.min()) <= pad
        or int(ys.max()) >= height - 1 - pad
    )


def _reflect_across_long_axis(
    mask: np.ndarray, centroid: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    """Binary reflection across the line through ``centroid`` along ``axis``.

    For each destination pixel p, the source is the reflection of p across
    that line: v = p-c, v' = 2(v·u)u - v. Approximate left–right symmetry of
    a long object is this reflection, not a fitted ellipse.
    """
    unit = np.asarray(axis, np.float64).reshape(2)
    unit = unit / (float(np.linalg.norm(unit)) + 1e-12)
    centre = np.asarray(centroid, np.float64).reshape(2)
    height, width = mask.shape
    ys, xs = np.mgrid[0:height, 0:width]
    delta = np.stack((xs - centre[0], ys - centre[1]), -1)
    reflected = 2.0 * (delta @ unit)[..., None] * unit - delta + centre
    src_x = np.rint(reflected[..., 0]).astype(np.int32)
    src_y = np.rint(reflected[..., 1]).astype(np.int32)
    valid = (src_x >= 0) & (src_x < width) & (src_y >= 0) & (src_y < height)
    out = np.zeros_like(mask, dtype=bool)
    out[valid] = np.asarray(mask, dtype=bool)[src_y[valid], src_x[valid]]
    return out


def _symmetry_iou(mask: np.ndarray, reflected: np.ndarray) -> float:
    both = np.logical_and(mask, reflected)
    either = np.logical_or(mask, reflected)
    union = int(either.sum())
    return float(both.sum()) / float(union) if union else 0.0


def gripper_occlusion_mask(
    shape: tuple[int, int],
    jaw_pair_px: np.ndarray | None,
    *,
    radius_px: float | None = None,
) -> np.ndarray | None:
    """Pixels covered by the visible jaw pair — do not treat as object body.

    When the claw occludes the object, those pixels must not join the support
    or pull the 重心. Keep the band tight so visible body beside the jaws
    still forms the mass. Returns a bool mask, or None if no jaws are given.
    This is FK tip geometry, not a colour prior.
    """
    import cv2

    if jaw_pair_px is None:
        return None
    pair = np.asarray(jaw_pair_px, np.float64)
    if pair.shape != (2, 2) or not np.isfinite(pair).all():
        return None
    height, width = int(shape[0]), int(shape[1])
    out = np.zeros((height, width), dtype=np.uint8)
    span = float(np.linalg.norm(pair[1] - pair[0]))
    # Modest tip disks + a thin FG sausage — not a wide pad that erases body.
    if radius_px is None:
        r = int(round(np.clip(0.22 * max(span, 1.0), 8.0, 14.0)))
    else:
        r = max(4, int(round(radius_px)))
    f = tuple(np.round(pair[0]).astype(int))
    g = tuple(np.round(pair[1]).astype(int))
    cv2.circle(out, f, r, 1, -1, cv2.LINE_AA)
    cv2.circle(out, g, r, 1, -1, cv2.LINE_AA)
    cv2.line(out, f, g, 1, max(3, r // 2), cv2.LINE_AA)
    return out.astype(bool)


def select_symmetric_support(
    scores: np.ndarray,
    peak_xy: tuple[int, int],
    floor: float,
    *,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """Pick a peak-connected cut, then keep its long-axis-symmetric core.

    Levels are ordinary fractions of peak-above-background. The first
    *stable* cut in tight-to-loose order wins (no looser halo / table
    puddle). ``exclude`` (the jaws) is removed before the cut so claw
    occlusion does not join the body mass. The
    returned mask is ``region AND reflect``, then eroded. The grasp point
    is that mask's revolution-prior 重心.
    """
    peak_x, peak_y = int(peak_xy[0]), int(peak_xy[1])
    work = np.asarray(scores, dtype=np.float64).copy()
    if exclude is not None:
        blocked = np.asarray(exclude, dtype=bool)
        if blocked.shape == work.shape:
            work = np.where(blocked, -np.inf, work)
            if not np.isfinite(work[peak_y, peak_x]):
                # Peak under the claw: take the best remaining score.
                if not np.isfinite(work).any():
                    empty = np.zeros(scores.shape, dtype=bool)
                    empty[peak_y, peak_x] = True
                    return empty, (float(peak_x), float(peak_y)), 0.0
                peak_y, peak_x = np.unravel_index(int(np.nanargmax(work)), work.shape)
    peak = float(scores[peak_y, peak_x]) if np.isfinite(work[peak_y, peak_x]) else float(np.nanmax(work))
    if not np.isfinite(peak):
        empty = np.zeros(scores.shape, dtype=bool)
        return empty, (float(peak_xy[0]), float(peak_xy[1])), 0.0
    limit = int(scores.size * MAX_SUPPORT_FRAME_FRACTION)
    chosen: tuple[np.ndarray, tuple[float, float], float] | None = None
    for frac in SYMMETRY_LEVEL_FRACS:
        level = floor + float(frac) * (peak - floor)
        candidate = np.isfinite(work) & (work >= level)
        raw = _region_containing(candidate, (peak_x, peak_y))
        if exclude is not None:
            blocked = np.asarray(exclude, dtype=bool)
            if blocked.shape == raw.shape:
                raw = np.logical_and(raw, ~blocked)
                raw = _region_containing(raw, (peak_x, peak_y))
        if int(raw.sum()) < MIN_STABLE_SUPPORT_PX or int(raw.sum()) > limit:
            continue
        moments = _mask_centroid_and_axis(raw)
        if moments is None:
            continue
        centroid, axis = moments
        reflected = _reflect_across_long_axis(raw, centroid, axis)
        iou = _symmetry_iou(raw, reflected)
        symmetric = np.logical_and(raw, reflected)
        refined = _mask_centroid_and_axis(symmetric)
        if refined is None:
            continue
        centre, axis = refined
        symmetric = np.logical_and(
            symmetric, _reflect_across_long_axis(symmetric, centre, axis)
        )
        symmetric = _region_containing(symmetric, (peak_x, peak_y))
        if exclude is not None:
            blocked = np.asarray(exclude, dtype=bool)
            if blocked.shape == symmetric.shape:
                symmetric = np.logical_and(symmetric, ~blocked)
        if _mask_centroid_and_axis(symmetric) is None:
            continue
        if int(symmetric.sum()) < MIN_STABLE_SUPPORT_PX or int(symmetric.sum()) > limit:
            continue
        grasp = estimated_shape_cog(symmetric)
        if grasp is None:
            continue
        chosen = (symmetric, grasp, float(iou))
        break
    if chosen is None:
        empty = np.zeros(scores.shape, dtype=bool)
        empty[peak_y, peak_x] = True
        return empty, (float(peak_x), float(peak_y)), 0.0
    mask, grasp, iou = chosen
    try:
        import cv2
        eroded = cv2.erode(
            mask.astype(np.uint8), np.ones((3, 3), np.uint8),
            iterations=int(SUPPORT_ERODE_ITERS),
        )
        if int(eroded.sum()) >= MIN_STABLE_SUPPORT_PX:
            tight = _region_containing(eroded.astype(bool), (peak_x, peak_y))
            if exclude is not None:
                blocked = np.asarray(exclude, dtype=bool)
                if blocked.shape == tight.shape:
                    tight = np.logical_and(tight, ~blocked)
            if int(tight.sum()) >= MIN_STABLE_SUPPORT_PX:
                cog = estimated_shape_cog(tight)
                if cog is not None:
                    return tight, cog, float(iou)
    except Exception:
        pass
    return mask, grasp, float(iou)


def support_contour_px(mask: np.ndarray, *, max_points: int = 96) -> list[list[float]]:
    """Body-hugging exterior polygon — stays close to the object edge.

    Uses a fine ``approxPolyDP`` so the polygon follows the silhouette
    instead of collapsing to a loose blob hull. Dense mask only; no ellipse.
    """
    import cv2

    array = np.asarray(mask).astype(np.uint8)
    if array.ndim != 2 or int(array.max()) == 0:
        return []
    contours, _ = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    # Keep the polygon tight to the body (small epsilon).
    eps = max(0.4, min(1.2, 0.0025 * perimeter))
    approx = cv2.approxPolyDP(contour, eps, True)
    points = approx.reshape(-1, 2)
    if len(points) < 4:
        # Too few vertices: subsample the raw outline instead of a triangle.
        raw = contour.reshape(-1, 2)
        step = max(1, len(raw) // max_points)
        points = raw[::step]
    if len(points) > max_points:
        index = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[index]
    return [[float(x), float(y)] for x, y in points]


class FrozenViTGrounder:
    """Nearest-neighbour lookup of an object prototype in frozen ViT features.

    Nothing in here is fitted. The backbone is downloaded once, frozen, and
    shared by every view and every task; the only per-task state is the
    prototype vector, which comes from the operator's registration crop and is
    then refreshed from whichever view can currently see the object clearly.
    """

    def __init__(
        self,
        *,
        backbone: str = DEFAULT_BACKBONE,
        device: torch.device | str = "cpu",
        patch_size: int = 14,
    ) -> None:
        self.backbone_name = str(backbone)
        self.device = torch.device(device)
        self.patch_size = int(patch_size)
        self._model: torch.nn.Module | None = None
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    @property
    def model(self) -> torch.nn.Module:
        """Load on first use so a run that never grounds never pays for it."""
        if self._model is None:
            import timm

            model = timm.create_model(
                self.backbone_name,
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True,
            )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._model = model.to(self.device).eval()
        return self._model

    def _to_pixels(self, rgb: np.ndarray) -> torch.Tensor:
        array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 RGB frame, received {array.shape}")
        tensor = torch.from_numpy(array).permute(2, 0, 1)[None].float().div_(255.0)
        return ((tensor - self._mean) / self._std).to(self.device)

    @torch.no_grad()
    def patch_features(self, rgb: np.ndarray) -> torch.Tensor:
        """[rows, columns, dim] unit-norm patch tokens for one frame.

        The frame is snapped to whole patches first. A backbone that cuts an
        image into 14 px squares cannot be handed a height of 480, and the grid
        is read from the frame's own proportions rather than assumed square,
        since a camera that is wider than it is tall is the ordinary case.
        """
        import cv2

        array = np.asarray(rgb)
        height, width = int(array.shape[0]), int(array.shape[1])
        rows = max(3, int(round(height / self.patch_size)))
        columns = max(3, int(round(width / self.patch_size)))
        snapped_h, snapped_w = rows * self.patch_size, columns * self.patch_size
        if (snapped_h, snapped_w) != (height, width):
            array = cv2.resize(
                array, (snapped_w, snapped_h), interpolation=cv2.INTER_CUBIC
            )
        model = self.model
        features = model.forward_features(self._to_pixels(array))
        patches = features[:, model.num_prefix_tokens :][0]
        if rows * columns != patches.shape[0]:
            raise ValueError(
                f"backbone returned {patches.shape[0]} patch tokens for a "
                f"{rows}x{columns} grid"
            )
        return F.normalize(patches, dim=-1).reshape(rows, columns, -1)

    def _resized_to_patch_grid(self, rgb: np.ndarray, scale: float) -> np.ndarray:
        import cv2

        side = int(round(rgb.shape[0] * float(scale)))
        side = max(self.patch_size * 3, int(round(side / self.patch_size)) * self.patch_size)
        return cv2.resize(np.asarray(rgb), (side, side), interpolation=cv2.INTER_CUBIC)

    @torch.no_grad()
    def prototype_from_crop(self, crop: np.ndarray) -> torch.Tensor:
        """Build a prototype from the operator's registration crop."""
        vectors = []
        for scale in BOOTSTRAP_SCALES:
            tokens = self.patch_features(self._resized_to_patch_grid(crop, scale))
            vectors.append(self._seed_core(tokens))
        return F.normalize(torch.stack(vectors).mean(0), dim=-1)

    @staticmethod
    def _seed_core(tokens: torch.Tensor) -> torch.Tensor:
        seed = tokens[tokens.shape[0] // 2, tokens.shape[1] // 2]
        flat = tokens.reshape(-1, tokens.shape[-1])
        agreement = flat @ seed
        count = max(1, int(round(flat.shape[0] * SEED_CORE_FRACTION)))
        chosen = torch.topk(agreement, count).indices
        return F.normalize(flat[chosen].mean(0), dim=-1)

    @torch.no_grad()
    def prototype_from_frame(
        self,
        frame: np.ndarray,
        centroid_px: tuple[float, float] | np.ndarray,
        radius_px: float,
    ) -> torch.Tensor:
        """Re-describe the object from a view that can see it now.

        The registration crop is one small picture of the object from one
        camera at one distance. The view that currently holds it sees it at the
        size and pose it actually has, so once a view is confident it becomes
        the better description, and the handoff to the other view rides on it.

        The picture changes; the way it is described does not. Cutting the
        object out and handing it to :meth:`prototype_from_crop` is what makes
        the two prototypes comparable, and it is not a detail: pooling the
        frame's own patches instead gathers about five of them at the far
        view's scale, most straddling the object's edge and half full of table.
        Measured against colour labels on 38 recorded grasp-range frames, that
        version put the object at 16.1 px median where the operator's crop
        managed 6.7 px -- the handoff was losing to the thing it replaced.
        """
        centre = np.asarray(centroid_px, dtype=np.float32).reshape(-1)
        # Half a patch of margin, so the object's edge is described by whole
        # patches rather than by patches that are mostly what surrounds it.
        reach = max(float(radius_px) + 0.5 * self.patch_size, float(self.patch_size))
        left = int(np.clip(round(centre[0] - reach), 0, frame.shape[1] - 1))
        right = int(np.clip(round(centre[0] + reach), left + 1, frame.shape[1]))
        top = int(np.clip(round(centre[1] - reach), 0, frame.shape[0] - 1))
        bottom = int(np.clip(round(centre[1] + reach), top + 1, frame.shape[0]))
        return self.prototype_from_crop(frame[top:bottom, left:right])

    @torch.no_grad()
    def similarity(self, frame: np.ndarray, prototype: torch.Tensor) -> np.ndarray:
        """Per-pixel cosine similarity to the prototype, at frame resolution."""
        tokens = self.patch_features(frame)
        scores = tokens @ prototype.to(tokens.device)
        upsampled = F.interpolate(
            scores[None, None].float(),
            size=(int(frame.shape[0]), int(frame.shape[1])),
            mode="bilinear",
            align_corners=False,
        )
        return upsampled[0, 0].cpu().numpy()

    def locate(
        self,
        frame: np.ndarray,
        prototype: torch.Tensor,
        *,
        search_centre_px: tuple[float, float] | None = None,
        search_radius_px: float | None = None,
        jaw_pair_px: np.ndarray | None = None,
    ) -> GroundingHit:
        """Find the prototype's best match and say how strongly it stood out.

        `separation` is the peak's height above the rest of the frame in
        background standard deviations. Support comes from prototype
        similarity + long-axis symmetry — not table colour. Optional jaw
        pixels (FK geometry) may be excluded when the claw occludes the body.
        """
        scores = self.similarity(frame, prototype)
        height, width = scores.shape
        ys, xs = np.mgrid[0:height, 0:width]

        searched = scores
        if search_centre_px is not None and search_radius_px is not None:
            within = (
                np.hypot(xs - float(search_centre_px[0]), ys - float(search_centre_px[1]))
                <= float(search_radius_px)
            )
            if bool(within.any()):
                searched = np.where(within, scores, -np.inf)

        peak_y, peak_x = np.unravel_index(int(np.argmax(searched)), searched.shape)
        peak = float(scores[peak_y, peak_x])

        exclude = None
        jaws = gripper_occlusion_mask((height, width), jaw_pair_px)
        if jaws is not None:
            exclude = jaws.copy()
            # Never exclude the peak itself or support collapses.
            exclude[peak_y, peak_x] = False

        floor_estimate = float(np.percentile(scores[np.isfinite(scores)], BACKGROUND_PERCENTILE))
        inside, centroid, symmetry_iou = select_symmetric_support(
            scores, (peak_x, peak_y), floor_estimate, exclude=exclude
        )
        weight = inside.astype(np.float64)
        shape = _support_shape(xs, ys, weight, centroid)

        background = scores[~inside]
        if background.size == 0:
            separation = 0.0
        else:
            separation = (peak - float(background.mean())) / (
                float(background.std()) + 1e-9
            )
        return GroundingHit(
            centroid_px=centroid,
            separation=float(separation),
            peak=peak,
            shape=shape,
            support_mask=inside,
            symmetry_iou=float(symmetry_iou),
        )
