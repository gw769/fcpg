"""Active column-camera gripper localization from a short open/close pulse."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np


DEFAULT_MAXIMUM_LIVE_ROTATION_DEG = 60.0
DEFAULT_MINIMUM_LIVE_SCALE = 0.5
DEFAULT_MAXIMUM_LIVE_SCALE = 1.5


def similarity_bound_residual_px(
    nominal_sweep_px: float,
    *,
    minimum_scale: float = DEFAULT_MINIMUM_LIVE_SCALE,
    maximum_scale: float = DEFAULT_MAXIMUM_LIVE_SCALE,
    maximum_rotation_deg: float = DEFAULT_MAXIMUM_LIVE_ROTATION_DEG,
) -> float:
    """Translate the similarity admissibility box into a translation residual.

    A translation fit on a two-point track has residual ``|d_obs - d_nom| / 2``.
    When the two tracks are related by a similarity, ``d_obs = s R d_nom``, so

        residual = |d_nom| / 2 * sqrt(1 + s^2 - 2 s cos(theta))

    which is an exact, monotone image of the ``(scale, rotation)`` box that
    :func:`select_live_moving_tip_similarity_transform` already enforces.  Taking
    the maximum over that box therefore yields the residual bound *equivalent to
    today's contract*, introducing no new tuning constant and admitting exactly
    the set of fits the deployed bounds admit.

    The expression is maximised on the box corners: it increases monotonically in
    ``|theta|``, and in ``s`` it is convex with an interior minimum at
    ``s = cos(theta)``, so both extremes are evaluated rather than assumed.
    """

    sweep = float(nominal_sweep_px)
    if not np.isfinite(sweep) or sweep <= 0.0:
        raise ValueError("nominal_sweep_px must be finite and positive")
    bounds = np.asarray(
        [minimum_scale, maximum_scale, maximum_rotation_deg], dtype=np.float64
    )
    if not np.all(np.isfinite(bounds)) or np.any(bounds <= 0.0):
        raise ValueError("similarity bounds must be finite and positive")
    if minimum_scale >= maximum_scale:
        raise ValueError("minimum_scale must be below maximum_scale")
    cosine = float(np.cos(np.radians(maximum_rotation_deg)))
    corners = [
        1.0 + scale**2 - 2.0 * scale * cosine
        for scale in (float(minimum_scale), float(maximum_scale))
    ]
    return sweep / 2.0 * float(np.sqrt(max(corners)))


@dataclass(frozen=True)
class ActiveGripperLocalization:
    center_px: tuple[float, float]
    open_tip_px: tuple[float, float]
    # One entry per (mask level, region endpoint) pair.  Both a quantile mask
    # and an Otsu mask are emitted because neither wins everywhere: the quantile
    # mask admits low-contrast arm shadow at some poses, and the Otsu mask can
    # clip a genuinely dim jaw at others.  The caller resolves the ambiguity by
    # fitting each candidate and keeping the one whose residual against the
    # forward-kinematics prediction is smallest, which is a physical criterion
    # rather than a tuned one.
    moving_tip_track_candidates_px: tuple[
        tuple[tuple[float, float], tuple[float, float]], ...
    ]
    closed_frame_index: int
    open_frame_index: int
    opening_span: float
    max_arm_motion: float
    moving_area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    contrast: float
    elongation: float
    #: How the open/closed correspondence was resolved from the pixels, kept so
    #: a committed calibration records which polarity it was fitted under.
    jaw_polarity: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def next_gripper_position(current: float, target: float, maximum_step: float) -> float:
    """Move one bounded normalized-position step toward a gripper target."""

    values = np.asarray([current, target, maximum_step], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("gripper positions and maximum step must be finite")
    if not 0.0 <= current <= 100.0 or not 0.0 <= target <= 100.0:
        raise ValueError("gripper positions must lie in [0,100]")
    if maximum_step <= 0.0:
        raise ValueError("maximum gripper step must be positive")
    delta = float(np.clip(target - current, -maximum_step, maximum_step))
    return float(current + delta)


def _validate_pulse(frames: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(frames)
    joints = np.asarray(states, dtype=np.float64)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("frames must be uint8 RGB [T,H,W,3]")
    if joints.shape != (len(images), 6) or not np.all(np.isfinite(joints)):
        raise ValueError("states must be finite [T,6] and align with frames")
    if len(images) < 5:
        raise ValueError("active localization requires at least five frames")
    return images, joints


def _component_elongation(component_xy: np.ndarray) -> float:
    centered = component_xy - component_xy.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return float(
        np.sqrt(max(float(eigenvalues[-1]), 1.0e-9) / max(float(eigenvalues[0]), 1.0e-9))
    )


def _principal_endpoint_centers(
    component_xy: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Estimate both physical ends of one elongated signed-difference region."""

    if len(component_xy) < 8:
        raise ValueError("signed jaw region is too small for endpoint estimation")
    center = component_xy.mean(axis=0)
    centered = component_xy - center[None]
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    minor_axis = eigenvectors[:, 0]
    major_axis = eigenvectors[:, -1]
    major = centered @ major_axis
    minor = centered @ minor_axis
    low, high = np.quantile(major, [0.02, 0.98])
    span = max(float(high - low), 1.0)
    endpoints = []
    for endpoint, band in (
        (low, major <= low + 0.12 * span),
        (high, major >= high - 0.12 * span),
    ):
        band_minor = minor[band]
        transverse = float(np.median(band_minor)) if len(band_minor) else 0.0
        point = (
            center
            + float(endpoint) * major_axis
            + transverse * minor_axis
        )
        endpoints.append((float(point[0]), float(point[1])))
    return endpoints[0], endpoints[1]


def resolve_open_signed_region_index(
    open_gray: np.ndarray,
    closed_gray: np.ndarray,
    regions: tuple[np.ndarray, np.ndarray],
    component_mask: np.ndarray,
    signed_difference: np.ndarray,
    threshold: float,
    *,
    minimum_polarity_margin: float = 6.0,
) -> tuple[int, dict[str, float]]:
    """Measure which signed region holds the moving jaw in the OPEN frame.

    The sign of ``open - closed`` cannot answer this on its own.  Relabelling the
    two regions exchanges which frame is called "jaw" and which is called
    "background", and every statistic of the difference is invariant under that
    exchange -- a bright jaw arriving is pixel-for-pixel indistinguishable from a
    dark jaw leaving.  Assuming one polarity therefore silently decides the
    open/closed correspondence, and getting it backwards inverts the observed
    track, which a similarity fit can only report as a ~180 degree rotation.

    The asymmetry that does exist is on the vacated side: wherever the jaw is
    *absent*, the camera sees the scene the jaw was standing in front of, so
    those samples must match the static background immediately surrounding the
    swept area.  The jaw itself is under no such obligation.  Measure that, and
    refuse to guess when the two hypotheses are not clearly separated.
    """

    open_frame = np.asarray(open_gray, dtype=np.float32)
    closed_frame = np.asarray(closed_gray, dtype=np.float32)
    signed = np.asarray(signed_difference, dtype=np.float32)
    selected = np.asarray(component_mask, dtype=bool)
    if not (open_frame.shape == closed_frame.shape == signed.shape == selected.shape):
        raise ValueError("polarity inputs must all share the image shape")
    if minimum_polarity_margin <= 0.0 or not np.isfinite(minimum_polarity_margin):
        raise ValueError("minimum polarity margin must be finite and positive")
    negative, positive = (np.asarray(region, dtype=bool) for region in regions)
    if not negative.any() or not positive.any():
        raise ValueError("both signed jaw regions must be non-empty")

    # A collar of *static* pixels around the swept area: the background the jaw
    # sweeps across. Restricting to |signed| < threshold keeps the moving jaw
    # itself out of the reference.
    kernel = np.ones((5, 5), np.uint8)
    solid = selected.astype(np.uint8)
    outer = cv2.dilate(solid, kernel, iterations=3).astype(bool)
    inner = cv2.dilate(solid, kernel, iterations=1).astype(bool)
    collar = outer & ~inner & (np.abs(signed) < threshold)
    if int(collar.sum()) < 24:
        raise ValueError(
            "no static background collar around the swept jaw region; "
            "cannot measure jaw polarity"
        )
    background_reference = float(
        np.median(np.concatenate([open_frame[collar], closed_frame[collar]]))
    )

    # Hypothesis "brighter": the jaw is in whichever frame is brighter, so it
    # occupies `negative` in the closed frame and `positive` in the open frame
    # -> the OPEN region is index 1.  Hypothesis "darker" is the mirror image
    # -> the OPEN region is index 0.
    occupied = {
        1: (closed_frame[negative], open_frame[positive]),
        0: (open_frame[negative], closed_frame[positive]),
    }
    vacated = {
        1: np.concatenate([open_frame[negative], closed_frame[positive]]),
        0: np.concatenate([closed_frame[negative], open_frame[positive]]),
    }
    # Two independent votes, both grounded in physics rather than in a constant:
    #   * the vacated pixels expose the scene, so they must match the collar;
    #   * the occupied pixels are ONE object photographed twice, so they must
    #     agree with each other, whereas the vacated pixels are two different
    #     patches of scene and carry no such requirement.
    background_residual = {
        index: abs(float(np.mean(samples)) - background_reference)
        for index, samples in vacated.items()
    }
    jaw_disagreement = {
        index: abs(float(np.mean(first)) - float(np.mean(second)))
        for index, (first, second) in occupied.items()
    }
    residuals = {
        index: background_residual[index] + jaw_disagreement[index]
        for index in (0, 1)
    }
    open_region_index = min(residuals, key=lambda index: residuals[index])
    margin = abs(residuals[0] - residuals[1])
    if margin < minimum_polarity_margin:
        raise ValueError(
            "jaw polarity is ambiguous, so the open/closed correspondence "
            "cannot be established: background_reference="
            f"{background_reference:.3f}, vacated_residuals="
            f"{{darker: {residuals[0]:.3f}, brighter: {residuals[1]:.3f}}}, "
            f"margin={margin:.3f} < {minimum_polarity_margin:.3f}"
        )
    diagnostics = {
        "open_signed_region_index": float(open_region_index),
        "jaw_brighter_than_background": float(open_region_index == 1),
        "background_reference_intensity": background_reference,
        "score_if_jaw_darker": residuals[0],
        "score_if_jaw_brighter": residuals[1],
        "background_residual_if_jaw_darker": background_residual[0],
        "background_residual_if_jaw_brighter": background_residual[1],
        "jaw_disagreement_if_jaw_darker": jaw_disagreement[0],
        "jaw_disagreement_if_jaw_brighter": jaw_disagreement[1],
        "polarity_margin": margin,
    }
    return open_region_index, diagnostics


def _moving_tip_track_candidates(
    signed_difference: np.ndarray,
    component_mask: np.ndarray,
    threshold: float,
    open_gray: np.ndarray,
    closed_gray: np.ndarray,
) -> tuple[
    tuple[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
    ],
    dict[str, float],
]:
    """Return candidate tracks for the same moving-jaw point in open/closed frames."""

    signed = np.asarray(signed_difference, dtype=np.float32)
    selected = np.asarray(component_mask, dtype=bool)
    if signed.shape != selected.shape:
        raise ValueError("signed difference and component mask must align")
    # signed = open - closed. Which sign carries the jaw in the OPEN frame is a
    # property of the scene, not a constant: it flips with the jaw/background
    # contrast. Measure it rather than assume it -- assuming it inverts the
    # track whenever the jaws are brighter than what is behind them.
    signed_regions = (
        selected & (signed <= -threshold),
        selected & (signed >= threshold),
    )
    open_region_index, polarity = resolve_open_signed_region_index(
        open_gray,
        closed_gray,
        signed_regions,
        selected,
        signed,
        threshold,
    )
    regions = (
        signed_regions[open_region_index],
        signed_regions[1 - open_region_index],
    )
    endpoint_sets = []
    for region in regions:
        rows, columns = np.nonzero(region)
        points = np.column_stack((columns, rows)).astype(np.float64)
        endpoint_sets.append(_principal_endpoint_centers(points))
    first = np.asarray(endpoint_sets[0], dtype=np.float64)
    second = np.asarray(endpoint_sets[1], dtype=np.float64)
    aligned_cost = float(np.linalg.norm(first[0] - second[0])) + float(
        np.linalg.norm(first[1] - second[1])
    )
    reversed_cost = float(np.linalg.norm(first[0] - second[1])) + float(
        np.linalg.norm(first[1] - second[0])
    )
    if reversed_cost < aligned_cost:
        second = second[::-1]
    pairs = [(first[index], second[index]) for index in range(2)]
    pairs.sort(
        key=lambda pair: float(np.linalg.norm(pair[0] - pair[1])), reverse=True
    )
    tracks = (
        tuple(tuple(point.tolist()) for point in pairs[0]),
        tuple(tuple(point.tolist()) for point in pairs[1]),
    )
    return tracks, polarity


def fit_point_track_similarity_transform(
    nominal_point_track_px: np.ndarray,
    observed_point_track_px: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Fit the exact 2-D similarity mapping from one two-frame point track."""

    nominal = np.asarray(nominal_point_track_px, dtype=np.float64)
    observed = np.asarray(observed_point_track_px, dtype=np.float64)
    if nominal.shape != (2, 2) or observed.shape != (2, 2):
        raise ValueError("point tracks must both have shape [2,2]")
    if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(observed)):
        raise ValueError("point tracks must be finite")
    source = nominal[1] - nominal[0]
    target = observed[1] - observed[0]
    source_length = float(np.linalg.norm(source))
    target_length = float(np.linalg.norm(target))
    if source_length < 1.0 or target_length < 1.0:
        raise ValueError("point tracks must have at least one pixel of displacement")

    source_angle = float(np.arctan2(source[1], source[0]))
    target_angle = float(np.arctan2(target[1], target[0]))
    angle_delta = target_angle - source_angle
    rotation = float(np.arctan2(np.sin(angle_delta), np.cos(angle_delta)))
    scale = target_length / source_length
    cosine = float(np.cos(rotation))
    sine = float(np.sin(rotation))
    linear = scale * np.asarray([[cosine, -sine], [sine, cosine]])
    translation = observed[0] - linear @ nominal[0]
    transform = np.concatenate((linear, translation[:, None]), axis=1)
    return transform.astype(np.float32), float(scale), float(np.degrees(rotation))


def fit_point_track_translation(
    nominal_point_track_px: np.ndarray,
    observed_point_track_px: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a translation-only image correction from one two-frame point track.

    This is the over-determined counterpart of
    :func:`fit_point_track_similarity_transform`.  A two-point track carries four
    scalar observations.  A similarity consumes all four, so it reproduces the
    track exactly and reports a residual that is identically zero no matter how
    wrong the underlying geometry is.  A translation consumes two, leaving two
    degrees of redundancy, so the residual becomes an observable measurement of
    how far the observed jaw sweep departs from the forward-kinematics
    prediction of that sweep.

    The least-squares offset is the mean correspondence offset, and the residual
    has a closed form: with ``d_nom = n1 - n0`` and ``d_obs = o1 - o0`` the two
    residual vectors are ``-(d_obs - d_nom)/2`` and ``+(d_obs - d_nom)/2``, so

        residual_rms = |d_obs - d_nom| / 2

    Equivalently the residual vanishes exactly when the similarity fit would have
    reported unit scale and zero rotation.  It is therefore a direct readout of
    the quantity the similarity fit silently absorbed into scale and rotation --
    parameters that act about the image origin and so amplify that same
    disagreement in proportion to distance from the origin.

    Returns:
        The 2x3 transform in the same convention as the similarity fit (identity
        linear part, fitted translation column) together with residual
        diagnostics.
    """

    nominal = np.asarray(nominal_point_track_px, dtype=np.float64)
    observed = np.asarray(observed_point_track_px, dtype=np.float64)
    if nominal.shape != (2, 2) or observed.shape != (2, 2):
        raise ValueError("point tracks must both have shape [2,2]")
    if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(observed)):
        raise ValueError("point tracks must be finite")
    source = nominal[1] - nominal[0]
    target = observed[1] - observed[0]
    if float(np.linalg.norm(source)) < 1.0 or float(np.linalg.norm(target)) < 1.0:
        raise ValueError("point tracks must have at least one pixel of displacement")

    offsets = observed - nominal
    offset = offsets.mean(axis=0)
    residuals = offsets - offset[None]
    residual_norms = np.linalg.norm(residuals, axis=1)
    transform = np.concatenate((np.eye(2), offset[:, None]), axis=1)
    diagnostics: dict[str, Any] = {
        "offset_px": [float(offset[0]), float(offset[1])],
        "residual_px": residuals.tolist(),
        "residual_rms_px": float(np.sqrt(np.mean(residual_norms**2))),
        "residual_max_px": float(np.max(residual_norms)),
        "sweep_disagreement_px": float(np.linalg.norm(target - source)),
        "nominal_sweep_px": float(np.linalg.norm(source)),
        "observed_sweep_px": float(np.linalg.norm(target)),
        "degrees_of_freedom": 2,
        "observations": 4,
    }
    return transform.astype(np.float32), diagnostics


def apply_live_similarity_transform(
    q_nominal_px: np.ndarray,
    transform_px: np.ndarray,
) -> np.ndarray:
    """Apply one image-plane similarity transform to any anchor array."""

    nominal = np.asarray(q_nominal_px, dtype=np.float64)
    transform = np.asarray(transform_px, dtype=np.float64)
    if nominal.ndim < 2 or nominal.shape[-1] != 2:
        raise ValueError("q_nominal_px must end in an XY coordinate")
    if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("transform_px must be one finite 2x3 matrix")
    linear = transform[:, :2]
    translation = transform[:, 2]
    return (nominal @ linear.T + translation).astype(np.float32)


def invert_live_similarity_transform(
    q_live_px: np.ndarray,
    transform_px: np.ndarray,
    *,
    max_condition_number: float = 1.0e4,
) -> np.ndarray:
    """Map live-camera pixels back into the nominal calibration image frame."""

    live = np.asarray(q_live_px, dtype=np.float64)
    transform = np.asarray(transform_px, dtype=np.float64)
    if live.ndim < 1 or live.shape[-1] != 2:
        raise ValueError("q_live_px must end in an XY coordinate")
    if not np.all(np.isfinite(live)):
        raise ValueError("q_live_px must be finite")
    if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("transform_px must be one finite 2x3 matrix")
    if max_condition_number <= 1.0 or not np.isfinite(max_condition_number):
        raise ValueError("max_condition_number must be finite and greater than one")
    linear = transform[:, :2]
    condition = float(np.linalg.cond(linear))
    if not np.isfinite(condition) or condition > max_condition_number:
        raise ValueError(
            "live similarity transform is not safely invertible: "
            f"condition_number={condition:.3g}"
        )
    translation = transform[:, 2]
    flat = live.reshape(-1, 2)
    nominal = np.linalg.solve(linear, (flat - translation[None]).T).T
    return nominal.reshape(live.shape).astype(np.float32)


def select_live_moving_tip_similarity_transform(
    nominal_anchors_px: np.ndarray,
    moving_tip_track_candidates_px: np.ndarray,
    *,
    center_anchor_index: int = 0,
    fixed_tip_anchor_index: int = 1,
    moving_tip_anchor_index: int = 2,
    minimum_scale: float = 0.5,
    maximum_scale: float = 1.5,
    maximum_rotation_deg: float = DEFAULT_MAXIMUM_LIVE_ROTATION_DEG,
    maximum_center_offset_px: float = 35.0,
) -> dict[str, Any]:
    """Select the physically local image similarity fit for an open/close pulse."""

    nominal = np.asarray(nominal_anchors_px, dtype=np.float32)
    candidates = np.asarray(moving_tip_track_candidates_px, dtype=np.float32)
    if nominal.shape != (2, 3, 2) or not np.all(np.isfinite(nominal)):
        raise ValueError("nominal calibration anchors must be finite [2,3,2]")
    if (
        candidates.ndim != 3
        or candidates.shape[1:] != (2, 2)
        or len(candidates) == 0
        or not np.all(np.isfinite(candidates))
    ):
        raise ValueError("moving-tip track candidates must be finite [N,2,2]")
    indices = (center_anchor_index, fixed_tip_anchor_index, moving_tip_anchor_index)
    if len(set(indices)) != 3 or any(index not in range(3) for index in indices):
        raise ValueError("center, fixed-tip, and moving-tip indices must be distinct")
    bounds = np.asarray(
        [
            minimum_scale,
            maximum_scale,
            maximum_rotation_deg,
            maximum_center_offset_px,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(bounds)) or np.any(bounds <= 0.0):
        raise ValueError("similarity calibration bounds must be finite and positive")
    if minimum_scale >= maximum_scale:
        raise ValueError("minimum_scale must be below maximum_scale")

    nominal_track = nominal[:, moving_tip_anchor_index]
    valid_fits: list[tuple[float, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        transform, scale, rotation_deg = fit_point_track_similarity_transform(
            nominal_track, candidate
        )
        transformed = apply_live_similarity_transform(nominal, transform)
        center_offsets = (
            transformed[:, center_anchor_index] - nominal[:, center_anchor_index]
        )
        center_distances = np.linalg.norm(center_offsets, axis=1)
        maximum_center_distance = float(np.max(center_distances))
        score = (
            float(np.mean(center_distances))
            + 0.2 * abs(rotation_deg)
            + 15.0 * abs(float(np.log(scale)))
        )
        row = {
            "candidate_index": candidate_index,
            "observed_moving_tip_track_px": candidate.copy(),
            "nominal_moving_tip_track_px": nominal_track.copy(),
            "transform_px": transform.copy(),
            "scale": scale,
            "rotation_deg": rotation_deg,
            "center_offsets_px": center_offsets.copy(),
            "maximum_center_offset_norm_px": maximum_center_distance,
            "transformed_anchors_px": transformed.copy(),
            "selection_score": score,
        }
        admissible = (
            minimum_scale <= scale <= maximum_scale
            and abs(rotation_deg) <= maximum_rotation_deg
            and maximum_center_distance <= maximum_center_offset_px
        )
        if admissible:
            valid_fits.append((score, row))
        else:
            rejected.append(row)
    if not valid_fits:
        diagnostics = [
            {
                "candidate_index": row["candidate_index"],
                "scale": row["scale"],
                "rotation_deg": row["rotation_deg"],
                "maximum_center_offset_norm_px": row[
                    "maximum_center_offset_norm_px"
                ],
            }
            for row in rejected
        ]
        raise ValueError(
            "no automatic moving-tip track fit passed the calibration bounds: "
            f"{diagnostics}"
        )

    _, selected = min(valid_fits, key=lambda item: item[0])
    return {
        **selected,
        "nominal_anchors_px": nominal.copy(),
        "center_anchor_index": center_anchor_index,
        "fixed_tip_anchor_index": fixed_tip_anchor_index,
        "moving_tip_anchor_index": moving_tip_anchor_index,
        "rejected_candidate_count": len(rejected),
    }


def _select_moving_component(
    absolute_difference: np.ndarray,
    level: float,
    *,
    minimum_component_area: int,
    maximum_component_area: int,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Threshold at ``level`` and return the strongest admissible component."""

    mask = (absolute_difference >= level).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    scored: list[tuple[float, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_component_area or area > maximum_component_area:
            continue
        contrast = float(np.mean(absolute_difference[labels == label]))
        scored.append((area * contrast, label))
    if not scored:
        raise ValueError("no reliable moving-finger component was found")
    _, selected = max(scored)
    return labels, selected, stats


_MAX_ROLE_HALO_PX = 24.0


def bbox_halo_px(bbox_xyxy: Sequence[float]) -> float:
    """Highlight slack around an OWL box. Capped so a large drop does not eat the arm."""

    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    return float(min(_MAX_ROLE_HALO_PX, max(12.0, 0.5 * max(x1 - x0, y1 - y0))))


def track_hits_bbox_xyxy(
    track_px: np.ndarray,
    bbox_xyxy: Sequence[float],
    *,
    halo_px: float | None = None,
) -> bool:
    """True if either endpoint of an open/closed track sits in the boxed object."""

    points = np.asarray(track_px, dtype=np.float64).reshape(2, 2)
    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    halo = bbox_halo_px(bbox_xyxy) if halo_px is None else float(halo_px)
    x0 -= halo
    y0 -= halo
    x1 += halo
    y1 += halo
    return bool(
        np.any(
            (points[:, 0] >= x0)
            & (points[:, 0] <= x1)
            & (points[:, 1] >= y0)
            & (points[:, 1] <= y1)
        )
    )


GRIPPER_TRACK_ROLE = "gripper"


def role_boxes_for_pulse_tracks(
    role_hits: Mapping[str, Any],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Split task-table locks into exclude (objects) vs prefer (gripper).

    Any new skill role with a box is excluded automatically. Colour is not used.
    """

    exclude: list[tuple[float, ...]] = []
    prefer: list[tuple[float, ...]] = []
    for role, measurement in (role_hits or {}).items():
        if not isinstance(measurement, Mapping):
            continue
        if not measurement.get("visible"):
            continue
        box = measurement.get("bbox_xyxy")
        if box is None:
            continue
        box_t = tuple(float(value) for value in box)
        if str(role) == GRIPPER_TRACK_ROLE:
            prefer.append(box_t)
        else:
            exclude.append(box_t)
    return tuple(exclude), tuple(prefer)


def _finite_xy(value: Any) -> np.ndarray | None:
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return None
    return point


def _point_in_bbox_xyxy(
    point_px: np.ndarray,
    bbox_xyxy: Sequence[float],
    *,
    pad_px: float = 0.0,
) -> bool:
    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    pad = float(pad_px)
    x, y = float(point_px[0]), float(point_px[1])
    return (x0 - pad) <= x <= (x1 + pad) and (y0 - pad) <= y <= (y1 + pad)


def track_metal_tip_hint_distance_px(
    track_px: np.ndarray,
    hint_px: Sequence[float],
) -> float:
    """Min distance from either open/closed endpoint to an open-frame tip hint."""

    hint = _finite_xy(hint_px)
    points = np.asarray(track_px, dtype=np.float64).reshape(2, 2)
    if hint is None or not np.all(np.isfinite(points)):
        return float("inf")
    return float(np.min(np.linalg.norm(points - hint[None], axis=1)))


def _usable_metal_tip_hint(
    hint_px: Sequence[float] | None,
    radius_px: float | None,
) -> tuple[np.ndarray, float] | None:
    hint = _finite_xy(hint_px)
    if hint is None or radius_px is None:
        return None
    radius = float(radius_px)
    if not np.isfinite(radius) or radius <= 0.0:
        return None
    return hint, radius


def tracks_matching_metal_tip_hint(
    tracks_px: np.ndarray,
    hint_px: Sequence[float],
    radius_px: float,
) -> np.ndarray:
    """Keep tracks whose open or closed end lands on the open-frame tip hint."""

    tracks = np.asarray(tracks_px, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[1:] != (2, 2) or len(tracks) == 0:
        raise ValueError("tracks_px must be finite [N,2,2]")
    usable = _usable_metal_tip_hint(hint_px, radius_px)
    if usable is None:
        raise ValueError("metal tip hint must be a finite XY and positive radius")
    hint, radius = usable
    matched = [
        track
        for track in tracks
        if track_metal_tip_hint_distance_px(track, hint) <= radius
    ]
    if not matched:
        raise ValueError(
            "no moving-tip track matched the open-frame metal tip hint: "
            f"hint_px={hint.tolist()} radius_px={radius:.3f}"
        )
    return np.asarray(matched, dtype=np.float32)


def independent_open_frame_metal_tip_hint(
    open_frame: np.ndarray | None,
    *,
    support_bbox_xyxy: Sequence[float] | None = None,
    candidate_tracks_px: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Independent distal metal-tip identity from the pulse's exact open frame.

    Residual-vs-FK cannot tell a knuckle from a claw tip: both are on the moving
    jaw, and a gripper role box often covers only the proximal band.  The same
    pulse's open frame carries an independent appearance measurement.  That
    measurement becomes a hint only after a distal-identity check against the
    pulse motion support (or against the extracted tracks).  An unvalidated
    full-frame detection is not a hint -- it is how a letterbox / table blob
    would otherwise fail the pulse closed.

    Missing or unvalidated measurements return ``None`` so ranking stays
    historical.  No residual bound is widened and no coordinate is hard-coded.
    """

    from fcpg.so101_column_tip_measurement import (
        _hanging_distal,
        column_content_rows,
        column_metal_operator_tips,
        gripper_like_mask,
    )

    if open_frame is None:
        return None
    frame = np.asarray(open_frame)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        return None
    support = None
    if support_bbox_xyxy is not None and len(support_bbox_xyxy) == 4:
        support = tuple(float(value) for value in support_bbox_xyxy)
    tracks = None
    if candidate_tracks_px is not None:
        proposed = np.asarray(candidate_tracks_px, dtype=np.float32)
        if proposed.ndim == 3 and proposed.shape[1:] == (2, 2) and len(proposed) > 0:
            tracks = proposed
    pad = float(DEFAULT_MOTION_ROI_EXPAND_PX)

    def _near_tracks(point: np.ndarray, radius: float) -> bool:
        if tracks is None:
            return False
        return any(
            track_metal_tip_hint_distance_px(track, point) <= radius for track in tracks
        )

    metal = column_metal_operator_tips(frame)
    if metal is not None:
        moving = _finite_xy(metal[2])
        fixed = _finite_xy(metal[1])
        if moving is not None and fixed is not None:
            span = float(np.linalg.norm(moving - fixed))
            radius = 0.5 * span if span > 0.0 else None
            supported_moving = support is not None and _point_in_bbox_xyxy(
                moving, support, pad_px=pad
            )
            supported_fixed = support is not None and _point_in_bbox_xyxy(
                fixed, support, pad_px=pad
            )
            hints: list[np.ndarray] = []
            if supported_moving:
                hints.append(moving)
            if supported_fixed:
                hints.append(fixed)
            elif radius is not None and _near_tracks(moving, radius):
                hints.append(moving)
            if tracks is not None and radius is not None:
                hints = [point for point in hints if _near_tracks(point, radius)]
            hint = hints[0] if hints else None
            if hint is not None and radius is not None:
                return {
                    "hint_px": (float(hint[0]), float(hint[1])),
                    "radius_px": float(radius),
                    "source": "column_metal_operator_tips",
                    "metal_tips_px": np.asarray(metal, dtype=np.float32).copy(),
                    "support_bbox_xyxy": support,
                }

    if support is None:
        return None
    y0, y1 = column_content_rows(frame)
    grip = gripper_like_mask(frame, y0, y1)
    distal, _attach = _hanging_distal(grip, y0, y1)
    supported = [
        (float(x), float(y))
        for x, y in distal.items()
        if _point_in_bbox_xyxy((x, y), support, pad_px=pad)
    ]
    if not supported:
        return None
    # Column view hangs down the image: the most distal support peak is max y.
    hint = np.asarray(max(supported, key=lambda point: point[1]), dtype=np.float64)
    width = max(support[2] - support[0], 1.0)
    height = max(support[3] - support[1], 1.0)
    radius = 0.5 * min(width, height)
    # Motion support can itself lock onto a moving person or exposure edge.
    # When pulse tracks exist, the appearance hint is valid only if at least
    # one extracted trajectory actually reaches it. Otherwise return no hint
    # and let the unchanged translation-residual/role-box selector decide.
    # Passing a support-only false hint downstream would fail every candidate
    # closed, even when the true distal track is already in the candidate set.
    if tracks is not None and not _near_tracks(hint, radius):
        return None
    if tracks is None and not _point_in_bbox_xyxy(hint, support, pad_px=pad):
        return None
    return {
        "hint_px": (float(hint[0]), float(hint[1])),
        "radius_px": float(radius),
        "source": "support_hanging_distal",
        "metal_tips_px": None if metal is None else np.asarray(metal, dtype=np.float32).copy(),
        "support_bbox_xyxy": support,
    }


def filter_tracks_by_role_boxes(
    tracks_px: np.ndarray,
    *,
    exclude_bboxes_xyxy: Sequence[Sequence[float]] = (),
    prefer_bboxes_xyxy: Sequence[Sequence[float]] = (),
    metal_tip_hint_px: Sequence[float] | None = None,
    metal_tip_hint_radius_px: float | None = None,
) -> np.ndarray:
    """Drop tracks on any non-gripper role box; prefer the gripper box.

    Residual-vs-FK ranking cannot do this: when the factory projection already
    sits on a scene object, that object's motion wins the residual. Identity
    comes from the task-table open-vocab roles, not from colour.

    When an open-frame metal-tip hint is present it is the distal identity:
    matching tracks are kept even if they sit outside the proximal gripper box
    or overlap a pickup box.  A present hint with no match fails closed.
    A missing hint leaves this filter unchanged.
    """

    tracks = np.asarray(tracks_px, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[1:] != (2, 2) or len(tracks) == 0:
        raise ValueError("tracks_px must be finite [N,2,2]")
    usable_hint = _usable_metal_tip_hint(metal_tip_hint_px, metal_tip_hint_radius_px)
    if usable_hint is not None:
        return tracks_matching_metal_tip_hint(
            tracks, usable_hint[0], usable_hint[1]
        )
    prefer = [bbox for bbox in prefer_bboxes_xyxy if bbox is not None]
    exclude = [bbox for bbox in exclude_bboxes_xyxy if bbox is not None]

    def hits_any(track: np.ndarray, boxes: list, *, halo_px: float | None) -> bool:
        return any(
            track_hits_bbox_xyxy(track, bbox, halo_px=halo_px) for bbox in boxes
        )

    if prefer:
        preferred = [
            track
            for track in tracks
            if hits_any(track, prefer, halo_px=8.0)
            and not hits_any(track, exclude, halo_px=None)
        ]
        if preferred:
            return np.asarray(preferred, dtype=np.float32)
        raise ValueError("no moving-tip track in the gripper role box")
    keep = np.ones(len(tracks), dtype=bool)
    for bbox in exclude:
        for index, track in enumerate(tracks):
            if track_hits_bbox_xyxy(track, bbox):
                keep[index] = False
    filtered = tracks[keep]
    if len(filtered) == 0:
        raise ValueError(
            "every moving-tip track landed on a locked non-gripper role"
        )
    return np.asarray(filtered, dtype=np.float32)


DEFAULT_MINIMUM_OPENING_SPAN = 20.0


@dataclass(frozen=True)
class PulseTrackDecision:
    """Hard identity gates for one jaw pulse. Fail-closed; no nearest-blob."""

    allowed: bool
    reason: str
    gripper_box_exists: bool
    opening_span_ok: bool
    moving_tracks_in_gripper: int
    tracks_explained_by_other: int
    geometry_consistent: bool
    kept_tracks_px: np.ndarray | None = None


def pulse_tracks_allowed(
    tracks_px: np.ndarray,
    *,
    gripper_role_configured: bool,
    prefer_bboxes_xyxy: Sequence[Sequence[float]] = (),
    exclude_bboxes_xyxy: Sequence[Sequence[float]] = (),
    opening_span: float | None = None,
    minimum_opening_span: float = DEFAULT_MINIMUM_OPENING_SPAN,
    nominal_anchors_px: np.ndarray | None = None,
    maximum_center_offset_px: float = 60.0,
    maximum_residual_px: float | None = None,
    metal_tip_hint_px: Sequence[float] | None = None,
    metal_tip_hint_radius_px: float | None = None,
) -> PulseTrackDecision:
    """PULSE ALLOWED only if all five gates hold. Never pick a leftover blob.

    1. Gripper role box exists when the task table configures a gripper role.
    2. Open/close produced sufficient image motion (existing opening_span).
    3. A moving-tip track lies inside the gripper role box.
    4. That track is not explained by any non-gripper role.
    5. Track geometry is consistent with the FK jaw sweep (existing residual).

    An open-frame metal-tip hint, when present, is distal identity for gates
    1/3/4.  A missing hint leaves those gates historical.
    """

    prefer = tuple(bbox for bbox in prefer_bboxes_xyxy if bbox is not None)
    exclude = tuple(bbox for bbox in exclude_bboxes_xyxy if bbox is not None)
    gripper_box_exists = bool(prefer)
    hint_usable = _usable_metal_tip_hint(metal_tip_hint_px, metal_tip_hint_radius_px)
    if gripper_role_configured and not gripper_box_exists and hint_usable is None:
        return PulseTrackDecision(
            allowed=False,
            reason=(
                "pulse has no open-vocab gripper lock; "
                "refusing residual-only correspondence"
            ),
            gripper_box_exists=False,
            opening_span_ok=False,
            moving_tracks_in_gripper=0,
            tracks_explained_by_other=0,
            geometry_consistent=False,
        )
    span_ok = True
    if opening_span is not None:
        span = float(opening_span)
        floor = float(minimum_opening_span)
        span_ok = np.isfinite(span) and np.isfinite(floor) and span >= floor
        if not span_ok:
            return PulseTrackDecision(
                allowed=False,
                reason=(
                    "opening/closing did not produce sufficient image motion: "
                    f"opening_span={opening_span} < {minimum_opening_span}"
                ),
                gripper_box_exists=gripper_box_exists,
                opening_span_ok=False,
                moving_tracks_in_gripper=0,
                tracks_explained_by_other=0,
                geometry_consistent=False,
            )
    tracks = np.asarray(tracks_px, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[1:] != (2, 2) or len(tracks) == 0:
        return PulseTrackDecision(
            allowed=False,
            reason="no moving-tip track could be extracted",
            gripper_box_exists=gripper_box_exists,
            opening_span_ok=span_ok,
            moving_tracks_in_gripper=0,
            tracks_explained_by_other=0,
            geometry_consistent=False,
        )

    def _hits(track: np.ndarray, boxes: Sequence[Sequence[float]], halo: float | None) -> bool:
        return any(track_hits_bbox_xyxy(track, bbox, halo_px=halo) for bbox in boxes)

    in_gripper = 0
    explained_by_other = 0
    for track in tracks:
        hits_gripper = bool(prefer) and _hits(track, prefer, 8.0)
        hits_other = bool(exclude) and _hits(track, exclude, None)
        if hits_gripper and not hits_other:
            in_gripper += 1
        if hits_other:
            explained_by_other += 1
    try:
        kept = filter_tracks_by_role_boxes(
            tracks,
            exclude_bboxes_xyxy=exclude,
            prefer_bboxes_xyxy=prefer,
            metal_tip_hint_px=metal_tip_hint_px,
            metal_tip_hint_radius_px=metal_tip_hint_radius_px,
        )
    except ValueError as error:
        return PulseTrackDecision(
            allowed=False,
            reason=str(error),
            gripper_box_exists=gripper_box_exists,
            opening_span_ok=span_ok,
            moving_tracks_in_gripper=in_gripper,
            tracks_explained_by_other=explained_by_other,
            geometry_consistent=False,
        )
    geometry_ok = True
    if nominal_anchors_px is not None:
        try:
            select_live_moving_tip_translation(
                nominal_anchors_px,
                kept,
                maximum_center_offset_px=maximum_center_offset_px,
                maximum_residual_px=maximum_residual_px,
                metal_tip_hint_px=metal_tip_hint_px,
                metal_tip_hint_radius_px=metal_tip_hint_radius_px,
            )
        except ValueError as error:
            return PulseTrackDecision(
                allowed=False,
                reason=str(error),
                gripper_box_exists=gripper_box_exists,
                opening_span_ok=span_ok,
                moving_tracks_in_gripper=in_gripper,
                tracks_explained_by_other=explained_by_other,
                geometry_consistent=False,
                kept_tracks_px=kept,
            )
    return PulseTrackDecision(
        allowed=True,
        reason="ok",
        gripper_box_exists=gripper_box_exists,
        opening_span_ok=span_ok,
        moving_tracks_in_gripper=int(len(kept)),
        tracks_explained_by_other=explained_by_other,
        geometry_consistent=geometry_ok,
        kept_tracks_px=kept,
    )


def _high_contrast_level(absolute_difference: np.ndarray, level: float) -> float:
    """Split the suprathreshold difference population into strong and weak.

    The quantile cut alone cannot separate the moving jaw from the shadow the
    jaw casts: both move with the aperture and they can sit within one closing
    kernel of each other, so morphology merges them.  What does separate them is
    contrast -- a shadow is an attenuated copy of the object casting it.  Otsu's
    method finds that split from the measured histogram, so this adds no tuning
    constant of its own.
    """

    values = absolute_difference[absolute_difference >= level]
    if values.size == 0:
        return float(level)
    scaled = np.clip(values, 0.0, 255.0).astype(np.uint8).reshape(-1, 1)
    otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(float(level), float(otsu))


def _multi_level_track_candidates(
    signed_difference: np.ndarray,
    absolute_difference: np.ndarray,
    threshold: float,
    open_gray: np.ndarray,
    closed_gray: np.ndarray,
    *,
    minimum_component_area: int,
    maximum_component_area: int,
) -> tuple[
    tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    dict[str, float],
]:
    """Propose moving-tip tracks from both the quantile and the Otsu mask.

    Neither mask level is right everywhere, and nothing available inside this
    function can tell which one is right here -- that judgement needs the
    forward-kinematics prediction of the sweep, which lives in the planner.  So
    both are proposed and the caller discriminates on fit residual.  Levels that
    yield no admissible component are skipped rather than raising, provided at
    least one level succeeds.

    Polarity is measured per mask level (upstream of pairing).  A level whose
    polarity is ambiguous raises ``ValueError`` and is skipped like any other
    failed level, rather than killing the whole pulse.  The recorded polarity
    is the first successful level's -- quantile first, matching the original
    production path -- because polarity is a scene property, not a mask-level
    one.
    """

    levels = [float(threshold)]
    high = _high_contrast_level(absolute_difference, threshold)
    if high > threshold:
        levels.append(high)
    proposals: list[tuple[tuple[float, float], tuple[float, float]]] = []
    polarities: list[dict[str, float]] = []
    failures: list[str] = []
    for level in levels:
        try:
            labels, selected, _ = _select_moving_component(
                absolute_difference,
                level,
                minimum_component_area=minimum_component_area,
                maximum_component_area=maximum_component_area,
            )
            tracks, polarity = _moving_tip_track_candidates(
                signed_difference,
                labels == selected,
                level,
                open_gray,
                closed_gray,
            )
            proposals.extend(tracks)
            polarities.append(polarity)
        except ValueError as error:
            failures.append(f"level={level:.4f}: {error}")
    if not proposals:
        raise ValueError(
            "no moving-tip track could be extracted at any mask level: "
            + "; ".join(failures)
        )
    return tuple(proposals), polarities[0]


def select_live_moving_tip_translation(
    nominal_anchors_px: np.ndarray,
    moving_tip_track_candidates_px: np.ndarray,
    *,
    center_anchor_index: int = 0,
    fixed_tip_anchor_index: int = 1,
    moving_tip_anchor_index: int = 2,
    # 35 px was the old local envelope. Clean Jieli shifts now span 46-56 px
    # across the calibrated free-space poses; 60 admits those tracks and still
    # rejects the measured ~91 px shadow-contaminated candidate.
    maximum_center_offset_px: float = 60.0,
    maximum_residual_px: float | None = None,
    exclude_bboxes_xyxy: Sequence[Sequence[float]] = (),
    prefer_bboxes_xyxy: Sequence[Sequence[float]] = (),
    metal_tip_hint_px: Sequence[float] | None = None,
    metal_tip_hint_radius_px: float | None = None,
) -> dict[str, Any]:
    """Select a translation-only live anchor correction from pulse candidates.

    This is the over-determined replacement for
    :func:`select_live_moving_tip_similarity_transform`.  Because a translation
    leaves two degrees of redundancy in a two-point track, every candidate now
    reports how far it disagrees with the forward-kinematics prediction of the
    same jaw sweep.  That residual does three jobs the similarity fit could not
    do at all: it ranks the candidates on physical evidence rather than on a
    hand-weighted score, it rejects a mis-extracted track outright, and it is
    exported so the runtime can gate on it.

    The residual is a check on the *measurement*, not on the correction's
    validity elsewhere in the workspace.  A pulse fitted at one pose can be
    perfectly self-consistent and still be wrong at another pose; that is the
    support-radius gate's job, and this function does not duplicate it.
    """

    nominal = np.asarray(nominal_anchors_px, dtype=np.float32)
    candidates = np.asarray(moving_tip_track_candidates_px, dtype=np.float32)
    if nominal.shape != (2, 3, 2) or not np.all(np.isfinite(nominal)):
        raise ValueError("nominal calibration anchors must be finite [2,3,2]")
    if (
        candidates.ndim != 3
        or candidates.shape[1:] != (2, 2)
        or len(candidates) == 0
        or not np.all(np.isfinite(candidates))
    ):
        raise ValueError("moving-tip track candidates must be finite [N,2,2]")
    indices = (center_anchor_index, fixed_tip_anchor_index, moving_tip_anchor_index)
    if len(set(indices)) != 3 or any(index not in range(3) for index in indices):
        raise ValueError("center, fixed-tip, and moving-tip indices must be distinct")
    if not np.isfinite(maximum_center_offset_px) or maximum_center_offset_px <= 0.0:
        raise ValueError("maximum_center_offset_px must be finite and positive")
    usable_hint = _usable_metal_tip_hint(metal_tip_hint_px, metal_tip_hint_radius_px)
    source_hint_distances = [
        track_metal_tip_hint_distance_px(track, usable_hint[0])
        if usable_hint is not None
        else float("inf")
        for track in candidates
    ]
    source_indices = list(range(len(candidates)))
    if exclude_bboxes_xyxy or prefer_bboxes_xyxy or usable_hint is not None:
        kept = filter_tracks_by_role_boxes(
            candidates,
            exclude_bboxes_xyxy=exclude_bboxes_xyxy,
            prefer_bboxes_xyxy=prefer_bboxes_xyxy,
            metal_tip_hint_px=metal_tip_hint_px,
            metal_tip_hint_radius_px=metal_tip_hint_radius_px,
        )
        recovered: list[int] = []
        for track in kept:
            for index, source in enumerate(candidates):
                if np.allclose(track, source, atol=1.0e-5):
                    recovered.append(index)
                    break
            else:
                recovered.append(-1)
        candidates = kept
        source_indices = recovered

    nominal_track = nominal[:, moving_tip_anchor_index]
    if maximum_residual_px is None:
        # Inherit the deployed similarity contract exactly rather than inventing
        # a residual budget: see `similarity_bound_residual_px`.
        residual_bound = similarity_bound_residual_px(
            float(np.linalg.norm(nominal_track[1] - nominal_track[0]))
        )
    else:
        residual_bound = float(maximum_residual_px)
        if not np.isfinite(residual_bound) or residual_bound <= 0.0:
            raise ValueError("maximum_residual_px must be finite and positive")
    valid_fits: list[tuple[float, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for filtered_index, proposed in enumerate(candidates):
        candidate_index = int(source_indices[filtered_index])
        if candidate_index < 0:
            candidate_index = filtered_index
        # Which observed point belongs to the open frame and which to the closed
        # one is not always recovered correctly by the extraction: the third
        # pulse arrives with both tracks end-swapped relative to the first.  A
        # swap is a pure correspondence error, and a 2-DOF fit cannot hide it --
        # with no rotational freedom the reversed pairing simply produces a large
        # residual.  So the redundancy that makes the residual observable also
        # resolves the ambiguity: fit both pairings and keep the one the data
        # prefers.  No threshold is introduced; the two residuals are compared
        # against each other.  A 4-DOF similarity cannot do this, because it
        # reproduces either pairing exactly and reports zero for both, and can
        # only fall back on rejecting the near-180-degree rotation it invents to
        # explain the swap.
        orderings = (
            ("as_extracted", proposed),
            ("reversed", proposed[::-1].copy()),
        )
        fitted = []
        for ordering_name, candidate in orderings:
            transform, diagnostics = fit_point_track_translation(
                nominal_track, candidate
            )
            fitted.append(
                (float(diagnostics["residual_rms_px"]), ordering_name, candidate,
                 transform, diagnostics)
            )
        _, ordering_name, candidate, transform, diagnostics = min(
            fitted, key=lambda item: item[0]
        )
        rejected_ordering_residual = max(item[0] for item in fitted)
        transformed = apply_live_similarity_transform(nominal, transform)
        center_offsets = (
            transformed[:, center_anchor_index] - nominal[:, center_anchor_index]
        )
        center_distances = np.linalg.norm(center_offsets, axis=1)
        maximum_center_distance = float(np.max(center_distances))
        residual = float(diagnostics["residual_rms_px"])
        hint_point = _finite_xy(metal_tip_hint_px)
        hint_distance = (
            track_metal_tip_hint_distance_px(proposed, hint_point)
            if hint_point is not None
            else float("inf")
        )
        open_hint_distance = (
            float(np.linalg.norm(np.asarray(candidate[0], dtype=np.float64) - hint_point))
            if hint_point is not None
            else float("inf")
        )
        row = {
            "candidate_index": candidate_index,
            "observed_moving_tip_track_px": candidate.copy(),
            "track_ordering": ordering_name,
            "rejected_ordering_residual_rms_px": rejected_ordering_residual,
            "nominal_moving_tip_track_px": nominal_track.copy(),
            "transform_px": transform.copy(),
            "offset_px": diagnostics["offset_px"],
            "residual_rms_px": residual,
            "hint_distance_px": hint_distance,
            "open_hint_distance_px": open_hint_distance,
            "residual_max_px": diagnostics["residual_max_px"],
            "sweep_disagreement_px": diagnostics["sweep_disagreement_px"],
            "nominal_sweep_px": diagnostics["nominal_sweep_px"],
            "observed_sweep_px": diagnostics["observed_sweep_px"],
            "center_offsets_px": center_offsets.copy(),
            "maximum_center_offset_norm_px": maximum_center_distance,
            "transformed_anchors_px": transformed.copy(),
            # Unlike the similarity score this is not a weighted blend of
            # unrelated quantities: it is the fit's own residual, in pixels.
            "selection_score": residual,
            "hint_matched": bool(
                usable_hint is not None
                and np.isfinite(hint_distance)
                and hint_distance <= usable_hint[1]
            ),
        }
        row["maximum_residual_px"] = residual_bound
        admissible = (
            residual <= residual_bound
            and maximum_center_distance <= maximum_center_offset_px
        )
        if admissible:
            valid_fits.append((residual, row))
        else:
            rejected.append(row)
    if not valid_fits:
        diagnostics_rows = [
            {
                "candidate_index": row["candidate_index"],
                "residual_rms_px": row["residual_rms_px"],
                "sweep_disagreement_px": row["sweep_disagreement_px"],
                "maximum_center_offset_norm_px": row["maximum_center_offset_norm_px"],
                "hint_distance_px": row.get("hint_distance_px"),
            }
            for row in rejected
        ]
        raise ValueError(
            "no moving-tip track fit passed the translation calibration bounds: "
            f"{diagnostics_rows}"
        )

    if usable_hint is not None:
        hinted_fits = [
            item for item in valid_fits if bool(item[1].get("hint_matched"))
        ]
        if not hinted_fits:
            raise ValueError(
                "no moving-tip track matched the open-frame metal tip hint: "
                f"hint_px={usable_hint[0].tolist()} radius_px={usable_hint[1]:.3f} "
                f"candidate_hint_distances_px={source_hint_distances}"
            )
        valid_fits = hinted_fits
        selection_reason = "open_frame_metal_tip_hint"
    else:
        selection_reason = "translation_residual"

    _, selected = min(valid_fits, key=lambda item: item[0])
    selected = {
        **selected,
        "selection_reason": selection_reason,
        "metal_tip_hint_px": (
            None
            if usable_hint is None
            else (float(usable_hint[0][0]), float(usable_hint[0][1]))
        ),
        "metal_tip_hint_radius_px": (
            None if usable_hint is None else float(usable_hint[1])
        ),
        "candidate_hint_distances_px": [float(value) for value in source_hint_distances],
    }
    return {
        **selected,
        "nominal_anchors_px": nominal.copy(),
        "center_anchor_index": center_anchor_index,
        "fixed_tip_anchor_index": fixed_tip_anchor_index,
        "moving_tip_anchor_index": moving_tip_anchor_index,
        "rejected_candidate_count": len(rejected),
        "candidate_residuals_px": [
            float(row["residual_rms_px"])
            for row in sorted(
                [row for _, row in valid_fits] + rejected,
                key=lambda row: row["candidate_index"],
            )
        ],
    }


def localize_gripper_from_pulse(
    frames: np.ndarray,
    states: np.ndarray,
    *,
    minimum_opening_span: float = 20.0,
    maximum_arm_motion: float = 2.0,
    minimum_component_area: int = 24,
    maximum_component_fraction: float = 0.18,
) -> ActiveGripperLocalization:
    """Locate the closed moving fingertip while the five arm joints stay fixed.

    Only image changes caused by the commanded gripper aperture are used.  The
    scene background and target may be anywhere, but they must remain still
    during the short pulse.
    """

    images, joints = _validate_pulse(frames, states)
    opening = joints[:, 5]
    closed_index = int(np.argmin(opening))
    open_index = int(np.argmax(opening))
    opening_span = float(opening[open_index] - opening[closed_index])
    if opening_span < minimum_opening_span:
        raise ValueError(
            f"gripper opening span {opening_span:.3f} is below {minimum_opening_span:.3f}"
        )
    arm_motion = float(np.max(np.ptp(joints[:, :5], axis=0)))
    if arm_motion > maximum_arm_motion:
        raise ValueError(
            f"arm moved {arm_motion:.3f} during gripper pulse; maximum is {maximum_arm_motion:.3f}"
        )

    open_gray = cv2.cvtColor(images[open_index], cv2.COLOR_RGB2GRAY).astype(np.float32)
    closed_gray = cv2.cvtColor(images[closed_index], cv2.COLOR_RGB2GRAY).astype(np.float32)
    open_gray = cv2.GaussianBlur(open_gray, (5, 5), 0)
    closed_gray = cv2.GaussianBlur(closed_gray, (5, 5), 0)
    signed = open_gray - closed_gray
    absolute = np.abs(signed)
    threshold = max(12.0, float(np.quantile(absolute, 0.96)))
    image_area = int(images.shape[1] * images.shape[2])
    maximum_area = int(round(maximum_component_fraction * image_area))

    labels, selected, stats = _select_moving_component(
        absolute,
        threshold,
        minimum_component_area=minimum_component_area,
        maximum_component_area=maximum_area,
    )
    rows, columns = np.nonzero(labels == selected)
    component_xy = np.column_stack((columns, rows)).astype(np.float64)
    flow = cv2.calcOpticalFlowFarneback(
        open_gray.astype(np.uint8),
        closed_gray.astype(np.uint8),
        None,
        0.5,
        5,
        21,
        5,
        7,
        1.5,
        0,
    )
    magnitude = np.linalg.norm(flow, axis=-1)
    component_magnitude = magnitude[rows, columns]
    moving = component_magnitude >= np.quantile(component_magnitude, 0.85)
    sources = component_xy[moving]
    destinations = sources + flow[rows[moving], columns[moving]]
    open_tip = np.median(sources, axis=0)
    closed_tip = np.median(destinations, axis=0)
    elongation = _component_elongation(component_xy)
    moving_tip_track_candidates, jaw_polarity = _multi_level_track_candidates(
        signed,
        absolute,
        threshold,
        open_gray,
        closed_gray,
        minimum_component_area=minimum_component_area,
        maximum_component_area=maximum_area,
    )
    x, y, width, height, area = stats[selected].tolist()
    contrast = float(np.mean(absolute[labels == selected]))
    return ActiveGripperLocalization(
        # ``center_px`` is retained for existing offline artifacts. It is the
        # closed moving fingertip, not the midpoint between both jaws.
        center_px=(float(closed_tip[0]), float(closed_tip[1])),
        open_tip_px=(float(open_tip[0]), float(open_tip[1])),
        moving_tip_track_candidates_px=moving_tip_track_candidates,
        closed_frame_index=closed_index,
        open_frame_index=open_index,
        opening_span=opening_span,
        max_arm_motion=arm_motion,
        moving_area_px=int(area),
        bbox_xyxy=(int(x), int(y), int(x + width), int(y + height)),
        contrast=contrast,
        elongation=elongation,
        jaw_polarity=jaw_polarity,
    )


def apply_live_center_translation(
    q_nominal_px: np.ndarray,
    live_center_px: np.ndarray,
    *,
    anchor_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate all anchor priors so one selected anchor matches observation."""

    nominal = np.asarray(q_nominal_px, dtype=np.float64)
    center = np.asarray(live_center_px, dtype=np.float64)
    if nominal.ndim != 3 or nominal.shape[-2:] != (3, 2):
        raise ValueError("q_nominal_px must have shape [T,3,2]")
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("live_center_px must be one finite XY coordinate")
    if anchor_index not in range(nominal.shape[-2]):
        raise ValueError("anchor_index is outside the nominal anchor axis")
    offset = center - nominal[-1, anchor_index]
    return (nominal + offset[None, None, :]).astype(np.float32), offset.astype(np.float32)


DEFAULT_MOTION_ROI_EXPAND_PX = 12
MOTION_GRIPPER_ROI_PROVENANCE = "open_close_motion_bbox"


def expand_motion_gripper_bbox(
    bbox_xyxy: Sequence[float],
    frame_hw: tuple[int, int],
    *,
    expand_px: int = DEFAULT_MOTION_ROI_EXPAND_PX,
    min_side_px: int = 16,
    min_area_fraction: float = 0.008,
    max_area_fraction: float = 0.50,
) -> dict[str, Any]:
    """Bounded expansion of a pulse-difference bbox.  Image-only."""

    from fcpg.so101_self_contact import assess_gripper_roi

    if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) != 4:
        return {
            "valid": False,
            "xyxy": None,
            "provenance": MOTION_GRIPPER_ROI_PROVENANCE,
            "reason": "gripper_roi_required",
            "diagnostics": {"source": "pulse_difference_bbox_missing"},
        }
    height, width = int(frame_hw[0]), int(frame_hw[1])
    x0, y0, x1, y1 = (int(round(float(v))) for v in bbox_xyxy)
    raw = (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )
    pad = max(0, int(expand_px))
    expanded = (
        max(0, raw[0] - pad),
        max(0, raw[1] - pad),
        min(width, raw[2] + pad),
        min(height, raw[3] + pad),
    )
    assessed_expanded = assess_gripper_roi(
        expanded,
        frame_hw,
        allow_full_image=False,
        provenance=MOTION_GRIPPER_ROI_PROVENANCE,
        min_side_px=min_side_px,
        min_area_fraction=min_area_fraction,
        max_area_fraction=max_area_fraction,
    )
    if assessed_expanded["valid"]:
        diagnostics = dict(assessed_expanded["diagnostics"])
        diagnostics.update(
            {
                "source_bbox_xyxy": list(raw),
                "expand_px": pad,
                "used_expansion": True,
            }
        )
        assessed_expanded["diagnostics"] = diagnostics
        return assessed_expanded
    assessed_raw = assess_gripper_roi(
        raw,
        frame_hw,
        allow_full_image=False,
        provenance=MOTION_GRIPPER_ROI_PROVENANCE,
        min_side_px=min_side_px,
        min_area_fraction=min_area_fraction,
        max_area_fraction=max_area_fraction,
    )
    diagnostics = dict(assessed_raw["diagnostics"] or {})
    diagnostics.update(
        {
            "source_bbox_xyxy": list(raw),
            "expand_px": pad,
            "used_expansion": False,
            "expanded_rejected": assessed_expanded.get("reason"),
        }
    )
    assessed_raw["diagnostics"] = diagnostics
    return assessed_raw


def motion_gripper_roi_from_localization(
    localization: Any,
    frame_hw: tuple[int, int],
    *,
    expand_px: int = DEFAULT_MOTION_ROI_EXPAND_PX,
    view: str | None = None,
) -> dict[str, Any]:
    """Reuse ``localize_gripper_from_pulse`` bbox/component, then expand.

    Does not use FK, metal, fish/box masks, M0/M1, FCPG, or pulse gate 5.
    ``view`` stamps provenance so a column box cannot be silently reused
    as a wrist box.
    """

    bbox = None
    area = None
    if isinstance(localization, Mapping):
        bbox = localization.get("bbox_xyxy")
        area = localization.get("moving_area_px")
        views = localization.get("views")
        if isinstance(views, Mapping):
            chosen = views.get(view) if view in {"column", "wrist"} else None
            if chosen is None and view is None:
                chosen = views.get("column")
            if isinstance(chosen, Mapping):
                bbox = chosen.get("bbox_xyxy", bbox)
                area = chosen.get("moving_area_px", area)
    else:
        bbox = getattr(localization, "bbox_xyxy", None)
        area = getattr(localization, "moving_area_px", None)
    assessed = expand_motion_gripper_bbox(
        bbox if bbox is not None else (),
        frame_hw,
        expand_px=expand_px,
    )
    diagnostics = dict(assessed.get("diagnostics") or {})
    if area is not None:
        diagnostics["moving_area_px"] = int(area)
    diagnostics["source"] = "pulse_open_close_component_bbox"
    diagnostics["view"] = view
    assessed["diagnostics"] = diagnostics
    assessed["view"] = view
    return assessed


def motion_gripper_rois_from_pulse(
    pulse_result: Mapping[str, Any],
    frame_hw: tuple[int, int] | Mapping[str, tuple[int, int]] = (224, 224),
    *,
    expand_px: int = DEFAULT_MOTION_ROI_EXPAND_PX,
    require_complete_in_frame: bool = False,
) -> dict[str, dict[str, Any]]:
    """Independent column and wrist motion ROIs.  Never copy coordinates."""

    from fcpg.so101_self_contact import assess_gripper_roi

    hw_by_view: dict[str, tuple[int, int]]
    if isinstance(frame_hw, Mapping):
        hw_by_view = {
            str(name): (int(value[0]), int(value[1])) for name, value in frame_hw.items()
        }
    else:
        shared = (int(frame_hw[0]), int(frame_hw[1]))
        hw_by_view = {"column": shared, "wrist": shared}
    views = pulse_result.get("views") if isinstance(pulse_result, Mapping) else None
    rois: dict[str, dict[str, Any]] = {}
    for view in ("column", "wrist"):
        source: Any
        if isinstance(views, Mapping) and isinstance(views.get(view), Mapping):
            source = views[view]
        elif view == "column":
            source = pulse_result
        else:
            source = {}
        raw = motion_gripper_roi_from_localization(
            source, hw_by_view.get(view, (224, 224)), expand_px=expand_px, view=view
        )
        assessed = assess_gripper_roi(
            raw.get("xyxy"),
            hw_by_view.get(view, (224, 224)),
            allow_full_image=False,
            provenance=str(raw.get("provenance") or MOTION_GRIPPER_ROI_PROVENANCE),
            require_complete_in_frame=bool(require_complete_in_frame),
            view=view,
        )
        diagnostics = dict(assessed.get("diagnostics") or {})
        diagnostics.update(raw.get("diagnostics") or {})
        diagnostics["view"] = view
        assessed["diagnostics"] = diagnostics
        assessed["view"] = view
        rois[view] = assessed
    return rois


def observe_self_contact_knife_edges(
    frame: np.ndarray,
    *,
    view: str,
    previous_frame: np.ndarray | None = None,
    gap_close_threshold_px: float = 1.0,
    gripper_roi_xyxy: Any = None,
    gripper_roi_provenance: str | None = None,
    allow_full_image: bool = False,
):
    """Image-only knife-edge observation for the opt-in self-contact path.

    This does not call ``select_live_moving_tip_*`` or ``pulse_tracks_allowed``
    gate 5, and it does not rank candidates by FK residual.  Formal callers
    must pass a pulse-derived ROI; ``allow_full_image`` is debug-only.
    ``u_hat`` is column-image-only.
    """

    from fcpg.so101_self_contact import observe_knife_edges

    return observe_knife_edges(
        frame,
        view=view,
        previous_frame=previous_frame,
        gap_close_threshold_px=gap_close_threshold_px,
        gripper_roi_xyxy=gripper_roi_xyxy,
        gripper_roi_provenance=gripper_roi_provenance,
        allow_full_image=allow_full_image,
    )
