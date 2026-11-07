"""FK-flow consistency test-time adaptation (TTA) core module.

Implements design document ``docs/tta_2b_design.md`` §1 (mechanism spec) for
the R0/R1 replay stages:

* Adjacent-round pair extraction from archived sessions
  (``outputs/lerobot/so101/<session>/rounds/r-<ns>-<hash>/plan.json``).  The
  expected image motion ``u_hat_fk`` is the *exact* finite difference of the
  archived per-round ``grounding[view].nominal`` projections (design §1.2 /
  line 21: the archived ``fk_delta_px`` slot is identically zero because the
  posterior chain was never enabled, so it must not be used).
* The eight hard gates of design §1.4 plus the §2.1 phase freeze
  (grasp / transport rounds never contribute supervision).
* A deterministic offline optical-flow front end (OpenCV DIS by default,
  RAFT behind an optional interface without a hard dependency): 64x64 crops
  centred at the corrected anchor position ``p_c(t0)``, forward/backward
  consistency, structure-tensor texture score, and differentiable flow
  sampling via :func:`fcpg.modules.sample_feature_map_at_pixels`.
* The consistency loss of design §1.4 (Huber kappa = 2 px, weights
  ``w_k = reliability.detach() * c_k * g_k``) with per-pair accept/reject
  decisions and machine-readable rejection reasons (fail-visible).
* The R1 learning machinery of design §1.5/§1.6: a zero-initialised
  ``nn.Linear(model_dim, 2)`` side head injected *inside* the tanh residual
  bound, single-step AdamW updates on a ring buffer, EMA inference weights,
  a hard Frobenius trust region on the side-head weight, adapt/holdout
  splitting with automatic rollback, and a divergence monitor.

The module never performs a model-backbone forward pass itself: the R0
measurement path runs from archived JSON + JPEG only, and the R1 path
consumes cached ``keypoint_latent`` / pre-tanh logits supplied by the caller.
Nothing here touches hardware, services, or GPUs.
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import nn

from fcpg.modules import sample_feature_map_at_pixels

__all__ = [
    "SO101_ANCHOR_ORDER",
    "DEFAULT_CROP_SIZE",
    "REASON_DT_EXCEEDED",
    "REASON_PHASE_FROZEN",
    "REASON_NO_ANCHOR_PASSED",
    "REASON_INVALID_ANCHOR",
    "REASON_MOTION_BELOW_MIN",
    "REASON_MOTION_ABOVE_MAX",
    "REASON_MARGIN_OUT_OF_BOUNDS",
    "REASON_TARGET_OVERLAP",
    "REASON_FB_ERROR_EXCEEDED",
    "REASON_LOW_TEXTURE",
    "REASON_ZERO_WEIGHTS",
    "RoundObservation",
    "RoundPair",
    "PairExtractionResult",
    "GateConfig",
    "GateDecision",
    "AnchorFlowMeasurement",
    "PairMeasurement",
    "DISFlowBackend",
    "RAFTFlowBackend",
    "create_flow_backend",
    "parse_round_timestamp_ns",
    "load_round_observation",
    "iter_adjacent_round_pairs",
    "load_round_image_gray",
    "structure_tensor_min_eigenvalue",
    "sample_flow_at",
    "forward_backward_error_px",
    "compute_anchor_flow",
    "huber_penalty",
    "flow_consistency_loss",
    "FlowConsistencyPipeline",
    "TTAUpdateConfig",
    "TTAPairSample",
    "UpdateResult",
    "HoldoutResult",
    "TTAFlowAdapter",
    "compute_pretanh_logits",
    "adapt_corrector_output",
    "make_pair_sample",
]


SO101_ANCHOR_ORDER = ("center", "fixed_tip", "moving_tip")
NUM_ANCHORS = len(SO101_ANCHOR_ORDER)
DEFAULT_CROP_SIZE = 64

# --- rejection reason codes (fail-visible vocabulary; design §1.4 order) ---
REASON_DT_EXCEEDED = "dt_exceeded"  # gate 7
REASON_PHASE_FROZEN = "phase_frozen"  # design §2.1 (grasp / holding freeze)
REASON_NO_ANCHOR_PASSED = "no_anchor_passed"
REASON_INVALID_ANCHOR = "invalid_anchor"  # gate 8
REASON_MOTION_BELOW_MIN = "motion_below_min"  # gate 1
REASON_MOTION_ABOVE_MAX = "motion_above_max"  # gate 2
REASON_MARGIN_OUT_OF_BOUNDS = "margin_out_of_bounds"  # gate 3
REASON_TARGET_OVERLAP = "target_overlap"  # gate 6
REASON_FB_ERROR_EXCEEDED = "fb_error_exceeded"  # gate 4
REASON_LOW_TEXTURE = "low_texture"  # gate 5
REASON_ZERO_WEIGHTS = "zero_weights"  # w_k = rbar*c*g all zero => drop pair

_ROUND_DIR_PATTERN = re.compile(r"^r-(\d+)-[0-9a-f]+$")


# ---------------------------------------------------------------------------
# Archived-session round loading and adjacent-pair extraction (design §4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundObservation:
    """One archived planning round for a single camera view."""

    round_id: str
    timestamp_ns: int
    round_dir: Path
    view: str
    state: np.ndarray  # [6] normalized joint state (plan.json:current_state)
    phase: str | None
    nominal_px: np.ndarray  # [3, 2]
    corrected_px: np.ndarray  # [3, 2]
    valid: np.ndarray  # [3] bool
    target_bboxes_xyxy: tuple[tuple[float, float, float, float], ...]
    image_path: Path

    @property
    def delta_px(self) -> np.ndarray:
        """Archived frozen-model residual ``corrected - nominal`` [3, 2]."""

        return self.corrected_px - self.nominal_px


@dataclass(frozen=True)
class RoundPair:
    """Two adjacent rounds of the same session and view (t0 < t1)."""

    t0: RoundObservation
    t1: RoundObservation
    dt_s: float
    u_hat_fk: np.ndarray  # [3, 2] exact nominal difference (design §1.2)


@dataclass(frozen=True)
class PairExtractionResult:
    pairs: tuple[RoundPair, ...]
    skipped_rounds: tuple[tuple[str, str], ...]  # (round_dir_name, reason)


def parse_round_timestamp_ns(round_dir_name: str) -> int:
    """Parse the nanosecond timestamp out of an ``r-<ns>-<hash>`` name."""

    match = _ROUND_DIR_PATTERN.match(round_dir_name)
    if match is None:
        raise ValueError(f"not a round directory name: {round_dir_name!r}")
    return int(match.group(1))


def _as_points(value: object, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.shape != (NUM_ANCHORS, 2):
        raise ValueError(f"{name} must have shape [3, 2], got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains non-finite values")
    return points


def load_round_observation(round_dir: Path, view: str) -> RoundObservation:
    """Load one round's plan.json into a :class:`RoundObservation`.

    Raises ``ValueError`` on malformed rounds (missing grounding for the
    view, wrong anchor count, missing raw image) so callers can record the
    skip reason instead of silently ingesting bad data.
    """

    round_dir = Path(round_dir)
    timestamp_ns = parse_round_timestamp_ns(round_dir.name)
    plan_path = round_dir / "plan.json"
    if not plan_path.is_file():
        raise ValueError("missing plan.json")
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)

    grounding = plan.get("grounding") or {}
    view_grounding = grounding.get(view)
    if not isinstance(view_grounding, dict):
        raise ValueError(f"grounding missing view {view!r}")
    nominal = _as_points(view_grounding.get("nominal"), f"grounding[{view}].nominal")
    corrected = _as_points(
        view_grounding.get("corrected"), f"grounding[{view}].corrected"
    )
    valid_raw = view_grounding.get("valid")
    if not isinstance(valid_raw, list) or len(valid_raw) != NUM_ANCHORS:
        raise ValueError(f"grounding[{view}].valid must be a length-3 list")
    valid = np.asarray([bool(v) for v in valid_raw], dtype=bool)

    state = np.asarray(plan.get("current_state"), dtype=np.float64)
    if state.shape != (6,):
        raise ValueError("current_state must have shape [6]")

    bboxes: list[tuple[float, float, float, float]] = []
    targets = plan.get("targets")
    if isinstance(targets, dict):
        for entry in targets.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("view") != view or not entry.get("visible"):
                continue
            bbox = entry.get("bbox_xyxy")
            if bbox is None:
                continue
            values = tuple(float(v) for v in bbox)
            if len(values) == 4:
                bboxes.append(values)  # type: ignore[arg-type]

    image_path = round_dir / f"{view}_raw.jpg"
    if not image_path.is_file():
        raise ValueError(f"missing {image_path.name}")

    phase = plan.get("phase")
    return RoundObservation(
        round_id=str(plan.get("round_id", round_dir.name)),
        timestamp_ns=timestamp_ns,
        round_dir=round_dir,
        view=view,
        state=state,
        phase=str(phase) if phase is not None else None,
        nominal_px=nominal,
        corrected_px=corrected,
        valid=valid,
        target_bboxes_xyxy=tuple(bboxes),
        image_path=image_path,
    )


def iter_adjacent_round_pairs(session_dir: Path, view: str) -> PairExtractionResult:
    """Extract adjacent-round pairs from one archived session directory.

    Rounds are ordered by the monotonic nanosecond timestamp embedded in the
    round directory name.  Every adjacent pair is returned; gate evaluation
    (including the dt gate) is a separate step so rejected pairs remain
    visible with reasons.  ``u_hat_fk`` is the exact difference of archived
    nominal projections (design §1.2; the ``fk_delta_px`` slot is unusable).
    """

    session_dir = Path(session_dir)
    rounds_dir = session_dir / "rounds"
    if not rounds_dir.is_dir():
        raise ValueError(f"no rounds/ directory under {session_dir}")

    entries: list[tuple[int, Path]] = []
    skipped: list[tuple[str, str]] = []
    for child in sorted(rounds_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            entries.append((parse_round_timestamp_ns(child.name), child))
        except ValueError:
            skipped.append((child.name, "unrecognized_round_dir_name"))
    entries.sort(key=lambda item: item[0])

    observations: list[RoundObservation] = []
    for _, child in entries:
        try:
            observations.append(load_round_observation(child, view))
        except (ValueError, json.JSONDecodeError) as exc:
            skipped.append((child.name, f"load_error: {exc}"))

    pairs: list[RoundPair] = []
    for t0, t1 in zip(observations, observations[1:]):
        dt_s = (t1.timestamp_ns - t0.timestamp_ns) / 1e9
        pairs.append(
            RoundPair(
                t0=t0,
                t1=t1,
                dt_s=dt_s,
                u_hat_fk=t1.nominal_px - t0.nominal_px,
            )
        )
    return PairExtractionResult(pairs=tuple(pairs), skipped_rounds=tuple(skipped))


def load_round_image_gray(image_path: Path) -> np.ndarray:
    """Load an archived raw JPEG as a uint8 grayscale array [H, W]."""

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read image {image_path}")
    return image


# ---------------------------------------------------------------------------
# Hard gates (design §1.4 items 1-8 plus §2.1 phase freeze)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for the eight hard gates of design §1.4.

    All values are the design-document draft numbers; the texture threshold
    is a draft default (``prereg_status: draft``) because §1.4 fixes the
    statistic (structure-tensor minimum eigenvalue) but not the number.
    """

    min_motion_px: float = 2.0  # gate 1
    max_motion_px: float = 40.0  # gate 2
    margin_px: float = 16.0  # gate 3
    max_fb_error_px: float = 3.0  # gate 4
    min_texture_eigenvalue: float = 1e-4  # gate 5 (gray in [0,1]; draft)
    max_target_overlap: float = 0.5  # gate 6 (crop-area fraction)
    max_dt_s: float = 30.0  # gate 7
    freeze_phases: tuple[str, ...] = ("grasp", "transport")  # design §2.1
    sigma_fb_px: float = 1.0  # c_k = exp(-e_fb^2 / (2 sigma_fb^2))


@dataclass(frozen=True)
class GateDecision:
    """Kinematic (pre-flow) gate outcome for one pair."""

    pair_passed: bool
    anchor_passed: np.ndarray  # [3] bool
    pair_reasons: tuple[str, ...]
    anchor_reasons: tuple[tuple[str, ...], ...]  # length 3


def _point_within_margin(
    point: np.ndarray, width: int, height: int, margin: float
) -> bool:
    x, y = float(point[0]), float(point[1])
    return (
        math.isfinite(x)
        and math.isfinite(y)
        and margin <= x <= float(width - 1) - margin
        and margin <= y <= float(height - 1) - margin
    )


def clamp_crop_origin(
    center_px: np.ndarray, width: int, height: int, crop_size: int
) -> np.ndarray:
    """Integer crop origin so the crop stays inside the image (design §1.3)."""

    ox = int(round(float(center_px[0]))) - crop_size // 2
    oy = int(round(float(center_px[1]))) - crop_size // 2
    ox = min(max(ox, 0), max(width - crop_size, 0))
    oy = min(max(oy, 0), max(height - crop_size, 0))
    return np.asarray([ox, oy], dtype=np.int64)


def _crop_target_overlap_fraction(
    origin: np.ndarray,
    crop_size: int,
    bboxes: Sequence[tuple[float, float, float, float]],
) -> float:
    if not bboxes:
        return 0.0
    cx0, cy0 = float(origin[0]), float(origin[1])
    cx1, cy1 = cx0 + crop_size, cy0 + crop_size
    crop_area = float(crop_size * crop_size)
    best = 0.0
    for x0, y0, x1, y1 in bboxes:
        ix0, iy0 = max(cx0, x0), max(cy0, y0)
        ix1, iy1 = min(cx1, x1), min(cy1, y1)
        if ix1 > ix0 and iy1 > iy0:
            best = max(best, (ix1 - ix0) * (iy1 - iy0) / crop_area)
    return best


def evaluate_kinematic_gates(
    pair: RoundPair,
    config: GateConfig,
    image_size: tuple[int, int],
    crop_size: int = DEFAULT_CROP_SIZE,
) -> GateDecision:
    """Evaluate the gates that need no optical flow (gates 1,2,3,6,7,8 + phase).

    Flow-dependent gates (4 FB, 5 texture) are applied later by the pipeline
    on anchors that survive this stage.  Every failed check is recorded as a
    machine-readable reason (fail-visible discipline).
    """

    width, height = image_size
    pair_reasons: list[str] = []
    if pair.dt_s > config.max_dt_s or pair.dt_s < 0.0:
        pair_reasons.append(REASON_DT_EXCEEDED)
    for obs in (pair.t0, pair.t1):
        if obs.phase is not None and obs.phase in config.freeze_phases:
            pair_reasons.append(REASON_PHASE_FROZEN)
            break

    anchor_passed = np.zeros(NUM_ANCHORS, dtype=bool)
    anchor_reasons: list[tuple[str, ...]] = []
    for k in range(NUM_ANCHORS):
        reasons: list[str] = []
        if not (bool(pair.t0.valid[k]) and bool(pair.t1.valid[k])):
            reasons.append(REASON_INVALID_ANCHOR)
        motion = float(np.linalg.norm(pair.u_hat_fk[k]))
        if motion < config.min_motion_px:
            reasons.append(REASON_MOTION_BELOW_MIN)
        if motion > config.max_motion_px:
            reasons.append(REASON_MOTION_ABOVE_MAX)
        p_c0 = pair.t0.corrected_px[k]
        if not (
            _point_within_margin(p_c0, width, height, config.margin_px)
            and _point_within_margin(
                p_c0 + pair.u_hat_fk[k], width, height, config.margin_px
            )
        ):
            reasons.append(REASON_MARGIN_OUT_OF_BOUNDS)
        origin = clamp_crop_origin(p_c0, width, height, crop_size)
        overlap = _crop_target_overlap_fraction(
            origin, crop_size, pair.t0.target_bboxes_xyxy
        )
        if overlap > config.max_target_overlap:
            reasons.append(REASON_TARGET_OVERLAP)
        anchor_passed[k] = not reasons and not pair_reasons
        anchor_reasons.append(tuple(reasons))

    if not pair_reasons and not anchor_passed.any():
        pair_reasons.append(REASON_NO_ANCHOR_PASSED)
    return GateDecision(
        pair_passed=not pair_reasons and bool(anchor_passed.any()),
        anchor_passed=anchor_passed,
        pair_reasons=tuple(pair_reasons),
        anchor_reasons=tuple(anchor_reasons),
    )


# ---------------------------------------------------------------------------
# Optical-flow backends and per-anchor crop flow (design §1.3)
# ---------------------------------------------------------------------------


class DISFlowBackend:
    """Deterministic OpenCV DIS dense optical flow (offline default)."""

    name = "dis"

    def __init__(self, preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) -> None:
        self._preset = preset

    def compute(self, gray0: np.ndarray, gray1: np.ndarray) -> np.ndarray:
        """Dense flow gray0 -> gray1; returns float32 [H, W, 2] (dx, dy)."""

        if gray0.shape != gray1.shape or gray0.ndim != 2:
            raise ValueError("flow inputs must be two equal-shape 2D arrays")
        # A fresh instance per call keeps the backend stateless/deterministic.
        dis = cv2.DISOpticalFlow_create(self._preset)
        flow = dis.calc(
            np.ascontiguousarray(_to_uint8_gray(gray0)),
            np.ascontiguousarray(_to_uint8_gray(gray1)),
            None,
        )
        return np.asarray(flow, dtype=np.float32)


class RAFTFlowBackend:
    """Optional RAFT backend interface (no hard dependency, design §1.3).

    Instantiation fails with an actionable error when torchvision's RAFT is
    unavailable; the rest of the module never imports it.
    """

    name = "raft"

    def __init__(self, device: str = "cpu") -> None:
        try:
            from torchvision.models.optical_flow import (
                Raft_Small_Weights,
                raft_small,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "RAFT flow backend requires torchvision with optical-flow "
                "models; install it or use the default 'dis' backend"
            ) from exc
        weights = Raft_Small_Weights.DEFAULT
        self._transforms = weights.transforms()
        self._model = raft_small(weights=weights).eval().to(device)
        self._device = device

    def compute(self, gray0: np.ndarray, gray1: np.ndarray) -> np.ndarray:
        if gray0.shape != gray1.shape or gray0.ndim != 2:
            raise ValueError("flow inputs must be two equal-shape 2D arrays")
        rgb0 = np.repeat(_to_uint8_gray(gray0)[..., None], 3, axis=-1)
        rgb1 = np.repeat(_to_uint8_gray(gray1)[..., None], 3, axis=-1)
        t0 = torch.from_numpy(rgb0).permute(2, 0, 1)[None]
        t1 = torch.from_numpy(rgb1).permute(2, 0, 1)[None]
        t0, t1 = self._transforms(t0, t1)
        with torch.no_grad():
            flows = self._model(t0.to(self._device), t1.to(self._device))
        flow = flows[-1][0].permute(1, 2, 0).cpu().numpy()
        return np.asarray(flow, dtype=np.float32)


def create_flow_backend(name: str = "dis", **kwargs: object):
    """Factory for flow backends; 'dis' is the deterministic offline default."""

    key = name.lower()
    if key == "dis":
        return DISFlowBackend(**kwargs)  # type: ignore[arg-type]
    if key == "raft":
        return RAFTFlowBackend(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown flow backend {name!r} (expected 'dis' or 'raft')")


def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("expected a single-channel grayscale image")
    if array.dtype == np.uint8:
        return array
    scaled = array
    if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array, initial=0.0)) <= 1.0:
        scaled = array * 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def structure_tensor_min_eigenvalue(gray_crop: np.ndarray) -> float:
    """Minimum eigenvalue of the aggregated crop structure tensor (gate 5).

    Gray values are normalised to [0, 1]; gradients are central differences.
    This is the Lucas-Kanade conditioning statistic: a small value means the
    aperture problem makes the flow direction unobservable in this crop.
    """

    gray = _to_uint8_gray(gray_crop).astype(np.float64) / 255.0
    gy, gx = np.gradient(gray)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    trace_half = 0.5 * (jxx + jyy)
    root = math.sqrt(max((0.5 * (jxx - jyy)) ** 2 + jxy**2, 0.0))
    return max(trace_half - root, 0.0)


def sample_flow_at(flow_fields: torch.Tensor, points_px: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample flow vectors at crop-frame pixel coordinates.

    Args:
        flow_fields: ``[K, 2, S, S]`` flow crops (channels = dx, dy).
        points_px: ``[K, 2]`` sample positions in crop pixel coordinates.

    Returns:
        ``[K, 2]`` flow vectors.  Differentiable w.r.t. ``points_px`` — this
        is the gauge-breaking gradient path of design §1.3 and must not be
        detached by callers.
    """

    if flow_fields.ndim != 4 or flow_fields.shape[1] != 2:
        raise ValueError("flow_fields must have shape [K, 2, S, S]")
    if points_px.ndim != 2 or points_px.shape != (flow_fields.shape[0], 2):
        raise ValueError("points_px must have shape [K, 2] matching flow_fields")
    size = (int(flow_fields.shape[-1]), int(flow_fields.shape[-2]))
    sampled = sample_feature_map_at_pixels(
        flow_fields, points_px.unsqueeze(1), image_size=size
    )
    return sampled[:, 0, :]


def forward_backward_error_px(
    flow_forward: np.ndarray,
    flow_backward: np.ndarray,
    point_px: np.ndarray,
) -> float:
    """Forward-backward flow inconsistency at one crop-frame point (gate 4)."""

    fwd = torch.from_numpy(np.ascontiguousarray(flow_forward)).permute(2, 0, 1)[None]
    bwd = torch.from_numpy(np.ascontiguousarray(flow_backward)).permute(2, 0, 1)[None]
    point = torch.as_tensor(point_px, dtype=fwd.dtype).reshape(1, 2)
    u_f = sample_flow_at(fwd.float(), point.float())
    end = point.float() + u_f
    size = float(flow_forward.shape[0] - 1)
    end = end.clamp(min=0.0, max=size)
    u_b = sample_flow_at(bwd.float(), end)
    return float(torch.linalg.vector_norm(u_f + u_b))


@dataclass(frozen=True)
class AnchorFlowMeasurement:
    """Per-anchor crop-flow measurement at the corrected position p_c(t0)."""

    crop_origin: np.ndarray  # [2] int
    flow_forward: np.ndarray  # [S, S, 2] float32
    fb_error_px: float
    texture_min_eigenvalue: float
    u_px: np.ndarray  # [2] flow sampled at p_c(t0)


def compute_anchor_flow(
    image0_gray: np.ndarray,
    image1_gray: np.ndarray,
    center_px: np.ndarray,
    backend,
    crop_size: int = DEFAULT_CROP_SIZE,
) -> AnchorFlowMeasurement:
    """Crop 64x64 at ``center_px`` (clamped), compute fwd/bwd DIS flow.

    ``u_px`` is the forward flow sampled exactly at ``center_px`` (crop
    frame), matching design §1.3's measurement definition.
    """

    height, width = image0_gray.shape[:2]
    origin = clamp_crop_origin(np.asarray(center_px, dtype=np.float64), width, height, crop_size)
    ox, oy = int(origin[0]), int(origin[1])
    crop0 = _to_uint8_gray(image0_gray)[oy : oy + crop_size, ox : ox + crop_size]
    crop1 = _to_uint8_gray(image1_gray)[oy : oy + crop_size, ox : ox + crop_size]
    flow_fwd = backend.compute(crop0, crop1)
    flow_bwd = backend.compute(crop1, crop0)
    point = np.asarray(
        [float(center_px[0]) - ox, float(center_px[1]) - oy], dtype=np.float64
    )
    fb_error = forward_backward_error_px(flow_fwd, flow_bwd, point)
    texture = structure_tensor_min_eigenvalue(crop0)
    fwd_tensor = torch.from_numpy(np.ascontiguousarray(flow_fwd)).permute(2, 0, 1)[None]
    u = sample_flow_at(
        fwd_tensor.float(), torch.as_tensor(point, dtype=torch.float32).reshape(1, 2)
    )[0].numpy()
    return AnchorFlowMeasurement(
        crop_origin=origin,
        flow_forward=flow_fwd,
        fb_error_px=float(fb_error),
        texture_min_eigenvalue=float(texture),
        u_px=np.asarray(u, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Consistency loss (design §1.4)
# ---------------------------------------------------------------------------


def huber_penalty(residual_norm: torch.Tensor, kappa: float = 2.0) -> torch.Tensor:
    """Huber penalty rho_H on residual norms (kappa in pixels)."""

    if kappa <= 0:
        raise ValueError("kappa must be positive")
    quadratic = 0.5 * residual_norm**2
    linear = kappa * (residual_norm - 0.5 * kappa)
    return torch.where(residual_norm <= kappa, quadratic, linear)


def flow_consistency_loss(
    residuals: torch.Tensor,
    weights: torch.Tensor,
    kappa: float = 2.0,
) -> torch.Tensor:
    """Weighted Huber consistency loss  sum_k w_k rho_H(||r_k||) / sum_k w_k.

    Args:
        residuals: ``[K, 2]`` per-anchor residual vectors.
        weights: ``[K]`` non-negative weights ``w_k = rbar_k * c_k * g_k``
            (reliability already detached by the caller; never a graph leaf).
    """

    if residuals.ndim != 2 or residuals.shape[-1] != 2:
        raise ValueError("residuals must have shape [K, 2]")
    if weights.shape != residuals.shape[:1]:
        raise ValueError("weights must have shape [K]")
    weights = weights.detach()
    total = weights.sum()
    if float(total) <= 0.0:
        raise ValueError("all weights are zero; the pair must be rejected upstream")
    norms = torch.linalg.vector_norm(residuals, dim=-1)
    return (weights * huber_penalty(norms, kappa)).sum() / total


# ---------------------------------------------------------------------------
# R0 measurement pipeline (no learning, design §4.2 R0)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairMeasurement:
    """Full fail-visible record for one adjacent-round pair."""

    view: str
    round_id_t0: str
    round_id_t1: str
    dt_s: float
    accepted: bool
    anchor_passed: np.ndarray  # [3] bool, final (kinematic + flow gates)
    pair_reasons: tuple[str, ...]
    anchor_reasons: tuple[tuple[str, ...], ...]
    u_hat_fk: np.ndarray  # [3, 2]
    u_px: np.ndarray  # [3, 2]; NaN where flow was not computed
    cos_uhat_u: np.ndarray  # [3]; NaN where undefined
    magnitude_ratio: np.ndarray  # [3] ||u|| / ||u_hat||; NaN where undefined
    fb_error_px: np.ndarray  # [3]; NaN where not computed
    texture_min_eigenvalue: np.ndarray  # [3]; NaN where not computed
    weights: np.ndarray  # [3] w_k = rbar * c * g
    crop_origins: np.ndarray  # [3, 2] int; -1 where not computed
    loss_flow: float  # NaN when rejected
    flow_fields: torch.Tensor | None = None  # [3, 2, S, S] when retained


class FlowConsistencyPipeline:
    """R0/R1 measurement front end: gates + crop flow + consistency loss.

    The pipeline is model-free: the archived ``corrected`` grounding plays
    the role of the frozen-model output, so the R0 path needs no GPU and no
    checkpoint.  Reliability weights default to 1 (LeWM residual-only
    architecture, design §1.4).
    """

    def __init__(
        self,
        gate_config: GateConfig | None = None,
        flow_backend=None,
        crop_size: int = DEFAULT_CROP_SIZE,
        huber_kappa_px: float = 2.0,
    ) -> None:
        self.gate_config = gate_config or GateConfig()
        self.flow_backend = flow_backend or DISFlowBackend()
        self.crop_size = int(crop_size)
        self.huber_kappa_px = float(huber_kappa_px)

    def measure_pair(
        self,
        pair: RoundPair,
        image0_gray: np.ndarray,
        image1_gray: np.ndarray,
        reliability: np.ndarray | None = None,
        keep_flow_fields: bool = False,
    ) -> PairMeasurement:
        """Measure one adjacent pair; never raises on gate rejection."""

        if image0_gray.shape != image1_gray.shape:
            raise ValueError("pair images must share the same shape")
        height, width = image0_gray.shape[:2]
        decision = evaluate_kinematic_gates(
            pair, self.gate_config, (width, height), self.crop_size
        )
        rbar = (
            np.ones(NUM_ANCHORS, dtype=np.float64)
            if reliability is None
            else np.asarray(reliability, dtype=np.float64)
        )
        if rbar.shape != (NUM_ANCHORS,):
            raise ValueError("reliability must have shape [3]")

        u_px = np.full((NUM_ANCHORS, 2), np.nan)
        cos_vals = np.full(NUM_ANCHORS, np.nan)
        ratio_vals = np.full(NUM_ANCHORS, np.nan)
        fb_vals = np.full(NUM_ANCHORS, np.nan)
        tex_vals = np.full(NUM_ANCHORS, np.nan)
        weights = np.zeros(NUM_ANCHORS, dtype=np.float64)
        origins = np.full((NUM_ANCHORS, 2), -1, dtype=np.int64)
        flow_stack = (
            torch.zeros(NUM_ANCHORS, 2, self.crop_size, self.crop_size)
            if keep_flow_fields
            else None
        )

        anchor_passed = decision.anchor_passed.copy()
        anchor_reasons = [list(r) for r in decision.anchor_reasons]
        if anchor_passed.any():
            for k in np.flatnonzero(anchor_passed):
                measurement = compute_anchor_flow(
                    image0_gray,
                    image1_gray,
                    pair.t0.corrected_px[k],
                    self.flow_backend,
                    self.crop_size,
                )
                origins[k] = measurement.crop_origin
                fb_vals[k] = measurement.fb_error_px
                tex_vals[k] = measurement.texture_min_eigenvalue
                u_px[k] = measurement.u_px
                if flow_stack is not None:
                    flow_stack[k] = torch.from_numpy(
                        np.ascontiguousarray(measurement.flow_forward)
                    ).permute(2, 0, 1)
                if measurement.fb_error_px > self.gate_config.max_fb_error_px:
                    anchor_reasons[k].append(REASON_FB_ERROR_EXCEEDED)
                    anchor_passed[k] = False
                if (
                    measurement.texture_min_eigenvalue
                    < self.gate_config.min_texture_eigenvalue
                ):
                    anchor_reasons[k].append(REASON_LOW_TEXTURE)
                    anchor_passed[k] = False
                u_hat_norm = float(np.linalg.norm(pair.u_hat_fk[k]))
                u_norm = float(np.linalg.norm(measurement.u_px))
                if u_hat_norm > 0.0 and u_norm > 0.0:
                    cos_vals[k] = float(
                        np.dot(pair.u_hat_fk[k], measurement.u_px)
                        / (u_hat_norm * u_norm)
                    )
                if u_hat_norm > 0.0:
                    ratio_vals[k] = u_norm / u_hat_norm
                if anchor_passed[k]:
                    c_k = math.exp(
                        -(measurement.fb_error_px**2)
                        / (2.0 * self.gate_config.sigma_fb_px**2)
                    )
                    weights[k] = float(rbar[k]) * c_k

        pair_reasons = list(decision.pair_reasons)
        if not pair_reasons and not anchor_passed.any():
            pair_reasons.append(REASON_NO_ANCHOR_PASSED)
        if not pair_reasons and float(weights[anchor_passed].sum()) <= 0.0:
            pair_reasons.append(REASON_ZERO_WEIGHTS)
        accepted = not pair_reasons and bool(anchor_passed.any())

        loss = float("nan")
        if accepted:
            delta0 = pair.t0.delta_px
            delta1 = pair.t1.delta_px
            residual = pair.u_hat_fk + delta1 - delta0 - u_px
            mask = anchor_passed & np.isfinite(residual).all(axis=-1)
            loss = float(
                flow_consistency_loss(
                    torch.as_tensor(residual[mask], dtype=torch.float64),
                    torch.as_tensor(weights[mask], dtype=torch.float64),
                    kappa=self.huber_kappa_px,
                )
            )

        return PairMeasurement(
            view=pair.t0.view,
            round_id_t0=pair.t0.round_id,
            round_id_t1=pair.t1.round_id,
            dt_s=pair.dt_s,
            accepted=accepted,
            anchor_passed=anchor_passed,
            pair_reasons=tuple(pair_reasons),
            anchor_reasons=tuple(tuple(r) for r in anchor_reasons),
            u_hat_fk=np.asarray(pair.u_hat_fk, dtype=np.float64),
            u_px=u_px,
            cos_uhat_u=cos_vals,
            magnitude_ratio=ratio_vals,
            fb_error_px=fb_vals,
            texture_min_eigenvalue=tex_vals,
            weights=weights,
            crop_origins=origins,
            loss_flow=loss,
            flow_fields=flow_stack,
        )


# ---------------------------------------------------------------------------
# R1 learning machinery: zero-init side head, AdamW, EMA, trust region,
# adapt/holdout rollback (design §1.5 / §1.6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TTAUpdateConfig:
    """Design §1.5/§1.6 draft hyper-parameters (grid frozen there for R1)."""

    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    lambda_trust: float = 1e-3  # soft trust region L2 on W_tta (design §1.4)
    tau_w: float = 0.5  # hard Frobenius trust region (grid {0.25, 0.5, 1.0})
    ema_decay: float = 0.98
    huber_kappa_px: float = 2.0
    buffer_capacity: int = 16  # ring buffer K per view
    minibatch_size: int = 8
    holdout_window: int = 10
    rollback_ratio: float = 1.25
    cooldown_rounds: int = 50
    divergence_p99_px: float = 8.0
    divergence_window: int = 256
    posterior_reset_ema_delta: float = 0.01  # design §1.6 item 7
    seed: int = 0


@dataclass(frozen=True)
class TTAPairSample:
    """Cached tensors for one accepted pair, ready for side-head updates.

    All tensors are detached constants except through the documented
    gradient paths: the side head output and the flow sampling position.
    """

    latent_t0: torch.Tensor  # [N, D] keypoint latents at t0 (detached)
    latent_t1: torch.Tensor  # [N, D]
    z_t0: torch.Tensor  # [N, 2] frozen pre-tanh logits at t0
    z_t1: torch.Tensor  # [N, 2]
    p_nom_t0: torch.Tensor  # [N, 2] nominal projection at t0
    u_hat_fk: torch.Tensor  # [N, 2]
    flow_fields: torch.Tensor  # [N, 2, S, S] forward flow crops
    crop_origins: torch.Tensor  # [N, 2]
    base_weights: torch.Tensor  # [N] w_k = rbar * c * g (zeros where gated)


@dataclass(frozen=True)
class UpdateResult:
    stepped: bool
    loss: float
    weight_norm: float
    trust_region_projected: bool
    rolled_back: bool
    reason: str | None = None


@dataclass(frozen=True)
class HoldoutResult:
    residual_adapted: float
    residual_frozen: float
    ratio: float
    rolled_back: bool


def compute_pretanh_logits(corrector, keypoint_latent: torch.Tensor) -> torch.Tensor:
    """Recompute frozen pre-tanh residual logits from cached latents.

    ``corrector.delta_head`` is a public attribute of every grounding
    corrector in ``fcpg/transformer_grounding.py`` (design §0), so this does
    not modify or subclass existing files.
    """

    with torch.no_grad():
        return corrector.delta_head(keypoint_latent.detach())


class TTAFlowAdapter(nn.Module):
    """Zero-initialised residual side head with the §1.6 protection stack.

    Owns exactly one trainable module: ``tta_head = nn.Linear(model_dim, 2)``
    (386 parameters at model_dim=192, 514 at 256).  The backbone, the
    original ``delta_head`` and any ``reliability_head`` are never touched;
    reliability enters only as detached weights inside
    :class:`TTAPairSample`.  Inference uses EMA weights (Mean-Teacher style);
    updates use instantaneous weights.
    """

    def __init__(
        self,
        model_dim: int,
        max_residual_px: float = 20.0,
        config: TTAUpdateConfig | None = None,
    ) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if max_residual_px <= 0:
            raise ValueError("max_residual_px must be positive")
        self.model_dim = int(model_dim)
        self.max_residual_px = float(max_residual_px)
        self.config = config or TTAUpdateConfig()
        self.tta_head = nn.Linear(self.model_dim, 2)
        nn.init.zeros_(self.tta_head.weight)
        nn.init.zeros_(self.tta_head.bias)
        self.register_buffer("ema_weight", torch.zeros(2, self.model_dim))
        self.register_buffer("ema_bias", torch.zeros(2))
        self._optimizer = self._build_optimizer()
        self._rng = np.random.default_rng(self.config.seed)
        self._adapt_buffer: deque[TTAPairSample] = deque(
            maxlen=self.config.buffer_capacity
        )
        self._holdout_ratios: deque[float] = deque(maxlen=self.config.holdout_window)
        self._divergence_window: deque[float] = deque(
            maxlen=self.config.divergence_window
        )
        self._accepted_pair_count = 0
        self.cooldown_remaining = 0
        self.rollback_count = 0
        self.events: list[dict] = []
        self._posterior_reset_pending = False
        self._ema_signal_reference = torch.zeros(2, self.model_dim)

    # -- construction helpers -------------------------------------------------

    def _build_optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.tta_head.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )

    # -- inference ------------------------------------------------------------

    def adapted_delta(
        self,
        keypoint_latent: torch.Tensor,
        pre_tanh_logits: torch.Tensor,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """delta_adapted = tanh(z.detach() + tta_head(latent.detach())) * max_r.

        Injection happens inside the tanh, so the +/- ``max_residual_px``
        envelope holds by construction (design §1.5).  With zero side-head
        weights the result is bitwise identical to the frozen output.
        """

        latent = keypoint_latent.detach()
        z = pre_tanh_logits.detach()
        if use_ema:
            dz = torch.nn.functional.linear(
                latent, self.ema_weight.to(latent.dtype), self.ema_bias.to(latent.dtype)
            )
        else:
            dz = self.tta_head(latent)
        return torch.tanh(z + dz) * self.max_residual_px

    def is_identity(self) -> bool:
        """True when both instantaneous and EMA heads are exactly zero."""

        return (
            not bool(self.tta_head.weight.detach().any())
            and not bool(self.tta_head.bias.detach().any())
            and not bool(self.ema_weight.any())
            and not bool(self.ema_bias.any())
        )

    # -- loss -----------------------------------------------------------------

    def _head_output(
        self,
        latent: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if weight is None:
            return torch.zeros(
                *latent.shape[:-1], 2, dtype=latent.dtype, device=latent.device
            )
        return torch.nn.functional.linear(
            latent, weight.to(latent.dtype), None if bias is None else bias.to(latent.dtype)
        )

    def _loss_for_parameters(
        self,
        samples: Sequence[TTAPairSample],
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean over pairs of the §1.4 per-pair consistency loss.

        delta(t0) and delta(t1) are recomputed from the cached latents with
        the *given* head parameters (design §1.5: the t0 correction must be
        re-evaluated under current parameters or the sampling-position
        gradient path breaks and the gauge freedom reopens).
        """

        losses = []
        for sample in samples:
            mask = sample.base_weights > 0
            if not bool(mask.any()):
                continue
            dz0 = self._head_output(sample.latent_t0.detach(), weight, bias)
            dz1 = self._head_output(sample.latent_t1.detach(), weight, bias)
            delta0 = torch.tanh(sample.z_t0.detach() + dz0) * self.max_residual_px
            delta1 = torch.tanh(sample.z_t1.detach() + dz1) * self.max_residual_px
            p_c0 = sample.p_nom_t0 + delta0
            points = (p_c0 - sample.crop_origins.to(p_c0.dtype))[mask]
            u = sample_flow_at(sample.flow_fields[mask].to(p_c0.dtype), points)
            residual = (sample.u_hat_fk + delta1 - delta0)[mask] - u
            losses.append(
                flow_consistency_loss(
                    residual,
                    sample.base_weights[mask],
                    kappa=self.config.huber_kappa_px,
                )
            )
        if not losses:
            raise ValueError("no usable anchors in the provided samples")
        return torch.stack(losses).mean()

    def pair_loss(
        self, samples: Sequence[TTAPairSample], use_ema: bool = True
    ) -> torch.Tensor:
        """Consistency loss under the adapted head (no regulariser, no grad)."""

        with torch.no_grad():
            if use_ema:
                return self._loss_for_parameters(samples, self.ema_weight, self.ema_bias)
            return self._loss_for_parameters(
                samples, self.tta_head.weight, self.tta_head.bias
            )

    def frozen_loss(self, samples: Sequence[TTAPairSample]) -> torch.Tensor:
        with torch.no_grad():
            return self._loss_for_parameters(samples, None, None)

    # -- streaming interface ----------------------------------------------------

    def observe_pair(self, sample: TTAPairSample) -> UpdateResult | HoldoutResult:
        """Consume one *accepted* pair (design §1.5 trigger: 1 step per pair).

        Accepted pairs alternate adapt / holdout by parity (design §1.6
        item 5).  Holdout pairs never contribute gradients.
        """

        index = self._accepted_pair_count
        self._accepted_pair_count += 1
        if index % 2 == 1:
            return self._observe_holdout(sample)
        return self._observe_adapt(sample)

    def _observe_adapt(self, sample: TTAPairSample) -> UpdateResult:
        if self.cooldown_remaining > 0:
            self.events.append(
                {"event": "cooldown_skip", "cooldown_remaining": self.cooldown_remaining}
            )
            return UpdateResult(
                stepped=False,
                loss=float("nan"),
                weight_norm=self._weight_norm(),
                trust_region_projected=False,
                rolled_back=False,
                reason="cooldown",
            )
        self._adapt_buffer.append(sample)
        return self.update_step()

    def _observe_holdout(self, sample: TTAPairSample) -> HoldoutResult:
        if self.cooldown_remaining > 0:
            self.events.append({"event": "cooldown_holdout_skip"})
            return HoldoutResult(
                residual_adapted=float("nan"),
                residual_frozen=float("nan"),
                ratio=float("nan"),
                rolled_back=False,
            )
        return self.evaluate_holdout(sample)

    def update_step(self) -> UpdateResult:
        """One AdamW step on <=8 pairs sampled from the ring buffer."""

        if not self._adapt_buffer:
            return UpdateResult(
                stepped=False,
                loss=float("nan"),
                weight_norm=self._weight_norm(),
                trust_region_projected=False,
                rolled_back=False,
                reason="empty_buffer",
            )
        count = min(len(self._adapt_buffer), self.config.minibatch_size)
        indices = self._rng.choice(len(self._adapt_buffer), size=count, replace=False)
        samples = [self._adapt_buffer[int(i)] for i in indices]
        loss_flow = self._loss_for_parameters(
            samples, self.tta_head.weight, self.tta_head.bias
        )
        loss = loss_flow + self.config.lambda_trust * (
            self.tta_head.weight**2
        ).sum()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        projected = self._project_trust_region()
        self._update_ema()
        rolled_back = self._check_divergence(samples)
        return UpdateResult(
            stepped=True,
            loss=float(loss_flow.detach()),
            weight_norm=self._weight_norm(),
            trust_region_projected=projected,
            rolled_back=rolled_back,
            reason="divergence" if rolled_back else None,
        )

    def evaluate_holdout(self, sample: TTAPairSample) -> HoldoutResult:
        """Compare EMA-adapted vs frozen residual on a holdout pair.

        The rolling median ratio over the last ``holdout_window`` holdout
        pairs above ``rollback_ratio`` triggers a full fail-closed rollback
        (design §1.6 item 5).
        """

        eps = 1e-9
        try:
            res_adapted = float(self.pair_loss([sample], use_ema=True))
            res_frozen = float(self.frozen_loss([sample]))
        except ValueError:
            return HoldoutResult(
                residual_adapted=float("nan"),
                residual_frozen=float("nan"),
                ratio=float("nan"),
                rolled_back=False,
            )
        ratio = res_adapted / (res_frozen + eps)
        self._holdout_ratios.append(ratio)
        rolled_back = False
        if (
            len(self._holdout_ratios) >= self.config.holdout_window
            and float(np.median(list(self._holdout_ratios))) > self.config.rollback_ratio
        ):
            self.rollback("holdout_ratio")
            rolled_back = True
        return HoldoutResult(
            residual_adapted=res_adapted,
            residual_frozen=res_frozen,
            ratio=ratio,
            rolled_back=rolled_back,
        )

    def notify_round(self) -> None:
        """Advance the per-round cooldown clock (call once per planner round)."""

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    # -- protections ------------------------------------------------------------

    def _weight_norm(self) -> float:
        return float(torch.linalg.matrix_norm(self.tta_head.weight.detach()))

    def _project_trust_region(self) -> bool:
        """Hard Frobenius projection ||W_tta||_F <= tau_W (design §1.6 item 3)."""

        with torch.no_grad():
            norm = torch.linalg.matrix_norm(self.tta_head.weight)
            if float(norm) > self.config.tau_w:
                self.tta_head.weight.mul_(self.config.tau_w / float(norm))
                return True
        return False

    def _update_ema(self) -> None:
        decay = self.config.ema_decay
        with torch.no_grad():
            self.ema_weight.mul_(decay).add_(
                self.tta_head.weight.detach(), alpha=1.0 - decay
            )
            self.ema_bias.mul_(decay).add_(
                self.tta_head.bias.detach(), alpha=1.0 - decay
            )
            drift = torch.linalg.matrix_norm(
                self.ema_weight - self._ema_signal_reference
            )
            if float(drift) > self.config.posterior_reset_ema_delta:
                self._posterior_reset_pending = True
                self._ema_signal_reference = self.ema_weight.clone()

    def _check_divergence(self, samples: Sequence[TTAPairSample]) -> bool:
        """||delta_adapted - delta_frozen|| p99 > 8 px alarm (design §1.6 item 6)."""

        with torch.no_grad():
            for sample in samples:
                for latent, z in (
                    (sample.latent_t0, sample.z_t0),
                    (sample.latent_t1, sample.z_t1),
                ):
                    adapted = self.adapted_delta(latent, z, use_ema=True)
                    frozen = torch.tanh(z.detach()) * self.max_residual_px
                    norms = torch.linalg.vector_norm(adapted - frozen, dim=-1)
                    self._divergence_window.extend(float(v) for v in norms)
        if not self._divergence_window:
            return False
        p99 = float(np.percentile(list(self._divergence_window), 99))
        if p99 > self.config.divergence_p99_px:
            self.rollback("divergence_p99")
            return True
        return False

    def rollback(self, reason: str) -> None:
        """Fail-closed reset: W_tta <- 0, buffers cleared, cooldown armed.

        After a rollback the adapter is bitwise identical to the frozen
        model (design §1.6 item 5; audited by G2-c).
        """

        with torch.no_grad():
            self.tta_head.weight.zero_()
            self.tta_head.bias.zero_()
            self.ema_weight.zero_()
            self.ema_bias.zero_()
        self._optimizer = self._build_optimizer()
        self._adapt_buffer.clear()
        self._holdout_ratios.clear()
        self._divergence_window.clear()
        self.cooldown_remaining = self.config.cooldown_rounds
        self.rollback_count += 1
        self._posterior_reset_pending = True
        self._ema_signal_reference = torch.zeros_like(self.ema_weight)
        self.events.append({"event": "rollback", "reason": reason})

    def consume_posterior_reset(self) -> bool:
        """Read-and-clear the §1.6 item 7 posterior-reset signal.

        Returns True when the effective (EMA) weights changed beyond the
        configured threshold or a rollback happened since the last call; the
        caller must then simulate ``reset_grounding_posterior()``.
        """

        pending = self._posterior_reset_pending
        self._posterior_reset_pending = False
        return pending


# ---------------------------------------------------------------------------
# Glue helpers
# ---------------------------------------------------------------------------


def make_pair_sample(
    measurement: PairMeasurement,
    latent_t0: torch.Tensor,
    latent_t1: torch.Tensor,
    z_t0: torch.Tensor,
    z_t1: torch.Tensor,
    p_nom_t0: np.ndarray | torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> TTAPairSample:
    """Assemble a :class:`TTAPairSample` from a measured (accepted) pair.

    ``measurement`` must have been produced with ``keep_flow_fields=True``.
    Reliability is already folded into ``measurement.weights`` (detached
    upstream; the reliability head is never updated, design §1.4).
    """

    if measurement.flow_fields is None:
        raise ValueError("measurement was produced without keep_flow_fields=True")
    if not measurement.accepted:
        raise ValueError("cannot build a training sample from a rejected pair")
    return TTAPairSample(
        latent_t0=latent_t0.detach().to(dtype),
        latent_t1=latent_t1.detach().to(dtype),
        z_t0=z_t0.detach().to(dtype),
        z_t1=z_t1.detach().to(dtype),
        p_nom_t0=torch.as_tensor(p_nom_t0, dtype=dtype),
        u_hat_fk=torch.as_tensor(measurement.u_hat_fk, dtype=dtype),
        flow_fields=measurement.flow_fields.to(dtype),
        crop_origins=torch.as_tensor(measurement.crop_origins, dtype=dtype),
        base_weights=torch.as_tensor(measurement.weights, dtype=dtype),
    )


def adapt_corrector_output(
    corrector,
    output,
    adapter: TTAFlowAdapter,
    use_ema: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrap a frozen corrector output with the TTA side head (design §1.5).

    Works for any corrector exposing ``delta_head`` on full keypoint latents
    (``AnchorConditionedResidualTransformer``, ``LeWMAnchorResidualCorrector``)
    and for ``TwoTipStructuredLeWMAnchorCorrector``, where the side head acts
    on the two fingertip latents and the center is rebuilt as their midpoint
    (structure constraint preserved).  Returns ``(delta_adapted,
    q_corrected_adapted)``; nothing in the existing files is modified.
    """

    q_nominal = output.q_corrected_px - output.delta_px
    from fcpg.transformer_grounding import TwoTipStructuredLeWMAnchorCorrector

    if isinstance(corrector, TwoTipStructuredLeWMAnchorCorrector):
        tips = list(corrector.tip_indices)
        tip_latent = output.keypoint_latent[:, tips]
        z_tips = compute_pretanh_logits(corrector, tip_latent)
        delta_tips = adapter.adapted_delta(tip_latent, z_tips, use_ema=use_ema)
        q_tips = q_nominal[:, tips] + delta_tips
        center = q_tips.mean(dim=1, keepdim=True)
        q_corrected = torch.cat((center, q_tips), dim=1)
        return q_corrected - q_nominal, q_corrected

    z = compute_pretanh_logits(corrector, output.keypoint_latent)
    delta = adapter.adapted_delta(output.keypoint_latent, z, use_ema=use_ema)
    return delta, q_nominal + delta
