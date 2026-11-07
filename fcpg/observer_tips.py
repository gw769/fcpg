"""Motion-consistent gripper-tip measurement (milestone M3).

The live path finds tips by colour: ``column_metal_operator_tips`` looks for
the two lowest blue-metal peaks hanging from the top of the frame.  Knuckles
are the same colour and the same shape, so the detector locks onto the finger
roots and then asserts ``on_tips=True`` because it found itself.

A knuckle and a tip are indistinguishable in one still frame.  They are not
indistinguishable under motion.  Given ``Δq``, forward kinematics predicts a
pixel displacement ``u_hat`` at the true tip; a knuckle, being closer to the
wrist axis, moves less and in a different direction.  The measurement is
therefore the pixel whose *observed* optical flow matches that prediction,
and the test is a chi-square on the innovation -- the same gate the filter
already uses, applied one step earlier so a bad point never enters it.

Colour is kept only as a cold-start seed: it centres the search window.
It is never the accepted measurement.  When predicted flow is too small
to tell a tip from a knuckle the last accepted lock is held; rejecting
the idle frame and falling back to colour is how the 22% knuckle-lock
rate re-enters the filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fcpg.observer_filter import CHI2_95_2DOF
from fcpg.so101_column_tip_measurement import column_metal_operator_tips
from fcpg.tta_flow_consistency import DISFlowBackend, create_flow_backend


__all__ = [
    "TipMeasurement",
    "measure_motion_tips",
    "color_seed_tips",
]


@dataclass(frozen=True)
class TipMeasurement:
    """One frame's tip pixels, with the reason each one was accepted or not."""

    points_px: np.ndarray  # [3, 2]
    valid: np.ndarray  # [3] bool
    source: tuple[str, ...]  # per-anchor: motion / hold / none
    mahalanobis: np.ndarray  # [3]
    flow_px: np.ndarray  # [3, 2] observed flow at the accepted pixel
    predicted_flow_px: np.ndarray  # [3, 2]
    diagnostics: dict[str, Any]

    @property
    def n_motion(self) -> int:
        return int(sum(source == "motion" for source in self.source))

    @property
    def n_hold(self) -> int:
        return int(sum(source == "hold" for source in self.source))


def color_seed_tips(frame_rgb: np.ndarray) -> np.ndarray | None:
    """The colour heuristic, used only to centre the motion search."""

    result = column_metal_operator_tips(frame_rgb)
    if result is None:
        return None
    array = np.asarray(result, dtype=np.float64)
    if array.shape != (3, 2) or not np.all(np.isfinite(array)):
        return None
    return array


def _sample_flow(flow: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Bilinear sample of a dense flow field at one pixel."""

    height, width = flow.shape[:2]
    x, y = float(point[0]), float(point[1])
    if not (0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0):
        return np.array([np.nan, np.nan], dtype=np.float64)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0
    top = (1.0 - dx) * flow[y0, x0] + dx * flow[y0, x1]
    bottom = (1.0 - dx) * flow[y1, x0] + dx * flow[y1, x1]
    return ((1.0 - dy) * top + dy * bottom).astype(np.float64)


def measure_motion_tips(
    gray0: np.ndarray,
    gray1: np.ndarray,
    predicted_px0: np.ndarray,
    predicted_flow_px: np.ndarray,
    *,
    seed_px: np.ndarray | None = None,
    held_px: np.ndarray | None = None,
    search_radius_px: float = 14.0,
    sigma_flow_px: float = 2.5,
    chi2_gate: float = CHI2_95_2DOF,
    min_predicted_flow_px: float = 1.5,
    backend: DISFlowBackend | None = None,
) -> TipMeasurement:
    """Find tip pixels whose observed flow matches the FK-predicted flow.

    For each anchor the search is a disc of radius ``search_radius_px``
    around the seed (colour if present, otherwise the filter prediction).
    A candidate is accepted when
    ``||u_obs - u_hat||^2 / sigma^2`` is below the 2-DoF 95% chi-square
    and the predicted flow itself is large enough to be informative.

    An idle frame (predicted flow below ``min_predicted_flow_px``) cannot
    distinguish a tip from a knuckle.  That is not a rejection: there is
    simply no new motion evidence, so the last accepted lock in
    ``held_px`` is held.  Falling back to the colour detector on those
    frames reintroduces the 22% knuckle-lock rate.  Colour remains a
    search seed only.
    """

    predicted = np.asarray(predicted_px0, dtype=np.float64)
    u_hat = np.asarray(predicted_flow_px, dtype=np.float64)
    if predicted.shape != (3, 2) or u_hat.shape != (3, 2):
        raise ValueError("predicted_px0 and predicted_flow_px must have shape [3, 2]")
    if gray0.shape != gray1.shape or gray0.ndim != 2:
        raise ValueError("gray0 and gray1 must be equal-shape single-channel images")

    flow_backend = backend if backend is not None else create_flow_backend("dis")
    flow = flow_backend.compute(gray0, gray1)

    height, width = gray0.shape
    points = np.full((3, 2), np.nan, dtype=np.float64)
    valid = np.zeros(3, dtype=bool)
    source: list[str] = []
    mahalanobis = np.full(3, np.inf, dtype=np.float64)
    observed_flow = np.full((3, 2), np.nan, dtype=np.float64)
    diagnostics: dict[str, Any] = {"search_radius_px": search_radius_px}

    seed = None if seed_px is None else np.asarray(seed_px, dtype=np.float64)
    held = None if held_px is None else np.asarray(held_px, dtype=np.float64)
    if held is not None and held.shape != (3, 2):
        raise ValueError("held_px must have shape [3, 2]")
    for anchor in range(3):
        centre = predicted[anchor]
        if seed is not None and np.all(np.isfinite(seed[anchor])):
            centre = seed[anchor]
        if not np.all(np.isfinite(centre)):
            source.append("none")
            continue
        predicted_speed = float(np.linalg.norm(u_hat[anchor]))
        if predicted_speed < min_predicted_flow_px:
            # Idle: no motion evidence.  Hold the last accepted tip rather
            # than rejecting the frame or falling back to colour.
            if held is not None and np.all(np.isfinite(held[anchor])):
                points[anchor] = held[anchor]
                valid[anchor] = True
                source.append("hold")
                diagnostics[f"anchor_{anchor}_reason"] = "predicted_flow_too_small_hold"
            else:
                source.append("none")
                diagnostics[f"anchor_{anchor}_reason"] = "predicted_flow_too_small"
            continue

        xs = np.arange(
            max(0, int(np.floor(centre[0] - search_radius_px))),
            min(width, int(np.ceil(centre[0] + search_radius_px)) + 1),
        )
        ys = np.arange(
            max(0, int(np.floor(centre[1] - search_radius_px))),
            min(height, int(np.ceil(centre[1] + search_radius_px)) + 1),
        )
        if xs.size == 0 or ys.size == 0:
            source.append("none")
            continue
        grid_x, grid_y = np.meshgrid(xs, ys)
        offset = np.stack((grid_x - centre[0], grid_y - centre[1]), axis=-1)
        disc = np.sum(offset * offset, axis=-1) <= search_radius_px**2
        if not bool(np.any(disc)):
            source.append("none")
            continue

        best_chi = np.inf
        best_pixel = None
        best_flow = None
        for y, x in zip(grid_y[disc], grid_x[disc]):
            pixel = np.array((float(x), float(y)), dtype=np.float64)
            u_obs = _sample_flow(flow, pixel)
            if not np.all(np.isfinite(u_obs)):
                continue
            chi = float(np.sum((u_obs - u_hat[anchor]) ** 2) / (sigma_flow_px**2))
            if chi < best_chi:
                best_chi = chi
                best_pixel = pixel
                best_flow = u_obs

        mahalanobis[anchor] = best_chi
        if best_pixel is not None and best_chi <= chi2_gate:
            points[anchor] = best_pixel
            valid[anchor] = True
            observed_flow[anchor] = best_flow
            source.append("motion")
        else:
            source.append("none")
            diagnostics[f"anchor_{anchor}_reason"] = (
                "chi2_gate" if best_pixel is not None else "no_candidate"
            )

    return TipMeasurement(
        points_px=points,
        valid=valid,
        source=tuple(source),
        mahalanobis=mahalanobis,
        flow_px=observed_flow,
        predicted_flow_px=u_hat,
        diagnostics=diagnostics,
    )
