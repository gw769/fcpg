"""Channel-discriminant segmentation detector.

A TargetDetectorInterface implementation that segments a frame using linear
discriminants over raw channel values, then returns connected components as
candidates.

The detector holds NO knowledge of what it is looking for.  A discriminant is
a generic linear constraint on channel intensities::

    sum_i (weight_i * channel_i) + offset > 0

Supplying the weight vectors and offsets from config is what selects a target
appearance, so the same class serves any object, sensor, or channel ordering.
Nothing here names a colour, an object, or a task.

Config
------
discriminants   : list of {weights: [w0, w1, ...], offset: float}
                  ALL must hold for a pixel to be foreground (logical AND).
min_area_px     : reject components smaller than this
max_area_px     : reject components larger than this (null = no bound)
min_solidity    : reject components below area/bbox_area (null = no bound)
morph_open_px   : optional opening kernel size to suppress speckle (0 = off)
descriptor_bins : histogram bins per channel for appearance descriptors
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from interfaces.hardware import Frame
from interfaces.perception import (
    TargetDetectorInterface,
    DetectionCandidate,
    AppearanceDescriptor,
)


class ChannelDiscriminantDetector(TargetDetectorInterface):
    def __init__(
        self,
        discriminants: list[dict[str, Any]],
        *,
        min_area_px: int = 30,
        max_area_px: int | None = None,
        min_solidity: float | None = None,
        morph_open_px: int = 0,
        descriptor_bins: int = 16,
    ) -> None:
        if not discriminants:
            raise ValueError("at least one channel discriminant is required")
        self._weights: list[np.ndarray] = []
        self._offsets: list[float] = []
        for spec in discriminants:
            if "weights" not in spec or "offset" not in spec:
                raise ValueError("each discriminant needs 'weights' and 'offset'")
            w = np.asarray(spec["weights"], dtype=np.float64)
            if w.ndim != 1 or not np.all(np.isfinite(w)):
                raise ValueError("discriminant weights must be a finite 1-D vector")
            off = float(spec["offset"])
            if not np.isfinite(off):
                raise ValueError("discriminant offset must be finite")
            self._weights.append(w)
            self._offsets.append(off)

        if min_area_px < 1:
            raise ValueError("min_area_px must be >= 1")
        if max_area_px is not None and max_area_px <= min_area_px:
            raise ValueError("max_area_px must exceed min_area_px")
        if min_solidity is not None and not (0.0 < min_solidity <= 1.0):
            raise ValueError("min_solidity must lie in (0, 1]")
        if descriptor_bins < 2:
            raise ValueError("descriptor_bins must be >= 2")

        self._min_area = int(min_area_px)
        self._max_area = None if max_area_px is None else int(max_area_px)
        self._min_solidity = min_solidity
        self._morph_open = int(morph_open_px)
        self._bins = int(descriptor_bins)

    # ── internals ────────────────────────────────────────────────────────────

    def _foreground(self, data: np.ndarray) -> np.ndarray:
        """Apply every discriminant as a logical AND over channel intensities."""
        if data.ndim != 3:
            raise ValueError("frame data must be (H, W, C)")
        channels = data.shape[2]
        values = data.astype(np.float64)
        mask = np.ones(data.shape[:2], dtype=bool)
        for w, off in zip(self._weights, self._offsets):
            if w.shape[0] != channels:
                raise ValueError(
                    f"discriminant expects {w.shape[0]} channels, frame has {channels}"
                )
            mask &= (values @ w + off) > 0.0
        out = mask.astype(np.uint8)
        if self._morph_open > 0:
            k = np.ones((self._morph_open, self._morph_open), np.uint8)
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
        return out

    # ── interface ────────────────────────────────────────────────────────────

    def detect(self, frame: Frame) -> list[DetectionCandidate]:
        data = np.asarray(frame.data)
        mask = self._foreground(data)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

        candidates: list[DetectionCandidate] = []
        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self._min_area:
                continue
            if self._max_area is not None and area > self._max_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            bbox_area = max(w * h, 1)
            solidity = area / bbox_area
            if self._min_solidity is not None and solidity < self._min_solidity:
                continue
            component = (labels == i).astype(np.uint8)
            candidates.append(
                DetectionCandidate(
                    centroid_px=np.asarray(centroids[i], dtype=np.float64),
                    bbox_xyxy=np.asarray([x, y, x + w, y + h], dtype=np.float64),
                    area_px=area,
                    solidity=float(solidity),
                    score=float(area),
                    descriptor=self.describe(frame, component),
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def describe(self, frame: Frame, region_mask: np.ndarray) -> AppearanceDescriptor:
        data = np.asarray(frame.data)
        mask = np.asarray(region_mask).astype(bool)
        if mask.shape != data.shape[:2]:
            raise ValueError("region_mask shape must match the frame")
        if not mask.any():
            raise ValueError("region_mask selects no pixels")
        pixels = data[mask]
        channels = pixels.shape[1]
        parts = []
        for c in range(channels):
            hist, _ = np.histogram(
                pixels[:, c].astype(np.float64), bins=self._bins, range=(0.0, 256.0)
            )
            total = hist.sum()
            parts.append(hist / total if total > 0 else hist.astype(np.float64))
        return AppearanceDescriptor(
            histogram=np.concatenate(parts).astype(np.float64),
            n_channels=channels,
            n_bins=self._bins,
        )
