"""Layer 3 — Perception: visual grounding, coordinate projection, target tracking.

Abstracts all visual perception so upper layers receive structured observations
independent of the underlying vision models or calibration methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np

from .hardware import Frame


# ── data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GripperLocalization:
    """Gripper location in image coordinates."""
    center_px: np.ndarray           # (2,) [u, v]
    fixed_tip_px: np.ndarray        # (2,)
    moving_tip_px: np.ndarray       # (2,)
    confidence: float
    method: str                     # e.g., "fcpg", "global_heatmap", "template"


@dataclass(frozen=True)
class TargetObservation:
    """Target object observation from template tracking or detection."""
    visible: bool
    centroid_px: np.ndarray | None  # (2,) [u, v] if visible
    bbox_xyxy: np.ndarray | None    # (4,) if visible
    score: float | None
    reason: str | None              # if not visible


@dataclass(frozen=True)
class AppearanceDescriptor:
    """Channel-marginal histogram descriptor for cross-view consistency.

    Layout is ``(n_channels * n_bins,)``, L1-normalised per channel.  No
    colour space is assumed — the frame's own channel order is used, so this
    works for any sensor encoding.
    """
    histogram: np.ndarray
    n_channels: int
    n_bins: int


@dataclass(frozen=True)
class DetectionCandidate:
    """One candidate target located without any prior reference."""
    centroid_px: np.ndarray          # (2,) [u, v]
    bbox_xyxy: np.ndarray            # (4,)
    area_px: int
    solidity: float                  # area_px / bbox area, in (0, 1]
    score: float
    descriptor: AppearanceDescriptor | None = None


@dataclass(frozen=True)
class ProjectionCalibration:
    """Camera projection parameters for pixel ↔ world transforms."""
    coefficients: np.ndarray        # shape depends on projection model
    feature_names: tuple[str, ...]  # e.g., ("x_base", "y_base", "z_base", "bias")
    live_transform: np.ndarray | None = None  # (2,3) affine for drift correction


# ── interfaces ────────────────────────────────────────────────────────────────

class ProjectionInterface(ABC):
    """Pixel ↔ world coordinate transforms using camera calibration."""

    @abstractmethod
    def pixel_to_world(
        self,
        pixel: np.ndarray,
        z_plane: float,
        *,
        live_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        """Project image pixel to world (x,y) on a given Z plane.

        Args:
            pixel: (2,) [u, v] in image coordinates
            z_plane: Z height in world frame (meters)
            live_transform: optional (2,3) affine correction for camera drift

        Returns:
            (3,) [x, y, z] in world coordinates
        """

    @abstractmethod
    def world_to_pixel(
        self,
        world_point: np.ndarray,
        *,
        live_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        """Project world point to image pixel.

        Args:
            world_point: (3,) [x, y, z] in world frame
            live_transform: optional (2,3) affine correction

        Returns:
            (2,) [u, v] in image coordinates
        """

    @abstractmethod
    def load_calibration(self, calibration_path: str) -> None:
        """Load projection parameters from file."""


class GripperGroundingInterface(ABC):
    """Locate the robot gripper in camera frames.

    May use local FCPG, global heatmaps, template matching, or learned models.
    """

    @abstractmethod
    def locate(self, frame: Frame, *, prior: GripperLocalization | None = None) -> GripperLocalization:
        """Return gripper center and fingertip positions in frame.

        Args:
            frame: camera frame
            prior: optional prior location (from FK or previous frame)
        """

    @abstractmethod
    def update_calibration(self, live_transform: np.ndarray) -> None:
        """Update live anchor transform after gripper calibration."""


class TargetDetectorInterface(ABC):
    """Find candidate targets in a frame with NO prior reference.

    This is the seam that makes autonomous registration possible: a detector
    answers "what plausible targets are in this frame?" without needing a
    template.  Template matching cannot do this — it requires the very
    reference we are trying to produce.

    Implementations are free to use colour statistics, saliency, learned
    segmentation, or open-vocabulary models.  All thresholds must come from
    adapter config so no task, object, or appearance assumption is baked in.
    """

    @abstractmethod
    def detect(self, frame: Frame) -> list[DetectionCandidate]:
        """Return candidates ordered by descending score.

        Returns an empty list when nothing qualifies.  Implementations must
        not raise merely because the frame contains no target.
        """

    @abstractmethod
    def describe(self, frame: Frame, region_mask: np.ndarray) -> AppearanceDescriptor:
        """Build an appearance descriptor for a region, for later comparison."""


class TargetRegistrationInterface(ABC):
    """Create and remove persistent target references.

    Registration is deliberately separate from detection: a reference may
    originate from a detector, from geometry, or from an operator, and the
    consumer should not care which.
    """

    @abstractmethod
    def register(
        self,
        frame: Frame,
        centroid_px: np.ndarray,
        role: str,
        view: str,
        *,
        radius_px: float = 28.0,
    ) -> str:
        """Register a reference at a frame location.  Returns target_id."""

    @abstractmethod
    def clear(self, target_id: str) -> None:
        """Remove a registered reference."""

    @abstractmethod
    def is_registered(self, role: str, view: str) -> bool:
        """Return True if a reference exists for this role and view."""

    @abstractmethod
    def list_registered(self) -> dict[str, dict[str, Any]]:
        """Return all references and their metadata."""


class TargetTrackerInterface(ABC):
    """Follow an already-registered target across frames.

    Tracking presupposes registration; it cannot bootstrap itself.
    """

    @abstractmethod
    def track(self, frame: Frame, target_id: str) -> TargetObservation:
        """Locate a registered target in the current frame."""
