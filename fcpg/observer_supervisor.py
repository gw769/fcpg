"""Covariance-gated close and 3-D drop-zone persistence (milestone M5).

Two live failures, two missing numbers:

1. **Close.**  ``close_allowed_for_pickup`` currently treats
   ``gripper_to_pickup_m`` as a metric distance.  It is not: it is an affine
   back-projection onto a constant-z plane.  The observer's posterior std
   along the jaw-close direction *is* a metric number.  A close is allowed
   only when that std is below a threshold; a stale or exploded covariance
   fails closed, which is the opposite of today's ``on_tips=True`` self-vote.
2. **Drop.**  ``maybe_register_drop`` re-finds a brown blob on the table
   every time the registered drop leaves the frame.  The first live failure
   was exactly that re-registration locking onto the wrong patch.  Once a
   drop is seen it is stored as a base-frame 3-D point and reprojected
   through the current camera estimate.  Leaving the image is not a reason
   to forget where the bowl is.

Both helpers are additive: the existing close predicate keeps its signature
and the 3-D drop path is opt-in, so ``:8877`` does not change until a
caller asks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fcpg.observer_filter import InformationObserver
from fcpg.so101_realtime_mpc import close_allowed_for_pickup, pickup_is_metrically_near


__all__ = [
    "DEFAULT_CLOSE_SIGMA_M",
    "DropZone3D",
    "close_allowed_with_covariance",
    "register_drop_zone_3d",
    "reproject_drop_zone",
]


DEFAULT_CLOSE_SIGMA_M = 0.012


def close_allowed_with_covariance(
    *,
    three_point_verdict: str | None,
    gripper_to_pickup_px: float | None,
    gripper_to_pickup_m: float | None,
    pickup_near_px: float,
    pickup_near_m: float,
    along_close_sigma_m: float | None,
    max_sigma_m: float = DEFAULT_CLOSE_SIGMA_M,
) -> bool:
    """Existing close predicate plus a covariance gate.

    A missing or non-finite sigma fails closed.  ``held`` still short-circuits
    the metric-near test -- the jaws already pinched -- but it does not
    bypass the covariance gate: a ``held`` verdict produced while the
    observer is unconverged is how the second live failure armed a place
    with empty jaws.
    """

    if (
        along_close_sigma_m is None
        or not np.isfinite(along_close_sigma_m)
        or float(along_close_sigma_m) > float(max_sigma_m)
    ):
        return False
    return close_allowed_for_pickup(
        three_point_verdict=three_point_verdict,
        gripper_to_pickup_px=gripper_to_pickup_px,
        gripper_to_pickup_m=gripper_to_pickup_m,
        pickup_near_px=pickup_near_px,
        pickup_near_m=pickup_near_m,
    )


@dataclass(frozen=True)
class DropZone3D:
    """A drop target stored in the robot base frame, not in pixels."""

    xyz_base: np.ndarray
    covariance: np.ndarray
    registered_round_id: str
    source: str = "column_reprojection"

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_base, dtype=np.float64).reshape(3)
        cov = np.asarray(self.covariance, dtype=np.float64)
        if cov.shape != (3, 3):
            raise ValueError("drop-zone covariance must be 3x3")
        object.__setattr__(self, "xyz_base", xyz)
        object.__setattr__(self, "covariance", cov)

    def to_dict(self) -> dict[str, Any]:
        return {
            "xyz_base": self.xyz_base.tolist(),
            "covariance": self.covariance.tolist(),
            "registered_round_id": self.registered_round_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DropZone3D":
        return cls(
            xyz_base=np.asarray(payload["xyz_base"], dtype=np.float64),
            covariance=np.asarray(payload["covariance"], dtype=np.float64),
            registered_round_id=str(payload["registered_round_id"]),
            source=str(payload.get("source", "column_reprojection")),
        )


def register_drop_zone_3d(
    observer: InformationObserver,
    centroid_px: np.ndarray,
    *,
    plane_z_base: float,
    round_id: str,
    source: str = "column_reprojection",
) -> DropZone3D:
    """Back-project a drop centroid onto a horizontal plane in the base frame.

    The plane height is the table, which the live path already treats as
    known (``contact_height_m``).  Using that plane is honest: a single
    pixel does not determine a 3-D point, and inventing a depth from the
    camera model alone is how the affine path produced a metric-looking
    number that was not metric.  Once the point is in the base frame, later
    frames reproject it; they do not go looking for another brown blob.
    """

    pixel = np.asarray(centroid_px, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(pixel)):
        raise ValueError("drop centroid must be finite")
    rotation = observer.state.rotation()
    translation = observer.state.translation
    # Ray in camera frame through the pixel.
    x = (pixel[0] - observer.state.cx) / observer.state.fx
    y = (pixel[1] - observer.state.cy) / observer.state.fy
    direction_cam = np.array((x, y, 1.0), dtype=np.float64)
    direction_cam /= np.linalg.norm(direction_cam)
    direction_base = rotation.T @ direction_cam
    origin_base = -rotation.T @ translation
    if abs(float(direction_base[2])) < 1e-9:
        raise ValueError("drop ray is parallel to the table; cannot intersect")
    scale = (float(plane_z_base) - float(origin_base[2])) / float(direction_base[2])
    if scale <= 0.0:
        raise ValueError("drop ray does not hit the table in front of the camera")
    xyz = origin_base + scale * direction_base
    # First-order covariance: pixel noise mapped through the ray, plus the
    # observer's translation uncertainty.  Exact enough to gate a place.
    sigma_px = observer.sigma_px
    pixel_jac = np.array(
        (
            (1.0 / observer.state.fx, 0.0),
            (0.0, 1.0 / observer.state.fy),
        ),
        dtype=np.float64,
    )
    cam_jac = np.zeros((3, 2), dtype=np.float64)
    cam_jac[:2] = pixel_jac * scale
    base_jac = rotation.T @ cam_jac
    pixel_cov = (sigma_px**2) * (base_jac @ base_jac.T)
    pose_cov = observer.state.covariance()[3:6, 3:6]
    covariance = pixel_cov + pose_cov
    return DropZone3D(
        xyz_base=xyz,
        covariance=covariance,
        registered_round_id=str(round_id),
        source=source,
    )


def reproject_drop_zone(
    drop: DropZone3D, observer: InformationObserver
) -> dict[str, Any]:
    """Project a stored base-frame drop through the current camera estimate."""

    pixels, depth = observer.project(drop.xyz_base.reshape(1, 3))
    visible = bool(
        np.all(np.isfinite(pixels[0]))
        and depth[0] > 0.02
        and 0.0 <= pixels[0, 0] < 224.0
        and 28.0 <= pixels[0, 1] < 196.0
    )
    return {
        "centroid_px": pixels[0].tolist(),
        "depth_m": float(depth[0]),
        "visible": visible,
        "xyz_base": drop.xyz_base.tolist(),
        "registered_round_id": drop.registered_round_id,
    }


# Re-export so callers that already import the metric-near helper do not
# have to remember which module grew the covariance argument.
pickup_is_metrically_near = pickup_is_metrically_near
