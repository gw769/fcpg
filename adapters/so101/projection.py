"""SO-101 column/wrist projection adapter.

Wraps the linear feature-to-pixel projection from so101_camera_calibration.py
behind ProjectionInterface.  Calibration path and live transform come from
config — nothing hardcoded here.
"""

from __future__ import annotations

import json
import numpy as np

from interfaces.perception import ProjectionInterface, ProjectionCalibration


class SO101LinearProjection(ProjectionInterface):
    """ProjectionInterface using a linear feature-to-pixel regression fit.

    The column fit uses features [x_base, y_base, z_base, bias].
    The wrist fit uses local gripper-frame features.

    Config keys
    -----------
    calibration_path : path to calibration JSON
    view             : "column" or "wrist"
    """

    def __init__(self, calibration_path: str, view: str) -> None:
        self._view = view
        self._coeff: np.ndarray | None = None
        self._feature_names: tuple[str, ...] = ()
        if calibration_path:
            self.load_calibration(calibration_path)

    def load_calibration(self, calibration_path: str) -> None:
        with open(calibration_path) as f:
            data = json.load(f)
        fit = data["fits"][self._view]
        self._coeff = np.array(fit["coefficients"], dtype=np.float64)   # (F, 2)
        self._feature_names = tuple(fit["feature_names"])

    def _build_features(self, world_point: np.ndarray) -> np.ndarray:
        """Build feature vector from world point — column view only."""
        x, y, z = float(world_point[0]), float(world_point[1]), float(world_point[2])
        feature_map = {"x_base": x, "y_base": y, "z_base": z, "bias": 1.0}
        return np.array([feature_map[n] for n in self._feature_names], dtype=np.float64)

    def world_to_pixel(
        self,
        world_point: np.ndarray,
        *,
        live_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._coeff is None:
            raise RuntimeError("projection not calibrated; call load_calibration first")
        feat = self._build_features(world_point)
        nom_px = feat @ self._coeff                     # (2,) nominal pixel
        if live_transform is not None:
            T = np.array(live_transform, dtype=np.float64)  # (2, 3)
            nom_px = T @ np.array([nom_px[0], nom_px[1], 1.0])
        return nom_px

    def pixel_to_world(
        self,
        pixel: np.ndarray,
        z_plane: float,
        *,
        live_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve for (x, y) at z=z_plane given pixel.

        If live_transform is provided, it maps nominal → live pixels, so we
        apply its inverse before solving the fit.
        """
        if self._coeff is None:
            raise RuntimeError("projection not calibrated; call load_calibration first")
        px = np.array(pixel, dtype=np.float64)

        # invert live transform: live_px → nominal_px
        if live_transform is not None:
            T = np.array(live_transform, dtype=np.float64)    # (2,3)
            # solve T @ [u_nom, v_nom, 1] = px  (least-squares)
            A = T[:, :2]
            b = px - T[:, 2]
            px = np.linalg.lstsq(A, b, rcond=None)[0]

        # coeff columns: u = feat @ coeff[:,0],  v = feat @ coeff[:,1]
        # with z fixed, feat = [x, y, z, 1], solve for x,y
        # u = c00*x + c10*y + c20*z + c30
        # v = c01*x + c11*y + c21*z + c31
        c = self._coeff   # (4,2) [x,y,z,bias] × [u,v]
        u_target = px[0] - c[2, 0] * z_plane - c[3, 0]
        v_target = px[1] - c[2, 1] * z_plane - c[3, 1]
        A = np.array([[c[0, 0], c[1, 0]],
                      [c[0, 1], c[1, 1]]], dtype=np.float64)
        xy = np.linalg.solve(A, np.array([u_target, v_target]))
        return np.array([xy[0], xy[1], z_plane], dtype=np.float64)
