"""Constant-parameter information filter for the column camera (milestone M2).

The quantity being estimated is a *constant* (camera pose + a masked set of
joint offsets).  The prediction step is therefore the identity plus a tiny
process covariance that exists only to absorb collisions and slow drift, and
the information matrix accumulates as ``Y += H^T R^{-1} H``.  ``trace(P)``
shrinks when the arm walks through poses that make the measurement Jacobian
full rank -- that is the precise meaning of "越来越准", and it is a number
that can be plotted.

Intrinsics are **not** in the state.  Milestone M1's identifiability audit
showed that task motion does not separate ``f`` from ``t_z``: the free-f
bundle lands on a bound, and a 10% holdout-error band spans more than an
octave of focal length.  Treating an unidentified ``f`` as a filter state
would report a shrinking variance on a number that is still wrong.  ``K`` is
therefore the 45° FOV prior (the same formula the runtime already uses), and
the filter estimates the things the data actually constrain: the six-DoF
camera pose and the three non-gauge joint offsets.

The update is an iterated extended Kalman filter written in information form.
One Gauss-Newton step around a bad linearization is enough to accept a
knuckle as a tip; re-linearizing two or three times, and gating the
innovation with a chi-square test, is what keeps that from happening.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from fcpg.observer_bundle import (
    ARM_JOINT_COUNT,
    DEFAULT_JOINT_OFFSET_MASK,
    LETTERBOX_FOV45_FOCAL_PX,
    intrinsic_matrix,
)


__all__ = [
    "CHI2_95_2DOF",
    "ObserverState",
    "ObserverUpdate",
    "InformationObserver",
    "fim_spectrum",
    "nees",
    "motion_type",
    "observability_by_motion",
]


CHI2_95_2DOF = float(chi2.ppf(0.95, df=2))
_MIN_DEPTH_M = 0.02


@dataclass
class ObserverState:
    """Mean and information of the constant-parameter estimate."""

    rotvec: np.ndarray
    translation: np.ndarray
    joint_offsets: np.ndarray
    information: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    joint_offset_mask: tuple[int, ...]
    updates: int = 0

    @property
    def dimension(self) -> int:
        return 6 + len(self.joint_offset_mask)

    def covariance(self) -> np.ndarray:
        return np.linalg.pinv(self.information)

    def sigma(self) -> np.ndarray:
        diagonal = np.clip(np.diag(self.covariance()), 0.0, None)
        return np.sqrt(diagonal)

    def rotation(self) -> np.ndarray:
        return Rotation.from_rotvec(self.rotvec).as_matrix()

    def extrinsic(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation()
        matrix[:3, 3] = self.translation
        return matrix

    def intrinsic(self) -> np.ndarray:
        return intrinsic_matrix(self.fx, self.fy, self.cx, self.cy)

    def mean_vector(self) -> np.ndarray:
        return np.concatenate((self.rotvec, self.translation, self.packed_offsets()))

    def packed_offsets(self) -> np.ndarray:
        return np.asarray(
            [self.joint_offsets[int(j)] for j in self.joint_offset_mask],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        sigma = self.sigma()
        return {
            "rotvec": self.rotvec.tolist(),
            "translation_m": self.translation.tolist(),
            "joint_offsets_normalized": self.joint_offsets.tolist(),
            "sigma": sigma.tolist(),
            "trace_p": float(np.trace(self.covariance())),
            "updates": self.updates,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }


@dataclass(frozen=True)
class ObserverUpdate:
    accepted: bool
    innovation_px: np.ndarray
    mahalanobis: float
    chi2_threshold: float
    residual_rms_px: float
    n_measurements: int
    rejection_reason: str | None = None


def _project(
    rotvec: np.ndarray,
    translation: np.ndarray,
    points_camera_ready: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    camera = points_camera_ready @ rotation.T + translation
    depth = camera[:, 2]
    safe = np.maximum(depth, _MIN_DEPTH_M)
    pixels = np.column_stack(
        (fx * camera[:, 0] / safe + cx, fy * camera[:, 1] / safe + cy)
    )
    return pixels, depth


def _shift_points(
    points_base_xyz: np.ndarray,
    joint_jacobian: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    return points_base_xyz + joint_jacobian @ offsets


def measurement_jacobian(
    state: ObserverState,
    points_base_xyz: np.ndarray,
    joint_jacobian: np.ndarray,
    *,
    rel_step: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference ``dh/dtheta`` and the predicted pixels.

    Analytic SE(3) Jacobians exist, but the joint-offset path already goes
    through a numerical FK linearization, so a consistent numerical Jacobian
    on the packed state is the honest one.  Nine columns is cheap.
    """

    points = _shift_points(points_base_xyz, joint_jacobian, state.joint_offsets)
    predicted, _ = _project(
        state.rotvec, state.translation, points, state.fx, state.fy, state.cx, state.cy
    )
    theta = state.mean_vector()
    jacobian = np.empty((predicted.size, theta.size), dtype=np.float64)
    for index in range(theta.size):
        step = rel_step * max(abs(float(theta[index])), 1.0)
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (
            _pixels_from_vector(plus, state, points_base_xyz, joint_jacobian)
            - _pixels_from_vector(minus, state, points_base_xyz, joint_jacobian)
        ).ravel() / (2.0 * step)
    return jacobian, predicted


def _pixels_from_vector(
    theta: np.ndarray,
    state: ObserverState,
    points_base_xyz: np.ndarray,
    joint_jacobian: np.ndarray,
) -> np.ndarray:
    offsets = state.joint_offsets.copy()
    for position, joint in enumerate(state.joint_offset_mask):
        offsets[int(joint)] = theta[6 + position]
    points = _shift_points(points_base_xyz, joint_jacobian, offsets)
    pixels, _ = _project(
        theta[:3], theta[3:6], points, state.fx, state.fy, state.cx, state.cy
    )
    return pixels


def fim_spectrum(jacobian: np.ndarray, sigma_px: float) -> np.ndarray:
    """Eigenvalues of the Gauss-Newton information ``H^T R^{-1} H``."""

    if jacobian.size == 0:
        return np.zeros(0, dtype=np.float64)
    weight = 1.0 / max(float(sigma_px) ** 2, 1e-12)
    information = weight * (jacobian.T @ jacobian)
    values = np.sort(np.linalg.eigvalsh(information))[::-1]
    return np.maximum(values, 0.0)


def nees(error: np.ndarray, covariance: np.ndarray) -> float:
    """Normalized estimation error squared.  Honest covariance => NEES ≈ dim."""

    residual = np.asarray(error, dtype=np.float64).reshape(-1)
    try:
        solved = np.linalg.solve(covariance, residual)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance) @ residual
    return float(residual @ solved)


def motion_type(delta_q: np.ndarray, *, floor: float = 1.0) -> str:
    """Name the dominant joint motion of one step, for the FIM-by-motion table.

    The names are the question the observability analysis answers: which
    physical motion lights up which degree of freedom.  A step that moves
    several joints above ``floor`` is ``mixed``; a step below the floor on
    every joint is ``idle`` and contributes almost no information.
    """

    delta = np.abs(np.asarray(delta_q, dtype=np.float64).reshape(-1))
    if delta.size < 6:
        delta = np.pad(delta, (0, 6 - delta.size))
    if float(delta.max()) < floor:
        return "idle"
    active = [index for index, value in enumerate(delta[:6]) if value >= floor]
    if active == [0]:
        return "pan"
    if set(active) <= {1, 2}:
        return "lift"
    if set(active) <= {3}:
        return "wrist_flex"
    if set(active) <= {4}:
        return "wrist_roll"
    if active == [5]:
        return "jaw"
    return "mixed"


class InformationObserver:
    """Iterated information filter on ``(xi_cb, dq)`` with a fixed pinhole ``K``."""

    def __init__(
        self,
        *,
        fx: float = LETTERBOX_FOV45_FOCAL_PX,
        fy: float | None = None,
        cx: float = 112.0,
        cy: float = 112.0,
        joint_offset_mask: Sequence[int] = DEFAULT_JOINT_OFFSET_MASK,
        prior_rot_std_rad: float = 0.4,
        prior_t_std_m: float = 0.4,
        prior_dq_std: float = 5.0,
        process_rot_std_rad: float = 1e-4,
        process_t_std_m: float = 1e-4,
        process_dq_std: float = 1e-3,
        sigma_px: float = 4.0,
        chi2_gate: float = CHI2_95_2DOF,
        iterations: int = 3,
        initial_rotvec: np.ndarray | None = None,
        initial_translation: np.ndarray | None = None,
    ) -> None:
        mask = tuple(int(j) for j in joint_offset_mask)
        dim = 6 + len(mask)
        prior = np.concatenate(
            (
                np.full(3, prior_rot_std_rad**2),
                np.full(3, prior_t_std_m**2),
                np.full(len(mask), prior_dq_std**2),
            )
        )
        process = np.concatenate(
            (
                np.full(3, process_rot_std_rad**2),
                np.full(3, process_t_std_m**2),
                np.full(len(mask), process_dq_std**2),
            )
        )
        self.joint_offset_mask = mask
        self.sigma_px = float(sigma_px)
        self.chi2_gate = float(chi2_gate)
        self.iterations = int(iterations)
        self._process = np.diag(process)
        self.state = ObserverState(
            rotvec=np.zeros(3, dtype=np.float64)
            if initial_rotvec is None
            else np.asarray(initial_rotvec, dtype=np.float64).reshape(3),
            translation=(
                np.array([0.0, 0.0, 0.8], dtype=np.float64)
                if initial_translation is None
                else np.asarray(initial_translation, dtype=np.float64).reshape(3)
            ),
            joint_offsets=np.zeros(ARM_JOINT_COUNT, dtype=np.float64),
            information=np.diag(1.0 / np.maximum(prior, 1e-12)),
            fx=float(fx),
            fy=float(fy if fy is not None else fx),
            cx=float(cx),
            cy=float(cy),
            joint_offset_mask=mask,
        )
        if self.state.dimension != dim:
            raise RuntimeError("observer state dimension mismatch")

    def predict(self) -> None:
        """Random-walk prediction: add process covariance, keep the mean."""

        covariance = self.state.covariance() + self._process
        # pinv rather than inv: a direction the data has never constrained
        # has infinite variance, and inverting that is not a number.
        self.state.information = np.linalg.pinv(covariance)

    def project(
        self, points_base_xyz: np.ndarray, joint_jacobian: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_base_xyz, dtype=np.float64)
        if joint_jacobian is not None:
            points = _shift_points(points, joint_jacobian, self.state.joint_offsets)
        return _project(
            self.state.rotvec,
            self.state.translation,
            points,
            self.state.fx,
            self.state.fy,
            self.state.cx,
            self.state.cy,
        )

    def update(
        self,
        points_base_xyz: np.ndarray,
        pixels_xy: np.ndarray,
        joint_jacobian: np.ndarray,
        *,
        valid: np.ndarray | None = None,
    ) -> ObserverUpdate:
        """One iterated information update.  Rejects on a chi-square gate."""

        points = np.asarray(points_base_xyz, dtype=np.float64)
        pixels = np.asarray(pixels_xy, dtype=np.float64)
        jacobian_fk = np.asarray(joint_jacobian, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_base_xyz must have shape [N, 3]")
        if pixels.shape != (len(points), 2):
            raise ValueError("pixels_xy must have shape [N, 2]")
        if jacobian_fk.shape != (len(points), 3, ARM_JOINT_COUNT):
            raise ValueError(
                "joint_jacobian must have shape [N, 3, 5] matching the points"
            )
        if valid is None:
            keep = np.ones(len(points), dtype=bool)
        else:
            keep = np.asarray(valid, dtype=bool)
        keep = keep & np.all(np.isfinite(points), axis=1) & np.all(
            np.isfinite(pixels), axis=1
        )
        if int(keep.sum()) < 2:
            return ObserverUpdate(
                accepted=False,
                innovation_px=np.zeros((0, 2)),
                mahalanobis=float("inf"),
                chi2_threshold=self.chi2_gate,
                residual_rms_px=float("inf"),
                n_measurements=0,
                rejection_reason="too_few_measurements",
            )

        points = points[keep]
        pixels = pixels[keep]
        jacobian_fk = jacobian_fk[keep]
        n = len(points)
        r_inv = (1.0 / (self.sigma_px**2)) * np.eye(2 * n, dtype=np.float64)

        working = ObserverState(
            rotvec=self.state.rotvec.copy(),
            translation=self.state.translation.copy(),
            joint_offsets=self.state.joint_offsets.copy(),
            information=self.state.information.copy(),
            fx=self.state.fx,
            fy=self.state.fy,
            cx=self.state.cx,
            cy=self.state.cy,
            joint_offset_mask=self.state.joint_offset_mask,
            updates=self.state.updates,
        )
        jacobian = np.zeros((2 * n, working.dimension), dtype=np.float64)
        predicted = np.zeros((n, 2), dtype=np.float64)
        for _ in range(max(1, self.iterations)):
            jacobian, predicted = measurement_jacobian(working, points, jacobian_fk)
            innovation = (pixels - predicted).reshape(-1)
            # Joseph-form information update around the current linearization.
            information = self.state.information + jacobian.T @ r_inv @ jacobian
            rhs = (
                self.state.information @ self.state.mean_vector()
                + jacobian.T @ r_inv @ (innovation + jacobian @ working.mean_vector())
            )
            try:
                theta = np.linalg.solve(information, rhs)
            except np.linalg.LinAlgError:
                theta = np.linalg.pinv(information) @ rhs
            working.rotvec = theta[:3]
            working.translation = theta[3:6]
            for position, joint in enumerate(working.joint_offset_mask):
                working.joint_offsets[int(joint)] = theta[6 + position]
            working.information = information

        predicted, _ = _project(
            working.rotvec,
            working.translation,
            _shift_points(points, jacobian_fk, working.joint_offsets),
            working.fx,
            working.fy,
            working.cx,
            working.cy,
        )
        innovation = pixels - predicted
        residual = innovation.reshape(-1)
        # Innovation covariance S = H P H^T + R, using the *prior* P so the
        # gate is a true residual test and not a self-fulfilling one.
        prior_p = self.state.covariance()
        s_matrix = jacobian @ prior_p @ jacobian.T + np.linalg.inv(r_inv)
        try:
            mahalanobis = float(residual @ np.linalg.solve(s_matrix, residual))
        except np.linalg.LinAlgError:
            mahalanobis = float(residual @ (np.linalg.pinv(s_matrix) @ residual))
        # Per-point 2-DoF gate: the stacked chi-square grows with N, so a
        # 2-DoF threshold on the mean per-point mahalanobis is the one that
        # stays calibrated as more anchors arrive.
        per_point = mahalanobis / max(n, 1)
        rms = float(np.sqrt(np.mean(np.sum(innovation**2, axis=1))))
        if per_point > self.chi2_gate:
            return ObserverUpdate(
                accepted=False,
                innovation_px=innovation,
                mahalanobis=per_point,
                chi2_threshold=self.chi2_gate,
                residual_rms_px=rms,
                n_measurements=n,
                rejection_reason="chi2_gate",
            )
        working.updates = self.state.updates + 1
        self.state = working
        return ObserverUpdate(
            accepted=True,
            innovation_px=innovation,
            mahalanobis=per_point,
            chi2_threshold=self.chi2_gate,
            residual_rms_px=rms,
            n_measurements=n,
        )

    def along_close_sigma_m(
        self, points_base_xyz: np.ndarray, close_direction_base: np.ndarray
    ) -> float:
        """Posterior std of the tip position along the jaw-close direction.

        This is the number the supervisor gates on.  Pixel covariance is the
        wrong unit: a 4 px error at 0.2 m is a different close decision from
        the same 4 px at 0.8 m.
        """

        direction = np.asarray(close_direction_base, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return float("inf")
        direction = direction / norm
        # First-order: camera-frame point covariance from the pose block of P.
        # A translation error maps 1-1; a rotation error maps as -[p]x omega.
        point = np.asarray(points_base_xyz, dtype=np.float64).reshape(3)
        camera_point = self.state.rotation() @ point + self.state.translation
        skew = np.array(
            (
                (0.0, -camera_point[2], camera_point[1]),
                (camera_point[2], 0.0, -camera_point[0]),
                (-camera_point[1], camera_point[0], 0.0),
            ),
            dtype=np.float64,
        )
        pose_jacobian = np.concatenate((-skew, np.eye(3)), axis=1)
        pose_cov = self.state.covariance()[:6, :6]
        point_cov = pose_jacobian @ pose_cov @ pose_jacobian.T
        camera_direction = self.state.rotation() @ direction
        variance = float(camera_direction @ point_cov @ camera_direction)
        return float(np.sqrt(max(variance, 0.0)))


def observability_by_motion(
    records: Sequence[tuple[str, np.ndarray]],
) -> dict[str, dict[str, Any]]:
    """Average FIM spectrum per named motion type.

    ``records`` is ``(motion_type, jacobian)`` from sequential updates.  The
    table answers "which motion lights up which degree of freedom" without
    claiming a motion that never happened was informative.
    """

    grouped: dict[str, list[np.ndarray]] = {}
    for name, jacobian in records:
        grouped.setdefault(name, []).append(np.asarray(jacobian, dtype=np.float64))
    report: dict[str, dict[str, Any]] = {}
    for name, jacobians in grouped.items():
        spectra = [fim_spectrum(item, 1.0) for item in jacobians if item.size]
        if not spectra:
            continue
        width = max(item.size for item in spectra)
        padded = np.stack(
            [np.pad(item, (0, width - item.size)) for item in spectra], axis=0
        )
        report[name] = {
            "count": int(len(spectra)),
            "mean_eigenvalues": padded.mean(axis=0).tolist(),
            "min_eigenvalue": float(padded[:, -1].mean()),
        }
    return report
