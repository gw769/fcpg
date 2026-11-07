"""Cross-round pinhole bundle for the column camera (milestone M1).

The live path projects FK anchors through an affine ridge fit
(``fcpg.so101_camera_calibration.ProjectionFit``): four coefficients per pixel
axis, no depth division.  That model cannot express metric depth, so every
"distance in metres" the supervisor computes downstream is really a constant-z
plane back-projection.  This module fits the missing thing -- a genuine pinhole
camera plus joint offsets -- from the task's own archived rounds.

What is estimated
-----------------

* ``log_fx, log_fy, cx, cy``: shared intrinsics.  The focal lengths are
  parameterized logarithmically (as in
  ``scripts/fit_so101_historical_column_center_pilot.py``) so the optimizer
  cannot walk them through zero or negative values.
* one ``cam_from_base`` SE(3) **per calibration epoch**.  The archived
  ``nominal`` channel is not a static map of FK even within one session --
  the live-anchor transform is recalibrated and toggled -- so grouping by
  session alone leaves 8-19 px that no camera model can explain.
* ``dq``: constant offsets on a masked subset of arm joints.  ``shoulder_pan``
  and ``wrist_roll`` are excluded by default because a pan offset is absorbed
  almost exactly by a camera rotation about the base vertical, following the
  gauge choice already frozen in ``fcpg.error_field``.

Joint offsets enter through a **precomputed FK Jacobian**:
``p(q + dq) ~= p(q) + J_q(q) dq``.  Re-running MuJoCo FK inside the optimizer
would cost one forward pass per frame per Jacobian column per iteration, which
is minutes per fit; the linearization is evaluated once, costs five extra
forward passes per frame, and is accurate for the small offsets a calibrated
arm can have.  It is an approximation and is reported as one.

What is audited
---------------

Fitting is the easy half.  A camera model fitted to nearly-constant-depth data
has a ``f`` <-> ``t_z`` valley, and a model fitted to a channel that was itself
seeded from the affine prediction will happily reproduce the affine prediction.
:class:`BundleFit` therefore carries the Jacobian SVD (condition number and the
named parameter directions the data barely constrains), the achieved depth
excitation, and per-parameter posterior standard deviations, so a
well-conditioned fit can be told apart from a lucky one.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from fcpg.observer_dataset import ANCHOR_COUNT, ObserverFrame
from fcpg.so101_kinematics import SO101MuJoCoKinematics


__all__ = [
    "ARM_JOINT_COUNT",
    "DEFAULT_JOINT_OFFSET_MASK",
    "LETTERBOX_FOV45_FOCAL_PX",
    "BundleObservations",
    "BundleFit",
    "CameraParameters",
    "build_observations",
    "epoch_group",
    "session_group",
    "focal_sweep_initialization",
    "fit_bundle",
    "project_points",
    "intrinsic_matrix",
    "apply_joint_offsets",
]


ARM_JOINT_COUNT = 5

#: Arm joints whose constant offset is estimated.  ``shoulder_pan`` (0) and
#: ``wrist_roll`` (4) are gauge directions against the camera extrinsic.
DEFAULT_JOINT_OFFSET_MASK = (1, 2, 3)

#: 224-px letterbox equivalent of a 45° vertical FOV on 640x480.  This is the
#: same formula ``fcpg.mujoco_camera.camera_intrinsic_from_fovy`` uses, mapped
#: through the uniform letterbox scale.  Task data does not identify focal
#: length (the ``f``/``t_z`` valley is flatter than the measurement noise), so
#: this prior is the operating point, not a fitted value.
LETTERBOX_FOV45_FOCAL_PX = (
    0.5 * 480.0 / math.tan(math.radians(45.0) / 2.0) * (224.0 / 640.0)
)

#: Finite-difference step for the FK joint Jacobian, in normalized joint units.
_FK_STEP = 0.5

#: Depth floor (metres).  Points closer than this are penalized rather than
#: projected, so the optimizer is pushed out of behind-the-camera basins
#: instead of dividing by something near zero.
_MIN_DEPTH_M = 0.02
_DEPTH_PENALTY_PX_PER_M = 1000.0

#: Physical box the parameters are confined to.  Without it the fit walks into
#: nonsense that still lowers the loss -- a principal point outside the canvas,
#: or a joint offset a quarter of the way across the joint range standing in for
#: camera-model error.  The focal bounds are the 224-px letterbox equivalents of
#: roughly 143-2286 px at 640x480.
_LOG_FOCAL_BOUNDS = (float(np.log(50.0)), float(np.log(800.0)))
_JOINT_OFFSET_BOUND = 10.0
_TRANSLATION_BOUND_M = 5.0
_MIN_CAMERA_DISTANCE_M = 0.05


@dataclass(frozen=True)
class BundleObservations:
    """Flattened correspondences plus the FK linearization, ready to fit.

    Rows are individual anchor observations, not frames, so per-anchor
    validity is already applied and nothing downstream has to carry masks.
    """

    group_index: np.ndarray  # [N] int, index into ``groups``
    groups: tuple[str, ...]
    points_base_xyz: np.ndarray  # [N, 3] FK anchor position at the encoder q
    joint_jacobian: np.ndarray  # [N, 3, ARM_JOINT_COUNT] d(point)/d(joint)
    pixels_xy: np.ndarray  # [N, 2] measured planning pixels
    anchor_index: np.ndarray  # [N] int in [0, ANCHOR_COUNT)
    frame_index: np.ndarray  # [N] int, index into the source frame list
    round_ids: tuple[str, ...]  # per source frame
    channel: str

    def __post_init__(self) -> None:
        n = len(self.points_base_xyz)
        for name, expected in (
            ("group_index", (n,)),
            ("points_base_xyz", (n, 3)),
            ("joint_jacobian", (n, 3, ARM_JOINT_COUNT)),
            ("pixels_xy", (n, 2)),
            ("anchor_index", (n,)),
            ("frame_index", (n,)),
        ):
            values = np.asarray(getattr(self, name))
            if values.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {values.shape}"
                )
        if n == 0:
            raise ValueError("no correspondences: nothing to fit")

    @property
    def count(self) -> int:
        return len(self.points_base_xyz)

    @property
    def group_count(self) -> int:
        return len(self.groups)


@dataclass(frozen=True)
class CameraParameters:
    """Unpacked bundle parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    rotations_cam_from_base: np.ndarray  # [S, 3, 3]
    translations_cam_from_base: np.ndarray  # [S, 3]
    joint_offsets: np.ndarray  # [ARM_JOINT_COUNT]
    joint_offset_mask: tuple[int, ...]

    def intrinsic(self) -> np.ndarray:
        return intrinsic_matrix(self.fx, self.fy, self.cx, self.cy)

    def extrinsic(self, group_index: int) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotations_cam_from_base[group_index]
        matrix[:3, 3] = self.translations_cam_from_base[group_index]
        return matrix

    def to_dict(self, groups: Sequence[str]) -> dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "joint_offsets_normalized": self.joint_offsets.tolist(),
            "joint_offset_mask": list(self.joint_offset_mask),
            "extrinsics_cam_from_base": {
                str(name): self.extrinsic(index).tolist()
                for index, name in enumerate(groups)
            },
        }


@dataclass(frozen=True)
class BundleFit:
    """A fitted bundle with the audit needed to decide whether to trust it."""

    parameters: CameraParameters
    parameter_labels: tuple[str, ...]
    parameter_vector: np.ndarray
    residuals_px: np.ndarray  # [N] per-correspondence reprojection error
    rms_px: float
    median_px: float
    p90_px: float
    sigma_px: float
    parameter_sigma: np.ndarray
    jacobian_singular_values: np.ndarray
    jacobian_condition: float
    weak_directions: tuple[dict[str, Any], ...]
    depth_m: np.ndarray
    optimizer_status: int
    n_observations: int
    joint_offset_mask: tuple[int, ...]
    #: Parameters resting on a bound.  Such a parameter is not an estimate: the
    #: data pushed it as far as it was allowed to go, so its value reports the
    #: bound and its sigma is meaningless.
    bound_hits: tuple[str, ...] = ()
    shared_focal: bool = True

    @property
    def depth_excitation(self) -> dict[str, float]:
        """How much the camera-frame depth actually varied.

        ``f`` and ``t_z`` are separable only through depth variation: at
        constant depth the projection depends on ``f / Z`` alone.  The
        relative spread reported here is the physical quantity that decides
        whether a focal length recovered from this data means anything.
        """

        depth = np.asarray(self.depth_m, dtype=np.float64)
        median = float(np.median(depth))
        return {
            "min_m": float(depth.min()),
            "median_m": median,
            "max_m": float(depth.max()),
            "relative_spread": float((depth.max() - depth.min()) / max(median, 1e-9)),
            "relative_std": float(depth.std() / max(median, 1e-9)),
        }

    def parameter_report(self) -> dict[str, dict[str, float]]:
        return {
            label: {
                "value": float(value),
                "sigma": float(sigma),
            }
            for label, value, sigma in zip(
                self.parameter_labels, self.parameter_vector, self.parameter_sigma
            )
        }


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.asarray(
        ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float64
    )


def _anchor_joint_jacobian(
    kinematics: SO101MuJoCoKinematics, states: np.ndarray
) -> np.ndarray:
    """Central-difference ``d(anchor xyz) / d(arm joint)`` at each state.

    Shape ``[N, ANCHOR_COUNT, 3, ARM_JOINT_COUNT]``.  Evaluated once, outside
    the optimizer.
    """

    values = np.asarray(states, dtype=np.float64)
    n = len(values)
    jacobian = np.zeros((n, ANCHOR_COUNT, 3, ARM_JOINT_COUNT), dtype=np.float64)
    for joint in range(ARM_JOINT_COUNT):
        step = np.zeros(6, dtype=np.float64)
        step[joint] = _FK_STEP
        plus = kinematics.forward_anchors(np.clip(values + step, -100.0, 100.0))
        minus = kinematics.forward_anchors(np.clip(values - step, -100.0, 100.0))
        jacobian[..., joint] = (plus - minus) / (2.0 * _FK_STEP)
    return jacobian


def epoch_group(frame: ObserverFrame) -> str:
    """Default extrinsic grouping: one camera pose per calibration epoch."""

    return frame.epoch_key


def session_group(frame: ObserverFrame) -> str:
    """Coarser grouping: one camera pose per session.

    Kept as an explicit alternative so the epoch/session choice can be ablated
    rather than assumed; on this archive the session-level fit is visibly worse
    because ``nominal`` is not a static map within a session.
    """

    return frame.session


def build_observations(
    frames: Sequence[ObserverFrame],
    *,
    channel: str,
    kinematics: SO101MuJoCoKinematics,
    group_of: Callable[[ObserverFrame], str] = epoch_group,
    groups: Sequence[str] | None = None,
    min_group_observations: int = 24,
) -> BundleObservations:
    """Flatten frames into correspondences and precompute the FK linearization.

    Passing ``groups`` pins the group ordering, so a held-out set can be
    evaluated against the extrinsics fitted on the training set.  A frame whose
    group is absent from a pinned ordering is dropped rather than silently
    reassigned: it has no fitted camera pose, so scoring it would be scoring an
    unrelated extrinsic.

    ``min_group_observations`` drops groups too small to determine a 6-DoF pose.
    A group holding two or three correspondences would otherwise contribute a
    pose that fits its own points exactly and constrains the shared intrinsics
    not at all, while adding six free parameters -- residuals would improve and
    the estimate would get worse.
    """

    usable = [frame for frame in frames if frame.channel(channel) is not None]
    if groups is not None:
        allowed = set(groups)
        usable = [frame for frame in usable if group_of(frame) in allowed]
    elif min_group_observations > 0:
        counts: dict[str, int] = {}
        for frame in usable:
            entry = frame.channel(channel)
            assert entry is not None
            counts[group_of(frame)] = counts.get(group_of(frame), 0) + int(
                entry.valid.sum()
            )
        usable = [
            frame
            for frame in usable
            if counts.get(group_of(frame), 0) >= min_group_observations
        ]
    if not usable:
        raise ValueError(f"no frame carries channel {channel!r}")

    states = np.stack([frame.joints for frame in usable], axis=0)
    joint_jacobian = _anchor_joint_jacobian(kinematics, states)

    if groups is not None:
        group_names = list(groups)
        group_lookup = {name: index for index, name in enumerate(group_names)}
    else:
        group_names = []
        group_lookup = {}
        for frame in usable:
            key = group_of(frame)
            if key not in group_lookup:
                group_lookup[key] = len(group_names)
                group_names.append(key)

    rows_group: list[int] = []
    rows_xyz: list[np.ndarray] = []
    rows_jac: list[np.ndarray] = []
    rows_px: list[np.ndarray] = []
    rows_anchor: list[int] = []
    rows_frame: list[int] = []
    for frame_position, frame in enumerate(usable):
        entry = frame.channel(channel)
        assert entry is not None
        content = frame.geometry.content_mask(entry.points_px)
        for anchor in range(ANCHOR_COUNT):
            if not bool(entry.valid[anchor]) or not bool(content[anchor]):
                continue
            rows_group.append(group_lookup[group_of(frame)])
            rows_xyz.append(frame.anchors_base_xyz[anchor])
            rows_jac.append(joint_jacobian[frame_position, anchor])
            rows_px.append(entry.points_px[anchor])
            rows_anchor.append(anchor)
            rows_frame.append(frame_position)

    if not rows_xyz:
        raise ValueError(
            f"channel {channel!r} has no valid in-content anchor observations"
        )
    return BundleObservations(
        group_index=np.asarray(rows_group, dtype=np.int64),
        groups=tuple(group_names),
        points_base_xyz=np.stack(rows_xyz, axis=0),
        joint_jacobian=np.stack(rows_jac, axis=0),
        pixels_xy=np.stack(rows_px, axis=0),
        anchor_index=np.asarray(rows_anchor, dtype=np.int64),
        frame_index=np.asarray(rows_frame, dtype=np.int64),
        round_ids=tuple(frame.round_id for frame in usable),
        channel=channel,
    )


def _parameter_labels(
    groups: Sequence[str], joint_offset_mask: Sequence[int]
) -> tuple[str, ...]:
    labels = ["log_fx", "log_fy", "cx", "cy"]
    for index in range(len(groups)):
        labels.extend(
            f"g{index}:{axis}" for axis in ("rx", "ry", "rz", "tx", "ty", "tz")
        )
    labels.extend(f"dq_{int(joint)}" for joint in joint_offset_mask)
    return tuple(labels)


def _unpack(
    vector: np.ndarray, n_groups: int, joint_offset_mask: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(intrinsics4, rotvecs [S,3], translations [S,3], dq [5])``."""

    intrinsics = vector[:4]
    pose_block = vector[4 : 4 + 6 * n_groups].reshape(n_groups, 6)
    offsets = np.zeros(ARM_JOINT_COUNT, dtype=np.float64)
    tail = vector[4 + 6 * n_groups :]
    for position, joint in enumerate(joint_offset_mask):
        offsets[int(joint)] = tail[position]
    return intrinsics, pose_block[:, :3], pose_block[:, 3:], offsets


def _predict(
    vector: np.ndarray,
    observations: BundleObservations,
    joint_offset_mask: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project every correspondence; returns ``(pixels [N,2], depth [N])``."""

    intrinsics, rotvecs, translations, offsets = _unpack(
        vector, observations.group_count, joint_offset_mask
    )
    fx, fy = np.exp(intrinsics[:2])
    cx, cy = intrinsics[2:4]

    shifted = observations.points_base_xyz + observations.joint_jacobian @ offsets
    rotations = Rotation.from_rotvec(rotvecs).as_matrix()  # [S, 3, 3]
    per_row_rotation = rotations[observations.group_index]
    per_row_translation = translations[observations.group_index]
    camera_xyz = (
        np.einsum("nij,nj->ni", per_row_rotation, shifted) + per_row_translation
    )
    depth = camera_xyz[:, 2]
    safe = np.maximum(depth, _MIN_DEPTH_M)
    pixels = np.column_stack(
        (fx * camera_xyz[:, 0] / safe + cx, fy * camera_xyz[:, 1] / safe + cy)
    )
    return pixels, depth


def _residual(
    vector: np.ndarray,
    observations: BundleObservations,
    joint_offset_mask: Sequence[int],
    scale: np.ndarray,
) -> np.ndarray:
    pixels, depth = _predict(vector, observations, joint_offset_mask)
    error = pixels - observations.pixels_xy
    behind = depth < _MIN_DEPTH_M
    if np.any(behind):
        error[behind] += (
            (_MIN_DEPTH_M - depth[behind, None]) * _DEPTH_PENALTY_PX_PER_M
        )
    return (error * scale[:, None]).ravel()


def focal_sweep_initialization(
    observations: BundleObservations,
    *,
    principal_point: tuple[float, float],
    focal_candidates: Sequence[float] = (80.0, 120.0, 160.0, 202.8, 260.0, 320.0, 500.0),
) -> np.ndarray:
    """Seed the bundle by sweeping focal length and solving PnP per session.

    A pinhole fit from a single focal guess falls into whichever ``f``/``t_z``
    valley the guess sits in, so the guess is swept and the best-scoring seed
    kept -- the same strategy as the historical column-center pilot, extended
    to a per-session extrinsic.
    """

    import cv2

    cx, cy = principal_point
    best: tuple[float, np.ndarray] | None = None
    for focal in focal_candidates:
        intrinsic = intrinsic_matrix(focal, focal, cx, cy)
        poses: list[np.ndarray] = []
        feasible = True
        for group in range(observations.group_count):
            mask = observations.group_index == group
            points = observations.points_base_xyz[mask]
            pixels = observations.pixels_xy[mask]
            if len(points) < 6:
                feasible = False
                break
            solved = None
            for flag in (cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE):
                try:
                    ok, rotation, translation = cv2.solvePnP(
                        points.astype(np.float64),
                        pixels.astype(np.float64),
                        intrinsic,
                        None,
                        flags=flag,
                    )
                except cv2.error:
                    continue
                if ok:
                    solved = np.concatenate(
                        (rotation.ravel(), translation.ravel())
                    )
                    break
            if solved is None:
                feasible = False
                break
            poses.append(solved)
        if not feasible:
            continue
        vector = np.concatenate(
            (
                np.log((focal, focal)),
                np.asarray((cx, cy), dtype=np.float64),
                np.concatenate(poses),
            )
        )
        pixels, _ = _predict(
            np.concatenate((vector, np.zeros(0))), observations, ()
        )
        score = float(
            np.mean(np.linalg.norm(pixels - observations.pixels_xy, axis=1))
        )
        if best is None or score < best[0]:
            best = (score, vector)
    if best is None:
        raise RuntimeError(
            "no focal-sweep seed produced a valid PnP pose for every group"
        )
    return best[1]


def _weak_directions(
    singular_values: np.ndarray,
    right_vectors: np.ndarray,
    labels: Sequence[str],
    *,
    max_reported: int = 3,
    loading_threshold: float = 0.15,
) -> tuple[dict[str, Any], ...]:
    """Name the parameter combinations the data constrains least.

    A raw condition number says a fit is ill-posed; it does not say *which*
    trade-off is ill-posed, which is the part that decides whether the number
    can be used.
    """

    reported: list[dict[str, Any]] = []
    order = np.argsort(singular_values)
    for row in order[:max_reported]:
        vector = right_vectors[row]
        loading_order = np.argsort(-np.abs(vector))
        involved = [
            {"parameter": labels[i], "loading": float(vector[i])}
            for i in loading_order[:4]
            if abs(vector[i]) > loading_threshold
        ]
        reported.append(
            {
                "singular_value": float(singular_values[row]),
                "relative_to_largest": float(
                    singular_values[row] / max(singular_values.max(), 1e-30)
                ),
                "loadings": involved,
            }
        )
    return tuple(reported)


def _central_jacobian(
    fun, x: np.ndarray, *, rel_step: float = 1e-6
) -> np.ndarray:
    base = fun(x)
    jacobian = np.empty((base.size, x.size), dtype=np.float64)
    for index in range(x.size):
        step = rel_step * max(abs(float(x[index])), 1.0)
        forward = np.array(x, dtype=np.float64)
        backward = np.array(x, dtype=np.float64)
        forward[index] += step
        backward[index] -= step
        jacobian[:, index] = (fun(forward) - fun(backward)) / (2.0 * step)
    return jacobian


def _parameter_bounds(
    n_groups: int,
    n_offsets: int,
    *,
    canvas_bounds: tuple[float, float, float, float],
    joint_offset_bound: float = _JOINT_OFFSET_BOUND,
) -> tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = canvas_bounds
    lower = [_LOG_FOCAL_BOUNDS[0], _LOG_FOCAL_BOUNDS[0], left, top]
    upper = [_LOG_FOCAL_BOUNDS[1], _LOG_FOCAL_BOUNDS[1], right, bottom]
    for _ in range(n_groups):
        lower.extend(
            [-np.inf] * 3
            + [-_TRANSLATION_BOUND_M, -_TRANSLATION_BOUND_M, _MIN_CAMERA_DISTANCE_M]
        )
        upper.extend([np.inf] * 3 + [_TRANSLATION_BOUND_M] * 3)
    bound = float(joint_offset_bound)
    if bound <= 0.0:
        raise ValueError("joint_offset_bound must be positive")
    if not np.isfinite(bound) and not np.isposinf(bound):
        raise ValueError("joint_offset_bound must be finite or +inf")
    lower.extend([-bound] * n_offsets)
    upper.extend([bound] * n_offsets)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def fit_bundle(
    observations: BundleObservations,
    *,
    principal_point: tuple[float, float] = (112.0, 112.0),
    joint_offset_mask: Sequence[int] = DEFAULT_JOINT_OFFSET_MASK,
    fix_principal_point: bool = False,
    shared_focal: bool = True,
    fix_focal_px: float | None = None,
    canvas_bounds: tuple[float, float, float, float] = (0.0, 28.0, 224.0, 196.0),
    joint_offset_bound: float = _JOINT_OFFSET_BOUND,
    initial_vector: np.ndarray | None = None,
    huber_delta_px: float = 6.0,
    max_nfev: int = 400,
    weights: np.ndarray | None = None,
) -> BundleFit:
    """Fit shared intrinsics, per-session extrinsics and joint offsets.

    A Huber loss is used because the measurement channels available offline
    contain real gross errors (a locator locking onto a knuckle, a grounding
    head drifting during occlusion).  A plain least-squares fit would let a
    handful of those set the focal length.

    ``shared_focal`` ties ``fy`` to ``fx``.  The column camera has square
    pixels and ``letterbox_v1`` rescales both axes by the same factor, so an
    independent ``fy`` is not a physical degree of freedom -- it is a direction
    the fit can use to absorb pose error, and leaving it free produced a 2:1
    aspect ratio on this data.  ``canvas_bounds`` keeps the principal point
    inside the region that actually carries camera content (the letterbox pad
    bands do not).  ``joint_offset_bound`` is the box on each free ``dq``
    (default ``_JOINT_OFFSET_BOUND``); a tighter physical range can be passed
    without changing the historical default.
    """

    mask = tuple(int(joint) for joint in joint_offset_mask)
    if len(set(mask)) != len(mask):
        raise ValueError("joint_offset_mask must not repeat a joint")
    if any(joint < 0 or joint >= ARM_JOINT_COUNT for joint in mask):
        raise ValueError(f"joint_offset_mask entries must be in [0, {ARM_JOINT_COUNT})")

    n_groups = observations.group_count
    expected = 4 + 6 * n_groups + len(mask)
    seed_only = 4 + 6 * n_groups
    if initial_vector is None:
        seed = focal_sweep_initialization(
            observations, principal_point=principal_point
        )
        x0 = np.concatenate((seed, np.zeros(len(mask), dtype=np.float64)))
    else:
        seed = np.asarray(initial_vector, dtype=np.float64)
        if seed.shape == (seed_only,):
            x0 = np.concatenate((seed, np.zeros(len(mask), dtype=np.float64)))
        elif seed.shape == (expected,):
            x0 = seed.copy()
        else:
            raise ValueError(
                f"initial vector must have {seed_only} (intrinsics+poses) or "
                f"{expected} (full) entries, got {seed.shape[0]}"
            )

    scale = (
        np.ones(observations.count, dtype=np.float64)
        if weights is None
        else np.sqrt(np.asarray(weights, dtype=np.float64))
    )
    if scale.shape != (observations.count,):
        raise ValueError("weights must have one entry per correspondence")

    lower, upper = _parameter_bounds(
        n_groups,
        len(mask),
        canvas_bounds=canvas_bounds,
        joint_offset_bound=joint_offset_bound,
    )
    if fix_focal_px is not None:
        if not (np.isfinite(fix_focal_px) and fix_focal_px > 0.0):
            raise ValueError("fix_focal_px must be finite and positive")
        x0[0] = x0[1] = float(np.log(fix_focal_px))
    x0 = np.clip(x0, lower, upper)

    free = np.ones(expected, dtype=bool)
    if fix_principal_point:
        free[2:4] = False
    if shared_focal:
        free[1] = False
    if fix_focal_px is not None:
        free[0] = False
        free[1] = False
    frozen_values = x0.copy()

    def expand(reduced: np.ndarray) -> np.ndarray:
        full = frozen_values.copy()
        full[free] = reduced
        if shared_focal and fix_focal_px is None:
            full[1] = full[0]
        return full

    def residual(reduced: np.ndarray) -> np.ndarray:
        return _residual(expand(reduced), observations, mask, scale)

    solution = least_squares(
        residual,
        x0[free],
        bounds=(lower[free], upper[free]),
        method="trf",
        loss="huber",
        f_scale=float(huber_delta_px),
        x_scale="jac",
        max_nfev=max_nfev,
    )
    vector = expand(solution.x)

    pixels, depth = _predict(vector, observations, mask)
    errors = np.linalg.norm(pixels - observations.pixels_xy, axis=1)
    # Robust noise scale: the Huber fit is only meaningful against a sigma that
    # the outliers did not set.
    sigma_px = float(1.4826 * np.median(np.abs(errors - np.median(errors))))
    sigma_px = max(sigma_px, 1e-6)

    jacobian = _central_jacobian(residual, solution.x)
    _, singular_values, right_vectors = np.linalg.svd(jacobian, full_matrices=False)
    labels = _parameter_labels(observations.groups, mask)
    free_labels = tuple(label for label, keep in zip(labels, free) if keep)
    condition = float(
        singular_values.max() / max(singular_values.min(), 1e-30)
    )
    weak = _weak_directions(singular_values, right_vectors, free_labels)

    # Posterior sigma from the Gauss-Newton covariance, pseudo-inverted so a
    # rank-deficient direction reports as infinite rather than as a huge but
    # finite number that looks like an answer.
    tolerance = 1e-10 * singular_values.max()
    usable = singular_values > tolerance
    covariance_diagonal = (
        right_vectors[usable] ** 2 / singular_values[usable, None] ** 2
    ).sum(axis=0)
    # A parameter loaded on a rank-deficient direction has infinite variance;
    # reporting a large finite number there would look like an answer.
    unidentified = (right_vectors[~usable] ** 2).sum(axis=0) > 1e-12
    covariance_diagonal = np.where(unidentified, np.inf, covariance_diagonal)
    free_sigma = sigma_px * np.sqrt(covariance_diagonal)
    parameter_sigma = np.full(expected, np.nan, dtype=np.float64)
    parameter_sigma[free] = free_sigma

    bound_slack = 1e-6 * np.maximum(np.abs(upper - lower), 1.0)
    at_bound = np.zeros(expected, dtype=bool)
    finite = np.isfinite(lower) & np.isfinite(upper)
    at_bound[finite] = (vector[finite] <= lower[finite] + bound_slack[finite]) | (
        vector[finite] >= upper[finite] - bound_slack[finite]
    )
    bound_hits = tuple(
        label for label, hit, keep in zip(labels, at_bound, free) if hit and keep
    )

    intrinsics, rotvecs, translations, offsets = _unpack(vector, n_groups, mask)
    parameters = CameraParameters(
        fx=float(np.exp(intrinsics[0])),
        fy=float(np.exp(intrinsics[1])),
        cx=float(intrinsics[2]),
        cy=float(intrinsics[3]),
        rotations_cam_from_base=Rotation.from_rotvec(rotvecs).as_matrix(),
        translations_cam_from_base=translations,
        joint_offsets=offsets,
        joint_offset_mask=mask,
    )
    return BundleFit(
        parameters=parameters,
        parameter_labels=labels,
        parameter_vector=vector,
        residuals_px=errors,
        rms_px=float(np.sqrt(np.mean(errors**2))),
        median_px=float(np.median(errors)),
        p90_px=float(np.quantile(errors, 0.9)),
        sigma_px=sigma_px,
        parameter_sigma=parameter_sigma,
        jacobian_singular_values=singular_values,
        jacobian_condition=condition,
        weak_directions=weak,
        depth_m=depth,
        optimizer_status=int(solution.status),
        n_observations=observations.count,
        joint_offset_mask=mask,
        bound_hits=bound_hits,
        shared_focal=bool(shared_focal),
    )


def project_points(
    parameters: CameraParameters,
    points_base_xyz: np.ndarray,
    group_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project base-frame points through a fitted camera; returns pixels, depth."""

    points = np.asarray(points_base_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_base_xyz must have shape [N, 3]")
    rotation = parameters.rotations_cam_from_base[group_index]
    translation = parameters.translations_cam_from_base[group_index]
    camera_xyz = points @ rotation.T + translation
    depth = camera_xyz[:, 2]
    safe = np.maximum(depth, _MIN_DEPTH_M)
    pixels = np.column_stack(
        (
            parameters.fx * camera_xyz[:, 0] / safe + parameters.cx,
            parameters.fy * camera_xyz[:, 1] / safe + parameters.cy,
        )
    )
    return pixels, depth


def apply_joint_offsets(
    observations: BundleObservations, joint_offsets: np.ndarray
) -> np.ndarray:
    """FK anchor positions corrected by the fitted joint offsets."""

    offsets = np.asarray(joint_offsets, dtype=np.float64)
    if offsets.shape != (ARM_JOINT_COUNT,):
        raise ValueError(f"joint_offsets must have shape [{ARM_JOINT_COUNT}]")
    return observations.points_base_xyz + observations.joint_jacobian @ offsets
