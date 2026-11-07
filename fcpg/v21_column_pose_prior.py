"""Uncertain Column geometry from exposure-time full FK, never RGB evidence.

The historical base-XYZ affine fit supplies a nominal camera basis, not a
calibrated camera pose. A product Gaussian quadrature varies its orientation
and scale; a shared uncertain translation locates *both* fingers. An optional
RGB proposal updates only that shared centre. Its endpoints and separation
never become geometric labels. All uncertainty defaults are explicit modeling
assumptions, not measured calibration accuracy or joint-specific rules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from .so101_camera_calibration import ProjectionFit


@dataclass(frozen=True)
class ColumnCameraUncertainty:
    """Independent camera-axis rotation, log-scale and image translation priors.

    Translation and proposal-centre standard deviations are fractions of
    (image width, image height). ``pixel_noise_std`` defaults to the standard
    deviation of one pixel's uniform quantization error; callers should supply
    their measured contour uncertainty when available. These are continuous
    probability scales. The visual detector may check model consistency
    against the resulting shared-centre covariance; relative uncertainty is
    retained independently so foreshortening does not become a width rule.
    """

    rotation_std_deg: tuple[float, float, float] = (15., 15., 10.)
    log_scale_std: float = .2
    translation_std_fraction: tuple[float, float] = (.5, .5)
    proposal_center_std_fraction: tuple[float, float] = (.05, .05)
    pixel_noise_std: float = float(1. / np.sqrt(12.))

    def __post_init__(self):
        for name, size in (("rotation_std_deg", 3),
                           ("translation_std_fraction", 2),
                           ("proposal_center_std_fraction", 2)):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (size,) or not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"{name} must contain {size} finite nonnegative values")
        if not np.isfinite(self.log_scale_std) or not 0 <= self.log_scale_std <= 2:
            raise ValueError("log_scale_std must be finite and in [0, 2]")
        if not np.isfinite(self.pixel_noise_std) or self.pixel_noise_std <= 0:
            raise ValueError("pixel_noise_std must be finite and positive")


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return result.copy()


def _covariance(value, name):
    covariance = _array(value, (2, 2), name)
    if not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() < 0:
        raise ValueError(f"{name} must be symmetric positive semidefinite")
    return covariance


def _rotation(vector):
    angle = float(np.linalg.norm(vector))
    if angle == 0:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * (cross @ cross)


def _quadrature(std):
    return [(0., 1.)] if std == 0 else [
        (-np.sqrt(3.) * std, 1. / 6.), (0., 2. / 3.), (np.sqrt(3.) * std, 1. / 6.)]


def _gaussian(residual, covariance):
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("scoring covariance must be positive definite")
    distance = max(0., float(residual @ np.linalg.solve(covariance, residual)))
    return -.5 * (distance + logdet + len(residual) * np.log(2. * np.pi)), distance


@dataclass(frozen=True)
class ColumnPosePrior:
    """Numeric, moment-matched prior with auditable shared camera hypotheses.

    Pixel points are ordered [F, G]. ``score_pair`` ranks RGB evidence without
    a minimum separation gate. The prior cannot certify either endpoint's
    visibility, physical identity or correctness. ``q_at_image`` must be the
    actual exposure-aligned readback, not a target or future command.
    """

    world_pair_xyz: np.ndarray
    nominal_projection_matrix: np.ndarray
    nominal_pair_px: np.ndarray
    mean_pair_px: np.ndarray
    center_mean_px: np.ndarray
    center_cov_px2: np.ndarray
    relative_mean_px: np.ndarray
    relative_cov_px2: np.ndarray
    pair_cov_px2: np.ndarray
    camera_matrices: np.ndarray
    camera_parameters: np.ndarray
    hypothesis_weights: np.ndarray
    hypothesis_pairs_px: np.ndarray
    uncertainty: ColumnCameraUncertainty
    image_size: tuple[float, float]
    q_at_image: np.ndarray | None
    proposal_center_px: np.ndarray | None
    proposal_center_cov_px2: np.ndarray | None

    @classmethod
    def from_kinematics(cls, kinematics, q_at_image, fit: ProjectionFit, **kwargs):
        q = _array(q_at_image, (6,), "q_at_image")
        if (q < [-100., -100., -100., -100., -100., 0.]).any() or (q > 100.).any():
            raise ValueError("q_at_image exceeds the SO-101 FK convention; no silent clipping")
        if fit.view != "column" or tuple(fit.feature_names) != (
                "x_base", "y_base", "z_base", "bias"):
            raise ValueError("Column prior requires the base-XYZ Column fit")
        world, _local = kinematics.forward_anchor_frames(q[None])
        world = np.asarray(world, dtype=np.float64)
        if world.ndim != 3 or world.shape[0] != 1 or world.shape[1] < 3 or world.shape[2] != 3:
            raise ValueError("forward_anchor_frames must return [1, >=3, 3] world anchors")
        return cls.from_world_pair(world[0, 1:3], np.asarray(fit.coefficients).T,
                                   q_at_image=q, **kwargs)

    @classmethod
    def from_world_pair(cls, world_pair_xyz, projection_matrix, *,
                        image_size=(224, 224), proposal_pair=None,
                        proposal_center_cov_px2=None, uncertainty=None,
                        q_at_image=None):
        """Build from world F/G and affine [2,4] (or zero-bias [2,3]) matrix.

        Both points always share a camera matrix. Rotations are around the
        pair's world centre; uncertain translation absorbs the corresponding
        camera position. Image centre evidence updates translation analytically
        and never adjusts individual fingers. Bounds are not clipped to image
        borders: offscreen possibilities must remain visible to the caller.
        """
        uncertainty = uncertainty or ColumnCameraUncertainty()
        world = _array(world_pair_xyz, (2, 3), "world_pair_xyz")
        size = _array(image_size, (2,), "image_size")
        if (size <= 0).any():
            raise ValueError("image_size must be positive")
        matrix = np.asarray(projection_matrix, dtype=np.float64)
        if matrix.shape == (2, 3):
            matrix = np.column_stack((matrix, np.zeros(2)))
        matrix = _array(matrix, (2, 4), "projection_matrix")
        u, singular, vt = np.linalg.svd(matrix[:, :3], full_matrices=False)
        if singular[-1] <= np.finfo(float).eps * singular[0] or singular[0] == 0:
            raise ValueError("projection_matrix must identify two image axes")
        # Polar decomposition retains the affine fit's anisotropy and skew,
        # while exposing an orthonormal camera basis for shared 3-D rotations.
        basis2 = u @ vt
        basis = np.vstack((basis2, np.cross(basis2[0], basis2[1])))
        image_transform = (u * singular[None]) @ u.T
        nominal = world @ matrix[:, :3].T + matrix[:, 3]
        center = nominal.mean(0)
        center_cov = np.diag((size * uncertainty.translation_std_fraction) ** 2)
        proposal_center = proposal_cov = None
        if proposal_pair is not None:
            proposal_center = _array(proposal_pair, (2, 2), "proposal_pair").mean(0)
            proposal_cov = (_covariance(proposal_center_cov_px2, "proposal_center_cov_px2")
                            if proposal_center_cov_px2 is not None else
                            np.diag((size * uncertainty.proposal_center_std_fraction) ** 2))
            # Gaussian conditioning works with zero camera or proposal variance
            # too; sensor quantization makes the innovation covariance proper.
            innovation = center_cov + proposal_cov + np.eye(2) * uncertainty.pixel_noise_std ** 2 / 2.
            gain = np.linalg.solve(innovation, center_cov).T
            center = center + gain @ (proposal_center - center)
            center_cov = center_cov - gain @ center_cov
            center_cov = .5 * (center_cov + center_cov.T)
        elif proposal_center_cov_px2 is not None:
            raise ValueError("proposal_center_cov_px2 requires proposal_pair")
        stds = [*np.deg2rad(uncertainty.rotation_std_deg), uncertainty.log_scale_std]
        weights, params, matrices, pairs = [], [], [], []
        world_center = world.mean(0)
        for quadrature in product(*[_quadrature(std) for std in stds]):
            parameters = np.array([point[0] for point in quadrature])
            weight = float(np.prod([point[1] for point in quadrature]))
            linear = np.exp(parameters[3]) * image_transform @ _rotation(parameters[:3])[:2] @ basis
            offset = center - linear @ world_center
            matrices.append(np.column_stack((linear, offset)))
            pairs.append((world - world_center) @ linear.T + center)
            params.append(parameters)
            weights.append(weight)
        weights = np.asarray(weights)
        weights /= weights.sum()
        pairs = np.asarray(pairs)
        # Compute displacement before adding image translation. This avoids
        # catastrophic cancellation when the old camera bias is far offscreen.
        relative = np.einsum("nkj,j->nk", np.asarray(matrices)[:, :, :3], world[1] - world[0])
        relative_mean = weights @ relative
        residual = relative - relative_mean
        relative_cov = (residual.T * weights) @ residual
        mean_pair = center + np.array([-.5, .5])[:, None] * relative_mean
        pair_cov = (np.kron(np.ones((2, 2)), center_cov)
                    + np.kron(np.array([[.25, -.25], [-.25, .25]]), relative_cov))
        return cls(world, matrix, nominal, mean_pair, center, center_cov,
                   relative_mean, relative_cov, pair_cov, np.asarray(matrices),
                   np.asarray(params), weights, pairs, uncertainty, tuple(size),
                   None if q_at_image is None else _array(q_at_image, (6,), "q_at_image"),
                   proposal_center, proposal_cov)

    def score_pair(self, pair, *, allow_swap=True):
        """Continuous joint position/relative likelihood; no geometric gates.

        Returns the best ordering unless ``allow_swap=False``. Probability is
        an association under this *model*, never verified physical identity.
        Coincident candidate endpoints are valid and have association 0.5.
        """
        candidate = _array(pair, (2, 2), "pair")
        noise = self.uncertainty.pixel_noise_std ** 2
        center_log, center_d2 = _gaussian(candidate.mean(0) - self.center_mean_px,
                                         self.center_cov_px2 + np.eye(2) * noise / 2.)
        relative = candidate[1] - candidate[0]
        covariance = self.relative_cov_px2 + np.eye(2) * noise * 2.
        direct_log, direct_d2 = _gaussian(relative - self.relative_mean_px, covariance)
        reverse_log, reverse_d2 = _gaussian(-relative - self.relative_mean_px, covariance)
        swapped = bool(allow_swap and reverse_log > direct_log)
        relative_log, relative_d2 = ((reverse_log, reverse_d2) if swapped else (direct_log, direct_d2))
        probability = float(np.exp(relative_log - np.logaddexp(direct_log, reverse_log)))
        return dict(log_likelihood=float(center_log + relative_log),
                    relative_log_likelihood=float(relative_log),
                    likelihood=float(np.exp(-.5 * (center_d2 + relative_d2))),
                    relative_likelihood=float(np.exp(-.5 * relative_d2)),
                    center_mahalanobis_sq=center_d2, relative_mahalanobis_sq=relative_d2,
                    swapped=swapped, identity_probability=probability,
                    ordered_pair=(candidate[::-1] if swapped else candidate).tolist(),
                    physical_identity_verified=False)

    def as_dict(self, *, standard_deviations=2., include_hypotheses=False):
        """JSON-safe modeled ranges, parameters and evidence provenance.

        Intervals use propagated moments; they are not calibrated coverage
        guarantees. ``overlap_affinity`` is a continuous pixel-resolution
        affinity, not a declaration that two independently visible tips exist.
        """
        if not np.isfinite(standard_deviations) or standard_deviations <= 0:
            raise ValueError("standard_deviations must be finite and positive")
        std = np.sqrt(np.maximum(0., np.diag(self.pair_cov_px2))).reshape(2, 2)
        bounds = np.stack((self.mean_pair_px - standard_deviations * std,
                           self.mean_pair_px + standard_deviations * std), axis=1)
        physical_span = float(np.linalg.norm(self.world_pair_xyz[1] - self.world_pair_xyz[0]))
        singular = np.linalg.svd(self.nominal_projection_matrix[:, :3], compute_uv=False)
        nominal_span = float(np.linalg.norm(self.nominal_pair_px[1] - self.nominal_pair_px[0]))
        spans = np.linalg.norm(self.hypothesis_pairs_px[:, 1] - self.hypothesis_pairs_px[:, 0], axis=1)
        mean_span = float(self.hypothesis_weights @ spans)
        # Gaussian expectation of exp(-||relative||^2 / (4 pixel_noise^2)).
        # This remains smooth and finite as the nominal projection collapses.
        resolution_cov = np.eye(2) * 2. * self.uncertainty.pixel_noise_std ** 2
        overlap_log, _ = _gaussian(self.relative_mean_px, self.relative_cov_px2 + resolution_cov)
        _, resolution_logdet = np.linalg.slogdet(resolution_cov)
        overlap = float(np.exp(overlap_log + np.log(2. * np.pi) + .5 * resolution_logdet))
        result = dict(
            available=True, source="column-full-fk-camera-uncertainty", inferred=True,
            observed=False, accuracy_verified=False, physical_identity_verified=False,
            q_at_image=None if self.q_at_image is None else self.q_at_image.tolist(),
            q_conditioned=self.q_at_image is not None, full_joint_fk=self.q_at_image is not None,
            camera_model="uncertain affine weak perspective; not metric calibration",
            coordinate_space="caller projection pixels", image_size=list(self.image_size),
            world_fg_xyz=self.world_pair_xyz.tolist(),
            nominal_projection_matrix=self.nominal_projection_matrix.tolist(),
            nominal_fg_px=self.nominal_pair_px.tolist(), mean_fg_px=self.mean_pair_px.tolist(),
            fg_bounds_px=bounds.tolist(), bounds_standard_deviations=float(standard_deviations),
            bounds_kind="unclipped propagated-moment intervals; uncalibrated coverage",
            center_mean_px=self.center_mean_px.tolist(), center_cov_px2=self.center_cov_px2.tolist(),
            relative_mean_px=self.relative_mean_px.tolist(), relative_cov_px2=self.relative_cov_px2.tolist(),
            pair_cov_px2=self.pair_cov_px2.tolist(), physical_tip_separation_m=physical_span,
            nominal_projected_span_px=nominal_span, expected_projected_span_px=mean_span,
            projected_span_std_px=float(np.sqrt(self.hypothesis_weights @ (spans - mean_span) ** 2)),
            foreshortening_fraction=nominal_span / (singular[0] * physical_span) if physical_span else 0.,
            overlap_affinity=overlap, zero_projected_separation_allowed=True,
            proposal_center_px=None if self.proposal_center_px is None else self.proposal_center_px.tolist(),
            proposal_center_cov_px2=None if self.proposal_center_cov_px2 is None else self.proposal_center_cov_px2.tolist(),
            proposal_used_as_endpoint=False, proposal_used_for_shared_center=self.proposal_center_px is not None,
            uncertainty=asdict(self.uncertainty), uncertainty_status="explicit assumptions; not empirically calibrated",
            hypothesis_count=len(self.hypothesis_weights),
            hypothesis_parameter_names=["camera_rx_rad", "camera_ry_rad", "camera_rz_rad", "log_scale"],
            hypothesis_method="independent three-node Gaussian quadrature; shared camera for F and G")
        if include_hypotheses:
            result.update(hypothesis_weights=self.hypothesis_weights.tolist(),
                          hypothesis_parameters=self.camera_parameters.tolist(),
                          hypothesis_camera_matrices=self.camera_matrices.tolist(),
                          hypothesis_fg_px=self.hypothesis_pairs_px.tolist())
        return result
