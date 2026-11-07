"""Episode-local mechanical-reference memory, driven by *actual* readbacks.

There is no learned head, online optimizer, stereo triangulation or six-DoF
pose estimate here. Each visual pair is lifted into its camera's mechanical
reference (wrist/tool, column/base). Metric cameras retain the nominal-depth
hypothesis and explicitly uncertain ray depth; affine fits retain only their
model-coordinate interpretation, never a claim of measured metric depth.

Hold one captured memory through an occlusion. ``elapsed_s`` is time SINCE
that capture, not time since the last call. Projecting the memory twice does
not constitute two measurements or add process/readback noise twice.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch

from .prod_v20_1_belief import JointPairBelief, pair_from_means
from .prod_v20_1_geometry import V201FittedMechanicalProjection, V201MechanicalProjection
from .robot_kinematic_graph import project_reference_points


@dataclass(frozen=True)
class V203CorrectionMemory:
    source_belief: JointPairBelief
    reference_correction: torch.Tensor  # [B,2,4,2,3], tool / base coordinates
    reference_covariance: torch.Tensor  # [B,2,4,6,6], includes unseen ray depth
    source_q_position: torch.Tensor  # [B,2,J], per-view capture readback
    available: torch.Tensor  # [B,2], valid geometry AND actual capture
    swap: torch.Tensor  # [B,2,4], physical F/M -> incoming visual gauge
    observed_rank: torch.Tensor  # [B,2,4], <=4 of six endpoint coordinates
    update_mask: torch.Tensor  # [B,2], THIS capture accepted actual visual evidence
    metric_3d: bool
    geometry_kind: str
    reference_frames: tuple[str, str] = ("tool", "base")

    def detach(self):
        return replace(self, **{f.name: getattr(self, f.name).detach()
            for f in fields(self) if isinstance(getattr(self, f.name), (torch.Tensor, JointPairBelief))})

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device)
            for f in fields(self) if isinstance(getattr(self, f.name), (torch.Tensor, JointPairBelief))})


def _choose(mask, new, old):
    if isinstance(new, JointPairBelief):
        return JointPairBelief(**{f.name: _choose(mask, getattr(new, f.name), getattr(old, f.name))
                                 for f in fields(new)})
    return torch.where(mask.reshape(*mask.shape, *([1] * (new.ndim - mask.ndim))), new, old)


class _ActualTransport:
    # These are explicit modeling defaults, not claimed calibrated sensor noise.
    def __init__(self, *args, process_variance_reference_per_s=1e-5,
                 nullspace_std_reference=.25, unavailable_variance_px2_per_s=4., **kwargs):
        super().__init__(*args, **kwargs)
        rates = (process_variance_reference_per_s, nullspace_std_reference,
                 unavailable_variance_px2_per_s)
        if not all(torch.isfinite(torch.tensor(x)) and x >= 0 for x in rates) or nullspace_std_reference == 0:
            raise ValueError("finite non-negative process rates and positive nullspace std required")
        self.process_variance_reference_per_s = float(process_variance_reference_per_s)
        self.nullspace_std_reference = float(nullspace_std_reference)
        self.unavailable_variance_px2_per_s = float(unavailable_variance_px2_per_s)

    def _check_q(self, q):
        if not isinstance(q, torch.Tensor) or q.ndim != 2 or not q.is_floating_point() or not torch.isfinite(q).all():
            raise ValueError("q must be finite floating calibrated positions [B,J]")
        return q.float()

    def _reference_points(self, q):
        indices = [self.kinematics.spec.fixed_tip_index, self.kinematics.spec.moving_tip_index]
        return self.kinematics.anchors_in_view_references(q)[:, :, indices]

    def _camera_mask(self, q):
        valid = self.camera_valid.to(device=q.device, dtype=torch.bool)
        return valid[None].expand(len(q), -1) if valid.ndim == 1 else valid

    def _nominal_in_gauge(self, q, swap):
        points = self._reference_points(q)[:, :, None].expand(-1, -1, 4, -1, -1)
        return torch.where(swap[..., None, None], points.flip(-2), points)

    def _reference_jacobian(self, points, q):
        # Full four-pixel / six-reference-coordinate Jacobian retains cross-tip
        # covariance. No pixel expectation is used as a neural feature readout.
        derivatives = []
        for coordinate in range(6):
            shift = torch.zeros_like(points)
            shift[..., coordinate // 3, coordinate % 3] = 1e-4
            plus, _ = self._project_references(points + shift, q)
            minus, _ = self._project_references(points - shift, q)
            derivatives.append(((plus - minus) / 2e-4).flatten(-2))
        return torch.stack(derivatives, -1)

    @torch.no_grad()
    def capture_correction(self, belief, q, view_valid=None, previous_memory=None):
        """Capture actual visual evidence; merge unobserved views unchanged.

        The 3-D mean is a reference hypothesis, NOT measured depth. Its null
        space covariance is explicitly retained. Caller owns capture clocks
        and must reset only clocks of views that actually received evidence.
        """
        q = self._check_q(q)
        belief.validate()
        if len(belief.log_weights) != len(q):
            raise ValueError("belief and q batch sizes differ")
        valid = torch.ones(len(q), 2, device=q.device, dtype=torch.bool) if view_valid is None else torch.as_tensor(view_valid, device=q.device, dtype=torch.bool)
        if valid.shape != (len(q), 2):
            raise ValueError("view_valid must be [B,2]")
        with torch.autocast(device_type=q.device.type, enabled=False):
            # A missing H0 image is not missing proprioception: seed a broad
            # nominal mechanical hypothesis, explicitly without visual credit.
            if previous_memory is None:
                belief = _choose(valid, belief, self.make_prior(q))
            nominal, geometric_valid = self.project(q)
            direct = (belief.means_px - nominal[:, :, None]).square().sum((-1, -2))
            swapped = (belief.means_px - nominal[:, :, None].flip(-2)).square().sum((-1, -2))
            identity = belief.identity_probability
            swap = (identity < .5) | ((identity == .5) & (swapped < direct))
            reference = self._nominal_in_gauge(q, swap)
            corrected = self._lift_pixels(belief.means_px.double(), reference.double(), q)
            corrected = torch.where(geometric_valid.all(-1)[..., None, None, None], corrected, reference)
            jacobian = self._reference_jacobian(corrected, q).double()
            lift = torch.linalg.pinv(jacobian, rtol=1e-5)
            null = torch.eye(6, device=q.device, dtype=torch.float64) - lift @ jacobian
            covariance = lift @ belief.endpoint_joint_covariance.double() @ lift.transpose(-1, -2)
            covariance += self.nullspace_std_reference**2 * (null @ null.transpose(-1, -2))
            covariance = (covariance + covariance.transpose(-1, -2)) * .5
            available = valid & geometric_valid.all(-1) & self._camera_mask(q)
            rank = torch.linalg.matrix_rank(jacobian, rtol=1e-5)
            rank = torch.where(available[..., None], rank, torch.zeros_like(rank))
            memory = V203CorrectionMemory(belief.detach(), corrected-reference, covariance,
                q[:, None].expand(-1, 2, -1).clone(), available, swap, rank,
                available.clone(), self.metric_3d, self.geometry_kind)
        if previous_memory is not None:
            if (previous_memory.geometry_kind != self.geometry_kind or
                    previous_memory.source_q_position.shape != memory.source_q_position.shape):
                raise ValueError("cannot merge a different geometry or batch memory")
            # Unknown geometry is not permission to overwrite usable memory.
            updates = memory.available
            memory = replace(memory, **{f.name: _choose(updates, getattr(memory, f.name), getattr(previous_memory, f.name))
                for f in fields(memory) if f.name != "update_mask" and isinstance(getattr(memory, f.name), (torch.Tensor, JointPairBelief))})
        return memory

    def _elapsed(self, elapsed_s, q):
        elapsed = torch.as_tensor(elapsed_s, device=q.device, dtype=torch.float32)
        if elapsed.ndim == 0:
            elapsed = elapsed.expand(len(q), 2)
        elif elapsed.shape == (len(q),):
            elapsed = elapsed[:, None].expand(-1, 2)
        if elapsed.shape != (len(q), 2) or not torch.isfinite(elapsed).all() or (elapsed < 0).any():
            raise ValueError("elapsed_s must be non-negative scalar, [B] or [B,2] since capture")
        return elapsed

    def project_correction(self, memory, q_actual, elapsed_s, readback_std=None):
        """Follow absolute measured q; no constant-velocity extrapolation.

        Last-observed visibility metadata is retained, not multiplied into
        position and not repeatedly zeroed outside the image. New images own
        visibility updates. Camera projection validity is handled separately.
        ``readback_std`` describes this readback, never LeWM forecast error.
        """
        return self._project_memory(memory, q_actual, elapsed_s, readback_std, "readback_std")

    def project_forecast(self, memory, q_predicted, elapsed_s, forecast_std=None):
        """Read-only forecast for LeWM innovation/planning, never actual memory.

        This separate API prevents treating forecast error as readback sensor
        uncertainty. Neither API captures observations or modifies memory.
        """
        return self._project_memory(memory, q_predicted, elapsed_s, forecast_std, "forecast_std")

    def _joint_pixel_jacobian(self, memory, q):
        derivatives = []
        for joint in range(q.shape[-1]):
            step = torch.zeros_like(q)
            step[:, joint] = .01
            plus, _ = self._project_references(self._nominal_in_gauge(q+step, memory.swap)+memory.reference_correction, q+step)
            minus, _ = self._project_references(self._nominal_in_gauge(q-step, memory.swap)+memory.reference_correction, q-step)
            derivatives.append(((plus-minus)/.02).flatten(-2))
        return torch.stack(derivatives, -1).double()

    @torch.no_grad()
    def _project_memory(self, memory, q_position, elapsed_s, q_std, std_name):
        q = self._check_q(q_position)
        if memory.geometry_kind != self.geometry_kind or memory.source_q_position.shape != (len(q), 2, q.shape[-1]):
            raise ValueError("memory does not match current geometry/batch/joints")
        elapsed = self._elapsed(elapsed_s, q)
        if q_std is not None:
            q_std = torch.as_tensor(q_std, device=q.device, dtype=torch.float32)
            if q_std.shape != q.shape or not torch.isfinite(q_std).all() or (q_std < 0).any():
                raise ValueError(f"{std_name} must be finite non-negative [B,J] uncertainty")
        with torch.autocast(device_type=q.device.type, enabled=False):
            current_reference = self._nominal_in_gauge(q, memory.swap)
            points = current_reference + memory.reference_correction
            pixels, projected_valid = self._project_references(points, q)
            jacobian = self._reference_jacobian(points, q).double()
            covariance = jacobian @ memory.reference_covariance.double() @ jacobian.transpose(-1, -2)
            covariance += (elapsed[..., None, None, None] * self.process_variance_reference_per_s *
                           (jacobian @ jacobian.transpose(-1, -2)))
            # Keep the selected pair gauge, but do not hide uncertainty about
            # which physical jaw produced each visual endpoint. Anchor the
            # alternative trajectory to the SAME actual capture; averaging
            # opposing jaw identities into a zero-span mean is forbidden.
            identity = memory.source_belief.identity_probability
            other_probability = torch.where(memory.swap, identity, 1-identity)
            if bool((other_probability > 0).any()):
                source_reference = torch.stack([
                    self._nominal_in_gauge(memory.source_q_position[:, view], memory.swap)[:, view]
                    for view in range(2)], 1)
                alternative = (source_reference + memory.reference_correction +
                               current_reference.flip(-2)-source_reference.flip(-2))
                alternate_pixels, _ = self._project_references(alternative, q)
                alternate_error = (alternate_pixels.double()-pixels.double()).flatten(-2)
                covariance += (other_probability[..., None, None] *
                    alternate_error[..., :, None]*alternate_error[..., None, :])
            if q_std is not None:
                joint_jacobian = self._joint_pixel_jacobian(memory, q)
                scaled = joint_jacobian * q_std[:, None, None, None, :]
                covariance += scaled @ scaled.transpose(-1, -2)
            covariance = (covariance + covariance.transpose(-1, -2)) * .5 + torch.eye(4, device=q.device)*1e-4
            source = memory.source_belief
            result = pair_from_means(pixels, joint_covariance=covariance,
                log_weights=source.log_weights, visible_probability=source.visible_probability,
                identity_probability=source.identity_probability)
            unavailable = replace(source, state_covariance=source.state_covariance +
                elapsed[..., None, None, None] * self.unavailable_variance_px2_per_s * torch.eye(4, device=q.device))
            # available is *visual* calibration, not whether known FK works.
            # H0 with no image still follows q using its broad nominal prior.
            usable = self._camera_mask(q)[..., None] & projected_valid.all(-1)
            return _choose(usable, result, unavailable)

    def propagate_actual(self, previous_belief, q_previous, q_actual, elapsed_s, readback_std=None):
        """One-step convenience; persistent occlusions should retain memory.

        The capture must represent an actual visual endpoint. Do not repeatedly
        recapture projections as if they were new independent measurements.
        """
        memory = self.capture_correction(previous_belief, q_previous)
        return self.project_correction(memory, q_actual, elapsed_s, readback_std)


class V203MechanicalProjection(_ActualTransport, V201MechanicalProjection):
    """Known metric K/T; ray-depth mean is still only a nominal hypothesis."""
    metric_3d = True
    geometry_kind = "metric-camera-projection-reference-correction"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for matrix, size in ((self.camera_from_reference, 4), (self.intrinsics, 3), (self.image_transform, 3)):
            if matrix is None:
                continue
            if matrix.shape[-2:] != (size, size):
                raise ValueError("metric camera matrix shape differs")
            finite = torch.isfinite(matrix).all((-1, -2))
            safe = torch.where(finite[..., None, None], matrix, torch.eye(size, device=matrix.device))
            nonsingular = torch.linalg.det(safe).abs() > 1e-12
            if (self.camera_valid & ~(finite & nonsingular)).any():
                raise ValueError("known metric cameras require finite nonsingular K/T/image matrices")

    def _matrices(self, q, dtype=None):
        batch = len(q)
        transform = self._batch(self.camera_from_reference.to(q), batch, 4)
        intrinsic = self._batch(self.intrinsics.to(q), batch, 4)
        image = None if self.image_transform is None else self._batch(self.image_transform.to(q), batch, 4)
        # Inactive camera rows may be empty; they must not leak NaNs through
        # pinv/Jacobian computation, and still never become available evidence.
        valid = self._camera_mask(q)[..., None, None]
        transform = torch.where(valid, transform, torch.eye(4, device=q.device))
        intrinsic = torch.where(valid, intrinsic, torch.eye(3, device=q.device))
        image = None if image is None else torch.where(valid, image, torch.eye(3, device=q.device))
        dtype = q.dtype if dtype is None else dtype
        return transform.to(dtype=dtype), intrinsic.to(dtype=dtype), None if image is None else image.to(dtype=dtype)

    def _project_references(self, points, q):
        transform, intrinsic, image = self._matrices(q, points.dtype)
        pixels, _ = project_reference_points(points.flatten(-3, -2), transform, intrinsic, image)
        homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), -1)
        camera = torch.einsum("bvij,bvkaj->bvkai", transform, homogeneous)
        pixels = pixels.reshape(*points.shape[:-1], 2)
        valid = self._camera_mask(q)[..., None, None] & (camera[..., 2] > 1e-6) & torch.isfinite(pixels).all(-1)
        return pixels, valid

    def _reference_jacobian(self, points, q):
        transform, intrinsic, image = self._matrices(q, points.dtype)
        camera_matrix = intrinsic if image is None else image @ intrinsic
        homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), -1)
        camera = torch.einsum("bvij,bvkaj->bvkai", transform, homogeneous)[..., :3]
        projected = torch.einsum("bvij,bvkaj->bvkai", camera_matrix, camera)
        depth = projected[..., 2:3].clamp_min(1e-8)
        pixel = projected[..., :2]/depth
        derivative = (camera_matrix[..., None, None, :2, :] -
            pixel[..., :, None]*camera_matrix[..., None, None, 2:3, :])/depth[..., None]
        derivative = derivative @ transform[..., None, None, :3, :3]
        jacobian = points.new_zeros(*points.shape[:3], 4, 6)
        jacobian[..., :2, :3] = derivative[..., 0, :, :]
        jacobian[..., 2:, 3:] = derivative[..., 1, :, :]
        return jacobian

    def _lift_pixels(self, pixels, nominal_reference, q):
        transform, intrinsic, image = self._matrices(q, pixels.dtype)
        homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
        if image is not None:
            homogeneous = torch.einsum("bvij,bvkaj->bvkai", torch.linalg.pinv(image), homogeneous)
        rays = torch.einsum("bvij,bvkaj->bvkai", torch.linalg.pinv(intrinsic), homogeneous)
        nominal = torch.cat((nominal_reference, torch.ones_like(nominal_reference[..., :1])), -1)
        camera = torch.einsum("bvij,bvkaj->bvkai", transform, nominal)
        # Keep nominal depth, not an invented visual depth observation. The
        # entire unobservable ray direction receives null-space covariance.
        points_camera = rays * (camera[..., 2:3] / rays[..., 2:3].clamp_min(1e-8))
        points_camera = torch.cat((points_camera, torch.ones_like(points_camera[..., :1])), -1)
        lifted = torch.einsum("bvij,bvkaj->bvkai", torch.linalg.pinv(transform), points_camera)[..., :3]
        return torch.where(self._camera_mask(q)[..., None, None, None], lifted, nominal_reference)


class V203FittedMechanicalProjection(_ActualTransport, V201FittedMechanicalProjection):
    """Existing pixel fits: mechanical-coordinate lift, NOT metric 3-D evidence."""
    metric_3d = False
    geometry_kind = "existing-affine-pixel-fit-not-metric-extrinsics"

    def _affine_pixels(self, points, q):
        ones = torch.ones_like(points[:, 0, ..., :1])
        opening = (q[:, -1].clamp(0, 100)/100)[:, None, None, None].expand_as(ones)
        wrist = torch.cat((points[:, 0], opening, ones), -1) @ self.wrist_coefficients.to(points)
        column = torch.cat((points[:, 1], ones), -1) @ self.column_coefficients.to(points)
        return torch.stack((wrist, column), 1)

    def _project_references(self, points, q):
        pixels = self._affine_pixels(points, q)
        valid = self._camera_mask(q)[..., None, None] & torch.isfinite(pixels).all(-1)
        return pixels, valid

    def _joint_pixel_jacobian(self, memory, q):
        # Identical central differences, evaluated in one batched FK rather
        # than twelve Python/GPU launch sequences. Camera validity is applied
        # by the caller; affine coefficients are shared across the batch.
        batch, joints = q.shape
        shifts = torch.eye(joints, device=q.device, dtype=q.dtype) * .01
        perturbed = torch.cat((q[:, None] + shifts, q[:, None] - shifts), dim=1)
        flat_q = perturbed.reshape(batch * 2 * joints, joints)
        swap = memory.swap[:, None].expand(-1, 2*joints, -1, -1).reshape(batch * 2*joints, 2, 4)
        correction = memory.reference_correction[:, None].expand(-1, 2*joints, -1, -1, -1, -1)
        correction = correction.reshape(batch * 2*joints, 2, 4, 2, 3)
        points = self._nominal_in_gauge(flat_q, swap) + correction
        pixels = self._affine_pixels(points, flat_q).reshape(batch, 2*joints, 2, 4, 4)
        derivative = (pixels[:, :joints] - pixels[:, joints:]) / .02
        return derivative.permute(0, 2, 3, 4, 1).double()

    def _lift_pixels(self, pixels, nominal_reference, q):
        nominal, _ = self._project_references(nominal_reference, q)
        jacobian = self._reference_jacobian(nominal_reference, q)
        correction = torch.linalg.pinv(jacobian, rtol=1e-5) @ (pixels-nominal).flatten(-2)[..., None]
        return nominal_reference + correction.squeeze(-1).reshape_as(nominal_reference)

    def _reference_jacobian(self, points, q):
        local = torch.stack((self.wrist_coefficients[:3].T, self.column_coefficients[:3].T))
        jacobian = points.new_zeros(*points.shape[:3], 4, 6)
        jacobian[..., :2, :3] = local[None, :, None]
        jacobian[..., 2:, 3:] = local[None, :, None]
        return jacobian
