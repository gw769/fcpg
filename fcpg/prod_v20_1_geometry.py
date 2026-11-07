"""V20.1 shared geometry readout with full *joint-pair* uncertainty.

Reuses V20's explicit calibrated-position conversion and real/CAD camera
adapters unchanged.  It does not invent camera extrinsics for archive rows.
The endpoint Jacobian has four rows; J Q J^T retains cross-tip covariance.
"""
import torch

from .prod_v20_observer import V20MechanicalProjection
from .prod_v20_geometry import V20FittedMechanicalProjection
from .prod_v20_1_belief import (
    JointPairBelief, broad_pair_prior, pair_from_means, select_pair_belief,
)


class _JointProjectionMixin:
    def make_prior(self, q_position, sigma_px=36.0):
        if sigma_px <= 0:
            raise ValueError("prior sigma must be positive")
        pixels, valid = self.project(q_position)
        means = pixels[:, :, None].expand(-1, -1, 4, -1, -1)
        endpoint_covariance = torch.eye(4, device=pixels.device).expand(*means.shape[:3], 4, 4) * sigma_px**2
        # Geometry provides identity, not direct evidence of visibility.
        prior = pair_from_means(means, visible_probability=torch.full_like(means[..., 0], .5),
                                joint_covariance=endpoint_covariance,
                                identity_probability=torch.ones_like(means[..., 0, 0]))
        unknown = broad_pair_prior(len(pixels), pixels.device)
        return select_pair_belief(valid.all(-1), prior, unknown)

    def propagate(self, previous_actual, q_previous, q_predicted, q_std=None, *,
                  q_covariance=None, process_variance_px2=4.0):
        """Carry the last actual visual correction and actuator uncertainty.

        The mean is previous visual pair + FK(q_predicted)-FK(q_previous).
        Covariance receives the full predicted-q Jacobian covariance and
        independent endpoint process variance.  q_std is in public calibrated
        position units, never radians/degrees guessed inside this layer.
        Unknown geometry carries mean unchanged and increases uncertainty.
        """
        if not isinstance(previous_actual, JointPairBelief):
            raise TypeError("V20.1 propagation requires JointPairBelief")
        if q_std is not None and q_covariance is not None:
            raise ValueError("provide q_std or q_covariance, not both")
        if process_variance_px2 < 0:
            raise ValueError("process variance cannot be negative")
        before, before_valid = self.project(q_previous)
        after, after_valid = self.project(q_predicted)
        valid = before_valid.all(-1) & after_valid.all(-1)
        # Projection returns physical F/M; belief endpoints retain the incoming
        # visual geometric gauge.  Never add physical-M motion to geometric F.
        identity = previous_actual.identity_probability.float()
        before_pair = before[:, :, None]
        direct_distance = (previous_actual.means_px - before_pair).square().sum((-1, -2))
        swapped_distance = (previous_actual.means_px - before_pair.flip(-2)).square().sum((-1, -2))
        swap = (identity < .5) | ((identity == .5) & (swapped_distance < direct_distance))
        physical_delta = (after - before)[:, :, None]
        delta = torch.where(swap[..., None, None], physical_delta.flip(-2), physical_delta)
        means = previous_actual.means_px.float() + delta
        covariance = previous_actual.endpoint_joint_covariance.float()
        covariance = covariance + torch.eye(4, device=means.device) * process_variance_px2
        # MAP-gauge mean is retained instead of averaging opposing identities.
        # Add the mean-square alternative displacement about that chosen mean.
        # This is conservative branch uncertainty, not exact mixture reduction.
        alternative_offset = (physical_delta.flip(-2) - physical_delta).flatten(-2)
        other_probability = torch.where(swap, identity, 1 - identity)
        covariance = covariance + (other_probability[..., None, None] *
            alternative_offset[..., :, None] * alternative_offset[..., None, :])
        if q_std is not None:
            if q_std.shape != q_predicted.shape or not torch.isfinite(q_std).all() or (q_std < 0).any():
                raise ValueError("q_std must be finite non-negative calibrated-position uncertainty")
            q_covariance = torch.diag_embed(q_std.float().square())
        if q_covariance is not None:
            shape = (*q_predicted.shape, q_predicted.shape[-1])
            if q_covariance.shape != shape or not torch.isfinite(q_covariance).all():
                raise ValueError("q_covariance must be finite [B,J,J]")
            q_covariance = q_covariance.float()
            if not torch.allclose(q_covariance, q_covariance.transpose(-1, -2), atol=1e-5) or (torch.linalg.eigvalsh(q_covariance) < -1e-6).any():
                raise ValueError("q_covariance must be symmetric positive semidefinite")
            derivatives = []
            for joint in range(q_predicted.shape[-1]):
                shift = torch.zeros_like(q_predicted)
                shift[:, joint] = .01
                plus, _ = self.project(q_predicted + shift)
                minus, _ = self.project(q_predicted - shift)
                derivatives.append(((plus - minus) / .02).flatten(-2))
            jacobian = torch.stack(derivatives, -1)  # [B,V,4,J]
            with torch.autocast(device_type=means.device.type, enabled=False):
                propagated = jacobian @ q_covariance[:, None] @ jacobian.transpose(-1, -2)
            permutation = [2, 3, 0, 1]
            swapped_covariance = propagated[..., permutation, :][..., permutation]
            covariance = covariance + torch.where(swap[..., None, None],
                swapped_covariance[:, :, None], propagated[:, :, None])
        # In-frame alone is not proof of visibility; outside cannot be visible.
        inside = ((means >= 0) & (means < 224)).all(-1)
        predicted = pair_from_means(means, joint_covariance=covariance,
                                    visible_probability=previous_actual.visible_probability * inside.to(means),
                                    log_weights=previous_actual.log_weights,
                                    identity_probability=previous_actual.identity_probability)
        unknown = JointPairBelief(previous_actual.log_weights, previous_actual.state_mean,
                                  previous_actual.state_covariance + torch.eye(4, device=means.device) * 16,
                                  previous_actual.visible_probability, previous_actual.identity_probability)
        return select_pair_belief(valid, predicted, unknown)


class V201MechanicalProjection(_JointProjectionMixin, V20MechanicalProjection):
    """CAD actual K/T adapter; wrist tool frame and column base frame."""


class V201FittedMechanicalProjection(_JointProjectionMixin, V20FittedMechanicalProjection):
    """Existing approximate real pixel fits, not metric camera extrinsics."""


# Explicit spelling aliases for external callers, not alternative contracts.
V20_1MechanicalProjection = V201MechanicalProjection
V20_1FittedMechanicalProjection = V201FittedMechanicalProjection
