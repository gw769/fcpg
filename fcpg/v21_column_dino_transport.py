"""Exposure-paired Column DINO F/G transport for CEM ranking.

This module is deliberately small and has no camera, LeWM, or motor imports.
It fits only achieved readback motion against same-exposure Column F/G midpoints.
The resulting transport is local: requests outside the measured joint envelope
are rejected rather than replaced with an FK projection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# One CEM step of bounded linear extrapolation beyond the measured joint
# envelope; see ``ColumnDinoFGTransport.project``.
COLUMN_DINO_EXTRAPOLATION_DEG = 1.5


def _finite(value, shape, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f'{name} must be finite {shape}')
    return out.copy()


@dataclass(frozen=True)
class ColumnDinoFGSample:
    """One same-exposure Column M=mid(F,G) and its actual q readback."""

    q_at_image: np.ndarray
    midpoint_px: np.ndarray
    image_time_s: float

    def __post_init__(self):
        object.__setattr__(self, 'q_at_image', _finite(self.q_at_image, (6,), 'q_at_image'))
        object.__setattr__(self, 'midpoint_px', _finite(self.midpoint_px, (2,), 'Column DINO midpoint'))
        if not np.isfinite(self.image_time_s):
            raise ValueError('image time must be finite')


@dataclass(frozen=True)
class ColumnDinoFGTransport:
    """A response fit bound to one current DINO F/G exposure.

    ``jacobian_px_per_joint`` has five columns when the gripper was held during
    the fit (the recorded arm-response contract) and six when the retained
    window contains real gripper motion, in which case the gripper is its own
    regressor instead of invalidating every sample.  The strict measured
    envelope stays on J1--J5; the J6 delta is clamped to the measured gripper
    range so a candidate that opens or closes the jaws is still scored.
    """

    jacobian_px_per_joint: np.ndarray  # [u,v] / J1..J5 (+J6 when measured)
    capture_q: np.ndarray
    observed_midpoint_px: np.ndarray
    lower_q: np.ndarray
    upper_q: np.ndarray
    residual_rms_px: float
    design_condition: float
    sample_count: int
    lower_q6: float | None = None
    upper_q6: float | None = None

    def __post_init__(self):
        jacobian = np.asarray(self.jacobian_px_per_joint, dtype=np.float64)
        if (jacobian.ndim != 2 or jacobian.shape[0] != 2
                or jacobian.shape[1] not in (5, 6) or not np.isfinite(jacobian).all()):
            raise ValueError('Column DINO Jacobian must be finite [2,5] or [2,6]')
        object.__setattr__(self, 'jacobian_px_per_joint', jacobian.copy())
        object.__setattr__(self, 'capture_q', _finite(self.capture_q, (6,), 'capture q'))
        object.__setattr__(self, 'observed_midpoint_px',
                           _finite(self.observed_midpoint_px, (2,), 'observed Column M'))
        object.__setattr__(self, 'lower_q', _finite(self.lower_q, (5,), 'measured lower q'))
        object.__setattr__(self, 'upper_q', _finite(self.upper_q, (5,), 'measured upper q'))
        if ((self.upper_q < self.lower_q).any() or not np.isfinite(self.residual_rms_px)
                or self.residual_rms_px < 0 or not np.isfinite(self.design_condition)
                or self.design_condition <= 0 or self.sample_count < 6):
            raise ValueError('invalid Column DINO transport metadata')
        if jacobian.shape[1] == 6:
            if (self.lower_q6 is None or self.upper_q6 is None
                    or not np.isfinite([self.lower_q6, self.upper_q6]).all()
                    or float(self.upper_q6) < float(self.lower_q6)):
                raise ValueError('six-column Column DINO transport needs measured gripper bounds')
        elif self.lower_q6 is not None or self.upper_q6 is not None:
            raise ValueError('five-column Column DINO transport must not declare gripper bounds')

    def project(self, candidate_q):
        """Predict M from the fitted DINO response, inside a bounded envelope.

        The measured envelope is built from the arm's own motion, so its boundary
        is always the pose the arm happens to be at: a strict rule would discard
        every candidate that continues in the same direction (observed: 0
        admitted and a held arm).  One CEM step (1.5 deg) beyond the measured
        range is therefore clamped to the boundary and scored, which is the same
        bounded treatment the gripper column already gets; anything further out
        is still rejected.
        """
        q = _finite(candidate_q, (6,), 'candidate q')
        over = np.maximum(np.maximum(self.lower_q - q[:5], q[:5] - self.upper_q), 0.0)
        if float(over.max()) > COLUMN_DINO_EXTRAPOLATION_DEG + 1e-9:
            raise ValueError('candidate leaves measured Column DINO response envelope')
        delta = np.clip(q[:5], self.lower_q, self.upper_q) - self.capture_q[:5]
        if self.jacobian_px_per_joint.shape[1] == 6:
            gripper = float(np.clip(q[5], float(self.lower_q6), float(self.upper_q6)))
            delta = np.append(delta, gripper - self.capture_q[5])
        return self.observed_midpoint_px + self.jacobian_px_per_joint @ delta


def fit_column_dino_fg_transport(samples, *, max_gripper_delta=0.25,
                                 max_residual_rms_px=4.0, max_design_condition=100.0,
                                 minimum_samples=10):
    """Fit a local response from actual Column DINO F/G samples.

    The five-column path keeps the recorded contract: the gripper is held almost
    fixed during the arm-response fit so F/G midpoint motion is not silently
    attributed to J1--J5.  When the retained window does contain real gripper
    motion (a passive trial window), the gripper becomes a sixth regressor
    instead of invalidating the whole window: J1--J5 keep their support,
    identifiability and envelope rules, and the J6 delta is clamped to its
    measured range at projection time.  The caller must bind the resulting
    response to the *current* Column M using ``bind_column_dino_fg``.
    """
    if (type(minimum_samples) is not int or minimum_samples < 6
            or not np.isfinite([max_gripper_delta, max_residual_rms_px, max_design_condition]).all()
            or max_gripper_delta < 0 or max_residual_rms_px <= 0 or max_design_condition <= 1):
        raise ValueError('invalid Column DINO transport fit limits')
    rows = tuple(samples)
    if len(rows) < minimum_samples:
        raise ValueError('insufficient actual Column DINO F/G samples')
    if not all(isinstance(row, ColumnDinoFGSample) for row in rows):
        raise TypeError('Column DINO transport needs exposure-paired samples')
    times = np.asarray([row.image_time_s for row in rows], dtype=np.float64)
    if not np.all(np.diff(times) > 0):
        raise ValueError('Column DINO sample image times must strictly advance')
    q = np.stack([row.q_at_image for row in rows])
    midpoint = np.stack([row.midpoint_px for row in rows])
    columns = 6 if float(np.max(np.abs(q[:, 5] - q[0, 5]))) > float(max_gripper_delta) else 5
    design = q[1:, :columns] - q[0, :columns]
    response = midpoint[1:] - midpoint[0]
    scale = np.max(np.abs(design), axis=0)
    if (scale[:5] <= 1e-6).any():
        raise ValueError('an arm joint lacks actual Column DINO support')
    if columns == 6 and float(scale[5]) <= 1e-6:
        raise ValueError('measured gripper motion lacks actual Column DINO support')
    normalized = design / scale
    coefficients, _, rank, singular = np.linalg.lstsq(normalized, response, rcond=1e-6)
    if rank != columns or singular[-1] <= singular[0] * 1e-6:
        raise ValueError('Column DINO arm response is not independently identifiable')
    jacobian = (coefficients / scale[:, None]).T
    residual = response - normalized @ coefficients
    residual_rms = float(np.sqrt(np.mean(np.square(residual))))
    condition = float(singular[0] / singular[-1])
    if residual_rms > max_residual_rms_px:
        raise ValueError('Column DINO response residual exceeds the declared local fit limit')
    if condition > max_design_condition:
        raise ValueError('Column DINO response design is ill-conditioned')
    if float(np.linalg.norm(jacobian[:, :5])) < 1e-6:
        raise ValueError('Column DINO F/G response has no candidate-sensitive motion')
    result = dict(
        jacobian_px_per_joint=jacobian,
        lower_q=q[:, :5].min(axis=0),
        upper_q=q[:, :5].max(axis=0),
        residual_rms_px=residual_rms,
        design_condition=condition,
        sample_count=len(rows),
    )
    if columns == 6:
        result.update(lower_q6=float(q[:, 5].min()), upper_q6=float(q[:, 5].max()))
    return result


def bind_column_dino_fg(response, capture):
    """Bind a validated response to the newest exposure-time q and DINO M."""
    if not isinstance(response, dict):
        raise TypeError('Column DINO response must be the measured fit dictionary')
    if not isinstance(capture, ColumnDinoFGSample):
        raise TypeError('Column DINO capture must be exposure-paired')
    lower = _finite(response.get('lower_q'), (5,), 'measured lower q')
    upper = _finite(response.get('upper_q'), (5,), 'measured upper q')
    q = capture.q_at_image
    if (q[:5] < lower - 1e-9).any() or (q[:5] > upper + 1e-9).any():
        raise ValueError('current Column DINO capture leaves the measured response envelope')
    lower_q6, upper_q6 = response.get('lower_q6'), response.get('upper_q6')
    if lower_q6 is not None and upper_q6 is not None:
        if not float(lower_q6) - 1e-9 <= float(q[5]) <= float(upper_q6) + 1e-9:
            raise ValueError('current Column DINO capture leaves the measured gripper envelope')
    return ColumnDinoFGTransport(
        jacobian_px_per_joint=response.get('jacobian_px_per_joint'),
        capture_q=q,
        observed_midpoint_px=capture.midpoint_px,
        lower_q=lower,
        upper_q=upper,
        residual_rms_px=float(response.get('residual_rms_px')),
        design_condition=float(response.get('design_condition')),
        sample_count=int(response.get('sample_count')),
        lower_q6=None if lower_q6 is None else float(lower_q6),
        upper_q6=None if upper_q6 is None else float(upper_q6),
    )


def _actual_response_endpoint(result, measured):
    if 'endpoint_q_position' not in result:
        return measured[-1]
    endpoint = _finite(result['endpoint_q_position'], (6,), 'actual endpoint q')
    stamp = float(result['endpoint_time_s'])
    times = np.asarray(result['readback_times_s'], dtype=np.float64)
    if (times.shape != (len(measured),) or not np.isfinite(times).all()
            or not np.isfinite(stamp) or stamp < times[-1]):
        raise ValueError('actual endpoint must follow the original measured q path')
    return endpoint


def require_actual_probe_response(result, joint, direction, *, minimum_motion):
    """Return the achieved arm delta or reject a commanded-but-unmoved probe.

    A completion/Goal_Position receipt is not an exposure-paired response.  The
    caller must retain a calibration sample only when its source-to-final
    Present_Position readback moved in the requested arm-joint direction.
    """
    if (type(joint) is not int or not 0 <= joint < 6 or type(direction) is not int
            or direction not in (-1, 1) or not np.isfinite(minimum_motion)
            or minimum_motion <= 0):
        raise ValueError('invalid actual Column DINO probe response requirement')
    try:
        source = _finite(result.get('source_q_position'), (6,), 'probe source q')
        measured = np.asarray(result.get('measured_q_path'), dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        raise ValueError('Column DINO probe lacks actual q readback') from None
    if measured.ndim != 2 or measured.shape[1] != 6 or len(measured) < 2 or not np.isfinite(measured).all():
        raise ValueError('Column DINO probe lacks a finite actual q path')
    delta = _actual_response_endpoint(result, measured) - source
    if direction * delta[joint] <= minimum_motion:
        raise ValueError('Column DINO probe did not achieve the requested actual joint response')
    if np.max(np.abs(np.delete(delta, joint))) > minimum_motion:
        raise ValueError('Column DINO probe produced mixed actual joint motion')
    return delta


class CEMActualResponseError(ValueError):
    """A valid actual trace failed the existing directional response check."""

    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        blocked = diagnostic['blocked_joint_indices']
        details = ', '.join(
            f'J{axis + 1}: command_peak={diagnostic["commanded_peak_delta"][axis]:.6g}, '
            f'actual_directional_peak={diagnostic["achieved_directional_motion"][axis]:.6g}'
            for axis in blocked)
        scope = 'arm' if diagnostic['checked_joint_count'] == 5 else 'joint'
        super().__init__(f'CEM command did not achieve the requested actual {scope} response; '
                         f'{details}; required > {diagnostic["minimum_motion"]:.6g}')


def require_actual_cem_response(result, *, minimum_motion, checked_joint_count=5):
    """Reject a CEM block when a checked commanded axis did not actually respond.

    The regular hardware completion tolerance is intentionally broad enough for
    an established normal-follow hold.  It cannot by itself certify that a
    newly planned CEM joint displacement happened.  This check is therefore
    directional rather than an invented servo-tuning rule: each checked axis whose
    final CEM command differs by more than ``minimum_motion`` must have an
    exposure-time to endpoint Present_Position delta of that same sign and
    magnitude.  Five joints remains the compatibility default; callers that
    command J6 can explicitly require all six.  It does not require an exact
    commanded endpoint.
    """
    if not np.isfinite(minimum_motion) or minimum_motion <= 0.0:
        raise ValueError('minimum CEM motion must be finite and positive')
    if type(checked_joint_count) is not int or checked_joint_count not in (5, 6):
        raise ValueError('checked_joint_count must be exactly 5 or 6')
    try:
        source = _finite(result.get('source_q_position'), (6,), 'CEM source q')
        commands = np.asarray(result.get('commands_position'), dtype=np.float64)
        measured = np.asarray(result.get('measured_q_path'), dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        raise ValueError('CEM execution lacks actual q readback') from None
    if commands.shape != (5, 6) or not np.isfinite(commands).all():
        raise ValueError('CEM execution lacks a finite five-command path')
    if measured.ndim != 2 or measured.shape[1] != 6 or len(measured) < 2 or not np.isfinite(measured).all():
        raise ValueError('CEM execution lacks a finite actual q path')
    endpoint = _actual_response_endpoint(result, measured)
    count = checked_joint_count
    actual = endpoint[:count] - source[:count]
    commanded = commands[:, :count] - source[:count]
    response = np.vstack((measured[:, :count], endpoint[:count])) - source[:count]
    driven = np.max(np.abs(commanded), axis=0) > float(minimum_motion)
    peak = commanded[np.argmax(np.abs(commanded), axis=0), np.arange(count)]
    achieved = np.max(response * np.sign(peak), axis=0)
    blocked = driven & (achieved <= float(minimum_motion))
    if np.any(blocked):
        diagnostic = dict(
            position_unit='lerobot_calibrated_position',
            minimum_motion=float(minimum_motion),
            joint_index_base=0, checked_joint_count=count,
            commanded_peak_delta=peak.tolist(),
            achieved_directional_motion=achieved.tolist(),
            driven_axes=driven.tolist(),
            blocked_joint_indices=np.flatnonzero(blocked).tolist(),
            actual_endpoint_delta=actual.tolist(),
            response_samples='measured path plus validated actual response endpoint',
            control_influence=False, predictions_used_as_evidence=False)
        if count == 5:
            # Compatibility aliases for existing five-arm-joint consumers.
            diagnostic.update(arm_joint_index_base=0,checked_arm_joint_count=5,
                driven_arm_axes=diagnostic['driven_axes'],
                blocked_arm_joint_indices=diagnostic['blocked_joint_indices'])
        raise CEMActualResponseError(diagnostic)
    return actual.copy()


def short_window_keeps_actual_response(result, error):
    """A 0.4 s chunk records the real readback instead of failing the request.

    Heavy joints can be commanded past the 0.25 response floor and still move
    less than that before the short follow ends. That readback is the LeWM
    sample. It is not a tracking success and it does not extend the window.
    """
    follow = result.get('normal_follow') if isinstance(result, dict) else None
    if not isinstance(follow, dict):
        return False
    return (isinstance(error, CEMActualResponseError)
            and follow.get('budget_s') == 0.4
            and result.get('completed') is True
            and result.get('policy_commands_returned') == 5)
