"""Readback-addressed Column display prediction; never factual visual memory.

The deployed forecast predicts joint positions, not FCPG pixel coordinates.
Its completed transition can transport a previously observed RGB F/G pair
through the same uncertain Column camera matrices and full-joint FK.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .v21_column_pose_prior import ColumnPosePrior
from .v21_temporal_contract import READBACKS_PER_COMPLETED_BLOCK


SOURCE = 'previous-column-rgb-plus-lewm-q-full-fk-displacement'
METHOD_LABEL = 'previous actual RGB + LeWM/FK candidate prior'
MAX_SCORE_BONUS = .35


def _array(value, shape):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('temporal evidence has invalid numeric shape')
    return result


def _address(state):
    session = getattr(state, 'progress_session_id', None)
    value = getattr(getattr(state, 'memory', None), 'current_readback_index', None)
    index = _array(value, (1,))[0]
    if not isinstance(session, str) or not session or index < 0 or index != int(index):
        raise ValueError('temporal evidence requires a session and completed readback index')
    return session, int(index)


@dataclass(frozen=True)
class ColumnTemporalPrior:
    record: dict

    def as_dict(self):
        return deepcopy(self.record)

    def as_suggestion(self, *, reason):
        """Explicit display prediction when current visual detection is absent."""
        return dict(available=False, observed=False, display_only=True, control_input=False,
            fm_px=None, m_px=None, suggested_fm_px=deepcopy(self.record['predicted_pair_px']),
            suggested_m_px=deepcopy(self.record['predicted_midpoint_px']),
            suggested_pair_bounds_px=deepcopy(self.record.get('predicted_pair_bounds_px')),
            source=self.record['source'], method_label=self.record.get('method_label'),
            progress_session_id=self.record.get('progress_session_id'),
            target_readback_index=self.record.get('target_readback_index'),
            coordinate_space='letterbox224', reason=reason,
            actual_visual_memory_write=False, fcpg_pixel_prediction_available=False)

    def score_pair(self, pair):
        candidate = _array(pair, (2, 2))
        center = candidate.mean(0)
        residual = center - np.asarray(self.record['predicted_midpoint_px'])
        covariance = np.asarray(self.record['midpoint_cov_px2'])
        center_distance = max(0., float(residual @ np.linalg.solve(covariance, residual)))
        covariance = np.asarray(self.record['pair_cov_px2'])
        associations = self.record['source_pair_associations']
        maximum_probability = max(row['probability'] for row in associations)
        best = None
        # Locator candidates have no proven F/G order yet. Retain both current
        # assignments even if the previous RGB pair was confidently ordered.
        for index, association in enumerate(associations):
            expected = np.asarray(association['predicted_pair_px'])
            penalty = -2. * np.log(max(association['probability'], 1e-300) / maximum_probability)
            for swapped in (False, True):
                residual = ((candidate[::-1] if swapped else candidate) - expected).reshape(4)
                distance = max(0., float(residual @ np.linalg.solve(covariance, residual)))
                ranked_distance = distance + penalty
                if best is None or ranked_distance < best[0]:
                    best = ranked_distance, distance, index, swapped
        return dict(midpoint_mahalanobis_sq=center_distance,
                    pair_mahalanobis_sq=best[1], source_association_index=best[2],
                    candidate_pair_swapped=best[3],
                    score_bonus=float(MAX_SCORE_BONUS * np.exp(-.5 * best[0])),
                    maximum_score_bonus=MAX_SCORE_BONUS,
                    source=SOURCE, endpoint_coordinates_used=False)


class ColumnTemporalTracker:
    """Ephemeral publication state keyed only by completed action samples.

    Pending/replanned forecasts are never read. Duplicate publications at the
    same index use an immutable prior from the original completed transition.
    """

    def __init__(self):
        self.session = None
        self.observations = {}
        self._receipt = self._forecast = self._prior = None
        self._dispatch_address = None
        self.diagnostic = dict(available=False, reason='no_completed_transition',
            method_label=METHOD_LABEL, fcpg_pixel_prediction_available=False)

    def _unavailable(self, reason, **details):
        self._prior = None
        self.diagnostic = dict(available=False, reason=reason,
            method_label=METHOD_LABEL, fcpg_pixel_prediction_available=False, **details)
        return None

    def _available(self):
        self.diagnostic = dict(available=True, reason=None,
            method_label=METHOD_LABEL, fcpg_pixel_prediction_available=False)
        return self._prior

    def clear(self):
        self.observations.clear()
        self._receipt = self._forecast = self._prior = None
        self._dispatch_address = None
        self._unavailable('observation_history_cleared')

    def _session(self, state):
        address = _address(state)
        if address[0] != self.session:
            self.clear()
            self.session = address[0]
        return address

    def record_observation(self, state, tip):
        try:
            _, index = self._session(state)
            if (not isinstance(tip, dict) or tip.get('available') is not True
                    or tip.get('source') != 'column-rgb-jaw-contour'
                    or tip.get('fk_based') is True):
                return
            pair = _array(tip.get('fm_px'), (2, 2)).copy()
            q = _array(tip.get('q_at_image'), (6,)).copy()
            if (pair < [0., 28.]).any() or (pair >= [224., 196.]).any():
                return
            self.observations[index] = (pair, q)
            self.observations = {i: row for i, row in self.observations.items()
                                 if index - READBACKS_PER_COMPLETED_BLOCK <= i <= index}
        except (ValueError, TypeError, AttributeError):
            return

    def for_current(self, state, bridge, kinematics, fit):
        """Only an executed forecast with complete actual readbacks may predict i."""
        try:
            session, index = self._session(state)
            runtime = getattr(bridge, 'runtime', None)
            receipt = getattr(bridge, 'last_attempt', None)
            if (getattr(runtime, 'requires_home_after_incomplete_execution', False)
                    or (receipt is not None and (receipt.get('completed') is not True
                        or receipt.get('error') or receipt.get('response_verified') is False))):
                self.clear()
                return self._unavailable('completed_response_not_verified',
                    requires_home=bool(getattr(runtime, 'requires_home_after_incomplete_execution', False)),
                    response_error=(receipt or {}).get('response_error'),
                    execution_error=(receipt or {}).get('error'))
            if index == 0 or not isinstance(receipt, dict):
                return self._unavailable('no_completed_transition')
            source = getattr(bridge, 'last_source_state', None)
            forecast = getattr(bridge, 'last_forecast', None)
            source_session, source_index = _address(source)
            # i counts individual policy readbacks. This service publishes
            # one endpoint image after all five commands, so consecutive
            # camera endpoints are i -> i+5 (or chunk k -> k+1), not i -> i+1.
            target_index = source_index + READBACKS_PER_COMPLETED_BLOCK
            if (source_session != session or target_index != index
                    or _array(state.feedback_readback_index, (1,))[0] != index
                    or not bool(_array(state.feedback_available, (1,))[0])
                    or receipt.get('policy_commands_attempted') != 5
                    or receipt.get('policy_commands_returned') != 5):
                return self._unavailable('completed_readback_address_mismatch')
            source_q = _array(source.q, (1, 6))[0]
            path = _array(receipt.get('measured_q_path'), (6, 6))
            predicted = _array(forecast.response.predicted_q_path, (1, 6, 6))[0]
            commands = _array(forecast.trace.positions, (1, 5, 6))[0]
            if (not np.allclose(source_q, _array(forecast.source_q, (1, 6))[0], atol=1e-4, rtol=0)
                    or not np.allclose(source_q, path[0], atol=1e-4, rtol=0)
                    or not np.allclose(commands, _array(receipt.get('commands_position'), (5, 6)), atol=1e-4, rtol=0)):
                return self._unavailable('executed_commands_or_source_mismatch')
            # This reuses a prediction only at its one addressed endpoint. A
            # new plan at i cannot replace the forecast that actually ran.
            dispatch_address = (session, source_index, target_index)
            if receipt is self._receipt and forecast is self._forecast:
                if self._dispatch_address != dispatch_address:
                    return self._unavailable('cached_dispatch_address_mismatch')
                if self._prior is not None:
                    return self._available()
            if source_index not in self.observations:
                return self._unavailable('source_visual_pair_unavailable',
                    source_readback_index=source_index)
            pair, image_q = self.observations[source_index]
            prior = ColumnPosePrior.from_kinematics(kinematics, image_q, fit, proposal_pair=pair)
            target = ColumnPosePrior.from_kinematics(kinematics, predicted[-1], fit)
            association = prior.score_pair(pair)
            ordered_pair = np.asarray(association['ordered_pair'])
            probability = float(association['identity_probability'])
            delta = target.world_pair_xyz - prior.world_pair_xyz
            # The same camera matrix projects both endpoint displacements.
            # Translation cancels; RGB supplies their actual source locations.
            displacements = np.einsum('nkj,pj->npk', prior.camera_matrices[:, :, :3], delta)
            hypotheses = ordered_pair[None] + displacements
            mean_pair = np.einsum('n,npk->pk', prior.hypothesis_weights, hypotheses)
            residual = (hypotheses - mean_pair).reshape(len(hypotheses), 4)
            # Explicit tolerance for contour noise and uncalibrated response;
            # no metric confidence or empirical coverage is claimed.
            pair_covariance = ((residual.T * prior.hypothesis_weights) @ residual
                + np.kron(np.ones((2, 2)), np.eye(2) * 8.**2) + np.eye(4) * 3.**2)
            center_projection = np.tile(np.eye(2) * .5, (1, 2))
            covariance = center_projection @ pair_covariance @ center_projection.T
            center = mean_pair.mean(0)
            extent = 2. * np.sqrt(np.diag(covariance))
            pair_extent = 2. * np.sqrt(np.diag(pair_covariance)).reshape(2, 2)
            source_associations = [dict(probability=probability,
                source_pair_swapped=bool(association['swapped']), predicted_pair_px=mean_pair.tolist())]
            if probability < .95:
                # Ambiguous source identity is a second association mode, not
                # an average that would collapse two separately visible tips.
                alternate = ordered_pair[::-1] + np.einsum('n,npk->pk', prior.hypothesis_weights, displacements)
                source_associations.append(dict(probability=1. - probability,
                    source_pair_swapped=not bool(association['swapped']), predicted_pair_px=alternate.tolist()))
            record = dict(available=True, source=SOURCE, inferred=True, observed=False,
                method_label=METHOD_LABEL,
                progress_session_id=session, source_readback_index=source_index,
                target_readback_index=target_index, current_readback_index=index,
                address_unit='completed_policy_readback_sample_index',
                completed_actual_readbacks=READBACKS_PER_COMPLETED_BLOCK,
                readbacks_per_completed_chunk=READBACKS_PER_COMPLETED_BLOCK,
                source_chunk_index=(source_index // READBACKS_PER_COMPLETED_BLOCK
                                    if source_index % READBACKS_PER_COMPLETED_BLOCK == 0 else None),
                target_chunk_index=(target_index // READBACKS_PER_COMPLETED_BLOCK
                                    if target_index % READBACKS_PER_COMPLETED_BLOCK == 0 else None),
                forecast_q_path_row=READBACKS_PER_COMPLETED_BLOCK,
                target_observation_kind='completed_five_command_chunk_endpoint',
                source_rgb_pair_px=pair.tolist(),
                source_q_at_image=image_q.tolist(), forecast_source_q=source_q.tolist(),
                predicted_target_q=predicted[-1].tolist(),
                predicted_pair_px=mean_pair.tolist(), pair_cov_px2=pair_covariance.tolist(),
                predicted_pair_bounds_px=np.stack((mean_pair - pair_extent, mean_pair + pair_extent), axis=1).tolist(),
                pair_covariance_order=['F.x', 'F.y', 'G.x', 'G.y'],
                source_pair_associations=source_associations,
                source_identity_probability=probability,
                candidate_pair_association='best direct or swapped; current RGB order is unverified',
                predicted_midpoint_px=center.tolist(), midpoint_cov_px2=covariance.tolist(),
                midpoint_bounds_px=[(center - extent).tolist(), (center + extent).tolist()],
                prediction_kind='RGB F/G pair transported by LeWM q forecast and full FK endpoint displacements',
                shared_center_noise_std_px=8., endpoint_noise_std_px=3.,
                fcpg_pixel_prediction_available=False, full_joint_fk=True,
                shared_camera_transform=True, coordinate_space='letterbox224',
                uncertainty_status='explicit uncalibrated assumptions',
                maximum_score_bonus=MAX_SCORE_BONUS,
                # The forecast never becomes a tip observation or a direct
                # control target. It can still change which current-image
                # proposal is inspected, so describing the prior itself as
                # display-only would hide its indirect effect on enabled
                # engineering consumers of the resulting RGB observation.
                display_only=False, direct_control_input=False,
                proposal_ranking_input=True,
                may_influence_control_via_current_rgb_observation=True,
                actual_visual_memory_write=False,
                predicted_endpoints_published_as_observation=False,
                endpoint_coordinates_used=False, current_rgb_and_pose_gates_required=True)
            self._prior = ColumnTemporalPrior(record)
            self._receipt, self._forecast = receipt, forecast
            self._dispatch_address = dispatch_address
            return self._available()
        except (ValueError, TypeError, AttributeError, KeyError, np.linalg.LinAlgError) as exc:
            return self._unavailable('invalid_temporal_evidence', error=f'{type(exc).__name__}: {exc}')
