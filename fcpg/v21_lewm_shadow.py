"""Read-only, paired LeWM predictions on one factual command transaction.

This module has no runtime, service or hardware callbacks. Predictions never
advance history, select commands, or certify physical accuracy. Call ``plan``
outside motor deadlines and ``complete`` only after the existing HOLD path.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

import numpy as np
import torch

from .prod_v20_policy import ActualHistory, _validate_history
from .v21_models import CommandTrace


def _json_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _tensor_record(value):
    return dict(dtype=str(value.dtype), shape=list(value.shape), values=value.detach().cpu().tolist())


def _clone_history(history, device=None):
    return ActualHistory(**{
        field.name: getattr(history, field.name).detach().to(device=device).clone()
        for field in fields(ActualHistory)
    })


def _packet(q, history, trace, queries):
    return dict(
        q=_tensor_record(q),
        history={field.name: _tensor_record(getattr(history, field.name)) for field in fields(ActualHistory)},
        trace=dict(positions=_tensor_record(trace.positions), offsets=_tensor_record(trace.offsets),
                   valid=_tensor_record(trace.valid), complete=trace.complete),
        query_offsets=_tensor_record(queries),
    )


def _validate_inputs(q, history, trace, queries):
    if q.shape != (1, 6) or not q.is_floating_point() or not torch.isfinite(q).all():
        raise ValueError('shadow source q must be finite [1,6]')
    _validate_history(history, 1)
    trace.validate(1)
    if trace.positions.shape != (1, 5, 6) or not trace.valid.all():
        raise ValueError('shadow comparison requires exactly five commands')
    if (queries.shape != (1, 5) or queries.dtype != torch.float64
            or not torch.isfinite(queries).all() or (queries <= 0).any()
            or (queries.diff(dim=1) <= 0).any()):
        raise ValueError('shadow readback offsets must be five ordered float64 times')
    if (trace.offsets[trace.valid] >= queries[:, -1:]).any():
        raise ValueError('shadow commands must precede the final readback')


def _array(value, shape, label):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f'{label} must be finite {shape}')
    return out


def _metrics(predicted, measured):
    """Descriptive calibrated-position errors; q0 is never an evaluated point."""
    residual = measured[1:] - predicted[1:]
    predicted_delta, actual_delta = predicted[-1] - measured[0], measured[-1] - measured[0]
    ratios = [None if a == 0 else float(p / a) for p, a in zip(predicted_delta, actual_delta)]
    return dict(
        position_unit='lerobot_calibrated_position', samples=5,
        error_sign='actual_minus_prediction',
        per_joint_mae=np.abs(residual).mean(0).tolist(),
        per_joint_rmse=np.sqrt(np.square(residual).mean(0)).tolist(),
        per_joint_max_abs=np.abs(residual).max(0).tolist(),
        per_joint_bias=residual.mean(0).tolist(),
        per_joint_terminal_error=residual[-1].tolist(),
        predicted_terminal_delta=predicted_delta.tolist(),
        actual_terminal_delta=actual_delta.tolist(),
        terminal_delta_ratio_predicted_over_actual=ratios,
        terminal_direction_agreement=[None if a == 0 or p == 0 else bool(np.sign(a) == np.sign(p))
                                      for p, a in zip(predicted_delta, actual_delta)],
        per_readback_mae=np.abs(residual).mean(1).tolist(),
        residual_path=residual.tolist(),
        mae=float(np.abs(residual).mean()), rmse=float(np.sqrt(np.square(residual).mean())),
        max_abs=float(np.abs(residual).max()),
    )


class LeWMShadowEvaluator:
    """Keep the incumbent and candidate predictors separate from the controller."""

    def __init__(self, parent_model, candidates, *, controller_candidate_sha256=None):
        self.parent_model = parent_model
        self.candidates = tuple(candidates)
        self._controller_identity = None
        if controller_candidate_sha256 is not None:
            if (type(controller_candidate_sha256) is not str or
                    re.fullmatch(r'[0-9a-f]{64}', controller_candidate_sha256) is None):
                raise ValueError('controller candidate SHA256 must be a lowercase SHA256 digest')
            matches = [candidate for candidate in self.candidates
                       if candidate.candidate_sha256 == controller_candidate_sha256]
            if len(matches) != 1:
                raise ValueError('controller candidate SHA256 must uniquely match a shadow candidate')
            if matches[0].method != 'lewm_mean':
                raise ValueError('controller candidate must be an adapted lewm_mean predictor')
        self.evaluator_id = uuid.uuid4().hex
        self._models = [('parent', parent_model, dict(role='parent'))]
        parent_shas = {candidate.parent_sha256 for candidate in self.candidates}
        if len(parent_shas) > 1:
            raise ValueError('shadow candidates name different parent checkpoints')
        if parent_shas:
            self._models[0][2]['parent_sha256'] = next(iter(parent_shas))
        candidate_shas = [candidate.candidate_sha256 for candidate in self.candidates]
        if len(set(candidate_shas)) != len(candidate_shas):
            raise ValueError('the same shadow candidate SHA256 was supplied more than once')
        base_names = [f'{candidate.method}-seed{candidate.seed}-{candidate.held_out_run}'
                      for candidate in self.candidates]
        base_counts = Counter(base_names)
        for candidate, base_name in zip(self.candidates, base_names):
            metadata = {name: getattr(candidate, name) for name in (
                'parent_sha256', 'candidate_sha256', 'method', 'seed', 'held_out_run',
                'parameter_names', 'requires_independent_validation')}
            metadata['parameter_names'] = list(metadata['parameter_names'])
            # Preserve established names unless two distinct artifacts share
            # the same method/seed/held-out label. Full digests avoid truncation
            # collisions; the final identifier check also catches alias clashes.
            name = (base_name if base_counts[base_name] == 1 else
                    f'{base_name}-{candidate.candidate_sha256}')
            self._models.append((name, candidate.model, metadata))
        names = [name for name, _, _ in self._models]
        if len(set(names)) != len(names):
            raise ValueError('shadow candidate identifiers must be unique')
        if any(model.training for _, model, _ in self._models):
            raise ValueError('shadow predictors must already be in eval mode')
        self._model_identities = {name: dict(metadata, model_state_sha256=self._state_sha(model))
                                  for name, model, metadata in self._models}
        if controller_candidate_sha256 is not None:
            for identity in self._model_identities.values():
                identity['used_for_control'] = identity.get('candidate_sha256') == controller_candidate_sha256
                if identity['used_for_control']:
                    self._controller_identity = dict(
                        role='adapted_lewm_mean', candidate_sha256=controller_candidate_sha256,
                        parent_sha256=identity['parent_sha256'],
                        model_state_sha256=identity['model_state_sha256'],
                        control_influence=True, experimental=True, uncertainty_calibrated=False)

    @property
    def controller_identity(self):
        """An owned metadata copy; the evaluator never installs a controller."""
        return None if self._controller_identity is None else dict(self._controller_identity)

    @staticmethod
    def _state_sha(model):
        # Construction is outside the motor loop. Include normalization buffers.
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _base(self, kind):
        record = dict(kind=kind, schema=1, evaluator_id=self.evaluator_id,
                      control_influence=False, physical_acceptance_validated=False,
                      physical_accuracy_accepted=None)
        if self._controller_identity is not None:
            record['controller_identity'] = self.controller_identity
        return record

    @staticmethod
    def _source(source_state, forecast):
        q = source_state.q.detach().clone()
        history = _clone_history(source_state.history)
        trace = CommandTrace(forecast.trace.positions.detach().clone(),
                             forecast.trace.offsets.detach().clone(),
                             forecast.trace.valid.detach().clone(), forecast.trace.source,
                             forecast.trace.complete)
        queries = forecast.query_offsets.detach().clone()
        if not torch.equal(q, forecast.source_q.to(q)):
            raise ValueError('shadow source q differs from the dispatch forecast')
        _validate_inputs(q, history, trace, queries)
        return q, history, trace, queries

    @torch.inference_mode()
    def _predict(self, q, history, trace, queries):
        records = {}
        for name, model, _ in self._models:
            entry = dict(self._model_identities[name], valid=False)
            try:
                if model.training:
                    raise ValueError('shadow predictor left eval mode')
                if self._state_sha(model) != entry['model_state_sha256']:
                    raise ValueError('shadow predictor weights changed after evaluator setup')
                parameter = next(model.parameters(), None)
                device = q.device if parameter is None else parameter.device
                mq = q.detach().to(device).clone()
                mh = _clone_history(history, device)
                mt = CommandTrace(trace.positions.detach().to(device).clone(),
                                  trace.offsets.detach().to(device).clone(),
                                  trace.valid.detach().to(device).clone(), trace.source, trace.complete)
                qt = queries.detach().to(device).clone()
                before = _json_sha(_packet(mq, mh, mt, qt))
                cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
                # Even a misconfigured stochastic candidate cannot consume the
                # controller's random stream. No seed is set or model is trained.
                with torch.random.fork_rng(devices=cuda_devices):
                    response = model(mq, mt, mh, qt)
                if before != _json_sha(_packet(mq, mh, mt, qt)):
                    raise ValueError('shadow predictor mutated its owned input copy')
                pred, sigma = response.predicted_q_path, response.q_uncertainty
                if (pred.shape != (1, 6, 6) or not pred.is_floating_point()
                        or not torch.isfinite(pred).all() or not torch.equal(pred[:, 0].to(mq), mq)):
                    raise ValueError('shadow prediction is malformed or not source anchored')
                if (sigma.shape != (1, 5, 6) or not sigma.is_floating_point()
                        or not torch.isfinite(sigma).all() or (sigma <= 0).any()):
                    raise ValueError('shadow uncertainty is malformed; calibration is not asserted')
                entry.update(valid=True, predicted_q_path=pred[0].detach().cpu().double().tolist(),
                             q_uncertainty=sigma[0].detach().cpu().double().tolist(),
                             uncertainty_calibrated=False)
            except Exception as exc:
                entry['error'] = f'{type(exc).__name__}: {exc}'
            records[name] = entry
        return records

    def plan(self, source_state, forecast):
        q, history, trace, queries = self._source(source_state, forecast)
        packet = _packet(q, history, trace, queries)
        predictions = self._predict(q, history, trace, queries)
        return dict(self._base('lewm-shadow-plan'), input_sha256=_json_sha(packet),
                    source_q_position=q[0].cpu().double().tolist(),
                    history_sha256=_json_sha(packet['history']),
                    source_time_s=float(forecast.source_time.item()),
                    command_offsets_s=trace.offsets[0].cpu().tolist(),
                    readback_offsets_s=queries[0].cpu().tolist(),
                    clock_semantics='nominal_plan_offsets_from_source',
                    parent=predictions.pop('parent'), candidates=predictions)

    def _receipt_match(self, receipt, input_sha, first_write_s, returned_commands_match):
        reasons = []
        result = dict(matched=False, semantics='prewrite_saved_nominal_prediction_only', reasons=reasons)
        if not receipt:
            reasons.append('no_persisted_plan_receipt')
            return result
        try:
            record = receipt['record']
            raw = Path(receipt['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
                reasons.append('receipt_file_sha256_mismatch')
            if json.loads(raw) != record:
                reasons.append('receipt_record_differs_from_saved_file')
            if record.get('evaluator_id') != self.evaluator_id:
                reasons.append('receipt_from_different_evaluator_instance')
            if record.get('input_sha256') != input_sha:
                reasons.append('plan_inputs_differ_from_dispatch_inputs')
            if (self._controller_identity is not None and
                    _json_sha(record.get('controller_identity')) != _json_sha(self._controller_identity)):
                reasons.append('receipt_controller_identity_differs')
            persisted = float(receipt['persisted_at_s'])
            if not math.isfinite(persisted) or not 0 <= persisted < first_write_s:
                reasons.append('receipt_not_persisted_before_first_write')
            if not returned_commands_match:
                reasons.append('returned_commands_differ_from_forecast')
            saved_models = dict(record.get('candidates', {}), parent=record.get('parent', {}))
            if set(saved_models) != set(self._model_identities):
                reasons.append('receipt_predictor_set_differs')
            for name, identity in self._model_identities.items():
                if saved_models.get(name, {}).get('model_state_sha256') != identity['model_state_sha256']:
                    reasons.append(f'receipt_predictor_identity_differs:{name}')
                if (self._controller_identity is not None and
                        saved_models.get(name, {}).get('used_for_control') is not identity['used_for_control']):
                    reasons.append(f'receipt_control_routing_differs:{name}')
                if saved_models.get(name, {}).get('valid') is not True:
                    reasons.append(f'receipt_prediction_invalid:{name}')
            result.update(path=str(receipt['path']), sha256=receipt['sha256'], persisted_at_s=persisted,
                          first_write_s=first_write_s)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            reasons.append(f'invalid_receipt:{type(exc).__name__}:{exc}')
        result['matched'] = not reasons
        return result

    def _clock_scores(self, q, history, trace, queries, measured, *, origin, label, actual_alignment):
        _validate_inputs(q, history, trace, queries)
        predictions = self._predict(q, history, trace, queries)
        for record in predictions.values():
            if record['valid']:
                record['metrics'] = _metrics(np.asarray(record['predicted_q_path']), measured)
        source = measured[:1]
        hold = np.repeat(source, 6, axis=0)
        active = np.concatenate((source, trace.active_at(queries, q)[0].cpu().double().numpy()))
        parent = predictions.pop('parent')
        for record in predictions.values():
            if parent['valid'] and record['valid']:
                record['paired_mae_change_from_parent'] = record['metrics']['mae'] - parent['metrics']['mae']
                record['per_joint_mae_change_from_parent'] = (
                    np.asarray(record['metrics']['per_joint_mae']) - parent['metrics']['per_joint_mae']).tolist()
        return dict(scored=True, clock_origin_s=origin, clock_semantics=label,
                    computation='post_execution_from_frozen_source_inputs',
                    actual_clock_aligned=actual_alignment,
                    input_sha256=_json_sha(_packet(q, history, trace, queries)),
                    command_offsets_s=trace.offsets[0].cpu().tolist(),
                    readback_offsets_s=queries[0].cpu().tolist(),
                    parent=parent, candidates=predictions,
                    baselines=dict(hold_last=dict(predicted_q_path=hold.tolist(), metrics=_metrics(hold, measured)),
                                   active_command=dict(predicted_q_path=active.tolist(), metrics=_metrics(active, measured))))

    @staticmethod
    def _prospective_scores(receipt, measured, replay):
        saved = receipt['record']
        predictions = dict(saved['candidates'], parent=saved['parent'])
        scored = {}
        for name, prediction in predictions.items():
            path = _array(prediction.get('predicted_q_path'), (6, 6), 'saved prospective prediction')
            if prediction.get('valid') is not True or not np.array_equal(path[0], measured[0]):
                raise ValueError('saved prospective prediction is invalid or not source anchored')
            scored[name] = dict(prediction, metrics=_metrics(path, measured))
            repeated = replay['parent'] if name == 'parent' else replay['candidates'][name]
            if repeated['valid']:
                scored[name]['nominal_replay_max_difference'] = float(
                    np.abs(path - np.asarray(repeated['predicted_q_path'])).max())
        parent = scored.pop('parent')
        for candidate in scored.values():
            candidate['paired_mae_change_from_parent'] = candidate['metrics']['mae'] - parent['metrics']['mae']
            candidate['per_joint_mae_change_from_parent'] = (
                np.asarray(candidate['metrics']['per_joint_mae']) - parent['metrics']['per_joint_mae']).tolist()
        return dict(scored=True, computation='saved_before_first_motor_write',
                    clock_semantics='saved_nominal_plan_offsets; actual_sample_times_may_differ',
                    actual_clock_aligned=False, parent=parent, candidates=scored,
                    baselines=replay['baselines'])

    def complete(self, source_state, forecast, result, plan_receipt=None):
        report = dict(self._base('lewm-shadow-comparison'), scored=False,
                      prospective_match=dict(matched=False, reasons=['not_assessed']),
                      prospective_predictions=dict(scored=False, reason='no_matched_prewrite_receipt'),
                      clock_evaluations={})
        if (result.get('completed') is not True or result.get('policy_commands_attempted') != 5
                or result.get('policy_commands_returned') != 5):
            report.update(reason='incomplete_five_command_transaction',
                          policy_commands_attempted=result.get('policy_commands_attempted'),
                          policy_commands_returned=result.get('policy_commands_returned'),
                          measured_samples=len(result.get('measured_q_path', [])))
            return report
        if not result.get('protective_hold_after_q5'):
            report['reason'] = 'completed_transaction_hold_receipt_missing'
            return report
        try:
            q, history, nominal_trace, nominal_queries = self._source(source_state, forecast)
            measured = _array(result.get('measured_q_path'), (6, 6), 'actual measured path')
            commands = _array(result.get('commands_position'), (5, 6), 'actual returned commands')
            read_times = _array(result.get('readback_times_s'), (6,), 'actual readback times')
            write_times = _array(result.get('command_times_s'), (5,), 'actual command times')
            source_q = _array(result.get('source_q_position'), (6,), 'recorded source q')
            source = q[0].cpu().double().numpy()
            if not np.array_equal(measured[0], source) or not np.array_equal(source_q, source):
                raise ValueError('recorded source or measured q0 differs from frozen dispatch source')
            epoch = float(result['source_time_s'])
            if (not math.isfinite(epoch) or epoch != float(forecast.source_time.item())
                    or epoch != float(source_state.time.item()) or read_times[0] != epoch):
                raise ValueError('dispatch source time differs from forecast or recorded q0 convention')
            if (np.any(np.diff(read_times) <= 0) or np.any(np.diff(write_times) <= 0)
                    or write_times[0] < epoch or np.any(write_times >= read_times[1:])
                    or np.any(read_times[1:-1] > write_times[1:])):
                raise ValueError('actual clocks are unordered or commands lie outside the transaction')
            returned_match = np.array_equal(commands, nominal_trace.positions[0].cpu().double().numpy())
            nominal_sha = _json_sha(_packet(q, history, nominal_trace, nominal_queries))
            report['prospective_match'] = self._receipt_match(plan_receipt, nominal_sha, float(write_times[0]), returned_match)
            report.update(source_time_s=epoch, source_q_position=source.tolist(),
                          actual_commands_match_nominal=returned_match,
                          legacy_response_verified=result.get('response_verified'),
                          legacy_response_error=result.get('response_error'),
                          legacy_response_evaluation='recorded' if 'response_verified' in result else 'not_yet_evaluated',
                          legacy_response_is_prediction_accuracy=False)
            groups = report['clock_evaluations']
            # The dispatch forecast may belong to the adapted controller.
            # Recompute the original parent independently; forecast.response
            # must never be relabelled as the parent baseline.
            groups['nominal_dispatch'] = self._clock_scores(
                q, history, nominal_trace, nominal_queries, measured, origin=epoch,
                label='planned_command_and_query_offsets; actual_sample_times_may_differ', actual_alignment=False)
            if report['prospective_match']['matched']:
                try:
                    report['prospective_predictions'] = self._prospective_scores(
                        plan_receipt, measured, groups['nominal_dispatch'])
                except (ValueError, TypeError, KeyError) as exc:
                    report['prospective_match']['matched'] = False
                    report['prospective_match']['reasons'].append(f'invalid_saved_prediction:{exc}')
            origins = [('actual_dispatch_epoch', epoch, 'actual_times_minus_dispatch_epoch')]
            read_origin = result.get('source_q_readback_time_s')
            if read_origin is None:
                groups['actual_source_readback'] = dict(scored=False, reason='source_q_readback_time_missing')
            else:
                read_origin = float(read_origin)
                if not math.isfinite(read_origin) or read_origin > epoch:
                    raise ValueError('source q readback time must precede or equal dispatch epoch')
                report['source_q_age_at_dispatch_s'] = epoch - read_origin
                origins.append(('actual_source_readback', read_origin, 'actual_times_minus_factual_source_q_readback_time'))
            for name, origin, label in origins:
                trace = CommandTrace(torch.tensor(commands, dtype=q.dtype, device=q.device)[None],
                                     torch.tensor(write_times - origin, dtype=torch.float64, device=q.device)[None],
                                     torch.ones((1, 5), dtype=torch.bool, device=q.device), 'recorded', True)
                queries = torch.tensor(read_times[1:] - origin, dtype=torch.float64, device=q.device)[None]
                groups[name] = self._clock_scores(q, history, trace, queries, measured,
                                                  origin=origin, label=label, actual_alignment=True)
            report.update(scored=True, reason='descriptive_paired_comparison_no_physical_acceptance_threshold')
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            report.update(scored=False, reason=f'invalid_alignment:{type(exc).__name__}:{exc}')
        return report
