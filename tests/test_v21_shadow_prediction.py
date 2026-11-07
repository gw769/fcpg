from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.prod_v20_policy import empty_history
from fcpg.v21_core import Forecast
from fcpg.v21_lewm_shadow import LeWMShadowEvaluator
from fcpg.v21_models import ARMOutput, CommandTrace


class Predictor(torch.nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.register_buffer('gain', torch.tensor(gain))
        self.calls = []

    def forward(self, q, trace, history, queries):
        self.calls.append((q.clone(), trace, history, queries.clone()))
        active = trace.active_at(queries, q)
        path = q[:, None] + self.gain * (active - q[:, None]) + queries[:, :, None].to(q) * .01
        path = path + history.readbacks_position.sum() * .001
        return ARMOutput(torch.cat((q[:, None], path), 1), torch.ones_like(path), path)


def fixture():
    parent, candidate = Predictor(.5).eval(), Predictor(.75).eval()
    adapted = SimpleNamespace(model=candidate, parent_sha256='parent-sha', candidate_sha256='candidate-sha',
                              method='response-head', seed=21, held_out_run='fresh', parameter_names=('gain',),
                              requires_independent_validation=True)
    evaluator = LeWMShadowEvaluator(parent, [adapted])
    q = torch.zeros(1, 6)
    state = SimpleNamespace(q=q, time=torch.tensor([10.], dtype=torch.float64), history=empty_history(1))
    commands = torch.arange(1., 6.)[None, :, None].expand(1, 5, 6).clone()
    trace = CommandTrace.policy(commands, torch.full((1, 5), .1))
    queries = torch.tensor([[.1, .2, .3, .4, .5]], dtype=torch.float64)
    forecast = Forecast(q, state.time, trace, queries, parent(q, trace, state.history, queries))
    measured = np.concatenate((np.zeros((1, 6)), commands[0].numpy() * .8))
    result = dict(completed=True, policy_commands_attempted=5, policy_commands_returned=5,
                  protective_hold_after_q5=True, source_q_position=q[0].tolist(), source_time_s=10.,
                  source_q_readback_time_s=9.98, commands_position=commands[0].tolist(),
                  command_times_s=[10.002, 10.106, 10.207, 10.308, 10.409],
                  readback_times_s=[10., 10.105, 10.206, 10.307, 10.408, 10.511],
                  measured_q_path=measured.tolist(), response_verified=True)
    return evaluator, parent, candidate, state, forecast, result


def persisted(tmp_path, record, stamp=9.99):
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(record))
    return dict(record=record, path=str(path), persisted_at_s=stamp,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_prospective_receipt_matches_despite_absolute_source_clock_change(tmp_path):
    evaluator, _, _, state, forecast, result = fixture()
    earlier = replace(forecast, source_time=torch.tensor([9.5], dtype=torch.float64))
    receipt = persisted(tmp_path, evaluator.plan(state, earlier))
    report = evaluator.complete(state, forecast, result, receipt)
    assert report['scored'] and report['prospective_match']['matched']
    assert report['control_influence'] is False
    assert report['physical_acceptance_validated'] is False
    assert report['physical_accuracy_accepted'] is None
    prospective = report['prospective_predictions']
    assert prospective['scored'] and prospective['computation'] == 'saved_before_first_motor_write'
    assert prospective['parent']['predicted_q_path'] == receipt['record']['parent']['predicted_q_path']
    assert prospective['parent']['nominal_replay_max_difference'] == 0
    assert set(report['clock_evaluations']) == {'nominal_dispatch', 'actual_dispatch_epoch', 'actual_source_readback'}
    actual = report['clock_evaluations']['actual_dispatch_epoch']
    assert actual['actual_clock_aligned'] is True
    assert actual['parent']['metrics']['samples'] == 5
    assert len(actual['parent']['metrics']['per_joint_mae']) == 6
    np.testing.assert_allclose(actual['baselines']['active_command']['predicted_q_path'][1:], result['commands_position'])
    candidate = next(iter(actual['candidates'].values()))
    assert candidate['paired_mae_change_from_parent'] < 0
    assert report['source_q_age_at_dispatch_s'] == pytest.approx(.02)
    np.testing.assert_allclose(report['clock_evaluations']['actual_source_readback']['readback_offsets_s'],
                               [.125, .226, .327, .428, .531])


@pytest.mark.parametrize('damage', ['q', 'history', 'query', 'trace', 'returned', 'time', 'file'])
def test_receipt_cannot_claim_matching_changed_inputs_or_late_persistence(tmp_path, damage):
    evaluator, _, _, state, forecast, result = fixture()
    receipt = persisted(tmp_path, evaluator.plan(state, forecast))
    if damage == 'q':
        state.q = state.q + .01
        forecast = replace(forecast, source_q=state.q)
        result['source_q_position'] = state.q[0].tolist()
        result['measured_q_path'][0] = state.q[0].tolist()
    elif damage == 'history':
        state.history = replace(state.history, readbacks_position=state.history.readbacks_position + .01)
    elif damage == 'query':
        forecast = replace(forecast, query_offsets=forecast.query_offsets + .001)
    elif damage == 'trace':
        forecast = replace(forecast, trace=replace(forecast.trace, positions=forecast.trace.positions + .01))
    elif damage == 'returned':
        result['commands_position'][0][0] += 1e-6
    elif damage == 'time':
        receipt['persisted_at_s'] = result['command_times_s'][0]
    else:
        from pathlib import Path
        Path(receipt['path']).write_text('{}')
    report = evaluator.complete(state, forecast, result, receipt)
    assert not report['prospective_match']['matched']
    assert report['prospective_match']['reasons']


def test_actual_labels_and_following_never_enter_predictor_inputs(tmp_path):
    evaluator, parent, candidate, state, forecast, result = fixture()
    receipt = persisted(tmp_path, evaluator.plan(state, forecast))
    first = evaluator.complete(state, forecast, result, receipt)
    result['measured_q_path'][1:] = [[999.] * 6 for _ in range(5)]
    result['following_q_path'] = [[-999.] * 6 for _ in range(6)]
    result['endpoint_q_position'] = [777.] * 6
    second = evaluator.complete(state, forecast, result, receipt)
    for name in first['clock_evaluations']:
        assert first['clock_evaluations'][name]['parent']['predicted_q_path'] == second['clock_evaluations'][name]['parent']['predicted_q_path']
    assert second['clock_evaluations']['actual_dispatch_epoch']['parent']['metrics']['mae'] > 900
    for model in (parent, candidate):
        assert all(torch.equal(call[0], state.q) for call in model.calls)
        assert all(torch.count_nonzero(call[2].readbacks_position) == 0 for call in model.calls)


def test_partial_execution_is_unscored_without_any_inference():
    evaluator, parent, candidate, state, forecast, result = fixture()
    before = (len(parent.calls), len(candidate.calls))
    result.update(completed=False, policy_commands_attempted=2, policy_commands_returned=2,
                  measured_q_path=result['measured_q_path'][:2], commands_position=result['commands_position'][:2])
    report = evaluator.complete(state, forecast, result)
    assert report['scored'] is False and report['reason'] == 'incomplete_five_command_transaction'
    assert report['clock_evaluations'] == {}
    assert (len(parent.calls), len(candidate.calls)) == before
    assert len(result['measured_q_path']) == 2


def test_response_gate_failure_does_not_hide_complete_prediction_samples():
    evaluator, _, _, state, forecast, result = fixture()
    result['response_verified'] = False
    result['response_error'] = 'no material response'
    report = evaluator.complete(state, forecast, result)
    assert report['scored'] is True
    assert report['legacy_response_verified'] is False
    assert report['legacy_response_error'] == 'no material response'
    assert report['physical_acceptance_validated'] is False


def test_missing_raw_source_time_does_not_invent_retiming():
    evaluator, _, _, state, forecast, result = fixture()
    result.pop('source_q_readback_time_s')
    report = evaluator.complete(state, forecast, result)
    assert report['scored']
    assert report['clock_evaluations']['actual_source_readback'] == dict(scored=False, reason='source_q_readback_time_missing')


def test_active_command_baseline_respects_read_before_exactly_tied_write():
    evaluator, _, _, state, forecast, result = fixture()
    result['command_times_s'][1] = result['readback_times_s'][1]
    report = evaluator.complete(state, forecast, result)
    active = report['clock_evaluations']['actual_dispatch_epoch']['baselines']['active_command']
    assert active['predicted_q_path'][1] == [1.] * 6


def test_input_and_rng_are_isolated_from_bad_candidate():
    evaluator, _, candidate, state, forecast, _ = fixture()
    original = candidate.forward
    def bad(q, trace, history, queries):
        torch.rand(3)
        history.readbacks_position.add_(100)
        return original(q, trace, history, queries)
    candidate.forward = bad
    rng = torch.get_rng_state().clone()
    report = evaluator.plan(state, forecast)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.count_nonzero(state.history.readbacks_position) == 0
    bad_record = next(iter(report['candidates'].values()))
    assert bad_record['valid'] is False
    assert 'mutated' in bad_record['error']


def test_scores_saved_forecast_directly_not_a_silent_nominal_replacement(tmp_path):
    evaluator, parent, _, state, forecast, result = fixture()
    record = evaluator.plan(state, forecast)
    saved = record['parent']['predicted_q_path']
    receipt = persisted(tmp_path, record)
    original = parent.forward
    def numerically_different(q, trace, history, queries):
        output = original(q, trace, history, queries)
        path = output.predicted_q_path.clone()
        path[:, 1:] += .01
        return replace(output, predicted_q_path=path)
    parent.forward = numerically_different
    report = evaluator.complete(state, forecast, result, receipt)
    prospective = report['prospective_predictions']['parent']
    assert prospective['predicted_q_path'] == saved
    assert prospective['nominal_replay_max_difference'] == pytest.approx(.01, abs=1e-6)
    assert prospective['metrics']['mae'] != report['clock_evaluations']['nominal_dispatch']['parent']['metrics']['mae']


def test_changed_model_weights_are_not_labelled_as_frozen_predictions():
    evaluator, _, candidate, state, forecast, _ = fixture()
    candidate.gain.add_(.1)
    record = next(iter(evaluator.plan(state, forecast)['candidates'].values()))
    assert not record['valid'] and 'weights changed' in record['error']


@pytest.mark.parametrize('damage', ['source', 'clock', 'shape', 'hold'])
def test_untrustworthy_actual_alignment_is_unscored(damage):
    evaluator, _, _, state, forecast, result = fixture()
    if damage == 'source': result['measured_q_path'][0][0] = 1.
    elif damage == 'clock': result['command_times_s'][0] = result['readback_times_s'][1]
    elif damage == 'shape': result['measured_q_path'] = result['measured_q_path'][:-1]
    else: result['protective_hold_after_q5'] = False
    assert evaluator.complete(state, forecast, result)['scored'] is False


def test_same_base_name_different_artifacts_keep_separate_saved_predictions(tmp_path):
    original, parent, _, state, forecast, result = fixture()
    first = SimpleNamespace(**vars(original.candidates[0]))
    first.candidate_sha256 = 'a'*64
    second = SimpleNamespace(**vars(first))
    second.candidate_sha256 = 'b'*64
    second.model = Predictor(.9).eval()
    evaluator = LeWMShadowEvaluator(parent, [first, second])
    base = f'{first.method}-seed{first.seed}-{first.held_out_run}'
    expected = {f'{base}-{first.candidate_sha256}', f'{base}-{second.candidate_sha256}'}
    plan = evaluator.plan(state, forecast)
    assert set(plan['candidates']) == expected
    assert {entry['candidate_sha256'] for entry in plan['candidates'].values()} == {'a'*64, 'b'*64}
    first_path, second_path = [entry['predicted_q_path'] for entry in plan['candidates'].values()]
    assert first_path != second_path
    receipt = persisted(tmp_path, plan)
    report = evaluator.complete(state, forecast, result, receipt)
    assert report['prospective_match']['matched']
    assert set(report['prospective_predictions']['candidates']) == expected
    for name in expected:
        assert report['prospective_predictions']['candidates'][name]['predicted_q_path'] == plan['candidates'][name]['predicted_q_path']
    for clock_record in report['clock_evaluations'].values():
        assert set(clock_record['candidates']) == expected


@pytest.mark.parametrize('change_base_name', [False, True])
def test_duplicate_candidate_digest_is_rejected_even_with_another_label(change_base_name):
    original, parent, _, _, _, _ = fixture()
    first = original.candidates[0]
    second = SimpleNamespace(**vars(first))
    if change_base_name:
        second.seed = 42
    with pytest.raises(ValueError, match='same shadow candidate SHA256'):
        LeWMShadowEvaluator(parent, [first, second])


def test_two_unique_base_names_keep_the_existing_labels():
    original, parent, _, state, forecast, _ = fixture()
    first = original.candidates[0]
    second = SimpleNamespace(**vars(first))
    second.seed = 42
    second.candidate_sha256 = 'another-candidate-sha'
    evaluator = LeWMShadowEvaluator(parent, [first, second])
    expected = {f'{candidate.method}-seed{candidate.seed}-{candidate.held_out_run}'
                for candidate in (first, second)}
    assert set(evaluator.plan(state, forecast)['candidates']) == expected
