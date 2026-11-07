from copy import deepcopy
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

    def forward(self, q, trace, history, queries):
        active = trace.active_at(queries, q)
        future = q[:, None] + self.gain * (active - q[:, None])
        return ARMOutput(torch.cat((q[:, None], future), 1), torch.ones_like(future), future)


@pytest.fixture
def setup():
    parent = Predictor(.25).eval()
    candidates = [SimpleNamespace(
        model=Predictor(gain).eval(), parent_sha256='f' * 64, candidate_sha256=digest * 64,
        method='lewm_mean', seed=21, held_out_run='reviewed_recent_four',
        parameter_names=('gain',), requires_independent_validation=True)
        for gain, digest in [(.75, 'a'), (.9, 'b')]]
    q = torch.zeros(1, 6)
    state = SimpleNamespace(q=q, time=torch.tensor([10.], dtype=torch.float64), history=empty_history(1))
    commands = torch.arange(1., 6.)[None, :, None].expand(1, 5, 6).clone()
    trace = CommandTrace.policy(commands, torch.full((1, 5), .1))
    queries = torch.tensor([[.1, .2, .3, .4, .5]], dtype=torch.float64)
    # The actual control forecast deliberately comes from the selected candidate.
    response = candidates[0].model(q, trace, state.history, queries)
    forecast = Forecast(q, state.time, trace, queries, response)
    result = dict(completed=True, policy_commands_attempted=5, policy_commands_returned=5,
                  protective_hold_after_q5=True, source_q_position=q[0].tolist(), source_time_s=10.,
                  source_q_readback_time_s=9.98, commands_position=commands[0].tolist(),
                  command_times_s=[10.002, 10.106, 10.207, 10.308, 10.409],
                  readback_times_s=[10., 10.105, 10.206, 10.307, 10.408, 10.511],
                  measured_q_path=response.predicted_q_path[0].tolist(), response_verified=True)
    return parent, candidates, state, forecast, result


def save_receipt(tmp_path, record, stamp=9.99):
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(record))
    return dict(record=record, path=str(path), persisted_at_s=stamp,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_default_and_explicit_none_preserve_existing_schema_and_predictions(setup, monkeypatch):
    parent, candidates, state, forecast, result = setup
    monkeypatch.setattr('fcpg.v21_lewm_shadow.uuid.uuid4', lambda: SimpleNamespace(hex='fixed'))
    old = LeWMShadowEvaluator(parent, candidates)
    explicit = LeWMShadowEvaluator(parent, candidates, controller_candidate_sha256=None)
    first, second = old.plan(state, forecast), explicit.plan(state, forecast)
    assert first == second
    assert old.complete(state, forecast, result) == explicit.complete(state, forecast, result)
    assert old.controller_identity is None
    assert set(old._base('plan')) == {
        'kind', 'schema', 'evaluator_id', 'control_influence',
        'physical_acceptance_validated', 'physical_accuracy_accepted'}
    assert 'controller_identity' not in first
    assert all('used_for_control' not in value for value in old._model_identities.values())


@pytest.mark.parametrize('digest', ['', 'a' * 63, 'A' * 64, 1, True, 'c' * 64])
def test_invalid_or_missing_controller_digest_is_rejected(setup, digest):
    parent, candidates, _, _, _ = setup
    with pytest.raises(ValueError, match='controller candidate SHA256'):
        LeWMShadowEvaluator(parent, candidates, controller_candidate_sha256=digest)


def test_controller_digest_must_have_exactly_one_match(setup):
    parent, candidates, _, _, _ = setup
    with pytest.raises(ValueError, match='uniquely match'):
        LeWMShadowEvaluator(parent, [], controller_candidate_sha256='a' * 64)
    with pytest.raises(ValueError, match='uniquely match'):
        LeWMShadowEvaluator(parent, [candidates[0], candidates[0]], controller_candidate_sha256='a' * 64)


def test_non_mean_candidate_cannot_claim_adapted_mean_controller_role(setup):
    parent, candidates, _, _, _ = setup
    candidates[0].method = 'jointwise_shadow_composition'
    with pytest.raises(ValueError, match='adapted lewm_mean'):
        LeWMShadowEvaluator(parent, candidates, controller_candidate_sha256='a' * 64)


def test_selected_controller_identity_routes_without_relabelling_parent(setup, tmp_path):
    parent, candidates, state, forecast, result = setup
    parent_gain = parent.gain.clone()
    evaluator = LeWMShadowEvaluator(parent, candidates, controller_candidate_sha256='a' * 64)
    earlier = replace(forecast, source_time=torch.tensor([9.5], dtype=torch.float64))
    planned = evaluator.plan(state, earlier)
    identity = planned['controller_identity']
    assert identity == dict(role='adapted_lewm_mean', candidate_sha256='a' * 64,
                           parent_sha256='f' * 64,
                           model_state_sha256=evaluator._state_sha(candidates[0].model),
                           control_influence=True, experimental=True, uncertainty_calibrated=False)
    assert planned['control_influence'] is False
    assert planned['parent']['role'] == 'parent'
    assert planned['parent']['used_for_control'] is False
    selected = next(key for key, value in planned['candidates'].items() if value['used_for_control'])
    assert planned['candidates'][selected]['candidate_sha256'] == 'a' * 64
    assert sum(value['used_for_control'] for value in planned['candidates'].values()) == 1
    receipt = save_receipt(tmp_path, planned)
    report = evaluator.complete(state, forecast, result, receipt)
    assert report['scored'] and report['prospective_match']['matched']
    assert report['controller_identity'] == identity
    assert report['control_influence'] is False
    original = parent(state.q, forecast.trace, state.history, forecast.query_offsets).predicted_q_path[0]
    prospective = report['prospective_predictions']
    np.testing.assert_array_equal(prospective['parent']['predicted_q_path'], original.numpy())
    assert prospective['parent']['predicted_q_path'] != forecast.response.predicted_q_path[0].tolist()
    assert prospective['parent']['metrics']['mae'] > 0
    assert prospective['candidates'][selected]['metrics']['mae'] == 0
    for clock_record in report['clock_evaluations'].values():
        assert clock_record['parent']['used_for_control'] is False
        assert clock_record['candidates'][selected]['used_for_control'] is True
    assert torch.equal(parent.gain, parent_gain)


def test_returned_controller_identity_cannot_mutate_evaluator_metadata(setup):
    parent, candidates, state, forecast, _ = setup
    evaluator = LeWMShadowEvaluator(parent, candidates, controller_candidate_sha256='a' * 64)
    public = evaluator.controller_identity
    public['candidate_sha256'] = 'b' * 64
    evaluator.plan(state, forecast)['controller_identity']['experimental'] = False
    assert evaluator.controller_identity['candidate_sha256'] == 'a' * 64
    assert evaluator.controller_identity['experimental'] is True


@pytest.mark.parametrize('damage', ['controller_sha', 'controller_boolean_type', 'missing_controller',
                                   'parent_routing', 'candidate_routing', 'late', 'source', 'file'])
def test_control_mode_receipt_mismatch_remains_fail_closed(setup, tmp_path, damage):
    parent, candidates, state, forecast, result = setup
    evaluator = LeWMShadowEvaluator(parent, candidates, controller_candidate_sha256='a' * 64)
    record = deepcopy(evaluator.plan(state, forecast))
    if damage == 'controller_sha': record['controller_identity']['candidate_sha256'] = 'b' * 64
    elif damage == 'controller_boolean_type': record['controller_identity']['experimental'] = 1
    elif damage == 'missing_controller': record.pop('controller_identity')
    elif damage == 'parent_routing': record['parent']['used_for_control'] = True
    elif damage == 'candidate_routing':
        next(value for value in record['candidates'].values() if value['used_for_control'])['used_for_control'] = False
    receipt = save_receipt(tmp_path, record)
    if damage == 'late': receipt['persisted_at_s'] = result['command_times_s'][0]
    elif damage == 'source':
        state.q = state.q + .01
        forecast = replace(forecast, source_q=state.q)
        result['source_q_position'] = state.q[0].tolist()
        result['measured_q_path'][0] = state.q[0].tolist()
    elif damage == 'file':
        (tmp_path / 'plan.json').write_text('{}')
    report = evaluator.complete(state, forecast, result, receipt)
    assert not report['prospective_match']['matched']
    assert report['prospective_match']['reasons']
    assert report['prospective_predictions']['scored'] is False
