"""Malformed LeWM output must never authorize a CEM backup; no hardware IO."""
from types import SimpleNamespace

import pytest
import torch

from fcpg.v21_lewm_cem import pick_after_lewm


def forecast():
    return SimpleNamespace(predicted_q_path=torch.zeros(1, 6, 6),
                           q_uncertainty=torch.ones(1, 5, 6))


def malformed(kind):
    result = forecast()
    if kind == 'missing':
        del result.predicted_q_path
    elif kind == 'list':
        result.predicted_q_path = result.predicted_q_path.tolist()
    elif kind == 'no_source':
        result.predicted_q_path = torch.zeros(1, 5, 6)
    elif kind == 'extra_step':
        result.predicted_q_path = torch.zeros(1, 7, 6)
    elif kind == 'wrong_batch':
        result.predicted_q_path = torch.zeros(2, 6, 6)
    elif kind == 'nan_source':
        result.predicted_q_path[:, 0, 0] = float('nan')
    elif kind == 'nan_future':
        result.predicted_q_path[:, 1, 0] = float('nan')
    elif kind == 'wrong_source':
        result.predicted_q_path[:, 0, 0] = 1.
    elif kind == 'missing_uncertainty':
        del result.q_uncertainty
    elif kind == 'bad_uncertainty_shape':
        result.q_uncertainty = torch.ones(1, 6, 6)
    elif kind == 'nan_uncertainty':
        result.q_uncertainty[:, 0, 0] = float('nan')
    elif kind == 'negative_uncertainty':
        result.q_uncertainty[:, 0, 0] = -1.
    return result


@pytest.mark.parametrize('kind', [
    'missing', 'list', 'no_source', 'extra_step', 'wrong_batch',
    'nan_source', 'nan_future', 'wrong_source', 'missing_uncertainty',
    'bad_uncertainty_shape', 'nan_uncertainty', 'negative_uncertainty',
])
def test_invalid_forecast_cannot_authorize_any_backup(kind):
    model = SimpleNamespace(q=torch.zeros(1, 6), forecast=lambda _: malformed(kind))
    # Even a permissive physical callback cannot turn malformed model output
    # into evidence that the predicted trajectory passed the gate.
    with pytest.raises(RuntimeError, match='no physically admissible backup after LeWM'):
        pick_after_lewm(model, torch.zeros(2, 5, 6), lambda _q, _u: None)


def test_invalid_forecast_is_skipped_and_valid_next_backup_is_checked():
    outputs = iter([malformed('missing'), forecast()])
    model = SimpleNamespace(q=torch.zeros(1, 6), forecast=lambda _: next(outputs))
    checks = []
    backups = torch.zeros(2, 5, 6)
    backups[1, :, 0] = .5
    selected, index, result = pick_after_lewm(
        model, backups, lambda q, u: checks.append(u.clone()))
    assert index == 1
    assert torch.equal(selected, backups[1:2])
    assert len(checks) == 3  # two command checks, one valid predicted path
    assert torch.equal(checks[-1], result.predicted_q_path[:, 1:])
