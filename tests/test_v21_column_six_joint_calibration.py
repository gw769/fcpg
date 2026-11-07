from types import SimpleNamespace
import numpy as np
import pytest
import torch

from fcpg.v21_column_dino_transport import (
    ColumnDinoFGSample, fit_column_dino_fg_transport, require_actual_probe_response,
)
from fcpg.v21_live_service import V21LiveService
from fcpg import v21_live_service as live


@pytest.mark.parametrize('schedule', [
    [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1),
     (3, 1), (4, 1), (5, 1)],
    [(0, -1), (5, 1), (5, -1), (0, 1), (4, -1), (4, 1),
     (2, -1), (1, -1), (3, 1)],
])
def test_nine_probe_schedule_identifies_all_six_columns_without_lowering_fit_gates(schedule):
    q = np.array([0., -97., 99., 61., -53., 31.])
    jacobian = np.array([[2., 1., -3., 2., 1., .5], [-1., 3., 1., -2., 4., -.25]])
    start = q.copy()
    rows = [ColumnDinoFGSample(q.copy(), [70., 100.], 1.)]
    for index, (axis, direction) in enumerate(schedule):
        q = q.copy(); q[axis] += 2.*direction
        rows.append(ColumnDinoFGSample(q, np.array([70., 100.])+jacobian@(q-start), index+2.))
    with pytest.raises(ValueError, match='insufficient'):
        fit_column_dino_fg_transport(rows[:-1])
    result = fit_column_dino_fg_transport(rows)
    assert result['sample_count'] == 10
    np.testing.assert_allclose(result['jacobian_px_per_joint'], jacobian, atol=1e-12)
    assert result['lower_q6'] == 31. and result['upper_q6'] == 33.
    assert result['design_condition'] < 100. and result['residual_rms_px'] < 4.


def test_j6_probe_requires_actual_independent_readback_not_goal():
    result = dict(source_q_position=[0.]*6, measured_q_path=[[0.]*6]*6)
    with pytest.raises(ValueError, match='did not achieve'):
        require_actual_probe_response(result, 5, 1, minimum_motion=.25)
    result['measured_q_path'] = [[0.]*6]*5 + [[0., 0., 0., 0., 0., 1.]]
    assert require_actual_probe_response(result, 5, 1, minimum_motion=.25)[5] == 1.
    result['measured_q_path'][-1][2] = .3
    with pytest.raises(ValueError, match='mixed actual joint'):
        require_actual_probe_response(result, 5, 1, minimum_motion=.25)


def service(tmp_path, monkeypatch):
    obj = object.__new__(V21LiveService)
    home = np.array([0., -97., 99., 61., -53., 31.34])
    monkeypatch.setattr(live.established, '_load_v177_home_state', lambda *_args, **_kwargs: home.copy())
    obj.column_dino_fg_calibration_enabled = True
    obj.column_dino_fg_response = dict(jacobian_px_per_joint=np.zeros((2, 5)))
    obj.column_dino_fg_transport_status = dict(available=True)
    obj._column_dino_fg_samples = [ColumnDinoFGSample(home, [50., 50.], 1.)]
    obj.auxiliary_budget = SimpleNamespace(active=True, block_attempts=8, maximum_blocks=10)
    obj.owner = SimpleNamespace(runtime=SimpleNamespace(pending=None, executing=False),
                                io=SimpleNamespace(settle_step_tolerance=.25))
    obj.fingertips = {'column': dict(available=True, source='column-rgb-jaw-contour',
        q_at_image=home, m_px=[50., 50.], image_time_s=1.)}
    obj.output = tmp_path
    obj._refit_column_dino_fg_transport = lambda: None
    obj._record_column_dino_fg_sample = lambda _tips: obj._column_dino_fg_samples.append(
        ColumnDinoFGSample(home+np.array([0., 0., 0., 0., 0., 2.]), [51., 50.], 2.))
    def run(build, source):
        commands = build(SimpleNamespace(q=torch.tensor(home[None], dtype=torch.float32)))
        np.testing.assert_allclose(commands[0, :, :5], np.tile(home[:5], (5, 1)), atol=1e-5)
        np.testing.assert_allclose(commands[0, :, 5].numpy()-home[5], [.4, .8, 1.2, 1.6, 2.], atol=1e-5)
        # Intermediate observations are not independent successful probes.
        obj._column_dino_fg_samples.append(ColumnDinoFGSample(home, [50., 50.], 1.5))
        return dict(source_q_position=home.tolist(), measured_q_path=np.vstack((home, commands[0].numpy())).tolist())
    obj._run_alignment_motion = run
    return obj


def test_service_allows_five_to_six_column_upgrade_preserves_home_budget_and_one_sample(tmp_path, monkeypatch):
    obj = service(tmp_path, monkeypatch)
    result = obj._column_dino_fg_probe(dict(joint=5, direction=1, confirmed=True))
    assert len(obj._column_dino_fg_samples) == 2
    assert result['column_dino_fg_probe']['gripper_command_held'] is False
    assert result['column_dino_fg_probe']['actual_joint_response_verified'] is True
    assert result['column_dino_fg_probe']['task_success_verified'] is False


def test_successful_probe_publishes_acceptance_receipt_for_http_polling(tmp_path, monkeypatch):
    obj = service(tmp_path, monkeypatch)
    # The shared motion runner records execution before calibration validation.
    obj.last_execution = dict(completed=True)
    result = obj._column_dino_fg_probe(dict(joint=5, direction=1, confirmed=True))
    public = live._compact_execution_status(obj.last_execution)
    assert public['column_dino_fg_probe'] == result['column_dino_fg_probe']
    assert public['actual_probe_delta'] == result['actual_probe_delta']
    result['column_dino_fg_probe']['actual_joint_response_verified'] = False
    assert obj.last_execution['column_dino_fg_probe']['actual_joint_response_verified'] is True


def test_tenth_calibration_probe_cannot_consume_reserved_home(tmp_path, monkeypatch):
    obj = service(tmp_path, monkeypatch); obj.auxiliary_budget.block_attempts = 9
    with pytest.raises(RuntimeError, match='retain one'):
        obj._column_dino_fg_probe(dict(joint=5, direction=1, confirmed=True))


def test_j6_probe_rejects_unfolded_source_without_any_dispatch(tmp_path, monkeypatch):
    obj = service(tmp_path, monkeypatch)
    def run(build, source):
        return build(SimpleNamespace(q=torch.tensor([[0., -50., 99., 61., -53., 31.34]])))
    obj._run_alignment_motion = run
    with pytest.raises(RuntimeError, match='near HOME'):
        obj._column_dino_fg_probe(dict(joint=5, direction=1, confirmed=True))


def test_fitted_six_joint_calibration_is_frozen_across_unrelated_later_observations(tmp_path, monkeypatch):
    obj = service(tmp_path, monkeypatch)
    obj._column_dino_fg_calibration_frozen = True
    before = tuple(obj._column_dino_fg_samples)
    V21LiveService._record_column_dino_fg_sample(obj, obj.fingertips)
    assert tuple(obj._column_dino_fg_samples) == before
