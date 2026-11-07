from types import SimpleNamespace

import torch

from fcpg import v21_lewm_cem as cem
from fcpg.prod_v20_policy import empty_history
from fcpg.v21_cem_trace import CEMTrace


def assert_smooth(q, commands):
    source = q.reshape(*([1] * (commands.ndim - 1)), 6)
    movement = torch.diff(torch.cat((source.expand(*commands.shape[:-2], 1, 6), commands), -2), dim=-2)
    direction = (commands[..., -1:, :] - source).sign()
    assert (movement * direction >= -2e-5).all()
    assert (movement.abs() <= commands.new_tensor(cem.MAX_STEP) + 2e-5).all()
    end = commands[..., -1, :]
    expected = cem.smooth_endpoint_chunk(q, end, end.new_full((6,), -100), end.new_full((6,), 100), end.new_tensor(cem.MAX_STEP))
    torch.testing.assert_close(commands, expected, atol=2e-5, rtol=0)


def test_endpoint_chunk_ignores_zigzags_without_shrinking_reachable_q5():
    q = torch.tensor([[0., -50., 50., 50., 0., 30.]])
    actor = q[:, None].expand(1, 5, 6).clone()
    actor[:, 0] += 20
    actor[:, 1] -= 10
    actor[:, -1] += torch.tensor([2.5, 1., -.5, 0., 1., 1.])
    path, metadata = cem.prepare_actor_reference(actor, q, 'smooth_actor_endpoint', action_scale=3.)
    torch.testing.assert_close(path[:, -1], q + 3 * (actor[:, -1] - q))
    assert_smooth(q, path)
    assert metadata['time_interpolated'] and not metadata['actor_time_shape_preserved']


def test_lewm_and_physical_gate_see_smoothed_candidates_before_selection(monkeypatch):
    monkeypatch.setattr(cem, 'CEM_POPULATION', 6)
    monkeypatch.setattr(cem, 'CEM_ELITES', 2)
    monkeypatch.setattr(cem, 'CEM_ITERS', 2)
    q = torch.tensor([[0., -50., 50., 50., 0., 30.]])
    actor = q[:, None] + torch.arange(1, 6)[None, :, None] * torch.tensor([.3, .8, -.7, .2, -.3, .2])
    calls = []
    def dynamics(source, trace, history, query):
        assert_smooth(source[:1], trace.positions)
        calls.append(len(source))
        return SimpleNamespace(predicted_q_path=torch.cat([source[:, None], trace.positions], 1), q_uncertainty=torch.ones_like(trace.positions))
    def physical(source, commands):
        if (commands[..., 0] < -.1).any():
            raise RuntimeError('exercise candidate replacement')
    trace = CEMTrace()
    selected, receipt = cem.plan_commands(dynamics, SimpleNamespace(q=q, history=empty_history(1)),
        actor, torch.full((1, 5), .1), physical, objective='actor_response_tracking',
        candidate_parameterization='smooth_actor_endpoint', action_scale=3.,
        generator=torch.Generator().manual_seed(20), trace=trace)
    assert_smooth(q, selected)
    assert calls and receipt['response_forecast_admission']['checked_joint_count'] == 6
    for event in trace.payload()['events']:
        if event['event'] == 'cem_iteration':
            assert_smooth(q, event['values']['candidates'])
