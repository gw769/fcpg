"""Configured trial limits use fake owners only; no sockets, cameras or serial IO."""
from types import SimpleNamespace

import pytest

from fcpg.v21_live_service import V21LiveService
from scripts import serve_so101_v21_research as cli
from test_v21_live_service import service  # noqa: F401 -- fake IO fixture


@pytest.mark.parametrize('budget', [0, -1, 33, True, False, 2.0, '2', None])
def test_invalid_service_budget_precedes_owner_and_output(tmp_path, budget):
    calls = []
    output = tmp_path / 'uncreated'
    with pytest.raises(ValueError, match='trial_max_blocks'):
        V21LiveService(tmp_path / 'model', output, trial_max_blocks=budget,
                       owner_factory=lambda *a, **kw: calls.append((a, kw)))
    assert calls == []
    assert not output.exists()


def test_default_service_budget_remains_sixteen(service):
    assert service.trial.max_blocks == 16
    assert service.trial.status()['max_blocks'] == 16
    assert service.owner.writes == 0


def test_two_block_budget_reaches_actual_trial_and_stops_at_two(service, tmp_path, monkeypatch):
    import fcpg.v21_live_service as live

    configured = V21LiveService(tmp_path / 'model2', tmp_path / 'out2', device='cpu',
                               owner_factory=type(service.owner), target_diagnostics=False,
                               clock=service.clock, trial_max_blocks=2)
    try:
        assert configured.trial.status()['max_blocks'] == 2
        monkeypatch.setattr(live.established, '_load_v177_home_state',
                            lambda *a, **kw: configured.owner.q.copy())
        original = configured.owner.bridge.execute

        def execute():
            result = original()
            result['endpoint_time_s'] = result['readback_times_s'][-1]
            return result

        configured.owner.bridge.execute = execute
        configured.handle('trial')
        configured.trial.join(timeout=30.)
        result = configured.trial.status()
        assert result['max_blocks'] == result['blocks_started'] == result['blocks_completed'] == 2
        assert result['outcome'] == 'budget_exhausted'
        assert result['policy_commands_returned'] == configured.owner.writes == 10
        assert configured.owner.runtime.stopped and not result['running']
        assert result['task_success_verified'] is False
    finally:
        configured.close()


@pytest.mark.parametrize('value', ['0', '-1', '17', '2.5', 'True'])
def test_invalid_cli_budget_precedes_server_and_owner(monkeypatch, value):
    def forbidden(*a, **kw):
        pytest.fail('invalid budget must fail before server or owner creation')

    monkeypatch.setattr(cli, 'make_server', forbidden)
    monkeypatch.setattr(cli, 'V21LiveService', forbidden)
    with pytest.raises(SystemExit) as caught:
        cli.main(['--enable-hardware', '--trial-max-blocks', value])
    assert caught.value.code == 2


@pytest.mark.parametrize(('args', 'expected'), [([], 16), (['--trial-max-blocks', '1'], 1),
                                               (['--trial-max-blocks', '2'], 2),
                                               (['--trial-max-blocks', '16'], 16)])
def test_cli_budget_is_forwarded(monkeypatch, args, expected):
    calls = []
    closed = []
    server = SimpleNamespace(serve_forever=lambda **kw: None, shutdown=lambda: None,
                             server_close=lambda: None)
    monkeypatch.setattr(cli, 'make_server', lambda *a: server)
    monkeypatch.setattr(cli.signal, 'signal', lambda *a: None)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')

    def owner(*a, **kw):
        calls.append(kw)
        return SimpleNamespace(close=lambda: closed.append(True))

    monkeypatch.setattr(cli, 'V21LiveService', owner)
    cli.main(['--enable-hardware', *args])
    assert len(calls) == 1 and calls[0]['trial_max_blocks'] == expected
    assert closed == [True]
