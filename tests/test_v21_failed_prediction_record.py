"""Failure evidence preserves the original prediction after HOLD, never fake q."""
import pytest
import torch

from test_v21_hardware_bridge import setup


def test_partial_failure_keeps_forecast_after_hold_without_inventing_readbacks():
    bridge, runtime, writes, holds = setup(cancel_at=2)
    class Forecast:
        def detach(self): return self
        def cpu(self):
            assert holds and len(writes) == 2
            return torch.ones(1, 6, 6)
    runtime.pending.response.predicted_q_path = Forecast()
    with pytest.raises(RuntimeError, match='cancelled'):
        bridge.execute()
    result = bridge.last_attempt
    assert result['policy_commands_returned'] == 2 and result['completed'] is False
    assert len(result['measured_q_path']) == 2  # q0 and one real pre-command sample
    assert len(result['predicted_q_path'][0]) == 6
    assert result['prediction_scope'] == 'original_complete_plan_not_complete_execution'
    assert runtime.requires_home_after_incomplete_execution


def test_failure_before_dispatch_has_no_prediction_to_invent():
    bridge, runtime, writes, holds = setup()
    runtime.stopped = True
    with pytest.raises(RuntimeError, match='fresh planned'):
        bridge.execute()
    assert not writes and holds
    assert 'predicted_q_path' not in bridge.last_attempt
