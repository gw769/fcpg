"""CEM visual ranking expires even when the measured robot pose stays fixed."""
import pytest
import torch

from fcpg.v21_runtime import PreExecutionStateChanged
from test_v21_planning_memory import runtime, endpoint  # noqa: F401


@pytest.mark.parametrize('mode', ['always', 'bounded'])
@pytest.mark.parametrize('drift', [0., .1])
def test_expired_cem_is_rejected_before_reusing_visual_ranking(runtime, mode, drift):
    value, now = runtime
    if mode == 'always':
        value.configure_pre_execution_recompute('always')
    before = value.state
    calls = []
    def builder(state):
        calls.append(state)
        return state.q[:, None].expand(-1, 5, -1).clone()
    image = endpoint(now[0], 3)
    value.plan(endpoint=image, command_builder=builder, action_source='v21_lewm_cem')
    now[0] += value.max_plan_age_s + .01
    with pytest.raises(PreExecutionStateChanged):
        value.begin_execution(image.q + drift, torch.tensor([now[0]], dtype=torch.float64))
    assert len(calls) == 1
    assert value.state is before
    assert value.stopped and value.pending is None and not value.executing


def test_fresh_cem_still_dispatches_without_changing_action_source(runtime):
    value, now = runtime
    image = endpoint(now[0], 3)
    commands = value.plan(endpoint=image,
        command_builder=lambda state: state.q[:, None].expand(-1, 5, -1).clone(),
        action_source='v21_lewm_cem')
    now[0] += .01
    dispatched = value.begin_execution(image.q, torch.tensor([now[0]], dtype=torch.float64))
    torch.testing.assert_close(dispatched, commands)
    assert value.executing and value.pending_action_source == 'v21_lewm_cem'
