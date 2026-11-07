"""Fake-clock checks of CEM follow early exit; no hardware construction."""
from dataclasses import make_dataclass
from copy import deepcopy

import numpy as np
import pytest
import torch

from fcpg.v21_models import CommandTrace
from test_v21_hardware_bridge import setup


def cem_bridge(late_response_at=None, budget=.4, checked_joint_count=5):
    bridge, runtime, writes, holds = setup(normal_follow_budget_s=budget)
    bridge.cem_response_joint_count = checked_joint_count
    pending_type = make_dataclass('Pending', ('source_q', 'source_time', 'trace', 'response'))
    state_type = make_dataclass('State', ('q', 'time'))
    commands = torch.zeros(1, 5, 6)
    commands[0, :, 2] = torch.tensor([-1.5, -3., -3., -3., -3.])
    old = runtime.pending
    runtime.pending = pending_type(old.source_q, torch.tensor([10.], dtype=torch.float64),
                                   CommandTrace.policy(commands, torch.full((1, 5), .1)), old.response)
    runtime.state = state_type(old.source_q, torch.tensor([10.], dtype=torch.float64))
    runtime.pending_action_source = 'v21_lewm_cem'
    def observed():
        q = np.zeros(6, np.float32)
        if bridge.clock() >= 10.2: q[2] = -.177
        if late_response_at is not None and bridge.clock() >= late_response_at-1e-8: q[2] = -.355
        return q
    bridge.read_q = observed
    bridge.capture_endpoint = lambda: state_type(torch.from_numpy(observed())[None],
                                                torch.tensor([bridge.clock()], dtype=torch.float64))
    return bridge, runtime, writes, holds


def test_six_axis_follow_rejects_commanded_but_unresponsive_gripper():
    bridge, runtime, writes, holds = cem_bridge(checked_joint_count=6)
    runtime.pending.trace.positions[0, :, 2] = 0.
    runtime.pending.trace.positions[0, :, 5] = torch.tensor([-1.5, -3., -3., -3., -3.])
    result = bridge.execute()
    assert len(writes) == 5 and holds
    assert result['normal_follow']['cem_response_satisfied'] is False
    diagnostic = result['normal_follow']['cem_response_diagnostic']
    assert diagnostic['checked_joint_count'] == 6
    assert diagnostic['blocked_joint_indices'] == [5]


def test_unresponsive_cem_uses_existing_budget_without_rewriting_q5():
    bridge, runtime, writes, holds = cem_bridge()
    result = bridge.execute()
    assert len(writes) == 5 and holds
    assert len(result['normal_follow']['positions']) == 8
    assert result['normal_follow']['times_s'][-1] == pytest.approx(10.9)
    assert result['normal_follow']['goal_reached'] is False
    assert result['normal_follow']['cem_response_satisfied'] is False
    assert result['normal_follow']['cem_response_diagnostic']['blocked_arm_joint_indices'] == [2]
    assert result['readback_times_s'][-1] == pytest.approx(10.5)
    assert len(result['measured_q_path']) == 6 and runtime.actual[1][0, -1, 2] == pytest.approx(-.177)
    assert runtime.feedback == {} and result['model_feedback_path'] == 'original_five_step_readback'


@pytest.mark.parametrize('budget,response_at', [(.4,10.7),(2.,11.7)])
def test_late_actual_response_is_observed_without_shortening_full_window(budget,response_at):
    bridge, runtime, writes, holds = cem_bridge(late_response_at=response_at,budget=budget)
    result = bridge.execute()
    assert len(writes) == 5 and holds
    assert result['normal_follow']['times_s'][-1] == pytest.approx(12.5 if budget==2. else response_at)
    assert result['normal_follow']['goal_reached'] is False
    assert result['normal_follow']['hold_eligible'] is True
    assert result['normal_follow']['cem_response_satisfied'] is True
    assert result['measured_q_path'][-1][2] == pytest.approx(-.177)
    assert result['following_q_path'][-1][2] == pytest.approx(-.355)
    assert result['endpoint_q_position'][2] == pytest.approx(-.355)
    assert runtime.actual[2][0, -1] == pytest.approx(10.5)
    assert runtime.feedback == {}


@pytest.mark.parametrize('budget', [.4,2.])
def test_full_window_does_not_exit_on_three_quiet_response_samples(budget):
    bridge, runtime, writes, holds = cem_bridge(late_response_at=10.3,budget=budget)
    result = bridge.execute()
    assert result['normal_follow']['times_s'][-1] == pytest.approx(12.5 if budget==2. else 10.65)
    assert len(writes) == 5 and result['normal_follow']['hold_eligible']
    assert result['normal_follow']['goal_reached'] is False


@pytest.mark.parametrize('budget,stop_at', [(.4,10.75),(2.,11.65),(2.,12.5)])
def test_stop_interrupts_extended_follow(budget,stop_at):
    bridge, runtime, writes, holds = cem_bridge(budget=budget)
    bridge.cancelled = lambda: bridge.clock() >= stop_at-1e-8
    with pytest.raises(RuntimeError, match='cancelled'):
        bridge.execute()
    assert len(writes) == 5 and holds and runtime.stopped
    assert stop_at-.000001 <= holds[-1] <= stop_at+.01
    assert bridge.last_attempt['readback_times_s'][-1] == pytest.approx(10.5)


def test_late_physical_guard_still_interrupts_two_second_follow():
    bridge, runtime, writes, holds = cem_bridge(budget=2.)
    def geometry(q, commands):
        if bridge.clock() >= 11.15-1e-8:
            raise RuntimeError('late following geometry violation')
    bridge.physical_check=geometry
    with pytest.raises(RuntimeError,match='late following geometry'):
        bridge.execute()
    assert len(writes)==5 and runtime.stopped and holds[-1]==pytest.approx(11.15)
    assert bridge.last_attempt['readback_times_s'][-1]==pytest.approx(10.5)


def test_two_second_follow_commits_only_five_actual_labels_and_readback_indices():
    from fcpg.v21_core import V21Config, V21ResearchCore
    from fcpg.v21_data import TaskVocabulary
    from fcpg.v21_memory import VisualObservation
    from fcpg.v21_runtime import EncodedEndpoint, V21ResearchRuntime

    bridge, old_runtime, writes, _ = cem_bridge(late_response_at=11.7,budget=2.)
    commands=old_runtime.pending.trace.positions.clone()
    vocabulary=TaskVocabulary.build(['pick red block'])
    core=V21ResearchCore(V21Config(len(vocabulary.words))).eval()
    runtime=V21ResearchRuntime(core,vocabulary.encode(['pick red block']),
        physical_check=lambda q,u:None,lease_check=lambda:None,clock_fn=bridge.clock)
    q=torch.zeros(1,6)
    initial=VisualObservation(torch.zeros(1,2,256,384),q[:,None].expand(-1,2,-1),
        torch.full((1,2),10.,dtype=torch.float64),torch.ones(1,2,dtype=torch.bool))
    bridge.runtime=runtime
    bridge.capture_endpoint=lambda:EncodedEndpoint(torch.from_numpy(bridge.read_q())[None],
        torch.tensor([bridge.clock()],dtype=torch.float64),None)
    with torch.inference_mode():
        runtime.observe(EncodedEndpoint(q,torch.tensor([10.],dtype=torch.float64),initial))
        runtime.plan(command_builder=lambda state:commands,action_source='v21_lewm_cem')
        result=bridge.execute()
    assert len(writes)==5 and result['normal_follow']['times_s'][-1]==pytest.approx(12.5)
    assert runtime.state.memory.current_readback_index.tolist()==[5]
    assert runtime.state.history.valid.sum()==1 and runtime.state.q_error.shape==(1,5,6)
    torch.testing.assert_close(runtime.state.history.readbacks_position[:,-1],
        torch.tensor(result['measured_q_path'])[None],atol=0,rtol=0)
    assert runtime.state.history.readbacks_position[0,-1,-1,2]==pytest.approx(-.177)
    assert runtime.state.q[0,2]==pytest.approx(-.355)
    assert result['model_feedback_path']=='original_five_step_readback'


def test_invalid_follow_endpoint_is_not_treated_as_waiting_for_motion():
    bridge, _, _, _ = cem_bridge()
    q5 = np.array([0., 0., -.177, 0., 0., 0.], dtype=np.float32)
    record = dict(action_source='v21_lewm_cem', source_q_position=[0.]*6,
                  commands_position=bridge.runtime.pending.trace.positions[0].tolist(),
                  measured_q_path=[[0.]*6]+[q5.tolist()]*5,
                  readback_times_s=[10., 10.1, 10.2, 10.3, 10.4, 11.])
    before = deepcopy(record['measured_q_path'])
    with pytest.raises(ValueError, match='endpoint must follow'):
        bridge._follow(record, 10., q5)
    assert record['measured_q_path'] == before
