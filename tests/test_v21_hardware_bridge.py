"""Hardware-free tests for V21's bounded physical transaction adapter."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from fcpg.v21_hardware_bridge import V21TransactionBridge, completion_hold_record
from fcpg.v21_models import CommandTrace


def setup(*, delay_begin=0., slow_write=False, mutate=False, cancel_at=None, normal_follow_budget_s=0.,
          maximum_lateness_s=.03):
    now = [10.]
    q = np.zeros(6, np.float32)
    writes, holds = [], []
    class Runtime:
        stopped = False
        pending_action_source = 'v21_actor'
        def __init__(self):
            self.pending = SimpleNamespace(source_q=torch.zeros(1, 6),
                trace=CommandTrace.policy(torch.ones(1, 5, 6), torch.full((1, 5), .1)),
                response=SimpleNamespace(predicted_q_path=torch.zeros(1, 6, 6)))
        def begin_execution(self, source, stamp):
            now[0] += delay_begin
            return self.pending.trace.positions
        def complete(self, trace, path, times, endpoint, **feedback):
            self.actual = (trace, path, times)
            self.feedback=feedback
            result={'feedback_retimed': [True], 'history_chunks': [1]}
            if feedback:result['nominal_retimed_prediction']=self.pending.response.predicted_q_path.tolist()
            return result
        def stop(self): self.stopped = True; self.pending = None
    runtime = Runtime()
    def write(command, previous):
        writes.append(command.copy()); now[0] += .04 if slow_write else .002
        return command+1 if mutate else command.copy()
    bridge = V21TransactionBridge(runtime, read_q=lambda:q.copy(), write_q=write,
        capture_endpoint=lambda: SimpleNamespace(time=torch.tensor([now[0]], dtype=torch.float64),
                                                 q=torch.from_numpy(q.copy())[None]),
        physical_check=lambda q, c:None, halt=lambda:holds.append(now[0]),
        clock=lambda:now[0], wait=lambda delay:now.__setitem__(0, now[0]+delay),
        normal_follow_budget_s=normal_follow_budget_s,
        maximum_lateness_s=maximum_lateness_s,
        cancelled=lambda:cancel_at is not None and len(writes) >= cancel_at)
    return bridge, runtime, writes, holds


def test_five_policy_writes_actual_clock_and_protective_hold():
    bridge, runtime, writes, holds = setup()
    result = bridge.execute()
    assert result['completed'] and len(writes) == 5
    assert len(holds) == 1 and result['protective_hold_after_q5']
    assert result['endpoint_q_position'] == [0.] * 6
    assert result['endpoint_time_s'] >= result['readback_times_s'][-1]
    assert runtime.stopped
    trace, path, times = runtime.actual
    assert trace.source == 'recorded' and path.shape == (1, 6, 6) and times.shape == (1, 6)
    assert trace.offsets.shape == (1, 5) and trace.offsets.dtype == torch.float64
    torch.testing.assert_close(trace.offsets, torch.arange(5,dtype=torch.float64)[None]*.1, atol=1e-9, rtol=0)


def test_forecast_logging_is_after_motor_deadlines_and_hold():
    bridge, runtime, writes, holds = setup()
    class LoggedForecast:
        def detach(self): return self
        def cpu(self):
            assert len(writes) == 5 and holds, 'telemetry copied inside motor deadline'
            return torch.zeros(1, 6, 6)
    runtime.pending.response.predicted_q_path = LoggedForecast()
    result = bridge.execute()
    assert result['completed'] and len(result['predicted_q_path'][0]) == 6


def test_cem_keeps_actual_q0_read_time_separate_from_dispatch_epoch():
    from dataclasses import make_dataclass
    bridge,runtime,writes,holds=setup(delay_begin=.02)
    Pending=make_dataclass('Pending',('source_q','source_time','trace','response'))
    State=make_dataclass('State',('q','time'))
    old=runtime.pending
    runtime.pending=Pending(old.source_q,torch.tensor([10.],dtype=torch.float64),old.trace,old.response)
    runtime.state=State(old.source_q,torch.tensor([10.],dtype=torch.float64))
    runtime.pending_action_source='v21_lewm_cem'
    result=bridge.execute()
    assert result['source_q_readback_time_s']==10.
    assert result['source_time_s']==pytest.approx(10.02)
    assert result['source_q_age_at_dispatch_epoch_s']==pytest.approx(.02)
    assert result['command_times_s'][0]>=result['source_time_s']
    assert bridge.last_forecast is not None and runtime.pending is None
    assert len(writes)==5 and len(holds)==1


@pytest.mark.parametrize('option,expected_writes', [({'delay_begin':.04},0),({'slow_write':True},1),
                                                   ({'mutate':True},1),({'cancel_at':2},2)])
def test_fault_or_stop_halts_without_remaining_policy_writes(option, expected_writes):
    bridge, runtime, writes, holds = setup(**option)
    with pytest.raises(RuntimeError): bridge.execute()
    assert len(writes) == expected_writes and holds and runtime.stopped
    assert not bridge.last_attempt['completed']


def test_no_second_execution_or_plan_without_new_observation():
    bridge, runtime, writes, holds = setup()
    bridge.execute()
    with pytest.raises(RuntimeError, match='fresh planned'):
        bridge.execute()
    assert len(writes) == 5


def test_invalid_actual_readback_stops_before_write():
    bridge, runtime, writes, holds = setup()
    bridge.read_q = lambda: np.full(6, np.nan)
    with pytest.raises(ValueError, match='finite'): bridge.execute()
    assert not writes and holds and runtime.stopped


def test_operation_guard_cannot_modify_real_readback():
    bridge,runtime,writes,holds=setup()
    bridge.extra_readback_check=lambda q:q.__setitem__(0,10.)
    with pytest.raises(RuntimeError,match='changed an actual readback'):bridge.execute()
    assert not writes and holds and runtime.stopped


def test_operation_guard_also_checks_the_post_image_endpoint():
    bridge,runtime,writes,holds=setup()
    original=bridge.capture_endpoint
    bridge.capture_endpoint=lambda:SimpleNamespace(time=original().time,q=torch.ones(1,6)*2)
    def check(q):
        if q[0]>1.:raise RuntimeError('operation clearance')
    bridge.extra_readback_check=check
    with pytest.raises(RuntimeError,match='operation clearance'):bridge.execute()
    assert len(writes)==5 and holds and runtime.stopped
    assert not bridge.last_attempt['completed'] and not hasattr(runtime,'actual')


def test_slow_read_is_reported_without_sending_the_late_fifth_command():
    bridge,runtime,writes,holds=setup()
    original=bridge.read_q
    calls=[0]
    def read():
        calls[0]+=1
        if calls[0]==6:bridge.wait(.1)
        return original()
    bridge.read_q=read
    with pytest.raises(RuntimeError,match='deadline'):bridge.execute()
    assert len(writes)==4 and holds and runtime.stopped
    assert bridge.last_attempt['read_calls'][-1]['duration_s']==pytest.approx(.1)
    assert bridge.last_attempt['deadline_miss']['excess_s']==pytest.approx(.07)


def test_write_overrun_reports_exact_excess_without_relaxing_deadline():
    bridge,runtime,writes,holds=setup(slow_write=True)
    with pytest.raises(RuntimeError,match='deadline'):bridge.execute()
    miss=bridge.last_attempt['deadline_miss']
    assert miss['command_index']==0 and miss['allowed_lateness_s']==.03
    assert miss['excess_s']==pytest.approx(.01)
    assert len(writes)==1 and holds and runtime.stopped


def test_hardware_lateness_window_can_cover_transport_jitter_without_changing_action_period():
    bridge, runtime, writes, holds = setup(slow_write=True, maximum_lateness_s=.08)
    result = bridge.execute()
    assert result['completed'] and len(writes) == 5 and len(holds) == 1
    assert result['maximum_lateness_s'] == pytest.approx(.08)
    assert min(np.diff(result['command_times_s'])) >= .1 - 1e-6


def test_hold_evidence_does_not_replace_q5_or_add_a_policy_command():
    import json
    bridge,runtime,writes,holds=setup()
    def hold():
        holds.append(True)
        return dict(event='safe_hold_verified',attempts=[dict(held_state=np.full(6,.25),mismatched_joints={})])
    bridge.halt=hold
    result=bridge.execute()
    assert len(writes)==5 and len(holds)==1
    assert result['measured_q_path'][-1]==[0.]*6
    assert result['completion_hold']['held_position']==[.25]*6
    assert result['completion_hold']['goal_register_verified'] is True
    assert result['completion_hold']['started_s'] >= result['readback_times_s'][-1]
    json.dumps(result,allow_nan=False)


def test_missing_hold_record_is_not_filled_with_q5():
    record=completion_hold_record(None,1.,1.1)
    assert record['held_position'] is None and not record['record_available']
    assert record['goal_register_verified'] is None


@pytest.mark.parametrize('budget', [.4, 2.])
def test_normal_follow_keeps_five_writes_and_original_q5(budget):
    bridge,runtime,writes,holds=setup(normal_follow_budget_s=budget)
    result=bridge.execute()
    assert len(writes)==5 and len(holds)==1
    assert result['readback_times_s'][-1]==pytest.approx(10.5)
    assert result['normal_follow']['goal_reached'] is False  # fixed fake readback is one unit short
    assert result['normal_follow']['hold_eligible'] is True
    final_time = 12.5 if budget == 2. else 10.65
    assert result['normal_follow']['times_s'][-1]==pytest.approx(final_time)
    assert runtime.actual[2][0,-1].item()==pytest.approx(10.5)
    assert runtime.feedback == {}  # no later following sample may replace q5
    assert 'feedback_q_path' not in result
    assert result['model_feedback_path']=='original_five_step_readback'
    assert result['following_readback_times_s'][-1]==pytest.approx(final_time)
    assert result['following_q_path'][-1]==result['normal_follow']['positions'][-1]
    assert result['completion_hold']['started_s']>=final_time


@pytest.mark.parametrize('budget,samples,final_time', [(.4, 8, 10.9), (2., 40, 12.5)])
def test_follow_budget_stops_unreached_goal_without_more_commands(budget,samples,final_time):
    bridge,runtime,writes,holds=setup(normal_follow_budget_s=budget)
    bridge.follow_position_tolerance=.01
    result=bridge.execute()
    assert len(writes)==5 and len(result['normal_follow']['positions'])==samples
    assert result['normal_follow']['goal_reached'] is False
    assert result['normal_follow']['reason']==('fixed_window_complete' if budget==2. else 'budget_exhausted')
    assert result['normal_follow']['times_s'][-1]==pytest.approx(final_time)
    assert runtime.actual[2][0,-1].item()==pytest.approx(10.5)
    assert runtime.feedback == {}
    assert runtime.stopped and len(holds)==1


def test_two_second_follow_keeps_absolute_schedule_and_records_actual_read_latency():
    bridge,runtime,writes,holds=setup(normal_follow_budget_s=2.)
    bridge.follow_position_tolerance=.01
    original=bridge.read_q
    def delayed_follow_read():
        if bridge.clock() >= 10.55-1e-8:
            bridge.wait(.01)
        return original()
    bridge.read_q=delayed_follow_read
    result=bridge.execute()
    follow=result['normal_follow']
    assert len(writes)==5 and len(follow['positions'])==40
    assert follow['times_s']==pytest.approx([10.5+i*.05+.01 for i in range(1,41)])
    assert holds[0]==pytest.approx(12.51)
    assert result['readback_times_s'][-1]==pytest.approx(10.5)
    assert runtime.actual[1].shape==(1,6,6) and runtime.actual[2][0,-1]==pytest.approx(10.5)
    assert runtime.feedback=={} and result['model_feedback_path']=='original_five_step_readback'


@pytest.mark.parametrize('budget', [-.1, .5, 1., 2.001, float('nan'), float('inf')])
def test_follow_constructor_rejects_unapproved_budgets(budget):
    with pytest.raises(ValueError,match='bounded-follow'):
        setup(normal_follow_budget_s=budget)


def test_stop_during_normal_follow_still_holds_immediately():
    bridge,runtime,writes,holds=setup();bridge.normal_follow_budget_s=.4
    bridge.cancelled=lambda:bridge.clock()>=10.6
    with pytest.raises(RuntimeError,match='cancelled'):bridge.execute()
    assert len(writes)==5 and len(holds)==1 and runtime.stopped
    assert not bridge.last_attempt['completed']
    assert bridge.last_attempt['readback_times_s'][-1]==pytest.approx(10.5)


def test_physical_guard_is_checked_during_follow():
    bridge,runtime,writes,holds=setup();bridge.normal_follow_budget_s=.4
    def guard(q,commands):
        if bridge.clock()>=10.55:raise RuntimeError('following physical guard')
    bridge.physical_check=guard
    with pytest.raises(RuntimeError,match='physical guard'):bridge.execute()
    assert len(writes)==5 and len(holds)==1 and runtime.stopped
    assert len(bridge.last_attempt['normal_follow']['positions'])==1
