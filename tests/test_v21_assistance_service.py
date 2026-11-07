"""Service-level assistance transactions using fake IO only.

These tests drive actual handle/plan/execute/stop code with the existing runtime
fixture.  The assistance builder deliberately mutates every relevant field so
the test catches accidental /plan writes, fresh-plan double counting and STOP
races rather than merely testing a standalone snapshot helper.
"""
from dataclasses import replace
import threading

import numpy as np
import pytest

from test_v21_live_service import service  # noqa: F401 -- shared no-hardware fixture


@pytest.fixture
def assisted(service, monkeypatch):
    service.grounding_correction = True
    monkeypatch.setattr(service.owner.runtime.core, "config",
                        replace(service.owner.runtime.core.config, object_relative=True))
    service.builder_calls = 0

    def builder(state):
        service.builder_calls += 1
        service._grounding_bias += np.array([1., -2.])
        service._grounding_prev_end = np.array([3., 4.])
        service._roll_prev = (-50., 10.)
        service._roll_gain = 1.5
        service._search["blocks"] += 1
        service._search["votes"].append(True)
        previous = service._carry or {"blocks": 0}
        service._carry = {"phase": "lift", "blocks": previous["blocks"] + 1}
        service._grounding_vetoes += 1
        return state.q[:, None, :].repeat(1, 5, 1).clone()

    monkeypatch.setattr(service, "_grounding_builder", builder)
    return service


def _assert_assistance(service, completed):
    np.testing.assert_array_equal(service._grounding_bias, [completed, -2 * completed])
    assert service._search["blocks"] == completed
    assert service._search["votes"] == [True] * completed
    assert service._grounding_vetoes == completed
    if completed:
        assert service._carry == {"phase": "lift", "blocks": completed}
        np.testing.assert_array_equal(service._grounding_prev_end, [3., 4.])
    else:
        assert service._carry is None
        assert service._grounding_prev_end is None


def test_repeated_plan_changes_neither_helper_nor_actual_history(assisted):
    value = assisted
    before = value.owner.runtime.state
    for _ in range(4):
        result = value.handle("plan")
        assert result["ready"]
        assert result["plan"]["action_source"] == "v21_actor_grounding_corrected"
        _assert_assistance(value, 0)
        assert value.owner.runtime.state is before
        assert value.owner.writes == 0
    assert value.builder_calls == 4
    assert value._assistance.pending_proposal_id is not None
    value.handle("stop")
    _assert_assistance(value, 0)
    assert value._assistance.pending_proposal_id is None


def test_fresh_execute_replan_commits_once_only_after_completed_readback(assisted):
    value = assisted
    value.handle("plan")
    first_time = value.plan_record["source_time_s"]
    proposal_id = value._assistance_proposal_id
    _assert_assistance(value, 0)
    outcome = value.handle("execute")
    assert value.builder_calls == 2  # preview plan + fresh execute-time plan
    assert outcome["last_execution"]["source_time_s"] > first_time
    assert outcome["last_execution"]["assistance_state_committed"] is True
    assert outcome["runtime"]["history_chunks"] == [1]
    assert value.owner.writes == 5
    _assert_assistance(value, 1)
    assert value._assistance.pending_proposal_id is None
    assert not value._assistance.commit_completed(proposal_id, completed=True)
    value.handle("plan")
    _assert_assistance(value, 1)
    value.handle("execute")
    _assert_assistance(value, 2)
    assert value.owner.writes == 10


def test_rejected_plan_rolls_back_helper_and_preserves_actual_history(assisted):
    value = assisted
    before = value.owner.runtime.state
    def reject(q, commands):
        raise RuntimeError("synthetic mechanical rejection")
    value.owner._physical_check = reject
    with pytest.raises(RuntimeError, match="mechanical rejection"):
        value.handle("plan")
    assert value.builder_calls == 1  # rejection after the builder mutated state
    _assert_assistance(value, 0)
    assert value.owner.runtime.state is before
    assert value.owner.writes == 0
    assert value._assistance.pending_proposal_id is None
    assert value.owner.runtime.stopped


def test_partial_execution_cannot_commit_helper_or_fake_completed_history(assisted, monkeypatch):
    value = assisted
    value.handle("plan")
    before_count = value.owner.runtime.state.history.valid.sum().item()
    def partial():
        pending = value.owner.runtime.pending
        value.owner.runtime.begin_execution(pending.source_q, pending.source_time)
        value.owner.writes += 2  # fake IO: two writes then interrupted readback
        value.owner.bridge.last_attempt = dict(completed=False, sent_commands=2)
        raise RuntimeError("synthetic partial readback failure")
    monkeypatch.setattr(value.owner.bridge, "execute", partial)
    with pytest.raises(RuntimeError, match="partial readback"):
        value.handle("execute")
    _assert_assistance(value, 0)
    assert value.owner.writes == 2
    assert value.owner.runtime.state.history.valid.sum().item() == before_count
    assert value.owner.runtime.stopped and value.owner.stop_event.is_set()
    assert value._assistance.pending_proposal_id is None
    assert value.last_attempt["sent_commands"] == 2


def test_explicit_completed_false_does_not_commit_assistance(assisted, monkeypatch):
    value = assisted
    value.handle("plan")
    before = value.owner.runtime.state
    def incomplete():
        q = value.owner.runtime.pending.source_q[0].tolist()
        value.owner.runtime.stop()
        return dict(completed=False, measured_q_path=[q] * 6)
    monkeypatch.setattr(value.owner.bridge, "execute", incomplete)
    outcome = value.handle("execute")
    _assert_assistance(value, 0)
    assert outcome["last_execution"]["assistance_state_committed"] is False
    assert value.owner.runtime.state is before
    assert outcome["runtime"]["history_chunks"] == [0]


def _fake_home(service, monkeypatch, *, fails=False):
    import fcpg.v21_live_service as module
    target = np.full(6, 20., np.float32)
    service.owner.read_checked = lambda: service.owner.q.copy()
    service.owner.io.settle_position_tolerance = 3.
    monkeypatch.setattr(module.established, "_load_v177_home_state", lambda *a, **k: target.copy())
    monkeypatch.setattr(module, "plan_home_chunk", lambda *a: np.repeat(target[None], 5, 0))
    def run(owner, commands, **kwargs):
        if fails:
            raise RuntimeError("synthetic HOME readback failure")
        owner.q = target.copy()
        return dict(completed=True, model_transition=False)
    monkeypatch.setattr(module, "execute_home_readback_chunk", run)


def test_successful_home_resets_committed_and_pending_helper_state(assisted, monkeypatch):
    value = assisted
    value.handle("plan"); value.handle("execute")
    _assert_assistance(value, 1)
    value.handle("plan")  # an unexecuted second candidate is also cleared
    _fake_home(value, monkeypatch)
    outcome = value.handle("home")
    _assert_assistance(value, 0)
    assert value._roll_gain is None and value._roll_prev is None
    assert value._assistance.pending_proposal_id is None
    assert value._assistance_proposal_id is None
    assert outcome["runtime"]["history_chunks"] == [0]
    assert value.owner.writes == 5  # only previous learned fake transaction
    assert value.owner.runtime.stopped


def test_failed_home_preserves_actual_history_but_invalidates_helper_state(assisted, monkeypatch):
    value = assisted
    value.handle("plan"); value.handle("execute")
    before = value.owner.runtime.state
    value.handle("plan")
    _fake_home(value, monkeypatch, fails=True)
    with pytest.raises(RuntimeError, match="HOME readback failure"):
        value.handle("home")
    assert value.owner.runtime.state is before
    # HOME may have moved physically before a readback failure; the old carry
    # and sag helper state is no longer valid, but actual history is not erased.
    _assert_assistance(value, 0)
    assert value._assistance.pending_proposal_id is None
    assert value.owner.runtime.stopped and value.owner.stop_event.is_set()


def test_stop_during_plan_publication_discards_staged_assistance(assisted, monkeypatch):
    value = assisted
    original = value._publish
    def publish_then_stop():
        value.stop()
        original()
    monkeypatch.setattr(value, "_publish", publish_then_stop)
    value.handle("plan")
    _assert_assistance(value, 0)
    assert value._assistance.pending_proposal_id is None
    assert value.owner.runtime.stopped and value.owner.writes == 0


def test_stop_racing_completed_execute_never_commits_cancelled_assistance(assisted, monkeypatch):
    value = assisted
    value.handle("plan")
    original_execute = value.owner.bridge.execute
    executed = threading.Event()
    allow_return = threading.Event()
    errors = []
    def execute_then_wait():
        result = original_execute()  # completed actual history is now committed
        executed.set()
        assert allow_return.wait(3.)
        return result
    monkeypatch.setattr(value.owner.bridge, "execute", execute_then_wait)
    def execute_request():
        try:
            value.handle("execute")
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=execute_request)
    worker.start()
    try:
        assert executed.wait(3.)
        value.handle("stop")  # must remain available without operation lock
    finally:
        allow_return.set()
        worker.join(3.)
    assert not worker.is_alive()
    _assert_assistance(value, 0)
    assert value.owner.runtime.state.history.valid.sum().item() == 1
    assert value.owner.writes == 5
    assert value._assistance.pending_proposal_id is None
    assert value.owner.runtime.stopped and value.owner.stop_event.is_set()
    assert not errors, errors  # completed transaction must not turn into id=None error
