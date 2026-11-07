from types import SimpleNamespace
import threading

import numpy as np
import pytest

from fcpg.v21_assistance_state import (
    AssistanceProposalCancelled,
    AssistanceStateCoordinator,
    AssistanceStateError,
)


class FakeOwner:
    def __init__(self):
        self._grounding_bias = np.array([0.0, 0.0])
        self._grounding_prev_end = None
        self._roll_prev = None
        self._roll_gain = None
        self._search = dict(dir=1.0, prev=None, votes=[], blocks=0)
        self._carry = None
        self._home_after_release = False
        self._grounding_vetoes = 0
        # This stands for actual model/visual memory.  The coordinator must not
        # copy, replace or roll it back when assistance planning is speculative.
        self.actual_memory = SimpleNamespace(values=["real-frame-1"])


def mutate_one_step(owner):
    owner._grounding_bias += np.array([1.0, -2.0])
    owner._grounding_prev_end = np.array([3.0, 4.0])
    owner._roll_prev = (-60.0, 12.0)
    owner._roll_gain = 1.7 if owner._roll_gain is None else owner._roll_gain + 1.7
    owner._search["blocks"] += 1
    owner._search["votes"].append(True)
    owner._carry = dict(phase="lift", blocks=1, nested=dict(seen=1))
    owner._home_after_release = True
    owner._grounding_vetoes += 1
    return "commands"


def assert_initial(owner):
    np.testing.assert_array_equal(owner._grounding_bias, [0.0, 0.0])
    assert owner._grounding_prev_end is None
    assert owner._roll_prev is None
    assert owner._roll_gain is None
    assert owner._search == dict(dir=1.0, prev=None, votes=[], blocks=0)
    assert owner._carry is None
    assert owner._home_after_release is False
    assert owner._grounding_vetoes == 0


def test_repeated_plan_does_not_accumulate_without_physical_motion():
    owner = FakeOwner(); memory = owner.actual_memory
    coordinator = AssistanceStateCoordinator(owner)
    first = coordinator.propose(lambda: mutate_one_step(owner))
    assert first.result == "commands"
    assert_initial(owner)
    second = coordinator.propose(lambda: mutate_one_step(owner))
    assert second.proposal_id != first.proposal_id
    assert_initial(owner)
    assert coordinator.pending_proposal_id == second.proposal_id
    assert owner.actual_memory is memory
    assert coordinator.commit_completed(second.proposal_id, completed=True)
    assert owner._search["blocks"] == 1
    assert owner._grounding_vetoes == 1
    assert owner._roll_gain == 1.7
    assert owner.actual_memory is memory


def test_completed_candidate_commits_exactly_once_and_is_deep_copied():
    owner = FakeOwner(); memory = owner.actual_memory
    coordinator = AssistanceStateCoordinator(owner)
    proposal = coordinator.propose(lambda: mutate_one_step(owner))
    assert coordinator.commit_completed(proposal.proposal_id, completed=True)
    np.testing.assert_array_equal(owner._grounding_bias, [1.0, -2.0])
    assert owner._search["blocks"] == 1
    assert owner._carry["nested"]["seen"] == 1
    assert owner._grounding_vetoes == 1
    assert owner._home_after_release is True
    assert coordinator.pending_proposal_id is None
    assert not coordinator.commit_completed(proposal.proposal_id, completed=True)
    assert owner._search["blocks"] == 1
    assert owner.actual_memory is memory


@pytest.mark.parametrize("completed", [False])
def test_partial_or_rejected_execution_discards_candidate(completed):
    owner = FakeOwner(); coordinator = AssistanceStateCoordinator(owner)
    proposal = coordinator.propose(lambda: mutate_one_step(owner))
    assert not coordinator.commit_completed(proposal.proposal_id, completed=completed)
    assert coordinator.pending_proposal_id is None
    assert_initial(owner)


def test_failed_planner_restores_snapshot_and_stages_nothing():
    owner = FakeOwner(); memory = owner.actual_memory
    coordinator = AssistanceStateCoordinator(owner)
    def rejected():
        mutate_one_step(owner)
        raise RuntimeError("mechanical rejection")
    with pytest.raises(RuntimeError, match="mechanical rejection"):
        coordinator.propose(rejected)
    assert coordinator.pending_proposal_id is None
    assert_initial(owner)
    assert owner.actual_memory is memory


def test_cancel_removes_stale_candidate_and_cannot_remove_newer_one():
    owner = FakeOwner(); coordinator = AssistanceStateCoordinator(owner)
    first = coordinator.propose(lambda: mutate_one_step(owner))
    assert coordinator.abort(first.proposal_id)
    assert coordinator.pending_proposal_id is None
    assert not coordinator.commit_completed(first.proposal_id, completed=True)
    second = coordinator.propose(lambda: mutate_one_step(owner))
    with pytest.raises(AssistanceStateError, match="stale proposal"):
        coordinator.commit_completed(first.proposal_id, completed=True)
    assert coordinator.pending_proposal_id == second.proposal_id
    assert coordinator.commit_completed(second.proposal_id, completed=True)


def test_stop_cancels_inflight_plan_and_restores_without_blocking():
    owner = FakeOwner(); memory = owner.actual_memory
    coordinator = AssistanceStateCoordinator(owner)
    entered = threading.Event(); release = threading.Event(); errors = []
    def planner():
        mutate_one_step(owner)
        entered.set()
        assert release.wait(2.0)
        return "commands"
    def run():
        try:
            coordinator.propose(planner)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start(); assert entered.wait(1.0)
    assert coordinator.abort()  # STOP path does not wait for the planner
    release.set(); thread.join(2.0)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], AssistanceProposalCancelled)
    assert coordinator.pending_proposal_id is None
    assert_initial(owner)
    assert owner.actual_memory is memory


def test_reset_is_deferred_safely_during_plan_and_preserves_actual_memory():
    owner = FakeOwner(); memory = owner.actual_memory
    coordinator = AssistanceStateCoordinator(owner)
    committed = coordinator.propose(lambda: mutate_one_step(owner))
    assert coordinator.commit_completed(committed.proposal_id, completed=True)

    entered = threading.Event(); release = threading.Event(); errors = []
    def planner():
        owner._search["blocks"] += 10
        entered.set()
        assert release.wait(2.0)
        return "commands"
    def run():
        try:
            coordinator.propose(planner)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start(); assert entered.wait(1.0)
    assert coordinator.reset() is False
    release.set(); thread.join(2.0)
    assert len(errors) == 1 and isinstance(errors[0], AssistanceProposalCancelled)
    assert_initial(owner)
    assert owner.actual_memory is memory


def test_owner_contract_is_explicit_and_reset_restores_initial_state():
    incomplete = SimpleNamespace(_grounding_bias=np.zeros(2))
    with pytest.raises(AssistanceStateError, match="missing explicit state"):
        AssistanceStateCoordinator(incomplete)

    owner = FakeOwner(); coordinator = AssistanceStateCoordinator(owner)
    proposal = coordinator.propose(lambda: mutate_one_step(owner))
    assert coordinator.commit_completed(proposal.proposal_id, completed=True)
    assert coordinator.reset() is True
    assert_initial(owner)
