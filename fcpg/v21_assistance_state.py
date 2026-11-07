"""Transactional state for planning-only V21 grasp assistance.

The grounding and carry builders are intentionally stateful: completed robot
motion teaches them servo sag, wrist-roll response, search direction and carry
progress.  Planning, however, is speculative and may run more than once before
one command is physically executed.  This module isolates those speculative
updates and makes a completed real transaction the only commit point.

It does not import the live service, robot adapters, cameras or model runtime.
Only the explicitly listed assistance attributes are copied; model/visual
memory and every unrelated owner attribute remain untouched.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import threading
from typing import Callable, Generic, TypeVar


ResultT = TypeVar("ResultT")


class AssistanceStateError(RuntimeError):
    """The speculative assistance-state lifecycle was used inconsistently."""


class AssistanceProposalCancelled(AssistanceStateError):
    """STOP/reset cancelled a planner while it was producing a proposal."""


@dataclass(frozen=True)
class AssistanceSnapshot:
    """An explicit deep snapshot of every planning-mutated assistance field."""

    grounding_bias: object
    grounding_prev_end: object
    roll_prev: object
    roll_gain: object
    search: object
    carry: object
    home_after_release: object
    grounding_vetoes: object

    @classmethod
    def capture(cls, owner: object) -> "AssistanceSnapshot":
        names = (
            "_grounding_bias",
            "_grounding_prev_end",
            "_roll_prev",
            "_roll_gain",
            "_search",
            "_carry",
            "_home_after_release",
            "_grounding_vetoes",
        )
        missing = [name for name in names if not hasattr(owner, name)]
        if missing:
            raise AssistanceStateError(
                "assistance owner is missing explicit state fields: " + ", ".join(missing)
            )
        return cls(*(deepcopy(getattr(owner, name)) for name in names))

    def restore(self, owner: object) -> None:
        # Copy again so later builder mutations cannot mutate the frozen snapshot
        # through an ndarray/list/dict alias.
        owner._grounding_bias = deepcopy(self.grounding_bias)
        owner._grounding_prev_end = deepcopy(self.grounding_prev_end)
        owner._roll_prev = deepcopy(self.roll_prev)
        owner._roll_gain = deepcopy(self.roll_gain)
        owner._search = deepcopy(self.search)
        owner._carry = deepcopy(self.carry)
        owner._home_after_release = deepcopy(self.home_after_release)
        owner._grounding_vetoes = deepcopy(self.grounding_vetoes)


@dataclass(frozen=True)
class AssistanceProposal(Generic[ResultT]):
    """Planner output paired with the only id allowed to commit its state."""

    proposal_id: int
    result: ResultT


@dataclass(frozen=True)
class _PendingCandidate:
    proposal_id: int
    snapshot: AssistanceSnapshot


class AssistanceStateCoordinator:
    """Snapshot/propose/commit coordinator for one live-service owner.

    ``propose`` always restores the assistance fields before returning, even if
    the planner raises.  A newer proposal replaces an older unexecuted one.
    ``commit_completed`` applies a candidate at most once and only when its
    caller explicitly supplies ``completed=True``.  ``abort`` is suitable for
    preview rejection, failed/partial execution and STOP.  ``reset`` additionally
    returns assistance fields to their construction-time values; it never
    snapshots or restores model/visual memory.
    """

    def __init__(self, owner: object):
        self._owner = owner
        self._initial = AssistanceSnapshot.capture(owner)
        self._lock = threading.RLock()
        self._next_id = 1
        self._cancel_epoch = 0
        self._planning = False
        self._active_id: int | None = None
        self._candidate: _PendingCandidate | None = None
        self._reset_requested = False

    @property
    def pending_proposal_id(self) -> int | None:
        with self._lock:
            return None if self._candidate is None else self._candidate.proposal_id

    @property
    def planning(self) -> bool:
        with self._lock:
            return self._planning

    def propose(self, planner: Callable[[], ResultT]) -> AssistanceProposal[ResultT]:
        """Run one candidate planner without committing any assistance update."""
        if not callable(planner):
            raise TypeError("planner must be callable")
        with self._lock:
            if self._planning:
                raise AssistanceStateError("an assistance proposal is already being planned")
            proposal_id = self._next_id
            self._next_id += 1
            epoch = self._cancel_epoch
            self._candidate = None  # repeated plan invalidates the earlier candidate
            self._planning = True
            self._active_id = proposal_id

        baseline = AssistanceSnapshot.capture(self._owner)
        candidate: AssistanceSnapshot | None = None
        succeeded = False
        cancelled = False
        reset_requested = False
        try:
            result = planner()
            candidate = AssistanceSnapshot.capture(self._owner)
            succeeded = True
        finally:
            try:
                baseline.restore(self._owner)
            finally:
                with self._lock:
                    cancelled = self._cancel_epoch != epoch
                    reset_requested = self._reset_requested
                    self._planning = False
                    self._active_id = None
                    if reset_requested:
                        self._initial.restore(self._owner)
                        self._reset_requested = False
                    if succeeded and not cancelled and not reset_requested:
                        self._candidate = _PendingCandidate(proposal_id, candidate)

        if cancelled or reset_requested:
            raise AssistanceProposalCancelled(
                f"assistance proposal {proposal_id} was cancelled before it could be staged"
            )
        return AssistanceProposal(proposal_id, result)

    def commit_completed(self, proposal_id: int, *, completed: bool) -> bool:
        """Commit one matching proposal iff a full real transaction completed.

        ``completed=False`` deliberately discards the matching candidate.  A
        second commit of an already consumed proposal is an idempotent ``False``;
        trying to commit an older id while a newer proposal is pending fails
        closed and leaves the newer proposal intact.
        """
        if type(proposal_id) is not int or proposal_id < 1:
            raise AssistanceStateError("proposal id must be a positive integer")
        if type(completed) is not bool:
            raise AssistanceStateError("completed must be an explicit bool")
        with self._lock:
            if self._planning:
                raise AssistanceStateError("cannot commit while candidate planning is active")
            candidate = self._candidate
            if candidate is None:
                return False
            if candidate.proposal_id != proposal_id:
                raise AssistanceStateError(
                    f"stale proposal {proposal_id}; pending proposal is {candidate.proposal_id}"
                )
            self._candidate = None
            if not completed:
                return False
            candidate.snapshot.restore(self._owner)
            return True

    def abort(self, proposal_id: int | None = None) -> bool:
        """Discard a staged proposal or cancel an in-flight one without state change."""
        if proposal_id is not None and (type(proposal_id) is not int or proposal_id < 1):
            raise AssistanceStateError("proposal id must be a positive integer")
        with self._lock:
            matched = proposal_id is None
            if self._planning and (proposal_id is None or proposal_id == self._active_id):
                self._cancel_epoch += 1
                matched = True
            if self._candidate is not None and (
                    proposal_id is None or proposal_id == self._candidate.proposal_id):
                self._candidate = None
                matched = True
            return matched

    def reset(self) -> bool:
        """Cancel pending work and restore construction-time assistance defaults.

        Returns ``True`` when restoration was immediate.  If a planner is still
        running, reset is marked for its ``finally`` block and returns ``False``;
        this keeps STOP/reset non-blocking while guaranteeing eventual restore.
        """
        with self._lock:
            self._cancel_epoch += 1
            self._candidate = None
            if self._planning:
                self._reset_requested = True
                return False
            self._initial.restore(self._owner)
            self._reset_requested = False
            return True


__all__ = [
    "AssistanceProposal",
    "AssistanceProposalCancelled",
    "AssistanceSnapshot",
    "AssistanceStateCoordinator",
    "AssistanceStateError",
]
