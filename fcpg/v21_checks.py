"""Section 17.80: input checks that do not stop the GPU.

The FCPG hot path checks its inputs at almost every call (finite values, clocks, command traces, history, readback
progress). Each check read a GPU value on the CPU, so one training update of the Actor trainer made the CPU wait for
the GPU about 280 times and the A100 sat idle in between (measured 10/23 on the V100: 3,700 GPU kernels and 280 waits
per update). ``fail_if(condition, message)`` raises at once, as every check did before, unless the calling thread is
inside ``deferred()``: then the condition (a GPU bool) is kept, and the caller reads all of them at once later and
raises the first failed check's error. ``later(values, use)`` does the same for numbers that are only counted.
Nothing computed changes; only when an error is raised and when a count is read.
11/02 (17.171, James 「不要什么安全检查了 不要了」): inside ``unchecked()`` (the LeWM trainer's updates) the checks are
not made at all: ``fail_if`` returns at once and its call sites skip computing their conditions (``checking()``). Every
other caller (the robot service, the exams, the tests) checks as before.
"""
import threading
from contextlib import contextmanager

import torch

_state = threading.local()


def deferring():
    return getattr(_state, 'pending', None) is not None


def checking():
    """False inside unchecked(): this thread's GPU checks are not made."""
    return not getattr(_state, 'unchecked', False)


@contextmanager
def unchecked():
    """Inside, this thread's fail_if() checks are skipped, and their call sites do not compute them (checking())."""
    outer = getattr(_state, 'unchecked', False)
    _state.unchecked = True
    try:
        yield
    finally:
        _state.unchecked = outer


def fail_if(condition, message, error=ValueError):
    """Raise error(message) if the bool tensor ``condition`` is true: now, or (inside deferred()) when read; never
    inside unchecked()."""
    if not checking():
        return
    pending = getattr(_state, 'pending', None)
    if pending is not None and isinstance(condition, torch.Tensor) and condition.is_cuda:
        pending.checks.append((condition, message, error))
    elif bool(condition):
        raise error(message)


def later(values, use):
    """use(list of floats) with the values of the 1-D tensor ``values``: now, or (inside deferred()) when read."""
    pending = getattr(_state, 'pending', None)
    if pending is not None and values.is_cuda:
        pending.counts.append((values, use))
    else:
        use(values.double().tolist())


class Pending:
    """The checks and counts of one deferred() block, copied to the CPU without waiting (start) and read (finish)."""

    def __init__(self):
        self.checks, self.counts = [], []
        self._copy = self._done = None

    def start(self, *extra):
        """Queue one copy of every flag, count and extra value (1-D float64 GPU tensors) to page-locked memory."""
        parts = [torch.stack([c.reshape(()) for c, _, _ in self.checks]).double()] if self.checks else []
        parts += [v.reshape(-1).double() for v, _ in self.counts] + [e.reshape(-1).double() for e in extra]
        if not parts:
            return
        values = torch.cat(parts)
        self._copy = torch.empty(values.shape, dtype=values.dtype, pin_memory=True)
        self._copy.copy_(values, non_blocking=True)
        self._done = torch.cuda.Event()
        self._done.record()

    def finish(self, extra_sizes=()):
        """Wait for the copy, raise the first failed check's error, hand out the counts; returns the extra values."""
        if self._copy is None:
            return [[] for _ in extra_sizes]
        self._done.synchronize()
        values = self._copy.tolist()
        for failed, (_, message, error) in zip(values, self.checks):
            if failed:
                raise error(message)
        at = len(self.checks)
        for counted, use in self.counts:
            n = counted.numel()
            use(values[at:at + n])
            at += n
        extra = []
        for n in extra_sizes:
            extra.append(values[at:at + n])
            at += n
        return extra


@contextmanager
def deferred(pending):
    """Inside, this thread's GPU checks and counts go into ``pending`` (a Pending) instead of being read."""
    outer = getattr(_state, 'pending', None)
    _state.pending = pending
    try:
        yield pending
    finally:
        _state.pending = outer
