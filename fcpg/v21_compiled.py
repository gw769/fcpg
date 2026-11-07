"""Compiled regions of a training update (docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md, section 17.84).

On the A100 about seven tenths of an Actor update's GPU time went to small steps that wait for memory, not for the
compute units: casts between bfloat16 and float32, LayerNorm and its backward pass, element-wise steps (section 17.83).
torch.compile (Inductor) merges such steps into a few larger GPU programs. As PyTorch's regional compilation recipe and
Hugging Face Accelerate's ``compile_regions`` do, only self-contained pieces are compiled (the FCPG read, the Actor's
layers, the memory shift, the picture checks' reductions), not the whole update with its Python bookkeeping.

``region(fn)`` runs ``fn`` itself unless the calling thread is inside ``regions()``; there it runs ``fn`` compiled.
The Actor trainer turns the regions on only around a training update's forward and backward passes. Evaluations,
rehearsals and the live service never do, so every number they report is computed exactly as before. Inside, values
agree with ``fn``'s up to the order of floating-point operations (checked against a float64 reference with PyTorch's
own rule, ``torch._dynamo.utils.same``), and dropout draws its masks from Inductor's random stream."""
import functools
import threading

import torch

CACHE_SIZE_LIMIT = 64       # compiled variants kept per function (torch's default 8: shapes x layers x train/no-grad)

_state = threading.local()
_compiled = []              # every region's compiled function, once made (reset() forgets them)


def active():
    """True inside regions() on this thread."""
    return getattr(_state, 'on', False)


class regions:
    """Turn the compiled regions on (or off) for the calling thread while inside."""
    def __init__(self, on=True):
        self.on = bool(on)

    def __enter__(self):
        self.before = active()
        _state.on = self.on
        return self

    def __exit__(self, *exc):
        _state.on = self.before
        return False


def region(fn):
    """fn, compiled the first time it is called inside regions(); fn itself everywhere else. Shapes are specialised
    as torch.compile does by default: the first call's shapes are fixed, a dimension that changes later (the text
    length, the last batch, the fish stage's batches) is compiled once more as a dynamic dimension."""
    compiled = []
    _compiled.append(compiled)

    @functools.wraps(fn)
    def call(*args, **kwargs):
        if not active():
            return fn(*args, **kwargs)
        if not compiled:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, CACHE_SIZE_LIMIT)
            compiled.append(torch.compile(fn))
        return compiled[0](*args, **kwargs)

    call.plain = fn
    return call


def reset():
    """Forget every region's compiled function (tests; the next call inside regions() compiles again)."""
    for compiled in _compiled:
        compiled.clear()
    torch._dynamo.reset()


def start_compile_workers():
    """Start Inductor's compile worker processes now. The trainer calls this before it pins its training thread to
    two fast cores (section 17.82): processes started later from that thread would inherit only those two cores."""
    from torch._inductor.async_compile import AsyncCompile
    AsyncCompile.warm_pool()
