"""Mixed-locality Actor sampling with complete, deterministic pass coverage.

Rows are still shuffled, but within a bounded set of independently shuffled
contiguous source blocks. This intentionally changes the old global-permutation
training order: it is an I/O-aware schedule, not a bitwise-equivalent optimizer
trajectory. Model inputs, per-pass population, and effective batch size are
unchanged. A new Actor run and any matched control must use the same schedule.

The iterator yields CPU row-ID windows, not tensors or observations. Consumers
may preload the unique endpoints for a window in physical order, then consume
these IDs in the exact yielded order. No global RNG or GPU state is touched.
"""
from __future__ import annotations

import operator
from collections.abc import Iterator

import numpy as np


def _integer(name: str, value, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def deterministic_mixed_locality_windows(
    population: int,
    passes: int,
    batch: int,
    seed: int,
    block_rows: int = 2048,
    blocks_per_window: int = 8,
) -> Iterator[np.ndarray]:
    """Yield shuffled, bounded windows covering each row once in every pass.

    Each pass shuffles contiguous block identities, groups up to
    ``blocks_per_window`` blocks, and shuffles the rows within each group.
    Iterating ``window[start:start + batch]`` produces exactly
    ``ceil(population / batch)`` batches per pass, including arbitrary small
    populations and non-divisible block sizes. Only the final window of a pass
    may end in a short batch; an earlier remainder is carried into the next
    window before its row shuffle. No carry crosses a pass boundary.

    Memory contains at most ``block_rows * blocks_per_window + batch - 1`` row
    IDs in one assembled window. With the production 2048-row blocks and a
    64-chain batch, no remainder occurs and windows contain at most 16384 rows.
    Restart/resume must recreate this same iterator and skip already-completed
    batches without materializing their image data.
    """
    population = _integer("population", population, minimum=0)
    passes = _integer("passes", passes, minimum=0)
    batch = _integer("batch", batch, minimum=1)
    seed = _integer("seed", seed, minimum=0)
    block_rows = _integer("block_rows", block_rows, minimum=1)
    blocks_per_window = _integer("blocks_per_window", blocks_per_window, minimum=1)
    if population == 0 or passes == 0:
        return

    rng = np.random.default_rng(seed)
    block_count = (population + block_rows - 1) // block_rows
    for _ in range(passes):
        block_order = rng.permutation(block_count)
        carry = np.empty(0, dtype=np.int64)
        for begin in range(0, block_count, blocks_per_window):
            blocks = block_order[begin:begin + blocks_per_window]
            parts = [carry] if len(carry) else []
            for block in blocks:
                start = int(block) * block_rows
                parts.append(np.arange(start, min(start + block_rows, population), dtype=np.int64))
            window = np.concatenate(parts)
            rng.shuffle(window)
            if begin + blocks_per_window >= block_count:
                # One and only one possible short batch, at this pass's end.
                yield window
                carry = np.empty(0, dtype=np.int64)
            else:
                complete = (len(window) // batch) * batch
                # Own the small remainder so it cannot retain a full window.
                carry = window[complete:].copy()
                if complete:
                    yield window[:complete]


__all__ = ["deterministic_mixed_locality_windows"]
