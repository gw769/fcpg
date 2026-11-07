"""Bounded source-video reads without changing observer GPU batch boundaries.

Eight decoders cannot work concurrently when a 128-endpoint batch contains
only one episode. Adjacent *input batches* are loaded together, then sliced
back into their original filtered batches. No endpoints are shuffled, padded,
duplicated, or dropped, and the observer still receives the same batch shape.

There are at most two loaded superbatch payloads: the one being consumed and
one next payload. The loader must be CPU-only and must not use a mutable GPU
cache. Geometry objects are shared constants by default (as for fish geometry);
batch-dependent geometry must supply ``geometry_slicer`` explicitly.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch


def _superbatches(batches, written, limit):
    iterator = iter(batches)
    pending = []
    count = 0
    try:
        for raw in iterator:
            ids = np.asarray(raw)
            if ids.ndim != 1 or ids.dtype.kind not in "iu":
                raise ValueError("observation batch indices must be an integer vector")
            if np.any(ids < 0) or np.any(ids >= len(written)):
                raise ValueError("observation batch indices are out of range")
            ids = ids.astype(np.int64, copy=True)
            ids = ids[~written[ids]]
            if not len(ids):
                continue
            if len(np.unique(ids)) != len(ids):
                raise ValueError("observation batch indices must be unique within a batch")
            if len(ids) > limit:
                raise ValueError("superbatch_size must fit each original observation batch")
            if count + len(ids) > limit:
                yield pending
                pending, count = [], 0
            pending.append(ids)
            count += len(ids)
            if count == limit:
                yield pending
                pending, count = [], 0
        if pending:
            yield pending
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def _validate_payload(value, count):
    if not isinstance(value, Mapping):
        raise TypeError("observation loader must return a mapping")
    for name, item in value.items():
        if name == "geometry":
            continue
        if not isinstance(item, (np.ndarray, torch.Tensor)):
            raise TypeError(f"observation {name} must be an endpoint-aligned array or tensor")
        if item.ndim == 0 or item.shape[0] != count:
            raise ValueError(f"observation {name} does not match loaded endpoint count")
        if isinstance(item, torch.Tensor) and item.device.type != "cpu":
            raise ValueError("observation prefetch loader must return CPU tensors only")
    return value


def _slices(group, value, geometry_slicer):
    begin = 0
    for ids in group:
        end = begin + len(ids)
        selection = slice(begin, end)
        part = {}
        for name, item in value.items():
            if name == "geometry":
                part[name] = (geometry_slicer(item, selection)
                              if item is not None and geometry_slicer is not None else item)
            else:
                part[name] = item[selection]
        yield ids, part
        begin = end


def iter_observation_batches(
    batches: Iterable[Any],
    load_observation: Callable[[np.ndarray], Mapping[str, Any]],
    *,
    written: np.ndarray,
    superbatch_size: int = 1024,
    prefetch: bool = True,
    geometry_slicer: Callable[[Any, slice], Any] | None = None,
) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    """Return original GPU batches from larger, ordered CPU source reads.

    ``written`` is the ordinary endpoint completion bitmap. Filtering occurs
    before loading. The surviving IDs in each original batch remain together;
    a partly completed batch is not refilled from its neighbours. There is no
    batch-size-dependent resampling or floating-point conversion here.

    Closing the returned generator cancels queued work and joins the one
    prefetch thread. An already-running source read must finish or reach its
    own timeout. The helper does not close the caller-owned source decoder.
    """
    written = np.asarray(written)
    if written.ndim != 1 or written.dtype != np.bool_:
        raise ValueError("written must be a one-dimensional boolean completion bitmap")
    if isinstance(superbatch_size, bool) or not isinstance(superbatch_size, (int, np.integer)) or superbatch_size <= 0:
        raise ValueError("superbatch_size must be a positive integer")
    groups = _superbatches(batches, written, int(superbatch_size))

    def load(group):
        ids = np.concatenate(group)
        return _validate_payload(load_observation(ids), len(ids))

    pool = None
    future = None
    try:
        if not prefetch:
            for group in groups:
                value = load(group)
                yield from _slices(group, value, geometry_slicer)
            return
        group = next(groups, None)
        if group is None:
            return
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v203-observation-input")
        future = pool.submit(load, group)
        while group is not None:
            value = future.result()
            following = next(groups, None)
            future = pool.submit(load, following) if following is not None else None
            yield from _slices(group, value, geometry_slicer)
            # Do not retain the preceding payload while a third is submitted.
            del value
            group = following
    finally:
        if future is not None:
            future.cancel()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        groups.close()
