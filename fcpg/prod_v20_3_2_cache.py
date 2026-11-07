"""V20.3.2 raw current-image cache.

The tensor layout stays compact and compatible with the existing endpoint LRU,
but the binding is deliberately new: a V20.3 posterior cache must never be
silently interpreted as a V20.3.2 raw field.
"""
import numpy as np

from .prod_v20_3_cache import RawObservationCache as _Raw, build_observation_cache as _build


class RawObservationCache(_Raw):
    observation_contract = "proprioceptive-pair-raw-fp64-v2-dual-view-attention"
    measurement_path = "same-RGB-DINO-V20.3.2-raw-field"


def build_observation_cache(*args, **kwargs):
    return _build(*args, **kwargs)


def estimate_cache_bytes(count):
    from .prod_v20_3_cache import estimate_cache_bytes as estimate
    return estimate(count)


__all__ = ["RawObservationCache", "build_observation_cache", "estimate_cache_bytes"]
