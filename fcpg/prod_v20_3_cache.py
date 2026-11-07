"""V20.3 probability precision is retained across storage and GPU staging.

Only the tiny paired-belief fields use float64; the large learned current
pooling field and frozen DINO features retain their original float16 storage.
The explicit new binding rejects legacy float32 belief caches instead of
claiming an upcast recovers covariance information already rounded away.
"""
import numpy as np

from .prod_v20_1_cache import (RawObservationCache as ParentCache, SHAPES,
                              build_observation_cache as _build, estimate_cache_bytes as _estimate)
from .prod_v20_3_observation_io import iter_observation_batches

FIELD_DTYPES = {name: np.dtype(np.float16 if name == "pooling_map" else np.float64) for name in SHAPES}


class RawObservationCache(ParentCache):
    field_dtypes = FIELD_DTYPES
    observation_contract = "proprioceptive-pair-raw-fp64-v1"
    measurement_path = "same-RGB-DINO-V20.3-observer"


def estimate_cache_bytes(count):
    return _estimate(count, FIELD_DTYPES)


def build_observation_cache(observer, cache, batches, load_observation, *, device,
                            prefetch=False, cpu_superbatch=None, raw_measurement=False):
    loaded = (iter_observation_batches(batches, load_observation, written=cache.written,
              superbatch_size=cpu_superbatch, prefetch=prefetch)
              if cpu_superbatch is not None else None)
    return _build(observer, cache, batches, load_observation, device=device, prefetch=prefetch,
                  loaded_batches=loaded, raw_measurement=raw_measurement)
