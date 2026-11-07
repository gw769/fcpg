"""V20.1 current-observation cache, including the field needed for re-pooling.

Continuous fusion changes component locations/covariance. Cached old feature
integrals alone are therefore insufficient. The frozen eight-channel current
56x56 field is stored explicitly, not silently replaced by stale pooled tokens.
"""
from dataclasses import fields
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .prod_v20_1 import OBSERVATION_CONTRACT
from .prod_v20_1_belief import JointPairBelief
from .prod_v20_1_observer import ObserverOutput, POOL_CHANNELS
from .prod_v20_training import write_json

SHAPES = dict(log_weights=(2,4), state_mean=(2,4,4),
              state_covariance=(2,4,4,4), visible_probability=(2,4,2),
              pooling_map=(2,POOL_CHANNELS,56,56))
if "identity_probability" in JointPairBelief.__dataclass_fields__:
    SHAPES["identity_probability"]=(2,4)


FIELD_DTYPES = {name: np.dtype(np.float16 if name == "pooling_map" else np.float32) for name in SHAPES}


def estimate_cache_bytes(count, field_dtypes=None):
    dtypes = FIELD_DTYPES if field_dtypes is None else field_dtypes
    return int(count)*(sum(int(np.prod(shape))*np.dtype(dtypes[name]).itemsize
                          for name,shape in SHAPES.items())+1)+4096*(len(SHAPES)+2)


def check_cache_space(root,count,reserve_bytes=10*1024**3,field_dtypes=None):
    target=Path(root).resolve()
    existing=target
    while not existing.exists():
        existing=existing.parent
    free=shutil.disk_usage(existing).free
    required=estimate_cache_bytes(count,field_dtypes)
    if free<required+reserve_bytes:
        raise OSError(f"V20.1 cache needs {required} bytes plus {reserve_bytes} reserve; available {free}; no cache created")
    return dict(required_bytes=required,available_bytes=free,reserve_bytes=reserve_bytes)


class RawObservationCache:
    field_dtypes = FIELD_DTYPES
    observation_contract = OBSERVATION_CONTRACT
    measurement_path = "same-RGB-DINO-V20.1-observer"

    def __init__(self,root,count,observer_run_id,*,create=False):
        self.root=Path(root)
        self.count=int(count)
        self.binding=dict(contract=self.observation_contract,observer_run_id=observer_run_id,endpoints=count,
                          measurement_path=self.measurement_path,view_order=["wrist","column"],
                          contents="raw-joint-pair-plus-current-pooling-field",pooling_channels=POOL_CHANNELS)
        config_path=self.root/"config.json"
        exists=config_path.exists()
        if exists:
            config=json.loads(config_path.read_text())
            if any(config.get(k)!=v for k,v in self.binding.items()):
                raise ValueError("V20.1 cache binding differs; never reinterpret a V20 cache")
        elif not create:
            raise FileNotFoundError(config_path)
        else:
            # Tiny test/smoke caches do not require a machine-specific reserve.
            check_cache_space(self.root,count,reserve_bytes=0 if count<1000 else 10*1024**3,
                              field_dtypes=self.field_dtypes)
            self.root.mkdir(parents=True,exist_ok=True)
            write_json(config_path,{**self.binding,"complete":False})
        self.arrays={}
        for name,shape in SHAPES.items():
            path=self.root/f"{name}.npy"
            dtype=self.field_dtypes[name]
            if exists:
                array=np.load(path,mmap_mode="r+" if create else "r",allow_pickle=False)
                if array.shape!=(count,*shape) or array.dtype!=dtype:
                    raise ValueError(f"invalid V20.1 cache {name}")
            else:
                array=np.lib.format.open_memmap(path,mode="w+",dtype=dtype,shape=(count,*shape))
            self.arrays[name]=array
        path=self.root/"written.npy"
        if exists:
            self.written=np.load(path,mmap_mode="r+" if create else "r",allow_pickle=False)
            if self.written.shape!=(count,) or self.written.dtype!=np.bool_:
                raise ValueError("invalid V20.1 completion mask")
        else:
            self.written=np.lib.format.open_memmap(path,mode="w+",dtype=np.bool_,shape=(count,))
            self.written[:]=False
            self.written.flush()

    def _indices(self,indices,unique=False):
        selected=np.asarray(indices,dtype=np.int64)
        if selected.ndim!=1 or not len(selected) or np.any((selected<0)|(selected>=self.count)) or (unique and len(np.unique(selected))!=len(selected)):
            raise ValueError("invalid observation cache indices")
        return selected

    def put(self,indices,measurement):
        selected=self._indices(indices,unique=True)
        measurement.belief.validate()
        prepared={}
        for name,shape in SHAPES.items():
            tensor=measurement.pooling_map if name=="pooling_map" else getattr(measurement.belief,name)
            if tensor.shape!=(len(selected),*shape) or not torch.isfinite(tensor).all():
                raise ValueError(f"invalid current observation {name}")
            transfer_dtype = torch.float64 if self.arrays[name].dtype == np.float64 else torch.float32
            value=tensor.detach().to(dtype=transfer_dtype).cpu().numpy().astype(self.arrays[name].dtype)
            if not np.isfinite(value).all():
                raise ValueError("cache precision overflow")
            prepared[name]=value
        for name,value in prepared.items():
            self.arrays[name][selected]=value
        for array in self.arrays.values():
            array.flush()
        self.written[selected]=True
        self.written.flush()

    def get(self,indices,scene_tokens,*,device,observer=None,view_valid=None,device_cache=None):
        selected=self._indices(indices)
        if not self.written[selected].all():
            raise RuntimeError("raw RGB observation missing; no feature-only substitution")
        if device_cache is None:
            values={name:torch.as_tensor(np.array(array[selected],copy=True),device=device)
                    for name,array in self.arrays.items()}
        else:
            values={name:device_cache.get_rows((str(self.root),self.binding["observer_run_id"],name),selected,
                        lambda ids,array=array:np.array(array[ids],copy=True))
                    for name,array in self.arrays.items()}
        pooling=values.pop("pooling_map").float()
        if not torch.isfinite(pooling).all():
            raise ValueError("nonfinite cached current spatial field")
        belief=JointPairBelief(**values)
        belief.validate()
        if scene_tokens.shape!=(len(selected),2,256,384) or not torch.isfinite(scene_tokens).all():
            raise ValueError("invalid current scene tokens")
        # A frozen learned 8->384 projection must be applied by the observer.
        # Requiring it avoids pretending zeros are actual appearance features.
        if observer is None:
            raise ValueError("V20.1 cache.get requires the frozen observer for exact current re-pooling")
        observation=ObserverOutput(belief,scene_tokens.new_zeros(len(selected),2,4,384),
                                  scene_tokens.detach(),scene_tokens.new_empty(len(selected),2,4,0),pooling)
        if view_valid is None or view_valid.shape!=(len(selected),2):
            raise ValueError("cache re-pooling requires the source view-valid mask")
        valid=torch.as_tensor(view_valid,device=device,dtype=torch.bool)
        return observer.repool(observation,belief,valid)

    def mark_complete(self):
        if not self.written.all():
            raise RuntimeError("cannot finalize partial V20.1 observation cache")
        write_json(self.root/"config.json",{**self.binding,"complete":True})


@torch.inference_mode()
def build_observation_cache(observer,cache,batches,load_observation,*,device,prefetch=False,
                            loaded_batches=None,raw_measurement=False):
    observer.eval().requires_grad_(False)
    def incomplete():
        for selected in batches:
            selected=np.asarray(selected,dtype=np.int64)
            selected=selected[~cache.written[selected]]
            if len(selected):
                yield selected
    def loaded():
        iterator=iter(incomplete())
        if not prefetch:
            for ids in iterator:
                yield ids,load_observation(ids)
            return
        # Only CPU source IO runs here. No mutable GPU LRU/model is shared
        # across threads. One next batch bounds host memory and preserves order.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix="v201-input") as pool:
            current=next(iterator,None)
            if current is None:
                return
            future=pool.submit(load_observation,current)
            for selected in iterator:
                value=future.result()
                future=pool.submit(load_observation,selected)
                yield current,value
                current=selected
            yield current,future.result()
    def transfer(value,dtype=None):
        tensor=torch.as_tensor(value)
        if torch.device(device).type=="cuda" and tensor.device.type=="cpu" and not tensor.is_pinned():
            tensor=tensor.pin_memory()
        return tensor.to(device=device,dtype=dtype,non_blocking=True)
    import time
    started=time.monotonic(); completed=0
    from contextlib import closing
    measurement = observer.measurement_for_cache if raw_measurement else observer.measurement
    with closing(loaded_batches if loaded_batches is not None else loaded()) as iterator:
        for selected,value in iterator:
            rgb=transfer(value["rgb"])
            # Preserve compact host storage across PCIe, then expand on GPU.
            dino=transfer(value["dino"]).to(dtype=torch.float32)
            valid=transfer(value["view_valid"],torch.bool)
            geometry=value.get("geometry")
            prior=geometry.make_prior(torch.as_tensor(value["q_position"],device=device,dtype=torch.float32)) if geometry is not None else None
            cache.put(selected,measurement(rgb,dino,valid,mechanical_prior=prior))
            completed+=len(selected)
            if completed%1024<len(selected):
                print(json.dumps(dict(stage="observation-cache",endpoints_completed_this_run=completed,
                                      endpoints_total=cache.count,elapsed_s=round(time.monotonic()-started,2),
                                      endpoints_per_s=completed/max(time.monotonic()-started,1e-9))),flush=True)
    cache.mark_complete()
