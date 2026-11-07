from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from scripts.serve_so101_realtime_fcpg_lewm import configured_camera_views
from scripts.canary_so101_v21_research import TrialOwner, cpu_actual_state
from fcpg.v21_core import V21Config,V21ResearchCore
from fcpg.v21_memory import VisualObservation
from fcpg.v21_runtime import V21ResearchRuntime,EncodedEndpoint


def test_default_camera_contract_is_unchanged_and_partial_mode_is_explicit():
    assert configured_camera_views(SimpleNamespace())==('wrist','column')
    assert configured_camera_views(SimpleNamespace(active_camera_views=('wrist',)))==('wrist',)
    for value in ((),('column',),('column','wrist'),'wrist'):
        with pytest.raises(ValueError):configured_camera_views(SimpleNamespace(active_camera_views=value))


def test_cached_column_is_never_recaptured_or_retimestamped(monkeypatch):
    import scripts.canary_so101_v21_research as module
    torch.set_num_threads(2)
    owner=TrialOwner.__new__(TrialOwner)
    owner.device=torch.device('cpu');owner.active_views=('wrist',);owner.acquisitions=[];owner.frames={}
    owner._lease_check=lambda:None;owner.read_checked=lambda:np.zeros(6,np.float32);owner.dino=object()
    reads=[]
    def camera(view):
        assert view=='wrist';reads.append(view);return np.zeros((224,224,3),np.uint8)
    owner.io=SimpleNamespace(camera_lock=threading.Lock(),_read_camera=camera,_last_raw_frames={})
    def dino(_backbone,rgb,**kwargs):
        assert len(rgb)==1
        return np.full((1,256,384),2.,np.float16)
    monkeypatch.setattr(module,'extract_dinov2_tokens',dino)
    core=V21ResearchCore(V21Config(4)).eval()
    old=VisualObservation(torch.ones(1,2,256,384),torch.zeros(1,2,6),
        torch.tensor([[0.,1.]],dtype=torch.float64),torch.tensor([[False,True]]))
    with torch.no_grad():
        state=core.observe(None,old,torch.zeros(1,6),torch.tensor([1.1],dtype=torch.float64))
        new=owner.capture()
        state=core.observe(state,new.observation,new.q,new.time)
    assert reads==['wrist']
    assert new.observation.measurement_available.tolist()==[[True,False]]
    assert new.observation.features[0,1].count_nonzero()==0
    assert state.memory.available.tolist()==[[True,True]]
    assert state.memory.image_times[0,1,-1]==1.
    assert state.memory.features[0,1,-1].eq(1.).all()
    assert state.history.valid.sum()==0
    saved=cpu_actual_state(state)
    assert saved.memory.features.device.type=='cpu'
    assert saved.memory.features.data_ptr()!=state.memory.features.data_ptr()
