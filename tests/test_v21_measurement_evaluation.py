from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from fcpg import v21_clock_evaluation as evaluation
from fcpg.v21_memory import VisualMemory, VisualObservation
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_training import synthetic_chain


def test_evaluator_masks_only_inputs_and_scores_real_dense40_even_for_cold_camera(monkeypatch):
    torch.set_num_threads(2)
    chain=synthetic_chain(TaskVocabulary.build(['pick']),batch=2)
    chain=replace(chain,capture_times=chain.endpoint_times[:,:,None].expand(-1,-1,2),
                  capture_q=chain.q[:,:,None].expand(-1,-1,2,-1))
    calls=[]
    class Bank:
        def load(self,rows,kind,context_frames):
            return chain.features[:,:context_frames],chain.view_valid[:,:context_frames].numpy()
    class Targets:
        def __init__(self,*args,supervision):
            assert supervision=='dense40'
        def load(self,rows,kind,h,device):
            assert calls[-1]=='factual'
            return SimpleNamespace(features=torch.ones(2,40,2,1,1),valid=torch.ones(2,40,2,dtype=torch.bool),
                                   query_indices=torch.arange(40))
        query_indices=torch.arange(40)
        def identities(self,rows,kind,h): return np.broadcast_to(np.arange(40),(2,40))
    class QTargets:
        trace=chain.traces[0]
        readback_offsets=torch.arange(1,41,dtype=torch.float64)[None].expand(2,-1)*.1
        readbacks=torch.zeros(2,40,6)
        valid=torch.ones(2,40,dtype=torch.bool)
        episode_ids=np.full(2,3)
        def to(self,device): return self
    class Labels:
        def gather(self,ids):
            return dict(fg_px=np.ones((len(ids),2,2,2),np.float32)*10,
                pair_available=np.ones((len(ids),2),bool),current_image_valid=np.ones((len(ids),2),bool),
                ordered_identity_available=np.ones((len(ids),2),bool))
    def factual(core,masked,h,images):
        assert masked.q is chain.q and masked.capture_q is chain.capture_q
        assert masked.capture_times is chain.capture_times
        memory=VisualMemory.empty(2)
        for t in range(h+1):
            memory,_=memory.write(VisualObservation(masked.features[:,t],masked.capture_q[:,t],
                masked.capture_times[:,t],masked.view_valid[:,t]),masked.endpoint_times[:,t])
        calls.append('factual')
        return SimpleNamespace(q=chain.q[:,h],time=chain.endpoint_times[:,h],history=None,memory=memory)
    class LeWM:
        def scene_encoder(self,features,valid): return torch.ones(*valid.shape,384)
        def fg_readout(self,z): return torch.zeros(*z.shape[:-2],2,2,2)
        def __call__(self,q,trace,history,offsets,**kwargs):
            memory=kwargs['memory']
            return SimpleNamespace(predicted_q_path=torch.ones(2,41,6),
                predicted_scene=torch.zeros(2,40,2,384),observation_scene=torch.ones(2,2,3,384)*.5,
                observation_valid=memory.valid)
    monkeypatch.setattr(evaluation,'FutureSceneTargets',Targets)
    monkeypatch.setattr(evaluation,'factual_state',factual)
    monkeypatch.setattr(evaluation,'horizon_targets',lambda *args:QTargets())
    # Device staging is orthogonal here; its real implementation has independent tests.
    import scripts.train_v21_from_scratch as staging
    monkeypatch.setattr(staging,'stage_chain_to_device',lambda value,device:value)
    source=SimpleNamespace(chain=lambda *args,**kw:chain)
    rows={'fish':np.array([6,6])}  # cold Column in the deterministic protocol
    core=SimpleNamespace(lewm=LeWM())
    factual_result=evaluation.evaluate_core(core,source,Bank(),rows,device=torch.device('cpu'),
        precision='fp32',scene_supervision='dense40',fg_table=Labels())['fish']
    masked_result=evaluation.evaluate_core(core,source,Bank(),rows,device=torch.device('cpu'),
        precision='fp32',scene_supervision='dense40',fg_table=Labels(),
        measurement_protocol='burst-cold-v1')['fish']
    assert factual_result['coverage']['future_view_slots']==masked_result['coverage']['future_view_slots']==[80,80]
    assert masked_result['views']['column']['targets']==0
    assert masked_result['views']['column']['relative_scene_error'] is None
    assert masked_result['fg_forecast']['full40']['column']['pairs']==80
    assert masked_result['fg_forecast']==factual_result['fg_forecast']
    assert masked_result['valid_joint_values']==factual_result['valid_joint_values']==[60,480]
    assert masked_result['measurement_input']['cold_latest_views']==[0,2]
