import numpy as np
import pytest
import torch
from fcpg.v21_saved_observation import load_saved_column
from fcpg.v21_core import V21Config,V21ResearchCore
from fcpg.v21_memory import VisualObservation


def saved(path):
    np.savez(path,rgb=np.zeros((2,224,224,3),np.uint8),
        dino_tokens=np.ones((1,2,256,384),np.float16),q_position=np.zeros((1,6),np.float32),
        q_at_image=np.zeros((1,2,6),np.float32),image_times_s=np.asarray([[.8,.9]],np.float64),
        q_time_s=np.asarray([1.],np.float64))


def test_old_column_is_kept_at_old_time_and_not_a_fresh_measurement(tmp_path):
    torch.set_num_threads(2)
    path=tmp_path/'actual.npz';saved(path)
    now=torch.tensor([100.],dtype=torch.float64)
    old=load_saved_column(path,now=now)
    assert old.age_s==pytest.approx(99.1)
    assert old.observation.measurement_available.tolist()==[[False,True]]
    core=V21ResearchCore(V21Config(4)).eval()
    fresh=VisualObservation(torch.full((1,2,256,384),2.),torch.ones(1,2,6),
                            torch.tensor([[99.9,0.]],dtype=torch.float64),torch.tensor([[True,False]]))
    with torch.no_grad():
        past=core.observe(None,old.observation,old.source_q,old.source_time)
        current=core.observe(past,fresh,torch.ones(1,6),now)
    assert current.memory.available.tolist()==[[True,True]]
    assert current.measurement_available.tolist()==[[True,False]]
    assert current.memory.image_times[0,1,-1].item()==pytest.approx(.9)
    assert current.memory.image_times[0,0,-1].item()==pytest.approx(99.9)
    assert current.history.valid.sum()==0 and not current.feedback_available.any()


def test_future_saved_view_or_wrong_shape_is_rejected(tmp_path):
    path=tmp_path/'actual.npz';saved(path)
    with pytest.raises(ValueError,match='future'):
        load_saved_column(path,now=torch.tensor([.5],dtype=torch.float64))
    np.savez(path,q_position=np.zeros(6))
    with pytest.raises(ValueError,match='aligned'):
        load_saved_column(path,now=torch.tensor([2.],dtype=torch.float64))
