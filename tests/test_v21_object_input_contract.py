"""The Actor consumes the exact current fish-body target shown as C."""
from types import SimpleNamespace

import numpy as np
import torch

from fcpg.v21_object_relative import encode_vector, assistance_cue


def test_vector_uses_cog_not_a_different_centroid():
    record=dict(cog_px=[130.,180.],centroid_px=[130.,100.],axis_px=[0.,1.],length_px=100.,width_px=30.,score=.8)
    actual=encode_vector(record,np.array([[80.,150.],[180.,150.]]))
    assert np.isclose(actual[3],30/224)


def test_helper_uses_separate_diagnostic_only_for_a_current_wrist_view():
    observation=SimpleNamespace(measurement_available=torch.tensor([[True,True]]))
    owner=SimpleNamespace(last_endpoint=SimpleNamespace(observation=observation),
        last_object_relative=[1.,.4,0.,-.2,.1,.1,1.,0.],
        last_object_diagnostic=[1.,.8,0.,.2,.4,.2,-1.,0.])
    assert assistance_cue(owner)==owner.last_object_diagnostic
    observation.measurement_available[0,0]=False
    assert not np.asarray(assistance_cue(owner)).any()
    assert owner.last_object_relative[3]==-.2


