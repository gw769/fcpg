"""Wrist display validity only; no models, cameras, serial ports, or services."""
from copy import deepcopy
import json
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

from fcpg.v21_live_service import V21LiveService


@pytest.fixture
def recorded():
    path=Path(__file__).parent/'fixtures'/'wrist_clipped_overlay_20260920.json'
    return json.loads(path.read_text())


@pytest.fixture
def service():
    value=V21LiveService.__new__(V21LiveService)
    value.head_grasp=False
    value.grasp_target='body'
    value._draw_cog_caption=lambda image,*args:image
    return value


def valid_overlay(**updates):
    value=dict(source='operator_colour_body',cog_px=[80.,80.],axis_px=[0.,1.],
        contour_px=[[70.,65.],[90.,65.],[90.,95.],[70.,95.]],length_px=30.,width_px=20.,
        jaw_pair_px=[[95.,140.],[205.,140.]],vector=[1.,.8,0.,0.,.2,.1,1.,0.])
    value.update(updates)
    return value


def test_recorded_clipped_diagnostic_does_not_draw_any_target_geometry(service,recorded):
    overlay=recorded['object_overlay']
    assert overlay['source']=='frozen_symmetric_support'
    assert overlay['support_clipped'] and overlay['aimable'] is False
    assert overlay['policy_token_available'] is False and overlay['vector']==[0.]*8
    before=deepcopy(overlay)
    image=np.full((224,224,3),45,np.uint8)
    np.testing.assert_array_equal(service._draw_object_geometry(image.copy(),overlay=overlay),image)
    assert overlay==before


@pytest.mark.parametrize('updates',[
    dict(policy_token_available=False),
    dict(vector=[0.]*8),
    dict(source='frozen_symmetric_support',support_clipped=True),
    dict(source='frozen_vit_features',aimable=False),
    dict(visible=False),
    dict(vector=[float('nan')]*8),
    dict(vector=[]),
    dict(vector=['invalid']*8),
])
def test_explicit_unavailable_or_invalid_cue_suppresses_contour_axes_c_and_cm(service,updates):
    image=np.zeros((224,224,3),np.uint8)
    actual=service._draw_object_geometry(image.copy(),overlay=valid_overlay(**updates))
    np.testing.assert_array_equal(actual,image)


@pytest.mark.parametrize('updates',[
    {},
    dict(body_truncated=True,support_clipped=True,aimable=None,policy_token_available=True),
    dict(vector=None),
    dict(vector=None,body_truncated=True,support_clipped=True,aimable=None),
    dict(source='frozen_symmetric_support',vector=None,support_clipped=False,aimable=True),
])
def test_valid_body_and_legacy_source_contracts_keep_target_geometry(service,updates):
    overlay=valid_overlay(**updates);before=deepcopy(overlay)
    image=service._draw_object_geometry(np.zeros((224,224,3),np.uint8),overlay=overlay)
    assert image[80,80].any()  # C
    assert image[69,80].any()  # long axis
    assert image[80,68].any()  # grasp axis
    assert image[110,115].any()  # C to FK midpoint
    assert overlay==before


def test_invalid_endpoint_overlay_preserves_independent_fk_tips_and_column(service,recorded):
    service.closing=threading.Event()
    service.display_lock=threading.Lock()
    image=np.full((224,224,3),45,np.uint8)
    image[:28]=0;image[196:]=0
    service.snapshot=dict(sequence=4,rgb={'wrist':image,'column':image},targets={'targets':{}},
        fingertips=deepcopy(recorded['fingertips']),object_overlay=deepcopy(recorded['object_overlay']),
        head_targets={},view_sources={'wrist':'current_observation','column':'current_observation'},
        image_times=recorded['image_times'],measurement_available=[[True,True]])
    before=deepcopy(service.snapshot['fingertips'])
    wrist=service.frame('wrist','endpoint',4)
    column=service.frame('column','endpoint',4)
    service.snapshot['object_overlay']=None
    assert service.frame('wrist','endpoint',4)==wrist
    assert service.frame('column','endpoint',4)==column
    assert service.snapshot['fingertips']==before
    decoded=cv2.imdecode(np.frombuffer(wrist,np.uint8),cv2.IMREAD_COLOR)
    for xy in before['wrist']['fm_px']:
        x,y=np.rint(xy).astype(int)
        assert decoded[y-2:y+3,x-2:x+3].max()>150
    assert before['wrist']['source']=='wrist-fk-projection'
    assert before['column']['source']=='column-rgb-jaw-contour'


def test_head_display_keeps_its_separate_path(service):
    service.head_grasp=True
    image=np.zeros((224,224,3),np.uint8)
    np.testing.assert_array_equal(service._draw_object_geometry(image.copy(),overlay=valid_overlay()),image)
