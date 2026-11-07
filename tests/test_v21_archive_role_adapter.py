from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_archive_camera_roles import ReviewedArchiveRoles, RoleAwareClock, research_adapter


def fixture():
    archive = SimpleNamespace(endpoint_episode_index=np.array([0,0,1,1,2,2]), episodes=[
        dict(manifest_row=dict(dataset_root='/recorded', episode_index=100+i,
             lewm_v12_slot_views=['base','left'])) for i in range(3)])
    review = dict(schema='v21-archive-camera-role-review-v1', purpose='diagnostic-only',
        scope='reviewed-canonical-episodes-only', production_enabled=False,
        source_wide_mapping_verified=False, dataset_root='/recorded', historical_views=['left','base'],
        episodes=[dict(canonical_episode=i, source_episode=100+i, upstream='source',
            physical_roles=['external','wrist'] if i==0 else ['wrist','external'],
            permutation=[1,0] if i==0 else [0,1], reviewed_frames=[1,5]) for i in range(2)])
    return archive, review


def test_only_reviewed_reversed_episode_changes_without_mutating_raw_data():
    archive, review = fixture(); roles = ReviewedArchiveRoles(research_adapter(review), archive)
    ids = np.array([[0,1],[2,4]])
    for value in (np.arange(8).reshape(2,2,2), np.arange(48).reshape(2,2,2,6),
                  np.arange(8).reshape(2,2,2)%3==0):
        original = value.copy(); expected = value.copy(); expected[0] = expected[0, :, ::-1]
        assert np.array_equal(roles.remap(value,ids),expected)
        assert torch.equal(roles.remap(torch.as_tensor(value),ids), torch.as_tensor(expected))
        assert np.array_equal(value,original)
    assert roles.coverage(ids) == dict(examples=4,reviewed=3,swapped=2,unverified=1,
                                      unknown_policy='retain-historical-unverified')


def test_identity_control_preserves_tensor_and_raw_order():
    archive, review=fixture(); roles=ReviewedArchiveRoles(research_adapter(review),archive)
    value=torch.randn(2,2,256,384)
    assert roles.remap(value,np.array([2,4])) is value


def test_capture_mask_time_and_q_move_together_fish_stays_identical():
    archive, review=fixture(); roles=ReviewedArchiveRoles(research_adapter(review),archive)
    packets=(np.array([[True,False]]),np.array([[.1,.2]]),np.arange(12).reshape(1,2,6))
    clock=RoleAwareClock(SimpleNamespace(observations=lambda ids,kind:packets),roles)
    result=clock.observations(np.array([0]),'archive')
    for old,new in zip(packets,result):
        assert np.array_equal(old[:,::-1],new)
    assert clock.observations(np.array([0]),'fish') is packets


def test_review_cannot_be_extrapolated_or_rebound_to_another_source():
    archive,review=fixture()
    bad=deepcopy(review);bad['source_wide_mapping_verified']=True
    with pytest.raises(ValueError): research_adapter(bad)
    bad=deepcopy(review);bad['episodes'][0]['source_episode']=999
    with pytest.raises(ValueError): ReviewedArchiveRoles(research_adapter(bad),archive)
    bad=deepcopy(review);bad['episodes'].append(deepcopy(bad['episodes'][0]))
    with pytest.raises(ValueError): ReviewedArchiveRoles(research_adapter(bad),archive)


def test_bad_shape_or_endpoint_identity_is_rejected():
    archive,review=fixture();roles=ReviewedArchiveRoles(research_adapter(review),archive)
    with pytest.raises(ValueError): roles.remap(np.zeros((1,3)),np.array([0]))
    with pytest.raises(ValueError): roles.orders(np.array([-1]))


def test_two_external_cameras_cannot_be_relabelled_as_a_wrist_external_pair():
    archive, review = fixture()
    review['episodes'][0]['physical_roles'] = ['external', 'external']
    for permutation in ([0, 1], [1, 0]):
        review['episodes'][0]['permutation'] = permutation
        with pytest.raises(ValueError, match='physical roles disagree'):
            ReviewedArchiveRoles(research_adapter(review), archive)


def test_archive_feature_gather_uses_same_adapter_and_preserves_fish_helper(monkeypatch):
    from fcpg.v21_data import V21Sources, V201Sources
    archive,review=fixture();roles=ReviewedArchiveRoles(research_adapter(review),archive)
    raw=torch.arange(4).reshape(2,2,1,1);mask=np.array([[True,False],[True,True]])
    monkeypatch.setattr(V201Sources,'_get_archive_dino',lambda self,ids:(raw,mask))
    source=SimpleNamespace(archive_camera_roles=roles)
    value,valid=V21Sources._get_archive_dino(source,np.array([0,2]))
    assert torch.equal(value[0],raw[0].flip(0))
    assert np.array_equal(valid[0],mask[0,::-1])
    assert torch.equal(value[1],raw[1])
