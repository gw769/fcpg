import numpy as np
import pytest

from fcpg import v21_cache_rebuild as rebuild
from fcpg.lerobot_archive import letterbox_rgb
from fcpg.v21_camera_quality import (raw_camera_quality, reviewed_exposure_mask,
    camera_quality_job, REVIEWED_CAMERA_QUALITY_VERSION, REVIEW_SCHEMA)


@pytest.mark.parametrize('kind', ['gray-band', 'uniform-gray'])
def test_opt_in_raw_quality_rejects_gray_fill_before_letterbox(monkeypatch, kind):
    rng=np.random.default_rng(17)
    good=rng.integers(20,230,(480,640,3),dtype=np.uint8)
    bad=good.copy()
    bad[80 if kind=='gray-band' else 0:]=128
    frames=np.stack([letterbox_rgb(x[...,::-1]) for x in (good,bad)])
    # Legacy whole-letterbox range cannot detect this partly/fully gray content.
    assert np.ptp(frames[1]) > 2
    record=dict(time=np.array([0.,1/30]),q=np.zeros((2,6)),action=np.zeros((2,6)),
        masks=np.ones((2,2),bool),mask_present=np.ones((2,2),bool))
    monkeypatch.setattr(rebuild,'read_recording',lambda *a:record)
    monkeypatch.setattr(rebuild,'resolve_media',lambda *a:dict(path='video',requested_pts=record['time'],
        start=0.,stop=1.,clock_origin=0.,provenance='test',declared_video_fps=30))
    def decode(*args,raw_rgb_callback=None):
        if raw_rgb_callback:
            for i,image in enumerate((good,bad)): raw_rgb_callback(i,image)
        return frames.copy(),record['time'].copy(),np.ones(2,bool),{}
    monkeypatch.setattr(rebuild,'decode_at_recorded_times',decode)
    job=dict(identity=0,row={},views=['wrist','column'],local=[0,1],endpoint_ids=[0,1])
    assert rebuild.episode_job(job)['valid'].all()  # Old generation remains reproducible.
    actual=rebuild.episode_job(dict(job,camera_quality_version='raw-neutral-fill-v1'))
    assert actual['valid'].tolist()==[[True,True],[False,False]]
    assert not actual['rgb'][1].any()
    assert actual['endpoint_ids'].tolist()==[0,1]  # No difficult row silently removed.


def test_unknown_quality_recipe_cannot_silently_fall_back():
    with pytest.raises(ValueError,match='camera quality'):
        rebuild.episode_job(dict(camera_quality_version='unknown',row={},views=[],local=[]))


@pytest.mark.parametrize('kind', ['gray-object', 'short-band', 'black-padding'])
def test_gray_object_and_small_band_are_not_neutral_fill(kind):
    rng=np.random.default_rng(19)
    frame=rng.integers(20,230,(480,640,3),dtype=np.uint8)
    if kind=='gray-object': frame[:,100:500]=128
    if kind=='short-band': frame[:20]=128
    if kind=='black-padding': frame[:60]=frame[-60:]=0
    assert raw_camera_quality(frame)['valid']


def test_quality_does_not_modify_raw_pixels():
    image=np.full((480,640,3),128,np.uint8)
    before=image.copy()
    assert not raw_camera_quality(image)['valid']
    np.testing.assert_array_equal(image,before)


def reviewed_manifest():
    return dict(schema=REVIEW_SCHEMA, review_id='test', dataset_root='/recording',
        exposures=[dict(episode=0, endpoint=939, local_frame=1, view='column', reason='reviewed band')])


def test_review_masks_only_exact_camera_and_identity(monkeypatch):
    rng=np.random.default_rng(91)
    images=rng.integers(20,230,(2,480,640,3),dtype=np.uint8)
    images[1,240:,:,0]=180
    # No colour threshold has been introduced: this raw check still accepts.
    assert raw_camera_quality(images[1])['valid']
    frames=np.stack([letterbox_rgb(x[...,::-1]) for x in images])
    times=np.array([0.,1/30]); q=np.arange(12,dtype=np.float32).reshape(2,6)
    record=dict(time=times,q=q,action=q+.5,masks=np.ones((2,2),bool),mask_present=np.ones((2,2),bool))
    monkeypatch.setattr(rebuild,'read_recording',lambda *a:record)
    monkeypatch.setattr(rebuild,'resolve_media',lambda *a:dict(path='video',requested_pts=times,
        start=0.,stop=1.,clock_origin=0.,provenance='test',declared_video_fps=30))
    def decode(*args,raw_rgb_callback=None):
        if raw_rgb_callback:
            for i,image in enumerate(images):raw_rgb_callback(i,image)
        return frames.copy(),times.copy(),np.ones(2,bool),{}
    monkeypatch.setattr(rebuild,'decode_at_recorded_times',decode)
    job=dict(identity=0,row=dict(dataset_root='/recording'),
             views=['observation.images.wrist','observation.images.column'],local=[0,1],endpoint_ids=[938,939])
    policy=dict(fish=REVIEWED_CAMERA_QUALITY_VERSION,fish_review=reviewed_manifest())
    actual=rebuild.episode_job(dict(job,**camera_quality_job(policy)))
    assert actual['valid'].tolist()==[[True,True],[True,False]]
    assert actual['reasons'].tolist()==[[0,0],[0,6]]
    assert not actual['rgb'][1,1].any()
    assert actual['rgb'][1,0].any()
    np.testing.assert_array_equal(actual['q'],q)
    np.testing.assert_array_equal(actual['actions'],q+.5)
    np.testing.assert_array_equal(actual['times'],times)
    assert actual['endpoint_ids'].tolist()==[938,939]
    # Replaying the old explicit generation has exactly its old behaviour.
    assert rebuild.episode_job(dict(job,camera_quality_version='raw-neutral-fill-v1'))['valid'].all()


def test_review_cannot_leak_to_another_dataset_or_wrong_local_identity():
    job=dict(identity=0,row=dict(dataset_root='/recording'),local=[0,1],endpoint_ids=[938,939])
    assert reviewed_exposure_mask(reviewed_manifest(),job,'column').tolist()==[False,True]
    assert not reviewed_exposure_mask(reviewed_manifest(),job,'wrist').any()
    with pytest.raises(ValueError,match='same-dataset'):
        reviewed_exposure_mask(reviewed_manifest(),dict(job,row=dict(dataset_root='/other')),'column')
    with pytest.raises(ValueError,match='local frame'):
        reviewed_exposure_mask(reviewed_manifest(),dict(job,local=[4,5]),'column')
    review=reviewed_manifest();review['exposures'][0]['episode']=2
    with pytest.raises(ValueError,match='admitted'):
        reviewed_exposure_mask(review,job,'column')


def test_review_version_never_silently_omits_its_manifest():
    with pytest.raises(ValueError,match='exposure manifest'):
        rebuild.episode_job(dict(camera_quality_version=REVIEWED_CAMERA_QUALITY_VERSION))
