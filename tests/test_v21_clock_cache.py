from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_clock_cache import RecordedClockCache
from fcpg.v21_clock_evaluation import short_training_gate, locked_development_rows
from fcpg.v21_recorded_clock import CLOCK_VERSION, EXTRACTOR_VERSION
from fcpg.v21_horizon_training import horizon_targets, factual_state


def test_real_clock_preserves_every_command_and_tail_not_next_episode():
    # Metadata says 10 Hz; actual state/command samples are 30 Hz with jitter.
    length = 101
    times = 12 + np.arange(length)/30 + (np.arange(length) % 3)*.0001
    q = np.repeat(np.arange(length, dtype=np.float32)[:, None], 6, axis=1)
    data = SimpleNamespace(counts={'archive': 1}, offsets=np.array([0, length]),
        archive=SimpleNamespace(row_endpoint_index=np.array([[0, 1, 2, 3, 4]]),
            endpoint_episode_index=np.zeros(5, np.int64), endpoint_local_frame_index=np.array([0, 15, 30, 45, 60]),
            fps=np.array([10.])), raw_q=q, raw_actions=q+.25,
        recorded_clock=SimpleNamespace(times=lambda kind: times))
    target = horizon_targets(data, [0], 'archive', 0)
    assert target.trace.valid.sum() == 99
    np.testing.assert_array_equal(target.trace.positions[0], data.raw_actions[:99])
    np.testing.assert_allclose(target.trace.offsets[0], times[:99]-times[0])
    assert target.valid.sum() == 33
    assert target.readbacks[0, 32, 0] == 99
    assert (target.readbacks[0, 33:] == 0).all()


def test_factual_input_uses_actual_capture_q_and_time():
    from fcpg.v21_data import TaskVocabulary
    from fcpg.v21_training import synthetic_chain
    from dataclasses import replace
    chain = synthetic_chain(TaskVocabulary.build(['pick']), batch=1)
    captured_q = chain.q[:, :, None].repeat(1, 1, 2, 1) + .125
    captured_times = chain.endpoint_times[:, :, None].repeat(1, 1, 2)-.05
    chain = replace(chain, capture_q=captured_q, capture_times=captured_times)
    class Core:
        def observe(self, previous, observation, q, time):
            assert torch.equal(observation.q_at_image, captured_q[:, 0])
            assert torch.equal(observation.image_times, captured_times[:, 0])
            assert torch.equal(q, chain.q[:, 0])
            return 'checked'
    assert factual_state(Core(), chain, 0, images=True) == 'checked'


def cache_fixture(root):
    meta = dict(complete=True, clock_version=CLOCK_VERSION, extractor_version=EXTRACTOR_VERSION,
        extraction_id='test', repository='/frozen/test/dino', weights='/frozen/test/weights',
        input_rgb=True, image_size=224, resize='letterbox', normalization={'mean': [.485, .456, .406]},
        compute_precision='bf16', storage_precision='float16', camera_role_policy='unchanged')
    (root/'manifest.json').write_text(json.dumps(meta))
    np.save(root/'archive_timestamps.npy', [0., .1])
    np.save(root/'fish_timestamps.npy', [0., np.nan, .1])
    values = []
    for name, ids in [('temporal-train', [0]), ('temporal-dev', [1]), ('fish', [0, 2])]:
        path = root/name
        path.mkdir()
        (path/'config.json').write_text(json.dumps(meta))
        np.save(path/'capture_times.npy', np.zeros((len(ids), 2)))
        np.save(path/'capture_q.npy', np.zeros((len(ids), 2, 6), np.float32))
        values.append(SimpleNamespace(root=path, endpoint_index=np.asarray(ids), view_valid=np.ones((len(ids), 2), bool)))
    paths = dict(dino_cache_root=str(root), dino_clock_version=CLOCK_VERSION, dino_extraction_id='test')
    archive = SimpleNamespace(episodes=[dict(frame_stop=2)], endpoint_count=2)
    def allowed(ids):
        if 1 in ids:
            raise ValueError('sealed episode')
    fish = SimpleNamespace(q=np.zeros((3, 6)), training_frames=np.array([0, 2]), validate_frames=allowed)
    return paths, archive, fish, values, meta


@pytest.mark.parametrize('mutation', ['incomplete', 'generation', 'weights', 'sealed'])
def test_reader_rejects_partial_mixed_and_sealed_cache(tmp_path, mutation):
    paths, archive, fish, values, meta = cache_fixture(tmp_path)
    good = RecordedClockCache(paths, archive, fish, values)
    assert good.observations(np.array([0, 2]), 'fish')[0].all()
    if mutation == 'sealed':
        with pytest.raises(ValueError, match='sealed'):
            good.observations(np.array([1]), 'fish')
        return
    if mutation == 'incomplete':
        meta['complete'] = False
    elif mutation == 'generation':
        meta['extraction_id'] = 'other'
    else:
        meta['weights'] = '/different/weights'
    (values[1].root/'config.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='mixed or incomplete'):
        RecordedClockCache(paths, archive, fish, values)


def metrics(ratio=1.):
    return {domain: dict(first5_mae=1., future_mae=1.,
        views={view: dict(relative_scene_error=ratio, targets=100) for view in ('wrist', 'column')})
        for domain in ('archive', 'fish')}


def test_camera_quality_generation_is_explicit_and_consistent(tmp_path):
    paths,archive,fish,values,meta=cache_fixture(tmp_path)
    policy=dict(fish='raw-neutral-fill-v1',archive='unchanged-legacy-validity')
    meta['camera_quality_policy']=policy
    (tmp_path/'manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='explicitly selected'):
        RecordedClockCache(paths,archive,fish,values)
    paths['dino_camera_quality_policy']=policy
    with pytest.raises(ValueError,match='mixed or incomplete'):
        RecordedClockCache(paths,archive,fish,values)
    for cache in values: (cache.root/'config.json').write_text(json.dumps(meta))
    values[-1].view_valid[0,1]=False
    loaded=RecordedClockCache(paths,archive,fish,values)
    assert loaded.observations(np.array([0]),'fish')[0].tolist()==[[True,False]]


def test_short_gate_does_not_accept_smaller_or_easier_evaluation_population():
    base, candidate, parent = metrics(), metrics(), metrics()
    candidate['archive']['views']['wrist']['relative_scene_error'] = .89
    assert short_training_gate(candidate, base, parent)['passed']
    candidate['fish']['views']['column']['targets'] = 99
    assert not short_training_gate(candidate, base, parent)['passed']
    candidate = deepcopy(base)
    assert not short_training_gate(candidate, base, parent)['passed']
    with pytest.raises(ValueError, match='split'):
        locked_development_rows({'rows': dict(archive=[0], fish=[1])}, dict(archive=np.array([0]), fish=np.array([3])))
