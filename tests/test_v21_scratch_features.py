"""Feature-bank parity against the actual V21 CPU chain path; no CUDA IO."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.prod_v18_1_data import V181FeatureCache
from fcpg.v21_data import TaskVocabulary, V21Sources
from fcpg.v21_scratch_features import ScratchDinoFeatureBank


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def actual_chain_source():
    """Small canonical motion/feature arrays consumed by unchanged chain()."""
    source = object.__new__(V21Sources)
    source.device, source.device_cache = torch.device('cpu'), None
    source.compact_host_features, source.motion_only = False, False
    source.counts = dict(archive=2, dense=1, fish=0)
    source.dense_rows = np.array([1], np.int64)
    source.vocabulary = TaskVocabulary.build(['pick red block'])
    source.raw_q = np.arange(120*6, dtype=np.float32).reshape(120, 6)/32
    source.raw_actions = source.raw_q+.125
    source.offsets = np.array([0], np.int64)
    source.episode_sources = np.array([4], np.int64)
    local = np.arange(8, dtype=np.int64)*15
    endpoints = np.array([[0, 1, 2, 3, 4], [3, 4, 5, 6, 7]], np.int64)
    validity = np.ones((8, 2), dtype=bool)
    validity[2, 1] = False
    action_ids = (local[endpoints[:, :-1, None]]+np.arange(5)[None, None]*3)
    archive = SimpleNamespace(row_endpoint_index=endpoints, endpoint_view_valid=validity,
        endpoint_local_frame_index=local, endpoint_episode_index=np.zeros(8, np.int64),
        action_frame_indices=action_ids.reshape(2, 20), chain_id=np.arange(2),
        episodes=[dict(frame_start=0)], fps=np.array([30.]))
    def chain_batch(rows):
        ep = endpoints[rows]
        return SimpleNamespace(endpoint_index=ep, commands_position=source.raw_actions[action_ids[rows]],
            q_position=source.raw_q[local[ep]], task_text=['pick red block']*len(rows),
            view_valid=validity[ep])
    archive.chain_batch = chain_batch
    source.archive = archive
    values = (np.arange(8*2*256*384, dtype=np.float32).reshape(8, 2, 256, 384)%997/128).astype(np.float16)
    values[~validity] = 0
    source.archive_dino = []
    for name, ids in [('train', [0, 2, 3, 5, 7]), ('dev', [1, 4, 6])]:
        cache = object.__new__(V181FeatureCache)
        cache.root = Path('/explicit-fixture')/name
        cache.endpoint_index = np.asarray(ids, np.int64)
        cache.tokens = values[ids].copy()
        cache.view_valid = validity[ids].copy()
        source.archive_dino.append(cache)
    return source


@pytest.mark.parametrize('kind,rows', [('archive', [1, 0, 1]), ('dense', [0, 0])])
def test_bank_matches_actual_v21_chain_values_order_dtype_and_host_ownership(kind, rows):
    source = actual_chain_source()
    baseline = source.chain(np.asarray(rows), kind, images=True)
    bank = ScratchDinoFeatureBank(source, 'cpu', 16*2**20)
    features, valid = bank.load(rows, kind)
    assert features.dtype == torch.float32 and features.device.type == 'cpu'
    assert torch.equal(features, baseline.features)
    assert np.array_equal(valid, baseline.view_valid.numpy())
    assert source.device == torch.device('cpu') and source.device_cache is None
    assert source.compact_host_features is False
    assert all(block.tensor.dtype == torch.float16 for block in bank.cache._blocks.values())
    calls = bank.stats()['loader_calls']
    again, masks = bank.load(rows, kind)
    assert bank.stats()['loader_calls'] == calls
    assert torch.equal(again, baseline.features) and np.array_equal(masks, valid)
    stored, held = bank.load(rows, kind, as_stored=True)      # 17.173: the float16 values, exactly the same
    assert stored.dtype == torch.float16 and torch.equal(stored.float(), baseline.features) and np.array_equal(held, valid)


def test_cache_eviction_and_clear_resume_change_residency_only():
    source = actual_chain_source()
    bank = ScratchDinoFeatureBank(source, 'cpu', 2*2*256*384*2)
    for rows in ([1], [0], [1, 0, 1]):
        features, valid = bank.load(rows)
        baseline = source.chain(np.asarray(rows), images=True)
        assert torch.equal(features, baseline.features)
        assert np.array_equal(valid, baseline.view_valid.numpy())
        assert bank.stats()['resident_bytes'] <= 2*2*256*384*2
    assert bank.stats()['evictions'] > 0
    assert bank.clear() > 0 and bank.stats()['resident_bytes'] == 0
    features, _ = bank.load([1, 0, 1])
    fresh = ScratchDinoFeatureBank(source, 'cpu', 16*2**20)
    after_resume, _ = fresh.load([1, 0, 1])
    assert torch.equal(features, after_resume)


def test_picture_memories_read_through_the_bank_equal_the_source_and_cache_their_rows():
    source = actual_chain_source()
    ids = np.array([5, 2, 5, 0, 6], np.int64)        # train and dev caches, a repeat, an absent view (endpoint 2)
    expected, expected_valid = source._get_archive_dino(ids)
    bank = ScratchDinoFeatureBank(source, 'cpu', 16*2**20)
    tokens, valid = bank.pictures('archive', ids)
    assert tokens.dtype == torch.float16 and tokens.device.type == 'cpu'
    assert torch.equal(tokens, torch.as_tensor(expected).to(torch.float16))
    assert np.array_equal(valid, expected_valid)
    calls = bank.stats()['loader_calls']
    again, _ = bank.pictures('archive', ids)
    assert bank.stats()['loader_calls'] == calls and torch.equal(again, tokens)
    assert source.device == torch.device('cpu') and source.device_cache is None
    with pytest.raises(ValueError, match='unknown picture memory source'):
        bank.pictures('dense', ids)


@pytest.mark.parametrize('context_frames', [1, 2, 3, 4])
def test_feature_prefix_matches_original_context_without_fetching_future_endpoints(context_frames):
    source = actual_chain_source()
    baseline = source.chain(np.array([0]), images=True)
    bank = ScratchDinoFeatureBank(source, 'cpu', 16*2**20)
    original = bank._features._get_archive_dino
    seen = []
    def load(ids):
        seen.extend(ids.tolist())
        return original(ids)
    bank._features._get_archive_dino = load
    features, valid = bank.load([0], context_frames=context_frames)
    assert seen == list(range(context_frames))
    assert torch.equal(features, baseline.features[:, :context_frames])
    assert np.array_equal(valid, baseline.view_valid[:, :context_frames].numpy())
