"""Independent tiny-data CPU checks for frozen DINO row caching, no GPU/robot IO."""
from types import SimpleNamespace
import json
import random

import numpy as np
import pytest
import torch

from fcpg.prod_v18_1_data import V181FeatureCache
from fcpg.prod_v20_1_sources import V201FishTrainingSource
from fcpg.v21_data import V21Sources
from fcpg.v21_scratch_features import ScratchDinoFeatureBank


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('GPU access forbidden'))
    yield
    torch.set_num_threads(threads)


def feature_cache(root, endpoint_ids, valid, base):
    root.mkdir()
    root.joinpath('config.json').write_text(json.dumps(dict(view_order=['wrist', 'column'],
        backbone='dinov2_vits14', kind=root.name, complete=True)))
    endpoint_ids = np.asarray(endpoint_ids, np.int64)
    valid = np.asarray(valid, bool)
    patch = ((np.arange(256*384) % 256)/256).reshape(1, 1, 256, 384)
    values = (base+np.arange(len(endpoint_ids))[:, None, None, None]/4
        +np.arange(2)[None, :, None, None]/8+patch).astype(np.float16)
    values[~valid] = 0
    np.save(root/'endpoint_index.npy', endpoint_ids)
    np.save(root/'tokens.npy', values)
    np.save(root/'view_valid.npy', valid)
    return V181FeatureCache(root)


@pytest.fixture
def host_source(tmp_path):
    train_valid, dev_valid = np.ones((6, 2), bool), np.ones((6, 2), bool)
    train_valid[2, 1] = False; train_valid[3] = False; dev_valid[1, 0] = False
    archive_dino = [feature_cache(tmp_path/'temporal-train', np.arange(6), train_valid, 1),
        feature_cache(tmp_path/'temporal-dev', np.arange(100, 106), dev_valid, 7)]
    fish_endpoints = (np.asarray([0, 61])[:, None]+np.arange(5)[None]*15).reshape(-1)
    fish_dino = feature_cache(tmp_path/'fish', fish_endpoints, np.ones((10, 2), bool), 19)
    source = object.__new__(V21Sources)
    source.device = torch.device('cpu'); source.device_cache = None; source.compact_host_features = False
    source.counts = dict(archive=4, dense=2, fish=3)
    source.dense_rows = np.asarray([3, 0])
    view_valid = np.zeros((106, 2), bool)
    view_valid[:6] = train_valid; view_valid[100:] = dev_valid
    source.archive = SimpleNamespace(row_endpoint_index=np.asarray([
        [0, 1, 2, 3, 4], [100, 101, 102, 103, 104], [1, 2, 3, 4, 5], [101, 102, 103, 104, 105]]),
        endpoint_view_valid=view_valid)
    source.archive_dino, source.fish_dino = archive_dino, fish_dino
    source.fish = object.__new__(V201FishTrainingSource)
    source.fish.starts = np.asarray([0, 61, 122])
    source.fish.episodes = np.repeat([0, 3, 2], 61)
    return source


def baseline(source, rows, kind):
    rows = np.asarray(rows, np.int64)
    if kind == 'dense':
        rows = source.dense_rows[rows]
    if kind == 'fish':
        endpoint = source.fish.starts[rows, None]+np.arange(5)[None]*15
        source.fish.validate_frames(endpoint.reshape(-1))
    else:
        endpoint = source.archive.row_endpoint_index[rows]
    unique, inverse = np.unique(endpoint, return_inverse=True)
    tokens, valid = (source._feature_values(source.fish_dino, unique) if kind == 'fish'
                     else source._get_archive_dino(unique))
    return torch.from_numpy(tokens)[inverse].reshape(len(rows), 5, 2, 256, 384), valid[inverse].reshape(len(rows), 5, 2)


def rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def same_rng(actual, expected):
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0] and actual[1][2:] == expected[1][2:]
    np.testing.assert_array_equal(actual[1][1], expected[1][1])
    assert torch.equal(actual[2], expected[2])


class CountedTokens:
    def __init__(self, array):
        self.array, self.reads = array, []
    def __getattr__(self, name):
        return getattr(self.array, name)
    def __getitem__(self, index):
        self.reads.append(np.asarray(index).copy())
        return self.array[index]


@pytest.mark.parametrize('budget', [0, 16*2**20])
def test_out_of_order_duplicate_archive_dense_fish_equal_uncached_without_rng_or_host_mutation(host_source, budget):
    host = host_source
    requests = [('archive', [2, 1, 0, 2]), ('archive', [3, 0, 1, 3]),
        ('dense', [1, 0, 1]), ('fish', [1, 0, 1])]
    expected = [baseline(host, rows, kind) for kind, rows in requests]
    before = rng_snapshot()
    identity = dict(host.__dict__)
    token_copies = [np.array(cache.tokens, copy=True) for cache in (*host.archive_dino, host.fish_dino)]
    bank = ScratchDinoFeatureBank(host, 'cpu', budget)
    for _ in range(2):
        for (kind, rows), (features, valid) in zip(requests, expected):
            actual_features, actual_valid = bank.load(np.asarray(rows), kind)
            assert actual_features.device.type == 'cpu' and actual_features.dtype == torch.float32
            assert actual_features.shape == features.shape and torch.equal(actual_features, features)
            assert isinstance(actual_valid, np.ndarray) and actual_valid.dtype == np.bool_
            np.testing.assert_array_equal(actual_valid, valid)
            assert not actual_features.requires_grad and actual_features.grad_fn is None
    same_rng(rng_snapshot(), before)
    assert host.__dict__.keys() == identity.keys()
    for name, value in identity.items():
        assert host.__dict__[name] is value, name
    assert host.device == torch.device('cpu') and host.device_cache is None and host.compact_host_features is False
    for cache, expected_tokens in zip((*host.archive_dino, host.fish_dino), token_copies):
        np.testing.assert_array_equal(cache.tokens, expected_tokens)


def test_cache_namespaces_separate_train_dev_fish_local_row_zero_and_hits_do_not_reread(host_source):
    host = host_source
    expected = [baseline(host, [row], kind) for kind, row in [('archive', 0), ('archive', 1), ('fish', 0)]]
    counted = [CountedTokens(cache.tokens) for cache in (*host.archive_dino, host.fish_dino)]
    for cache, tokens in zip((*host.archive_dino, host.fish_dino), counted):
        cache.tokens = tokens
    bank = ScratchDinoFeatureBank(host, 'cpu', 16*2**20)
    requests = [('archive', 0), ('archive', 1), ('fish', 0)]
    for (kind, row), (features, valid) in zip(requests, expected):
        actual, available = bank.load(np.asarray([row]), kind)
        assert torch.equal(actual, features)
        np.testing.assert_array_equal(available, valid)
    read_counts = [len(tokens.reads) for tokens in counted]
    assert read_counts == [1, 1, 1]
    for (kind, row), (features, valid) in zip(reversed(requests), reversed(expected)):
        actual, available = bank.load(np.asarray([row]), kind)
        assert torch.equal(actual, features)
        np.testing.assert_array_equal(available, valid)
        actual.fill_(-99)  # Returned batch must not alter the resident immutable bank.
        unchanged, _ = bank.load(np.asarray([row]), kind)
        assert torch.equal(unchanged, features)
    assert [len(tokens.reads) for tokens in counted] == read_counts


@pytest.mark.parametrize('corruption', ['nan', 'inf', 'invalid_view_nonzero', 'view_identity_mismatch', 'missing_endpoint'])
def test_bad_or_missing_archive_features_are_rejected(host_source, corruption):
    host = host_source
    cache = host.archive_dino[0]
    if corruption in ('nan', 'inf', 'invalid_view_nonzero'):
        cache.tokens = np.array(cache.tokens, copy=True)
        if corruption == 'invalid_view_nonzero':
            cache.tokens[2, 1, 0, 0] = .25
        else:
            cache.tokens[0, 0, 0, 0] = np.nan if corruption == 'nan' else np.inf
    elif corruption == 'view_identity_mismatch':
        host.archive.endpoint_view_valid[0, 0] = False
    else:
        cache.endpoint_index = cache.endpoint_index[1:]
        cache.tokens = cache.tokens[1:]
        cache.view_valid = cache.view_valid[1:]
    bank = ScratchDinoFeatureBank(host, 'cpu', 16*2**20)
    with pytest.raises(ValueError):
        bank.load(np.asarray([0]), 'archive')


def test_fish_cache_cannot_invent_a_missing_view(host_source):
    cache = host_source.fish_dino
    cache.tokens, cache.view_valid = np.array(cache.tokens, copy=True), np.array(cache.view_valid, copy=True)
    cache.view_valid[0, 0] = False; cache.tokens[0, 0] = 0
    bank = ScratchDinoFeatureBank(host_source, 'cpu', 16*2**20)
    with pytest.raises(ValueError):
        bank.load(np.asarray([0]), 'fish')


def test_fish_episode_two_and_invalid_row_ids_rejected_before_any_feature_values(host_source):
    host = host_source
    class SealedTokens:
        def __getitem__(self, index):
            pytest.fail('unadmitted row reached frozen feature values')
    for cache in (*host.archive_dino, host.fish_dino):
        cache.tokens = SealedTokens()
    bank = ScratchDinoFeatureBank(host, 'cpu', 16*2**20)
    with pytest.raises(ValueError, match='episode 2'):
        bank.load(np.asarray([2]), 'fish')
    for rows, kind in [([-1], 'archive'), ([4], 'archive'), ([2], 'dense'),
                       ([3], 'fish'), ([.5], 'archive'), ([], 'archive')]:
        with pytest.raises(ValueError):
            bank.load(np.asarray(rows), kind)


def test_episode_admission_is_rechecked_even_if_all_fish_features_are_cached(host_source):
    bank = ScratchDinoFeatureBank(host_source, 'cpu', 16*2**20)
    bank.load(np.asarray([0]), 'fish')
    before = bank.stats()
    assert before['resident_rows'] == 5
    # The guard must precede cache lookup, so prior residents cannot grant access.
    host_source.fish.episodes[:61] = 2
    with pytest.raises(ValueError, match='episode 2'):
        bank.load(np.asarray([0]), 'fish')
    assert bank.stats()['row_hits'] == before['row_hits']
