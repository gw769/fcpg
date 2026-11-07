"""17.106: the Archive LeWM stage's window reading keeps ordered_groups()' draws and names every draw's window."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.train_v21_horizon40 import archive_windows, one_ahead, ordered_groups


def schedule(population, passes=2, batch=128):
    args = SimpleNamespace(seed=21004, lewm_batch_size=batch, batch_size=64, archive_passes=passes, canary=False,
                           from_scratch=False)
    return ordered_groups('lewm_archive', dict(train=dict(archive=population)), args)


@pytest.mark.parametrize('readers', [1, 3])
def test_the_window_schedule_is_ordered_groups_with_its_windows(readers):
    population = np.sort(np.random.default_rng(0).choice(20000, 9000, replace=False))
    plain = list(schedule(population))
    windowed = list(archive_windows(schedule(population), population, readers))
    assert len(windowed) == len(plain) == 2 * sum(-(-len(population[i:i + 2048]) // 128) for i in range(0, 9000, 2048))
    drawn, order = {}, []
    for (h, groups), (h2, groups2, (key, rows, last, following)) in zip(plain, windowed):
        assert h == h2 and len(groups2) == 1 and np.array_equal(groups[0][1], groups2[0][1])
        assert np.isin(groups2[0][1], rows).all() and (np.diff(rows) > 0).all()
        assert any(np.array_equal(rows, population[i:i + 2048]) for i in range(0, 9000, 2048))
        if key not in drawn:
            order.append(key)
            drawn[key] = dict(rows=rows, batches=[])
        drawn[key]['batches'].append((groups2[0][1], last))
        assert [k for k, _ in following] == list(range(key + 1, min(key + 1 + readers, 10)))
    assert order == list(range(10))                                   # 5 windows per pass, in drawing order
    for key, window in drawn.items():
        rows = np.concatenate([batch for batch, _ in window['batches']])
        assert np.array_equal(np.sort(rows), window['rows'])          # every row of the window, once
        assert [last for _, last in window['batches']] == [False] * (len(window['batches']) - 1) + [True]


def test_a_batch_outside_its_window_is_refused():
    population = np.arange(5000)
    bad = [(0, [('archive', np.array([1, 2, 3000]))])]
    with pytest.raises(ValueError, match='spans two'):
        list(archive_windows(bad, population, 2))


def test_one_ahead_pairs_every_item_with_the_next():
    assert list(one_ahead('abc')) == [('a', 'b'), ('b', 'c'), ('c', None)]
    assert list(one_ahead([])) == []
