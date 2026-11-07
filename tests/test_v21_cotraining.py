"""Section 17.121 (James 10/28 「继续」): the Actor trainer's co-training (scripts/train_v21_fcpg_actor.py
--archive-fish-share): fish rows of demonstrations 0 and 1 beside the Archive rows of every Archive-stage update, from
their own generators, so the Archive rows' draws stay those of the same run without them. No data, no GPU."""
import numpy as np
import pytest
import torch

from scripts.train_v21_fcpg_actor import cotraining_generators, cotraining_rows, dropout_stream, main

COMMON = ['--source-config', 'c.json', '--camera-layout', 'l.npz', '--lewm', 'lewm.pt', '--output', 'run-that-is-never-made']


def test_the_cotraining_options_are_checked_before_anything_is_read():
    for bad, match in ((['--archive-fish-share', '1'], 'share in'),
                       (['--archive-fish-share', '-.1'], 'share in'),
                       (['--archive-fish-share', '.001'], 'no fish row'),
                       (['--archive-fish-share', '.1', '--archive-checkpoint', 'a.pt'], '--archive-checkpoint'),
                       (['--archive-fish-share', '.1', '--evaluate-checkpoint', 'a.pt'], '--evaluate-checkpoint'),
                       (['--archive-fish-share', '.1', '--world-head'], 'not with'),
                       (['--archive-fish-share', '.1', '--view-dropout', '.2'], 'not with')):
        with pytest.raises(ValueError, match=match):
            main(COMMON + bad)


def test_the_fish_rows_beside_the_archive_rows_make_the_share():
    assert cotraining_rows(0., 128) == 0
    assert cotraining_rows(.1, 128) == 14            # 14 of 142 rows: 9.9%
    assert cotraining_rows(.5, 128) == 128           # Mobile ALOHA's equal share
    for share in (.02, .1, .25, .5):
        k = cotraining_rows(share, 128)
        assert abs(k / (128 + k) - share) < .5 / 128


def test_the_cotraining_draws_come_from_seed_plus_21_to_25():
    g = cotraining_generators(7, torch.device('cpu'))
    assert g['rows'].random(3).tolist() == np.random.default_rng(28).random(3).tolist()
    assert g['paste'].random(3).tolist() == np.random.default_rng(29).random(3).tolist()
    for name, seed in (('mask', 30), ('cue', 31)):
        assert torch.equal(torch.rand(3, generator=g[name]), torch.rand(3, generator=torch.Generator().manual_seed(seed)))


def test_the_dropout_stream_keeps_the_global_draws_and_continues_its_own():
    stream = cotraining_generators(7, torch.device('cpu'))['dropout']
    torch.manual_seed(123)
    expected = torch.rand(5)                         # what the Archive rows draw next without the fish rows
    torch.manual_seed(123)
    with dropout_stream(stream, torch.device('cpu')):
        inside = torch.rand(5)
    assert torch.equal(torch.rand(5), expected)      # unchanged by the fish rows' draws
    with dropout_stream(stream, torch.device('cpu')):
        again = torch.rand(5)
    torch.manual_seed(32)                            # seed + 25: where the stream starts
    assert torch.equal(inside, torch.rand(5))
    assert torch.equal(again, torch.rand(5))         # and it goes on where it stopped
