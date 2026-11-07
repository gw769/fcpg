"""Section 17.120 (James 10/28 「真机器算了，你还是训练最强模型把」「也可以参考nero」): the Actor trainer's fish-stage recipe
options (scripts/train_v21_fcpg_actor.py): the learning rates rising and falling to zero, the weight average and the fish
stage's own seed. No data, no GPU."""
import numpy as np
import pytest
import torch

from scripts.train_v21_fcpg_actor import WeightAverage, archive_learning_rate_factor, fish_stage_generators, main

COMMON = ['--source-config', 'c.json', '--camera-layout', 'l.npz', '--lewm', 'lewm.pt', '--output', 'run-that-is-never-made']


def test_the_fish_options_are_checked_before_anything_is_read():
    for bad, match in ((['--fish-warmup', '50'], '--fish-warmup'),
                       (['--fish-schedule', 'warmup-linear'], '--fish-warmup'),
                       (['--fish-schedule', 'warmup-linear', '--fish-warmup', '1000'], '--fish-warmup'),
                       (['--fish-ema', '1'], '--fish-ema'),
                       (['--fish-ema', '-.1'], '--fish-ema'),
                       (['--fish-seed', '-1'], '--fish-seed'),
                       (['--fish-ema', '.999', '--evaluate-checkpoint', 'x.pt'], '--evaluate-checkpoint'),
                       (['--fish-seed', '7', '--evaluate-checkpoint', 'x.pt'], '--evaluate-checkpoint')):
        with pytest.raises(ValueError, match=match):
            main(COMMON + bad)


def test_the_fish_schedule_rises_to_the_fish_rates_and_falls_to_zero_at_the_last_update():
    factors = [archive_learning_rate_factor(step, 1000, 50, 'warmup-linear') for step in range(1, 1001)]
    assert factors[0] == pytest.approx(1 / 50) and factors[49] == 1. and factors[-1] == 0.
    assert all(a < b for a, b in zip(factors[:49], factors[1:50]))
    assert all(a > b for a, b in zip(factors[49:-1], factors[50:]))
    assert archive_learning_rate_factor(375, 1000, 0, 'constant') == 1.


def test_a_fish_seed_equal_to_the_seed_draws_as_the_archive_checkpoint_does():
    mask = torch.Generator()
    rng, paste, scene = fish_stage_generators(21004, mask)
    after = torch.rand(3)
    assert np.array_equal(rng.integers(0, 10 ** 9, 8), np.random.default_rng(21004 + 4).integers(0, 10 ** 9, 8))
    assert np.array_equal(paste.random(8), np.random.default_rng(21004 + 13).random(8))
    assert np.array_equal(scene.random(8), np.random.default_rng(21004 + 14).random(8))
    reference = torch.Generator()
    reference.manual_seed(21004 + 5)
    assert torch.equal(torch.rand(4, generator=mask), torch.rand(4, generator=reference))
    torch.manual_seed(21004 + 6)
    assert torch.equal(after, torch.rand(3))
    other, other_paste, _ = fish_stage_generators(7, torch.Generator())
    assert not np.array_equal(other.integers(0, 10 ** 9, 8), np.random.default_rng(21004 + 4).integers(0, 10 ** 9, 8))
    assert not np.array_equal(other_paste.random(8), np.random.default_rng(21004 + 13).random(8))


def test_the_average_the_fish_stage_saves_trails_the_trained_weights_and_is_kept_at_the_end():
    weight = torch.nn.Parameter(torch.zeros(2))
    average = WeightAverage([weight], .999)
    for step in range(1, 101):              # weights trained up to 100: the average stays behind, nearer the late ones
        with torch.no_grad():
            weight.fill_(float(step))
        average.update()
    with average.swapped():
        saved = weight.detach().clone()
    assert torch.equal(weight.detach(), torch.full((2,), 100.))       # training goes on with its own weights
    assert 80. < saved[0] < 100.
    average.keep()
    assert torch.equal(weight.detach(), saved)
