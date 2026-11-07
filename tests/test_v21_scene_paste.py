"""Section 17.120 (James 「我要的是通用场景解决方案」): the Actor trainer's scene paste (scripts/train_v21_fcpg_actor.py
--scene-paste-bank / --scene-paste-share): the lifting held fish rows the box paste left alone see another room's
wrist background, from their own generator. No data, no GPU."""
import numpy as np
import pytest
import torch

from scripts.train_v21_fcpg_actor import main, paste_background, scene_paste_rows

COMMON = ['--source-config', 'c.json', '--camera-layout', 'l.npz', '--lewm', 'lewm.pt', '--output', 'run-that-is-never-made']


def test_the_scene_paste_options_are_checked_before_anything_is_read():
    for bad, match in ((['--scene-paste-share', '2', '--scene-paste-bank', 'b.pt'], 'share in'),
                       (['--scene-paste-share', '.5'], 'go together'),
                       (['--scene-paste-bank', 'b.pt'], 'go together'),
                       (['--scene-paste-share', '.5', '--scene-paste-bank', 'b.pt'], 'grasp cue'),
                       (['--scene-paste-share', '.5', '--scene-paste-bank', 'b.pt', '--object-cues', 'cues',
                         '--evaluate-checkpoint', 'x.pt'], '--evaluate-checkpoint')):
        with pytest.raises(ValueError, match=match):
            main(COMMON + bad)


def test_the_scene_paste_takes_only_lifting_rows_the_box_paste_left_alone():
    lifting = np.array([1, 1, 1, 0, 0, 1], bool)
    box = np.array([1, 0, 0, 0, 1, 0], bool)
    draws = np.array([1, 1, 0, 1, 1, 1], bool)
    assert scene_paste_rows(lifting, box, draws).tolist() == [False, True, False, False, False, True]
    assert scene_paste_rows(lifting, np.zeros(6, bool), draws).tolist() == (lifting & draws).tolist()
    rng = np.random.default_rng(3)
    many = scene_paste_rows(np.ones(20000, bool), rng.random(20000) < .5, rng.random(20000) < .5)
    assert .22 < many.mean() < .28           # a half of the half the box paste left


def test_a_scene_entry_replaces_the_background_of_the_three_newest_wrist_exposures_only():
    background = torch.tensor([0, 1, 14, 15])
    features = torch.zeros(3, 5, 2, 196, 4)
    bank = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4) + 1
    out = paste_background(features, 3, np.array([1]), bank, np.array([1]), background)
    for k in range(3):                       # newest first: frame 3 takes entry k = 0, frame 1 takes k = 2
        assert torch.equal(out[1, 3 - k, 0, background], bank[1, k])
    assert torch.equal(out[1, 0], features[1, 0]) and torch.equal(out[1, 4], features[1, 4])
    assert torch.equal(out[1, :, 1], features[1, :, 1])                         # the other camera
    keep = torch.ones(196, dtype=torch.bool)
    keep[background] = False
    assert torch.equal(out[1, :, 0][:, keep], features[1, :, 0][:, keep])       # the jaws and the fish
    assert torch.equal(out[0], features[0]) and torch.equal(out[2], features[2])
