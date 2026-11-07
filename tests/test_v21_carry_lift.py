"""Section 17.120 (James 10/27 「搞好阿 重新训练也可以。 狠狠惩罚这种」): the Actor trainer's carry-lift penalty and box-background
paste (scripts/train_v21_fcpg_actor.py): which fish rows hold the fish, the jaws' height gradient, the pasted patches and
the penalty on a carry below the demonstrator's. No data, no GPU."""
import numpy as np
import pytest
import torch

from scripts.train_v21_fcpg_actor import (ARM, CARRY_MARGIN_MM, CARRY_SCALE_MM, HELD_GRIP, carry_lift_penalty, carry_rows,
                                          clearance_gradient, main, paste_background)

WEIGHTS = np.asarray([.5, -5., -4., -2., .1])      # mm per unit, as the robot's carry pose (17.120 weights section)


class LinearGate:
    """clearance_mm stand-in: linear in the five arm joints."""
    def clearance_mm(self, q):
        return 40. + float(np.asarray(q, np.float64)[:ARM] @ WEIGHTS)


def test_the_height_gradient_is_the_clearance_per_unit_of_each_arm_joint():
    assert np.allclose(clearance_gradient(LinearGate(), np.asarray([3., -20., 40., 60., -50., 8.])), WEIGHTS)


def test_held_rows_are_shut_jaws_with_the_grasp_cue_and_their_lift_is_linearised():
    q = np.zeros((4, 6))
    q[:, 5] = (HELD_GRIP - 4., HELD_GRIP - 4., HELD_GRIP + 10., HELD_GRIP - 4.)
    cue = np.zeros((4, 8))
    cue[:3, 0] = 1.                                  # row 3: shut jaws without the fish (after a release)
    commands = np.repeat(q[:, None], 5, 1)
    commands[:, 4, 1] -= 4.                          # the demonstrator lifts: lift -4 units -> +20 mm
    held, gradient, lift = carry_rows(LinearGate(), q, cue, commands)
    assert held.tolist() == [True, True, False, False]
    assert np.allclose(gradient[:2], WEIGHTS) and np.allclose(gradient[2:], 0.)
    assert np.allclose(lift, [20., 20., 0., 0.])
    held, _, lift = carry_rows(LinearGate(), q, None, commands)       # no cue: nothing counts as held
    assert not held.any() and np.allclose(lift, 0.)
    held, _, lift = carry_rows(LinearGate(), q, cue, commands, shut=False)   # Archive rows: the cue alone
    assert held.tolist() == [True, True, True, False] and np.allclose(lift, [20., 20., 20., 0.])


def test_the_paste_changes_only_the_chosen_rows_wrist_background_newest_to_newest():
    b, frames, patches, dim, h = 3, 4, 196, 8, 3
    features = torch.randn(b, frames, 2, patches, dim).half()
    background = torch.as_tensor([0, 1, 2, 50, 100])
    bank = torch.randn(2, 3, len(background), dim).half()
    out = paste_background(features, h, np.asarray([1]), bank, np.asarray([1]), background)
    assert torch.equal(out[[0, 2]], features[[0, 2]])                 # other rows untouched
    assert torch.equal(out[1, :, 1], features[1, :, 1])               # the external view untouched
    assert torch.equal(out[1, 0], features[1, 0])                     # the oldest exposure leaves the memory anyway
    for k in range(3):                                                # exposures h, h-1, h-2 take entries 0, 1, 2
        assert torch.equal(out[1, h - k, 0, background], bank[1, k])
    foreground = torch.ones(patches, dtype=torch.bool)
    foreground[background] = False
    assert torch.equal(out[1, :, 0][:, foreground], features[1, :, 0][:, foreground])
    assert torch.equal(features, features.clone())                    # the input is not changed in place
    early = paste_background(features[:, :2], 1, np.asarray([0, 2]), bank, np.asarray([0, 0]), background)
    assert torch.equal(early[0, 1, 0, background], bank[0, 0]) and torch.equal(early[0, 0, 0, background], bank[0, 1])


def test_the_penalty_punishes_only_a_held_carry_below_the_demonstrator():
    gradient = torch.as_tensor(np.repeat(WEIGHTS[None], 3, 0), dtype=torch.float32)
    held = torch.as_tensor([True, True, False])
    commands = torch.zeros(3, 5, 6)
    commands[:, :, 1] = -4.                                           # the demonstrator: +20 mm
    valid = torch.ones(3, 5, dtype=torch.bool)
    above = commands.clone()
    above[:, :, 1] = -6.                                              # 10 mm higher than the demonstrator
    assert float(carry_lift_penalty(above, commands, valid, gradient, held)) == 0.
    within = commands.clone()
    within[:, :, 1] = -4. + (CARRY_MARGIN_MM - .5) / 5.               # lower, within the margin
    assert float(carry_lift_penalty(within, commands, valid, gradient, held)) == 0.
    low = commands.clone()
    low[:, :, 1] = -2.                                                # 10 mm below the demonstrator
    low.requires_grad_(True)
    penalty = carry_lift_penalty(low, commands, valid, gradient, held)
    assert abs(float(penalty) - ((10. - CARRY_MARGIN_MM) / CARRY_SCALE_MM) ** 2) < 1e-5
    penalty.backward()
    assert float(low.grad[:2].abs().sum()) > 0 and float(low.grad[2].abs().sum()) == 0.   # pushes the held rows up only
    only_free = carry_lift_penalty(low.detach(), commands, valid, gradient, torch.zeros(3, dtype=torch.bool))
    assert float(only_free) == 0.                                     # rows not held never count
    unit = ((10. - CARRY_MARGIN_MM) / CARRY_SCALE_MM) ** 2
    seeing = carry_lift_penalty(low.detach(), commands, valid, gradient, held, torch.as_tensor([3., 1., 1.]))
    assert abs(float(seeing) - unit * (3. + 1.) / 2.) < 1e-5            # a row that sees the place counts 3 times
    valid[0] = False
    valid[1, 2:] = False
    partial = carry_lift_penalty(low.detach(), commands, valid, gradient, held)
    assert abs(float(partial) - ((10. - CARRY_MARGIN_MM) / CARRY_SCALE_MM) ** 2) < 1e-5   # a mean over valid commands


def test_the_over_margin_also_punishes_a_carry_far_above_the_demonstrator():
    """Robot trial 27 (section 17.120): HLtp, punished only below the demonstrator, climbed to 47 cm holding the fish."""
    gradient = torch.as_tensor(np.repeat(WEIGHTS[None], 2, 0), dtype=torch.float32)
    held = torch.as_tensor([True, False])
    commands = torch.zeros(2, 5, 6)
    commands[:, :, 1] = -4.                                           # the demonstrator: +20 mm
    valid = torch.ones(2, 5, dtype=torch.bool)
    above = commands.clone()
    above[:, :, 1] = -4. - 30. / 5.                                   # 30 mm above the demonstrator
    assert float(carry_lift_penalty(above, commands, valid, gradient, held)) == 0.          # below only: free
    above.requires_grad_(True)
    penalty = carry_lift_penalty(above, commands, valid, gradient, held, over_margin_mm=10.)
    assert abs(float(penalty) - ((30. - 10.) / CARRY_SCALE_MM) ** 2) < 1e-4
    penalty.backward()
    # a step against the gradient raises the lift joint towards -4, the jaws down to the demonstrator's; free rows untouched
    assert float(above.grad[0, :, 1].sum()) < 0 and float(above.grad[1].abs().sum()) == 0.
    within = commands.clone()
    within[:, :, 1] = -4. - 8. / 5.                                   # 8 mm above: inside the margin
    assert float(carry_lift_penalty(within, commands, valid, gradient, held, over_margin_mm=10.)) == 0.
    below = commands.clone()
    below[:, :, 1] = -2.                                              # 10 mm below: as before
    assert abs(float(carry_lift_penalty(below, commands, valid, gradient, held, over_margin_mm=10.))
               - ((10. - CARRY_MARGIN_MM) / CARRY_SCALE_MM) ** 2) < 1e-5


def test_the_options_are_checked_before_anything_is_read():
    common = ['--source-config', 'c.json', '--camera-layout', 'l.npz', '--lewm', 'lewm.pt', '--output', 'run-that-is-never-made']
    for bad, match in ((['--carry-lift-weight', '-1'], '--carry-lift-weight'),
                       (['--carry-lift-weight', '5', '--carry-place-factor', '.5'], '--carry-place-factor'),
                       (['--carry-place-factor', '3'], '--carry-place-factor'),
                       (['--box-paste-share', '.5'], '--box-paste-bank'),
                       (['--box-paste-bank', 'b.pt'], '--box-paste-bank'),
                       (['--box-paste-share', '1.5', '--box-paste-bank', 'b.pt'], '--box-paste-share'),
                       (['--carry-over-margin-mm', '10'], '--carry-over-margin-mm'),
                       (['--carry-fish-only'], '--carry-fish-only'),
                       (['--carry-lift-weight', '5', '--carry-over-margin-mm', '-1'], '--carry-over-margin-mm'),
                       (['--carry-lift-weight', '5'], 'grasp cue')):
        with pytest.raises(ValueError, match=match):
            main(common + bad)


def test_the_lewm_is_punished_only_for_forecasting_held_jaws_above_where_they_end():
    from scripts.train_v21_horizon40 import carry_forecast_penalty
    gradient = torch.as_tensor(np.repeat(WEIGHTS[None], 3, 0), dtype=torch.float32)
    held = torch.as_tensor([True, True, False])
    recorded = torch.zeros(3, 5, 6)
    valid = torch.ones(3, 5, dtype=torch.bool)
    lower = recorded.clone()
    lower[:, :, 1] = 2.                                               # forecast 10 mm BELOW the read-backs: no cost
    assert float(carry_forecast_penalty(lower, recorded, valid, gradient, held)) == 0.
    higher = recorded.clone()
    higher[:, :, 1] = -2.                                             # forecast 10 mm above (the arm sags more)
    higher.requires_grad_(True)
    penalty = carry_forecast_penalty(higher, recorded, valid, gradient, held)
    assert abs(float(penalty) - ((10. - 2.) / 10.) ** 2) < 1e-5
    penalty.backward()
    assert float(higher.grad[:2].abs().sum()) > 0 and float(higher.grad[2].abs().sum()) == 0.
    seeing = carry_forecast_penalty(higher.detach(), recorded, valid, gradient, held, torch.as_tensor([3., 1., 1.]))
    assert abs(float(seeing) - ((10. - 2.) / 10.) ** 2 * 2.) < 1e-5


def test_the_lewm_options_are_checked_before_anything_is_read():
    from scripts.train_v21_horizon40 import main
    base = ['--parent', 'parent.pt', '--source-config', 'config.json', '--output', 'run-that-is-never-made',
            '--lewm-only', '--full-patch-lewm']
    for bad, match in ((['--carry-forecast-weight', '-1'], '--carry-forecast-weight must'),
                       (['--carry-place-factor', '3'], '--carry-place-factor'),
                       (['--object-cues', 'cues'], '--object-cues needs --camera-layout'),
                       (['--carry-forecast-weight', '5'], 'finds held rows')):
        with pytest.raises(ValueError, match=match):
            main(base + bad)
