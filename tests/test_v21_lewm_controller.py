"""17.153: the arm's control setting as LeWM inputs (fcpg/v21_full_patch_lewm.py controller_input=True; the robot blocks'
arm P table, fcpg/v21_robot_block_pack.py): the untrained inputs change nothing, a controller LeWM never runs without
them, only rows that have a setting train its vector, every loader rebuilds it, and the sessions' P is the launch
record's."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from fcpg.v21_full_patch_lewm import CONTROLLER_INPUTS, V21FullPatchLeWM, architecture_from_state
from fcpg.v21_native_horizon import core_from_parent
from fcpg.v21_robot_block_pack import ARM_P_SCHEMA, RobotBlocks, session_arm_p
from v21_lewm_helpers import setup


def call(lewm, state, targets, **kwargs):
    return lewm(state.q, targets.trace, state.history, targets.readback_offsets, patches=state.memory.features,
                patch_valid=state.memory.valid, memory=state.memory, now_time=state.time, **kwargs)


def settings(n):
    return torch.tensor([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])[:n]


def test_untrained_controller_inputs_change_nothing():
    core, state, _, targets, _ = setup()
    plain = core.lewm
    torch.manual_seed(21004)
    lewm = V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40,
                            controller_input=True).eval()
    assert lewm.controller_weight.shape == (len(CONTROLLER_INPUTS), lewm.width) and not lewm.controller_weight.any()
    state_plain = plain.state_dict()
    assert all(torch.equal(v, state_plain[k]) for k, v in lewm.state_dict().items() if k != 'controller_weight')
    with torch.no_grad():
        expected = call(plain, state, targets).predicted_q_path
        for setting in settings(len(state.q)), torch.ones(len(state.q), 2):
            assert torch.equal(call(lewm, state, targets, controller=setting).predicted_q_path, expected)


def test_a_controller_lewm_never_runs_without_its_setting_and_a_plain_one_refuses_it():
    core, state, _, targets, _ = setup()
    with pytest.raises(ValueError, match='not built'):
        call(core.lewm, state, targets, controller=torch.zeros(len(state.q), 2))
    core, state, _, targets, _ = setup(controller_input=True)
    lewm = core.lewm
    with pytest.raises(ValueError, match='pass controller'):
        call(lewm, state, targets)
    with pytest.raises(ValueError, match='pass controller'):          # blind calls too
        lewm(state.q, targets.trace, state.history, targets.readback_offsets)
    with pytest.raises(ValueError, match='0 or 1'):
        call(lewm, state, targets, controller=torch.full((len(state.q), 2), .5))
    with pytest.raises(ValueError):
        call(lewm, state, targets, controller=torch.zeros(len(state.q), 3))
    with pytest.raises(ValueError, match='explicit booleans'):
        V21FullPatchLeWM(horizon=40, controller_input=1)


def test_only_rows_with_a_setting_train_its_vector_and_a_set_vector_moves_them():
    core, state, _, targets, _ = setup(controller_input=True)
    lewm = core.lewm.train()
    zero = torch.zeros(len(state.q), 2)
    call(lewm, state, targets, controller=zero).predicted_q_path.sum().backward()
    assert lewm.controller_weight.grad is None or not lewm.controller_weight.grad.any()
    lewm.zero_grad()
    one = settings(len(state.q))
    call(lewm, state, targets, controller=one).predicted_q_path.sum().backward()
    grad = lewm.controller_weight.grad
    assert grad[0].abs().sum() > 0 and grad[1].abs().sum() > 0
    lewm.eval()
    with torch.no_grad():
        lewm.controller_weight[0] = .5
        before = call(lewm, state, targets, controller=zero).predicted_q_path
        after = call(lewm, state, targets, controller=torch.tensor([[1., 0.]]).expand(len(state.q), 2)).predicted_q_path
        assert not torch.equal(before, after)


def test_every_loader_rebuilds_a_controller_lewm():
    core, state, _, targets, _ = setup(controller_input=True)
    with torch.no_grad():
        core.lewm.controller_weight.normal_()
    whole = core.state_dict()
    assert architecture_from_state(whole, 'lewm.') == dict(width=384, layers=4, controller_input=True)
    rebuilt = core_from_parent(dict(config=asdict(core.config), model=whole)).eval()
    assert rebuilt.lewm.controller_input and rebuilt.lewm.architecture()['controller_input']
    setting = settings(len(state.q))
    with torch.no_grad():
        torch.testing.assert_close(call(rebuilt.lewm, state, targets, controller=setting).predicted_q_path,
                                   call(core.lewm, state, targets, controller=setting).predicted_q_path, atol=0, rtol=0)
    with pytest.raises(RuntimeError):
        V21FullPatchLeWM(horizon=40).load_state_dict(core.lewm.state_dict(), strict=True)


def test_the_core_gives_a_controller_lewm_the_setting_its_runtime_declared():
    """V21ResearchCore.predict (the service's forecast) passes the declared setting [P 32, 1] to a controller LeWM,
    the same as a direct call with it, and refuses to guess one."""
    from fcpg.v21_arm_gain import lewm_arm_setting
    core, state, _, targets, _ = setup(controller_input=True)
    with torch.no_grad():
        core.lewm.controller_weight.normal_()
    with pytest.raises(ValueError, match='arm_p_coefficient'):
        core.predict(state, targets.trace, targets.readback_offsets)
    assert torch.equal(lewm_arm_setting(None), torch.tensor([0., 1.]))
    assert torch.equal(lewm_arm_setting(32), torch.tensor([1., 1.]))
    with pytest.raises(ValueError):
        lewm_arm_setting(16)                  # the service's 16 is None (LeRobot's), as --arm-p-coefficient
    outputs = []
    for value in (None, 32):
        core.arm_setting = lewm_arm_setting(value)
        with torch.no_grad():
            got = core.predict(state, targets.trace, targets.readback_offsets).response.predicted_q_path
            expected = call(core.lewm, state, targets,
                            controller=core.arm_setting.expand(len(state.q), -1)).predicted_q_path
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
        outputs.append(got)
    assert not torch.equal(*outputs)


def test_the_sessions_arm_p_is_the_launch_records(tmp_path):
    (tmp_path / 'a.launch.txt').write_text('started 2026-10-27 mode control ... arm P 32 (17.92) steer approach on\n')
    (tmp_path / 'b.launch.txt').write_text('started 2026-10-30 mode control ... arm P 16 (lerobot) (17.92) steer\n')
    (tmp_path / 'c.launch.txt').write_text('started 2026-10-24 mode control ... grip latch (17.45)\n')
    (tmp_path / 'd.launch.txt').write_text('arm P 32 (17.92) ... arm P 16 (lerobot)\n')
    assert session_arm_p(tmp_path, 'a') == (32, 'arm P 32')
    assert session_arm_p(tmp_path, 'b') == (16, 'arm P 16')
    assert session_arm_p(tmp_path, 'c')[0] == 16
    with pytest.raises(ValueError):
        session_arm_p(tmp_path, 'd')
    with pytest.raises(ValueError):
        session_arm_p(tmp_path, 'missing')


def test_robot_blocks_give_each_block_its_sessions_setting():
    sessions = ['s16', 's32']
    session = torch.tensor([0, 1, 1, 0])
    pack = dict(schema='v21-h40-robot-block-pack-v1', split='train', sessions=sessions, days=['20261027'],
                tasks=['pick up the red fish toy and place it in the cardboard box'],
                data=dict(q=torch.zeros(4, 6), holding=torch.zeros(4, dtype=torch.bool), session=session))
    plain = RobotBlocks(pack, 'cpu')
    with pytest.raises(ValueError, match='no arm P table'):
        plain.controller([0])
    with pytest.raises(ValueError, match='every session'):
        RobotBlocks(pack, 'cpu', arm_p=dict(schema=ARM_P_SCHEMA, sessions=dict(s16=16)))
    blocks = RobotBlocks(pack, 'cpu', arm_p=dict(schema=ARM_P_SCHEMA, sessions=dict(s16=16, s32=32, other=32)))
    assert blocks.record['arm_p32_blocks'] == 2
    assert torch.equal(blocks.controller(np.array([0, 1, 2, 3])),
                       torch.tensor([[0., 1.], [1., 1.], [1., 1.], [0., 1.]]))
