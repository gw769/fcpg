"""17.167 (11/01; James 「改好阿，可以空著阿大哥，動態的」): the robot service's picture memory for a picture-memory LeWM
(17.166). A runtime's state (fcpg/v21_core.py ActualState.pictures, load_runtime sets core.live_pictures) starts every
task with N empty slots per camera and takes in each new exposure the three-picture memory takes, the oldest leaving
once N are held; predict and complete read it; the planner gives all its candidates one memory, read once, with the
forecasts of every candidate's own call; the snapshots keep its times, joints and validity without the tokens; a
resume restores it, saves from before it load with none; the runtime's loader rebuilds the LeWM. Training keeps no
such memory (its rows' memories come from the recordings). The inference fast path is off, as in the trainer."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_arm_gain import lewm_arm_setting
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_full_patch_lewm import V21FullPatchLeWM
from fcpg.v21_horizon_training import factual_state, horizon_targets
from fcpg.v21_models import CommandTrace
from fcpg.v21_picture_memory import empty_pictures
from test_v21_lewm_picture_memory import as_trained
from test_v21_scratch_features import actual_chain_source

N = 6
RECIPE = dict(controller_input=True, balanced_inputs=True, command_symlog=True, memory_tokens=4)


@pytest.fixture(autouse=True)
def no_attention_fast_path():
    before = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(before)


def live(pictures=N, h=2, live_pictures=True, with_chain=False):
    """A core with a picture-memory LeWM (its FiLM drawn, as trained), a runtime's arm setting, and the state after
    h + 1 observations of two synthetic chains, through the core's own observe / predict / complete."""
    torch.manual_seed(21004)
    source = actual_chain_source()
    source.offsets = np.array([0, len(source.raw_q)])
    source.archive.endpoint_episode_index = np.zeros(8, dtype=np.int64)
    core = V21ResearchCore(V21Config(len(source.vocabulary.words), prediction_horizon=40, object_relative=True,
                                     reader_task=True)).eval()
    core.lewm = as_trained(V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40,
                                            memory_pictures=pictures, **RECIPE)).eval()
    core.arm_setting = lewm_arm_setting(None, device='cpu')
    core.live_pictures = live_pictures
    rows = np.array([0, 1])
    chain = source.chain(rows, 'archive', images=True)
    with torch.no_grad():
        state = factual_state(core, chain, h, images=True)
    return (core, state, horizon_targets(source, rows, 'archive', h)) + ((chain,) if with_chain else ())


def test_a_state_starts_empty_and_keeps_the_last_n_new_exposures():
    core, _, _ = live()
    start = core.initial_state(torch.zeros(1, 6), torch.zeros(1, dtype=torch.float64))
    assert start.pictures.features.shape[:3] == (1, 2, N) and not start.pictures.valid.any()
    for pictures, kept in ((N, 3), (2, 2)):                    # three exposures: all of them / the newest two
        _, state, _ = live(pictures)
        memory, held = state.memory, state.pictures
        assert memory.valid.any() and held.valid.shape == (2, 2, pictures)
        assert not held.valid[..., :pictures - kept].any()
        for name in ('features', 'q_at_image', 'image_times', 'valid'):
            assert torch.equal(getattr(held, name)[:, :, pictures - kept:], getattr(memory, name)[:, :, 3 - kept:]), name
    _, state, _ = live(2, h=0)                                  # one exposure: the oldest slot still empty
    assert torch.equal(state.pictures.valid[..., 1], state.memory.valid[..., 2]) and not state.pictures.valid[..., 0].any()


def test_predict_and_complete_read_the_states_pictures():
    core, state, targets = live()
    with torch.no_grad():
        got = core.predict(state, targets.trace, targets.readback_offsets).response.predicted_q_path
        pictures = state.pictures
        direct = core.lewm(state.q, targets.trace, state.history, targets.readback_offsets, patches=pictures.features,
                           patch_valid=pictures.valid, memory=pictures, now_time=state.time,
                           controller=core.lewm_controller(len(state.q), state.q)).predicted_q_path
        moved = replace(state, pictures=replace(pictures, image_times=pictures.image_times - .7))
        other = core.predict(moved, targets.trace, targets.readback_offsets).response.predicted_q_path
    assert torch.equal(got, direct)
    assert not torch.allclose(got, other)                       # the memory it reads is the state's pictures
    with pytest.raises(ValueError, match="state's picture memory"):
        live(live_pictures=False)                               # training's cores keep none: never a silent default


def test_a_retimed_completion_forecasts_again_from_the_pictures():
    core, state, _, chain = live(h=1, with_chain=True)
    seen = []
    with torch.no_grad():
        forecast = core.predict(state, chain.traces[1], chain.readback_offsets[:, 1, 1:])
        later = chain.readback_offsets[:, 1].clone()
        later[:, 1:] += .01                                     # the readbacks came 10 ms later than planned
        original = core.lewm._memory_bank
        core.lewm._memory_bank = lambda *a: (seen.append(a[0].shape[2]), original(*a))[1]
        try:
            after = core.complete(state, forecast, chain.traces[1], chain.paths[:, 1], state.time[:, None] + later)
        finally:
            del core.lewm._memory_bank
    assert seen == [N] and after.feedback_retimed.all()         # forecast again, from the N-picture memory
    assert after.pictures is state.pictures                     # a completion takes no picture in


def test_the_planner_reads_one_shared_memory_with_every_candidates_forecast():
    from fcpg.v21_alignment_cem import AlignmentCEMAdapter
    core, state, _ = live()
    one = replace(state, **{name: getattr(state, name)[:1] for name in ('q', 'time')},
                  history=type(state.history)(**{k: v[:1] for k, v in vars(state.history).items()}),
                  pictures=type(state.pictures)(state.pictures.features[:1], state.pictures.q_at_image[:1],
                                                state.pictures.image_times[:1], state.pictures.valid[:1], None, None),
                  memory=type(state.memory)(**{k: (v[:1] if isinstance(v, torch.Tensor) else v)
                                               for k, v in vars(state.memory).items()}))
    adapter = AlignmentCEMAdapter.__new__(AlignmentCEMAdapter)
    adapter.dynamics, adapter.bf16 = 'lewm', False
    adapter.service = SimpleNamespace(owner=SimpleNamespace(runtime=SimpleNamespace(core=core)))
    offsets = torch.tensor([[.1, .2, .3, .4, .5]], dtype=torch.float64)
    commands = one.q.double().numpy()[:, None] + np.random.default_rng(1).normal(0, .02, (7, 5, 6))
    seen = []
    original = core.lewm._memory_bank
    core.lewm._memory_bank = lambda *a: (seen.append(len(a[0])), original(*a))[1]
    try:
        paths = adapter._forecast(one, offsets)(commands)
    finally:
        del core.lewm._memory_bank
    assert seen == [1] and paths.shape == (7, 6, 6)            # one memory for the seven candidates
    with torch.no_grad():
        for i in range(7):
            trace = CommandTrace(torch.as_tensor(commands[i:i + 1], dtype=torch.float32), offsets,
                                 torch.ones(1, 5, dtype=torch.bool), 'planned', True)
            own = core.predict(one, trace, offsets + .1).response.predicted_q_path
            np.testing.assert_allclose(paths[i], own[0].numpy(), atol=2e-5, rtol=0)


def test_snapshots_keep_times_joints_and_validity_without_the_tokens():
    from scripts.canary_so101_v21_research import cpu_actual_state
    core, state, targets = live()
    saved = cpu_actual_state(state)
    assert saved.pictures.features.numel() == 0 and saved.memory.features.numel() == state.memory.features.numel()
    for name in ('q_at_image', 'image_times', 'valid'):
        assert torch.equal(getattr(saved.pictures, name), getattr(state.pictures, name))
    with pytest.raises(ValueError, match='picture memory must be DINO patches'):
        core.predict(saved, targets.trace, targets.readback_offsets)


def test_a_resume_restores_the_pictures_and_saves_from_before_load_with_none():
    from fcpg.v21_actual_resume import pack, unpack_state
    _, state, _ = live()
    back = unpack_state(pack(state), 'cpu')
    for name in ('features', 'q_at_image', 'image_times', 'valid'):
        assert torch.equal(getattr(back.pictures, name), getattr(state.pictures, name))
    legacy = pack(replace(state, pictures=None))
    assert legacy.pop('pictures') is None                       # a save written before the field existed
    assert unpack_state(legacy, 'cpu').pictures is None


def package(root, **switches):
    """An overhaul run's package (as test_runtime_loads_an_overhaul_run writes it) with a picture-memory LeWM."""
    import json
    from fcpg.v21_core import CONTRACT, POSITION_UNIT
    from fcpg.v21_runtime import OVERHAUL_KIND
    from test_v21_fcpg_actor_overhaul import build
    from fcpg.v21_data import TaskVocabulary
    vocabulary = TaskVocabulary.build(['pick red block', 'pick up the red fish toy'])
    core = build(vocabulary, actor_history=False, reader_task=True, **switches)
    core.lewm = as_trained(V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40,
                                            memory_pictures=N, **RECIPE))
    payload = dict(version='21.0.4', contract=CONTRACT, position_unit=POSITION_UNIT, config=asdict(core.config),
                   vocabulary=list(vocabulary.words), text_max_length=vocabulary.max_length, model=core.state_dict(),
                   stage='fish', step=3, complete=True, smoke=False,
                   temporal_training_contract=core.temporal_training_contract,
                   temporal_feature_coverage=core.temporal_feature_coverage)
    (root / 'fish').mkdir(parents=True)
    torch.save(payload, root / 'fish' / 'final.pt')
    config = dict(kind=OVERHAUL_KIND, contract=CONTRACT, smoke=False, complete=True, model=asdict(core.config),
                  vocabulary=list(vocabulary.words), fish_updates=3)
    (root / 'config.json').write_text(json.dumps(config))
    return core


def test_the_runtimes_loader_rebuilds_the_lewm_and_keeps_pictures(tmp_path):
    from fcpg.v21_runtime import load_runtime
    core = package(tmp_path)
    runtime = load_runtime(tmp_path, 'pick up the red fish toy', device='cpu',
                           physical_check=lambda *a, **k: None, lease_check=lambda *a, **k: None)
    assert int(runtime.core.lewm.memory_pictures) == N and runtime.core.picture_slots() == N
    loaded = runtime.core.state_dict()
    assert all(torch.equal(loaded[k], v) for k, v in core.state_dict().items())
    state = runtime.core.initial_state(torch.zeros(1, 6), torch.zeros(1, dtype=torch.float64))
    assert state.pictures is not None and not state.pictures.valid.any()
    assert core.picture_slots() == 0                            # a core outside a runtime (training) keeps none
    blank = empty_pictures(1, N)
    assert blank.features.shape == (1, 2, N, *blank.features.shape[3:]) and blank.ids is None


def test_a_picture_memory_lewm_never_runs_on_the_wrist_camera_alone(tmp_path):
    """James 「雙視角雙視角」: a package whose model keeps the wrist camera only is refused for a picture-memory LeWM."""
    from fcpg.v21_runtime import load_runtime
    package(tmp_path, wrist_only=True)
    with pytest.raises(ValueError, match='both cameras'):
        load_runtime(tmp_path, 'pick up the red fish toy', device='cpu',
                     physical_check=lambda *a, **k: None, lease_check=lambda *a, **k: None)
