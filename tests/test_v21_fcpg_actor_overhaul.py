"""docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md: the new switches change nothing for earlier models, the
history-free Actor ignores how the arm just moved, the task-reading FCPG depends on the task, and the Archive camera
layout puts the wrist camera in slot 0."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_archive_camera_layout import ArchiveCameraLayout
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_horizon_training import factual_state
from fcpg.v21_memory import ReadbackMemoryReader
from fcpg.v21_temporal_contract import PROGRESS_TRAINING_POLICY, begin_training_temporal_contract
from fcpg.v21_training import synthetic_chain


def build(vocabulary, **switches):
    torch.manual_seed(5)
    core = V21ResearchCore(V21Config(len(vocabulary.words), object_relative=True, prediction_horizon=40,
                                     **switches)).eval()
    begin_training_temporal_contract(core, PROGRESS_TRAINING_POLICY)
    return core


@pytest.fixture
def vocabulary():
    return TaskVocabulary.build(['pick red block', 'pick up the red fish toy'])


def test_defaults_keep_earlier_models_unchanged(vocabulary):
    torch.manual_seed(5)
    old = V21ResearchCore(V21Config(len(vocabulary.words), object_relative=True, prediction_horizon=40))
    assert old.config.actor_history and not old.config.reader_task
    assert not any(name.startswith(('reader.task', 'reader.pose')) for name in old.state_dict())
    # an earlier checkpoint's saved config (no new fields) still builds the same model
    saved = {k: v for k, v in vars(old.config).items()
             if k not in ('actor_history', 'reader_task', 'gripper_separate', 'wrist_only', 'actor_width')}
    assert V21Config(**saved) == old.config
    assert not old.config.gripper_separate and not old.config.wrist_only
    # the second-round switches add no parameters and need the history-free model
    torch.manual_seed(5)
    new = V21ResearchCore(V21Config(len(vocabulary.words), object_relative=True, prediction_horizon=40,
                                    actor_history=False, gripper_separate=True, wrist_only=True))
    assert {k: v.shape for k, v in new.state_dict().items()} == {k: v.shape for k, v in old.state_dict().items()}
    with pytest.raises(ValueError, match='gripper_separate'):
        V21Config(len(vocabulary.words), gripper_separate=True)


def test_old_reader_ignores_task_ids(vocabulary):
    chain = synthetic_chain(vocabulary, batch=2)
    core = build(vocabulary)
    with torch.no_grad():
        state = factual_state(core, chain, 2, images=True)
        g = core.reader(state.memory, state.q, state.time, state.history)
        g_task = core.reader(state.memory, state.q, state.time, state.history,
                             task_ids=vocabulary.encode(['pick up the red fish toy'] * 2))
    torch.testing.assert_close(g, g_task, rtol=0, atol=0)


def test_task_reader_needs_and_uses_the_task(vocabulary):
    chain = synthetic_chain(vocabulary, batch=2)
    core = build(vocabulary, reader_task=True)
    with torch.no_grad():
        state = factual_state(core, chain, 2, images=True)
        assert torch.equal(state.task_ids, chain.text_ids)
        block = core.reader(state.memory, state.q, state.time, state.history, task_ids=chain.text_ids)
        fish = core.reader(state.memory, state.q, state.time, state.history,
                           task_ids=vocabulary.encode(['pick up the red fish toy'] * 2))
        with pytest.raises(ValueError, match='task'):
            core.reader(state.memory, state.q, state.time, state.history)
    assert (block - fish).abs().max() > 1e-4
    # an actual state keeps its task
    with pytest.raises(ValueError, match='keeps its task'):
        core.observe(state, None, state.q, state.time, task_ids=vocabulary.encode(['pick up the red fish toy'] * 2))


def test_history_free_actor_ignores_how_the_arm_just_moved(vocabulary):
    chain = synthetic_chain(vocabulary, batch=2)
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(2, -1) * .1
    for history, should_change in ((False, False), (True, True)):
        core = build(vocabulary, actor_history=history, reader_task=True)
        with torch.no_grad():
            state = factual_state(core, chain, 2, images=True)
            assert state.feedback_available.all()
            base = core.act(state, chain.text_ids, offsets).commands_position
            moved = replace(state, q_error=state.q_error + 3., history=replace(
                state.history, readbacks_position=state.history.readbacks_position + 2.))
            other = core.act(moved, chain.text_ids, offsets).commands_position
            seen = replace(state, memory=replace(state.memory, features=state.memory.features.flip(0)))
            looked = core.act(seen, chain.text_ids, offsets).commands_position
        assert ((other - base).abs().max() > 1e-5) == should_change
        assert (looked - base).abs().max() > 1e-5


class _Archive:
    def __init__(self):
        # 4 episodes, 2 endpoints each; chain rows start at endpoint 2*e
        self.episodes = [{}] * 4
        self.endpoint_episode_index = np.repeat(np.arange(4), 2)
        self.row_endpoint_index = np.stack([np.arange(0, 8, 2)] * 5, 1)


def test_camera_layout_orders_and_masks(tmp_path):
    path = tmp_path / 'layout.npz'
    np.savez(path, order=np.array([[0, 1], [1, 0], [0, 1], [0, 1]]),
             wrist_present=np.array([True, True, False, False]), decided=np.array([True, True, True, False]))
    layout = ArchiveCameraLayout(path, _Archive())
    ids = np.arange(8)
    tokens = np.zeros((8, 2, 256, 384), np.float16)
    tokens[:, 0] = 1.
    tokens[:, 1] = 2.
    valid = np.ones((8, 2), bool)
    times = np.stack([np.full(8, 10.), np.full(8, 20.)], 1)
    for to in (np.asarray, torch.as_tensor):
        t, v, s = (layout.remap(to(x), ids) for x in (tokens, valid, times))
        t, v, s = (np.asarray(x) for x in (t, v, s))
        assert (t[0:2, 0] == 1).all() and (t[0:2, 1] == 2).all()           # in order
        assert (t[2:4, 0] == 2).all() and (t[2:4, 1] == 1).all()           # swapped
        assert (s[2:4, 0] == 20).all() and v[2:4].all()
        assert (t[4:6, 0] == 0).all() and (t[4:6, 1] == 2).all()           # no wrist camera
        assert not v[4:6, 0].any() and v[4:6, 1].all() and (s[4:6, 0] == 10).all()
        assert (t[6:8] == tokens[6:8]).all() and v[6:8].all()               # undecided: untouched
    assert layout.admitted_rows(_Archive(), np.arange(4)).tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        bad = tmp_path / 'bad.npz'
        np.savez(bad, order=np.array([[0, 1]] * 4), wrist_present=np.array([True, True, True, True]),
                 decided=np.array([True, True, True, False]))
        ArchiveCameraLayout(bad, _Archive())


def test_camera_layout_masks_a_missing_external_camera(tmp_path):
    """17.68: a wrist camera only (the other slot a black video): wrist view in slot 0, slot 1 zero and invalid."""
    path = tmp_path / 'layout.npz'
    np.savez(path, order=np.array([[1, 0], [0, 1], [0, 1], [0, 1]]), wrist_present=np.array([True, True, False, False]),
             decided=np.array([True, True, True, False]), external_present=np.array([False, False, True, True]))
    layout = ArchiveCameraLayout(path, _Archive())
    tokens = np.zeros((8, 2, 256, 384), np.float16)
    tokens[:, 0], tokens[:, 1] = 1., 2.
    valid = np.ones((8, 2), bool)
    for to in (np.asarray, torch.as_tensor):
        t, v = (np.asarray(layout.remap(to(x), np.arange(8))) for x in (tokens, valid))
        assert (t[0:2, 0] == 2).all() and (t[0:2, 1] == 0).all() and v[0:2, 0].all() and not v[0:2, 1].any()
        assert (t[2:4, 0] == 1).all() and (t[2:4, 1] == 0).all() and not v[2:4, 1].any()
        assert (t[4:6, 0] == 0).all() and (t[4:6, 1] == 2).all() and not v[4:6, 0].any() and v[4:6, 1].all()
        assert (t[6:8] == tokens[6:8]).all() and v[6:8].all()
    assert layout.admitted_rows(_Archive(), np.arange(4), wrist=True).tolist() == [0, 1]
    assert layout.coverage(_Archive(), np.arange(4))['no_external'] == 2
    for bad in (dict(external_present=np.array([True, True, True, False])),        # undecided claims a camera gone
                dict(wrist_present=np.array([True, True, False, False]),
                     external_present=np.array([True, True, False, True]))):       # decided with neither camera
        values = dict(order=np.array([[0, 1]] * 4), wrist_present=np.array([True, True, False, False]),
                      decided=np.array([True, True, True, False]), external_present=np.ones(4, bool))
        values.update(bad)
        np.savez(tmp_path / 'bad.npz', **values)
        with pytest.raises(ValueError):
            ArchiveCameraLayout(tmp_path / 'bad.npz', _Archive())


def test_reader_rejects_a_task_reader_without_reserved_tokens():
    with pytest.raises(ValueError):
        ReadbackMemoryReader(task_vocabulary=3)


def test_runtime_loads_an_overhaul_run(tmp_path, vocabulary):
    import json
    from dataclasses import asdict
    from fcpg.v21_core import CONTRACT, POSITION_UNIT
    from fcpg.v21_full_patch_lewm import V21FullPatchLeWM
    from fcpg.v21_runtime import OVERHAUL_KIND, load_runtime
    core = build(vocabulary, actor_history=False, reader_task=True)
    core.lewm = V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40)
    payload = dict(version='21.0.4', contract=CONTRACT, position_unit=POSITION_UNIT, config=asdict(core.config),
                   vocabulary=list(vocabulary.words), text_max_length=vocabulary.max_length, model=core.state_dict(),
                   stage='fish', step=3, complete=True, smoke=False,
                   temporal_training_contract=core.temporal_training_contract,
                   temporal_feature_coverage=core.temporal_feature_coverage)
    (tmp_path / 'fish').mkdir()
    torch.save(payload, tmp_path / 'fish' / 'final.pt')
    config = dict(kind=OVERHAUL_KIND, contract=CONTRACT, smoke=False, complete=True, model=asdict(core.config),
                  vocabulary=list(vocabulary.words), fish_updates=3)
    (tmp_path / 'config.json').write_text(json.dumps(config))
    runtime = load_runtime(tmp_path, 'pick up the red fish toy', device='cpu',
                           physical_check=lambda *a, **k: None, lease_check=lambda *a, **k: None)
    assert runtime.core.config.reader_task and not runtime.core.config.actor_history
    assert runtime.core.temporal_training_contract == core.temporal_training_contract
    loaded = runtime.core.state_dict()
    assert all(torch.equal(loaded[k], v) for k, v in core.state_dict().items())
    (tmp_path / 'config.json').write_text(json.dumps(dict(config, fish_updates=4)))
    with pytest.raises(ValueError, match='incomplete'):
        load_runtime(tmp_path, 'pick up the red fish toy', device='cpu',
                     physical_check=lambda *a, **k: None, lease_check=lambda *a, **k: None)


def test_direct_rows_equal_the_memory_map(tmp_path):
    from fcpg.v21_direct_rows import DirectRows
    values = np.random.default_rng(3).standard_normal((300, 2, 5, 7)).astype(np.float16)
    np.save(tmp_path / 'rows.npy', values)
    mapped = np.load(tmp_path / 'rows.npy', mmap_mode='r')
    direct = DirectRows(mapped, threads=4)
    for rows in (np.arange(40, 140), np.array([7, 8, 9, 200, 3, 4, 299, 0]), np.array([-1, 5]), np.zeros(0, np.int64)):
        assert np.array_equal(direct[rows], values[rows])
    assert np.array_equal(direct[3:9], values[3:9])                         # other indices: the memory map
    assert direct.shape == values.shape and len(direct) == 300
    with pytest.raises(IndexError):
        direct[np.array([300])]
    direct.close()


def test_earlier_models_keep_task_free_states_and_snapshots_roundtrip(vocabulary):
    from fcpg.v21_actual_resume import pack, unpack_state
    chain = synthetic_chain(vocabulary, batch=1)
    with torch.no_grad():
        old = factual_state(build(vocabulary), chain, 1, images=True)
        new = factual_state(build(vocabulary, actor_history=False, reader_task=True), chain, 1, images=True)
    assert old.task_ids is None and torch.equal(new.task_ids, chain.text_ids)
    for state in (old, new):
        data = pack(state)
        back = unpack_state(data, 'cpu')
        assert (back.task_ids is None) == (state.task_ids is None)
        torch.testing.assert_close(back.grounding, state.grounding, rtol=0, atol=0)
    legacy = pack(old)
    del legacy['task_ids']                                   # a snapshot written before the field existed
    assert unpack_state(legacy, 'cpu').task_ids is None


def test_separate_gripper_is_not_read_and_is_output_as_an_opening(vocabulary):
    chain = synthetic_chain(vocabulary, batch=2)
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(2, -1) * .1
    for separate in (True, False):
        core = build(vocabulary, actor_history=False, reader_task=True, gripper_separate=separate)
        torch.nn.init.normal_(core.actor.output.weight, std=.05)             # make the proposal visibly non-trivial
        with torch.no_grad():
            state = factual_state(core, chain, 2, images=True)
            jaw = torch.tensor([0., 0, 0, 0, 0, 17.])
            other_memory = replace(state.memory, q_at_image=state.memory.q_at_image + jaw)
            read = core.reader(state.memory, state.q, state.time, state.history, task_ids=state.task_ids)
            reread = core.reader(other_memory, state.q + jaw, state.time, state.history, task_ids=state.task_ids)
            out = core.act(state, chain.text_ids, offsets)
            opened = core.act(replace(state, q=state.q + jaw), chain.text_ids, offsets)
        base, moved = out.commands_position, opened.commands_position
        if separate:
            torch.testing.assert_close(read, reread, rtol=0, atol=1e-6)      # FCPG does not read the jaw
            torch.testing.assert_close(base, moved, rtol=0, atol=1e-5)       # neither the arm nor the jaw command follow it
            assert (base[..., 5] - base[:, :1, 5]).abs().max() > 1e-4          # the jaw command is still a path
        else:
            assert (read - reread).abs().max() > 1e-6
            assert (base[..., 5] - moved[..., 5]).abs().max() > 1.           # "reading + changes" moves with the reading
        # every consumer may keep reading "pose + summed increments"
        inside = (base > core.actor.position_lower + 1e-3) & (base < core.actor.position_upper - 1e-3)
        rebuilt = state.q[:, None] + out.increments.cumsum(1)
        torch.testing.assert_close(rebuilt[inside], base[inside], rtol=0, atol=1e-3)


def test_wrist_only_never_admits_the_external_camera(vocabulary):
    chain = synthetic_chain(vocabulary, batch=2)
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(2, -1) * .1
    core = build(vocabulary, actor_history=False, reader_task=True, wrist_only=True)
    other = replace(chain, features=torch.cat((chain.features[:, :, :1], chain.features[:, :, 1:].flip(0) + 1.), 2))
    with torch.no_grad():
        state = factual_state(core, chain, 2, images=True)
        changed = factual_state(core, other, 2, images=True)
        assert state.memory.valid[:, 0].any() and not state.memory.valid[:, 1].any()
        assert not state.memory.features[:, 1].any()
        base = core.act(state, chain.text_ids, offsets).commands_position
        same = core.act(changed, chain.text_ids, offsets).commands_position
    torch.testing.assert_close(base, same, rtol=0, atol=0)


def test_rows_with_a_wrist_camera_and_collections(tmp_path):
    path = tmp_path / 'layout.npz'
    np.savez(path, order=np.array([[0, 1], [1, 0], [0, 1], [0, 1]]),
             wrist_present=np.array([True, True, False, False]), decided=np.array([True, True, True, False]))
    archive = _Archive()
    archive.episodes = [dict(manifest_row=dict(dataset_root=root)) for root in (
        '/data/so101/raw_expand/IRA_lego_so101/session_1', '/data/so101/raw_expand/IRA_desk_cleanup_so101/session_2',
        '/data/so101/raw_expand/bv_so101', '/data/so101/raw_expand/so101_pi05_base_dataset')]
    layout = ArchiveCameraLayout(path, archive)
    assert layout.admitted_rows(archive, np.arange(4), wrist=True).tolist() == [0, 1]
    names, index = layout.row_collections(archive, np.array([3, 0, 1, 2]))
    assert names == ['IRA', 'bv_so101', 'so101_pi05_base_dataset'] and index.tolist() == [2, 0, 0, 1]


def test_larger_model_reads_longer_summaries_with_a_wider_actor(vocabulary):
    chain = synthetic_chain(vocabulary, batch=2)
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(2, -1) * .1
    small = build(vocabulary, actor_history=False, reader_task=True)
    large = build(vocabulary, actor_history=False, reader_task=True, queries_per_view=32, actor_width=384)
    assert small.config.actor_width == 192 and small.config.queries_per_view == 8
    with torch.no_grad():
        state = factual_state(large, chain, 2, images=True)
        assert state.grounding.shape == state.visual_error.shape == (2, 2, 32, 192)
        out = large.act(state, chain.text_ids, offsets).commands_position
        seen = replace(state, memory=replace(state.memory, features=state.memory.features.flip(0)))
        assert large.reader(seen.memory, seen.q, seen.time, seen.history, task_ids=seen.task_ids).shape == (2, 2, 32, 192)
    assert out.shape == (2, 40, 6) and torch.isfinite(out).all()
    count = lambda module: sum(p.numel() for p in module.parameters())
    assert count(large.actor) > 3 * count(small.actor) and count(large.reader) > count(small.reader)
    assert set(large.state_dict()) == set(small.state_dict())               # same parts, other sizes
    for bad in (dict(queries_per_view=16), dict(actor_width=256)):
        with pytest.raises(ValueError):
            V21Config(len(vocabulary.words), **bad)


def test_actor_v2_layers_decode_the_whole_chunk_together(vocabulary):
    # section 17.75: 'v2' = 4 context layers, command queries attending both ways, dropout 0.1; 'v1' is every earlier model
    chain = synthetic_chain(vocabulary, batch=2)
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(2, -1) * .1
    later = offsets.clone()
    later[:, -1] += .05                                     # only the last command's time moves
    switches = dict(actor_history=False, reader_task=True, queries_per_view=32, actor_width=384)
    v1, v2 = build(vocabulary, **switches), build(vocabulary, actor_layers='v2', **switches)
    assert v1.config.actor_layers == 'v1' and isinstance(v2.actor.context, torch.nn.ModuleList)
    assert len(v2.actor.context) == 4 and not any(k.startswith('actor.context.0.') for k in v1.state_dict())
    assert {k for k in v1.state_dict() if not k.startswith('actor.')} == {k for k in v2.state_dict() if not k.startswith('actor.')}
    with torch.no_grad():
        state = factual_state(v2, chain, 2, images=True)
        first = {name: core.act(state, chain.text_ids, offsets).commands_position for name, core in (('v1', v1), ('v2', v2))}
        moved = {name: core.act(state, chain.text_ids, later).commands_position for name, core in (('v1', v1), ('v2', v2))}
        torch.testing.assert_close(moved['v1'][:, :39], first['v1'][:, :39], rtol=0, atol=0)   # v1 looks only back
        assert (moved['v2'][:, 0] - first['v2'][:, 0]).abs().max() > 0                          # v2 looks both ways
        torch.testing.assert_close(v2.act(state, chain.text_ids, offsets).commands_position, first['v2'], rtol=0, atol=0)
        v2.actor.train()                                                                       # dropout only in training
        assert (v2.act(state, chain.text_ids, offsets).commands_position - first['v2']).abs().max() > 0
    with pytest.raises(ValueError, match='actor_layers'):
        V21Config(len(vocabulary.words), actor_layers='v3')


def test_saved_configs_from_before_actor_layers_compare_equal(vocabulary):
    # a checkpoint saved before section 17.75 has no actor_layers; --archive-checkpoint / --continue-archive must still
    # accept it for the same model built now
    from dataclasses import asdict
    from scripts.train_v21_fcpg_actor import saved_model_config
    config = asdict(V21Config(len(vocabulary.words), actor_history=False, gripper_separate=True, wrist_only=True))
    earlier = {k: v for k, v in config.items() if k != 'actor_layers'}
    assert saved_model_config(earlier) == config and saved_model_config(None) is None
    assert saved_model_config(dict(earlier, actor_layers='v2'))['actor_layers'] == 'v2'


def test_weight_average_follows_swaps_back_and_keeps():
    from scripts.train_v21_fcpg_actor import WeightAverage, main
    layer = torch.nn.Linear(3, 2)
    start = [p.detach().clone() for p in layer.parameters()]
    average = WeightAverage(list(layer.parameters()), .9)
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(1.)
    average.update()                                        # first update: decay min(.9, 2/11)
    for a, s, p in zip(average.average, start, layer.parameters()):
        torch.testing.assert_close(a, s * (2 / 11) + p.detach() * (9 / 11))
    trained = [p.detach().clone() for p in layer.parameters()]
    with average.swapped():
        for p, a in zip(layer.parameters(), average.average):
            torch.testing.assert_close(p.detach(), a, rtol=0, atol=0)
    for p, t in zip(layer.parameters(), trained):
        torch.testing.assert_close(p.detach(), t, rtol=0, atol=0)
    average.keep()
    for p, a in zip(layer.parameters(), average.average):
        torch.testing.assert_close(p.detach(), a, rtol=0, atol=0)
    common = ['--source-config', 'c.json', '--camera-layout', 'l.npz', '--lewm', 'lewm.pt', '--output', 'run-that-is-never-made']
    for bad in (['--ema', '1'], ['--ema', '-.1'], ['--ema', '.999', '--archive-checkpoint', 'f.pt']):
        with pytest.raises(ValueError, match='--ema'):
            main(common + bad)


def test_best_weights_keep_the_lowest_error_and_count_the_evaluations_since():
    from scripts.train_v21_fcpg_actor import BestWeights, main
    layer = torch.nn.Linear(3, 2)
    best = BestWeights(list(layer.parameters()))
    seen = {}
    for step, error in ((500, 3.), (1000, 2.), (1500, 2.5), (2000, 2.), (2500, float('nan'))):
        with torch.no_grad():
            for p in layer.parameters():
                p.add_(1.)
        seen[step] = [p.detach().clone() for p in layer.parameters()]
        since = best.offer(step, error)
    assert (best.step, best.error, since) == (1000, 2., 3)     # an equal error and a NaN do not take its place
    best.restore()
    for p, s in zip(layer.parameters(), seen[1000]):
        torch.testing.assert_close(p.detach(), s, rtol=0, atol=0)
    common = ['--source-config', 'c.json', '--camera-layout', 'l.npz', '--lewm', 'lewm.pt', '--output', 'run-that-is-never-made']
    for bad, match in ((['--archive-patience', '-1'], '--archive-patience'), (['--archive-patience', '3'], '--archive-patience'),
                       (['--archive-select', 'development', '--archive-checkpoint', 'f.pt'], '--archive-select')):
        with pytest.raises(ValueError, match=match):
            main(common + bad)


def _targets(chain, steps=40):
    from fcpg.v21_horizon_training import HorizonTargets
    from fcpg.v21_models import CommandTrace
    b = len(chain.q)
    commands = chain.q[:, -1:, :].expand(-1, steps, -1) + torch.linspace(0, 1, steps)[None, :, None]
    valid = torch.ones(b, steps, dtype=torch.bool)
    valid[0, -5:] = False
    offsets = (torch.arange(1, steps + 1, dtype=torch.float64) * .1)[None].expand(b, -1)
    trace = CommandTrace(commands[:, :10].clone(), (torch.arange(10, dtype=torch.float64) * .1)[None].expand(b, -1).clone(),
                         torch.ones(b, 10, dtype=torch.bool), 'recorded', True)
    return HorizonTargets(trace, offsets.clone(), offsets.clone(), commands.clone(), commands.clone() + .5, valid,
                          np.arange(b))


def test_mirror_rows_flip_the_scene_consistently(vocabulary):
    from fcpg.v21_mirror import mirror_rows, LIMIT
    chain = synthetic_chain(vocabulary, batch=3)
    cue = torch.zeros(3, 5, 8)
    cue[..., 0], cue[..., 2], cue[..., 3], cue[..., 7] = 1., .2, .1, .3
    chain = replace(chain, object_relative=cue)
    targets = _targets(chain)
    centre = torch.tensor([0., -103., 0.])                 # row 1: roll near 0 would leave the range when mirrored
    asked = torch.tensor([True, True, False])
    mirrored, mirrored_targets, done = mirror_rows(chain, targets, asked, centre)
    assert done.tolist() == [True, False, False]
    for before, after in ((chain.q, mirrored.q), (chain.commands, mirrored.commands), (chain.paths, mirrored.paths),
                          (targets.commands[:, :-5], mirrored_targets.commands[:, :-5])):
        torch.testing.assert_close(after[0][..., 0], -before[0][..., 0], rtol=0, atol=0)
        torch.testing.assert_close(after[0][..., 4], 0. - before[0][..., 4], rtol=0, atol=0)
        torch.testing.assert_close(after[0][..., 1:4], before[0][..., 1:4], rtol=0, atol=0)
        torch.testing.assert_close(after[0][..., 5], before[0][..., 5], rtol=0, atol=0)
        torch.testing.assert_close(after[1:], before[1:], rtol=0, atol=0)
    grid = chain.features[0].reshape(5, 2, 16, 16, 384)
    torch.testing.assert_close(mirrored.features[0].reshape(5, 2, 16, 16, 384), grid.flip(3), rtol=0, atol=0)
    torch.testing.assert_close(mirrored.features[1:], chain.features[1:], rtol=0, atol=0)
    assert mirrored.object_relative[0, 0, 2] == -.2 and mirrored.object_relative[0, 0, 7] == -.3
    assert mirrored.object_relative[0, 0, 3] == .1 and torch.equal(mirrored.object_relative[1:], cue[1:])
    # padding of the targets stays as it was
    assert torch.equal(mirrored_targets.commands[0, -5:], targets.commands[0, -5:])
    # twice is the original
    back, back_targets, _ = mirror_rows(mirrored, mirrored_targets, asked, centre)
    torch.testing.assert_close(back.q, chain.q, rtol=0, atol=1e-5)
    torch.testing.assert_close(back_targets.readbacks, targets.readbacks, rtol=0, atol=1e-5)
    assert torch.equal(back.features, chain.features)
    assert (mirrored.q.abs() <= LIMIT).all()
    # a mirrored chain is still a consistent factual history
    core = build(vocabulary, actor_history=False, reader_task=True, wrist_only=True)
    with torch.no_grad():
        state = factual_state(core, mirrored, 3, images=True)
    assert torch.isfinite(state.grounding).all()


def test_reading_only_in_the_last_observation_gives_the_same_state(vocabulary):
    from scripts.train_v21_fcpg_actor import final_read_only
    chain = synthetic_chain(vocabulary, batch=2)
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(2, -1) * .1
    core = build(vocabulary, actor_history=False, reader_task=True, gripper_separate=True)
    calls, queries = [], core.reader._queries
    core.reader._queries = lambda *args, **kwargs: (calls.append(1), queries(*args, **kwargs))[1]   # real reads only
    for h in range(4):
        with torch.no_grad():
            calls.clear()
            full = factual_state(core, chain, h, images=True)
            full_calls = len(calls)
            calls.clear()
            with final_read_only(core, h):
                fast = factual_state(core, chain, h, images=True)
            fast_calls = len(calls)
            assert 'forward' not in core.reader.__dict__ and 'observe' not in core.__dict__
            torch.testing.assert_close(fast.grounding, full.grounding, rtol=0, atol=1e-6)
            torch.testing.assert_close(fast.visual_error, full.visual_error, rtol=0, atol=1e-6)
            torch.testing.assert_close(core.act(fast, chain.text_ids, offsets).commands_position,
                                       core.act(full, chain.text_ids, offsets).commands_position, rtol=0, atol=1e-6)
        assert full_calls == 2 * h + 1 and fast_calls == min(h + 1, 2)


def test_idle_start_is_found_from_the_commands_and_trimmed_per_context():
    from scripts.train_v21_fcpg_actor import command_onsets, moving_rows
    # demonstrations 0, 1 and 3 hold their first command for 60, 10 and 20 frames; episode 2 is never read
    lengths, held = {0: 120, 1: 100, 2: 90, 3: 110}, {0: 60, 1: 10, 3: 20}
    episodes = np.concatenate([np.full(n, e) for e, n in lengths.items()])
    actions = np.zeros((len(episodes), 6), np.float32)
    for e, n in held.items():
        frames = np.flatnonzero(episodes == e)
        actions[frames[n:], 2] = np.arange(1, len(frames) - n + 1) * .6   # first frame above one unit: n + 1
    read = []

    class Fish:
        def __init__(self):
            self.episodes, self.actions = episodes, actions
            self.starts = np.flatnonzero((episodes != 2) & (np.arange(len(episodes)) % 3 == 0))

        def validate_frames(self, frames):
            read.append(np.asarray(frames))
            assert not np.isin(episodes[frames], [2]).any()

    source = SimpleNamespace(fish=Fish())
    onsets = command_onsets(source)
    first = {e: int(np.flatnonzero(episodes == e)[0]) for e in held}
    assert onsets == {e: first[e] + held[e] + 1 for e in held}
    rows = np.arange(len(source.fish.starts))
    moving = moving_rows(source, rows, onsets)
    for h in range(4):
        current = source.fish.starts[moving[h]] + 15 * h
        assert (current >= np.asarray([onsets[e] for e in episodes[source.fish.starts[moving[h]]]])).all()
        left_out = np.setdiff1d(rows, moving[h])
        assert (source.fish.starts[left_out] + 15 * h < np.asarray([onsets[e] for e in episodes[source.fish.starts[left_out]]])).all()
    assert len(moving[0]) < len(moving[3]) < len(rows)


def test_fish_horizon_cuts_the_labels_to_the_first_commands():
    from fcpg.v21_horizon_training import HorizonTargets
    from fcpg.v21_models import CommandTrace
    from scripts.train_v21_fcpg_actor import first_commands
    b = 3
    trace = CommandTrace(torch.zeros(b, 8, 6), torch.zeros(b, 8, dtype=torch.float64), torch.ones(b, 8, dtype=torch.bool),
                         'recorded', True)
    times = torch.arange(1, 41, dtype=torch.float64)[None].expand(b, -1) * .1
    valid = torch.ones(b, 40, dtype=torch.bool)
    valid[0, 7:] = False                                                    # an episode tail stays masked
    target = HorizonTargets(trace, times - .1, times, torch.randn(b, 40, 6), torch.randn(b, 40, 6), valid,
                            np.zeros(b, np.int64))
    assert first_commands(target, 40) is target                             # the default changes nothing
    cut = first_commands(target, 5)
    for name in ('command_offsets', 'readback_offsets', 'commands', 'readbacks', 'valid'):
        torch.testing.assert_close(getattr(cut, name), getattr(target, name)[:, :5], rtol=0, atol=0)
    assert cut.trace is target.trace
    for bad in (0, 41):
        with pytest.raises(ValueError):
            first_commands(target, bad)


def test_archive_horizon_is_an_archive_stage_switch(tmp_path):
    # section 17.50: --archive-horizon 5 supervises the Archive stage's first 5 commands; the default keeps all 40, and
    # a run that skips the Archive stage refuses it instead of recording a horizon it never used
    from scripts.train_v21_fcpg_actor import main, parser
    common = ['--source-config', str(tmp_path / 'c.json'), '--camera-layout', str(tmp_path / 'l.npz'),
              '--lewm', str(tmp_path / 'lewm.pt'), '--output', str(tmp_path / 'run')]
    assert parser().parse_args(common).archive_horizon == 40
    assert parser().parse_args(common + ['--archive-horizon', '5']).archive_horizon == 5
    with pytest.raises(SystemExit):
        parser().parse_args(common + ['--archive-horizon', '10'])
    for skip in ('--archive-checkpoint', '--evaluate-checkpoint'):
        with pytest.raises(ValueError, match='--archive-horizon'):
            main(common + ['--archive-horizon', '5', skip, str(tmp_path / 'final.pt')])


def test_archive_learning_rate_rises_then_falls_to_zero(tmp_path):
    # section 17.74: BERT's schedule over the Archive stage; the default keeps the rate constant, as every earlier run
    from scripts.train_v21_fcpg_actor import archive_learning_rate_factor, main
    assert [archive_learning_rate_factor(s, 10, 0, 'constant') for s in (1, 5, 10)] == [1., 1., 1.]
    rates = [archive_learning_rate_factor(s, 10, 2, 'warmup-linear') for s in range(1, 11)]
    assert rates[:2] == [.5, 1.] and rates[-1] == 0. and all(a > b for a, b in zip(rates[1:], rates[2:]))
    assert archive_learning_rate_factor(6, 10, 2, 'warmup-linear') == pytest.approx(.5)
    for warmup in (0, 10, 11):
        with pytest.raises(ValueError, match='--archive-warmup'):
            archive_learning_rate_factor(1, 10, warmup, 'warmup-linear')
    common = ['--source-config', str(tmp_path / 'c.json'), '--camera-layout', str(tmp_path / 'l.npz'),
              '--lewm', str(tmp_path / 'lewm.pt'), '--output', str(tmp_path / 'run')]
    with pytest.raises(ValueError, match='--archive-warmup'):
        main(common + ['--archive-warmup', '100'])
    for skip in ('--archive-checkpoint', '--evaluate-checkpoint'):
        with pytest.raises(ValueError, match='--archive-schedule'):
            main(common + ['--archive-schedule', 'warmup-linear', '--archive-warmup', '100', skip, str(tmp_path / 'f.pt')])


def test_continue_archive_trains_the_archive_stage_further(tmp_path):
    # section 17.72: --continue-archive runs the Archive stage from saved weights, so the switches that skip that stage
    # are refused with it, before anything is read
    from scripts.train_v21_fcpg_actor import main, parser
    common = ['--source-config', str(tmp_path / 'c.json'), '--camera-layout', str(tmp_path / 'l.npz'),
              '--lewm', str(tmp_path / 'lewm.pt'), '--output', str(tmp_path / 'run')]
    assert parser().parse_args(common).continue_archive is None
    assert str(parser().parse_args(common + ['--continue-archive', 'a/final.pt']).continue_archive) == 'a/final.pt'
    for skip in ('--archive-checkpoint', '--evaluate-checkpoint'):
        with pytest.raises(ValueError, match='--continue-archive'):
            main(common + ['--continue-archive', str(tmp_path / 'final.pt'), skip, str(tmp_path / 'other.pt')])


def test_view_dropout_hides_one_camera_per_row_for_the_whole_chain():
    # section 17.52: one draw per row; the external camera below the share, the wrist camera from the share to twice it,
    # only in rows with both cameras; a hidden camera is invalid in every frame and its features zero
    from dataclasses import dataclass
    from scripts.train_v21_fcpg_actor import drop_views

    @dataclass
    class Chain:
        view_valid: torch.Tensor
        features: torch.Tensor

    n = 4000
    valid = torch.ones(n, 5, 2, dtype=torch.bool)
    valid[:500, :, 0] = False                       # no wrist camera: nothing is hidden
    valid[500:1000, :, 1] = False                   # no external camera: nothing is hidden
    chain = Chain(valid, torch.ones(n, 5, 2, 3, 4))
    generator = torch.Generator().manual_seed(1)
    out, external, wrist = drop_views(chain, .2, generator)
    hidden = valid.any(1) & ~out.view_valid.any(1)
    assert int(hidden[:, 1].sum()) == external and int(hidden[:, 0].sum()) == wrist
    assert not (hidden[:, 0] & hidden[:, 1]).any() and out.view_valid.any(1).any(1).all()
    assert not hidden[:1000].any()
    both = slice(1000, n)
    for slot in (0, 1):
        share = float(hidden[both, slot].float().mean())
        assert .17 < share < .23
        assert (out.features[hidden[:, slot]][:, :, slot] == 0).all()
    kept = ~hidden.any(1)
    assert torch.equal(out.features[kept], chain.features[kept]) and torch.equal(out.view_valid[kept], valid[kept])
    same, *_ = drop_views(chain, 0., generator)
    assert same is chain


def test_two_camera_switches_are_refused_where_they_mean_nothing(tmp_path):
    # section 17.52: --mirror-external and --view-dropout are for a model that reads both cameras
    from scripts.train_v21_fcpg_actor import main, parser
    common = ['--source-config', str(tmp_path / 'c.json'), '--camera-layout', str(tmp_path / 'l.npz'),
              '--lewm', str(tmp_path / 'lewm.pt'), '--output', str(tmp_path / 'run')]
    args = parser().parse_args(common)
    assert args.view_dropout == 0. and not args.mirror_external
    for bad, words in ((['--mirror', '--mirror-external', '--wrist-only'], '--mirror-external'),
                       (['--mirror-external'], '--mirror-external'),
                       (['--view-dropout', '.2', '--wrist-only'], '--view-dropout'),
                       (['--view-dropout', '.5'], '--view-dropout'),
                       (['--wrist-hold-back', '1'], '--wrist-hold-back'),
                       (['--wrist-hold-back', '-.1'], '--wrist-hold-back'),
                       (['--exam-hold-back'], '--exam-hold-back')):
        with pytest.raises(ValueError, match=words):
            main(common + bad)


def test_wrist_hold_back_hides_the_newest_wrist_pictures_of_a_share_of_rows():
    # section 17.63: per row, below the share, the newest k wrist pictures (k uniform in 1..min(3, h)) become invalid
    # with zero features; only rows whose wrist picture k blocks earlier is valid; the external camera never changes
    from dataclasses import dataclass
    from scripts.train_v21_fcpg_actor import hold_back_wrist

    @dataclass
    class Chain:
        view_valid: torch.Tensor
        features: torch.Tensor

    n, h = 6000, 3
    valid = torch.ones(n, 5, 2, dtype=torch.bool)
    valid[:600, :, 0] = False                       # no wrist camera: nothing is held back
    chain = Chain(valid, torch.ones(n, h + 1, 2, 3, 4))
    out, counts = hold_back_wrist(chain, h, .3, torch.Generator().manual_seed(1))
    lost = valid[:, :, 0] & ~out.view_valid[:, :, 0]                  # [rows, frames]
    held, k = lost.any(1), lost.sum(1)
    assert torch.equal(out.view_valid[..., 1], valid[..., 1])
    assert torch.equal(out.features[:, :, 1], chain.features[:, :, 1])
    assert not held[:600].any() and [int((k == m).sum()) for m in (1, 2, 3)] == counts
    assert .27 < float(held[600:].float().mean()) < .33
    for m in (1, 2, 3):
        rows = k == m
        assert .28 < counts[m - 1] / sum(counts) < .39                 # about a third each
        assert lost[rows][:, h - m + 1:h + 1].all() and not lost[rows][:, :h - m + 1].any()
        assert not lost[rows][:, h + 1:].any()
        assert (out.features[rows][:, h - m + 1:, 0] == 0).all() and (out.features[rows][:, :h - m + 1, 0] == 1).all()
    assert torch.equal(out.features[~held], chain.features[~held])
    # one completed block: always k = 1, only the current frame
    one, counts = hold_back_wrist(Chain(valid, torch.ones(n, 2, 2, 3, 4)), 1, .3, torch.Generator().manual_seed(2))
    lost = valid[:, :, 0] & ~one.view_valid[:, :, 0]
    assert counts[0] == int(lost.any(1).sum()) > 0 and counts[1:] == [0, 0] and not lost[:, [0, 2, 3, 4]].any()
    # no completed block or a zero share: the same chain
    for h0, share in ((0, .3), (2, 0.)):
        same, counts = hold_back_wrist(chain, h0, share, torch.Generator().manual_seed(3))
        assert same is chain and counts == [0, 0, 0]
    # the picture k blocks earlier must exist: without a wrist picture one block ago, k = 1 never happens
    gap = valid.clone()
    gap[:, h - 1, 0] = False
    _, counts = hold_back_wrist(Chain(gap, torch.ones(n, h + 1, 2, 3, 4)), h, .9, torch.Generator().manual_seed(4))
    assert counts[0] == 0 and counts[1] > 0 and counts[2] > 0


def test_a_held_back_wrist_picture_reaches_the_actor_with_the_joints_at_its_exposure(vocabulary):
    # section 17.63: after the newest k wrist pictures are held back, the wrist memory ends with the picture of k blocks
    # earlier (its features and the joints at its exposure, 5 k readbacks old, still within what the Actor may use), no
    # wrist picture is new at the current frame and no wrist object cue is current; the external camera stays new
    from scripts.train_v21_fcpg_actor import hold_back_wrist
    from fcpg.v21_temporal_contract import temporal_actor_state
    batch, h = 12, 3
    chain = synthetic_chain(vocabulary, batch=batch)
    chain = replace(chain, object_relative=torch.full((batch, 5, 8), .5))
    held, counts = hold_back_wrist(chain, h, .99, torch.Generator().manual_seed(6))
    k = (~held.view_valid[:, :, 0]).sum(1)
    assert sum(counts) == int((k > 0).sum()) and all(counts)
    core = build(vocabulary, actor_history=False, reader_task=True, gripper_separate=True)
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(batch, -1) * .1
    with torch.no_grad():
        state = factual_state(core, held, h, images=True)
        seen = temporal_actor_state(core, state)
        proposal = core.act(state, held.text_ids, offsets).commands_position
    assert torch.isfinite(proposal).all()
    for i in range(batch):
        frame = h - int(k[i])
        assert bool(state.measurement_available[i, 0]) == (k[i] == 0) and bool(state.measurement_available[i, 1])
        assert torch.equal(state.memory.features[i, 0, -1], chain.features[i, frame, 0].to(state.memory.features))
        torch.testing.assert_close(state.memory.q_at_image[i, 0, -1], chain.q[i, frame])
        age = state.memory.current_readback_index[i] - state.memory.readback_indices[i, 0, -1]
        assert int(age) == 5 * int(k[i]) and bool(seen.memory.valid[i, 0, -1])
        assert torch.equal(state.memory.features[i, 1, -1], chain.features[i, h, 1].to(state.memory.features))
        assert bool((state.object_relative[i] == 0).all()) == (k[i] > 0)


def test_the_exam_withholds_the_wrist_pictures_as_training_does():
    # section 17.63: withhold_wrist(k) gives every row what hold_back_wrist gives a row it holds back by k blocks; 'all'
    # hides the wrist camera in frames 0..h; the external camera never changes
    from dataclasses import dataclass
    from scripts.train_v21_fcpg_actor import hold_back_wrist, withhold_wrist

    @dataclass
    class Chain:
        view_valid: torch.Tensor
        features: torch.Tensor

    n, h = 400, 3
    chain = Chain(torch.ones(n, 5, 2, dtype=torch.bool), torch.rand(n, h + 1, 2, 3, 4) + .1)
    trained, counts = hold_back_wrist(chain, h, .99, torch.Generator().manual_seed(1))
    k = (~trained.view_valid[:, :, 0]).sum(1)
    assert all(c > 50 for c in counts)
    for m in (1, 2, 3):
        rows = k == m
        exam = withhold_wrist(chain, h, m)
        assert torch.equal(exam.view_valid[rows], trained.view_valid[rows])
        assert torch.equal(exam.features[rows], trained.features[rows])
        assert torch.equal(exam.view_valid[:, :, 1], chain.view_valid[:, :, 1])
    blind = withhold_wrist(chain, h, 'all')
    assert not blind.view_valid[:, :h + 1, 0].any() and blind.view_valid[:, h + 1:, 0].all()
    assert (blind.features[:, :, 0] == 0).all() and torch.equal(blind.features[:, :, 1], chain.features[:, :, 1])
    assert torch.equal(blind.view_valid[..., 1], chain.view_valid[..., 1])



def test_reread_loss_compares_held_rows_only_and_ignores_scale():
    # section 17.95: zero for the same read (at any scale), only held-back rows count, no gradient reaches the fresh read
    from scripts.train_v21_fcpg_actor import reread_loss
    torch.manual_seed(0)
    fresh = torch.randn(4, 8, 192)
    held = torch.tensor([True, False, True, False])
    assert float(reread_loss(fresh.clone(), fresh, held)) == 0.
    assert float(reread_loss(fresh * 3., fresh, held)) < 1e-9
    stale = fresh.clone()
    stale[1] = torch.randn(8, 192)                      # a row that was not held back does not count
    assert float(reread_loss(stale, fresh, held)) < 1e-9
    stale[2] = torch.randn(8, 192)
    norm = lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:])
    expected = (norm(stale[2]) - norm(fresh[2])).square().mean() / 2      # rows 0 and 2 held, row 0 equal
    torch.testing.assert_close(reread_loss(stale, fresh, held), expected)
    assert float(reread_loss(stale, fresh, torch.zeros(4, dtype=torch.bool))) == 0.
    s, f = stale.clone().requires_grad_(), fresh.clone().requires_grad_()
    reread_loss(s, f, held).backward()
    assert f.grad is None and float(s.grad[[0, 1, 3]].abs().sum()) == 0. and float(s.grad[2].abs().sum()) > 0
    for bad in ((stale, fresh[:, :4], held), (stale, fresh, held[:3]), (stale, fresh, held.long())):
        with pytest.raises(ValueError, match='re-read'):
            reread_loss(*bad)


def test_reread_target_is_the_read_the_actor_gets_with_the_newest_pictures(vocabulary):
    # section 17.95: prepare() composes the rows again without the hold-back; that read (temporal_actor_state, last
    # observation only) is exactly the wrist read the Actor gets from the unheld chain, a held-back row's read differs
    # from it, the others equal it; update() writes core.act() out and gets the same proposal
    from scripts.train_v21_fcpg_actor import final_read_only, hold_back_wrist
    from fcpg.v21_temporal_contract import temporal_actor_state
    batch, h = 8, 3
    chain = synthetic_chain(vocabulary, batch=batch)
    core = build(vocabulary, actor_history=False, reader_task=True, gripper_separate=True, wrist_only=True)
    held, counts = hold_back_wrist(chain, h, .6, torch.Generator().manual_seed(6))
    mask = chain.view_valid[:, h, 0] & ~held.view_valid[:, h, 0]
    assert 0 < int(mask.sum()) < batch and int(mask.sum()) == sum(counts)
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(batch, -1) * .1
    with torch.no_grad():
        with final_read_only(core, h):
            fresh = factual_state(core, chain, h, images=True)
        fresh = temporal_actor_state(core, fresh).grounding[:, 0]
        actor_sees = core.actor_inputs(factual_state(core, chain, h, images=True))['grounding'][:, 0]
        state = factual_state(core, held, h, images=True)
        inputs = core.actor_inputs(state)
        stale = inputs['grounding'][:, 0]
        out = core.actor(inputs.pop('grounding'), inputs.pop('visual_error'), inputs.pop('q_position'), held.text_ids,
                         command_offsets=offsets, **inputs)
        torch.testing.assert_close(out.commands_position, core.act(state, held.text_ids, offsets).commands_position,
                                   rtol=0, atol=0)
    torch.testing.assert_close(fresh, actor_sees, rtol=0, atol=1e-6)
    differs = (stale - fresh).abs().flatten(1).amax(1) > 1e-4
    assert torch.equal(differs, mask)


def test_memory_address_temporal_reads_without_the_joints_at_each_exposure(vocabulary):
    # section 17.95: --memory-address temporal (V21Config.memory_mode) builds a model whose reads do not change with the
    # joints at an exposure; the re-read target then trains no address weight of those joints, while the default
    # address (readback) trains them (the jaw columns stay unread with the separate gripper)
    from scripts.train_v21_fcpg_actor import hold_back_wrist, reread_loss
    from fcpg.v21_temporal_contract import temporal_actor_state
    batch, h = 6, 2
    chain = synthetic_chain(vocabulary, batch=batch)
    frames = chain.q.shape[1]
    moved = replace(chain, capture_q=(chain.q[:, :, None].expand(-1, -1, 2, -1) + 7.).clone())
    held, _ = hold_back_wrist(chain, h, .99, torch.Generator().manual_seed(2))
    mask = chain.view_valid[:, h, 0] & ~held.view_valid[:, h, 0]
    assert mask.any() and moved.capture_q.shape == (batch, frames, 2, 6)
    for mode in ('readback', 'temporal'):
        core = build(vocabulary, actor_history=False, reader_task=True, gripper_separate=True, memory_mode=mode)
        assert core.reader.mode == mode
        with torch.no_grad():
            a = factual_state(core, chain, h, images=True).grounding
            b = factual_state(core, moved, h, images=True).grounding
        assert bool((a - b).abs().max() > 1e-5) == (mode == 'readback')
        with torch.no_grad():
            fresh = temporal_actor_state(core, factual_state(core, chain, h, images=True)).grounding[:, 0]
        state = factual_state(core, held, h, images=True, train_reader=True)
        reread_loss(core.actor_inputs(state)['grounding'][:, 0], fresh, mask).backward()
        grad = core.reader.address[0].weight.grad                  # [64, 29]: exposure joints 0-5, displacement 6-11
        arm = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        assert float(grad[:, [5, 11]].abs().sum()) == 0.
        assert (float(grad[:, arm].abs().sum()) > 0) == (mode == 'readback')
        assert float(grad[:, 24:].abs().sum()) > 0                   # age, view and patch position are read in both


def test_reread_switches_are_refused_where_they_mean_nothing(tmp_path):
    # section 17.95: the re-read target needs held-back rows and the Archive stage; the address switch builds a new model
    from scripts.train_v21_fcpg_actor import main, parser
    common = ['--source-config', str(tmp_path / 'c.json'), '--camera-layout', str(tmp_path / 'l.npz'),
              '--lewm', str(tmp_path / 'lewm.pt'), '--output', str(tmp_path / 'run')]
    args = parser().parse_args(common)
    assert args.reread_weight == 0. and args.memory_address == 'readback'
    hold = ['--wrist-hold-back', '.5']
    for bad, words in ((['--reread-weight', '1'], '--reread-weight'),
                       (hold + ['--reread-weight', '-1'], '--reread-weight'),
                       (hold + ['--reread-weight', 'nan'], '--reread-weight'),
                       (hold + ['--reread-weight', '1', '--fish-train-reader'], '--reread-weight'),
                       (hold + ['--reread-weight', '1', '--archive-checkpoint', str(tmp_path / 'a.pt')], '--reread-weight'),
                       (hold + ['--reread-weight', '1', '--evaluate-checkpoint', str(tmp_path / 'e.pt')], '--reread-weight'),
                       (['--memory-address', 'temporal', '--evaluate-checkpoint', str(tmp_path / 'e.pt')], '--memory-address')):
        with pytest.raises(ValueError, match=words):
            main(common + bad)
    # a temporal model's later steps pass its switch: past that check, to the next one (the missing files)
    for later in (['--continue-archive', str(tmp_path / 'c.pt')], ['--archive-checkpoint', str(tmp_path / 'a.pt')]):
        with pytest.raises(ValueError) as refused:
            main(common + ['--memory-address', 'temporal'] + later)
        assert '--memory-address' not in str(refused.value)
    with pytest.raises(SystemExit):
        parser().parse_args(common + ['--memory-address', 'other'])

def test_window_rows_default_and_a_window_smaller_than_a_batch_is_refused(tmp_path):
    # section 17.55: the locality window is a setting (the .102 A100 reads the 1.2M Archive from a hard disk); its
    # default keeps every earlier run's 2048
    from scripts.train_v21_fcpg_actor import main, parser
    common = ['--source-config', str(tmp_path / 'c.json'), '--camera-layout', str(tmp_path / 'l.npz'),
              '--lewm', str(tmp_path / 'lewm.pt'), '--output', str(tmp_path / 'run')]
    assert parser().parse_args(common).window_rows == 2048
    with pytest.raises(ValueError, match='--window-rows'):
        main(common + ['--batch-size', '256', '--window-rows', '128'])


def test_the_first_five_commands_do_not_depend_on_how_far_the_actor_plans(vocabulary):
    # the decoder is causal: the 5 commands a block executes are the same whether 5 or 40 are asked for, so a fish
    # stage supervising only 5 trains exactly what the service runs
    chain = synthetic_chain(vocabulary, batch=2)
    offsets = torch.arange(40, dtype=torch.float64)[None].expand(2, -1) * .1
    core = build(vocabulary, actor_history=False, reader_task=True, gripper_separate=True)
    torch.nn.init.normal_(core.actor.output.weight, std=.05)
    with torch.no_grad():
        state = factual_state(core, chain, 2, images=True)
        long = core.act(state, chain.text_ids, offsets).commands_position
        short = core.act(state, chain.text_ids, offsets[:, :5]).commands_position
    assert short.shape == (2, 5, 6)
    # float32 rounding differs with the number of queries (about 2e-5 units on a jaw near 20); 1e-4 units is far
    # below anything a servo resolves, and a real dependence on later steps would be orders of magnitude larger
    assert (short - long[:, :5]).abs().max() < 1e-4


def test_an_archive_update_cap_the_passes_cannot_reach_is_refused():
    # section 17.58: HL2 asked for 6,960 Archive updates with 2 passes over 257,527 rows and trained 4,024 silently
    from scripts.train_v21_fcpg_actor import planned_archive_updates
    assert planned_archive_updates(257527, 2048, 128, 2) == 4024
    with pytest.raises(ValueError, match='gives 4024 updates, fewer than --archive-updates 6960'):
        planned_archive_updates(257527, 2048, 128, 2, 6960)
    assert planned_archive_updates(257527, 2048, 128, 4, 6960) == 6960
    assert planned_archive_updates(112665, 2048, 128, 5, 4024) == 4024        # HL3: 881 batches per pass
    assert planned_archive_updates(300, 128, 64, 1) == 5                      # windows of 128, 128, 44 rows


def test_the_fish_stage_archive_pool_keeps_rows_about_equally_likely_and_reads_few_blocks():
    # section 17.59: --fish-archive-pool draws the fish stage's Archive half from a rolling pool of store blocks
    from scripts.train_v21_fcpg_actor import ArchivePool
    sizes = np.array([64] * 50 + [8] * 50)                # 100 blocks, big and small
    keys = np.repeat(np.arange(100) * 7 + 3, sizes)
    rows = np.arange(len(keys)) * 2 + 11
    rng = np.random.default_rng(0)
    pool = ArchivePool(rows, keys, 32, 2, rng)
    counts = np.zeros(len(keys))
    for _ in range(4000):
        drawn = pool.draw(32)
        assert np.isin(drawn, rows).all()
        assert len(pool.pool) == len(set(pool.pool)) == 32
        counts[np.searchsorted(rows, drawn)] += 1
    assert pool.blocks_entered == 32 + 2 * 4000
    big, small = counts[:50 * 64].mean(), counts[50 * 64:].mean()
    assert abs(big / small - 1) < .1                      # a row of a big block about as likely as one of a small block
    with pytest.raises(ValueError, match='refresh'):
        ArchivePool(rows, keys, 10, 0, rng)


def test_held_development_rows_serve_the_window_values_from_a_copy():
    # section 17.65: the development rows are read once into host memory and copied to the GPU for each evaluation
    from fcpg.v21_encoder import DIM, PATCHES
    from scripts.train_v21_fcpg_actor import WindowFeatures
    rng = np.random.default_rng(7)
    endpoints = np.arange(40, dtype=np.int64) * 3                     # sorted picture identities
    tokens = rng.standard_normal((40, 2, PATCHES, DIM)).astype(np.float16)
    valid = rng.random((40, 2)) > .2
    tokens[~valid] = 0
    cache = SimpleNamespace(endpoint_index=endpoints, view_valid=valid, tokens=tokens,
                            rows_for=lambda ids: np.searchsorted(endpoints, ids))
    chains = np.stack([endpoints[i:i + 5] for i in range(30)])        # 30 rows of five consecutive pictures
    source = SimpleNamespace(archive=SimpleNamespace(row_endpoint_index=chains), archive_dino=[cache])
    holder = WindowFeatures(source, torch.device('cpu'))
    rows = np.array([3, 9, 10, 22])
    holder.load(rows)
    holder.close()
    held = holder.copy_to('cpu')
    for frames in range(1, 5):
        features, checked = held.features(rows[::-1], frames)
        index = np.searchsorted(endpoints, chains[rows[::-1], :frames])
        assert torch.equal(features, torch.from_numpy(tokens[index]).float())
        assert np.array_equal(checked, valid[index])
    with pytest.raises(ValueError, match='outside the loaded window'):
        held.features(np.array([0]), 1)


def test_flips_and_masks_in_float16_give_the_float32_values_bit_for_bit():
    # section 17.76: prepare() keeps a window's pictures float16 through the flips and masks and converts them once
    from fcpg.v21_encoder import DIM, PATCHES
    from fcpg.v21_mirror import flipped_features
    from scripts.train_v21_fcpg_actor import WindowFeatures
    rng = np.random.default_rng(8)
    endpoints = np.arange(40, dtype=np.int64) * 3
    tokens = rng.standard_normal((40, 2, PATCHES, DIM)).astype(np.float16)
    tokens[rng.random(tokens.shape) < .01] *= -0.                   # signed zeros and negative values survive too
    valid = np.ones((40, 2), bool)
    cache = SimpleNamespace(endpoint_index=endpoints, view_valid=valid, tokens=tokens,
                            rows_for=lambda ids: np.searchsorted(endpoints, ids))
    chains = np.stack([endpoints[i:i + 5] for i in range(30)])
    source = SimpleNamespace(archive=SimpleNamespace(row_endpoint_index=chains), archive_dino=[cache])
    window = WindowFeatures(source, torch.device('cpu'))
    rows = np.array([3, 9, 10, 22, 4])
    window.load(rows)
    window.close()
    flip = torch.tensor([True, False, True, True, False])
    for frames in range(1, 5):
        stored, checked = window.features(rows, frames, as_stored=True)
        converted, _ = window.features(rows, frames)
        assert stored.dtype == torch.float16 and torch.equal(stored.float(), converted)
        keep = torch.rand(stored.shape[:-1], generator=torch.Generator().manual_seed(frames)) >= .25
        hide = torch.ones(stored.shape[:3], dtype=torch.bool)
        hide[1, :, 0] = False                                         # a whole camera of one row, as the view drops
        def steps(features):
            features = flipped_features(features, flip)
            features = features * hide[..., None, None].to(features.dtype)
            return features * keep[..., None].to(features.dtype)
        before, after = steps(converted), steps(stored).float()
        assert torch.equal(before.view(torch.int32), after.view(torch.int32))


class _SwapCameras:
    """A camera layout for the window tests: recordings whose id is a multiple of 6 have their cameras swapped."""
    def remap(self, value, ids):
        swap = torch.as_tensor(np.asarray(ids) % 6 == 0)
        if isinstance(value, np.ndarray):
            return np.where(swap.numpy()[:, None], value[:, ::-1], value)
        swap = swap.to(value.device).reshape(-1, *([1] * (value.ndim - 1)))
        return torch.where(swap, value.flip(1), value)


def _int8_window(n=600, seed=12):
    """n pictures in two interleaved int8 store parts (values of absent views are zero)."""
    from fcpg.v21_encoder import DIM, PATCHES
    rng = np.random.default_rng(seed)
    unique = np.arange(n, dtype=np.int64) * 2
    valid = rng.random((n, 2)) > .3
    values = rng.integers(-128, 128, (n, 2, PATCHES, DIM)).astype(np.int8)
    values[~valid] = 0
    scales = rng.uniform(.001, .2, (n, 2, PATCHES)).astype(np.float16)
    first, second = np.arange(0, n, 2), np.arange(1, n, 2)
    return unique, [(first, values[first], scales[first]), (second, values[second], scales[second])], valid, values, scales


def test_a_window_is_placed_a_chunk_at_a_time_with_the_same_values_and_checks():
    # section 17.78: int8 rows multiplied out, checked and laid out 256 rows at a time (on a GPU, on a side stream)
    from scripts.train_v21_fcpg_actor import WindowFeatures, dequantize
    unique, parts, valid, values, scales = _int8_window()               # three chunks, the last one short
    second = parts[1][0]
    window = WindowFeatures(SimpleNamespace(archive_camera_roles=_SwapCameras()), torch.device('cpu'))
    got_unique, tokens, got_valid, broken = window._place(unique, parts, valid)
    expected = _SwapCameras().remap(dequantize(torch.from_numpy(values), torch.from_numpy(scales)), unique)
    assert not bool(broken) and np.array_equal(got_unique, unique)
    assert torch.equal(tokens.view(torch.int16), expected.view(torch.int16))
    assert np.array_equal(got_valid, _SwapCameras().remap(valid, unique))
    for spoil in ('nonfinite', 'absent'):
        spoilt = [(where, v.copy(), s.copy()) for where, v, s in parts]
        row = int(np.flatnonzero((~valid[second]).any(1))[3])         # a row of the second part with an absent view
        view = int(np.flatnonzero(~valid[second][row])[0])
        if spoil == 'nonfinite':
            spoilt[1][2][row, 1 - view, 0] = np.inf
            spoilt[1][1][row, 1 - view, 0, 0] = 1
        else:
            spoilt[1][1][row, view, 5, 7] = 3
        assert bool(window._place(unique, spoilt, valid)[3])
    window.close()


def test_the_pool_schedule_with_one_window_is_the_earlier_schedule_draw_for_draw():
    # section 17.82: --pool-windows 1 makes exactly the draws and batches the Archive loop made before the pool
    from scripts.train_v21_fcpg_actor import pool_batches
    windows = [np.arange(i, min(i + 50, 230)) for i in range(0, 230, 50)]            # the last window is short

    def earlier(rng):
        for epoch in range(3):
            order = rng.permutation(len(windows))
            for position, window_id in enumerate(order):
                window = rng.permutation(windows[window_id])
                for begin in range(0, len(window), 16):
                    rows = window[begin:begin + 16]
                    yield epoch, begin == 0, rows, int(rng.integers(4)), rng.random(len(rows)) < .5

    old, new = earlier(np.random.default_rng(5)), pool_batches(windows, 3, 1, 16, 50, 2, np.random.default_rng(5), True)
    count = 0
    for (epoch, first, rows, h, mirror), (pass_, entering, leaving, following, rows2, h2, mirror2) in zip(old, new):
        assert (epoch, h) == (pass_, h2) and np.array_equal(rows, rows2) and np.array_equal(mirror, mirror2)
        assert first == bool(entering) and len(entering) <= 1
        count += 1
    assert next(old, None) is None and next(new, None) is None and count == 3 * sum(-(-len(w) // 16) for w in windows)


def test_a_pool_of_windows_uses_every_row_once_per_pass_and_lets_windows_leave_one_at_a_time():
    # section 17.82 (James 「用过就淘汰」): each batch takes from every window in the pool; a window leaves with its last row
    from scripts.train_v21_fcpg_actor import planned_archive_updates, pool_batches, pool_counts
    assert pool_counts([7], 16, 4) == [7] and pool_counts([100, 100, 100], 30, 4) == [16, 8, 6]
    assert pool_counts([1, 100, 100], 30, 4) == [1, 17, 12] and pool_counts([5, 3], 30, 4) == [5, 3]
    windows = [np.arange(i, min(i + 64, 1000)) for i in range(0, 1000, 64)]
    seen, resident, most, batches = np.zeros(1000, np.int64), set(), 0, 0
    for _, entering, leaving, following, rows, h, mirror in pool_batches(windows, 2, 4, 16, 64, 3, np.random.default_rng(9), True):
        resident |= {key for key, _ in entering}
        most = max(most, len(resident))
        assert {key for key, _ in following}.isdisjoint(resident) and len(rows) == len(mirror) <= 16
        np.add.at(seen, rows, 1)
        resident -= set(leaving)
        batches += 1
    assert (seen == 2).all() and most == 4 and not resident
    assert batches == planned_archive_updates(1000, 64, 16, 2, pool=4) == 125


def test_rows_of_a_pool_come_from_the_windows_that_hold_them():
    # section 17.82: features() takes every row of a batch from its window in the pool, in the batch's order
    from fcpg.v21_encoder import DIM, PATCHES
    from scripts.train_v21_fcpg_actor import WindowFeatures
    rng = np.random.default_rng(11)
    endpoints = np.arange(60, dtype=np.int64) * 3
    tokens = rng.standard_normal((60, 2, PATCHES, DIM)).astype(np.float16)
    valid = rng.random((60, 2)) > .2
    tokens[~valid] = 0
    cache = SimpleNamespace(endpoint_index=endpoints, view_valid=valid, tokens=tokens,
                            rows_for=lambda ids: np.searchsorted(endpoints, ids))
    chains = np.stack([endpoints[i:i + 5] for i in range(50)])
    source = SimpleNamespace(archive=SimpleNamespace(row_endpoint_index=chains), archive_dino=[cache])
    pool = WindowFeatures(source, torch.device('cpu'))
    first, second = np.arange(0, 20), np.arange(25, 45)
    pool.admit('a', first)
    pool.admit('b', second)
    rows = np.array([30, 2, 44, 19, 25, 0])
    for frames in range(1, 5):
        features, checked = pool.features(rows, frames)
        index = np.searchsorted(endpoints, chains[rows, :frames])
        assert torch.equal(features, torch.from_numpy(tokens[index]).float())
        assert np.array_equal(checked, valid[index])
    pool.release('a')
    with pytest.raises(ValueError, match='outside the loaded window'):
        pool.features(np.array([2]), 1)
    pool.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='reading processes place their windows on a GPU')
def test_windows_read_by_processes_are_placed_ahead_and_pooled_as_the_cpu_placement():
    # sections 17.79 / 17.82: reading processes hand int8 windows over in shared memory; the next one is placed on the
    # GPU on a side stream before it is admitted, and nothing is read twice
    from scripts.train_v21_fcpg_actor import WindowFeatures
    unique, parts, valid, _, _ = _int8_window(n=520, seed=13)
    source = SimpleNamespace(archive_camera_roles=_SwapCameras(), archive_dino=[])
    expected = WindowFeatures(source, torch.device('cpu'))._place(unique, parts, valid)[1]
    window = WindowFeatures(source, torch.device('cuda:0'), readers=2)

    def read(rows, pinned=False, shared=False):
        assert shared                                                # in a reading process, into shared memory
        return unique, [(where, torch.from_numpy(v).share_memory_(), torch.from_numpy(s).share_memory_())
                        for where, v, s in parts], valid
    window._read = read
    rows, later = np.arange(5), np.arange(5, 9)
    window.prefetch([('a', rows), ('b', later)])
    window.prefetch([('a', rows), ('b', later)])                     # being read or placed: nothing is read twice
    assert window._placed[0] == 'a' and [key for key, _, _ in window._ahead] == ['b']
    for key, current in (('a', rows), ('b', later)):                  # each one placed on the GPU ahead of its admission
        window.admit(key, current)
        assert window.prefetched and window._placed is None
        got_unique, tokens, got_valid = window.resident[key]
        assert np.array_equal(got_unique, unique) and np.array_equal(got_valid, _SwapCameras().remap(valid, unique))
        assert torch.equal(tokens.cpu().view(torch.int16), expected.view(torch.int16))
        window.prefetch([('b', later)] if key == 'a' else [])
    window.release('a')
    assert list(window.resident) == ['b']
    window.close()


def test_the_pool_names_its_entering_blocks_and_warming_reads_their_pictures():
    # section 17.65: the blocks entering the fish pool are decoded on a background thread before their rows are drawn
    from scripts.train_v21_fcpg_actor import ArchivePool, warm_archive_blocks
    sizes = np.array([5, 7, 3, 6, 4, 8])
    keys = np.repeat(np.arange(6) * 10, sizes)
    rows = np.arange(len(keys)) * 3
    plain, named = ArchivePool(rows, keys, 3, 1, np.random.default_rng(4)), ArchivePool(rows, keys, 3, 1, np.random.default_rng(4))
    for _ in range(20):
        assert np.array_equal(plain.draw(4), named.draw(4))           # naming the blocks changes no draw
        assert named.entered == named.pool[-1:]                       # the block appended last (it may be the one that left)
        expected = np.concatenate([rows[keys == 10 * b] for b in named.entered])
        assert np.array_equal(np.sort(named.entered_rows()), np.sort(expected))

    asked = []
    endpoints = np.arange(100, dtype=np.int64) * 2
    class Tokens:
        source_row = None
        def __getitem__(self, index):
            asked.append(np.asarray(index))
    cache = SimpleNamespace(endpoint_index=endpoints, tokens=Tokens(), rows_for=lambda ids: np.searchsorted(endpoints, ids))
    plain_cache = SimpleNamespace(endpoint_index=endpoints, tokens=np.zeros(3), rows_for=None)   # a memory map: nothing kept
    chains = np.stack([endpoints[i:i + 5] for i in range(90)])
    source = SimpleNamespace(archive=SimpleNamespace(row_endpoint_index=chains), archive_dino=[plain_cache, cache])
    warm_archive_blocks(source, np.array([4, 2, 4]))
    assert len(asked) == 1 and np.array_equal(asked[0], np.arange(2, 9))  # every picture of rows 2 and 4, once
    warm_archive_blocks(source, np.zeros(0, np.int64))
    assert len(asked) == 1

    # a feature bank that multiplies an int8 store's rows on its GPU reads int8 blocks: those are the ones kept
    quantized = []
    class Int8Tokens(Tokens):
        int8 = True
        def quantized(self, index):
            quantized.append(np.asarray(index))
    int8_cache = SimpleNamespace(endpoint_index=endpoints, tokens=Int8Tokens(), rows_for=lambda ids: np.searchsorted(endpoints, ids))
    int8_source = SimpleNamespace(archive=source.archive, archive_dino=[int8_cache])
    warm_archive_blocks(int8_source, np.array([4, 2, 4]), SimpleNamespace(device='cuda:0'))
    assert len(quantized) == 1 and np.array_equal(quantized[0], np.arange(2, 9)) and len(asked) == 1
    warm_archive_blocks(int8_source, np.array([4, 2, 4]), SimpleNamespace(device='cpu'))   # a CPU bank: float16 blocks
    assert len(quantized) == 1 and len(asked) == 2


def test_dequantizing_on_a_device_is_the_store_conversion_bit_for_bit():
    # section 17.65: the window's int8 rows are multiplied on the GPU; the same arithmetic as the store's numpy decode
    from scripts.train_v21_fcpg_actor import dequantize
    rng = np.random.default_rng(11)
    values = rng.integers(-128, 128, (6, 2, 7, 33)).astype(np.int8)
    values[0] = 127
    values[1] = -128
    scales = np.concatenate([rng.uniform(1e-7, 6e-5, (2, 2, 7)),            # products below float16's normal range
                             rng.uniform(.001, .2, (2, 2, 7)), rng.uniform(100, 500, (2, 2, 7))]).astype(np.float16)
    expected = (values.astype(np.float32) * scales.astype(np.float32)[..., None]).astype(np.float16)
    got = dequantize(torch.from_numpy(values), torch.from_numpy(scales)).numpy()
    assert got.dtype == np.float16 and np.array_equal(got.view(np.uint16), expected.view(np.uint16))


def test_cue_only_rows_blank_both_pictures_where_the_fish_cue_is_seen():
    # 11/03: one draw per row; below the share, a row whose current wrist cue is seen gets blank pictures (features
    # zero) in both cameras for the whole chain, the exposures still delivered, so only the cue, the joints and the
    # task say where the object is; rows without a cue keep their pictures
    from dataclasses import dataclass
    from scripts.train_v21_fcpg_actor import cue_only_views

    @dataclass
    class Chain:
        view_valid: torch.Tensor
        features: torch.Tensor
        object_relative: torch.Tensor

    n, h = 4000, 2
    cue = torch.zeros(n, 5, 8)
    cue[:3000, h, 0] = 1.                           # the fish seen at context h in the first 3000 rows
    chain = Chain(torch.ones(n, 5, 2, dtype=torch.bool), torch.ones(n, 5, 2, 3, 4), cue)
    out, hidden = cue_only_views(chain, h, .3, torch.Generator().manual_seed(1))
    blind = (out.features == 0).flatten(1).all(1)
    assert int(blind.sum()) == hidden and not blind[3000:].any()
    assert .27 < float(blind[:3000].float().mean()) < .33
    assert (out.features[~blind] == 1).all()
    assert torch.equal(out.view_valid, chain.view_valid)     # still delivered
    assert torch.equal(out.object_relative, cue)
    no_wrist = chain.view_valid.clone()
    no_wrist[:, h, 0] = False                       # the wrist picture did not arrive: no cue for the Actor, no blank
    kept, none = cue_only_views(replace(chain, view_valid=no_wrist), h, .99, torch.Generator().manual_seed(1))
    assert none == 0 and (kept.features == 1).all()
    same, none = cue_only_views(chain, h, 0., torch.Generator().manual_seed(1))
    assert same is chain and none == 0
    with pytest.raises(ValueError, match='blanks the pictures'):
        cue_only_views(replace(chain, features=None), h, .3, torch.Generator().manual_seed(1))


def test_the_actor_keeps_the_cue_of_a_cue_only_row(vocabulary):
    # the Actor gets the wrist cue only with a current wrist exposure (temporal_actor_state): 9afe618 hid the cameras by
    # making their exposures invalid, and those rows lost the cue as well; blank pictures keep it
    from fcpg.v21_temporal_contract import temporal_actor_state
    from scripts.train_v21_fcpg_actor import cue_only_views

    chain = synthetic_chain(vocabulary, batch=2)
    cue = torch.zeros(2, 5, 8)
    cue[..., 0], cue[..., 2] = 1., .2
    chain = replace(chain, object_relative=cue)
    core = build(vocabulary, actor_history=False, reader_task=True, gripper_separate=True)
    blank, hidden = cue_only_views(chain, 2, .999, torch.Generator().manual_seed(0))
    invalid = replace(chain, view_valid=torch.zeros_like(chain.view_valid), features=chain.features * 0)
    with torch.no_grad():
        kept = temporal_actor_state(core, factual_state(core, blank, 2, images=True))
        lost = temporal_actor_state(core, factual_state(core, invalid, 2, images=True))
    assert hidden == 2 and (blank.features == 0).all()
    assert torch.equal(kept.object_relative, cue[:, 2]) and not lost.object_relative.any()
