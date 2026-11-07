"""17.166 (11/01; James 「直接30張」「不會抄人家的方案馬」): the full-patch LeWM with a memory of the last N pictures per
camera (fcpg/v21_full_patch_lewm.py memory_pictures=N, memory_tokens=K; fcpg/v21_picture_memory.py). A picture's K
pooled tokens depend on that picture only (its patches and camera: made once, kept), the forecast reads them moved by
the time since and the joints' travel since, plus each camera's newest picture in all its patches; empty slots change
nothing; a row without pictures forecasts as with no pictures; the planner's shared memory is read once and gives the
same forecasts as every row's own copy; the new parts train; every loader rebuilds it. a-control's other settings
(control setting, balanced inputs, symlog). nn.MultiheadAttention's inference fast path is off, as in the trainer."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from fcpg.v21_encoder import DIM as FEATURE_DIM, PATCHES
from fcpg.v21_full_patch_lewm import V21FullPatchLeWM, architecture_from_state
from fcpg.v21_native_horizon import core_from_parent
from fcpg.v21_picture_memory import PictureMemory, from_visual_memory, select_pictures, view_loss_drops
from v21_lewm_helpers import setup

N = 6
RECIPE = dict(controller_input=True, balanced_inputs=True, command_symlog=True, memory_pictures=N, memory_tokens=4)


@pytest.fixture(autouse=True)
def no_attention_fast_path():
    before = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(before)


def memory_for(state, seed=3, taken=(N, 2)):
    """A picture memory beside the state's time: camera 0 took taken[0] pictures, camera 1 taken[1], 0.5 s apart."""
    g = torch.Generator().manual_seed(seed)
    b = len(state.q)
    features = torch.randn(b, 2, N, PATCHES, FEATURE_DIM, generator=g)
    valid = torch.zeros(b, 2, N, dtype=torch.bool)
    for camera, count in enumerate(taken):
        if count:
            valid[:, camera, N - count:] = True
    features = torch.where(valid[..., None, None], features, 0.)
    times = state.time.double()[:, None, None] - .5 * torch.arange(N - 1, -1, -1, dtype=torch.float64)[None, None] - .1
    q = state.q[:, None, None].expand(b, 2, N, 6) + torch.randn(b, 2, N, 6, generator=g)
    return PictureMemory(features, torch.where(valid[..., None], q, 0.), torch.where(valid, times, 0.), valid,
                         np.full((b, 2, N), -1), np.ones(b, np.int64))


def call(lewm, state, targets, memory, **kwargs):
    b = len(state.q)
    controller = torch.tensor([[0., 0.], [1., 1.]])[:b]
    return lewm(state.q, targets.trace, state.history, targets.readback_offsets, patches=memory.features,
                patch_valid=memory.valid, memory=memory, now_time=state.time, controller=controller, **kwargs)


def forecast(lewm, state, targets, memory):
    with torch.no_grad():
        return call(lewm, state, targets, memory).predicted_q_path


def as_trained(lewm, seed=5):
    """The zero-started FiLM of the memory drawn as well (a trained one is not the identity)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in (lewm.memory_film, lewm.film):
            for weight in module.parameters():
                weight.copy_(torch.randn(weight.shape, generator=g) * .05)
    return lewm


def test_a_memory_lewm_has_its_parts_and_refuses_what_it_cannot_read():
    core, _, _, _, _ = setup(**RECIPE)
    lewm = core.lewm
    found = lewm.architecture()
    assert found['memory_pictures'] == N and found['memory_tokens'] == 4
    assert all(layer.direct for layer in lewm.layers)
    assert lewm.memory_queries.shape == (4, lewm.width)
    for wrong in (dict(memory_pictures=True), dict(memory_pictures=65), dict(memory_pictures=3, memory_tokens=0),
                  dict(memory_pictures=2.)):
        with pytest.raises(ValueError, match='whole number'):
            V21FullPatchLeWM(horizon=40, **wrong)
    for other in (dict(label_head=True), dict(patch_encoder_layers=1)):
        with pytest.raises(ValueError, match='has no label head or patch encoder'):
            V21FullPatchLeWM(horizon=40, memory_pictures=3, **other)


def test_the_bank_is_the_newest_patches_and_every_pictures_pooled_tokens():
    core, state, _, targets, _ = setup(**RECIPE)
    lewm = as_trained(core.lewm)
    memory = memory_for(state)
    bank, pad = lewm._memory_bank(memory.features, memory.valid, state.q, memory, state.time)
    assert bank.shape == (len(state.q), 2 * PATCHES + 2 * N * 4, lewm.width)
    readable = ~pad
    assert readable[:, :PATCHES].all() and readable[:, PATCHES:2 * PATCHES].all()       # both cameras' newest
    tokens = readable[:, 2 * PATCHES:].reshape(len(state.q), 2, N, 4)
    assert tokens[:, 0].all() and tokens[:, 1, -2:].all() and not tokens[:, 1, :-2].any()
    assert torch.isfinite(forecast(lewm, state, targets, memory)).all()
    gap = PictureMemory(memory.features, memory.q_at_image, memory.image_times, memory.valid.clone(), memory.ids,
                        memory.strides)
    gap.valid[:, 0, -1] = False                                       # a hole at the newest end: not right-aligned
    with pytest.raises(ValueError, match='right-aligned'):
        forecast(lewm, state, targets, gap)


def test_a_pictures_tokens_depend_on_that_picture_only():
    core, state, _, _, _ = setup(**RECIPE)
    lewm = core.lewm.eval()
    g = torch.Generator().manual_seed(9)
    tokens = torch.randn(5, PATCHES, lewm.width, generator=g)
    with torch.no_grad():
        alone = lewm.memory_pool(lewm.memory_queries, tokens[2:3])
        together = lewm.memory_pool(lewm.memory_queries, tokens)
        changed = tokens.clone()
        changed[0] += torch.randn(changed[0].shape, generator=g)
        other = lewm.memory_pool(lewm.memory_queries, changed)
    torch.testing.assert_close(together[2:3], alone, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(other[1:], together[1:], atol=0, rtol=0)
    assert not torch.allclose(other[0], together[0], atol=1e-5)


def test_empty_slots_change_nothing_and_time_is_read():
    core, state, _, targets, _ = setup(**RECIPE)
    lewm = as_trained(core.lewm).eval()
    memory = memory_for(state)
    base = forecast(lewm, state, targets, memory)
    noise = memory.features.clone()
    noise[:, 1, :-2] = torch.randn(noise[:, 1, :-2].shape, generator=torch.Generator().manual_seed(1))
    filled = PictureMemory(noise, memory.q_at_image + (~memory.valid[..., None]) * 7., memory.image_times,
                           memory.valid, memory.ids, memory.strides)
    torch.testing.assert_close(forecast(lewm, state, targets, filled), base, atol=0, rtol=0)
    older = PictureMemory(memory.features, memory.q_at_image, memory.image_times - 3., memory.valid, memory.ids,
                          memory.strides)
    assert not torch.allclose(forecast(lewm, state, targets, older), base, atol=1e-6)


def test_a_row_without_pictures_forecasts_as_with_none():
    core, state, _, targets, _ = setup(**RECIPE)
    lewm = as_trained(core.lewm).eval()
    memory = memory_for(state, taken=(0, 0))
    controller = torch.tensor([[0., 0.], [1., 1.]])[:len(state.q)]
    with torch.no_grad():
        without = lewm(state.q, targets.trace, state.history, targets.readback_offsets, controller=controller)
    torch.testing.assert_close(forecast(lewm, state, targets, memory), without.predicted_q_path, atol=1e-6, rtol=1e-6)


def test_the_planners_shared_memory_is_read_once_with_the_same_forecasts():
    core, state, _, targets, _ = setup(**RECIPE)
    lewm = as_trained(core.lewm).eval()
    one = memory_for(state)
    n = 5
    row = lambda x: x[:1].expand(n, *x.shape[1:])
    shared = PictureMemory(row(one.features), row(one.q_at_image), row(one.image_times), row(one.valid),
                           np.repeat(one.ids[:1], n, 0), np.ones(n, np.int64))
    copied = PictureMemory(*(x.contiguous() for x in (shared.features, shared.q_at_image, shared.image_times,
                                                      shared.valid)), shared.ids, shared.strides)
    q, history = row(state.q), type(state.history)(**{k: row(v) for k, v in vars(state.history).items()})
    trace = type(targets.trace)(row(targets.trace.positions) + torch.randn(n, *targets.trace.positions.shape[1:]) * .01,
                                row(targets.trace.offsets), row(targets.trace.valid), targets.trace.source,
                                targets.trace.complete)
    seen = []
    original = lewm._memory_bank
    lewm._memory_bank = lambda *a: (seen.append(len(a[0])), original(*a))[1]
    try:
        with torch.no_grad():
            kwargs = dict(controller=torch.zeros(n, 2), now_time=row(state.time))
            a = lewm(q, trace, history, row(targets.readback_offsets), patches=shared.features, patch_valid=shared.valid,
                     memory=shared, **kwargs).predicted_q_path
            b = lewm(q.contiguous(), trace, history, row(targets.readback_offsets), patches=copied.features,
                     patch_valid=copied.valid, memory=copied, **{**kwargs, 'now_time': kwargs['now_time'].contiguous()}
                     ).predicted_q_path
    finally:
        del lewm._memory_bank
    assert seen == [1, n]                                             # once for the shared memory, per row for copies
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_the_memory_parts_train():
    core, state, _, targets, _ = setup(**RECIPE)
    lewm = as_trained(core.lewm).train()
    call(lewm, state, targets, memory_for(state)).predicted_q_path.square().sum().backward()
    for name in ('memory_queries', 'memory_kind'):
        assert getattr(lewm, name).grad.abs().sum() > 0
    for module in (lewm.memory_pool, lewm.memory_film, lewm.feature_in):
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_every_loader_rebuilds_a_memory_lewm():
    core, state, _, targets, _ = setup(**RECIPE)
    as_trained(core.lewm)
    whole = core.state_dict()
    assert architecture_from_state(whole, 'lewm.') == dict(width=384, layers=4, controller_input=True,
                                                           balanced_inputs=True, command_symlog=True,
                                                           memory_pictures=N, memory_tokens=4)
    rebuilt = core_from_parent(dict(config=asdict(core.config), model=whole)).eval()
    memory = memory_for(state)
    torch.testing.assert_close(forecast(rebuilt.lewm, state, targets, memory),
                               forecast(core.lewm.eval(), state, targets, memory), atol=0, rtol=0)
    with pytest.raises(RuntimeError):
        V21FullPatchLeWM(horizon=40, controller_input=True, balanced_inputs=True,
                         command_symlog=True).load_state_dict(core.lewm.state_dict(), strict=True)


def test_selection_counts_new_valid_exposures_newest_first_and_drops_from_the_newest():
    valid = np.ones((1, 6, 2), bool)
    valid[0, 1, 0] = False                                            # camera 0 missed the second candidate
    times = np.array([[[5., 5.], [4.5, 4.5], [4., 4.5], [3.5, 3.5], [3., 3.], [2.5, 2.5]]])   # camera 1 repeats once
    out = select_pictures(valid, times, 3, np.zeros((1, 2), np.int64))
    assert out[0, 0].tolist() == [3, 2, 0] and out[0, 1].tolist() == [3, 1, 0]
    out = select_pictures(valid, times, 3, np.array([[1, 2]]))
    assert out[0, 0].tolist() == [4, 3, 2] and out[0, 1].tolist() == [5, 4, 3]
    out = select_pictures(valid, times, 8, np.zeros((1, 2), np.int64))
    assert (out[0, 0, :3] == -1).all() and out[0, 0, 3:].tolist() == [5, 4, 3, 2, 0]


def test_three_pictures_become_the_newest_slots_and_drops_stay_in_range():
    core, state, _, _, _ = setup(memory_pictures=N)
    memory = from_visual_memory(state.memory, N)
    assert memory.features.shape[2] == N and not memory.valid[:, :, :N - 3].any()
    assert torch.equal(memory.valid[:, :, N - 3:], state.memory.valid)
    assert torch.equal(memory.features[:, :, N - 3:], state.memory.features)
    rng = np.random.default_rng(0)
    strides = rng.integers(1, 4, 1000)
    drop = view_loss_drops(rng, 1000, 30, strides, .5, 5.)
    lost = drop.max(1)
    assert ((drop > 0).sum(1) <= 1).all() and .4 < (lost > 0).mean() < .6
    assert (lost <= np.maximum(1, (5. / (.5 * strides)).astype(int))).all()


def robot_pack(sessions=2, blocks=6, seed=11, reset=None):
    """A pack shaped like robot-pack/*.pt: each session's camera c took a picture every 1.2 s; block k's snapshot held
    the last three of each camera (fewer at the start), right-aligned as VisualMemory keeps them, and the service's
    completed-readback counter (5 per completed block). reset: from that block on a new actual state (the next typed
    task, 40 s later): an empty memory again, the counter from 0."""
    from fcpg.v21_picture_memory import robot_memory_index  # noqa: F401  (the module under test)
    g = torch.Generator().manual_seed(seed)
    feats, qs, times, valid, session, counter, now = [], [], [], [], [], [], []
    for s in range(sessions):
        stream = torch.randn(blocks + 2, 2, PATCHES, FEATURE_DIM, generator=g).half()
        q = torch.randn(blocks + 2, 2, 6, generator=g)
        for k in range(blocks):
            start = reset if reset is not None and k >= reset else 0
            f = torch.zeros(2, 3, PATCHES, FEATURE_DIM, dtype=torch.float16)
            qq, tt, vv = torch.zeros(2, 3, 6), torch.zeros(2, 3, dtype=torch.float64), torch.zeros(2, 3, dtype=torch.bool)
            for c in range(2):
                held = [j for j in range(start, k + 1 - c) if j > k - c - 3]       # camera 1 starts one picture later
                for slot, j in zip(range(3 - len(held), 3), held):
                    f[c, slot], qq[c, slot], vv[c, slot] = stream[j, c], q[j, c], True
                    tt[c, slot] = 100. * s + 1.2 * j + (40. if start else 0.)
            feats.append(f), qs.append(qq), times.append(tt), valid.append(vv), session.append(s)
            counter.append(5 * (k - start))
            now.append(100. * s + 1.2 * k + (40. if start else 0.) + .1)
    memory = dict(features=torch.stack(feats), q_at_image=torch.stack(qs), image_times=torch.stack(times),
                  valid=torch.stack(valid), current_readback_index=torch.tensor(counter))
    return dict(split='exam', data=dict(memory=memory, session=torch.tensor(session),
                                        now=torch.tensor(now, dtype=torch.float64)))


def test_the_robot_streams_give_each_block_its_own_pictures_then_the_ones_before():
    from fcpg.v21_picture_memory import RobotPictureMemory, robot_memory_index
    pack = robot_pack()
    index = robot_memory_index(pack, 5)
    assert index['blocks'] == 12 and len(index['times']) == 2 * (6 + 5)
    stream = RobotPictureMemory(index, 'cpu')
    memory, data = pack['data']['memory'], pack['data']
    ids = np.arange(12)
    now = data['memory']['image_times'].amax((1, 2)) + .1
    three = stream.memory(ids, now, 3)
    assert torch.equal(three.valid, memory['valid']) and torch.equal(three.image_times, memory['image_times'])
    assert torch.equal(three.features[three.valid], memory['features'][memory['valid']])
    five = stream.memory(ids, now, 5)
    assert five.valid[5].all()                            # block 5: five pictures of each camera before it
    assert five.valid[3, 0, 1:].all() and not five.valid[3, 0, 0]               # block 3: four of the wrist's
    assert five.valid[3, 1, 2:].all() and not five.valid[3, 1, :2].any()        # and three of the other
    assert (torch.diff(five.image_times[5, 0]) > 0).all()
    lost = np.zeros((12, 2))
    lost[:, 0] = 1.                                       # the wrist's pictures of the last second are not taken
    after = stream.memory(ids, now, 3, lost)
    torch.testing.assert_close(after.image_times[5, 0], five.image_times[5, 0, 1:4], atol=0, rtol=0)
    assert torch.equal(after.image_times[:, 1], three.image_times[:, 1])
    broken = robot_pack()
    broken['data']['memory']['features'][1, 0, 1] += 1   # the same exposure, other tokens, in the next block
    with pytest.raises(ValueError, match='two sets of tokens'):
        robot_memory_index(broken, 5)


def test_the_memory_starts_empty_with_every_actual_state():
    from fcpg.v21_picture_memory import RobotPictureMemory, robot_memory_index
    pack = robot_pack(blocks=8, reset=4)                  # blocks 4-7 of each session: the next typed task
    index = robot_memory_index(pack, 6)
    assert index['block_state'].tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    stream = RobotPictureMemory(index, 'cpu')
    memory, data = pack['data']['memory'], pack['data']
    ids = np.arange(16)
    three = stream.memory(ids, data['now'], 3)            # the snapshots' own pictures, also after the reset
    assert torch.equal(three.valid, memory['valid']) and torch.equal(three.image_times, memory['image_times'])
    six = stream.memory(ids, data['now'], 6)
    assert six.valid[3].sum(-1).tolist() == [4, 3]        # the first state's four and three pictures
    assert six.valid[4].sum(-1).tolist() == [1, 0]        # the next state starts empty: nothing from before
    assert six.valid[7].sum(-1).tolist() == [4, 3]
    assert (six.image_times[4:8][six.valid[4:8]] >= 40.).all()
    with pytest.raises(ValueError, match='counter'):
        robot_memory_index(dict(pack, data=dict(data, memory={k: v for k, v in memory.items()
                                                              if k != 'current_readback_index'})), 6)
