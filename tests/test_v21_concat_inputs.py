"""V21Config.concat_inputs and V21FullPatchLeWM(concat_inputs=True) (James 11/02 「能concat的就concat 然后改代码」
「能合并的就合并网络 不要把linear放在外面」, Fig. 1 b, c): the FCPG reader and the picture-memory LeWM concatenate their
inputs and the network that reads them reads them as they are, in place of added vectors and of the scale-and-shift
(FiLM), with no network before them (「或者是直接输入到fcpg呢」). The reader's attention: [picture features, patch position]
for its values, with the read address for its keys, [learned query, current joints] for its queries. The LeWM
(「不用30张压缩后的 直接都给lewm」): no pool; the query layers read every patch of every picture of the memory as [features,
position code, camera code, its picture's numbers] and each step as [learned step query, step inputs, recent motion,
control setting]. Both keep what the added versions guarantee, every loader rebuilds them, and the default builds
every earlier model.
nn.MultiheadAttention's inference fast path is off, as in the trainer."""
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from fcpg import v21_compiled, v21_full_patch_lewm
from fcpg.v21_compiled import regions
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_encoder import DIM, PATCHES
from fcpg.v21_full_patch_lewm import BANK_INPUTS, POSITION_CODE, V21FullPatchLeWM, _attend, architecture_from_state
from fcpg.v21_horizon_training import factual_state
from fcpg.v21_memory import ADDRESS, ReadLayer, ReadbackMemoryReader
from fcpg.v21_native_horizon import core_from_parent
from fcpg.v21_picture_memory import PictureMemory
from fcpg.v21_temporal_contract import PROGRESS_TRAINING_POLICY, begin_training_temporal_contract
from fcpg.v21_training import synthetic_chain
from test_v21_compiled import gradients, memory, read
from test_v21_lewm_picture_memory import N, call, forecast, memory_for
from v21_lewm_helpers import setup

CONCAT = dict(controller_input=True, balanced_inputs=True, command_symlog=True, memory_pictures=N, memory_tokens=4,
              concat_inputs=True)
ADDED = ('feature_in', 'patch_xy', 'camera', 'frame', 'film', 'memory_film', 'memory_kind', 'controller_weight',
         'history')


@pytest.fixture(autouse=True)
def no_attention_fast_path():
    before = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    v21_compiled.reset()
    yield
    v21_compiled.reset()
    torch.backends.mha.set_fastpath_enabled(before)


@pytest.fixture
def uncompiled(monkeypatch):
    """torch.compile replaced by the identity: the regions' code paths without Inductor."""
    monkeypatch.setattr(torch, 'compile', lambda fn, **kwargs: fn)


@pytest.fixture
def vocabulary():
    return TaskVocabulary.build(['pick red block', 'pick up the red fish toy'])


def build(vocabulary, **switches):
    torch.manual_seed(5)
    core = V21ResearchCore(V21Config(len(vocabulary.words), object_relative=True, prediction_horizon=40,
                                     actor_history=False, reader_task=True, gripper_separate=True, **switches)).eval()
    begin_training_temporal_contract(core, PROGRESS_TRAINING_POLICY)
    return core


def test_a_concatenating_reader_has_its_parts_and_the_actor_is_unchanged(vocabulary):
    core, added = build(vocabulary, task_text=False, concat_inputs=True), build(vocabulary, task_text=False)
    state, before = core.state_dict(), added.state_dict()
    gone = {f'reader.{part}.{p}' for part in ('feature_projection', 'position', 'address.0', 'address.2',
                                              'pose_queries.0', 'pose_queries.2', 'layers.0.kv_norm',
                                              'layers.1.kv_norm')
            for p in ('weight', 'bias')}
    assert set(before) - set(state) == gone and set(state) <= set(before)     # no network before the attention
    for i in (0, 1):
        assert state[f'reader.layers.{i}.key.weight'].shape == (192, DIM + 2 + ADDRESS)
        assert state[f'reader.layers.{i}.value.weight'].shape == (192, DIM + 2)  # the picture only
        assert state[f'reader.layers.{i}.q_projection.weight'].shape == (192, 192 + 6)  # [query, current joints]
    assert ({k: v.shape for k, v in state.items() if not k.startswith('reader.')}
            == {k: v.shape for k, v in before.items() if not k.startswith('reader.')})
    with pytest.raises(ValueError, match='concat_inputs'):
        V21Config(len(vocabulary.words), reader_task=True, concat_inputs=True)          # it would read the words
    with pytest.raises(ValueError, match='concat_inputs'):
        V21Config(len(vocabulary.words), concat_inputs=1)
    with pytest.raises(ValueError, match='concatenating reader'):
        ReadbackMemoryReader(task_vocabulary=9, concat=True)


def test_a_concatenating_reader_follows_the_joints_and_the_pictures_not_the_words(vocabulary):
    chain = synthetic_chain(vocabulary, batch=2)
    core = build(vocabulary, task_text=False, concat_inputs=True)
    with torch.no_grad():
        state = factual_state(core, chain, 2, images=True)
        assert state.memory.valid.any()
        read = core.reader(state.memory, state.q, state.time, state.history)
        words = core.reader(state.memory, state.q, state.time, state.history, task_ids=chain.text_ids)
        moved = core.reader(state.memory, state.q + 5., state.time, state.history)        # from the eval cache
        noise = torch.randn(state.memory.features.shape, generator=torch.Generator().manual_seed(3))
        other = replace(state.memory, features=state.memory.features + noise * state.memory.valid[..., None, None])
        seen = core.reader(other, state.q, state.time, state.history)
    torch.testing.assert_close(read, words, rtol=0, atol=0)
    assert (moved - read).abs().max() > 1e-5 and (seen - read).abs().max() > 1e-5


def test_a_concatenating_reader_trains(vocabulary):
    chain = synthetic_chain(vocabulary, batch=2)
    core = build(vocabulary, task_text=False, concat_inputs=True)
    with torch.no_grad():
        state = factual_state(core, chain, 2, images=True)
    core.reader(state.memory, state.q, state.time, state.history).square().sum().backward()
    reader = core.reader
    assert reader.queries.grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in reader.layers.parameters())


def test_a_concatenating_read_layer_keeps_a_row_without_pictures_empty():
    torch.manual_seed(0)
    available = torch.rand(64, 3 * PATCHES) < .3
    available[::3] = False
    has = available.any(-1)
    inputs = (torch.randn(64, 8, 192), torch.randn(64, 3 * PATCHES, DIM + 2), torch.randn(64, 3 * PATCHES, 192),
              torch.randn(64, 3 * PATCHES, ADDRESS), available)
    for out in (ReadLayer(True)(*inputs), ReadLayer(True, True)(*inputs, torch.randn(64, 6))):
        assert torch.isfinite(out).all() and not out[~has].any() and out[has].abs().sum() > 0


@pytest.mark.parametrize('pose', [False, True])
def test_the_region_read_of_a_concatenating_reader_is_its_plain_read(uncompiled, pose):
    torch.manual_seed(0)
    reader = ReadbackMemoryReader(history_address=False, jaw_address=False, queries=32, pose_queries=pose,
                                  concat=True).train()
    values = memory(5, wrist_only=False)
    plain = read(reader, values)
    plain.square().sum().backward()
    expected = gradients(reader)
    reader.zero_grad()
    with regions():
        same = read(reader, values)
        same.square().sum().backward()
    assert torch.equal(same, plain)                 # the same operations in the same order
    assert all(torch.equal(g, expected[k]) for k, g in gradients(reader).items())


def test_a_concatenating_lewm_has_its_parts_and_refuses_what_it_cannot_read():
    core, _, _, _, _ = setup(**CONCAT)
    lewm = core.lewm
    assert not any(hasattr(lewm, name) for name in ADDED + ('memory_pool', 'memory_queries'))      # no pool
    assert BANK_INPUTS == DIM + 2 * POSITION_CODE + 19 + 5 and BANK_INPUTS % 8 == 0    # aligned for the tensor cores
    assert all(layer.cross_attn.k_proj_weight.shape == (lewm.width, BANK_INPUTS) and not hasattr(layer, 'norm_m')
               for layer in lewm.layers)                         # every patch as it is, no per-layer bank norm
    assert lewm.query_in.in_features == lewm.width + 20 + 303 + 2          # [step query, inputs, motion, setting]
    assert lewm.memory_position.shape == (PATCHES, POSITION_CODE) and lewm.memory_camera.shape == (2, POSITION_CODE)
    assert lewm.architecture()['concat_inputs'] is True and all(layer.direct for layer in lewm.layers)
    base = dict(horizon=40, controller_input=True, balanced_inputs=True, memory_pictures=3)
    for wrong in (dict(concat_inputs=1), dict(memory_pictures=0), dict(controller_input=False),
                  dict(balanced_inputs=False)):
        with pytest.raises(ValueError, match='concat_inputs'):
            V21FullPatchLeWM(**{**base, 'concat_inputs': True, **wrong})
    default = V21FullPatchLeWM(**base)
    assert all(hasattr(default, name) for name in ADDED) and not hasattr(default, 'concat_inputs')


def test_the_bank_is_every_patch_of_every_picture_and_a_pictures_patches_depend_on_it_only():
    core, state, _, targets, _ = setup(**CONCAT)
    lewm = core.lewm.eval()
    memory = memory_for(state)
    b = len(state.q)
    with torch.no_grad():
        bank, pad = lewm._memory_bank(memory.features, memory.valid, state.q, memory, state.time)
        features = memory.features.clone()
        features[:, 0, 2] += torch.randn(features[:, 0, 2].shape, generator=torch.Generator().manual_seed(4))
        changed = replace(memory, features=features)
        other, _ = lewm._memory_bank(changed.features, changed.valid, state.q, changed, state.time)
    assert bank.shape == (b, 2 * N * PATCHES, BANK_INPUTS)
    readable = (~pad).reshape(b, 2, N, PATCHES)
    assert readable[:, 0].all() and readable[:, 1, -2:].all() and not readable[:, 1, :-2].any()   # taken pictures only
    picked = torch.zeros(2, N, dtype=torch.bool)
    picked[0, 2] = True
    patches, moved = (x.reshape(b, 2, N, PATCHES, -1) for x in (bank, other))
    torch.testing.assert_close(moved[:, ~picked], patches[:, ~picked], atol=0, rtol=0)
    assert not torch.allclose(moved[:, 0, 2], patches[:, 0, 2], atol=1e-5)
    torch.testing.assert_close(patches[:, 0, 2, :, :DIM], torch.nn.functional.layer_norm(memory.features[:, 0, 2],
                                                                                          (DIM,)))   # normalized
    assert not patches[..., -5:].any()                                                   # the alignment's zeros
    assert torch.isfinite(forecast(lewm, state, targets, memory)).all()


def test_empty_slots_change_nothing_and_time_joints_and_setting_are_read():
    core, state, _, targets, _ = setup(**CONCAT)
    lewm = core.lewm.eval()
    memory = memory_for(state)
    base = forecast(lewm, state, targets, memory)
    noise = memory.features.clone()
    noise[:, 1, :-2] = torch.randn(noise[:, 1, :-2].shape, generator=torch.Generator().manual_seed(1))
    filled = replace(memory, features=noise, q_at_image=memory.q_at_image + (~memory.valid[..., None]) * 7.)
    torch.testing.assert_close(forecast(lewm, state, targets, filled), base, atol=0, rtol=0)
    for changed in (replace(memory, image_times=memory.image_times - 3.),
                    replace(memory, q_at_image=memory.q_at_image + .5 * memory.valid[..., None])):
        assert not torch.allclose(forecast(lewm, state, targets, changed), base, atol=1e-6)
    with torch.no_grad():
        flipped = lewm(state.q, targets.trace, state.history, targets.readback_offsets, patches=memory.features,
                       patch_valid=memory.valid, memory=memory, now_time=state.time,
                       controller=torch.tensor([[1., 0.], [0., 1.]])).predicted_q_path
    assert not torch.allclose(flipped, base, atol=1e-6)


def test_a_row_without_pictures_forecasts_as_with_none():
    core, state, _, targets, _ = setup(**CONCAT)
    lewm = core.lewm.eval()
    memory = memory_for(state, taken=(0, 0))
    controller = torch.tensor([[0., 0.], [1., 1.]])[:len(state.q)]
    with torch.no_grad():
        without = lewm(state.q, targets.trace, state.history, targets.readback_offsets, controller=controller)
    torch.testing.assert_close(forecast(lewm, state, targets, memory), without.predicted_q_path, atol=1e-6, rtol=1e-6)


def test_the_planners_shared_memory_is_read_once_with_the_same_forecasts():
    core, state, _, targets, _ = setup(**CONCAT)
    lewm = core.lewm.eval()
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
            a = lewm(q, trace, history, row(targets.readback_offsets), patches=shared.features,
                     patch_valid=shared.valid, memory=shared, **kwargs).predicted_q_path
            b = lewm(q.contiguous(), trace, history, row(targets.readback_offsets), patches=copied.features,
                     patch_valid=copied.valid, memory=copied, **{**kwargs, 'now_time': kwargs['now_time'].contiguous()}
                     ).predicted_q_path
    finally:
        del lewm._memory_bank
    assert seen == [1, n]                                             # once for the shared memory, per row for copies
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_the_concatenated_parts_train():
    core, state, _, targets, _ = setup(**CONCAT)
    lewm = core.lewm.train()
    call(lewm, state, targets, memory_for(state)).predicted_q_path.square().sum().backward()
    for name in ('memory_position', 'memory_camera', 'query'):
        assert getattr(lewm, name).grad.abs().sum() > 0
    for module in (lewm.query_in, *(layer.cross_attn for layer in lewm.layers)):
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_an_attention_of_wider_keys_attends_as_the_module_does():
    torch.manual_seed(0)
    attention = torch.nn.MultiheadAttention(48, 4, batch_first=True, kdim=70, vdim=70).eval()
    query, memory = torch.randn(3, 5, 48), torch.randn(3, 11, 70)
    readable = torch.rand(3, 11) < .7
    readable[:, 0] = True
    with torch.no_grad():
        mine = _attend(attention, query, memory, readable=readable[:, None, None])
        theirs = attention(query, memory, memory, key_padding_mask=~readable, need_weights=False)[0]
    torch.testing.assert_close(mine, theirs, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or not v21_full_patch_lewm._flash_capable(torch.device('cuda', 0)),
                    reason='FlashAttention needs a GPU that runs it (the A100)')
def test_the_training_read_by_flash_attention_is_the_masked_read(monkeypatch):
    """James 11/02 「用那个FlashAttention加速也可以阿」: a concatenating LeWM's training read of its bank (FlashAttention,
    the padding keys shut by one more channel) gives the masked read's values and gradients, up to BF16 rounding."""
    calls = []
    flash = v21_full_patch_lewm._flash_read
    monkeypatch.setattr(v21_full_patch_lewm, '_flash_read', lambda *a: calls.append(1) or flash(*a))
    torch.manual_seed(0)
    attention = torch.nn.MultiheadAttention(96, 2, dropout=0., batch_first=True, kdim=72, vdim=72).cuda()
    query, memory = torch.randn(4, 40, 96, device='cuda'), torch.randn(4, 300, 72, device='cuda')
    readable = torch.rand(4, 300, device='cuda') < .6
    readable[:, 0] = True
    readable[1, 1:] = False                       # a row with one readable key, as _one_readable_token leaves it
    runs = []
    for training in (True, False):                # FlashAttention, then the masked read
        attention.zero_grad()
        q, m = query.clone().requires_grad_(), memory.clone().requires_grad_()
        with torch.autocast('cuda', torch.bfloat16):
            out = _attend(attention, q, m, readable=readable[:, None, None], training=training)
        out.float().square().sum().backward()
        runs.append([out.float(), q.grad, m.grad] + [p.grad.clone() for p in attention.parameters()])
    assert calls == [1]                           # the training read went through FlashAttention, the other did not
    for mine, theirs in zip(*runs):
        torch.testing.assert_close(mine, theirs, atol=2e-2 * theirs.abs().max().item(), rtol=2e-2)
    assert not runs[0][2][~readable].any()        # nothing flows to a padding key


def test_every_loader_rebuilds_a_concatenating_lewm():
    core, state, _, targets, _ = setup(**CONCAT)
    whole = core.state_dict()
    assert architecture_from_state(whole, 'lewm.') == dict(width=384, layers=4, controller_input=True,
                                                           balanced_inputs=True, command_symlog=True,
                                                           memory_pictures=N, concat_inputs=True)
    rebuilt = core_from_parent(dict(config=asdict(core.config), model=whole)).eval()
    memory = memory_for(state)
    torch.testing.assert_close(forecast(rebuilt.lewm, state, targets, memory),
                               forecast(core.lewm.eval(), state, targets, memory), atol=0, rtol=0)
    with pytest.raises(RuntimeError):
        V21FullPatchLeWM(horizon=40, controller_input=True, balanced_inputs=True, command_symlog=True,
                         memory_pictures=N, memory_tokens=4).load_state_dict(core.lewm.state_dict(), strict=True)
