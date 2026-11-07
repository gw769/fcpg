"""Section 17.84 (docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md): compiled regions of a training update."""
import pytest
import torch

from fcpg import v21_compiled
from fcpg.prod_v20_policy import empty_history
from fcpg.v21_compiled import region, regions
from fcpg.v21_encoder import DIM, PATCHES
from fcpg.v21_memory import ReadLayer, ReadbackMemoryReader, VisualMemory, VisualObservation
from fcpg.v21_mirror import flipped_features
from fcpg.v21_models import _DropoutDecoder, _DropoutEncoder

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU (Inductor programs)')


@pytest.fixture(autouse=True)
def fresh():
    v21_compiled.reset()
    yield
    v21_compiled.reset()


@pytest.fixture
def uncompiled(monkeypatch):
    """torch.compile replaced by the identity: the regions' code paths without Inductor."""
    monkeypatch.setattr(torch, 'compile', lambda fn, **kwargs: fn)


def memory(batch, *, wrist_only=True, seed=0, device='cpu'):
    g = torch.Generator().manual_seed(seed)
    valid = torch.rand(batch, 2, 3, generator=g) < .7
    valid[0] = False                                    # a row without any picture
    if wrist_only:
        valid[:, 1] = False
    features = torch.randn(batch, 2, 3, PATCHES, DIM, generator=g) * valid[..., None, None]
    times = (10. + torch.arange(3, dtype=torch.float64)).expand(batch, 2, 3).clone()
    return VisualMemory(features.to(device), (.3 * torch.randn(batch, 2, 3, 6, generator=g)).to(device),
                        times.to(device), valid.to(device), torch.zeros(batch, 2, 3, dtype=torch.long, device=device),
                        torch.zeros(batch, dtype=torch.long, device=device))


def read(reader, values, *, seed=1):
    b = len(values.features)
    g = torch.Generator().manual_seed(seed)
    device = values.features.device
    q = (.3 * torch.randn(b, 6, generator=g)).to(device)
    now = torch.full((b,), 20., dtype=torch.float64, device=device)
    task = None if reader.task_vocabulary is None else torch.randint(1, 9, (b, 5), generator=g).to(device)
    return reader(values, q, now, empty_history(b, device=device), task_ids=task)


def gradients(module):
    return {name: p.grad.clone() for name, p in module.named_parameters() if p.grad is not None}


def test_a_region_is_compiled_only_inside_regions(monkeypatch):
    made = []

    def fake_compile(fn, **kwargs):
        made.append(fn)
        return lambda *args, **kw: ('compiled', fn(*args, **kw))

    monkeypatch.setattr(torch, 'compile', fake_compile)
    double = region(lambda x: 2 * x)
    assert double(3) == 6 and not made
    with regions():
        assert v21_compiled.active() and double(3) == ('compiled', 6)
        with regions(False):
            assert not v21_compiled.active() and double(3) == 6
        assert double(4) == ('compiled', 8)
    assert not v21_compiled.active() and double(4) == 8 and len(made) == 1


def test_the_read_mask_is_the_indexed_assignment_it_replaced():
    torch.manual_seed(0)
    available = torch.rand(64, 3 * PATCHES) < .3
    available[::3] = False
    has = available.any(-1)
    before = available.clone()
    before[~has, 0] = True                                     # the assignment ReadLayer used to make
    after = available | ((~has)[:, None] & (torch.arange(available.shape[1]) == 0))
    assert torch.equal(before, after)
    layer = ReadLayer()
    out = layer(torch.randn(64, 8, 192), torch.randn(64, 3 * PATCHES, 192), torch.randn(64, 3 * PATCHES, 192),
                torch.randn(64, 3 * PATCHES, 192), available)
    assert torch.isfinite(out).all() and not out[~has].any()


@pytest.mark.parametrize('task', [False, True])
def test_the_region_read_is_the_plain_read(uncompiled, task):
    torch.manual_seed(0)
    reader = ReadbackMemoryReader(history_address=False, jaw_address=False, queries=32,
                                  task_vocabulary=9 if task else None).train()
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


@pytest.mark.parametrize('task', [False, True])
def test_a_wrist_only_read_skips_the_other_view(uncompiled, task):
    torch.manual_seed(0)
    reader = ReadbackMemoryReader(history_address=False, jaw_address=False, queries=32,
                                  task_vocabulary=9 if task else None).train()
    values = memory(5)
    plain = read(reader, values)
    plain.square().sum().backward()
    expected = gradients(reader)
    reader.zero_grad()
    reader.views = 1
    with regions():
        skipped = read(reader, values)
        skipped.square().sum().backward()
    assert not skipped[:, 1].any() and not plain[:, 1].any()
    torch.testing.assert_close(skipped, plain, rtol=1e-5, atol=1e-6)
    got = gradients(reader)
    assert got.keys() == expected.keys()
    # the same sums over fewer rows: float32 rounding only (relative size about 2e-7, all-l-20261022/
    # diagnose_wrist_read_20261023.py; in float64 the outputs are equal). Gradients that are zero in exact arithmetic
    # (key biases: softmax ignores them) are noise of the same absolute size, hence an absolute tolerance on the scale
    # of the largest gradient
    scale = float(max(g.abs().max() for g in expected.values()))
    for name, g in got.items():
        torch.testing.assert_close(g, expected[name], rtol=1e-4, atol=1e-6 * scale, msg=name)


def test_a_wrist_only_read_stops_at_a_picture_of_the_other_view(uncompiled):
    reader = ReadbackMemoryReader(queries=32).train()
    reader.views = 1
    with regions(), pytest.raises(ValueError, match='wrist-only memory holds a picture of the other view'):
        read(reader, memory(4, wrist_only=False))


def observation(batch, *, seed=0, time=10.):
    g = torch.Generator().manual_seed(seed)
    return VisualObservation(torch.randn(batch, 2, PATCHES, DIM, generator=g).half(), torch.randn(batch, 2, 6, generator=g),
                             torch.full((batch, 2), time, dtype=torch.float64),
                             torch.tensor([[True, False], [True, True], [False, False]])[torch.arange(batch) % 3])


def test_memory_writes_and_their_checks_are_unchanged(uncompiled):
    now = torch.full((6,), 10., dtype=torch.float64)
    start = VisualMemory.empty(6, dtype=torch.float16)
    for inside in (False, True):
        with regions(inside):
            written, inserted = start.write(observation(6), now)
            again, _ = written.write(observation(6), now)          # the same exposure again: nothing changes
            if inside:
                assert all(torch.equal(getattr(written, k), getattr(plain, k)) for k in
                           ('features', 'q_at_image', 'image_times', 'valid', 'readback_indices'))
                assert torch.equal(inserted, plain_inserted)
            plain, plain_inserted = written, inserted
            assert torch.equal(again.features, written.features)
            changed = observation(6, seed=1)
            with pytest.raises(ValueError, match='same exposure identity has changed features or q'):
                written.write(changed, now)
            broken = observation(6)
            broken.features[1, 0, 3, 5] = float('nan')
            with pytest.raises(ValueError, match='actual image features: expected finite floating values'):
                start.write(broken, now)


def test_the_flip_is_unchanged(uncompiled):
    features = torch.randn(4, 3, 2, PATCHES, DIM).half()
    rows = torch.tensor([True, False, True, False])
    plain = flipped_features(features, rows)
    with regions():
        assert torch.equal(flipped_features(features, rows), plain)


@cuda
def test_the_compiled_read_agrees_with_the_plain_read():
    torch.manual_seed(0)
    reader = ReadbackMemoryReader(history_address=False, jaw_address=False, queries=32, task_vocabulary=9).cuda().train()
    values = memory(6, device='cuda')
    plain = read(reader, values)
    plain.square().sum().backward()
    expected = gradients(reader)
    reader.zero_grad()
    reader.views = 1
    with regions():
        compiled = read(reader, values)
        compiled.square().sum().backward()
    torch.testing.assert_close(compiled, plain, rtol=1e-4, atol=1e-5)
    for name, g in gradients(reader).items():
        torch.testing.assert_close(g, expected[name], rtol=1e-3, atol=1e-5, msg=name)


@cuda
@pytest.mark.parametrize('kind', ['encoder', 'decoder'])
def test_compiled_actor_layers_agree_with_plain_layers(kind):
    torch.manual_seed(0)
    layer = (_DropoutEncoder(384, 12, .1) if kind == 'encoder' else _DropoutDecoder(384, 12, .1)).cuda().eval()
    context = torch.randn(8, 70, 384, device='cuda')
    padding = torch.rand(8, 70, device='cuda') < .3
    padding[:, 0] = False
    query = torch.randn(8, 40, 384, device='cuda', requires_grad=True)
    call = (lambda: layer(context, padding=padding)) if kind == 'encoder' else (lambda: layer(query, context, None, padding))
    plain = call()
    plain.square().sum().backward()
    expected = gradients(layer)
    layer.zero_grad()
    for text in (70, 61):                 # another context length: compiled once more, with a dynamic length
        with regions():
            compiled = call()
            compiled.square().sum().backward()
        if text == 70:
            torch.testing.assert_close(compiled, plain, rtol=1e-4, atol=1e-5)
            for name, g in gradients(layer).items():
                torch.testing.assert_close(g, expected[name], rtol=1e-3, atol=1e-4, msg=name)
        context, padding = context[:, :61], padding[:, :61]
        layer.zero_grad()
