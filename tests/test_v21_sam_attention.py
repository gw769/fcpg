"""17.163: the SAM 3.1 attention supervision and label head of the full-patch LeWM, the label table and the bank labels."""
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg import v21_full_patch_lewm
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_encoder import PATCHES
from fcpg.v21_full_patch_lewm import V21FullPatchLeWM, architecture_from_state
from fcpg.v21_horizon_training import horizon_targets
from fcpg.v21_lewm_inputs import lewm_inputs
from fcpg.v21_native_horizon import core_from_parent
from fcpg.v21_sam_labels import (EPS, SCHEMA, TARGET, SamLabelTable, attention_penalty, attention_weights,
                                 bank_labels, future_label_loss)
from test_v21_scratch_features import actual_chain_source


def inputs(**size):
    torch.manual_seed(21004)
    source = actual_chain_source()
    source.offsets = np.array([0, len(source.raw_q)])
    source.archive.endpoint_episode_index = np.zeros(8, dtype=np.int64)
    core = V21ResearchCore(V21Config(len(source.vocabulary.words), prediction_horizon=40, object_relative=True)).eval()
    rows = np.array([0, 1])
    with torch.no_grad():       # the LeWM stages' inputs (the trainer's lewm_inputs: no reader, no forecasts)
        s = lewm_inputs(core, source.chain(rows, 'archive', images=True), 2)
    core.lewm = V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40, **size).eval()
    targets = horizon_targets(source, rows, 'archive', 2)
    kwargs = dict(patches=s.memory.features, patch_valid=s.memory.valid, memory=s.memory, now_time=s.time)
    return core, (s.q, targets.trace, s.history, targets.readback_offsets), kwargs


def recorded_attention(lewm, args, kwargs):
    """Every layer's picture cross-attention probabilities [B, heads, queries, keys], as _attend scores them."""
    kept, original = [], v21_full_patch_lewm._attend

    def attend(attention, query, memory, *, causal=False, readable=None, training=False):
        if readable is not None:
            width, heads = attention.embed_dim, attention.num_heads
            w, b = attention.in_proj_weight, attention.in_proj_bias
            q = torch.nn.functional.linear(query, w[:width], b[:width]).unflatten(-1, (heads, -1)).transpose(1, 2)
            k = torch.nn.functional.linear(memory, w[width:2 * width], b[width:2 * width]).unflatten(-1, (heads, -1))
            scores = q @ k.transpose(1, 2).transpose(-1, -2) / (width // heads) ** .5
            kept.append(scores.masked_fill(~readable, float('-inf')).softmax(-1))
        return original(attention, query, memory, causal=causal, readable=readable, training=training)
    v21_full_patch_lewm._attend = attend
    try:
        with torch.no_grad():
            out = lewm(*args, **kwargs)
    finally:
        v21_full_patch_lewm._attend = original
    return kept, out


@pytest.mark.parametrize('per_head', [False, True])
@pytest.mark.parametrize('size', [dict(width=64, layers=2), dict(balanced_inputs=True)])
def test_the_mass_is_the_attention_the_first_heads_give_the_labelled_keys(size, per_head):
    core, args, kwargs = inputs(**size)
    shape = (2, 4, 6 * PATCHES) if per_head else (2, 6 * PATCHES)
    weights = torch.rand(*shape) * (torch.rand(*shape) > .7)
    attention, plain = recorded_attention(core.lewm, args, kwargs)
    with torch.no_grad():
        out = core.lewm(*args, **kwargs, attention_supervision=(4, weights))
    w = weights[:, :, None] if per_head else weights[:, None, None]
    expected = torch.stack([(p[:, :4] * w).sum(-1) for p in attention])
    assert out.attention_mass.shape == (len(core.lewm.layers), 2, 4, 40)
    torch.testing.assert_close(out.attention_mass, expected, atol=1e-5, rtol=1e-5)
    # the supervision only reads: the forecast is the same, bit for bit
    assert torch.equal(out.predicted_q_path, plain.predicted_q_path) and torch.equal(out.hidden, plain.hidden)
    assert type(plain).__name__ == 'ARMOutput' and not hasattr(plain, 'attention_mass')


def test_supervision_needs_pictures_direct_attention_and_weights_in_0_1():
    core, args, kwargs = inputs(width=64, layers=2)
    good = torch.zeros(2, 6 * PATCHES)
    with pytest.raises(ValueError):
        core.lewm(*args, attention_supervision=(4, good))                  # no pictures
    with pytest.raises(ValueError):
        core.lewm(*args, **kwargs, attention_supervision=(4, good + 2))     # weights above 1
    with pytest.raises(ValueError):
        core.lewm(*args, **kwargs, attention_supervision=(9, good))         # more heads than the layer has
    deployed, args, kwargs = inputs()
    assert not any(layer.direct for layer in deployed.lewm.layers)
    with pytest.raises(ValueError):
        deployed.lewm(*args, **kwargs, attention_supervision=(4, good))


def test_the_mass_has_gradients_and_the_penalty_pulls_it_up():
    core, args, kwargs = inputs(width=64, layers=2)
    core.lewm.train()
    weights = torch.zeros(2, 6 * PATCHES)
    weights[:, :PATCHES] = 1.                                               # the first picture of camera 0
    out = core.lewm(*args, **kwargs, attention_supervision=(4, weights))
    rows = torch.ones(2, 4, dtype=torch.bool)
    penalty, mass = attention_penalty(out.attention_mass, rows)
    penalty.backward()
    grad = core.lewm.layers[0].cross_attn.in_proj_weight.grad
    assert grad is not None and grad[:4 * 8].abs().sum() > 0                 # the supervised heads' queries move
    assert torch.allclose(penalty, (np.log(TARGET) - (out.attention_mass + EPS).log()).clamp_min(0).mean())
    assert 0 < mass < TARGET and float(penalty) > 0
    none, missing = attention_penalty(out.attention_mass, torch.zeros(2, 4, dtype=torch.bool))
    assert float(none) == 0 and missing is None
    # a head that already gives the labelled keys at least TARGET of its attention is left alone
    with torch.no_grad():
        full = core.lewm(*args, **kwargs, attention_supervision=(4, torch.ones(2, 6 * PATCHES)))
    assert float(attention_penalty(full.attention_mass, rows)[0]) == 0


def test_bank_labels_find_each_picture_by_its_capture_time():
    coverage = torch.arange(2 * 3 * 2 * 2 * PATCHES, dtype=torch.int64).remainder(251).to(torch.uint8)
    coverage = coverage.reshape(2, 3, 2, 2, PATCHES)                         # [B, J, view, kind, P]
    labelled = torch.ones(2, 3, 2, dtype=torch.bool)
    labelled[1, 2, 0] = False                                                # one valid picture without a label
    times = torch.tensor([[[1., 1.5], [2., 2.5], [3., 3.5]], [[10., 10.], [11., 12.], [13., 14.]]], dtype=torch.float64)
    valid = torch.ones(2, 3, 2, dtype=torch.bool)
    # memory, row 0: view 0 holds chain pictures 0, 1, 2; view 1 an empty first slot, then pictures 1 and 2;
    # row 1: both views hold pictures 0, 1, 2, and view 0's picture 2 has no label
    image_times = torch.tensor([[[1., 2., 3.], [0., 2.5, 3.5]], [[10., 11., 13.], [10., 12., 14.]]], dtype=torch.float64)
    bank_valid = torch.tensor([[[True, True, True], [False, True, True]], [[True, True, True], [True, True, True]]])
    got, kept = bank_labels(image_times, bank_valid, times, valid, coverage, labelled)
    assert got.shape == (2, 2, 3, 2, PATCHES) and kept.shape == (2, 2, 3)
    picks = {(0, 0): [0, 1, 2], (0, 1): [None, 1, 2], (1, 0): [0, 1, 2], (1, 1): [0, 1, 2]}
    for (b, v), js in picks.items():
        for s, j in enumerate(js):
            if j is None:
                assert not kept[b, v, s] and not got[b, v, s].any()
            elif not labelled[b, j, v]:
                assert not kept[b, v, s] and not got[b, v, s].any()
            else:
                assert kept[b, v, s] and torch.equal(got[b, v, s], coverage[b, j, v])
    image_times[0, 0, 1] = 2.25                                              # a bank picture the chain does not have
    with pytest.raises(ValueError):
        bank_labels(image_times, bank_valid, times, valid, coverage, labelled)


def test_half_the_heads_get_the_object_and_half_the_robot_on_labelled_pictures():
    coverage = torch.zeros(1, 2, 3, 2, PATCHES, dtype=torch.uint8)
    coverage[0, 0, 2, 0, 5], coverage[0, 0, 2, 1, 5] = 51, 255               # object .2 and robot 1 on one patch
    coverage[0, 1, 0, 0, 7] = 102
    labelled = torch.tensor([[[False, True, True], [True, True, True]]])
    weights, rows = attention_weights(coverage, labelled, 4)
    assert weights.shape == (1, 4, 6 * PATCHES) and rows.tolist() == [[True] * 4]
    for head in (0, 1):                                                       # the object's heads
        assert abs(float(weights[0, head, 2 * PATCHES + 5]) - .2) < 1e-6
        assert abs(float(weights[0, head, 3 * PATCHES + 7]) - .4) < 1e-6 and int((weights[0, head] > 0).sum()) == 2
    for head in (2, 3):                                                       # the robot's heads
        assert weights[0, head, 2 * PATCHES + 5] == 1. and int((weights[0, head] > 0).sum()) == 1
    coverage[..., 1, :] = 0                                                   # no robot anywhere
    assert attention_weights(coverage, labelled, 4)[1].tolist() == [[True, True, False, False]]
    labelled[:] = False
    assert not attention_weights(coverage, labelled, 4)[1].any()


def test_the_future_label_loss_reads_the_query_nearest_each_later_picture():
    torch.manual_seed(3)
    head = torch.nn.Linear(8, 4 * PATCHES)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    hidden = torch.randn(2, 40, 8)
    offsets = torch.arange(1, 41, dtype=torch.float64)[None].expand(2, -1) * .1
    endpoint_times = torch.tensor([[0., .5, 1., 1.5, 2.], [0., .5, 1., 1.5, 9.]], dtype=torch.float64)
    coverage = torch.randint(0, 256, (2, 5, 2, 2, PATCHES), dtype=torch.uint8)
    labelled = torch.ones(2, 5, 2, dtype=torch.bool)
    labelled[0, 3, 1] = False
    loss, pictures = future_label_loss(head, hidden, offsets, endpoint_times, 1, coverage, labelled)
    # zero logits: the binary cross-entropy is log 2 whatever the target; row 1's last picture is 8.5 s ahead (none)
    assert abs(float(loss) - float(np.log(2))) < 1e-6
    assert pictures == (3 * 2 - 1) + (2 * 2)
    none, count = future_label_loss(head, hidden, offsets, endpoint_times, 4, coverage, labelled)
    assert float(none) == 0 and count == 0


def test_the_label_head_draws_no_random_numbers_and_every_loader_rebuilds_it():
    torch.manual_seed(5)
    plain = V21FullPatchLeWM(horizon=40, width=64, layers=2)
    after_plain = torch.get_rng_state()
    torch.manual_seed(5)
    headed = V21FullPatchLeWM(horizon=40, width=64, layers=2, label_head=True)
    assert torch.equal(after_plain, torch.get_rng_state())
    shared = {k: v for k, v in headed.state_dict().items() if not k.startswith('label_head.')}
    assert all(torch.equal(v, plain.state_dict()[k]) for k, v in shared.items())
    assert not headed.label_head.weight.any() and not headed.label_head.bias.any()
    assert headed.architecture()['label_head'] is True and 'label_head' not in plain.architecture()
    assert architecture_from_state(headed.state_dict()) == dict(width=64, layers=2, label_head=True)
    core = V21ResearchCore(V21Config(20, prediction_horizon=40, object_relative=True))
    core.lewm = headed
    rebuilt = core_from_parent(dict(config=asdict(core.config), model=core.state_dict()))
    assert hasattr(rebuilt.lewm, 'label_head')
    assert all(torch.equal(a, b) for a, b in zip(rebuilt.lewm.state_dict().values(), headed.state_dict().values()))


def table(tmp_path, caches, *, labelled_on_invalid=False):
    root = tmp_path / 'table'
    for cache in caches:
        part = root / cache.root.name
        part.mkdir(parents=True)
        rows = len(cache.endpoint_index)
        np.save(part / 'endpoint_index.npy', cache.endpoint_index)
        coverage = (np.arange(rows * 2 * 2 * PATCHES) % 200).astype(np.uint8).reshape(rows, 2, 2, PATCHES)
        coverage += (cache.endpoint_index[:, None, None, None] % 50).astype(np.uint8)
        np.save(part / 'coverage.npy', coverage)
        labelled = cache.view_valid.copy()
        if labelled_on_invalid:
            labelled[~cache.view_valid] = True
        np.save(part / 'labelled.npy', labelled)
    (root / 'manifest.json').write_text(json.dumps(dict(schema=SCHEMA, complete=True, counts={}, missing_recordings=0)))
    return root


def caches(tmp_path):
    train = SimpleNamespace(root=tmp_path / 'temporal-train', endpoint_index=np.array([0, 2, 3, 7], np.int64),
                            view_valid=np.array([[1, 1], [1, 0], [0, 1], [1, 1]], bool))
    dev = SimpleNamespace(root=tmp_path / 'temporal-dev', endpoint_index=np.array([1, 4], np.int64),
                          view_valid=np.array([[1, 1], [1, 1]], bool))
    return [train, dev]


def test_the_label_table_serves_the_cache_rows_of_any_endpoint(tmp_path):
    parts = caches(tmp_path)
    labels = SamLabelTable(table(tmp_path, parts), parts)
    coverage, labelled = labels.get(np.array([[7, 1], [2, 0]]))
    assert coverage.shape == (2, 2, 2, 2, PATCHES) and labelled.shape == (2, 2, 2)
    stored = {name: np.load(tmp_path / 'table' / name / 'coverage.npy') for name in ('temporal-train', 'temporal-dev')}
    assert np.array_equal(coverage[0, 0], stored['temporal-train'][3]) and np.array_equal(coverage[0, 1], stored['temporal-dev'][0])
    assert np.array_equal(coverage[1, 0], stored['temporal-train'][1]) and labelled[1, 0].tolist() == [True, False]
    with pytest.raises(ValueError):
        labels.get(np.array([5]))                                            # no cache row has endpoint 5


def test_a_table_off_the_cache_rows_or_on_invalid_pictures_is_refused(tmp_path):
    parts = caches(tmp_path)
    root = table(tmp_path, parts)
    moved = SimpleNamespace(root=parts[0].root, endpoint_index=np.array([0, 2, 3, 8], np.int64),
                            view_valid=parts[0].view_valid)
    with pytest.raises(ValueError):
        SamLabelTable(root, [moved, parts[1]])
    other = tmp_path / 'other'
    other.mkdir()
    with pytest.raises(ValueError):
        SamLabelTable(table(other, caches(other), labelled_on_invalid=True), caches(other))
