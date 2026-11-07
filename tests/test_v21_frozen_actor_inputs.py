import numpy as np
import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_frozen_actor_inputs import FrozenActorInputBank
from fcpg.v21_horizon_training import actor_loss, factual_state, horizon_targets
from test_v21_scratch_features import actual_chain_source
from scripts.train_v21_horizon40 import initialize_scratch40, ordered_groups
from types import SimpleNamespace


def test_cached_factual_inputs_preserve_row_order_actor_gradients_and_invalidate_on_reader_update():
    torch.set_num_threads(2)
    torch.manual_seed(21004)
    source = actual_chain_source()
    source.offsets = np.array([0, len(source.raw_q)])
    source.archive.endpoint_episode_index = np.zeros(8, dtype=np.int64)
    core = V21ResearchCore(V21Config(len(source.vocabulary.words), prediction_horizon=40)).eval()
    core.requires_grad_(False)
    core.actor.requires_grad_(True)

    def prepare(kind, rows, h, images):
        chain = source.chain(np.asarray(rows), kind, images=images)
        return factual_state(core, chain, h, images=images), chain.text_ids, horizon_targets(source, rows, kind, h)

    bank = FrozenActorInputBank(core)
    for h in range(4):
        bank.build('archive', h, [0, 1], prepare, batch_size=1)
        cached = bank.get('archive', h, [1, 0, 1])
        state, ids, targets = prepare('archive', [1, 0, 1], h, True)
        direct = core.act(state, ids, targets.command_offsets).commands_position
        replayed = core.actor(**cached.inputs).commands_position
        torch.testing.assert_close(replayed, direct, atol=2e-5, rtol=1e-5)
        assert torch.equal(cached.commands, targets.commands) and torch.equal(cached.valid, targets.valid)
        core.zero_grad(set_to_none=True)
        actor_loss(replayed, cached, cached.inputs['q_position'], core.q_loss_scale).backward()
        cached_grad = core.actor.query.grad.clone()
        core.zero_grad(set_to_none=True)
        actor_loss(direct, targets, state.q, core.q_loss_scale).backward()
        torch.testing.assert_close(core.actor.query.grad, cached_grad, atol=1e-6, rtol=1e-4)
        assert all(p.grad is None for module in (core.reader, core.lewm) for p in module.parameters())
    with pytest.raises(ValueError, match='not included'):
        bank.get('archive', 0, [2])
    with torch.no_grad():
        core.actor.output.bias.add_(.01)
    changed = core.actor(**bank.get('archive', 3, [1, 0, 1]).inputs).commands_position
    assert not torch.equal(changed, replayed)
    with torch.no_grad():
        next(core.reader.parameters()).add_(.01)
    with pytest.raises(ValueError, match='changed'):
        bank.get('archive', 0, [0])


def test_scratch_native40_initialization_reads_no_checkpoint_and_updates_fcpg_and_actor(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: pytest.fail('scratch initialization read weights'))
    source = actual_chain_source()
    source.offsets = np.array([0, len(source.raw_q)])
    source.archive.endpoint_episode_index = np.zeros(8, dtype=np.int64)
    core = initialize_scratch40(source.vocabulary, np.full(6, 2., np.float32), 21004, 'cpu').eval()
    assert core.config.prediction_horizon == 40
    for module in (core.actor, core.lewm):
        assert module.query.shape == (1, 40, 192)
        assert not torch.equal(module.query[:, :5], module.query[:, 5:10])
    before = {name: value.clone() for name, value in core.state_dict().items()}
    core.lewm.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=1e-4)
    chain = source.chain(np.array([0, 1]), images=True)
    state = factual_state(core, chain, 3, images=True, train_reader=True)
    targets = horizon_targets(source, [0, 1], 'archive', 3)
    predicted = core.act(state, chain.text_ids, targets.command_offsets).commands_position
    actor_loss(predicted, targets, state.q, core.q_loss_scale).backward()
    for module in (core.reader, core.actor):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    optimizer.step()
    after = core.state_dict()
    for prefix in ('reader.', 'actor.'):
        assert any(not torch.equal(value, after[name]) for name, value in before.items() if name.startswith(prefix))
    assert all(torch.equal(value, after[name]) for name, value in before.items() if name.startswith('lewm.'))


def test_scratch_covers_every_actual_context_for_every_archive_row_each_pass():
    args = SimpleNamespace(seed=21004, batch_size=4, lewm_batch_size=8, archive_passes=2,
                           canary=False, from_scratch=True)
    splits = dict(train=dict(archive=np.arange(10)))
    for stage in ('lewm_archive', 'actor_archive'):
        counts = np.zeros((10, 4), dtype=int)
        for h, groups in ordered_groups(stage, splits, args):
            counts[groups[0][1], h] += 1
        assert np.all(counts == 2)
