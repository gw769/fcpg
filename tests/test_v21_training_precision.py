"""Explicit FP32 routing and a synthetic CPU step; no GPU or saved weights."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_training import V21Trainer, synthetic_chain


@pytest.mark.parametrize('device,precision,autocast_calls', [
    ('cpu', 'bf16', 0), ('cpu', 'fp32', 0), ('cuda', 'fp32', 0), ('cuda', 'bf16', 1),
])
def test_precision_routes_context_without_initializing_cuda(monkeypatch, device, precision, autocast_calls):
    calls = []
    def autocast(*args, **kwargs):
        calls.append((args, kwargs))
        return nullcontext()
    monkeypatch.setattr(torch, 'autocast', autocast)
    trainer = SimpleNamespace(device=torch.device(device), precision=precision)
    with V21Trainer._compute_context(trainer):
        pass
    assert len(calls) == autocast_calls
    if calls:
        assert calls == [(('cuda',), dict(dtype=torch.bfloat16))]


@pytest.mark.parametrize('precision', ['fp16', 'auto', None])
def test_unsupported_precision_rejected_before_model_or_device_access(tmp_path, precision):
    with pytest.raises(ValueError, match='precision'):
        V21Trainer(None, None, tmp_path, precision=precision)


def test_fp32_synthetic_step_is_finite_and_preserves_frozen_state(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        vocabulary = TaskVocabulary.build(['pick red block'])
        core = V21ResearchCore(V21Config(len(vocabulary.words), object_relative=True))
        batch = synthetic_chain(vocabulary, batch=2)
        cue = torch.tensor([1., .8, .1, .2, .3, .1, 1., 0.]).expand(2, 5, 8).clone()
        batch = replace(batch, object_relative=cue)
        trainer = V21Trainer(core, vocabulary, tmp_path, microbatch=2, precision='fp32')
        optimizer = trainer.configure('fish')
        before = {name: value.detach().clone() for name, value in core.state_dict().items()}
        labels = batch.commands.clone()
        result = trainer.step([batch], optimizer, 'fish')
        assert result['precision'] == 'fp32'
        assert torch.isfinite(torch.tensor([result['loss'], result['gradient_norm']])).all()
        changed = [name for name, value in core.state_dict().items() if not torch.equal(value, before[name])]
        assert changed and all(name.startswith(('actor.', 'reader.')) for name in changed)
        assert all(not value.requires_grad and value.grad is None for value in core.lewm.parameters())
        assert all(value.grad is None or torch.isfinite(value.grad).all() for value in core.parameters())
        assert torch.equal(batch.commands, labels) and not list(tmp_path.iterdir())
    finally:
        torch.set_num_threads(previous_threads)
