"""Hand-computable DEV trajectories and preservation of ongoing training state."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from fcpg.v21_data import TrainingChain
from fcpg.v21_models import CommandTrace
from scripts.evaluate_v21_scratch_development import development_selection, evaluate_development


def trajectory(rows, *, images):
    b = len(rows)
    q = torch.arange(5, dtype=torch.float32)[None, :, None].expand(b, 5, 6)*5
    paths = q[:, :-1, None]+torch.arange(6, dtype=torch.float32)[None, None, :, None]
    commands = q[:, :-1, None]+10+torch.arange(5, dtype=torch.float32)[None, None, :, None]
    offsets = torch.arange(6, dtype=torch.float64)[None, None].expand(b, 4, 6)*.1
    times = torch.arange(5, dtype=torch.float64)[None].expand(b, 5)*.5
    traces = tuple(CommandTrace(commands[:, h], offsets[:, h, :5],
        torch.ones(b, 5, dtype=torch.bool), 'recorded', True) for h in range(4))
    return TrainingChain(q, times, commands, paths, offsets, traces,
        torch.ones(b, 5, 2, 1, 1) if images else None,
        torch.ones(b, 5, 2, dtype=torch.bool), torch.ones(b, 2, dtype=torch.long),
        torch.ones(b, dtype=torch.long), torch.ones(b, 5, 8))


class DevSource:
    def __init__(self):
        self.counts = dict(archive=5, fish=5)
        self.archive = SimpleNamespace(archive=SimpleNamespace(split=np.array([0, 1, 0, 1, 1], np.uint8)))
        self.fish = SimpleNamespace(starts=np.array([0, 61, 122, 183, 244]),
            episodes=np.repeat([0, 1, 2, 3, 3], 61),
            frame_indices=np.concatenate([np.arange(61)]*3+[np.arange(122)]))
        self.reads = []

    def chain(self, rows, kind, *, images=True):
        rows = np.asarray(rows)
        if kind == 'archive':
            assert (self.archive.archive.split[rows] == 1).all()
        else:
            assert (self.fish.episodes[self.fish.starts[rows]] == 3).all()
        self.reads.append((kind, rows.copy(), images))
        return trajectory(rows, images=images)


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.))
        self._projection_cache = object()
        self.nonfinite = False

    def forward(self, q):
        # Exercise all three global RNGs so restoring them is actually tested.
        random.random(); np.random.random(); torch.rand(1)
        value = q[:, None, None, :1].expand(-1, 2, 8, 192)*self.weight
        return value+float('nan') if self.nonfinite else value


class HandComputedCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.reader = Reader()
        self.actor = nn.Linear(1, 1)
        self.lewm = nn.Linear(1, 1)
        self.temporal_feature_coverage = dict(actor_calls=123)
        self._temporal_collect_training_coverage = True
        self.completed = []

    def observe(self, previous, observation, q, now):
        assert now.dtype == torch.float64
        grounding = self.reader(q)
        return SimpleNamespace(q=q, time=now, grounding=grounding)

    def predict(self, state, trace, offsets):
        assert offsets.dtype == torch.float64
        steps = torch.arange(6, dtype=torch.float32)[None, :, None]
        path = state.q[:, None]+steps
        path[:, 1:] += torch.arange(1, 7)[None, None]
        return SimpleNamespace(response=SimpleNamespace(predicted_q_path=path))

    def act(self, state, text, offsets):
        if self._temporal_collect_training_coverage:
            self.temporal_feature_coverage['actor_calls'] += 1
        command = state.q[:, None]+17+torch.arange(5)[None, :, None]
        return SimpleNamespace(commands_position=command)

    def complete(self, previous, prediction, trace, actual_path, times, observation):
        # A predicted-q rollout would fail these checks: history must be actual.
        expected = previous.q[:, None]+torch.arange(6)[None, :, None]
        torch.testing.assert_close(actual_path, expected, rtol=0, atol=0)
        assert times.dtype == torch.float64
        torch.testing.assert_close(times[:, 0], previous.time)
        self.completed.append(actual_path.clone())
        return self.observe(previous, observation, actual_path[:, -1], times[:, -1])


def dev_rows():
    return dict(archive_dev_rows=np.array([1, 3, 4]), fish_dev_rows=np.array([3, 4]))


@pytest.mark.parametrize('stage', ['lewm', 'actor'])
def test_hand_calculated_q_terminal_hold_and_command_errors(stage):
    source, core = DevSource(), HandComputedCore()
    report = evaluate_development(core, source, dev_rows(), stage=stage, microbatch=1, archive_limit=2)
    assert report['stage'] == stage and report['teacher_forced'] and report['development']
    assert not report['unseen_claim'] and not report['physical_success_verified']
    assert report['episode_2_values_read'] is False
    for kind, result in report['results'].items():
        assert result['chains'] == 2 and result['predicted_readbacks'] == 40
        np.testing.assert_array_equal(result['lewm']['mae_per_joint'], np.arange(1, 7))
        np.testing.assert_array_equal(result['lewm']['terminal_mae_per_joint'], np.arange(1, 7))
        np.testing.assert_array_equal(result['hold_last']['mae_per_joint'], [3]*6)
        np.testing.assert_array_equal(result['hold_last']['terminal_mae_per_joint'], [5]*6)
        per_point = np.asarray(result['lewm']['mae_by_horizon_readback_joint'])
        assert per_point.shape == (4, 5, 6)
        if stage == 'actor':
            np.testing.assert_array_equal(result['actor_commands']['mae_per_joint'], [7]*6)
            assert result['fcpg']['finite'] and result['fcpg']['forward_calls'] > 0
        else:
            assert 'actor_commands' not in result and result['fcpg']['evaluated'] is False
    assert len(core.completed) == 4*3
    assert all(images == (stage != 'lewm') for _, _, images in source.reads)


@pytest.mark.parametrize('fault', ['train_archive', 'incomplete_archive', 'test_fish', 'incomplete_fish', 'cross_episode'])
def test_non_dev_or_incomplete_populations_fail_before_source_payload_read(fault):
    source = DevSource(); rows = dev_rows()
    if fault == 'train_archive': rows['archive_dev_rows'] = np.array([0, 1, 3, 4])
    elif fault == 'incomplete_archive': rows['archive_dev_rows'] = np.array([1, 3])
    elif fault == 'test_fish': rows['fish_dev_rows'] = np.array([2, 3, 4])
    elif fault == 'incomplete_fish': rows['fish_dev_rows'] = np.array([3])
    else: source.fish.episodes[190] = 2
    with pytest.raises(ValueError):
        evaluate_development(HandComputedCore(), source, rows, stage='lewm')
    assert not source.reads


def test_selection_is_deterministic_across_input_order_and_does_not_use_global_rng():
    source = DevSource(); rows = dev_rows()
    first, _ = development_selection(source, rows, archive_limit=2)
    np.random.seed(91); np.random.random(13)
    rows['archive_dev_rows'] = rows['archive_dev_rows'][::-1]
    second, _ = development_selection(source, rows, archive_limit=2)
    np.testing.assert_array_equal(first['archive'], second['archive'])
    np.testing.assert_array_equal(first['fish'], [3, 4])


@pytest.mark.parametrize('failure', [False, True])
def test_evaluation_preserves_training_parameters_gradients_modes_cache_coverage_rng(failure):
    source, core = DevSource(), HandComputedCore()
    core.train(); core.lewm.eval()
    core.reader.nonfinite = failure
    for p in core.parameters():
        p.grad = torch.full_like(p, .25)
    weights = deepcopy(core.state_dict())
    grads = [(p.grad, p.grad.clone(), p.requires_grad) for p in core.parameters()]
    modes = [module.training for module in core.modules()]
    cache, coverage = core.reader._projection_cache, core.temporal_feature_coverage
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    if failure:
        with pytest.raises(ValueError, match='FCPG reader output'):
            evaluate_development(core, source, dev_rows(), stage='actor', archive_limit=2)
    else:
        evaluate_development(core, source, dev_rows(), stage='actor', archive_limit=2)
    for name, value in core.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    for p, (ref, values, enabled) in zip(core.parameters(), grads):
        assert p.grad is ref and p.requires_grad is enabled
        torch.testing.assert_close(p.grad, values, rtol=0, atol=0)
    assert [module.training for module in core.modules()] == modes
    assert core.reader._projection_cache is cache
    assert core.temporal_feature_coverage is coverage and coverage == dict(actor_calls=123)
    assert core._temporal_collect_training_coverage is True
    assert random.getstate() == python_rng
    restored = np.random.get_state()
    assert restored[0] == numpy_rng[0] and restored[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(restored[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_float32_timestamps_are_not_silently_accepted():
    source = DevSource()
    original = source.chain
    def invalid(rows, kind, *, images):
        batch = original(rows, kind, images=images)
        return replace(batch, endpoint_times=batch.endpoint_times.float())
    source.chain = invalid
    with pytest.raises(ValueError, match='float64'):
        evaluate_development(HandComputedCore(), source, dev_rows(), stage='lewm')


def cli_config(recipe):
    return dict(recipe=recipe,
        initialization='module-default-random-all-components-created-before-first-stage',
        external_weights_loaded=False, vocabulary=['move'], model={}, schedule={},
        q_loss_scale=[1.]*6, source_paths={}, body_table='unused', smoke=False)


@pytest.mark.parametrize('recipe_name', ['RECIPE', 'GPU_RECIPE'])
def test_cli_accepts_recorded_cpu_and_gpu_scratch_runs_on_cpu(tmp_path, monkeypatch, recipe_name):
    from scripts import evaluate_v21_scratch_development as evaluator
    from scripts import train_v21_from_scratch as trainer
    config = cli_config(getattr(trainer, recipe_name))
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'lewm').mkdir()
    (tmp_path/'lewm'/'final.pt').write_bytes(b'checkpoint double')
    np.savez(tmp_path/'training-rows.npz', development_archive=[1], development_fish=[3])
    calls = []

    class CPUCore:
        q_loss_scale = torch.ones(6)

        def cpu(self):
            calls.append('cpu'); return self

        def float(self):
            calls.append('fp32'); return self

    def source(paths, *, device, object_relative_table, motion_only):
        assert device == 'cpu' and motion_only
        calls.append('cpu_source')
        return SimpleNamespace(vocabulary=evaluator.TaskVocabulary(('move',)))

    monkeypatch.setattr(evaluator.torch, 'set_num_threads', lambda _: None)
    monkeypatch.setattr(evaluator, 'V21Config', lambda **kwargs: kwargs)
    monkeypatch.setattr(evaluator, 'V21ResearchCore', lambda _: CPUCore())
    monkeypatch.setattr(evaluator, 'V21Sources', source)
    monkeypatch.setattr(evaluator, 'load_checkpoint', lambda *a, **k:
        dict(smoke=False, step=17000, complete=True))
    monkeypatch.setattr(evaluator, 'evaluate_development', lambda *a, **k: dict(stage=k['stage']))
    assert evaluator.main(['--run', str(tmp_path), '--stage', 'lewm']) == 0
    assert calls == ['cpu', 'fp32', 'cpu_source']
    report = json.loads((tmp_path/'lewm'/'development.json').read_text())
    assert report['stage'] == 'lewm' and report['checkpoint']['step'] == 17000
    assert report['smoke'] is False


@pytest.mark.parametrize('overrides', [
    dict(recipe='arbitrary-finetuning'), dict(recipe='v21-original-train-from-scratch-cuda-bf16-v1'),
    dict(external_weights_loaded=True), dict(external_weights_loaded=None),
    dict(initialization='pretrained'), dict(initialization=None),
])
def test_cli_rejects_non_scratch_before_checkpoint_access(tmp_path, overrides):
    from scripts import evaluate_v21_scratch_development as evaluator
    config = cli_config('v21-original-train-from-scratch-cuda-fp32-v1')
    config.update(overrides)
    (tmp_path/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='only a recorded new scratch run'):
        evaluator.main(['--run', str(tmp_path), '--stage', 'lewm'])
