#!/usr/bin/env python3
"""Recorded DEV evaluation for a new V21 scratch run; no hardware or optimization."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpg.v21_core import POSITION_UNIT, V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary, V21Sources
from fcpg.v21_memory import VisualObservation
from fcpg.v21_training import V21Schedule, load_checkpoint, write_json

STAGES = ('lewm', 'actor', 'dense', 'fish')
SCRATCH_RECIPES = ('v21-original-train-from-scratch-cpu-v1',
    'v21-original-train-from-scratch-cuda-fp32-v1')
SELECTION_SEED = 20000
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _rows(mapping, kind, count):
    aliases = (kind, kind+'_dev_rows', 'development_'+kind)
    found = [np.asarray(mapping[name]) for name in aliases if name in mapping]
    require(bool(found), 'missing full development row population: '+kind)
    rows = found[0]
    require(rows.ndim == 1 and rows.dtype.kind in 'iu' and len(rows) > 0
        and len(np.unique(rows)) == len(rows) and ((rows >= 0)&(rows < count)).all(),
        'development rows must be unique in-range integer identities: '+kind)
    require(all(np.array_equal(rows, other) for other in found[1:]),
        'conflicting development row aliases: '+kind)
    return np.sort(rows.astype(np.int64, copy=False))


def development_selection(source, development_rows, archive_limit=64):
    """Admit all DEV identities before reading any q/actions/visual features."""
    require(type(archive_limit) is int and archive_limit > 0, 'positive archive_limit required')
    archive = _rows(development_rows, 'archive', source.counts['archive'])
    split = np.asarray(source.archive.archive.split)
    require(split.shape == (source.counts['archive'],) and np.isin(split, [0, 1]).all(),
        'original archive TRAIN/DEV split required')
    require(np.array_equal(archive, np.flatnonzero(split == 1)),
        'archive development population must exactly match original DEV, with no TRAIN rows')
    fish = _rows(development_rows, 'fish', source.counts['fish'])
    starts = np.asarray(source.fish.starts)
    episodes, frames = source.fish.episodes, source.fish.frame_indices
    require(starts.shape == (source.counts['fish'],) and starts.dtype.kind in 'iu'
        and ((starts >= 0)&(starts+60 < len(episodes))).all(), 'invalid fish chain identities')
    expected = np.flatnonzero(np.asarray(episodes[starts]) == 3)
    require(np.array_equal(fish, expected),
        'fish development must contain every episode 3 chain and no episode 0/1/2')
    for row in fish:
        ids = starts[row]+np.arange(61)
        require((np.asarray(episodes[ids]) == 3).all()
            and np.array_equal(frames[ids], frames[ids[0]]+np.arange(61)),
            'fish development chain crosses an episode or missing actual frame')
    selected = np.sort(np.random.default_rng(SELECTION_SEED).choice(archive,
        size=min(archive_limit, len(archive)), replace=False))
    return dict(archive=selected, fish=fish), dict(archive=len(archive), fish=len(fish))


@contextmanager
def _evaluation_state(core):
    """Restore modes, gradients, RNG and training diagnostics even after failure."""
    modes = [(module, module.training) for module in core.modules()]
    parameters = list(core.parameters())
    weights = {name: value.detach().clone() for name, value in core.state_dict().items()}
    gradients = [(p, p.grad, None if p.grad is None else p.grad.detach().clone(), p.requires_grad)
                 for p in parameters]
    cache = getattr(core.reader, '_projection_cache', None)
    sentinel = object()
    attrs = {name: getattr(core, name, sentinel) for name in
             ('_temporal_collect_training_coverage', 'temporal_feature_coverage')}
    coverage = deepcopy(attrs['temporal_feature_coverage']) if attrs['temporal_feature_coverage'] is not sentinel else None
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in parameters if p.device.type == 'cuda'})
    fastpath = torch.backends.mha.get_fastpath_enabled()
    with torch.random.fork_rng(devices=devices):
        try:
            core.eval()
            core._temporal_collect_training_coverage = False
            torch.backends.mha.set_fastpath_enabled(False)
            with torch.inference_mode():
                yield
        finally:
            changed = []
            with torch.no_grad():
                for name, value in core.state_dict().items():
                    if not torch.equal(value, weights[name]):
                        changed.append(name)
                        value.copy_(weights[name])
                for p, grad, original, requires_grad in gradients:
                    if p.grad is not grad or (grad is not None and not torch.equal(grad, original)):
                        changed.append('gradient')
                        if grad is not None:
                            grad.copy_(original)
                        p.grad = grad
                    p.requires_grad_(requires_grad)
            for module, training in modes:
                module.training = training
            core.reader._projection_cache = cache
            for name, value in attrs.items():
                if value is sentinel:
                    if hasattr(core, name):
                        delattr(core, name)
                else:
                    if name == 'temporal_feature_coverage':
                        value.clear(); value.update(coverage)
                    setattr(core, name, value)
            torch.backends.mha.set_fastpath_enabled(fastpath)
            random.setstate(python_rng); np.random.set_state(numpy_rng)
            require(not changed, 'evaluation tried to modify model tensors/gradients; original values restored: '+str(changed[:3]))


def _finite(value, name, shape=None):
    require(isinstance(value, torch.Tensor) and (shape is None or tuple(value.shape) == shape)
        and bool(torch.isfinite(value).all()), 'nonfinite or incorrectly shaped '+name)


def _validate_chain(batch, images):
    b = len(batch.q)
    for name, shape in dict(q=(b, 5, 6), commands=(b, 4, 5, 6), paths=(b, 4, 6, 6),
                            endpoint_times=(b, 5), readback_offsets=(b, 4, 6)).items():
        _finite(getattr(batch, name), name, shape)
    require(batch.endpoint_times.dtype == batch.readback_offsets.dtype == torch.float64,
        'recorded endpoint and readback clocks must remain float64')
    require(bool((batch.endpoint_times.diff(dim=1) > 0).all())
        and bool((batch.readback_offsets.diff(dim=2) > 0).all())
        and bool((batch.readback_offsets[:, :, 0] == 0).all()), 'noncausal recorded clocks')
    require(torch.allclose(batch.endpoint_times[:, 1:]-batch.endpoint_times[:, :-1],
        batch.readback_offsets[:, :, -1], atol=1e-8, rtol=0), 'actual endpoint/readback clocks disagree')
    require(torch.equal(batch.paths[:, :, 0], batch.q[:, :-1])
        and torch.equal(batch.paths[:, :, -1], batch.q[:, 1:]), 'actual path endpoints disagree')
    require(len(batch.traces) == 4, 'four complete recorded command traces required')
    for trace in batch.traces:
        trace.validate(b)
        require(trace.source == 'recorded' and trace.complete and trace.offsets.dtype == torch.float64,
            'evaluation requires full recorded commands and float64 offsets')
    if images:
        _finite(batch.features, 'real DINO features')
        if batch.object_relative is not None:
            _finite(batch.object_relative, 'current wrist body cue', (b, 5, 8))


def _metrics(sums, chains):
    # sums retains H0..H3 x q1..q5 x six joints; q0 never dilutes the errors.
    mean = sums/chains
    return dict(mae_per_joint=mean.mean((0, 1)).tolist(),
        terminal_mae_per_joint=mean[:, -1].mean(0).tolist(),
        mae_by_horizon_readback_joint=mean.tolist(),
        terminal_mae_by_horizon_joint=mean[:, -1].tolist())


def evaluate_development(core, source, development_rows, *, stage, microbatch=8, archive_limit=64):
    """Teacher-forced original archive DEV and all fish episode 3 chains.

    ``development_rows`` accepts ``archive``/``fish`` (derive_splits output),
    ``archive_dev_rows``/``fish_dev_rows``, or saved ``development_*`` arrays.
    The source keeps original archive global IDs and original fish ordinals.
    """
    require(stage in STAGES, 'unknown scratch stage')
    require(type(microbatch) is int and microbatch > 0, 'positive microbatch required')
    selected, populations = development_selection(source, development_rows, archive_limit)
    images = stage != 'lewm'
    groups = {}
    with _evaluation_state(core):
        for kind, rows in selected.items():
            prediction_sum = np.zeros((4, 5, 6), np.float64)
            hold_sum = np.zeros_like(prediction_sum)
            command_sum = np.zeros_like(prediction_sum)
            chains = 0
            fcpg = dict(evaluated=images, finite=True if images else None, forward_calls=0, output_shapes=[])
            def check_fcpg(_module, _inputs, output):
                _finite(output, 'FCPG reader output')
                fcpg['forward_calls'] += 1
                shape = list(output.shape[1:])
                if shape not in fcpg['output_shapes']:
                    fcpg['output_shapes'].append(shape)
            hook = core.reader.register_forward_hook(check_fcpg) if images else None
            try:
                for begin in range(0, len(rows), microbatch):
                    batch = source.chain(rows[begin:begin+microbatch], kind, images=images)
                    _validate_chain(batch, images)
                    def observation(h):
                        if not images:
                            return None
                        return VisualObservation(batch.features[:, h].float(),
                            batch.q[:, h, None].expand(-1, 2, -1),
                            batch.endpoint_times[:, h, None].expand(-1, 2), batch.view_valid[:, h],
                            object_relative=None if batch.object_relative is None else batch.object_relative[:, h])
                    state = core.observe(None, observation(0), batch.q[:, 0], batch.endpoint_times[:, 0])
                    for h in range(4):
                        if images:
                            action = core.act(state, batch.text_ids, batch.readback_offsets[:, h, :5]).commands_position
                            _finite(action, 'Actor commands', (len(batch.q), 5, 6))
                            command_sum[h] += (action-batch.commands[:, h]).abs().double().sum(0).cpu().numpy()
                        forecast = core.predict(state, batch.traces[h], batch.readback_offsets[:, h, 1:])
                        prediction = forecast.response.predicted_q_path
                        _finite(prediction, 'LeWM path', (len(batch.q), 6, 6))
                        target = batch.paths[:, h, 1:]
                        prediction_sum[h] += (prediction[:, 1:]-target).abs().double().sum(0).cpu().numpy()
                        hold_sum[h] += (state.q[:, None]-target).abs().double().sum(0).cpu().numpy()
                        if h < 3:
                            state = core.complete(state, forecast, batch.traces[h], batch.paths[:, h],
                                state.time[:, None]+batch.readback_offsets[:, h], observation(h+1))
                    chains += len(batch.q)
            finally:
                if hook is not None:
                    hook.remove()
            report = dict(rows=rows.tolist(), chains=chains, transitions=chains*4,
                predicted_readbacks=chains*4*5, lewm=_metrics(prediction_sum, chains),
                hold_last=_metrics(hold_sum, chains), fcpg=fcpg)
            if images:
                report['actor_commands'] = _metrics(command_sum, chains)
            groups[kind] = report
    return dict(schema='v21-scratch-development-evaluation-v1', stage=stage,
        evidence='teacher-forced-recorded-development', teacher_forced=True, development=True,
        unseen_claim=False, physical_success_verified=False, hardware_used=False,
        checkpoint_selection=False, optimizer_steps=0, episode_2_values_read=False,
        position_unit=POSITION_UNIT, joint_names=list(JOINTS),
        archive_selection=dict(seed=SELECTION_SEED, population=populations['archive'],
            selected=len(selected['archive']), sampling='fixed uniform without replacement; sorted original DEV identities'),
        fish_selection=dict(episode=3, population=populations['fish'], full_population=True),
        history='complete recorded command trace and actual q/readback clocks; predictions never replace factual history',
        horizon_order=['H0', 'H1', 'H2', 'H3'], readback_order=['q1', 'q2', 'q3', 'q4', 'q5'],
        weights_gradients_rng_preserved=True, results=groups)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--checkpoint', type=Path, help='optional checkpoint inside this scratch run; default stage/final.pt')
    parser.add_argument('--output', type=Path, help='default stage/development.json')
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--archive-limit', type=int, default=64)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    require(1 <= args.threads <= 32, 'CPU threads must be between 1 and 32')
    root = args.run.resolve()
    config = json.loads((root/'config.json').read_text())
    require(config.get('recipe') in SCRATCH_RECIPES
        and config.get('initialization') == 'module-default-random-all-components-created-before-first-stage'
        and config.get('external_weights_loaded') is False, 'only a recorded new scratch run may be evaluated')
    checkpoint = (root/args.stage/'final.pt' if args.checkpoint is None else args.checkpoint).resolve()
    require(checkpoint.is_relative_to(root) and checkpoint.is_file(), 'checkpoint must exist inside this scratch run')
    torch.set_num_threads(args.threads)
    vocabulary = TaskVocabulary(tuple(config['vocabulary']))
    core = V21ResearchCore(V21Config(**config['model'])).cpu().float()
    payload = load_checkpoint(checkpoint, core, vocabulary, V21Schedule(**config['schedule']),
        stage=args.stage, allow_smoke=bool(config.get('smoke')))
    require(payload.get('smoke') is bool(config.get('smoke')), 'checkpoint and run canary status disagree')
    require(torch.equal(core.q_loss_scale, torch.tensor(config['q_loss_scale'], dtype=torch.float32)),
        'checkpoint TRAIN normalization differs from the scratch run')
    source = V21Sources(config['source_paths'], device='cpu',
        object_relative_table=config['body_table'], motion_only=args.stage == 'lewm')
    source.compact_host_features = False
    require(source.vocabulary == vocabulary, 'development source vocabulary differs from the scratch run')
    with np.load(root/'training-rows.npz', allow_pickle=False) as arrays:
        development_rows = {key: arrays[key] for key in ('development_archive', 'development_fish')}
    if args.stage != 'lewm':
        from scripts.build_v21_body_object_table import validate_manifest
        fish_rows = development_rows['development_fish']
        required = np.unique(source.fish.starts[fish_rows, None]+np.arange(5)[None]*15)
        validate_manifest(config['body_table'], config['body_manifest'], source.fish,
            config['source_paths']['fish_wrist_root'], required,
            source_wrist_root=config.get('source_paths_original', config['source_paths'])['fish_wrist_root'])
    report = evaluate_development(core, source, development_rows, stage=args.stage,
        microbatch=args.microbatch, archive_limit=args.archive_limit)
    report.update(checkpoint=dict(path=str(checkpoint), step=payload['step'], complete=payload['complete']),
        smoke=bool(payload['smoke']), two_step_real_data_canary=bool(payload['smoke']))
    output = args.output or root/args.stage/'development.json'
    write_json(output, report)
    print(json.dumps(dict(complete=True, stage=args.stage, smoke=report['smoke'], output=str(output))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
