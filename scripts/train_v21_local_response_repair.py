#!/usr/bin/env python3
"""Refine native LeWM on factual local TRAIN responses and original archive TRAIN.

The live service, parent checkpoint, Actor, FCPG and visual caches are untouched.
Validation records and the first scratch live block are never opened here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpg.v21_data import V21Sources
from fcpg.v21_local_response_data import LocalResponseDataset
from fcpg.v21_runtime import load_runtime
from fcpg.v21_training import write_json
from scripts.train_v21_from_scratch import configure_device, stage_chain_to_device
from scripts.v21_source_map import resolve_motion_source_paths


def response_loss(response, y, scale):
    prediction = response.predicted_q_path[:, 1:]
    target = y.detach()
    regression = F.smooth_l1_loss(prediction / scale, target / scale)
    error = (prediction - target) / scale
    sigma = (response.q_uncertainty / scale).clamp_min(1e-3)
    uncertainty = (.5 * (error.detach() / sigma).square() + sigma.log()).mean()
    return regression + .05 * uncertainty, regression, uncertainty


def archive_loss(core, chain):
    state = core.observe(None, None, chain.q[:, 0], chain.endpoint_times[:, 0])
    totals, regressions, uncertainties = [], [], []
    for h in range(4):
        forecast = core.predict(state, chain.traces[h], chain.readback_offsets[:, h, 1:])
        loss, regression, uncertainty = response_loss(
            forecast.response, chain.paths[:, h, 1:], core.q_loss_scale)
        totals.append(loss)
        regressions.append(regression)
        uncertainties.append(uncertainty)
        if h < 3:
            state = core.complete(state, forecast, chain.traces[h], chain.paths[:, h],
                state.time[:, None] + chain.readback_offsets[:, h], None)
    return tuple(torch.stack(values).mean() for values in (totals, regressions, uncertainties))


def no_hardware(*args, **kwargs):
    raise RuntimeError('Offline response repair cannot access hardware')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--train-manifest', type=Path, required=True)
    parser.add_argument('--source-map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--archive-chains', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--gpu-memory-limit-gib', type=float, default=6.)
    parser.add_argument('--canary', action='store_true')
    args = parser.parse_args(argv)
    if not (1 <= args.steps <= 5000 and 1 <= args.batch_size <= 128
            and 1 <= args.archive_chains <= 64 and 1 <= args.threads <= 8
            and 0 < args.learning_rate <= 1e-4):
        raise ValueError('Invalid bounded response training configuration')
    if args.canary and args.steps != 2:
        raise ValueError('A canary has exactly two optimizer steps')
    if args.output.exists():
        raise FileExistsError('Use a new output directory; do not overwrite a run')
    torch.set_num_threads(args.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device, device_info = configure_device(args.device, args.gpu_memory_limit_gib)
    rng = np.random.default_rng(args.seed)
    config = json.loads((args.base_run / 'config.json').read_text())
    source_paths, source_info = resolve_motion_source_paths(
        config.get('source_paths_original', config['source_paths']), args.source_map)
    source = V21Sources(source_paths, device='cpu', motion_only=True)
    split = np.asarray(source.archive.archive.split)
    if split.shape != (source.counts['archive'],) or not np.isin(split, [0, 1]).all():
        raise ValueError('Original TRAIN/DEV split is unavailable')
    archive_train = np.flatnonzero(split == 0)
    if len(archive_train) != 272000:
        raise ValueError('Original archive TRAIN population changed')
    local = LocalResponseDataset(args.train_manifest, split='train')
    if local.split != 'train' or not len(local):
        raise ValueError('Only eligible factual local TRAIN records can be optimized')
    run_ids = np.asarray(local.run_ids)
    run_names = sorted(set(run_ids.tolist()))
    groups = [np.flatnonzero(run_ids == name) for name in run_names]
    runtime = load_runtime(args.base_run, 'Pick up the red fish toy and place it in the cardboard box',
        device=device, physical_check=no_hardware, lease_check=no_hardware)
    core = runtime.core
    parent = torch.load(args.base_run / 'fish/final.pt', map_location='cpu', weights_only=False)
    core.requires_grad_(False).eval()
    core.lewm.requires_grad_(True).train()
    original = {name: value.detach().cpu().clone() for name, value in core.state_dict().items()}
    if set(original) != set(parent['model']) or any(
            not torch.equal(original[name], value) for name, value in parent['model'].items()):
        raise ValueError('Strict native checkpoint restoration failed')
    optimizer = torch.optim.AdamW(core.lewm.parameters(), lr=args.learning_rate,
        weight_decay=.01, foreach=False)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema='v21-local-response-refinement-v1',
        base_checkpoint=str(args.base_run / 'fish/final.pt'), train_manifest=str(args.train_manifest),
        source_map=str(args.source_map), source_mapping=source_info,
        source_paths=source_paths, original_archive_train_rows=len(archive_train),
        local_records=len(local), local_record_ids=list(local.record_ids),
        local_runs=dict(zip(run_names, map(len, groups))),
        sampling='equal probability per local source run; uniform original archive TRAIN',
        steps=args.steps, seed=args.seed, learning_rate=args.learning_rate,
        local_batch_size=args.batch_size, archive_chains=args.archive_chains,
        source_loss_weights=dict(local=.5, archive=.5),
        regression='scaled SmoothL1 beta=1', uncertainty='Gaussian NLL with detached regression residual',
        uncertainty_loss_weight=.05, gradient_clip=1., weight_decay=.01,
        sigreg_weight=0., recipe='supervised LeWM response refinement; no new representation regularizer',
        trainable_modules=['lewm'], frozen_modules=['reader', 'actor'],
        original_command_traces_preserved=True, command_resampling=False,
        predictions_used_as_labels=False, validation_used_for_optimization=False,
        checkpoint_selection='fixed final optimizer step', canary=args.canary,
        hardware_used=False, precision='fp32', device=str(device), device_info=device_info)
    write_json(args.output / 'protocol.json', protocol)
    write_json(args.output / 'run-status.json', dict(state='running', step=0, total_steps=args.steps))
    start = time.monotonic()
    draws = np.zeros(len(local), np.int64)
    archive_draws = []
    try:
        with (args.output / 'training.jsonl').open('x') as log:
            for step in range(1, args.steps + 1):
                chosen_groups = rng.integers(len(groups), size=args.batch_size)
                indices = np.array([rng.choice(groups[g]) for g in chosen_groups], np.int64)
                np.add.at(draws, indices, 1)
                rows = archive_train[rng.integers(len(archive_train), size=args.archive_chains)]
                if not (split[rows] == 0).all():
                    raise ValueError('Non-TRAIN archive row reached optimization')
                archive_draws.append(rows.tolist())
                batch = local.batch(indices.tolist(), device=device)
                chain = stage_chain_to_device(source.chain(rows, 'archive', images=False), device)
                optimizer.zero_grad(set_to_none=True)
                prediction = core.lewm(batch.q_position, batch.trace, batch.history, batch.query_offsets)
                local_values = response_loss(prediction, batch.y, core.q_loss_scale)
                archive_values = archive_loss(core, chain)
                loss = .5 * local_values[0] + .5 * archive_values[0]
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite response loss')
                loss.backward()
                if any(p.grad is not None for name, p in core.named_parameters() if not name.startswith('lewm.')):
                    raise ValueError('Frozen Actor/FCPG received a gradient')
                gradient = torch.nn.utils.clip_grad_norm_(core.lewm.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                record = dict(step=step, seconds=time.monotonic()-start, loss=float(loss.detach()),
                    local_regression=float(local_values[1].detach()),
                    archive_regression=float(archive_values[1].detach()),
                    local_uncertainty=float(local_values[2].detach()),
                    archive_uncertainty=float(archive_values[2].detach()), gradient_norm=float(gradient))
                log.write(json.dumps(record)+'\n')
                if step == 1 or step % 25 == 0 or step == args.steps:
                    log.flush()
                    write_json(args.output / 'run-status.json', dict(state='running', total_steps=args.steps, **record))
                    print(json.dumps(record), flush=True)
        state = {name: value.detach().cpu().clone() for name, value in core.state_dict().items()}
        changed = [name for name in state if not torch.equal(state[name], original[name])]
        if not changed or any(not name.startswith('lewm.') for name in changed):
            raise ValueError('The refinement must change only LeWM')
        if any(not torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
            raise ValueError('Nonfinite trained checkpoint')
        report = dict(protocol, complete=True, elapsed_s=time.monotonic()-start,
            changed_state_entries=changed, unchanged_non_lewm_entries=sum(not n.startswith('lewm.') for n in state),
            local_draw_counts=draws.tolist(), archive_train_draws=archive_draws,
            optimizer_steps=args.steps, final_training_metrics=record,
            max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            physical_success_verified=False)
        candidate = deepcopy(parent)
        candidate['model'] = state
        candidate['smoke'] = args.canary
        candidate['optimizer'] = None
        candidate['response_refinement'] = {k:v for k,v in report.items() if k not in ('archive_train_draws','local_draw_counts')}
        candidate['response_refinement_optimizer'] = optimizer.state_dict()
        candidate['metrics'] = dict(candidate.get('metrics') or {}, response_refinement=record)
        (args.output / 'fish').mkdir()
        temporary = args.output / 'fish/final.pt.tmp'
        torch.save(candidate, temporary)
        temporary.replace(args.output / 'fish/final.pt')
        config['smoke'] = args.canary
        config['response_refinement'] = candidate['response_refinement']
        write_json(args.output / 'config.json', config)
        metrics = json.loads((args.base_run / 'metrics.json').read_text())
        metrics['response_refinement'] = report
        write_json(args.output / 'metrics.json', metrics)
        write_json(args.output / 'run-status.json', dict(state='completed', step=args.steps,
            elapsed_s=report['elapsed_s'], checkpoint=str(args.output / 'fish/final.pt'),
            frozen_non_lewm_unchanged=True, changed_state_entries=len(changed)))
        print(json.dumps(dict(completed=True, checkpoint=str(args.output / 'fish/final.pt'),
            elapsed_s=report['elapsed_s'], changed_state_entries=len(changed))), flush=True)
    except BaseException as error:
        write_json(args.output / 'run-status.json', dict(state='failed', error=type(error).__name__+': '+str(error)))
        raise


if __name__ == '__main__':
    main()
