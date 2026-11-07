#!/usr/bin/env python3
"""Matched-window BN/LN/parent audit, with no robot I/O or training updates.

Each scene model is compared to persistence in ITS OWN encoder space, using
identical camera/time masks. Episode 2 is never admitted. A failed report is
not a deployment approval, even when one aggregate metric improves.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpg.v21_data import V21Sources
from fcpg.v21_horizon_precision import compute_context
from fcpg.v21_horizon_training import factual_state, horizon_targets
from fcpg.v21_native_horizon import core_from_parent
from fcpg.v21_scene_targets import FutureSceneTargets
from fcpg.v21_scratch_features import ScratchDinoFeatureBank
from fcpg.v21_training import write_json
from scripts.train_v21_from_scratch import configure_device, derive_splits, stage_chain_to_device


def acceptance(models):
    """No cross-encoder raw MSE comparisons; both domains/views must pass."""
    results = {}
    for name in ('bn', 'ln'):
        checks = {}
        for domain in ('archive', 'fish'):
            current, parent = models[name][domain], models['parent'][domain]
            for camera in ('wrist', 'column'):
                ratio = current['views'][camera]['relative_scene_error']
                checks[f'{domain}.{camera}.beats_persistence'] = ratio is not None and ratio < 1.
            for metric in ('first5_mae', 'future_mae'):
                checks[f'{domain}.{metric}.within_parent_5pct'] = (
                    current[metric] <= 1.05 * parent[metric])
        results[name] = dict(passed=all(checks.values()), checks=checks)
    return results


def view_metrics(prediction_sum, persistence_sum, count):
    return {view: dict(targets=int(count[c]),
        scene_mse=float(prediction_sum[c]/count[c]) if count[c] else None,
        persistence_mse=float(persistence_sum[c]/count[c]) if count[c] else None,
        relative_scene_error=float(prediction_sum[c]/persistence_sum[c]) if persistence_sum[c] > 0 else None)
        for c, view in enumerate(('wrist', 'column'))}


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bn', 'ln', 'parent', 'source-config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda:0'))
    p.add_argument('--precision', default='bf16', choices=('fp32', 'bf16'))
    p.add_argument('--gpu-memory-limit-gib', type=float, default=6.)
    p.add_argument('--archive-cases', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--context', type=int, choices=range(4), default=2)
    p.add_argument('--locked-development-report', type=Path,
                   help='Preserve exact original window identities, including changed camera coverage')
    args = p.parse_args(argv)
    if args.output.exists() or not 1 <= args.archive_cases <= 4096 or not 1 <= args.batch_size <= 16:
        raise ValueError('new output directory and bounded audit sizes required')
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device, device_info = configure_device(args.device, args.gpu_memory_limit_gib)
    compute_context(device, args.precision)
    config = json.loads(args.source_config.read_text())
    source = V21Sources(config['source_paths'], device='cpu', object_relative_table=config['body_table'])
    bank = ScratchDinoFeatureBank(source, device, 256 * 1024**2)
    targets = FutureSceneTargets(source, bank)
    splits, rows_by_domain = derive_splits(source)['development'], {}
    if args.locked_development_report is not None:
        from fcpg.v21_clock_evaluation import locked_development_rows, evaluate_core
        locked = json.loads(args.locked_development_report.read_text())
        if locked['context'] != args.context:
            raise ValueError('locked development context must not change')
        rows_by_domain = locked_development_rows(locked, splits)
        args.output.mkdir(parents=True)
        report = dict(schema='v21-scene-world-model-acceptance-v1', context=args.context,
            rows=locked['rows'], models={}, precision=args.precision, same_camera_time_masks=True,
            cache_generation=source.recorded_clock.metadata if source.recorded_clock else None,
            checkpoints={n: str(getattr(args, n)) for n in ('bn', 'ln', 'parent')},
            fish_train_episodes=[0, 1], fish_development_episode=3, sealed_episode_2_read=False,
            hardware_used=False, deployment_approved=False)
        started = time.monotonic()
        for name in ('parent', 'bn', 'ln'):
            payload = torch.load(getattr(args, name), map_location='cpu', weights_only=False)
            core = core_from_parent(payload).to(device).eval()
            del payload
            report['models'][name] = evaluate_core(core, source, bank, rows_by_domain,
                device=device, precision=args.precision, context=args.context, batch_size=args.batch_size)
            write_json(args.output/'report.json', report)
            print(json.dumps(dict(model=name, metrics=report['models'][name]), allow_nan=False), flush=True)
            del core
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        report['acceptance'] = acceptance(report['models'])
        report['elapsed_s'] = time.monotonic()-started
        write_json(args.output/'report.json', report)
        return
    for domain in ('archive', 'fish'):
        rows = splits[domain]
        rows = rows[(targets.identities(rows, domain, args.context) >= 0).all(1)]
        if domain == 'archive':
            rows = rows[np.unique(np.linspace(0, len(rows)-1, min(len(rows), args.archive_cases), dtype=int))]
        if not len(rows):
            raise ValueError(f'no fully labelled native40 development windows: {domain}')
        rows_by_domain[domain] = rows
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = dict(schema='v21-scene-world-model-acceptance-v1', device=device_info,
        precision=args.precision, context=args.context, same_camera_time_masks=True,
        rows={k: v.tolist() for k, v in rows_by_domain.items()},
        fish_train_episodes=[0, 1], fish_development_episode=3, sealed_episode_2_read=False,
        overlapping_windows=True, hardware_used=False, deployment_approved=False,
        checkpoints={n: str(getattr(args, n)) for n in ('bn', 'ln', 'parent')}, models={})
    report['population'] = {}
    for domain, rows in rows_by_domain.items():
        episodes = horizon_targets(source, rows, domain, args.context).episode_ids
        sources = source.episode_sources[episodes] if domain == 'archive' else np.full(len(rows), source.fish_source)
        unique, counts = np.unique(sources, return_counts=True)
        report['population'][domain] = dict(windows=len(rows), unique_episodes=len(np.unique(episodes)),
            source_windows={str(s): int(n) for s, n in zip(unique, counts)},
            task_text_ids=len(np.unique(source.archive.task_index[rows])) if domain == 'archive' else 1)
    for name in ('parent', 'bn', 'ln'):
        payload = torch.load(getattr(args, name), map_location='cpu', weights_only=False)
        core = core_from_parent(payload).to(device).eval()
        del payload
        model_report = {}
        for domain, rows in rows_by_domain.items():
            q_sum = np.zeros(2, dtype=np.float64)
            q_n = np.zeros(2, dtype=np.int64)
            pred_sum, persist_sum, view_n = np.zeros(2), np.zeros(2), np.zeros(2, dtype=np.int64)
            query_sums = np.zeros((3, 8, 2), dtype=np.float64)
            source_sums = {}
            for begin in range(0, len(rows), args.batch_size):
                ids = rows[begin:begin + args.batch_size]
                host = source.chain(ids, domain, images=False)
                features, checked = bank.load(ids, domain, context_frames=args.context + 1)
                if not np.array_equal(checked, host.view_valid[:, :args.context + 1].numpy()):
                    raise ValueError('feature validity disagrees with factual history')
                chain = replace(stage_chain_to_device(host, device), features=features)
                target = horizon_targets(source, ids, domain, args.context).to(device)
                if not target.valid.all() or (domain == 'fish' and (target.episode_ids != 3).any()):
                    raise ValueError('expected full native40, non-sealed development labels')
                with compute_context(device, args.precision):
                    state = factual_state(core, chain, args.context, images=True)
                    response = core.lewm(state.q, target.trace, state.history, target.readback_offsets,
                        patches=state.memory.features, patch_valid=state.memory.valid,
                        memory=state.memory, now_time=state.time)
                    if name != 'parent':
                        future = targets.load(ids, domain, args.context, device)
                        encoded = core.lewm.scene_encoder(future.features, future.valid).float()
                error = (response.predicted_q_path[:, 1:].float() - target.readbacks.float()).abs()
                if not torch.isfinite(error).all():
                    raise ValueError('nonfinite joint prediction')
                for i, end in enumerate((5, 40)):
                    q_sum[i] += float(error[:, :end].sum())
                    q_n[i] += error[:, :end].numel()
                if name != 'parent':
                    mask = future.valid & response.observation_valid[:, :, -1][:, None]
                    prediction = response.predicted_scene[:, future.query_indices].float()
                    persistence = response.observation_scene[:, :, -1].float()[:, None]
                    mse = (prediction - encoded).square().mean(-1)
                    base = (persistence - encoded).square().mean(-1)
                    if not torch.isfinite(mse[mask]).all() or not torch.isfinite(base[mask]).all():
                        raise ValueError('nonfinite valid scene prediction')
                    for c in range(2):
                        selected = mask[..., c]
                        pred_sum[c] += float(mse[..., c][selected].sum())
                        persist_sum[c] += float(base[..., c][selected].sum())
                        view_n[c] += int(selected.sum())
                    observed, predicted_error, baseline_error = [value.cpu().numpy() for value in (mask, mse, base)]
                    query_sums += np.stack(((predicted_error*observed).sum(0),
                                           (baseline_error*observed).sum(0), observed.sum(0)))
                    sources = (source.episode_sources[target.episode_ids] if domain == 'archive'
                               else np.full(len(ids), source.fish_source))
                    for identity in np.unique(sources):
                        selected = observed & (sources == identity)[:, None, None]
                        source_sums.setdefault(str(identity), np.zeros((3, 2), dtype=np.float64))
                        source_sums[str(identity)] += np.stack(((predicted_error*selected).sum((0, 1)),
                            (baseline_error*selected).sum((0, 1)), selected.sum((0, 1))))
            result = dict(windows=len(rows), first5_mae=float(q_sum[0]/q_n[0]),
                          future_mae=float(q_sum[1]/q_n[1]))
            if name != 'parent':
                result['views'] = view_metrics(pred_sum, persist_sum, view_n)
                result['per_query'] = {str((i+1)*5): view_metrics(*query_sums[:, i]) for i in range(8)}
                result['per_source'] = {key: view_metrics(*value) for key, value in source_sums.items()}
            model_report[domain] = result
            print(json.dumps(dict(model=name, domain=domain, **result), allow_nan=False), flush=True)
        report['models'][name] = model_report
        write_json(args.output/'report.json', report)
        del core, state, response
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report['acceptance'] = acceptance(report['models'])
    report['elapsed_s'] = time.monotonic() - started
    write_json(args.output/'report.json', report)
    print(json.dumps(report['acceptance']), flush=True)


if __name__ == '__main__':
    main()
