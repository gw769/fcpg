"""Locked-window metrics for the time-axis repair, not a new training loss."""
from dataclasses import replace

import numpy as np
import torch

from .v21_horizon_training import horizon_targets, factual_state
from .v21_horizon_precision import compute_context
from .v21_scene_targets import FutureSceneTargets


def locked_development_rows(report, splits):
    result = {}
    for domain in ('archive', 'fish'):
        rows = np.asarray(report['rows'][domain], np.int64)
        if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows) or not np.isin(rows, splits[domain]).all():
            raise ValueError('locked development identities changed or crossed training split')
        result[domain] = rows
    return result


@torch.no_grad()
def evaluate_core(core, source, bank, rows_by_domain, *, device, precision='bf16', context=2, batch_size=4,
                  measurement_protocol='none', dropout_variant=0, scene_supervision='sparse8', fg_table=None):
    from scripts.train_v21_from_scratch import stage_chain_to_device
    from scripts.accept_v21_scene_world_model import view_metrics
    from scripts.train_so101_v21_research import measurement_dropout
    from .v21_scene_fg import fg_batch, pair_loss
    if measurement_protocol not in ('none', 'burst-cold-v1'):
        raise ValueError('unknown evaluation measurement protocol')
    targets = FutureSceneTargets(source, bank, supervision=scene_supervision)
    reports = {}
    has_scene = hasattr(core.lewm, 'scene_encoder')
    for domain, rows in rows_by_domain.items():
        q_sum, q_n = np.zeros(2), np.zeros(2, np.int64)
        pred_sum, base_sum, view_n = np.zeros(2), np.zeros(2), np.zeros(2, np.int64)
        observations, future_count = np.zeros(2, np.int64), np.zeros(2, np.int64)
        removed, retained, cold = np.zeros(2, np.int64), np.zeros(2, np.int64), np.zeros(2, np.int64)
        fg_sum, fg_n = np.zeros((2, 2)), np.zeros((2, 2), np.int64)
        for begin in range(0, len(rows), batch_size):
            ids = rows[begin:begin+batch_size]
            host = source.chain(ids, domain, images=False)
            features, checked = bank.load(ids, domain, context_frames=context+1)
            if not np.array_equal(checked, host.view_valid[:, :context+1].numpy()):
                raise ValueError('input masks differ from corrected extraction')
            chain = replace(stage_chain_to_device(host, device), features=features.to(device))
            if measurement_protocol != 'none':
                # Evaluation deliberately applies a declared stress condition;
                # unlike training augmentation it must work under no_grad.
                chain = measurement_dropout(chain, ids, protocol=measurement_protocol, variant=dropout_variant)
            target = horizon_targets(source, ids, domain, context).to(device)
            if domain == 'fish' and (target.episode_ids != 3).any():
                raise ValueError('fish evaluation is restricted to episode 3')
            with compute_context(device, precision):
                state = factual_state(core, chain, context, images=True)
                current_mask = chain.view_valid[:, context]
                memory_mask = state.memory.valid[:, :, -1]
                removed += (host.view_valid[:, :context+1].to(device) &
                            ~chain.view_valid[:, :context+1]).sum((0, 1)).cpu().numpy()
                retained += (~current_mask & memory_mask).sum(0).cpu().numpy()
                cold += (~memory_mask).sum(0).cpu().numpy()
                response = core.lewm(state.q, target.trace, state.history, target.readback_offsets,
                    patches=state.memory.features, patch_valid=state.memory.valid,
                    memory=state.memory, now_time=state.time)
                if has_scene:
                    future = targets.load(ids, domain, context, device)
                    encoded = core.lewm.scene_encoder(future.features, future.valid).float()
            if fg_table is not None and domain == 'fish':
                if not has_scene or not hasattr(core.lewm, 'fg_readout') or scene_supervision != 'dense40':
                    raise ValueError('spatial stress evaluation requires a dense40 F/G scene model')
                from .v21_scene_fg import FGBatch
                labels = fg_batch(fg_table, targets.identities(ids, domain, context),
                                  device=device, image_valid=future.valid)
                predicted = core.lewm.fg_readout(response.predicted_scene[:, future.query_indices].float())
                if not torch.isfinite(predicted[labels.valid]).all():
                    raise ValueError('nonfinite F/G prediction on actual future labels')
                for i, end in enumerate((5, 40)):
                    selected = FGBatch(labels.xy[:, :end], labels.valid[:, :end], labels.ordered[:, :end])
                    _, entries = pair_loss(predicted[:, :end], selected)
                    for c, entry in enumerate(entries):
                        fg_sum[i, c] += entry['pixel_error_sum']; fg_n[i, c] += entry['pairs']
            error = (response.predicted_q_path[:, 1:].float()-target.readbacks).abs()
            for i, end in enumerate((5, 40)):
                selected = error[:, :end][target.valid[:, :end]]
                if not torch.isfinite(selected).all():
                    raise ValueError('nonfinite joint evaluation')
                q_sum[i] += float(selected.sum()); q_n[i] += selected.numel()
            if has_scene:
                mask = future.valid & response.observation_valid[:, :, -1][:, None]
                mse = (response.predicted_scene[:, future.query_indices].float()-encoded).square().mean(-1)
                base = (response.observation_scene[:, :, -1].float()[:, None]-encoded).square().mean(-1)
                for c in range(2):
                    selected = mask[..., c]
                    if not torch.isfinite(mse[..., c][selected]).all() or not torch.isfinite(base[..., c][selected]).all():
                        raise ValueError('nonfinite scene evaluation')
                    pred_sum[c] += float(mse[..., c][selected].sum())
                    base_sum[c] += float(base[..., c][selected].sum())
                    view_n[c] += int(selected.sum())
                observations += chain.view_valid[:, :context+1].sum((0, 1)).cpu().numpy()
                future_count += future.valid.sum((0, 1)).cpu().numpy()
        if (q_n == 0).any():
            raise ValueError('locked windows have no valid readbacks; do not substitute easier windows')
        reports[domain] = dict(windows=len(rows), first5_mae=float(q_sum[0]/q_n[0]),
            future_mae=float(q_sum[1]/q_n[1]), valid_joint_values=q_n.tolist())
        if has_scene:
            reports[domain].update(views=view_metrics(pred_sum, base_sum, view_n),
                coverage=dict(input_view_slots=observations.tolist(), future_view_slots=future_count.tolist(),
                              same_mask_targets=view_n.tolist(), rows_removed=0))
        if measurement_protocol != 'none':
            reports[domain]['measurement_input'] = dict(protocol=measurement_protocol, variant=dropout_variant,
                removed_slots=removed.tolist(), retained_latest_views=retained.tolist(), cold_latest_views=cold.tolist())
        if fg_table is not None and domain == 'fish':
            reports[domain]['fg_forecast'] = {name: {view: dict(pairs=int(fg_n[i, c]),
                mean_pixel_error=float(fg_sum[i, c]/fg_n[i, c]) if fg_n[i, c] else None)
                for c, view in enumerate(('wrist', 'column'))} for i, name in enumerate(('first5', 'full40'))}
    return reports


def short_training_gate(candidate, baseline, parent):
    """The user-approved 10% / 5% gate, including unchanged dev coverage."""
    checks = {}
    for domain in ('archive', 'fish'):
        for camera in ('wrist', 'column'):
            current = candidate[domain]['views'][camera]
            before = baseline[domain]['views'][camera]
            ratio, reference = current['relative_scene_error'], before['relative_scene_error']
            factor = .9 if (domain, camera) == ('archive', 'wrist') else 1.05
            checks[f'{domain}.{camera}.relative_error'] = (ratio is not None and reference is not None
                and np.isfinite([ratio, reference]).all() and ratio <= factor*reference)
            checks[f'{domain}.{camera}.same_coverage'] = current['targets'] == before['targets'] and current['targets'] > 0
        for metric in ('first5_mae', 'future_mae'):
            checks[f'{domain}.{metric}.parent_tolerance'] = candidate[domain][metric] <= 1.05*parent[domain][metric]
    return dict(passed=bool(all(checks.values())), checks={k: bool(v) for k, v in checks.items()})
