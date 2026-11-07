"""Compact development headline for a completed V20.3 ``evaluate`` run.

The formal JSON reports stay the source of truth. This module only folds
already-written sum/count rows into a one-page table. Missing pieces stay
``available=False`` with a reason; no number is invented. Development
evidence is not unseen-scene accuracy, anti-fog success, or a robot trial.
"""
from __future__ import annotations

import numpy as np

from .prod_v20_3_pipeline import MIN_CLUSTERS_FOR_CI, OCCLUSION_PROBE_REFERENCE, merge_sum_count_metrics

L1_CONDITIONS = (
    "mechanical_prior_known_geometry",
    "current_measurement_known_geometry",
    "h0_posterior_known_geometry",
    "h0_posterior_independent_endpoint_known_geometry",
)
L1_METRICS = ("tip_epe_px", "joint_pair_nll", "conditional_pair_hpd50_approx",
              "conditional_pair_hpd90_approx", "pair_span_absolute_error_px")
L3_INNOVATION = ("innovation_full", "innovation_belief_only", "innovation_q_only", "innovation_neither")
OCCLUSION_HORIZONS = ("h2", "h3", "h4")
OCCLUSION_ESTIMATORS = ("proprioceptive_memory", "independent_endpoint", "pixel_offset_carry",
                        "frozen_pixels", "nominal_prior", "memoryless_fusion")
VIEWS = ("wrist", "column")
CI_METRIC = "map_pair_distance_to_full_vision_px"


def clustered_bootstrap_ci(values, clusters, *, draws: int = 2000, seed: int = 20100, level: float = .95) -> dict:
    """Percentile bootstrap of the mean, resampling whole clusters (episodes).

    Rows inside one cluster stay together in every draw, so overlapping
    windows never masquerade as independent trials. Fewer than
    ``MIN_CLUSTERS_FOR_CI`` clusters returns ``available=False``.
    """
    pairs = [(float(v), c) for v, c in zip(values, clusters) if v is not None and np.isfinite(v)]
    if not pairs:
        return dict(available=False, reason="no finite values")
    sample = np.asarray([v for v, _ in pairs])
    labels = np.asarray([c for _, c in pairs])
    unique = np.unique(labels)
    if len(unique) < MIN_CLUSTERS_FOR_CI:
        return dict(available=False, reason=f"{len(unique)} clusters < {MIN_CLUSTERS_FOR_CI}; clustered CI not meaningful",
                    clusters=int(len(unique)), mean=float(sample.mean()), rows=int(len(sample)))
    groups = [sample[labels == c] for c in unique]
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        chosen = rng.integers(len(groups), size=len(groups))
        drawn = np.concatenate([groups[k] for k in chosen])
        means[i] = drawn.mean()
    alpha = (1 - level) / 2
    return dict(available=True, mean=float(sample.mean()), clusters=int(len(unique)), rows=int(len(sample)),
                low=float(np.quantile(means, alpha)), high=float(np.quantile(means, 1 - alpha)),
                level=level, draws=draws, seed=seed, resampling="whole clusters (recording episodes)")


def _occlusion_cis(occlusion: dict) -> dict:
    identity = occlusion.get("row_identity") or {}
    per_row = occlusion.get("per_row") or {}
    if not identity.get("available") or not per_row:
        return dict(available=False, reason="no per-row values or no episode identity")
    episodes = identity["episode"]
    result = {}
    for estimator in OCCLUSION_ESTIMATORS:
        for horizon in OCCLUSION_HORIZONS:
            for view in VIEWS:
                key = f"occlusion/both_h1_h3/{estimator}/{horizon}/{view}/{CI_METRIC}"
                if key in per_row and len(per_row[key]) == len(episodes):
                    result.setdefault(estimator, {}).setdefault(horizon, {})[view] = clustered_bootstrap_ci(per_row[key], episodes)
    feasible = bool(identity.get("clustered_ci_feasible"))
    return dict(available=feasible, clusters=identity.get("clusters"),
                reason=None if feasible else f"{identity.get('clusters')} recording episodes; need {MIN_CLUSTERS_FOR_CI} for a clustered interval",
                metric=CI_METRIC, schedule="both_h1_h3", intervals=result)


def _cell(value):
    if not isinstance(value, dict) or "count" not in value:
        return None
    return dict(sum=value.get("sum"), count=value["count"], mean=value.get("mean"))


def merge_labeled_training_fit(batches: dict) -> dict:
    """Sum/count merge of ``labeled_training_fit`` batches, per condition/view/metric."""
    merged, rows = {}, []
    for batch in batches.values():
        rows.extend(batch.get("rows") or [])
        for condition, views in (batch.get("conditions") or {}).items():
            if condition not in L1_CONDITIONS:
                continue
            for view in VIEWS:
                source = views.get(view) or {}
                target = merged.setdefault(condition, {}).setdefault(view, {})
                for metric in L1_METRICS:
                    if metric in source:
                        target.setdefault(metric, []).append(source[metric])
    result = {}
    for condition, views in merged.items():
        result[condition] = {}
        for view, metrics in views.items():
            parts = [{name: values[index] for name, values in metrics.items() if index < len(values)}
                     for index in range(max((len(values) for values in metrics.values()), default=0))]
            result[condition][view] = merge_sum_count_metrics(parts)
    return dict(conditions=result, rows=rows, row_count=len(rows))


def _delta(left, right):
    if left is None or right is None or left.get("mean") is None or right.get("mean") is None:
        return None
    return dict(left_minus_right=left["mean"] - right["mean"],
                left_count=left["count"], right_count=right["count"])


def _actor_innovation(metrics: dict) -> dict:
    result = {}
    for name in L3_INNOVATION:
        key = f"actor/{name}/normalized_mae"
        if key in metrics:
            result[name] = _cell(metrics[key])
    return result


def _occlusion_slice(summary: dict, schedule="both_h1_h3") -> dict:
    block = summary.get(schedule) or {}
    sliced = {}
    for estimator in OCCLUSION_ESTIMATORS:
        if estimator not in block:
            continue
        sliced[estimator] = {horizon: {view: (block[estimator].get(horizon) or {}).get(view)
                                       for view in VIEWS}
                             for horizon in OCCLUSION_HORIZONS}
    return sliced


def summarize_development_headline(report: dict) -> dict:
    """One-page table from an ``evaluate()`` development report. No new numbers."""
    labeled = merge_labeled_training_fit(report.get("labeled_training_fit") or {})
    posterior = labeled["conditions"].get("h0_posterior_known_geometry") or {}
    independent = labeled["conditions"].get("h0_posterior_independent_endpoint_known_geometry") or {}
    l1 = dict(available=bool(labeled["conditions"]),
              evidence="training-development", unseen_claim=False,
              physical_labels="already-exposed real labeled views; training-fit only",
              independent_endpoint="same trained measurement; cross-tip covariance removed at fusion",
              conditions=labeled["conditions"], row_count=labeled["row_count"])
    if not l1["available"]:
        l1["reason"] = "labeled_training_fit is empty"
    l1_delta = {view: {metric: _delta((posterior.get(view) or {}).get(metric),
                                      (independent.get(view) or {}).get(metric))
                       for metric in ("tip_epe_px", "joint_pair_nll")}
                for view in VIEWS}
    occlusion = report.get("occlusion_probe") or {}
    occlusion_headline = dict(available=bool(occlusion.get("available")),
                              evidence="training-development", unseen_claim=False,
                              reference=occlusion.get("reference", OCCLUSION_PROBE_REFERENCE),
                              geometry_kind=occlusion.get("geometry_kind"),
                              metric_geometry=occlusion.get("metric_geometry"),
                              chain_rows=occlusion.get("chain_rows"),
                              recording_episodes=(occlusion.get("row_identity") or {}).get("clusters"),
                              file=occlusion.get("file"),
                              both_h1_h3=_occlusion_slice(occlusion.get("summary") or {}),
                              clustered_ci=_occlusion_cis(occlusion))
    if not occlusion_headline["available"]:
        occlusion_headline["reason"] = occlusion.get("reason", "occlusion probe was not run")
    metrics = report.get("metrics") or {}
    innovation = _actor_innovation(metrics)
    l3 = dict(available=bool(innovation), evidence="training-development", unseen_claim=False,
              inference_masks=innovation,
              matched_no_innovation=dict(available=False,
                  reason="separate same-capacity job: actor --actor-mode no_innovation --frozen-from <full run>"),
              withdrawal_rule="if the visual residual adds nothing beyond the q path and history, "
                              "the visual-feedback control claim is withdrawn or narrowed")
    if not l3["available"]:
        l3["reason"] = "evaluate() actor intervention metrics are missing"
    return dict(evidence="training-development", unseen_claim=False,
                held_episode_opened=bool(report.get("held_episode_opened", False)),
                robot_commands_sent=bool(report.get("robot_commands_sent", False)),
                l1_representation=l1,
                l1_independent_endpoint_delta=l1_delta,
                occlusion=occlusion_headline,
                l3_feedback=l3,
                l4_fresh_real=dict(available=False,
                    reason="independently supervised real trials remain separate work"))


__all__ = ["summarize_development_headline", "merge_labeled_training_fit"]
