#!/usr/bin/env python3
"""V21 research: preflight / explicit formal training / isolated CPU smoke. No robot I/O."""
import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fcpg.v21_core import V21Config, V21ResearchCore, VERSION, CONTRACT
from fcpg.v21_data import V21Sources, TaskVocabulary
from fcpg.v21_training import (V21Schedule, V21Trainer, synthetic_chain, load_checkpoint,
                              save_checkpoint, reuse_frozen_lewm, write_json, record_stage_metrics)
from fcpg.prod_v20_3_actor_sampling import deterministic_mixed_locality_windows
from fcpg.v21_temporal_contract import TRAINING_POLICIES, temporal_training_configuration


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("preflight", "lewm", "actor", "all", "smoke"))
    p.add_argument("--source-config", type=Path, help="existing run config.json, or source-path mapping")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-formal", action="store_true", help="explicit opt-in to a long GPU1 training run")
    p.add_argument("--memory-mode", choices=("readback", "temporal"), default="readback")
    p.add_argument("--microbatch", type=int, default=8, help="chain microbatch; effective batch is fixed")
    p.add_argument("--feature-bank-gib", type=float, default=36.)
    p.add_argument("--workspace-gib", type=float, default=24.)
    p.add_argument("--sigreg-weight", type=float, choices=(0., .02), default=.02)
    p.add_argument("--lewm-from", type=Path, help="reuse compatible completed ARM only; new Actor/reader train from scratch")
    p.add_argument("--measurement-dropout", choices=("legacy", "burst-cold-v1"), default="legacy",
                   help="opt-in missing-view schedules; immutable per output, never applied to LeWM")
    p.add_argument('--temporal-feature-policy', choices=TRAINING_POLICIES, default='legacy',
                   help='explicit Actor/FCPG sequence feature contract; keeps factual clocks and LeWM intervals')
    return p


def measurement_dropout(batch, rows, *, protocol="legacy", variant=0):
    """Remove measurements only; never relabel images, q, actions or clocks.

    burst-cold-v1 covers outages within the existing chain duration. It does
    not simulate a long outage by stretching timestamps or repeating images.
    Variant is an explicit deterministic request index, not shared worker RNG.
    """
    if protocol not in ("legacy", "burst-cold-v1"):
        raise ValueError("unknown measurement-dropout protocol")
    rows = np.asarray(rows)
    if (rows.shape != (len(batch.q),) or rows.dtype.kind not in "iu"
            or (rows < 0).any() or type(variant) is not int or variant < 0):
        raise ValueError("dropout requires aligned nonnegative row IDs and variant")
    if batch.view_valid.shape != (len(batch.q), 5, 2) or batch.view_valid.dtype != torch.bool:
        raise ValueError("dropout requires H0-H4 boolean view validity")
    masks = batch.view_valid.clone()
    modulus = 4 if protocol == "legacy" else 10
    case = torch.as_tensor((rows % modulus + (0 if protocol == "legacy" else variant % modulus)) % modulus,
                           device=masks.device)
    for h in (1, 2, 3):
        masks[case == 1, h, h % 2] = False
        masks[case == 2, h, 1-h % 2] = False
        if h == 2:
            masks[case == 3, h] = False
    if protocol == "burst-cold-v1":
        masks[case == 4, 1:, 1] = False  # last real Column at H0, then outage
        masks[case == 5, 1:, 0] = False  # last real Wrist at H0, then outage
        masks[case == 6, :, 1] = False   # no Column history at cold start
        masks[case == 7, :, 0] = False   # no Wrist history at cold start
        masks[case == 8, 1:3, :] = False # both absent at H1/H2; actual H3 reacquisition
        masks[case == 9, 1:3, variant % 2] = False
    return replace(batch, view_valid=masks)


def dropout_contract(protocol):
    """Keep old configs byte-for-byte equivalent; opt-in runs record the change."""
    if protocol == "legacy":
        return {}
    if protocol != "burst-cold-v1":
        raise ValueError("unknown measurement-dropout protocol")
    return {"measurement_dropout": dict(protocol=protocol, variants=10,
            duration="recorded-chain-only", modifies="view_valid-only",
            selection="row-plus-request-index-modulo-10")}


def configuration(output, config):
    path = output/"config.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("output belongs to a different immutable training contract")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("nonempty output has no matching config; refusing to overwrite it")
    else:
        write_json(path, config)


def main(argv=None):
    args = parser().parse_args(argv)
    device = torch.device(args.device)
    smoke = args.stage == "smoke"
    if args.lewm_from is not None and args.stage in ("smoke", "lewm"):
        raise ValueError("--lewm-from is for preflight/all/actor, never smoke or LeWM training")
    if args.stage not in ("preflight", "smoke") and not args.run_formal:
        raise ValueError("formal training requires --run-formal; preflight never trains")
    if device.type == "cuda" and (os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or device.index not in (None, 0)):
        raise ValueError("use physical GPU1: CUDA_VISIBLE_DEVICES=1 and --device cuda:0")
    if args.stage not in ("preflight", "smoke") and device.type != "cuda":
        raise ValueError("formal training requires GPU1, not an accidental CPU run")
    torch.set_num_threads(2)
    random.seed(20000); np.random.seed(20000); torch.manual_seed(20000)
    torch.set_float32_matmul_precision("high")
    schedule = V21Schedule(sigreg_weight=args.sigreg_weight)
    output = args.output.resolve() / "smoke" if smoke else args.output.resolve()
    paths = None; source = None
    if smoke:
        vocabulary = TaskVocabulary.build(["pick red block", "push blue block"])
        scale = [2.]*6
    else:
        if args.source_config is None:
            raise ValueError("--source-config is required; source paths are never guessed")
        given = json.loads(args.source_config.read_text())
        given = given.get("sources", given)
        names = ("endpoint_root", "companion_root", "temporal_root", "rgb_root",
                 "fish_wrist_root", "fish_column_root", "dino_cache_root")
        paths = {name: str(given[name]) for name in names}
        source = V21Sources(paths, device=device, motion_only=args.stage == "lewm")
        vocabulary = source.vocabulary
        from fcpg.prod_v17_8_training import compute_joint_delta_loss_scale
        scale = np.asarray(compute_joint_delta_loss_scale(source.archive.archive)).tolist()
    model_config = V21Config(len(vocabulary.words), memory_mode=args.memory_mode)
    config = dict(version=VERSION, contract=CONTRACT, model=asdict(model_config), schedule=asdict(schedule),
                  vocabulary=list(vocabulary.words), source_paths=paths, q_loss_scale=scale, smoke=smoke,
                  lewm_from=None if args.lewm_from is None else str(args.lewm_from.resolve()),
                  sampler=dict(recipe="mixed-locality", block_rows=512, blocks_per_window=4),
                  visual_backbone="frozen-dinov2-vits14", task_encoder="native-word-embedding")
    config.update(dropout_contract(args.measurement_dropout))
    config.update(temporal_training_configuration(args.temporal_feature_policy))
    configuration(output, config)
    if args.stage == "preflight":
        report = source.audit(); write_json(output/"data-audit.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2)); return 0
    core = V21ResearchCore(model_config).to(device); core.set_loss_scale(scale)
    cache = None
    if source is not None and args.stage != "lewm" and device.type == "cuda" and args.feature_bank_gib > 0:
        from fcpg.prod_v20_1_acceleration import TensorRowBlockCache
        cache = TensorRowBlockCache(device, int(args.feature_bank_gib*2**30),
                                    workspace_reserve_bytes=int(args.workspace_gib*2**30))
        source.device_cache = cache
    trainer = V21Trainer(core, vocabulary, output, schedule, microbatch=args.microbatch, smoke=smoke,
                         clear_cache=None if cache is None else lambda: cache.clear(release_allocator=True),
                         temporal_feature_policy=args.temporal_feature_policy)
    if smoke:
        # No source data, hardware, downloads or long training budget.
        for stage in ("lewm", "actor", "dense", "fish"):
            fixture = synthetic_chain(vocabulary, device=device, images=stage != "lewm")
            trainer.run(stage, range(2), lambda i, x=fixture, s=stage: [x if s == "lewm" or args.measurement_dropout == "legacy" else
                measurement_dropout(x, np.arange(len(x.q)), protocol=args.measurement_dropout,
                                    variant=i)], total_steps=2)
        print(json.dumps(dict(smoke=True, real_data_used=False, hardware_used=False, output=str(output))))
        return 0

    active_window = None
    def batches(kind, passes, stage):
        for window_id, window in enumerate(deterministic_mixed_locality_windows(
                source.counts[kind], passes, schedule.chains_per_batch, schedule.seed,
                block_rows=512, blocks_per_window=4)):
            for i in range(0, len(window), schedule.chains_per_batch):
                yield kind, window_id, window, window[i:i+schedule.chains_per_batch], stage

    def prepare(request):
        nonlocal active_window
        kind, window_id, window, rows, stage = request
        if stage != "lewm" and active_window != (stage, kind, window_id):
            source.prefetch(window, kind); active_window = (stage, kind, window_id)
        batch = source.chain(rows, kind, images=stage != "lewm")
        return [measurement_dropout(batch, rows, protocol=args.measurement_dropout,
                                    variant=int(window_id))] if stage != "lewm" else [batch]

    try:
        if args.lewm_from is not None:
            frozen_final = output/"lewm"/"final.pt"
            if frozen_final.exists():
                payload = load_checkpoint(frozen_final, core, vocabulary, schedule, stage="lewm")
                metrics = payload["metrics"]
            else:
                metrics = reuse_frozen_lewm(args.lewm_from, core, vocabulary, schedule, paths)
                for name in ("latest.pt", "final.pt"):
                    save_checkpoint(output/"lewm"/name, core, vocabulary, schedule,
                        stage="lewm", step=metrics["updates"], complete=True, smoke=False, metrics=metrics)
                print(json.dumps(dict(stage="lewm", reused_from=str(args.lewm_from),
                                      additional_training_updates=0)), flush=True)
            record_stage_metrics(output, "lewm", metrics)
        elif args.stage in ("all", "lewm"):
            trainer.run("lewm", batches("archive", schedule.archive_passes, "lewm"), prepare,
                        math.ceil(source.counts["archive"]/schedule.chains_per_batch)*schedule.archive_passes)
        if args.stage in ("all", "actor"):
            load_checkpoint(output/"lewm"/"final.pt", core, vocabulary, schedule, stage="lewm")
            for stage, kind, passes in (("actor", "archive", schedule.archive_passes),
                                        ("dense", "dense", schedule.dense_passes)):
                trainer.run(stage, batches(kind, passes, stage), prepare,
                            math.ceil(source.counts[kind]/schedule.chains_per_batch)*passes)
            rng = np.random.default_rng(schedule.seed+7)
            count = schedule.fish_chains_per_batch//2
            selections = ((rng.integers(source.counts["fish"], size=count),
                           rng.integers(source.counts["archive"], size=count)) for _ in range(schedule.fish_updates))
            def mixed(request):
                update, ids = request
                return [measurement_dropout(source.chain(rows, kind), rows,
                                            protocol=args.measurement_dropout, variant=update)
                        for kind, rows in zip(("fish", "archive"), ids)]
            trainer.run("fish", enumerate(selections), mixed, schedule.fish_updates)
    finally:
        if cache is not None:
            write_json(output/"input-metrics.json", cache.stats())
            cache.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
