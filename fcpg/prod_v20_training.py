"""V20 training mechanics shared by synthetic checks and real-data stages.

Stage scheduling is explicit and fixed.  No held episode is opened by ``all``.
Checkpoints include optimizer and RNG state and use the final update, not best.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from .prod_v20 import V20Core, V20Config, VERSION, CONTRACT, POSITION_UNIT, detach_actual
from .prod_v20_policy import future_q_path_loss, normalized_action_loss


@dataclass(frozen=True)
class V20Schedule:
    seed: int = 20000
    appearance_endpoints: int = 100000
    cad_rows: int = 190000
    cad_passes: int = 4
    real_updates: int = 600
    archive_rows: int = 320000
    horizons: int = 4
    lewm_passes: int = 4
    actor_passes: int = 4
    dense_rows: int = 20000
    dense_passes: int = 1
    fish_updates: int = 4000
    effective_batch: int = 256
    fish_batch: int = 64
    learning_rate: float = 1.e-4
    fish_learning_rate: float = 3.e-5
    cad_prior_translation_std_px: float = 16.0
    q_uncertainty_loss_weight: float = 0.05

    @property
    def actor_pair_exposures(self):
        return self.archive_rows * self.horizons * self.actor_passes


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_all(seed: int = 20000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_gpu1(device: torch.device):
    if device.type != "cuda" or device.index not in (0, None) or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("formal V20 training requires CUDA_VISIBLE_DEVICES=1 and --device cuda:0")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU1 must support bf16; CPU is only permitted for synthetic smoke/tests")


def save_checkpoint(path: Path, core: V20Core, optimizer, *, stage: str, step: int,
                    schedule: V20Schedule, metrics: dict, run_id: str):
    payload = dict(version=VERSION, contract=CONTRACT, position_unit=POSITION_UNIT,
                   config=asdict(core.config), schedule=asdict(schedule), stage=stage, step=step,
                   run_id=run_id, model=core.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
                   rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), metrics=metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, core: V20Core, *, optimizer=None, stage=None,
                    schedule: V20Schedule | None = None, restore_rng=False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (("version", VERSION), ("contract", CONTRACT), ("position_unit", POSITION_UNIT),
                          ("config", asdict(core.config))):
        if payload.get(key) != expected:
            raise ValueError(f"checkpoint {key} differs; old model weights cannot be loaded as V20")
    if stage is not None and payload.get("stage") != stage:
        raise ValueError("checkpoint stage differs")
    if schedule is not None and payload.get("schedule") != asdict(schedule):
        raise ValueError("resume cannot change the training schedule")
    core.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        state = payload["rng"]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])
    return payload


def deterministic_batches(population: int, passes: int, batch: int, seed: int):
    """Every row once per pass, including tails; no replacement or hidden drop."""
    generator = np.random.default_rng(seed)
    for _ in range(passes):
        order = generator.permutation(population)
        for begin in range(0, population, batch):
            yield order[begin:begin + batch]


def chain_loss(core: V20Core, chain: Mapping, *, stage: str, actor_horizons=None):
    """Four causal transitions using the exact runtime predict/update methods.

    ``measurements`` contain RAW image measurements for H0..H4, never filtered
    posteriors. Targets come only from independent measured readbacks.  Cached
    endpoint measurements must have been made from real RGB plus DINO.
    """
    observations = chain["measurements"]
    paths = chain["measured_q_paths_position"]
    commands = chain["commands_position"]
    dt = chain["time_deltas"]
    q = chain["q_position"]
    valid = chain["view_valid"]
    geometry = chain.get("geometry")
    batch = len(q)
    if q.shape != (batch, 5, 6) or paths.shape != (batch, 4, 6, 6) or commands.shape != (batch, 4, 5, 6):
        raise ValueError("V20 chain needs H0..H4 and four exact command/readback transitions")
    if not torch.allclose(paths[:, :, 0], q[:, :-1], atol=1.e-4, rtol=0) or not torch.allclose(paths[:, :, -1], q[:, 1:], atol=1.e-4, rtol=0):
        raise ValueError("same-chain measured endpoint clock mismatch")
    if dt.shape != (batch, 4, 5) or not bool((dt > 0).all()):
        raise ValueError("five real elapsed time intervals required")
    initial_time = chain.get("initial_time_s", q.new_zeros(batch))
    first = observations[0]
    if geometry is not None:
        first = core.observer.assimilate(first, geometry.make_prior(q[:, 0]), valid[:, 0])
    state = core.initial_from_measurement(first, q[:, 0], valid[:, 0], initial_time)
    losses = []
    for horizon in range(4):
        if stage == "lewm":
            prediction = core.predict(state, commands[:, horizon], dt[:, horizon], geometry)
            loss = future_q_path_loss(prediction.predicted_q_path, paths[:, horizon], core.q_loss_scale)
            # Proper uncertainty is supervised by the independent readback;
            # q0 is never in either loss and cannot dilute future-state error.
            error = (prediction.predicted_q_path[:, 1:] - paths[:, horizon, 1:]) / core.q_loss_scale
            std = (prediction.q_uncertainty / core.q_loss_scale).clamp_min(1.e-3)
            uncertainty_loss = (0.5 * (error.detach() / std).square() + std.log()).mean()
            losses.append(loss + V20Schedule().q_uncertainty_loss_weight * uncertainty_loss)
        elif stage == "actor":
            output = core.act(state, chain["task_embedding"])
            # One chain forward accounts for four independent BC exposures;
            # the actual (not Actor-proposed) historical command is replayed.
            losses.append(normalized_action_loss(output.commands_position, commands[:, horizon],
                                                  q[:, horizon], core.q_loss_scale))
            with torch.no_grad():
                prediction = core.predict(state, commands[:, horizon], dt[:, horizon], geometry)
        else:
            raise ValueError("chain stage must be lewm or actor")
        if horizon < 3:
            target_time = state.endpoint_time_s + dt[:, horizon].sum(-1)
            with torch.no_grad():
                state = core.complete_from_measurement(state, prediction, observations[horizon + 1],
                                                       paths[:, horizon], valid[:, horizon + 1], target_time)
                state = detach_actual(state)
            if stage == "actor":
                # The frozen filter is detached; the Actor-owned encoders are
                # trained freshly at every actual endpoint, not only H0.
                from dataclasses import replace
                grounding, innovation = core.control_encoding(state.observation, prediction.prior,
                                              paths[:, horizon], prediction.predicted_q_path, core.q_loss_scale)
                state = replace(state, grounding_tracks=grounding, innovation_tracks=innovation)
    return torch.stack(losses).mean()


class StageTrainer:
    def __init__(self, core, output: Path, device, schedule=V20Schedule(), *, microbatch=16, run_id=None):
        self.core = core
        self.output = Path(output)
        self.device = torch.device(device)
        self.schedule = schedule
        self.microbatch = int(microbatch)
        if self.microbatch < 1:
            raise ValueError("microbatch must be positive")
        self.run_id = run_id or time.strftime("v20-%Y%m%d-%H%M%S", time.gmtime())

    def run(self, stage: str, indices: Iterable[np.ndarray], loss_fn: Callable,
            *, total_steps: int, learning_rate=None, pair_multiplier=1):
        self.core.freeze_for_stage(stage)
        parameters = [p for p in self.core.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate or self.schedule.learning_rate,
                                      weight_decay=0.01)
        root = self.output / stage
        latest = root / "latest.pt"
        step = 0
        metrics = dict(stage=stage, updates=0, samples=0, pair_exposures=0, loss_sum=0., evidence="training-development")
        if latest.exists():
            payload = load_checkpoint(latest, self.core, optimizer=optimizer, stage=stage,
                                      schedule=self.schedule, restore_rng=True)
            step, metrics, self.run_id = payload["step"], payload["metrics"], payload["run_id"]
        started = time.monotonic()
        for ordinal, selected in enumerate(indices):
            if ordinal < step:
                continue
            selected = np.asarray(selected, dtype=np.int64)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            # Weight tails by sample count, not by number of microbatches.
            for begin in range(0, len(selected), self.microbatch):
                part = selected[begin:begin + self.microbatch]
                context = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
                with context:
                    loss = loss_fn(part)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite/non-scalar {stage} loss before optimizer update")
                weight = len(part) / len(selected)
                (loss * weight).backward()
                loss_value += float(loss.detach()) * weight
            gradient = nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            step = ordinal + 1
            metrics.update(updates=step, samples=metrics["samples"] + len(selected),
                           pair_exposures=metrics["pair_exposures"] + pair_multiplier * len(selected),
                           loss_sum=metrics["loss_sum"] + loss_value * len(selected),
                           last_loss=loss_value, gradient_norm=float(gradient), total_updates=total_steps)
            if step % 20 == 0 or step == total_steps:
                print(json.dumps(dict(stage=stage, step=step, total=total_steps, loss=loss_value,
                                      elapsed_s=round(time.monotonic() - started, 2))), flush=True)
            if step % 500 == 0:
                save_checkpoint(latest, self.core, optimizer, stage=stage, step=step,
                                schedule=self.schedule, metrics=metrics, run_id=self.run_id)
                write_json(root / "metrics.json", metrics)
        if step != total_steps:
            raise RuntimeError(f"{stage} iterator ended at {step}, expected {total_steps}; no final produced")
        metrics["mean_loss"] = metrics["loss_sum"] / max(metrics["samples"], 1)
        metrics["complete"] = True
        for name in ("latest.pt", "final.pt"):
            save_checkpoint(root / name, self.core, optimizer, stage=stage, step=step,
                            schedule=self.schedule, metrics=metrics, run_id=self.run_id)
        write_json(root / "metrics.json", metrics)
        return metrics


def hold_commands(q_position):
    return q_position[:, None].expand(-1, 5, -1).clone()


def legal_action_shuffle(commands, q_position, lower, upper, permutation):
    increments = torch.diff(torch.cat((q_position[:, None], commands), 1), dim=1)
    donor = increments[permutation]
    return (q_position[:, None] + donor.cumsum(1)).clamp(lower, upper)


class MetricAccumulator:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def add(self, name, per_sample):
        values = torch.as_tensor(per_sample).double().flatten()
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite evaluation value")
        self.sums[name] = self.sums.get(name, 0.) + values.sum().item()
        self.counts[name] = self.counts.get(name, 0) + values.numel()

    def result(self):
        return {name: dict(mean=value / self.counts[name], count=self.counts[name]) for name, value in self.sums.items()}
