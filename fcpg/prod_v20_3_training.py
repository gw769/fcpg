"""V20.3 training mechanics shared by synthetic checks and real-data stages.

Stage scheduling is explicit and fixed.  No held episode is opened by ``all``.
Checkpoints include optimizer and RNG state and use the final update, not best.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from .prod_v20_3 import V203Core, V203Config, VERSION, CONTRACT, POSITION_UNIT, detach_actual
from .prod_v20_policy import future_q_path_loss, normalized_action_loss


@dataclass(frozen=True)
class V203Schedule:
    seed: int = 20100
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
    sigreg_weight: float = 0.02
    sigreg_recipe: str = "arm-source-local-ecf17-proj256-nscaled-v1"

    def __post_init__(self):
        if self.sigreg_weight not in (0.0, 0.02):
            raise ValueError("ARM SIGReg permits only the matched 0 or 0.02 weights")
        if self.sigreg_recipe != "arm-source-local-ecf17-proj256-nscaled-v1":
            raise ValueError("ARM SIGReg recipe is fixed: 17 knots, 256 directions, N-scaled ECF")

    @property
    def actor_pair_exposures(self):
        return self.archive_rows * self.horizons * self.actor_passes


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_all(seed: int = 20100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_gpu1(device: torch.device):
    if device.type != "cuda" or device.index not in (0, None) or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("formal V20.3 training requires CUDA_VISIBLE_DEVICES=1 and --device cuda:0")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU1 must support bf16; CPU is only permitted for synthetic smoke/tests")


def save_checkpoint(path: Path, core: V203Core, optimizer, *, stage: str, step: int,
                    schedule: V203Schedule, metrics: dict, run_id: str):
    # New compatible cores may carry a distinct contract while reusing this
    # trainer.  Legacy V20.3 cores keep the original constants unchanged.
    version = getattr(core, "CHECKPOINT_VERSION", VERSION)
    contract = getattr(core, "CHECKPOINT_CONTRACT", CONTRACT)
    position_unit = getattr(core, "CHECKPOINT_POSITION_UNIT", POSITION_UNIT)
    payload = dict(version=version, contract=contract, position_unit=position_unit,
                   config=asdict(core.config), schedule=asdict(schedule), stage=stage, step=step,
                   run_id=run_id, model=core.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
                   rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), metrics=metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, core: V203Core, *, optimizer=None, stage=None,
                    schedule: V203Schedule | None = None, restore_rng=False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_version = getattr(core, "CHECKPOINT_VERSION", VERSION)
    expected_contract = getattr(core, "CHECKPOINT_CONTRACT", CONTRACT)
    expected_position_unit = getattr(core, "CHECKPOINT_POSITION_UNIT", POSITION_UNIT)
    for key, expected in (("version", expected_version), ("contract", expected_contract), ("position_unit", expected_position_unit),
                          ("config", asdict(core.config))):
        if payload.get(key) != expected:
            raise ValueError(f"checkpoint {key} differs; old model weights cannot be loaded as V20.3")
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


@dataclass(frozen=True)
class LeWMLossParts:
    """Microbatch predictions; SIGReg is deliberately NOT reduced here.

    H is the causal H0..H3 transition index; five command positions remain
    separate temporal slices. Only B consists of independent source chains.
    Holding these small ARM graphs until one complete effective batch avoids
    changing its nonlinear ECF statistic when execution microbatch changes.
    """
    regression: torch.Tensor
    nll: torch.Tensor
    projected: torch.Tensor  # [H,B,5,D], training-only projector output
    sources: torch.Tensor  # [B], -1 denotes unavailable provenance


def combine_lewm_loss_parts(parts, weights=None, *, schedule=None):
    """One source-local regularizer after concatenating every execution slice.

    The zero-weight control still evaluates SIGReg and consumes the identical
    direction RNG. No EMA, detached feature queue, or temporal pseudo-samples
    are introduced. Returned metrics contain no live autograd references.
    """
    from .prod_v20_1_arm import source_local_sigreg
    schedule = V203Schedule() if schedule is None else schedule
    parts = list(parts)
    if not parts:
        raise ValueError("cannot combine an empty ARM effective batch")
    for value in parts:
        if (not isinstance(value, LeWMLossParts)
                or any(not isinstance(item, torch.Tensor) for item in
                       (value.regression, value.nll, value.projected, value.sources))
                or value.regression.ndim != 0 or value.nll.ndim != 0 or value.projected.ndim != 4
                or value.projected.shape[2] != 5 or value.projected.shape[1] < 1
                or value.sources.shape != (value.projected.shape[1],)):
            raise ValueError("ARM loss parts require scalars, [H,B,5,D], and source [B]")
    if weights is None:
        sizes = np.asarray([value.projected.shape[1] for value in parts], dtype=np.float64)
        weights = sizes / sizes.sum()
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.shape != (len(parts),) or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1.)):
        raise ValueError("ARM microbatch reduction weights must sum to one")
    regression = sum(value.regression * float(weight) for value, weight in zip(parts, weights))
    nll = sum(value.nll * float(weight) for value, weight in zip(parts, weights))
    projected = torch.cat([value.projected for value in parts], dim=1)
    sources = torch.cat([value.sources for value in parts], dim=0)
    regularization, coverage = source_local_sigreg(projected, sources, return_stats=True)
    weighted_nll = schedule.q_uncertainty_loss_weight * nll
    weighted_sigreg = schedule.sigreg_weight * regularization
    total = regression + weighted_nll + weighted_sigreg
    values = torch.stack((regression, nll, regularization, weighted_nll, weighted_sigreg, total)).detach().float().cpu().tolist()
    metrics = dict(zip(("q_regression", "uncertainty_nll", "sigreg", "weighted_nll", "weighted_sigreg", "total"), values))
    metrics.update(sigreg_weight=schedule.sigreg_weight, sigreg_recipe=schedule.sigreg_recipe,
                   source_coverage=coverage, independent_chain_rows=len(sources),
                   transition_slices=projected.shape[0], command_slices=projected.shape[2],
                   microbatches_combined=len(parts), statistic_scope="whole-effective-batch-per-source",
                   uncertainty_target="independent readback residual with detached mean error")
    return total, metrics


def chain_loss(core: V203Core, chain: Mapping, *, stage: str, actor_horizons=None,
               schedule=None, return_parts=False):
    """Four causal transitions using the exact runtime predict/update methods.

    ``measurements`` contain RAW image measurements for H0..H4, never filtered
    posteriors. Targets come only from independent measured readbacks.  Cached
    endpoint measurements must have been made from real RGB plus DINO.
    Formal LeWM trainers MUST use ``return_parts=True``: the scalar convenience
    return is only an already-complete batch, not a microbatch-averagable loss.
    """
    schedule = V203Schedule() if schedule is None else schedule
    if return_parts and stage != "lewm":
        raise ValueError("structured ARM loss parts are available only for the lewm stage")
    observations = chain["measurements"]
    paths = chain["measured_q_paths_position"]
    commands = chain["commands_position"]
    dt = chain["time_deltas"]
    q = chain["q_position"]
    valid = chain["view_valid"]
    geometry = chain.get("geometry")
    batch = len(q)
    if q.shape != (batch, 5, 6) or paths.shape != (batch, 4, 6, 6) or commands.shape != (batch, 4, 5, 6):
        raise ValueError("V20.3 chain needs H0..H4 and four exact command/readback transitions")
    if not torch.allclose(paths[:, :, 0], q[:, :-1], atol=1.e-4, rtol=0) or not torch.allclose(paths[:, :, -1], q[:, 1:], atol=1.e-4, rtol=0):
        raise ValueError("same-chain measured endpoint clock mismatch")
    if dt.shape != (batch, 4, 5) or not bool((dt > 0).all()):
        raise ValueError("five real elapsed time intervals required")
    initial_time = chain.get("initial_time_s", q.new_zeros(batch))
    first = observations[0]
    if getattr(core, "INITIAL_CONSUMES_RAW_MEASUREMENT", False):
        # V20.3.2 caches raw image fields and performs its q/history-conditioned
        # update inside the core. This keeps the training clock identical to
        # runtime and prevents a pre-fused posterior from being fused again.
        state = core.initial_from_measurement(first, q[:, 0], valid[:, 0], initial_time, geometry=geometry)
    else:
        if geometry is not None:
            first = core.observer.assimilate(first, geometry.make_prior(q[:, 0]), valid[:, 0])
        state = core.initial_from_measurement(first, q[:, 0], valid[:, 0], initial_time, geometry=geometry)
    losses, regression_terms, nll_terms, projected_terms = [], [], [], []
    for horizon in range(4):
        if stage == "lewm":
            prediction = core.predict(state, commands[:, horizon], dt[:, horizon], geometry, return_arm_hidden=True)
            loss = future_q_path_loss(prediction.predicted_q_path, paths[:, horizon], core.q_loss_scale)
            # Proper uncertainty is supervised by the independent readback;
            # q0 is never in either loss and cannot dilute future-state error.
            error = (prediction.predicted_q_path[:, 1:] - paths[:, horizon, 1:]) / core.q_loss_scale
            std = (prediction.q_uncertainty / core.q_loss_scale).clamp_min(1.e-3)
            uncertainty_loss = (0.5 * (error.detach() / std).square() + std.log()).mean()
            if prediction.arm_hidden is None or prediction.arm_hidden.shape != (batch, 5, 192):
                raise ValueError("ARM training requires all five true response-head hidden states")
            regression_terms.append(loss)
            nll_terms.append(uncertainty_loss)
            projected_terms.append(core.arm_sigreg_projector(prediction.arm_hidden))
        elif stage == "actor":
            output = core.act(state, chain["task_embedding"])
            # One chain forward accounts for four independent BC exposures;
            # the actual (not Actor-proposed) historical command is replayed.
            target = commands[:, horizon]
            # Event-balanced imitation: archive/fish windows are dominated by
            # near-static gripper motion.  Weight genuine closing and release
            # windows from the recorded q6 transition more strongly so the
            # Actor cannot minimize loss by always holding the jaw open.  This
            # is a training-only weight; no inference-time action is injected.
            # Event labels come from measured q6 readback, not requested
            # commands, so servo lag or saturation cannot fabricate a close.
            q6_delta = paths[:, horizon, -1, 5] - paths[:, horizon, 0, 5]
            closed = q6_delta < -1.0
            released = q6_delta > 1.0
            event_weights = 1.0 + 2.0 * closed.to(q.dtype) + released.to(q.dtype)
            losses.append(normalized_action_loss(output.commands_position, commands[:, horizon],
                                                  q[:, horizon], core.q_loss_scale, event_weights))
            with torch.no_grad():
                prediction = core.predict(state, commands[:, horizon], dt[:, horizon], geometry)
        else:
            raise ValueError("chain stage must be lewm or actor")
        if horizon < 3:
            target_time = state.endpoint_time_s + dt[:, horizon].sum(-1)
            with torch.no_grad():
                state = core.complete_from_measurement(state, prediction, observations[horizon + 1],
                                                       paths[:, horizon], valid[:, horizon + 1], target_time, geometry=geometry)
                state = detach_actual(state)
            if stage == "actor":
                # The frozen filter is detached; the Actor-owned encoders are
                # trained freshly at every actual endpoint, not only H0.
                from dataclasses import replace
                mode = "neither" if core.config.actor_mode == "no_innovation" else "full"
                grounding, innovation = core.control_encoding(state.observation, prediction.prior,
                                              paths[:, horizon], prediction.predicted_q_path, core.q_loss_scale,
                                              innovation_mode=mode, view_valid=valid[:, horizon + 1])
                state = replace(state, grounding_tracks=grounding, innovation_tracks=innovation)
    if stage == "lewm":
        source = chain.get("source_index")
        if source is None:
            source = torch.full((batch,), -1, device=q.device, dtype=torch.long)
        else:
            source = torch.as_tensor(source, device=q.device)
        parts = LeWMLossParts(torch.stack(regression_terms).mean(), torch.stack(nll_terms).mean(),
                             torch.stack(projected_terms), source)
        return parts if return_parts else combine_lewm_loss_parts([parts], schedule=schedule)[0]
    return torch.stack(losses).mean()


def _capture_batch_rng(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)


def _restore_batch_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class StageTrainer:
    """Fixed-effective-batch training with truthful timing and bounded OOM replay.

    ``timed_load`` instruments input preparation inside existing loss closures;
    alternatively ``run(load_fn=...)`` separates preparation explicitly. CUDA
    events are read only at telemetry-window boundaries, not per operation.
    CPU launch/wait time and device event time are reported separately.

    OOM during forward/backward/gradient validation is replayable. An optimizer
    step can be partially mutating, so a step-time OOM FAILS CLOSED: no current
    checkpoint is saved; restart from the existing latest checkpoint. Callbacks
    may evict caches but must not mutate the model/optimizer or sample order.
    """
    @staticmethod
    def checkpoint_interval_for_stage(stage):
        """Persistence cadence only; it never changes the training schedule."""
        return 100 if stage in ("appearance", "fcpg", "refit") else 500

    def __init__(self, core, output: Path, device, schedule=V203Schedule(), *, microbatch=16,
                 run_id=None, clear_cache_callback=None, telemetry_interval=20, cache_stats_callback=None):
        self.core = core
        self.output = Path(output)
        self.device = torch.device(device)
        self.schedule = schedule
        self.microbatch = int(microbatch)
        if self.microbatch < 1:
            raise ValueError("microbatch must be positive")
        if int(telemetry_interval) < 1:
            raise ValueError("telemetry interval must be positive")
        self.clear_cache_callback = clear_cache_callback
        self.cache_stats_callback = cache_stats_callback
        self.telemetry_interval = int(telemetry_interval)
        self.run_id = run_id or time.strftime("v203-%Y%m%d-%H%M%S", time.gmtime())
        self._compute_depth = self._loader_depth = 0
        self._reset_window()

    def _reset_window(self):
        self._window_started = time.monotonic()
        self._phase_wall = {name: 0. for name in ("loader", "loader_nested", "compute", "update")}
        self._phase_events = []
        self._loader_marks = self._window_samples = self._window_updates = self._window_ooms = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    @contextmanager
    def _phase(self, name):
        start_event = end_event = None
        if self.device.type == "cuda":
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(self.device))
        started = time.monotonic()
        try:
            yield
        finally:
            self._phase_wall[name] += time.monotonic() - started
            if end_event is not None:
                end_event.record(torch.cuda.current_stream(self.device))
                self._phase_events.append((name, start_event, end_event))

    def timed_load(self, function, *args, **kwargs):
        """Measure a real loader/preparation call without a device synchronize.

        Nested calls count once, so callers may safely instrument both a chain
        builder and its underlying sources. Preparation of frozen-model inputs
        can include device work; this is not labeled disk I/O bandwidth.
        """
        if self._loader_depth:
            return function(*args, **kwargs)
        self._loader_marks += 1
        self._loader_depth += 1
        try:
            with self._phase("loader_nested" if self._compute_depth else "loader"):
                return function(*args, **kwargs)
        finally:
            self._loader_depth -= 1

    def _telemetry(self, pair_multiplier):
        gpu = {name: 0. for name in self._phase_wall}
        peak_allocated = peak_reserved = None
        if self.device.type == "cuda":
            # One synchronization per reporting window. Phase timers never
            # insert operation-level waits or change the compute stream.
            torch.cuda.synchronize(self.device)
            for name, start, end in self._phase_events:
                gpu[name] += start.elapsed_time(end) / 1000.
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        elapsed = max(time.monotonic() - self._window_started, 1.e-9)
        result = dict(window_updates=self._window_updates, window_samples=self._window_samples,
                    elapsed_s=elapsed, samples_per_s=self._window_samples / elapsed,
                    pair_exposures_per_s=self._window_samples * pair_multiplier / elapsed,
                    loader_wall_s=self._phase_wall["loader"] + self._phase_wall["loader_nested"],
                    compute_host_wall_s=max(0., self._phase_wall["compute"] - self._phase_wall["loader_nested"]),
                    update_host_wall_s=self._phase_wall["update"],
                    loader_calls=self._loader_marks, timing_split_available=self._loader_marks > 0,
                    compute_includes_uninstrumented_loader=self._loader_marks == 0,
                    cuda_loader_s=(gpu["loader"] + gpu["loader_nested"]) if self.device.type == "cuda" else None,
                    cuda_compute_s=max(0., gpu["compute"] - gpu["loader_nested"]) if self.device.type == "cuda" else None,
                    cuda_update_s=gpu["update"] if self.device.type == "cuda" else None,
                    cuda_peak_allocated_bytes=peak_allocated, cuda_peak_reserved_bytes=peak_reserved,
                    oom_retries_in_window=self._window_ooms,
                    timing_method="host monotonic and current-stream CUDA events; window synchronization")
        if self.cache_stats_callback is not None:
            result["data_cache"] = self.cache_stats_callback()
        return result

    def _accumulate(self, selected, sample_weights, loss_fn, load_fn, optimizer, parameters, stage):
        optimizer.zero_grad(set_to_none=True)
        value = None
        arm_parts, arm_weights = [], []
        self._last_loss_parts = None
        for begin in range(0, len(selected), self.microbatch):
            part = selected[begin:begin + self.microbatch]
            prepared = self.timed_load(load_fn, part) if load_fn is not None else part
            context = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
            self._compute_depth += 1
            try:
                with self._phase("compute"):
                    with context:
                        loss = loss_fn(prepared)
                    weight = float(sample_weights[begin:begin + len(part)].sum() / sample_weights.sum())
                    if isinstance(loss, LeWMLossParts):
                        if stage != "lewm" or value is not None:
                            raise RuntimeError("structured ARM parts cannot mix with scalar/stage losses")
                        if loss.projected.ndim != 4 or loss.projected.shape[1] != len(part):
                            raise ValueError("ARM parts must retain every selected independent chain")
                        arm_parts.append(loss)
                        arm_weights.append(weight)
                    else:
                        if arm_parts:
                            raise RuntimeError("structured ARM parts cannot mix with scalar/stage losses")
                        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                            raise RuntimeError(f"non-scalar {stage} loss before optimizer update")
                        (loss * weight).backward()
                        weighted = loss.detach().float() * weight
                        value = weighted if value is None else value + weighted
            finally:
                self._compute_depth -= 1
            del loss, prepared
        with self._phase("compute"):
            if arm_parts:
                total, details = combine_lewm_loss_parts(arm_parts, arm_weights, schedule=self.schedule)
                if not torch.isfinite(total):
                    raise RuntimeError("nonfinite lewm loss before optimizer update")
                total.backward()
                value = total.detach().float()
                self._last_loss_parts = details
                # Do not keep the just-consumed full-batch graphs in trainer
                # state, telemetry, or an OOM retry callback.
                arm_parts.clear()
                del total
            if value is None or not torch.isfinite(value):
                raise RuntimeError(f"nonfinite {stage} loss before optimizer update")
            gradient = nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        # Only one loss scalar is read per effective batch, not per microbatch.
        return float(value), float(gradient)

    def _batch_with_replay(self, selected, sample_weights, loss_fn, load_fn, optimizer, parameters, stage, metrics):
        state = _capture_batch_rng(self.device)
        retried_after_clear = False
        while True:
            failed = False
            try:
                return self._accumulate(selected, sample_weights, loss_fn, load_fn, optimizer, parameters, stage)
            except torch.cuda.OutOfMemoryError as error:
                # Release traceback-held graphs BEFORE evicting allocator
                # cache. RuntimeError text alone never enters this branch.
                message = str(error)
                error.__traceback__ = None
                failed = True
            if failed:
                optimizer.zero_grad(set_to_none=True)
                if self.clear_cache_callback is not None:
                    self.clear_cache_callback()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                _restore_batch_rng(state)
                self._window_ooms += 1
                metrics["oom_retries"] = metrics.get("oom_retries", 0) + 1
                if retried_after_clear:
                    if min(self.microbatch, len(selected)) == 1:
                        raise torch.cuda.OutOfMemoryError(
                            f"{stage}: effective batch retained but microbatch=1 still OOM after cache clear; "
                            f"no optimizer update or sample advance occurred: {message}") from None
                    self.microbatch = max(1, min(self.microbatch, len(selected)) // 2)
                retried_after_clear = True
                print(json.dumps(dict(stage=stage, event="oom-replay", microbatch=self.microbatch,
                                      effective_samples=len(selected), retry=metrics["oom_retries"])), flush=True)

    def _write_failure(self, error, context):
        """Write diagnostics, never the possibly failed in-memory weights.

        Inspect only ordinary metadata in the last atomic checkpoint. Memory
        mapping avoids materializing a second model/optimizer on a failing
        process. An unreadable checkpoint is reported, not called recoverable.
        """
        path = self.output / context["stage"] / "latest.pt"
        checkpoint = dict(path=str(path), exists=path.exists(), step=None, stage=None,
                          complete=None, metadata_readable=False)
        if checkpoint["exists"]:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                checkpoint.update(step=int(payload["step"]), stage=str(payload["stage"]),
                                  complete=bool(payload.get("metrics", {}).get("complete", False)),
                                  metadata_readable=True, bytes=path.stat().st_size)
            except Exception as inspection_error:
                checkpoint["inspection_error"] = dict(type=type(inspection_error).__name__,
                                                       message=str(inspection_error))
        selected = context.get("selected")
        selected_rows = None
        if selected is not None:
            try:
                value = np.asarray(selected)
                if value.ndim == 1 and value.dtype.kind in "iu":
                    selected_rows = value.tolist()
            except Exception:
                # Malformed input is not converted into invented row IDs.
                pass
        report = dict(stage=context["stage"], run_id=self.run_id,
                      observed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      complete=False, completed_step=int(context["completed_step"]),
                      total_updates=int(context["total_steps"]),
                      attempted_batch_ordinal=context.get("ordinal"),
                      selected_row_ids=selected_rows, microbatch=int(self.microbatch),
                      failure_phase=context["phase"], error_type=type(error).__name__, error_message=str(error),
                      optimizer_state_possibly_mutated=bool(context["optimizer_state_possibly_mutated"]),
                      failed_batch_checkpointed=False, automatic_non_oom_retry=False,
                      checkpoint_interval_updates=self.checkpoint_interval_for_stage(context["stage"]),
                      expected_checkpoint_step=context["expected_checkpoint_step"],
                      checkpoint=checkpoint)
        if error.__cause__ is not None:
            report["cause"] = dict(type=type(error.__cause__).__name__, message=str(error.__cause__))
        write_json(self.output / context["stage"] / "failure.json", report)

    def run(self, stage: str, indices: Iterable[np.ndarray], loss_fn: Callable,
            *, total_steps: int, learning_rate=None, pair_multiplier=1, load_fn=None, sample_weights_fn=None):
        context = dict(stage=stage, completed_step=0, total_steps=total_steps,
                       expected_checkpoint_step=None, selected=None, ordinal=None,
                       phase="setup", optimizer_state_possibly_mutated=False)
        try:
            return self._run(stage, indices, loss_fn, total_steps=total_steps, learning_rate=learning_rate,
                             pair_multiplier=pair_multiplier, load_fn=load_fn,
                             sample_weights_fn=sample_weights_fn, recovery_context=context)
        except BaseException as error:
            try:
                self._write_failure(error, context)
            except Exception as report_error:
                # Full disks and filesystem faults must not hide the original
                # training failure or trigger a checkpoint of unsafe weights.
                note = f"failure.json could not be written: {type(report_error).__name__}: {report_error}"
                if hasattr(error, "add_note"):
                    error.add_note(note)
                else:  # Local CPU tests may use Python 3.10; training uses 3.11.
                    error.__notes__ = [*getattr(error, "__notes__", []), note]
                    print(note, file=sys.stderr)
            raise
        finally:
            self.core.zero_grad(set_to_none=True)

    def _run(self, stage: str, indices: Iterable[np.ndarray], loss_fn: Callable,
             *, total_steps: int, learning_rate=None, pair_multiplier=1, load_fn=None,
             sample_weights_fn=None, recovery_context):
        recovery = recovery_context
        self.core.freeze_for_stage(stage)
        parameters = [p for p in self.core.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate or self.schedule.learning_rate,
                                      weight_decay=0.01, foreach=False)
        root = self.output / stage
        latest = root / "latest.pt"
        step = 0
        metrics = dict(stage=stage, updates=0, samples=0, pair_exposures=0, loss_sum=0., evidence="training-development")
        if latest.exists():
            recovery["phase"] = "checkpoint-load"
            payload = load_checkpoint(latest, self.core, optimizer=optimizer, stage=stage,
                                      schedule=self.schedule, restore_rng=True)
            step, metrics, self.run_id = payload["step"], payload["metrics"], payload["run_id"]
            recovery.update(completed_step=step, expected_checkpoint_step=step)
        started = time.monotonic()
        self._reset_window()
        requested_microbatch = self.microbatch
        recovery["phase"] = "input-iteration"
        for ordinal, selected in enumerate(indices):
            if ordinal < step:
                continue
            recovery.update(ordinal=ordinal, selected=selected, phase="batch-input")
            selected = np.asarray(selected, dtype=np.int64)
            recovery["selected"] = selected
            if selected.ndim != 1 or not len(selected):
                raise ValueError("each effective batch must contain a nonempty vector of row identities")
            # Match the loss's own denominator across microbatch partitions:
            # e.g. valid-view count for appearance distillation. The callback
            # runs once and the exact same weights are reused on every replay.
            sample_weights = np.ones(len(selected), dtype=np.float64) if sample_weights_fn is None else np.asarray(sample_weights_fn(selected), dtype=np.float64)
            if (sample_weights.shape != selected.shape or not np.isfinite(sample_weights).all()
                    or (sample_weights < 0).any() or sample_weights.sum() <= 0):
                raise ValueError("sample weights must be finite nonnegative [effective_batch] with positive sum")
            recovery["phase"] = "forward-backward"
            loss_value, gradient = self._batch_with_replay(selected, sample_weights, loss_fn, load_fn, optimizer, parameters, stage, metrics)
            try:
                recovery.update(phase="optimizer-step", optimizer_state_possibly_mutated=True)
                with self._phase("update"):
                    optimizer.step()
            except torch.cuda.OutOfMemoryError as error:
                # AdamW may already have updated some parameters/moments.
                # Do not retry, save final/latest, or claim transactional rollback.
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("optimizer.step CUDA OOM may have partially mutated state; training stopped, "
                                   "no checkpoint saved for this batch; discard this process and resume existing latest.pt") from error
            step = ordinal + 1
            recovery.update(completed_step=step, phase="metrics", optimizer_state_possibly_mutated=False)
            metrics.update(updates=step, samples=metrics["samples"] + len(selected),
                           pair_exposures=metrics["pair_exposures"] + pair_multiplier * len(selected),
                           loss_sum=metrics["loss_sum"] + loss_value * len(selected),
                           last_loss=loss_value, gradient_norm=float(gradient), total_updates=total_steps)
            if self._last_loss_parts is not None:
                metrics["loss_parts"] = self._last_loss_parts
                totals = metrics.setdefault("sigreg_source_totals", {})
                for key in ("eligible_rows", "missing_rows", "singleton_rows", "total_rows"):
                    if key in self._last_loss_parts["source_coverage"]:
                        totals[key] = totals.get(key, 0) + int(self._last_loss_parts["source_coverage"][key])
            metrics["runtime"] = dict(microbatch=self.microbatch, requested_microbatch=requested_microbatch,
                                      optimizer_foreach=False, effective_batch_preserved=True,
                                      checkpoint_interval_updates=self.checkpoint_interval_for_stage(stage))
            self._window_samples += len(selected)
            self._window_updates += 1
            if step % self.telemetry_interval == 0 or step == total_steps:
                metrics["throughput"] = self._telemetry(pair_multiplier)
                print(json.dumps(dict(stage=stage, step=step, total=total_steps, loss=loss_value,
                                      elapsed_s=round(time.monotonic() - started, 2),
                                      throughput=metrics["throughput"],
                                      **({"loss_parts": metrics["loss_parts"]} if self._last_loss_parts is not None else {}))), flush=True)
                write_json(root / "metrics.json", metrics)
                self._reset_window()
            if step % self.checkpoint_interval_for_stage(stage) == 0:
                recovery.update(phase="checkpoint-save", expected_checkpoint_step=step)
                save_checkpoint(latest, self.core, optimizer, stage=stage, step=step,
                                schedule=self.schedule, metrics=metrics, run_id=self.run_id)
                write_json(root / "metrics.json", metrics)
            recovery.update(selected=None, ordinal=None, phase="input-iteration")
        if step != total_steps:
            raise RuntimeError(f"{stage} iterator ended at {step}, expected {total_steps}; no final produced")
        recovery.update(phase="final-save", expected_checkpoint_step=step)
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
