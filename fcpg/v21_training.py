"""V21 causal training and ordinary resumable checkpoints; no hardware imports."""
from dataclasses import asdict, dataclass, replace
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F

from .v21_core import VERSION, COMPATIBLE_VERSIONS, CONTRACT, POSITION_UNIT, V21Config, V21ResearchCore
from .v21_data import TaskVocabulary
from .v21_memory import VisualObservation
from .prod_v20_1_arm import ARMProjector
from .prod_v20_3_training import LeWMLossParts, combine_lewm_loss_parts
from .v21_temporal_contract import (begin_training_temporal_contract, constrain_training_age_columns,
    restore_training_temporal_consumer)


@dataclass(frozen=True)
class V21Schedule:
    archive_passes: int = 4
    dense_passes: int = 1
    fish_updates: int = 4000
    chains_per_batch: int = 64       # 4 horizons = effective 256 action pairs
    fish_chains_per_batch: int = 16
    learning_rate: float = 1e-4
    fish_learning_rate: float = 3e-5
    q_uncertainty_loss_weight: float = .05
    sigreg_weight: float = .02
    sigreg_recipe: str = "arm-source-local-ecf17-proj256-nscaled-v1"
    checkpoint_every: int = 500
    seed: int = 20000

    def __post_init__(self):
        if self.sigreg_weight not in (0., .02):
            raise ValueError("SIGReg controls are 0 and .02 only")
        if self.q_uncertainty_loss_weight != .05:
            raise ValueError("ARM uncertainty weight is fixed at .05")


def rng_state(device=None):
    cuda = None
    if device is not None and torch.device(device).type == "cuda":
        index = torch.device(device).index or 0
        cuda = dict(index=index, state=torch.cuda.get_rng_state(index))
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=cuda)


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"]["state"], state["cuda"]["index"])


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def record_stage_metrics(output, stage, metrics):
    path = Path(output)/"metrics.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report[stage] = metrics
    write_json(path, report)


def save_checkpoint(path, core, vocabulary, schedule, *, stage, step, complete, smoke,
                    optimizer=None, projector=None, metrics=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    version = "21.0.4" if core.config.prediction_horizon == 40 else VERSION
    payload = dict(version=version, contract=CONTRACT, position_unit=POSITION_UNIT,
        config=asdict(core.config), vocabulary=list(vocabulary.words), text_max_length=vocabulary.max_length,
        schedule=asdict(schedule), stage=stage, step=int(step), complete=bool(complete), smoke=bool(smoke),
        model=core.state_dict(), optimizer=None if optimizer is None else optimizer.state_dict(),
        projector=None if projector is None else projector.state_dict(),
        rng=rng_state(next(core.parameters()).device), metrics=metrics or {})
    if getattr(core, 'temporal_training_contract', None) is not None:
        payload['temporal_training_contract'] = core.temporal_training_contract
        payload['temporal_feature_coverage'] = core.temporal_feature_coverage
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def load_checkpoint(path, core, vocabulary, schedule=None, *, stage=None, optimizer=None,
                    projector=None, restore_random=False, allow_smoke=False, allow_missing_prefixes=()):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    requested_temporal = getattr(core, 'temporal_training_contract', None)
    saved_temporal = payload.get('temporal_training_contract')
    # Legacy weights omitted the two horizon fields; their trained capacity is five.
    payload['config'] = asdict(V21Config(**payload['config']))
    if optimizer is not None and requested_temporal != saved_temporal:
        raise ValueError('resume cannot change the temporal feature policy')
    if payload.get("version") not in COMPATIBLE_VERSIONS:
        raise ValueError("V21 research checkpoint version differs; unsupported version")
    expected = dict(contract=CONTRACT, position_unit=POSITION_UNIT,
                    config=asdict(core.config), vocabulary=list(vocabulary.words), text_max_length=vocabulary.max_length)
    if allow_missing_prefixes:
        # A warm start may ADD a declared module (e.g. the object-relative token); the
        # saved config then differs only in that flag and the new keys are absent.
        saved = dict(payload.get("config") or {}); expected_config = dict(expected["config"])
        saved.pop("object_relative", None); expected_config.pop("object_relative", None)
        saved.pop("place_relative", None); expected_config.pop("place_relative", None)
        saved.pop("place_goal", None); expected_config.pop("place_goal", None)
        if saved != expected_config:
            raise ValueError("warm-start checkpoint config differs beyond the added module")
        expected["config"] = payload.get("config")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"V21 research checkpoint {key} differs; deployment repair weights are incompatible")
    if payload.get("smoke") is not False and not allow_smoke:
        raise ValueError("smoke checkpoint cannot enter a formal run")
    if stage is not None and payload.get("stage") != stage:
        raise ValueError("checkpoint training stage differs")
    if schedule is not None and payload.get("schedule") != asdict(schedule):
        raise ValueError("resume cannot change the schedule")
    from .v21_models import admit_untrained_scene_projections
    admit_untrained_scene_projections(payload["model"], core.state_dict())
    if any(not torch.isfinite(value).all() for value in payload["model"].values() if value.is_floating_point()):
        raise ValueError("nonfinite checkpoint parameter")
    if allow_missing_prefixes:
        result = core.load_state_dict(payload["model"], strict=False)
        stray = [k for k in result.missing_keys if not k.startswith(tuple(allow_missing_prefixes))]
        if stray or result.unexpected_keys:
            raise ValueError(f"checkpoint keys differ beyond the added module: missing={stray} unexpected={result.unexpected_keys}")
    else:
        core.load_state_dict(payload["model"], strict=True)
    core.reader._projection_cache = None
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if projector is not None:
        projector.load_state_dict(payload["projector"], strict=True)
    if restore_random:
        restore_rng(payload["rng"])
    if saved_temporal is not None:
        core.temporal_training_contract = saved_temporal
        core.temporal_feature_coverage = payload['temporal_feature_coverage']
        restore_training_temporal_consumer(core)
    return payload


def reuse_frozen_lewm(path, core, vocabulary, schedule, source_paths):
    """Import ONLY an unchanged, completed ARM. Never import old Actor/reader.

    Ordinary configuration and state-key checks, no cryptographic identity.
    This is not optimizer resume and does not claim any new ARM exposures.
    """
    path = Path(path)
    source_config = json.loads((path.parent.parent/"config.json").read_text())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    allowed = {("21.0.0", "v21-readback-addressed-visual-memory-v1")} | {(v, CONTRACT) for v in COMPATIBLE_VERSIONS}
    if (payload.get("version"), payload.get("contract")) not in allowed:
        raise ValueError("unsupported frozen LeWM contract")
    if (payload.get("stage") != "lewm" or payload.get("complete") is not True
            or payload.get("smoke") is not False or source_config.get("smoke") is not False
            or payload.get("position_unit") != POSITION_UNIT):
        raise ValueError("only completed formal LeWM checkpoints may be reused")
    if (payload.get("schedule") != asdict(schedule) or payload.get("vocabulary") != list(vocabulary.words)
            or source_config.get("source_paths") != source_paths
            or source_config.get("schedule") != asdict(schedule)):
        raise ValueError("frozen LeWM data, vocabulary or schedule differs")
    expected_steps = (320000+schedule.chains_per_batch-1)//schedule.chains_per_batch*schedule.archive_passes
    metrics = payload.get("metrics", {})
    if (payload.get("step") != expected_steps or metrics.get("updates") != expected_steps
            or metrics.get("complete") is not True
            or metrics.get("pair_exposures") != 320000*4*schedule.archive_passes):
        raise ValueError("frozen LeWM exposure budget is incomplete")
    model = payload["model"]
    if not torch.equal(model["q_loss_scale"], core.q_loss_scale.detach().cpu()):
        raise ValueError("frozen LeWM calibrated normalization differs")
    from .v21_models import admit_untrained_scene_projections
    state = {k.removeprefix("lewm."): v for k, v in model.items() if k.startswith("lewm.")}
    admit_untrained_scene_projections(state, core.lewm.state_dict())
    expected = core.lewm.state_dict()
    if (state.keys() != expected.keys() or any(state[k].shape != expected[k].shape for k in expected)
            or any(not torch.isfinite(v).all() for v in state.values())):
        raise ValueError("frozen LeWM parameter contract differs or is nonfinite")
    core.lewm.load_state_dict(state, strict=True)
    core.lewm.requires_grad_(False).eval()
    return dict(metrics, reused_from=str(path.resolve()), reused_version=payload["version"],
                additional_training_updates=0, actor_or_reader_imported=False)


def event_weights(chain):
    delta = chain.paths[:, :, -1, 5] - chain.paths[:, :, 0, 5]
    return 1. + 2. * (delta < -1).float() + (delta > 1).float()


def actor_feature_branch_state(state, hidden):
    """Hide Actor visual memory while keeping real camera/cue availability truthful.

    Only the Actor receives this temporary state. Dynamics, observation writes
    and subsequent completed history continue to use the factual state.
    """
    if hidden is None:
        return state
    if hidden.dtype != torch.bool or hidden.shape != (len(state.q),):
        raise ValueError('Actor feature dropout must be bool [B]')
    hidden=hidden.to(state.q.device)
    memory=replace(state.memory,valid=state.memory.valid & ~hidden[:,None,None])
    return replace(state,memory=memory,
        grounding=torch.where(hidden[:,None,None,None],torch.zeros_like(state.grounding),state.grounding),
        visual_error=torch.where(hidden[:,None,None,None],torch.zeros_like(state.visual_error),state.visual_error),
        measurement_pending=state.measurement_pending & ~hidden[:,None],
        visual_feedback_available=state.visual_feedback_available & ~hidden[:,None])


def chain_forward(core, chain, stage, projector=None):
    """Four actual transitions; no Actor proposal is replayed as real history."""
    if stage not in ("lewm", "actor"):
        raise ValueError("unknown training stage")
    if chain.features is None and stage == "actor":
        raise ValueError("Actor requires the real visual feature path")
    def observation(h):
        if chain.features is None:
            return None
        return VisualObservation(chain.features[:, h], chain.q[:, h, None].expand(-1, 2, -1),
                                 chain.endpoint_times[:, h, None].expand(-1, 2), chain.view_valid[:, h],
                                 object_relative=None if chain.object_relative is None else chain.object_relative[:, h],
                                 place_relative=(None if getattr(chain, 'place_relative', None) is None
                                               else chain.place_relative[:, h]),
                                 place_goal=None if getattr(chain, 'place_goal', None) is None else chain.place_goal[:, h])
    state = core.observe(None, observation(0), chain.q[:, 0], chain.endpoint_times[:, 0])
    losses, regression, uncertainty, projected = [], [], [], []
    for h in range(4):
        if stage == "actor":
            actor_state=actor_feature_branch_state(state,chain.actor_visual_drop)
            action = core.act(actor_state, chain.text_ids, chain.readback_offsets[:, h, :5]).commands_position
            target = chain.commands[:, h].detach()
            scale = core.q_loss_scale
            position = F.smooth_l1_loss(action/scale, target/scale, reduction="none").mean((1, 2))
            pv = torch.diff(torch.cat((state.q[:, None], action), 1), dim=1)
            tv = torch.diff(torch.cat((state.q[:, None], target), 1), dim=1)
            velocity = F.smooth_l1_loss(pv/scale, tv/scale, reduction="none").mean((1, 2))
            losses.append(position + .2*velocity)
            with torch.no_grad():
                forecast = core.predict(state, chain.traces[h], chain.readback_offsets[:, h, 1:])
        else:
            forecast = core.predict(state, chain.traces[h], chain.readback_offsets[:, h, 1:])
            pred, target = forecast.response.predicted_q_path[:, 1:], chain.paths[:, h, 1:]
            scale = core.q_loss_scale
            regression.append(F.smooth_l1_loss(pred/scale, target/scale))
            error = (pred-target)/scale
            sigma = (forecast.response.q_uncertainty/scale).clamp_min(1e-3)
            uncertainty.append((.5*(error.detach()/sigma).square()+sigma.log()).mean())
            projected.append(projector(forecast.response.hidden))
        if h < 3:
            times = state.time[:, None]+chain.readback_offsets[:, h]
            state = core.complete(state, forecast, chain.traces[h], chain.paths[:, h], times, observation(h+1))
    if stage == "actor":
        return torch.stack(losses, 1)
    return LeWMLossParts(torch.stack(regression).mean(), torch.stack(uncertainty).mean(),
                        torch.stack(projected), chain.source_index)


class V21Trainer:
    def __init__(self, core, vocabulary, output, schedule=V21Schedule(), *, microbatch=8, smoke=False,
                 clear_cache=None, learning_rate_overrides=None, temporal_feature_policy='legacy',
                 precision='bf16'):
        if precision not in ('bf16', 'fp32'):
            raise ValueError('training precision must be bf16 or fp32')
        self.core, self.vocabulary, self.output, self.schedule = core, vocabulary, Path(output), schedule
        # {parameter-name prefix: lr}. A module added at a warm start (zero-initialised)
        # cannot grow at the repair rate meant for already-trained weights.
        self.learning_rate_overrides = dict(learning_rate_overrides or {})
        self.temporal_feature_policy = temporal_feature_policy
        self.precision = precision
        self.microbatch, self.smoke, self.clear_cache = int(microbatch), bool(smoke), clear_cache
        if self.microbatch < 1:
            raise ValueError("positive microbatch required")
        self.device = next(core.parameters()).device
        self.projector = ARMProjector().to(self.device)

    def _compute_context(self):
        if self.device.type == 'cuda' and self.precision == 'bf16':
            return torch.autocast('cuda', dtype=torch.bfloat16)
        return nullcontext()

    def configure(self, stage):
        if stage != 'lewm':
            begin_training_temporal_contract(self.core, self.temporal_feature_policy)
        self.core.zero_grad(set_to_none=True); self.projector.zero_grad(set_to_none=True)
        self.core.requires_grad_(False).eval()
        self.projector.requires_grad_(stage == "lewm").train(stage == "lewm")
        if stage == "lewm":
            self.core.lewm.requires_grad_(True).train()
            parameters = [*self.core.lewm.parameters(), *self.projector.parameters()]
        else:
            self.core.reader.requires_grad_(True).train()
            self.core.actor.requires_grad_(True).train()
            parameters = [*self.core.reader.parameters(), *self.core.actor.parameters()]
        lr = self.schedule.fish_learning_rate if stage == "fish" else self.schedule.learning_rate
        if not self.learning_rate_overrides:
            return torch.optim.AdamW(parameters, lr=lr, weight_decay=.01, foreach=False)
        named = {id(p): n for n, p in self.core.named_parameters()}
        groups = {None: []}
        for p in parameters:
            prefix = next((k for k in self.learning_rate_overrides if named.get(id(p), "").startswith(k)), None)
            groups.setdefault(prefix, []).append(p)
        missing = [k for k in self.learning_rate_overrides if k not in groups]
        if missing:
            raise ValueError(f"learning-rate override matches no trainable parameter: {missing}")
        return torch.optim.AdamW([dict(params=v, lr=(lr if k is None else float(self.learning_rate_overrides[k])))
                                  for k, v in groups.items() if v], lr=lr, weight_decay=.01, foreach=False)

    def step(self, batches, optimizer, stage):
        """Exact group reduction independent of microbatch; OOM retries before step only."""
        phase = "lewm" if stage == "lewm" else "actor"
        total = sum(len(batch.q) for batch in batches)
        weights = [event_weights(batch) for batch in batches]
        denominators = [weight.sum(0).clamp_min(1.) for weight in weights]
        saved_rng = rng_state(self.device)
        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                parts = []; total_loss = 0.
                for batch, event, denominator in zip(batches, weights, denominators):
                    for start in range(0, len(batch.q), self.microbatch):
                        stop = min(start+self.microbatch, len(batch.q))
                        micro = batch.slice(start, stop)
                        with self._compute_context():
                            value = chain_forward(self.core, micro, phase, self.projector)
                        if phase == "lewm":
                            parts.append(value)
                        else:
                            # Normalize event weights once per FULL source group,
                            # then preserve the requested fish/archive proportions.
                            loss = (value*event[start:stop]/denominator).sum()/4 * (len(batch.q)/total)
                            if not torch.isfinite(loss):
                                raise ValueError("nonfinite Actor loss")
                            loss.backward(); total_loss += float(loss.detach())
                metrics = {}
                if phase == "lewm":
                    loss, metrics = combine_lewm_loss_parts(parts, schedule=self.schedule)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite ARM loss")
                    loss.backward(); total_loss = float(loss.detach())
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                del parts
                value = loss = micro = None
                if self.clear_cache is not None:
                    self.clear_cache()
                if self.microbatch == 1:
                    raise
                self.microbatch = max(1, self.microbatch//2)
                restore_rng(saved_rng)
        params = [p for group in optimizer.param_groups for p in group["params"]]
        gradient = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        # Never retry a partially applied optimizer step; resume from a saved checkpoint.
        optimizer.step()
        constrain_training_age_columns(self.core, optimizer)
        return dict(loss=total_loss, gradient_norm=float(gradient), chains=total,
                    pair_exposures=total*4, microbatch=self.microbatch, precision=self.precision, **metrics)

    def run(self, stage, batch_ids, prepare, total_steps):
        optimizer = self.configure(stage)
        directory = self.output/stage; latest = directory/"latest.pt"; final = directory/"final.pt"
        resume = 0; metrics = dict(stage=stage, updates=0, pair_exposures=0, complete=False, smoke=self.smoke)
        path = final if final.exists() else latest
        if path.exists():
            payload = load_checkpoint(path, self.core, self.vocabulary, self.schedule, stage=stage,
                optimizer=optimizer, projector=self.projector if stage == "lewm" else None,
                restore_random=True, allow_smoke=self.smoke)
            resume, metrics = int(payload["step"]), payload["metrics"]
            if resume > total_steps:
                raise ValueError("checkpoint exceeds the fixed stage budget")
            if payload["complete"]:
                if resume != total_steps:
                    raise ValueError("incomplete final checkpoint budget")
                # Repair the ordinary report if a process ended after final.pt
                # was committed but before metrics.json was committed.
                record_stage_metrics(self.output, stage, metrics)
                return metrics
        started = time.monotonic()
        step = resume
        for ordinal, ids in enumerate(batch_ids, 1):
            if ordinal <= resume:
                continue
            if ordinal > total_steps:
                raise ValueError("sampler exceeds declared budget")
            loaded_at = time.monotonic()
            batch = prepare(ids)
            compute_at = time.monotonic()
            result = self.step(batch, optimizer, stage)
            result.update(loader_host_s=compute_at-loaded_at, compute_host_s=time.monotonic()-compute_at)
            step = ordinal
            metrics = dict(metrics, updates=step, last=result,
                pair_exposures=metrics["pair_exposures"]+result["pair_exposures"], elapsed_s=time.monotonic()-started)
            if step == resume+1 or step % 50 == 0 or self.smoke:
                print(json.dumps(dict(stage=stage, step=step, total_steps=total_steps, **result)), flush=True)
            if step % self.schedule.checkpoint_every == 0:
                save_checkpoint(latest, self.core, self.vocabulary, self.schedule, stage=stage, step=step,
                    complete=False, smoke=self.smoke, optimizer=optimizer,
                    projector=self.projector if stage == "lewm" else None, metrics=metrics)
        if step != total_steps:
            raise ValueError("sampler ended before the declared stage budget")
        metrics["complete"] = True
        for path in (latest, final):
            save_checkpoint(path, self.core, self.vocabulary, self.schedule, stage=stage, step=step,
                complete=True, smoke=self.smoke, optimizer=optimizer,
                projector=self.projector if stage == "lewm" else None, metrics=metrics)
        record_stage_metrics(self.output, stage, metrics)
        return metrics


def synthetic_chain(vocabulary, *, batch=4, device="cpu", images=True):
    """Synthetic interface fixture, not a robot dataset or trained-model evidence."""
    from .v21_data import TrainingChain
    from .v21_models import CommandTrace
    generator = torch.Generator().manual_seed(20000)
    raw = torch.randn(batch, 61, 6, generator=generator).cumsum(1)*.03
    raw[:, :, 5] += 20.
    actions = torch.cat((raw[:, 1:], raw[:, -1:]), 1)
    endpoint = torch.arange(5)*15
    ids = endpoint[:-1, None] + torch.arange(6)*3
    q = raw[:, endpoint].to(device); paths = raw[:, ids].to(device)
    target = actions[:, ids[:, :-1]].to(device)
    times = torch.arange(5, dtype=torch.float64)[None].expand(batch, -1)*.5
    offsets = torch.arange(6, dtype=torch.float64)[None, None].expand(batch, 4, -1)*.1
    traces = tuple(CommandTrace(actions[:, h*15:(h+1)*15].to(device),
        (torch.arange(15, dtype=torch.float64)[None].expand(batch, -1)/30).to(device),
        torch.ones(batch, 15, dtype=torch.bool, device=device), "recorded", True) for h in range(4))
    from .v21_encoder import DIM, PATCHES
    features = torch.randn(batch, 5, 2, PATCHES, DIM, generator=generator).to(device) if images else None
    return TrainingChain(q, times.to(device), target, paths, offsets.to(device), traces, features,
        torch.ones(batch, 5, 2, dtype=torch.bool, device=device),
        vocabulary.encode(["pick red block"]*batch, device=device),
        (torch.arange(batch, device=device)//2).long())
