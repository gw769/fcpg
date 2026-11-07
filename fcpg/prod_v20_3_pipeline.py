"""Executable, existing-data V20.3 training pipeline (no robot commands)."""
from dataclasses import asdict, fields, replace
from contextlib import closing
from pathlib import Path
import math
import shutil

import numpy as np
import torch

from .prod_v20_3 import V203Core, V203Config, VERSION, CONTRACT, POSITION_UNIT
from .prod_v20_3_cache import RawObservationCache, build_observation_cache, estimate_cache_bytes
from .prod_v20_3_sources import V203Sources
from .prod_v20_3_training import (V203Schedule, StageTrainer, deterministic_batches, write_json,
                               seed_all, require_gpu1, load_checkpoint, chain_loss)


def development_batches(rows):
    """Fixed donor groups, independent of training microbatch; merge singleton tails."""
    rows = np.asarray(rows, dtype=np.int64)
    groups = [rows[start:start + 16] for start in range(0, len(rows), 16)]
    if len(groups) > 1 and len(groups[-1]) == 1:
        groups[-2] = np.concatenate((groups[-2], groups[-1]))
        groups.pop()
    return groups


def merge_sum_count_metrics(parts):
    """Merge ``{name: {"sum", "count", ...}}`` batches by summing, then recompute means.

    Means of differently sized batches are never averaged; a zero total count
    yields ``mean=None``.
    """
    metrics = {}
    for part in parts:
        for name, value in part.items():
            result = metrics.setdefault(name, {"sum": 0., "count": 0})
            result["sum"] += value["sum"]
            result["count"] += value["count"]
    for value in metrics.values():
        value["mean"] = value["sum"] / value["count"] if value["count"] else None
    return metrics


OCCLUSION_PROBE_REFERENCE = ("full-vision filtered estimate on already-exposed fish training chains; "
                             "not physical ground truth, not anti-fog evidence")
OCCLUSION_SUMMARY_METRICS = ("map_pair_distance_to_full_vision_px", "full_vision_map_nll")


MIN_CLUSTERS_FOR_CI = 8


def fish_row_identity(fish, rows):
    """Recording episode and start frame per fish chain row, for clustering.

    Overlapping windows from the same episode are not independent trials. The
    cluster count decides whether an episode-clustered confidence interval is
    meaningful at all; with fewer than ``MIN_CLUSTERS_FOR_CI`` episodes it is
    reported as unavailable rather than computed.
    """
    rows = np.asarray(rows, dtype=np.int64)
    starts = getattr(fish, "starts", None)
    episodes = getattr(fish, "episodes", None)
    frames = getattr(fish, "frame_indices", None)
    if starts is None or episodes is None or frames is None:
        return dict(available=False, reason="fish source exposes no episode/start metadata",
                    clusters=None, clustered_ci_feasible=False)
    start = np.asarray(starts)[rows]
    episode = np.asarray(episodes)[start].astype(np.int64)
    unique = np.unique(episode)
    return dict(available=True, episode=episode.tolist(), start_frame=np.asarray(frames)[start].astype(np.int64).tolist(),
                clusters=int(len(unique)), unique_episodes=unique.tolist(),
                clustered_ci_feasible=bool(len(unique) >= MIN_CLUSTERS_FOR_CI),
                note="chains are overlapping windows; cluster by episode, never by row")


def occlusion_probe_summary(metrics, names=OCCLUSION_SUMMARY_METRICS):
    """Nest ``occlusion/<schedule>/<estimator>/h<k>/<view>/<metric>`` means by schedule/estimator/horizon/view."""
    summary = {}
    for name, value in metrics.items():
        parts = name.split("/")
        if len(parts) != 6 or parts[0] != "occlusion" or parts[5] not in names:
            continue
        _, schedule, estimator, horizon, view, metric = parts
        summary.setdefault(schedule, {}).setdefault(estimator, {}).setdefault(horizon, {}).setdefault(view, {})[metric] = value["mean"]
    return summary


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("U", "S", "O"):
            return value  # source/frame provenance is metadata, not a tensor
        value = torch.from_numpy(np.ascontiguousarray(value) if value.flags.writeable else np.array(value,copy=True))
    if isinstance(value, torch.Tensor):
        # Time differences are represented in seconds, positions in calibrated
        # public units. No extra factor 100/180 occurs after the source loader.
        dtype = (torch.float64 if value.dtype == torch.float64 else torch.float32) if value.is_floating_point() else value.dtype
        if torch.device(device).type=="cuda" and value.device.type=="cpu" and value.numel()>=4096 and not value.is_pinned():
            value=value.pin_memory()
        return value.to(device=device, dtype=dtype, non_blocking=True)
    return value


def cache_disk_preflight(output: Path, endpoint_count: int, *, guard_bytes: int = 10 * 2**30):
    """Read-only capacity check against actual allocated blocks, including resumes."""
    output = Path(output).resolve()
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    expected = int(estimate_cache_bytes(int(endpoint_count)))
    cache = output / "observation-cache"
    allocated = 0
    if cache.exists():
        allocated = sum(path.stat().st_blocks * 512 for path in cache.rglob("*") if path.is_file())
    required = max(0, expected - allocated) + int(guard_bytes)
    free = int(shutil.disk_usage(ancestor).free)
    report = dict(cache_expected_bytes=expected, cache_allocated_bytes=allocated,
                  required_additional_with_guard_bytes=required, free_bytes=free,
                  activation_or_checkpoint_guard_bytes=int(guard_bytes),
                  filesystem=str(ancestor), enough_space=free >= required)
    if free < required:
        raise RuntimeError(f"V20.3 cache requires {required / 2**30:.2f} GiB additional including guard; "
                           f"only {free / 2**30:.2f} GiB available; no cache allocated")
    return report


class V203Pipeline:
    core_class = V203Core
    config_class = V203Config
    schedule_class = V203Schedule
    sources_class = V203Sources
    cache_class = RawObservationCache
    cache_builder = staticmethod(build_observation_cache)

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.output = Path(args.output).expanduser().resolve()
        self.frozen_base = Path(args.frozen_from).expanduser().resolve() if args.frozen_from else None
        self.observer_base = Path(args.observer_from).expanduser().resolve() if getattr(args,"observer_from",None) else None
        if self.observer_base is not None and self.frozen_base is not None:
            raise ValueError("choose perception-only import OR matched full frozen source, not both")
        if self.observer_base == self.output:
            raise ValueError("perception import requires a fresh output; preserve the old run")
        if self.frozen_base is not None:
            # A matched Actor source may itself reuse completed perception.
            # Keep the ARM/Actor checkpoint root separate from its cache root.
            import json
            frozen_config=json.loads((self.frozen_base/"config.json").read_text())
            if frozen_config.get("observer_from"):
                self.observer_base=Path(frozen_config["observer_from"]).expanduser().resolve()
        self.schedule = self.schedule_class(sigreg_weight=getattr(args,"sigreg_weight",.02))
        if args.smoke:
            # Smoke has its own fixed tiny schedule/output and can never be
            # resumed as formal evidence. It still reads actual source data.
            self.output = self.output / "smoke"
        elif args.stage != "preflight":
            require_gpu1(self.device)
        seed_all(self.schedule.seed)
        torch.set_float32_matmul_precision("high")
        self.core = self.core_class(self.config_class(actor_mode=args.actor_mode)).to(self.device)
        self.version = getattr(self.core, "CHECKPOINT_VERSION", VERSION)
        self.contract = getattr(self.core, "CHECKPOINT_CONTRACT", CONTRACT)
        self.sources = self.sources_class(args, device=self.device, decode_workers=args.decode_workers)
        from .prod_v20_1_acceleration import TensorRowBlockCache, GIB
        self.gpu_cache = (TensorRowBlockCache(self.device,int(getattr(args,"gpu_cache_gib",0)*GIB),
            workspace_reserve_bytes=int(getattr(args,"workspace_gib",24)*GIB))
            if self.device.type=="cuda" and getattr(args,"gpu_cache_gib",0)>0 else None)
        self.sources.device_cache=self.gpu_cache
        self.trainer = StageTrainer(self.core, self.output, self.device, self.schedule, microbatch=args.microbatch,
            clear_cache_callback=self._clear_gpu_cache,
            cache_stats_callback=self.gpu_cache.stats if self.gpu_cache else None)
        self.banks = {}
        self.fish_ids = self.sources.fish.training_frames
        self.run_metrics = {}

    def _clear_gpu_cache(self):
        if getattr(self, "actor_window_cache", None) is not None:
            self.actor_window_cache.clear()
        if self.gpu_cache is not None:
            self.gpu_cache.clear(release_allocator=True)

    def preflight(self):
        if self.observer_base is not None:
            import json
            source_config = json.loads((self.observer_base / "config.json").read_text())
            calibration = json.loads(Path(self.sources.paths.real_calibration).read_text())
            if (source_config.get("smoke") is not False
                    or source_config.get("sources") != asdict(self.sources.paths)
                    or source_config.get("real_projection_fits") != calibration.get("fits")):
                raise ValueError("perception reuse requires identical real data/calibration and a formal source")
            self._observer_payload()
        if self.frozen_base is not None:
            import json
            frozen_config = json.loads((self.frozen_base / "config.json").read_text())
            if bool(frozen_config.get("smoke")) != bool(self.args.smoke):
                raise ValueError("smoke and formal frozen sources cannot be mixed")
            current_calibration = json.loads(Path(self.sources.paths.real_calibration).read_text())
            if (frozen_config.get("sources") != asdict(self.sources.paths)
                    or frozen_config.get("real_projection_fits") != current_calibration.get("fits")):
                raise ValueError("matched frozen source cannot change data identities or camera projection fits")
        report = self.sources.preflight(media=self.args.audit_media)
        expected = dict(archive=320000, dense=20000, cad=190000, appearance=100000)
        for name, count in expected.items():
            if self.sources.counts[name] != count:
                raise RuntimeError(f"formal {name} population drift: {self.sources.counts[name]} != {count}")
        report.update(version=self.version, contract=self.contract, schedule=asdict(self.schedule),
                      actor_pair_exposures=self.schedule.actor_pair_exposures,
                      actor_mode=self.args.actor_mode, training_device=str(self.device),
                      smoke=bool(self.args.smoke), physical_commands_sent=False)
        report["cache_capacity"] = cache_disk_preflight(self.observer_base or self.frozen_base or self.output,
            self.sources.archive.endpoint_count + len(self.fish_ids))
        performance=dict(microbatch=self.args.microbatch,cache_batch=self.args.cache_batch,
            cache_cpu_batch=getattr(self.args,"cache_cpu_batch",1024),cache_host_features="fp16",
            gpu_cache=self.gpu_cache.stats() if self.gpu_cache else None,
            cache_cpu_prefetch=not getattr(self.args,"no_cache_prefetch",False),decode_workers=self.args.decode_workers,
            loss_denominators="whole-effective-batch, separate pair/center view counts")
        report["performance"]=performance
        write_json(self.output/"performance-config.json",performance)
        write_json(self.output / "data-audit.json", report)
        configuration = dict(version=self.version, contract=self.contract,
                             model=asdict(self.core.config), schedule=asdict(self.schedule),
                             sources=asdict(self.sources.paths), smoke=bool(self.args.smoke))
        configuration.update(getattr(self, "configuration_extra", lambda: {})())
        import json
        calibration = json.loads(Path(self.sources.paths.real_calibration).read_text())
        configuration["real_projection_fits"] = calibration["fits"]
        if self.frozen_base is not None:
            configuration["frozen_from"] = str(self.frozen_base)
        if self.observer_base is not None:
            configuration["observer_from"] = str(self.observer_base)
            configuration["transfer"] = "observer-only; fresh ARM/Actor; no optimizer/RNG import"
        config_path = self.output / "config.json"
        if config_path.exists():
            import json
            if json.loads(config_path.read_text()) != configuration:
                raise ValueError("output already belongs to a different V20.3 contract; choose a fresh directory")
        write_json(config_path, configuration)
        return report

    def _indices(self, count, passes, batch):
        if self.args.smoke:
            yield np.arange(min(count, min(batch, self.args.microbatch)), dtype=np.int64)
        else:
            yield from deterministic_batches(count, passes, batch, self.schedule.seed)

    def _total(self, count, passes, batch):
        return 1 if self.args.smoke else math.ceil(count / batch) * passes

    def _load(self, stage):
        if getattr(self,"observer_base",None) is not None and stage=="refit":
            from .prod_v20_3_transfer import import_observer_only
            payload=self._observer_payload()
            report=import_observer_only(self.core,payload,self.schedule)
            self._record("perception-reuse",report)
            return payload
        if self.frozen_base is not None and stage in ("refit", "lewm"):
            payload = torch.load(self.frozen_base / stage / "final.pt", map_location="cpu", weights_only=False)
            expected = asdict(self.core.config)
            actual = dict(payload.get("config", {}))
            if actual.get("actor_mode") not in ("full", "no_innovation"):
                raise ValueError("frozen source has an unknown Actor contract")
            actual["actor_mode"] = expected["actor_mode"]
            if (payload.get("version") != self.version or payload.get("contract") != self.contract
                    or payload.get("position_unit") != POSITION_UNIT
                    or actual != expected or payload.get("stage") != stage
                    or payload.get("schedule") != asdict(self.schedule) or not payload.get("metrics", {}).get("complete")):
                raise ValueError("frozen V20.3 source differs from the matched Actor experiment")
            # LeWM-final contains the identical, still-untrained Actor initial
            # state. Only the configured innovation input differs afterwards.
            self._validate_final(payload, stage)
            self.core.load_state_dict(payload["model"], strict=True)
            return payload
        payload = load_checkpoint(self.output / stage / "final.pt", self.core, stage=stage, schedule=self.schedule)
        self._validate_final(payload, stage)
        return payload

    def _observer_payload(self):
        from .prod_v20_3_transfer import validate_observer_checkpoint
        payload=torch.load(self.observer_base/"refit"/"final.pt",map_location="cpu",weights_only=False)
        return validate_observer_checkpoint(payload,self.core.config,self.schedule)

    def _validate_final(self, payload, stage):
        schedule = self.schedule
        batch = schedule.effective_batch
        expected = dict(appearance=math.ceil(schedule.appearance_endpoints / batch),
                        fcpg=math.ceil(schedule.cad_rows / batch) * schedule.cad_passes,
                        refit=schedule.real_updates,
                        lewm=math.ceil(schedule.archive_rows / (batch // 4)) * schedule.lewm_passes,
                        actor=math.ceil(schedule.archive_rows / (batch // 4)) * schedule.actor_passes,
                        dense=math.ceil(schedule.dense_rows / (batch // 4)) * schedule.dense_passes,
                        fish=schedule.fish_updates)[stage]
        if self.args.smoke:
            expected = 1
        if payload.get("step") != expected or payload.get("stage") != stage or not payload.get("metrics", {}).get("complete"):
            raise ValueError(f"{stage} final is incomplete or belongs to a smoke schedule")

    def _record(self, name, result):
        import json
        path = self.output / "metrics.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        result=dict(result)
        if self.gpu_cache is not None:
            result["gpu_cache"]=self.gpu_cache.stats()
        current[name] = result
        write_json(path, current)
        return result

    def train_fcpg(self):
        batch = self.schedule.effective_batch
        reused_cad = getattr(self, "reuse_cad_stage", lambda: False)()

        def appearance(indices):
            value = self.trainer.timed_load(lambda:to_device(self.sources.appearance_batch(indices),self.device))
            return self.core.observer.appearance_distillation(value["rgb"], value["dino"], value["view_valid"])

        if not reused_cad:
            self._record("appearance", self.trainer.run("appearance", self._indices(self.schedule.appearance_endpoints, 1, batch),
                appearance, total_steps=self._total(self.schedule.appearance_endpoints, 1, batch),
                sample_weights_fn=self.sources.appearance_weights))

        normalization={}
        def spatial_weights(indices,real=False):
            normalization.clear()
            normalization.update(self.sources.spatial_normalization(indices,real=real))
            return np.ones(len(indices),dtype=np.float64)
        def spatial(indices, real=False):
            value = self.trainer.timed_load(lambda:to_device(
                self.sources.real_batch(indices) if real else self.sources.cad_batch(indices),self.device))
            geometry = value.get("geometry")
            prior = geometry.make_prior(value["q_position"]) if geometry is not None else None
            if prior is not None and not real:
                # Pre-registered soft-prior perturbations, never image-label
                # shifts. The full image remains searchable on every update.
                from dataclasses import replace
                perturbed = prior.state_mean.clone()
                if getattr(self.core.observer, "analytic_wrist", False):
                    perturbation = torch.randn_like(perturbed[:, 1, :, :2]) * self.schedule.cad_prior_translation_std_px
                    limit = self.core.observer.max_column_residual_px
                    perturbed[:, 1, :, :2] += perturbation.clamp(-limit, limit)
                else:
                    perturbed[..., :2] += torch.randn_like(perturbed[..., :2]) * self.schedule.cad_prior_translation_std_px
                prior = replace(prior, state_mean=perturbed)
            if hasattr(self.core.observer, "training_observation"):
                observation = self.core.observer.training_observation(
                    value["rgb"], value["dino"], value["view_valid"], prior=prior,
                    q_position=value["q_position"], q_available=value["q_available"],
                    actual_history=value.get("actual_history"))
            else:
                observation = self.core.observer(value["rgb"], value["dino"], value["view_valid"], prior)
            return self.core.observer.spatial_loss(observation, value["target_fm_px"], value["label_valid"],
                            value.get("ordered", True), visibility=value.get("visibility"),
                            visibility_valid=value.get("visibility_valid"),normalization=normalization)

        if not reused_cad:
            self._record("fcpg", self.trainer.run("fcpg", self._indices(self.schedule.cad_rows, self.schedule.cad_passes, batch),
                spatial, total_steps=self._total(self.schedule.cad_rows, self.schedule.cad_passes, batch),
                sample_weights_fn=spatial_weights))
        updates = 1 if self.args.smoke else self.schedule.real_updates
        # Deterministic update identities seed natural within-view sampling;
        # the source method never upweights rare aperture/span bins.
        real_batch = min(batch, self.args.microbatch) if self.args.smoke else batch
        real_batches = (np.arange(i * real_batch, (i + 1) * real_batch, dtype=np.int64) for i in range(updates))
        self._record("refit", self.trainer.run("refit", real_batches, lambda ids: spatial(ids, True), total_steps=updates,
            sample_weights_fn=lambda ids:spatial_weights(ids,True)))

    def _cache_identity(self):
        payload = self._load("refit")
        return f"{payload['run_id']}-seed{self.schedule.seed}-refit-step{payload['step']}"

    def cache_observations(self):
        if getattr(self,"observer_base",None) is not None:
            # Reuse is read-only. A separate old-contract cache-only process
            # may finish interrupted writes; never race it or re-label them.
            self._load("refit")
            self._open_banks()
            for name,bank in self.banks.items():
                if not bool(bank.written.all()):
                    raise ValueError(f"{name} perception source cache is incomplete; finish cache-only first")
            return self._record("cache",dict(reused_from=str(self.observer_base),complete=True))
        self._clear_gpu_cache()
        # Sequential construction has no row reuse; prefetch pure CPU reads
        # instead. The shared mutable GPU LRU stays in the main thread only.
        previous_device_cache = self.sources.device_cache
        previous_compact_features = getattr(self.sources,"compact_host_features",False)
        self.sources.device_cache=None
        self.sources.compact_host_features=True
        try:
            self.sources.fish_geometry()
            return self._cache_observations()
        finally:
            self.sources.device_cache=previous_device_cache
            self.sources.compact_host_features=previous_compact_features

    def _cache_observations(self):
        binding = self._cache_identity()
        for kind, count in (("archive", self.sources.archive.endpoint_count), ("fish", len(self.fish_ids))):
            cache = self.cache_class(self.output / "observation-cache" / kind, count, binding, create=True)
            if self.args.smoke:
                # Only the required chain prefix is measured; never mark a
                # partial smoke bank as a complete 320K observation cache.
                needed = self.sources.chain_batch(np.arange(min(self.args.microbatch, 2)), kind)["endpoint_index"].reshape(-1)
                ids = np.unique(needed)
                addresses = ids if kind == "archive" else np.searchsorted(self.fish_ids, ids)
                for begin in range(0, len(ids), self.args.cache_batch):
                    source_ids = ids[begin:begin + self.args.cache_batch]
                    value = to_device(self.sources.observation_batch(source_ids, kind), self.device)
                    with torch.no_grad():
                        geometry = value.get("geometry")
                        prior = geometry.make_prior(value["q_position"]) if geometry is not None else None
                        observation = self.core.observer.measurement(value["rgb"], value["dino"], value["view_valid"], mechanical_prior=prior)
                    cache.put(addresses[begin:begin + len(source_ids)], observation)
            else:
                batches = (np.arange(begin, min(begin + self.args.cache_batch, count)) for begin in range(0, count, self.args.cache_batch))
                def loader(ids, selected_kind=kind):
                    source_ids = ids if selected_kind == "archive" else self.fish_ids[ids]
                    return self.sources.observation_batch(source_ids, selected_kind)
                self.cache_builder(self.core.observer, cache, batches, loader, device=self.device,
                    prefetch=not getattr(self.args,"no_cache_prefetch",False),
                    cpu_superbatch=max(self.args.cache_batch,getattr(self.args,"cache_cpu_batch",1024)),
                    raw_measurement=True)
            self.banks[kind] = cache
        self._record("observation-cache", dict(complete=not self.args.smoke, measurement_path=self.cache_class.measurement_path,
                                               archive_endpoints=self.sources.archive.endpoint_count,
                                               fish_training_frames=len(self.fish_ids), episode_2_opened=False))

    def _open_banks(self):
        if self.banks:
            return
        # Reading refit identity must not replace a loaded LeWM/Actor with the
        # old all-module snapshot. Inspect only ordinary metadata here.
        base = getattr(self,"observer_base",None) or self.frozen_base or self.output
        payload = torch.load(base / "refit" / "final.pt", map_location="cpu", weights_only=False)
        expected = asdict(self.core.config)
        actual = dict(payload.get("config", {}))
        if self.frozen_base is not None and actual.get("actor_mode") in ("full", "no_innovation"):
            actual["actor_mode"] = expected["actor_mode"]
        if getattr(self,"observer_base",None) is not None:
            from .prod_v20_3_transfer import validate_observer_checkpoint
            validate_observer_checkpoint(payload,self.core.config,self.schedule)
        elif (payload.get("version") != self.version or payload.get("contract") != self.contract
                or payload.get("position_unit") != POSITION_UNIT or actual != expected
                or payload.get("schedule") != asdict(self.schedule)):
            raise ValueError("observation cache refit metadata differs from the V20.3 model/data contract")
        if getattr(self,"observer_base",None) is None:
            self._validate_final(payload, "refit")
        binding = f"{payload['run_id']}-seed{self.schedule.seed}-refit-step{payload['step']}"
        self.banks = {
            "archive": self.cache_class(base / "observation-cache" / "archive", self.sources.archive.endpoint_count, binding),
            "fish": self.cache_class(base / "observation-cache" / "fish", len(self.fish_ids), binding),
        }
        if getattr(self,"observer_base",None) is not None:
            for name,bank in self.banks.items():
                if not bool(bank.written.all()):
                    raise ValueError(f"{name} perception source cache is incomplete; finish cache-only first")

    def chain(self, rows, kind="archive", *, include_final_image=False):
        return self.trainer.timed_load(self._chain,rows,kind,include_final_image=include_final_image)

    def _chain(self, rows, kind="archive", *, include_final_image=False):
        value = self.sources.chain_batch(rows, kind)
        ids = value["endpoint_index"]
        if not include_final_image:
            # Actor makes H0..H3 decisions. H4 q/readback remains supervised,
            # but its image is never read by this loss. Evaluation keeps H4.
            ids = ids[:, :4]
        observations = []
        bank_kind = "archive" if kind == "dense" else kind
        if not self.args.smoke:
            # One endpoint-indexed read/validation/pool over all five frames,
            # then restore the exact original chain/horizon order.
            endpoints=ids.reshape(-1)
            if bank_kind == "archive" and getattr(self, "actor_window_cache", None) is not None:
                from .prod_v20_3_actor_io import observation_from_arrays
                staged = self.actor_window_cache.get_arrays(endpoints)
                measured = observation_from_arrays(staged, self.core.observer)
            else:
                features=to_device(self.sources.dino_batch(endpoints,bank_kind),self.device)
                addresses=endpoints if bank_kind=="archive" else np.searchsorted(self.fish_ids,endpoints)
                measured=self.banks[bank_kind].get(addresses,features["dino"],device=self.device,
                    observer=self.core.observer,view_valid=features["view_valid"],device_cache=self.gpu_cache)
            for horizon in range(ids.shape[1]):
                selected=torch.arange(len(ids),device=self.device)*ids.shape[1]+horizon
                belief=type(measured.belief)(**{field.name:getattr(measured.belief,field.name).index_select(0,selected)
                                               for field in fields(measured.belief)})
                observations.append(replace(measured,belief=belief,
                    **{name:getattr(measured,name).index_select(0,selected) for name in
                       ("component_features","scene_tokens","coarse_center_log_posterior","pooling_map")}))
            result=to_device(value,self.device)
            result["measurements"]=observations
            return result
        for horizon in range(ids.shape[1]):
            endpoints = ids[:, horizon]
            features = to_device(self.sources.dino_batch(endpoints, bank_kind), self.device)
            addresses = endpoints if bank_kind == "archive" else np.searchsorted(self.fish_ids, endpoints)
            if self.args.smoke and not self.banks[bank_kind].written[addresses].all():
                # In a real-data smoke all needed observations still run the
                # same frozen image observer. No legacy fallback is allowed.
                images = to_device(self.sources.observation_batch(endpoints, bank_kind), self.device)
                with torch.no_grad():
                    geometry = images.get("geometry")
                    prior = geometry.make_prior(images["q_position"]) if geometry is not None else None
                    measured = self.core.observer.measurement(images["rgb"], images["dino"], images["view_valid"], mechanical_prior=prior)
                # Smoke may open read-only banks after restarting a stage;
                # use the just-measured tensor directly in that case.
                observations.append(measured)
            else:
                observations.append(self.banks[bank_kind].get(addresses, features["dino"], device=self.device,
                    observer=self.core.observer, view_valid=features["view_valid"],device_cache=self.gpu_cache))
        result = to_device(value, self.device)
        result["measurements"] = observations
        return result

    def _scales(self):
        from .prod_v17_8_training import compute_joint_delta_loss_scale
        scale = torch.as_tensor(compute_joint_delta_loss_scale(self.sources.archive.archive), device=self.device)
        self.core.q_loss_scale.copy_(scale)
        self.core.lewm.increment_scale.copy_(scale)
        self.core.actor.increment_scale.copy_(scale)
        self._record("joint-normalization", dict(scale=scale.cpu().tolist(), source="all-320K-factual-command-increments",
                                                output_step_cap=False))

    def train_lewm(self):
        from .prod_v20_3_acceleration import ResidentMotionBank
        from .prod_v20_3_motion_training import motion_chain_loss
        self._load("refit")
        self._scales()
        self._clear_gpu_cache()
        motion_count = min(self.schedule.archive_rows, max(2, self.args.microbatch)) if self.args.smoke else self.schedule.archive_rows
        motion = ResidentMotionBank(self.sources.motion_chain_batch, motion_count,
                                    device=self.device, max_bytes=1024**3, block_rows=8192)
        self._record("motion-preload", motion.stats())
        seed_all(self.schedule.seed + 200)
        batch = self.schedule.effective_batch // 4
        try:
            self._record("lewm", self.trainer.run("lewm", self._indices(self.schedule.archive_rows, self.schedule.lewm_passes, batch),
                lambda rows: motion_chain_loss(self.core, self.trainer.timed_load(motion.get, rows),
                    schedule=self.schedule, return_parts=True),
                total_steps=self._total(self.schedule.archive_rows, self.schedule.lewm_passes, batch), pair_multiplier=4))
        finally:
            del motion

    def _actor_input_contract(self):
        import json
        name = getattr(self.args, "actor_io", "window_lru")
        contract = dict(recipe=name, seed=self.schedule.seed,
                        block_rows=2048, blocks_per_window=8, chain_batch=self.schedule.effective_batch//4,
                        horizons=4, passes=self.schedule.actor_passes, host_pool_max_gib=16,
                        prefetch_windows=1, changes_training_order=name == "window_lru",
                        samples_dropped=False, frozen_features_requantized=False)
        path = self.output / "actor-input-config.json"
        if path.exists():
            if json.loads(path.read_text()) != contract:
                raise ValueError("Actor input order differs from saved run; cannot silently resume")
        elif any((self.output/stage/file).exists() for stage in ("actor", "dense", "fish")
                 for file in ("latest.pt", "final.pt")) and name != "random_lru":
            raise ValueError("existing Actor has legacy random order; use --actor-io random_lru to resume")
        if self.frozen_base is not None:
            source_contract = self.frozen_base / "actor-input-config.json"
            if source_contract.exists() and json.loads(source_contract.read_text()) != contract:
                raise ValueError("matched Actor control must use the source sampling contract")
        write_json(path, contract)
        return contract

    def _actor_window_indices(self, stage, kind, count, passes, batch):
        """One prefetched CPU pool, main-thread window LRU, exact OOM replay.

        The mixed-block sampler intentionally changes epoch order. Every chain
        is still paired with its own q/task/images and visited once per pass. Fully resumed windows
        yield their indices without any disk prefetch or GPU allocation.
        """
        from concurrent.futures import ThreadPoolExecutor
        from .prod_v20_3_actor_sampling import deterministic_mixed_locality_windows
        from .prod_v20_3_actor_io import ActorEndpointReader, ActorWindowLRU, ENDPOINT_BYTES
        reader = ActorEndpointReader(self.sources, self.banks["archive"], workers=4)
        windows = []
        def bounded(rows):
            if len(reader.endpoint_ids(rows, kind))*ENDPOINT_BYTES <= reader.max_host_bytes:
                yield rows
                return
            middle = (len(rows)//2//batch)*batch
            if middle == 0:
                raise MemoryError("one Actor batch exceeds the host staging budget")
            yield from bounded(rows[:middle])
            yield from bounded(rows[middle:])
        for rows in deterministic_mixed_locality_windows(count, passes, batch, self.schedule.seed):
            windows.extend(bounded(rows))
        resume = 0
        latest = self.output / stage / "latest.pt"
        if latest.exists():
            # Trainer performs the full strict load. This read-only ordinal is
            # solely to avoid materializing already consumed input windows.
            resume = int(torch.load(latest, map_location="cpu", weights_only=False, mmap=True)["step"])
        self._clear_gpu_cache()
        cache = ActorWindowLRU(self.device, int(getattr(self.args, "gpu_cache_gib",48)*2**30),
                               workspace_bytes=int(getattr(self.args,"workspace_gib",24)*2**30))
        self.actor_window_cache = cache
        old_stats = self.trainer.cache_stats_callback
        self.trainer.cache_stats_callback = cache.stats
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="actor-next-window")
        future = None
        ordinal = 0
        try:
            for index, rows in enumerate(windows):
                batches = math.ceil(len(rows)/batch)
                if ordinal+batches <= resume:
                    ordinal += batches
                    for begin in range(0,len(rows),batch):
                        yield rows[begin:begin+batch]
                    continue
                if future is None:
                    future = pool.submit(reader.load, rows, kind)
                host = self.trainer.timed_load(future.result)
                self.trainer.timed_load(cache.activate, (stage,index), host)
                del host
                # The just-uploaded host pool is kept for OOM replay. Exactly
                # one other <=16 GiB CPU pool can be in flight; never full data.
                future = pool.submit(reader.load, windows[index+1], kind) if index+1 < len(windows) else None
                for begin in range(0,len(rows),batch):
                    yield rows[begin:begin+batch]
                ordinal += batches
        finally:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
                self._record(stage+"-input-io", cache.stats())
            finally:
                cache.close()
                self.actor_window_cache = None
                self.trainer.cache_stats_callback = old_stats

    def train_actor(self):
        self._load("lewm")
        self._open_banks()
        input_contract = self._actor_input_contract()
        # Matched full/no-innovation starts share weights AND stochastic state;
        # a latest checkpoint still restores its own continuation RNG below.
        seed_all(self.schedule.seed + 300)
        batch = self.schedule.effective_batch // 4
        def indices(stage, kind, count, passes):
            if self.args.smoke or input_contract["recipe"] == "random_lru":
                return self._indices(count, passes, batch)
            return self._actor_window_indices(stage, kind, count, passes, batch)
        # Explicit close is required on loss/optimizer failure: a retained
        # traceback may otherwise retain the generator, prefetch thread and
        # host pool indefinitely. Never rely on refcount/GC cleanup.
        with closing(indices("actor", "archive", self.schedule.archive_rows, self.schedule.actor_passes)) as batches:
            self._record("actor", self.trainer.run("actor", batches,
                lambda rows: chain_loss(self.core, self.chain(rows), stage="actor"),
                total_steps=self._total(self.schedule.archive_rows, self.schedule.actor_passes, batch), pair_multiplier=4))
        with closing(indices("dense", "dense", self.schedule.dense_rows, self.schedule.dense_passes)) as batches:
            self._record("dense", self.trainer.run("dense", batches,
                lambda rows: chain_loss(self.core, self.chain(rows, "dense"), stage="actor"),
                total_steps=self._total(self.schedule.dense_rows, self.schedule.dense_passes, batch), pair_multiplier=4))
        updates = 1 if self.args.smoke else self.schedule.fish_updates
        mixed_chains = 2 if self.args.smoke else self.schedule.fish_batch // 4
        rng = np.random.default_rng(self.schedule.seed + 7)
        selections = (np.concatenate((rng.integers(self.sources.counts["fish"], size=mixed_chains // 2),
                     rng.integers(self.sources.counts["archive"], size=mixed_chains // 2) + self.sources.counts["fish"]))
                     for _ in range(updates))

        def mixed_loss(ids):
            fish = ids < self.sources.counts["fish"]
            loss = next(self.core.actor.parameters()).sum() * 0
            if fish.any():
                loss = loss + chain_loss(self.core, self.chain(ids[fish], "fish"), stage="actor") * (fish.sum() / len(ids))
            if (~fish).any():
                rows = ids[~fish] - self.sources.counts["fish"]
                loss = loss + chain_loss(self.core, self.chain(rows), stage="actor") * ((~fish).sum() / len(ids))
            return loss

        self._record("fish", self.trainer.run("fish", selections, mixed_loss, total_steps=updates,
                                              learning_rate=self.schedule.fish_learning_rate, pair_multiplier=4))

    def evaluate(self):
        from .prod_v20_3_evaluation import evaluate_chain, observer_metrics, fit_servo_lag
        self._load("fish")
        self._open_banks()
        self.core.eval().requires_grad_(False)
        manifest = self.sources.development_manifest(self.args.evaluation_rows)
        selected = np.asarray([record["row"] for record in manifest["records"]], dtype=np.int64)
        evaluation_groups = development_batches(selected)
        manifest["shuffle_protocol"] = dict(group_rows=16, singleton_tail="join previous",
            independent_of_training_microbatch=True,
            groups=[dict(rows=rows.tolist(), donor_rows=np.roll(rows, 1).tolist()) for rows in evaluation_groups])
        write_json(self.output / "evaluation" / "development-manifest.json", manifest)
        prefix = self.sources.servo_training_prefix(selected, count=getattr(self.args, "servo_fit_rows", 256))
        fit = {"available": False, "reason": "no disjoint earlier training prefix"}
        if prefix["rows"]:
            # Read only positions/timestamps: no image bank or future pixel
            # target is needed to fit this plain actuator baseline.
            training = to_device(self.sources.chain_batch(np.asarray(prefix["rows"])), self.device)
            training["evidence_role"] = prefix["role"]
            fit = fit_servo_lag(training, evaluation_endpoint_ids=self.sources.archive.row_endpoint_index[selected])
        reports = []
        for rows in evaluation_groups:
            reports.append(evaluate_chain(self.core, self.chain(rows, include_final_image=True), servo_lag_fit=fit))
        metrics = merge_sum_count_metrics([part["metrics"] for part in reports])
        report = dict(evidence="training-development", unseen_claim=False, manifest=manifest,
                      servo_training_prefix=prefix, servo_lag_fit=fit, metrics=metrics,
                      batch_protocols=[part["protocol"] for part in reports],
                      actual_localization_geometry_covered=any(part["actual_localization_geometry_available"] for part in reports),
                      held_episode_opened=False, robot_commands_sent=False)
        report["occlusion_probe"] = self._occlusion_probe_report()
        report["labeled_training_fit"] = {}
        # These labels have already been used. Report their two views
        # separately and never present the fit as new real-scene accuracy.
        from .prod_v20_3_independent_endpoint import independent_endpoint_fusion
        real_count = min(2, self.sources.counts["real"]) if self.args.smoke else self.sources.counts["real"]
        for begin in range(0, real_count, self.args.microbatch):
            ids = np.arange(begin, min(begin + self.args.microbatch, real_count))
            value = to_device(self.sources.real_evaluation_batch(ids), self.device)
            with torch.no_grad():
                prior = value["geometry"].make_prior(value["q_position"])
                raw = self.core.observer.measurement(value["rgb"], value["dino"], value["view_valid"], mechanical_prior=prior)
                if hasattr(self.core.observer, "training_observation"):
                    filtered = self.core.observer.training_observation(
                        value["rgb"], value["dino"], value["view_valid"], prior=prior,
                        q_position=value["q_position"], q_available=value["q_available"])
                else:
                    filtered = self.core.observer.assimilate(raw, prior, value["view_valid"])
                try:
                    with independent_endpoint_fusion(self.core):
                        independent = self.core.observer.assimilate(raw, prior, value["view_valid"])
                except NotImplementedError as error:
                    # V20.3.1 overrides assimilation; the matched row is defined for V20.3 only.
                    independent = None
                    report.setdefault("labeled_training_fit_unavailable", {})["h0_posterior_independent_endpoint_known_geometry"] = str(error)
            from dataclasses import replace
            comparison = {"mechanical_prior_known_geometry": replace(raw, belief=prior),
                          "current_measurement": raw, "h0_posterior": filtered,
                          "current_measurement_known_geometry": raw, "h0_posterior_known_geometry": filtered}
            if getattr(self.core.observer, "raw_feature_cache", False):
                # Raw V20.3.2 cache entries contain a neutral schema placeholder,
                # not a predicted spatial measurement. Never score it as one.
                for name in ("current_measurement", "current_measurement_known_geometry"):
                    comparison.pop(name)
                report.setdefault("labeled_training_fit_unavailable", {})["current_measurement"] = (
                    "V20.3.2 spatial output requires explicit current q/prior; raw cache stores image fields only")
            if independent is not None:
                comparison["h0_posterior_independent_endpoint_known_geometry"] = independent
            report["labeled_training_fit"][str(begin)] = {
                "rows": ids.tolist(), "camera_available": value["camera_available"].cpu().tolist(),
                "conditions": {name: observer_metrics(observed, value["target_fm_px"],
                               value["point_valid"] & value["camera_available"][..., None] if name.endswith("known_geometry") else value["point_valid"], value["ordered"])
                               for name, observed in comparison.items()}}
        from .prod_v20_3_headline import summarize_development_headline
        report["headline"] = summarize_development_headline(report)
        write_json(self.output / "evaluation" / "development.json", report)
        write_json(self.output / "evaluation" / "headline.json", report["headline"])
        return report

    def _occlusion_probe_report(self):
        """Fish-chain occlusion persistence; skip if no rows or no camera geometry."""
        requested = getattr(self.args, "occlusion_rows", 0)
        fish = int(self.sources.counts.get("fish", 0))
        if requested < 0:
            raise ValueError("occlusion_rows must be non-negative")
        if requested == 0:
            return dict(available=False, reason="occlusion_rows is 0", evidence="training-development",
                        unseen_claim=False, held_episode_opened=False, robot_commands_sent=False)
        if fish == 0:
            return dict(available=False, reason="no fish chains with camera geometry",
                        evidence="training-development", unseen_claim=False,
                        held_episode_opened=False, robot_commands_sent=False)
        from .prod_v20_3_occlusion_probe import occlusion_probe
        count = min(requested, fish, 2 if self.args.smoke else requested)
        # Evenly spaced starts cover every training episode's timeline with the
        # least window overlap; a random draw from three episodes adds nothing.
        rows = np.unique(np.round(np.linspace(0, fish - 1, count)).astype(np.int64))
        identity = fish_row_identity(getattr(self.sources, "fish", None), rows)
        parts, per_row, schedules, estimators, protocol, geometry_kind, metric_geometry, labeled = [], {}, None, None, None, None, None, {}
        with torch.no_grad():
            for begin in range(0, len(rows), self.args.microbatch):
                part = occlusion_probe(self.core, self.chain(rows[begin:begin + self.args.microbatch], "fish",
                                                            include_final_image=True))
                parts.append(part["metrics"])
                for name, values in part.get("per_row", {}).items():
                    per_row.setdefault(name, []).extend(values)
                if schedules is None:
                    schedules, estimators, protocol = part["schedules"], part["estimators"], part["protocol"]
                    geometry_kind, metric_geometry = part["geometry_kind"], part["metric_geometry"]
                if part.get("labeled"):
                    labeled.update(part["labeled"])
        metrics = merge_sum_count_metrics(parts)
        result = dict(available=True, evidence="training-development", unseen_claim=False,
                      rows=rows.tolist(), chain_rows=len(rows), row_identity=identity,
                      geometry_kind=geometry_kind, metric_geometry=metric_geometry,
                      estimators=estimators, schedules=schedules,
                      metrics=metrics, per_row=per_row, labeled=labeled, protocol=protocol,
                      reference=protocol["reference"], summary=occlusion_probe_summary(metrics),
                      held_episode_opened=False, robot_commands_sent=False)
        path = self.output / "evaluation" / "occlusion-probe.json"
        write_json(path, result)
        result["file"] = str(path)
        return result

    def run(self, stage):
        if self.frozen_base is not None and stage not in ("actor", "evaluate", "preflight"):
            raise ValueError("shared frozen source is only valid for Actor training or development evaluation")
        if stage == "preflight":
            return self.preflight()
        self.preflight()
        if stage == "all":
            if self.observer_base is None:
                self.train_fcpg()
            self.cache_observations()
            self.train_lewm()
            self.train_actor()
            return {"training_complete": True, "held_evaluation_opened": False, "output": str(self.output)}
        operations = dict(fcpg=self.train_fcpg, cache=self.cache_observations, lewm=self.train_lewm,
                          actor=self.train_actor, evaluate=self.evaluate)
        return operations[stage]()

    def close(self):
        self._clear_gpu_cache()
        self.sources.close()
