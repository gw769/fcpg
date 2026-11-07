#!/usr/bin/env python3
"""Random-initialized FCPG/Actor/LeWM, original TRAIN splits, FP32 training.

The frozen DINO cache supplies real recorded visual features. The only model
checkpoint reads are stage continuation/resume inside this new output directory.
CPU is the default; CUDA requires one explicitly visible GPU and an allocator cap.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpg.v21_core import V21Config, V21ResearchCore, VERSION, CONTRACT
from fcpg.v21_data import TrainingChain, V21Sources
from fcpg.v21_training import V21Schedule, V21Trainer, write_json
from fcpg.v21_temporal_contract import PROGRESS_TRAINING_POLICY, temporal_training_configuration
from fcpg.prod_v17_5 import restore_companion_joint_units
from fcpg.prod_v20_3_actor_sampling import deterministic_mixed_locality_windows
from scripts.train_so101_v21_research import measurement_dropout, dropout_contract
from scripts.build_v21_body_object_table import validate_manifest
from scripts.v21_source_map import resolve_source_paths, pin_source_map

STAGES = ('lewm', 'actor', 'dense', 'fish')
RECIPE = 'v21-original-train-from-scratch-cpu-v1'
GPU_RECIPE = 'v21-original-train-from-scratch-cuda-fp32-v1'
DATA_CONTRACT = dict(schema='v21-original-source-split-v1', archive_train_rows=272000,
    archive_development_rows=48000, dense_train_rows=17000, dense_development_rows=3000,
    fish_train_episodes=[0, 1], fish_development_episodes=[3], fish_test_episode_used=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configure_device(name='cpu', gpu_memory_limit_gib=8.0):
    """Select only one admitted GPU; bound this process before tensor allocation."""
    require(name in ('cpu', 'cuda:0'), 'scratch device must be cpu or logical cuda:0')
    device = torch.device(name)
    if device.type == 'cpu':
        return device, {}
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(re.fullmatch(r'(?:[0-9]+|GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})', visible)
        is not None, 'CUDA_VISIBLE_DEVICES must identify exactly one explicit GPU index or full GPU UUID')
    require(math.isfinite(gpu_memory_limit_gib) and gpu_memory_limit_gib > 0,
        'GPU allocator limit must be finite and positive')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
        'CUDA training requires exactly one visible available GPU')
    total = int(torch.cuda.get_device_properties(device).total_memory)
    limit = float(gpu_memory_limit_gib)*1024**3
    require(total > 0 and limit <= total, 'GPU allocator limit exceeds the selected device memory')
    fraction = limit/total
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device, dict(cuda_visible_devices=visible, gpu_memory_limit_gib=float(gpu_memory_limit_gib),
        gpu_memory_fraction=fraction, gpu_total_memory_bytes=total, tf32_enabled=False)


def stage_chain_to_device(chain, device):
    """Move an already prepared CPU chain without changing values or dtypes."""
    require(isinstance(chain,TrainingChain), 'CPU staging requires a complete TrainingChain')
    device = torch.device(device)
    require(device in (torch.device('cpu'),torch.device('cuda:0')),
        'staged chain device must be cpu or logical cuda:0')
    def move(value):
        require(isinstance(value,torch.Tensor) and value.device.type == 'cpu',
            'staging requires all source chain tensors to be prepared on CPU')
        return value.to(device=device,non_blocking=True)
    values = {}
    for field in fields(chain):
        value = getattr(chain,field.name)
        if field.name == 'traces':
            values[field.name] = tuple(replace(trace,positions=move(trace.positions),
                offsets=move(trace.offsets),valid=move(trace.valid)) for trace in value)
        else:
            values[field.name] = None if value is None else move(value)
    return replace(chain,**values)


class DeviceStagedSources:
    """Scratch-only CPU sampling/validation followed by whole-chain transfers."""
    def __init__(self, host_source, device, feature_bank=None):
        require(torch.device(host_source.device).type == 'cpu', 'staged source must prepare chains on CPU')
        self.host_source, self.device = host_source, torch.device(device)
        self.feature_bank = feature_bank

    def __getattr__(self, name):
        return getattr(self.host_source,name)

    def chain(self, rows, kind='archive', *, images=True):
        cached = images and self.feature_bank is not None
        host = self.host_source.chain(rows,kind,images=images and not cached)
        features = None
        if cached:
            features, checked = self.feature_bank.load(rows,kind)
            require(np.array_equal(checked,host.view_valid.numpy()),
                'cached DINO and current chain view identities differ')
            require(features.device == self.device and features.dtype == torch.float32,
                'feature bank must preserve FP32 features on the selected device')
        chain = stage_chain_to_device(host,self.device)
        return replace(chain,features=features) if cached else chain


def derive_splits(source):
    """Restore original split IDs; never use the legacy all-row training alias."""
    split = np.asarray(source.archive.archive.split)
    require(split.shape == (source.counts['archive'],) and np.isin(split, [0, 1]).all(),
        'original archive split must contain only declared TRAIN/DEV identities')
    dense = np.asarray(source.dense_rows)
    require(dense.ndim == 1 and dense.dtype.kind in 'iu' and len(dense) == source.counts['dense']
        and len(np.unique(dense)) == len(dense) and ((dense >= 0)&(dense < len(split))).all(),
        'dense ordinal-to-archive identity differs')
    endpoint = np.asarray(source.archive.row_endpoint_index)
    episode = np.asarray(source.archive.endpoint_episode_index)[endpoint]
    require(episode.shape == (len(split), 5) and (episode == episode[:, :1]).all(),
        'archive chain crosses episode')
    require(not np.intersect1d(episode[split == 0, 0], episode[split == 1, 0]).size,
        'original archive TRAIN/DEV episodes overlap')
    fish_starts = np.asarray(source.fish.starts)
    fish_endpoints = fish_starts[:, None]+np.arange(5, dtype=np.int64)[None]*15
    source.fish.validate_frames(fish_endpoints.reshape(-1))
    fish_episodes = np.asarray(source.fish.episodes[fish_endpoints])
    require(fish_episodes.shape == (source.counts['fish'], 5)
        and (fish_episodes == fish_episodes[:, :1]).all() and np.isin(fish_episodes, [0, 1, 3]).all(),
        'fish chain episode identities differ or include sealed TEST')
    train = dict(archive=np.flatnonzero(split == 0), dense=dense[split[dense] == 0],
        fish=np.flatnonzero(np.isin(fish_episodes[:, 0], [0, 1])))
    development = dict(archive=np.flatnonzero(split == 1), dense=dense[split[dense] == 1],
        fish=np.flatnonzero(fish_episodes[:, 0] == 3))
    # the half-second Archive has no 20K dense subset (rgb_root null): its dense population is empty on both sides
    require(all(len(train[k]) and len(development[k]) for k in ('archive', 'fish'))
        and (not len(dense) or (len(train['dense']) and len(development['dense']))),
        'original TRAIN and DEV populations must both be nonempty')
    return dict(train=train, development=development,
        fish_required_endpoints=np.unique(fish_endpoints),
        dense_train_ordinals=np.flatnonzero(split[dense] == 0),
        dense_development_ordinals=np.flatnonzero(split[dense] == 1))


def training_loss_scale(source, archive_rows, chunk_rows=4096):
    """Original float64 population std, with only explicitly admitted TRAIN rows."""
    rows = np.asarray(archive_rows)
    archive = source.archive.archive
    require(rows.ndim == 1 and rows.dtype.kind in 'iu' and len(rows) > 0
        and len(np.unique(rows)) == len(rows) and ((rows >= 0)&(rows < archive.rows)).all()
        and (np.asarray(archive.split)[rows] == 0).all() and chunk_rows > 0,
        'loss normalization requires unique original TRAIN rows')
    total = 0
    summed, squared = np.zeros(6, np.float64), np.zeros(6, np.float64)
    for start in range(0, len(rows), chunk_rows):
        selected = rows[start:start+chunk_rows]
        commands = restore_companion_joint_units(np.asarray(archive.completed_commands[selected], np.float32)).astype(np.float64)
        q = restore_companion_joint_units(np.asarray(archive.measured_state_path[selected, :, 0], np.float32)).astype(np.float64)
        require(commands.shape == (len(selected), 4, 5, 6) and q.shape == (len(selected), 4, 6),
            'recorded normalization commands/source q shape differs')
        previous = np.concatenate((q[:, :, None], commands[:, :, :-1]), axis=2)
        delta = (commands-previous).reshape(-1, 6)
        require(np.isfinite(delta).all(), 'nonfinite factual TRAIN increments')
        summed += delta.sum(0, dtype=np.float64)
        squared += np.square(delta).sum(0, dtype=np.float64)
        total += len(delta)
    require(total == len(rows)*4*5, 'TRAIN scale exposure count differs')
    variance = np.maximum(squared/total-np.square(summed/total), 0.)
    scale = np.sqrt(variance).astype(np.float32)
    require(np.isfinite(scale).all() and (scale > 1e-6).all(), 'TRAIN joint delta scale is degenerate')
    return scale


def stage_steps(stage, populations, schedule, *, canary=False):
    require(stage in STAGES, 'unknown stage')
    if canary:
        return 2
    if stage == 'fish':
        return schedule.fish_updates
    kind = 'dense' if stage == 'dense' else 'archive'
    passes = schedule.dense_passes if stage == 'dense' else schedule.archive_passes
    return math.ceil(len(populations[kind])/schedule.chains_per_batch)*passes


def stage_requests(stage, populations, schedule, *, canary=False):
    """Sample filtered ordinals, then map to original identities before IO."""
    require(stage in STAGES, 'unknown stage')
    limit = stage_steps(stage, populations, schedule, canary=canary)
    if stage == 'fish':
        require(schedule.fish_chains_per_batch >= 2 and schedule.fish_chains_per_batch % 2 == 0,
            'fish stage needs an even mixed batch')
        rng = np.random.default_rng(schedule.seed+7)
        half = schedule.fish_chains_per_batch//2
        for update in range(limit):
            groups = [(kind, populations[kind][rng.integers(len(populations[kind]), size=half)])
                for kind in ('fish', 'archive')]
            yield dict(stage=stage, variant=update, groups=groups)
        return
    kind = 'dense' if stage == 'dense' else 'archive'
    passes = schedule.dense_passes if stage == 'dense' else schedule.archive_passes
    count, batch = len(populations[kind]), schedule.chains_per_batch
    if canary:
        passes = max(passes, math.ceil(2/math.ceil(count/batch)))
    produced = 0
    for window_id, window in enumerate(deterministic_mixed_locality_windows(count, passes, batch,
            schedule.seed, block_rows=512, blocks_per_window=4)):
        for begin in range(0, len(window), batch):
            if produced == limit:
                return
            rows = populations[kind][window[begin:begin+batch]]
            yield dict(stage=stage, variant=window_id, groups=[('archive', rows)])
            produced += 1
    require(produced == limit, 'filtered sampler ended before declared budget')


def prepare_request(source, request):
    stage = request['stage']
    require(stage in STAGES, 'unknown stage')
    batches = []
    for kind, rows in request['groups']:
        rows = np.asarray(rows)
        require(kind in ('archive', 'fish') and rows.ndim == 1 and rows.dtype.kind in 'iu'
            and len(rows) > 0 and ((rows >= 0)&(rows < source.counts[kind])).all(),
            'invalid original training row identities')
        # Requests were already admitted by the original TRAIN-only mapping.
        if kind == 'archive':
            require((np.asarray(source.archive.archive.split)[rows] == 0).all(), 'DEV archive row reached optimizer input')
        elif kind == 'fish':
            episodes = np.asarray(source.fish.episodes[source.fish.starts[rows]])
            require(np.isin(episodes, [0, 1]).all(), 'DEV/TEST fish row reached optimizer input')
        else:
            raise ValueError('only mapped original archive/fish rows may enter training')
        batch = source.chain(rows, kind, images=stage != 'lewm')
        if stage != 'lewm':
            require(batch.features is not None and torch.isfinite(batch.features).all(),
                'Actor/FCPG require recorded DINO features')
            batch = replace(batch, features=batch.features.float())
            batch = measurement_dropout(batch, rows, protocol='burst-cold-v1', variant=int(request['variant']))
        batches.append(batch)
    return batches


def initialize_core(vocabulary, scale, seed=20000, *, device='cpu'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device)
    require(device == torch.device('cpu') or device == torch.device('cuda:0'),
        'scratch initialization device must be cpu or logical cuda:0')
    # Both devices consume the same CPU default initializer before any migration.
    with torch.device('cpu'):
        core = V21ResearchCore(V21Config(len(vocabulary.words), seed=seed, object_relative=True)).float()
        core.set_loss_scale(scale)
    core = core.to(device=device, dtype=torch.float32)
    require(all(value.device == device and value.dtype == torch.float32 for value in core.parameters()),
        'all newly initialized modules must remain FP32 on the selected device')
    return core


def verify_frozen_model_state(core, frozen, frozen_names):
    """Check every tensor, with one host synchronization on a valid update."""
    require(set(frozen) == frozen_names, 'frozen tensor population changed')
    state = core.state_dict()
    checks, messages = [], []
    for name,expected in frozen.items():
        value = state[name]
        # Shape comparisons must remain exact; elementwise comparison alone
        # would admit a broadcastable shape change that torch.equal rejects.
        require(value.shape == expected.shape and value.device == expected.device,
            'frozen state changed: '+name)
        checks.append(value.eq(expected).all())
        messages.append('frozen state changed: '+name)
    parameters = list(core.named_parameters())
    for name,value in parameters:
        checks.append(torch.isfinite(value).all())
        messages.append('nonfinite model parameter: '+name)
    require(bool(checks), 'model has no tensors to verify')
    admitted = torch.stack(checks)
    if not admitted.all().item():
        # Transfer individual flags only after a failed aggregate; no tensor
        # values are substituted or skipped to avoid a synchronization.
        failures = admitted.detach().cpu().tolist()
        raise ValueError(next(message for valid,message in zip(failures,messages) if not valid))
    for name,value in parameters:
        if name in frozen_names:
            require(value.grad is None and not value.requires_grad, 'frozen parameter got gradients: '+name)


class ScratchTrainer(V21Trainer):
    """Keep the established losses/optimizer and verify every frozen tensor."""
    def configure(self, stage):
        require(stage in STAGES and self.device in (torch.device('cpu'), torch.device('cuda:0'))
            and self.precision == 'fp32', 'scratch stages require FP32 on cpu or logical cuda:0')
        optimizer = super().configure(stage)
        prefixes = ('lewm.',) if stage == 'lewm' else ('reader.', 'actor.')
        expected = {name for name,_ in self.core.named_parameters() if name.startswith(prefixes)}
        actual = {name for name,value in self.core.named_parameters() if value.requires_grad}
        require(actual == expected, 'stage trainable parameter population differs')
        self._scratch_frozen = None  # Capture after a possible internal resume load.
        self.verified_updates = 0
        return optimizer

    def step(self, batches, optimizer, stage):
        frozen_names = {name for name,value in self.core.named_parameters() if not value.requires_grad}
        frozen_names |= {name for name,_ in self.core.named_buffers()}
        if self._scratch_frozen is None:
            self._scratch_frozen = {name:value.detach().clone() for name,value in self.core.state_dict().items() if name in frozen_names}
        result = super().step(batches, optimizer, stage)
        verify_frozen_model_state(self.core,self._scratch_frozen,frozen_names)
        self.verified_updates += 1
        if self.device.type == 'cuda':
            result.update(gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)),
                gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(self.device)))
        if getattr(self,'feature_bank',None) is not None:
            result['feature_cache'] = self.feature_bank.stats()
        return dict(result, frozen_tensors_equal=True, external_checkpoint_reads=0)


def internal_checkpoint(output, path):
    output, path = Path(output).resolve(), Path(path)
    require(path.resolve().is_relative_to(output), 'checkpoint must remain inside this new output')
    return path


def resume_stage(output, schedule, *, canary=False):
    """The deepest stage owns the state; earlier random reader values cannot win."""
    deepest = 0
    present = []
    for index,stage in enumerate(STAGES):
        directory = Path(output)/stage
        final, latest = directory/'final.pt', directory/'latest.pt'
        path = final if final.exists() else latest
        if not path.exists():
            continue
        internal_checkpoint(output,path)
        require(all((Path(output)/before/'final.pt').is_file() for before in STAGES[:index]),
            'later stage has no completed predecessor')
        payload = torch.load(path,map_location='cpu',weights_only=False)
        require(payload.get('stage') == stage and payload.get('schedule') == asdict(schedule)
            and payload.get('smoke') is bool(canary), 'internal resume stage/schedule/canary differs')
        if path == final:
            require(payload.get('complete') is True, 'stage final must be complete')
        if present:
            require(present[-1][1], 'later stage follows an incomplete predecessor')
        present.append((stage,payload.get('complete') is True))
        deepest = index
    return deepest


def run_stages(trainer, source, populations, *, through='all', canary=False, after_stage=None):
    require(through == 'all' or through in STAGES, 'unknown final stage')
    finish = len(STAGES)-1 if through == 'all' else STAGES.index(through)
    start = resume_stage(trainer.output,trainer.schedule,canary=canary)
    require(start <= finish, 'cannot rewind a run behind its deepest stage')
    reports = {}
    for stage in STAGES[start:finish+1]:
        # V21Trainer.run loads this stage's own final/latest before returning or
        # resuming. In particular actor/dense state is never replaced by LeWM.
        requests = stage_requests(stage,populations,trainer.schedule,canary=canary)
        reports[stage] = trainer.run(stage,requests,lambda request:prepare_request(source,request),
            stage_steps(stage,populations,trainer.schedule,canary=canary))
        final, latest = trainer.output/stage/'final.pt', trainer.output/stage/'latest.pt'
        if reports[stage].get('complete') and not final.exists():
            # A crash may leave complete latest.pt before final.pt was committed.
            # The trainer has already loaded and validated this exact local state.
            internal_checkpoint(trainer.output,latest)
            temporary = final.with_suffix('.pt.tmp')
            shutil.copyfile(latest,temporary)
            temporary.replace(final)
        if after_stage is not None:
            after_stage(stage)
    return reports


def validate_gpu_continuation(previous, config):
    """Change execution resources only, continuing this same random-start run."""
    require(previous.get('recipe') == RECIPE and previous.get('device') == 'cpu'
        and config.get('recipe') == GPU_RECIPE and config.get('device') == 'cuda:0'
        and previous.get('external_weights_loaded') is False
        and previous.get('smoke') is False and config.get('smoke') is False,
        'GPU continuation requires this formal CPU scratch run')
    resources = {'recipe', 'device', 'threads', 'microbatch', 'cuda_visible_devices',
        'gpu_memory_limit_gib', 'gpu_memory_fraction', 'gpu_total_memory_bytes',
        'tf32_enabled', 'source_preparation_device', 'training_chain_transfer'}
    require({k:v for k,v in previous.items() if k not in resources}
        == {k:v for k,v in config.items() if k not in resources},
        'GPU continuation cannot change data, initialization, precision or schedule')


def validate_resource_continuation(previous, config):
    """A cache/cap change cannot change this run's data or optimizer recipe."""
    require(previous.get('recipe') == config.get('recipe') == GPU_RECIPE
        and previous.get('device') == config.get('device') == 'cuda:0'
        and previous.get('external_weights_loaded') is config.get('external_weights_loaded') is False
        and previous.get('smoke') is config.get('smoke') is False,
        'resource continuation requires this formal GPU scratch run')
    resources = {'gpu_memory_limit_gib', 'gpu_memory_fraction', 'feature_bank_gib'}
    require({k:v for k,v in previous.items() if k not in resources}
        == {k:v for k,v in config.items() if k not in resources},
        'resource continuation cannot change device identity, data, initialization, precision or schedule')


def configure_output(output, config, splits, *, resume, resume_on_gpu=False,
                     resume_with_resources=False):
    output = Path(output)
    current = output/'config.json'
    require(not resume_on_gpu or resume, 'GPU continuation requires --resume')
    require(not resume_with_resources or resume, 'resource continuation requires --resume')
    require(not (resume_on_gpu and resume_with_resources), 'continuation flags are mutually exclusive')
    if resume:
        require(current.is_file(), 'resume requires this exact scratch run configuration')
        previous = json.loads(current.read_text())
        changing_device = previous != config
        if changing_device and resume_on_gpu:
            validate_gpu_continuation(previous, config)
        elif changing_device and resume_with_resources:
            validate_resource_continuation(previous, config)
        else:
            require(not changing_device, 'resume requires this exact scratch run configuration')
        if changing_device:
            require(any((output/stage/'latest.pt').is_file() or (output/stage/'final.pt').is_file()
                for stage in STAGES), 'GPU continuation requires a saved checkpoint in this run')
        with np.load(output/'training-rows.npz',allow_pickle=False) as recorded:
            expected = {f'{role}_{kind}':rows for role in ('train','development') for kind,rows in splits[role].items()}
            require(set(recorded.files) == set(expected), 'resume split population changed')
            for name,rows in expected.items():
                require(np.array_equal(recorded[name],rows), 'resume original split identities changed: '+name)
        if changing_device and resume_on_gpu:
            before = output/'config-before-gpu.json'
            require(not before.exists() or json.loads(before.read_text()) == previous,
                'GPU continuation has a different recorded CPU configuration')
            write_json(before, previous)
            write_json(output/'device-continuation.json', dict(
                source_device='cpu', target_device='cuda:0', same_random_start_run=True,
                existing_stage_checkpoints_preserved=True, optimizer_state_resumed=True,
                sampler_step_resumed=True, external_pretrained_weights_loaded=False,
                numerical_note='CPU and CUDA kernels and random projection streams may differ; no bitwise cross-device claim',
                previous_microbatch=previous['microbatch'], microbatch=config['microbatch'],
                gpu_memory_limit_gib=config['gpu_memory_limit_gib']))
            write_json(current, config)
        elif changing_device:
            resource_keys = ('gpu_memory_limit_gib', 'gpu_memory_fraction', 'feature_bank_gib')
            write_json(output/'resources-history'/f'{time.time_ns()}.json', dict(
                timestamp_unix_s=time.time(), same_random_start_run=True,
                previous={key:previous.get(key) for key in resource_keys},
                current={key:config.get(key) for key in resource_keys},
                optimizer_and_sampler_resumed=True, existing_checkpoints_preserved=True))
            write_json(current, config)
    else:
        require(not output.exists() or not any(output.iterdir()), 'fresh scratch output must be empty; use explicit --resume only for this run')
        output.mkdir(parents=True,exist_ok=True)
        write_json(current,config)
        np.savez_compressed(output/'training-rows.npz',**{f'{role}_{kind}':rows
            for role in ('train','development') for kind,rows in splits[role].items()})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('source-map','body-table','body-manifest','output'):
        result.add_argument('--'+name,type=Path,required=True)
    result.add_argument('--stage',choices=('all',*STAGES),default='all',help='run through this stage')
    result.add_argument('--threads',type=int,default=2)
    result.add_argument('--microbatch',type=int,default=8)
    result.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    result.add_argument('--gpu-memory-limit-gib',type=float,default=8.,
        help='CUDA allocator cap in GiB for this process only')
    result.add_argument('--feature-bank-gib',type=float,default=0.,
        help='bounded CUDA cache for immutable DINO rows; leave 24 GiB within the allocator cap')
    result.add_argument('--resume',action='store_true')
    result.add_argument('--resume-on-gpu', action='store_true',
        help='continue this same formal CPU scratch run on GPU; preserve its checkpoints and schedule')
    result.add_argument('--resume-with-resources',action='store_true',
        help='continue this same GPU run with a changed allocator cap or frozen-feature cache size')
    result.add_argument('--two-step-canary',action='store_true',help='isolated real-data two-update run per stage, never formal')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    require(not args.resume_on_gpu or (args.resume and args.device == 'cuda:0'),
        '--resume-on-gpu requires --resume and --device cuda:0')
    require(not args.resume_with_resources or (args.resume and args.device == 'cuda:0'),
        '--resume-with-resources requires --resume and --device cuda:0')
    require(not (args.resume_on_gpu and args.resume_with_resources), 'continuation flags are mutually exclusive')
    require(math.isfinite(args.feature_bank_gib) and args.feature_bank_gib >= 0,
        'feature cache budget must be finite and nonnegative')
    require(not args.feature_bank_gib or (args.device == 'cuda:0'
        and args.feature_bank_gib + 24 <= args.gpu_memory_limit_gib),
        'CUDA feature cache requires 24 GiB of allocator workspace outside its budget')
    require(1 <= args.threads <= 32, 'CPU thread count must be within 1..32')
    maximum_microbatch = 64 if args.device == 'cuda:0' else 32
    require(1 <= args.microbatch <= maximum_microbatch,
        f'{args.device} microbatch count must be within 1..{maximum_microbatch}')
    output = args.output.resolve()
    if not args.resume:
        require(not output.exists() or not any(output.iterdir()), 'fresh scratch output must be empty')
    device, device_config = configure_device(args.device,args.gpu_memory_limit_gib)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    torch.backends.mha.set_fastpath_enabled(False)
    mapping = json.loads(args.source_map.read_text())
    original = mapping['source_paths_original']
    paths, source_mapping = resolve_source_paths(original,args.source_map)
    source = V21Sources(paths,device=torch.device('cpu'),object_relative_table=args.body_table)
    source.device_cache = None
    source.compact_host_features = False
    feature_bank = None
    if device.type == 'cuda':
        if args.feature_bank_gib:
            from fcpg.v21_scratch_features import ScratchDinoFeatureBank
            feature_bank = ScratchDinoFeatureBank(source,device,int(args.feature_bank_gib*2**30))
        source = DeviceStagedSources(source,device,feature_bank=feature_bank)
        device_config.update(source_preparation_device='cpu',
            training_chain_transfer='preserve-dtype-nonblocking-v1')
    splits = derive_splits(source)
    counts = {f'{kind}_{role}_rows':len(splits[role][kind]) for role in ('train','development') for kind in ('archive','dense')}
    require(all(counts[name] == DATA_CONTRACT[name] for name in counts), 'canonical original archive/dense split counts differ')
    require(len(splits['train']['fish']) == 381 and len(splits['development']['fish']) == 105,
        'canonical fish0/1 TRAIN and fish3 DEV populations differ')
    body = validate_manifest(args.body_table,args.body_manifest,source.fish,paths['fish_wrist_root'],
        splits['fish_required_endpoints'],source_wrist_root=original['fish_wrist_root'])
    scale = training_loss_scale(source,splits['train']['archive'])
    schedule = V21Schedule()
    model_config = V21Config(len(source.vocabulary.words),seed=schedule.seed,object_relative=True)
    config = dict(recipe=RECIPE if device.type == 'cpu' else GPU_RECIPE,
        version=VERSION,contract=CONTRACT,model=asdict(model_config),
        schedule=asdict(schedule),vocabulary=list(source.vocabulary.words),source_paths=paths,
        source_paths_original=original,source_mapping=source_mapping,q_loss_scale=scale.tolist(),
        training_data_contract=DATA_CONTRACT,normalization_train_chains=len(splits['train']['archive']),
        normalization_command_increments=len(splits['train']['archive'])*20,
        smoke=args.two_step_canary,two_step_real_data_canary=args.two_step_canary,
        device=str(device),precision='fp32',threads=args.threads,microbatch=args.microbatch,
        feature_bank_gib=float(args.feature_bank_gib),
        initialization='module-default-random-all-components-created-before-first-stage',
        external_weights_loaded=False,external_checkpoint_reads=0,
        visual_backbone='frozen-dinov2-vits14-real-cache',task_encoder='native-word-embedding',
        body_table=str(args.body_table.resolve()),body_manifest=str(args.body_manifest.resolve()),body_contract=body,
        validation_scope='previously reviewed development, excluded from gradients; not blind TEST',
        hardware_used=False,deployment=False,physical_success_verified=False)
    config.update(device_config)
    config.update(dropout_contract('burst-cold-v1'))
    config.update(temporal_training_configuration(PROGRESS_TRAINING_POLICY))
    configure_output(output,config,splits,resume=args.resume,resume_on_gpu=args.resume_on_gpu,
        resume_with_resources=args.resume_with_resources)
    pin_source_map(output,args.source_map)
    core = initialize_core(source.vocabulary,scale,schedule.seed,device=device)
    if not args.resume:
        write_json(output/'initialization.json',dict(seed=schedule.seed,external_weights_loaded=False,
            external_checkpoint_reads=0,all_components_created_before_training=True,object_relative=True,
            parameter_counts={name:sum(value.numel() for value in getattr(core,name).parameters())
                for name in ('reader','actor','lewm')}))
    trainer = ScratchTrainer(core,source.vocabulary,output,schedule,microbatch=args.microbatch,
        clear_cache=None if feature_bank is None else lambda: feature_bank.clear(release_allocator=True),
        smoke=args.two_step_canary,precision='fp32',temporal_feature_policy=PROGRESS_TRAINING_POLICY)
    trainer.feature_bank = feature_bank
    def development_after_fixed_save(stage):
        destination = output/'development'/f'{stage}.json'
        if destination.exists():
            require(json.loads(destination.read_text()).get('stage') == stage,
                'recorded development report stage differs')
            return
        from scripts.evaluate_v21_scratch_development import evaluate_development
        report = evaluate_development(core,source,splits['development'],stage=stage,
            microbatch=min(args.microbatch,8),archive_limit=64)
        require(report.get('stage') == stage, 'development report must identify the completed stage')
        write_json(destination,report)
    run_stages(trainer,source,splits['train'],through=args.stage,canary=args.two_step_canary,
        after_stage=None if args.two_step_canary else development_after_fixed_save)
    if feature_bank is not None:
        write_json(output/'input-metrics.json',feature_bank.stats())
    write_json(output/'run-status.json',dict(complete=not args.two_step_canary and (output/'fish/final.pt').is_file(),
        canary_complete=args.two_step_canary and (output/'fish/final.pt').is_file(),
        stopped_after=args.stage,external_weights_loaded=False,external_checkpoint_reads=0,
        real_data_used=True,frozen_tensors_verified_each_update=True,hardware_used=False,deployment=False,
        physical_success_verified=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
