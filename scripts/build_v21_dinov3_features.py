#!/usr/bin/env python3
"""DINOv3 ViT-L/16 features of the 1.2M archive index's block endpoints (H0-H4, both camera slots), stored as 8-bit.

James 2026-10-21: 「DINOv3 L，8 位 全部替换成这个」「先删掉，再生成，然后拷贝回来」「直接搞过去 然后pro6000d的cpu解码」.
Rows, row order and pictures are those of the two DINOv2 stores the 1.2M training roots read (old: Codex's `reused`
blocks of the 320k parent; fresh: scripts/build_v21_fresh_endpoint_features.py; per partition sorted global frames), so
a root's token sources (kind, source_row) address the new stores unchanged:
  - decode: fcpg.v21_cache_rebuild.episode_job, the recipe of both stores (recorded-time PTS decode, aspect letterbox
    224 cv2 INTER_AREA, original camera masks, placeholder check, view order [slots[1], slots[0]]), with the raw
    q/action identity check against the temporal index;
  - encoder: timm vit_large_patch16_dinov3.lvd1689m from the local safetensors, ImageNet normalization, fp32 weights
    and residual stream with fp16 autocast matrix products, forward_features (final norm), the 5 prefix tokens (class +
    4 registers) dropped -> [196, 1024] per picture. Against fp32 on 512 archive pictures (10/21, precision_dinov3_l.py,
    after the NERO sim session's finding): fp16 autocast median 0.19% / p99 0.93% / worst 10.6%, no overflow;
    bf16 autocast 2.7% / 17% / 89% (the first runs used bf16; set aside as stray). The robot's V100 runs fp16 too;
  - storage: every token as int8 with its own fp16 scale (max |value| / 127, rounded to fp16 before quantizing), blocks
    of 64 rows: block-<first row>.v21z = int8 [n, 2, 196, 1024], scale-<first row>.v21z = fp16 [n, 2, 196]
    (fcpg/v21_lossless_features.py, zstd1, both decoded and compared before they leave) and meta-<first row>.npz
    (global frame ids, valid, reasons, capture times and q, video pts); invalid views are zero.
    Measured 10/21 on 24 fresh episodes: 302.6 KB per row (the DINOv2 store 302.6 KB), 1.6% mean relative error.
Subcommands:
  jobs          (.102)     one part's decode jobs in row order, dataset roots moved to --raw-prefix-to, and the DINOv2
                           store's identities and validity
  extra-jobs    (.102)     17.68: jobs for the pictures of the recordings an extended half-second index added (no
                           DINOv2 reference); encode them like the rest. --new-recordings-only (17.90): of an index
                           over an extended temporal pool, only the recordings the pool's extension added (those of the
                           earlier --extra list already have their store)
  encode        (pro6000d) CPU decode workers -> DINOv3-L on the GPU -> blocks, as a tar stream on stdout (the .102
                           side extracts it onto the HDD: pro6000d has no room for the store beside the videos) or into
                           a folder; a row range per partition (--resume-blocks / --stop-blocks, block counts) so
                           several GPUs share a part; on .102 the fresh part can read the stage 1 RGB cache of the
                           DINOv2 build instead of decoding (--rgb-cache: the same pictures, checked bit for bit)
  resume-point  (.102)     consecutive complete blocks per partition of a received part
  verify        (.102)     every block decodes, identities match the DINOv2 store, valid wherever it was valid; optional
                           spot check against DINOv3-L recomputed from the fresh RGB cache
  check-dinov2  (.102)     decode a few episodes of a part and run DINOv2-S/14 as the store was made: bit-equal tokens
                           mean the decode gives the store's pictures
Never deletes anything; never reads fish data."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib.util
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tarfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fcpg.v21_lossless_features import LosslessFeatures

PARTITIONS = ('temporal-train', 'temporal-dev')
MODEL, PATCHES, DIM, PREFIX = 'vit_large_patch16_dinov3.lvd1689m', 196, 1024, 5
SCHEMA = 'v21-1p2m-dinov3-l16-int8-v1'
MEAN, STD = (.485, .456, .406), (.229, .224, .225)


def builder():
    """scripts/build_v21_fresh_endpoint_features.py as a module (jobs_for, check_motion, ordered_decode, sha256, ...)."""
    spec = importlib.util.spec_from_file_location('fresh_builder', REPO / 'scripts' / 'build_v21_fresh_endpoint_features.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def dumps(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


# ---------------------------------------------------------------- jobs (.102)

def make_jobs(args):
    b = builder()
    manifest = json.loads((args.index / 'manifest.json').read_text())
    temporal = Path(manifest['temporal_root'])
    offsets = np.load(temporal / 'episode_offsets.npy', allow_pickle=False)
    records = {int(r['temporal_episode_index']): r for r in
               json.loads((args.index / 'episode-path-map.json').read_text())['episodes']}
    args.out.mkdir(parents=True, exist_ok=False)
    counts = {}
    with open(args.out / 'jobs.jsonl', 'w') as handle:
        for partition in PARTITIONS:
            source = args.dinov2_store / partition
            ids = np.load(source / 'global-frame-index.npy', allow_pickle=False).astype(np.int64)
            valid = np.load(source / 'view_valid.npy', allow_pickle=False)
            if valid.shape != (len(ids), 2) or valid.dtype != np.bool_ or not (np.diff(ids) > 0).all():
                raise ValueError(f'{source}: identities must be sorted unique with [rows, 2] validity')
            (args.out / partition).mkdir()
            np.save(args.out / partition / 'global-frame-index.npy', ids, allow_pickle=False)
            np.save(args.out / partition / 'dinov2-view-valid.npy', valid, allow_pickle=False)
            episodes = 0
            for job in b.jobs_for(partition, ids, records, offsets, 0, 0):
                root = job['row']['dataset_root']
                if not root.startswith(args.raw_prefix_from):
                    raise ValueError(f'dataset outside the raw prefix: {root}')
                job['row'] = dict(job['row'], dataset_root=args.raw_prefix_to + root[len(args.raw_prefix_from):])
                handle.write(json.dumps(job, separators=(',', ':')) + '\n')
                episodes += 1
            counts[partition] = dict(rows=int(len(ids)), episodes=episodes)
    write_json(args.out / 'jobs.json', dict(part=args.part, index=str(args.index.resolve()),
        dinov2_store=str(args.dinov2_store.resolve()), temporal_root=str(temporal),
        raw_prefix_from=args.raw_prefix_from, raw_prefix_to=args.raw_prefix_to, partitions=counts))
    print(json.dumps(counts), flush=True)


def make_extra_jobs(args):
    """17.68 (James 10/23 「那四部分尽量搞好」): decode jobs for the pictures of the recordings an extended half-second
    index added with --extra (scripts/build_v21_halfsecond_index.py), per partition in global frame order, written as
    a 'fresh' part so that encode and the training roots builder read it as they read the first store. There is no
    DINOv2 store of these pictures: dinov2-view-valid.npy is all False, so encode compares nothing against it."""
    b = builder()
    manifest = json.loads((args.index / 'manifest.json').read_text())
    extra = manifest.get('extra') or {}
    if not extra.get('recordings'):
        raise ValueError('the index added no --extra recordings')
    added = np.asarray(sorted(int(x['episode']) for x in json.loads(Path(extra['file']).read_text())['episodes']))
    temporal = Path(manifest['temporal_root'])
    first_new = 0
    if args.new_recordings_only:                       # 17.90: only what the extended pool added
        pool = json.loads((temporal / 'manifest.json').read_text())
        if not manifest.get('temporal_root_extends') or not pool.get('extended_from'):
            raise ValueError('--new-recordings-only needs an index over an extended temporal pool')
        first_new = int(pool['extended_from']['base_counts']['episodes'])
        added = added[added >= first_new]
    offsets = np.load(temporal / 'episode_offsets.npy', allow_pickle=False)
    records = {int(r['temporal_episode_index']): r for r in
               json.loads((args.index / 'episode-path-map.json').read_text())['episodes']}
    pictures = np.load(args.index / 'endpoint_global_frame_index.npy', allow_pickle=False).astype(np.int64)
    mapping = np.load(args.index / 'row_endpoint_index.npy', allow_pickle=False)
    split = np.load(args.index / 'split.npy', allow_pickle=False)
    split_of = np.full(len(pictures), 255, np.uint8)
    split_of[mapping.reshape(-1)] = np.repeat(split, mapping.shape[1])
    episode = np.searchsorted(offsets, pictures, side='right') - 1
    args.out.mkdir(parents=True, exist_ok=False)
    counts = {}
    with open(args.out / 'jobs.jsonl', 'w') as handle:
        for code, partition in enumerate(PARTITIONS):
            ids = pictures[(split_of == code) & np.isin(episode, added)]
            (args.out / partition).mkdir()
            np.save(args.out / partition / 'global-frame-index.npy', ids, allow_pickle=False)
            np.save(args.out / partition / 'dinov2-view-valid.npy', np.zeros((len(ids), 2), bool), allow_pickle=False)
            episodes = 0
            for job in b.jobs_for(partition, ids, records, offsets, 0, 0):
                handle.write(json.dumps(job, separators=(',', ':')) + '\n')
                episodes += 1
            counts[partition] = dict(rows=int(len(ids)), episodes=episodes)
    write_json(args.out / 'jobs.json', dict(part='fresh', index=str(args.index.resolve()), extra=extra,
        temporal_root=str(temporal), reference='none (dinov2-view-valid all False)', partitions=counts,
        new_recordings_only=bool(args.new_recordings_only), first_new_episode=first_new, recordings=len(added)))
    print(json.dumps(counts), flush=True)


# ---------------------------------------------------------------- encode (pro6000d)

class TarSink:
    """Members of one tar stream (written only by the main thread); the .102 side runs tar -x on it."""
    def __init__(self, handle):
        self.handle = handle
        self.tar = tarfile.open(fileobj=handle, mode='w|', format=tarfile.PAX_FORMAT)

    def put(self, name, data):
        info = tarfile.TarInfo(name)
        info.size, info.mtime, info.mode = len(data), int(time.time()), 0o664
        self.tar.addfile(info, io.BytesIO(data))

    def close(self):
        self.tar.close()
        self.handle.close()


class FolderSink:
    def __init__(self, root):
        self.root = Path(root)

    def put(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)

    def close(self):
        pass


def encode_block(tokens, scales, meta):
    """-> the three payloads of one block; both feature arrays are decoded and compared first (thread-safe)."""
    codec = LosslessFeatures(level=1, shuffle=True)
    a, s = codec.encode(tokens), codec.encode(scales)
    if (not np.array_equal(codec.decode(a), tokens)
            or not np.array_equal(codec.decode(s).view(np.uint16), scales.view(np.uint16))):
        raise ValueError('a DINOv3 block does not decode to the same bits')
    buffer = io.BytesIO()
    np.savez(buffer, **meta)
    return a, s, buffer.getvalue()


def load_dinov3(weights, device):
    import timm
    model = timm.create_model(MODEL, pretrained=True, num_classes=0,
                              pretrained_cfg_overlay=dict(file=str(weights))).to(device).eval()
    model.requires_grad_(False)
    if (int(model.num_prefix_tokens) != PREFIX or int(model.embed_dim) != DIM
            or tuple(model.patch_embed.patch_size) != (16, 16)):
        raise ValueError('not DINOv3 ViT-L/16 with 5 prefix tokens')
    return model


def dinov3_int8(model, pictures, device, batch_size):
    """uint8 [m, 224, 224, 3] -> int8 tokens [m, 196, 1024] and fp16 scales [m, 196] (one scale per token)."""
    import torch
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    tokens, scales = [], []
    with torch.inference_mode():
        for start in range(0, len(pictures), batch_size):
            x = torch.from_numpy(pictures[start:start + batch_size]).to(device).permute(0, 3, 1, 2).float().div_(255.)
            with torch.autocast('cuda', dtype=torch.float16):        # bf16 is 14x less exact here (docstring)
                features = model.forward_features((x - mean) / std)
            t = features[:, PREFIX:].float()
            if t.shape[1:] != (PATCHES, DIM) or not bool(torch.isfinite(t).all()):
                raise ValueError(f'DINOv3-L patch tokens {tuple(t.shape)} or not finite')
            scale = (t.abs().amax(-1, keepdim=True) / 127.).clamp_min(1e-6).half()
            tokens.append(torch.round(t / scale.float()).clamp_(-127, 127).to(torch.int8).cpu())
            scales.append(scale.squeeze(-1).cpu())
    return torch.cat(tokens).numpy(), torch.cat(scales).numpy()


def encode(args):
    import torch
    b = builder()
    jobs_meta = json.loads((args.jobs / 'jobs.json').read_text())
    part = jobs_meta['part']
    resume = json.loads(args.resume_blocks) if args.resume_blocks else {}
    stops = json.loads(args.stop_blocks) if args.stop_blocks else {}
    partitions = args.partitions.split(',') if args.partitions else list(PARTITIONS)
    if not set(partitions) <= set(PARTITIONS):
        raise ValueError(f'partitions must be among {PARTITIONS}')
    if args.output_tar:                     # the tar stream owns the real stdout; prints (also workers') go to stderr
        stream = os.fdopen(os.dup(1), 'wb')
        os.dup2(2, 1)
        sink = TarSink(stream)
    else:
        sink = FolderSink(args.output)
    by_partition = {p: [] for p in PARTITIONS}
    with open(args.jobs / 'jobs.jsonl') as handle:
        for line in handle:
            job = json.loads(line)
            job['decode'] = args.decode
            by_partition[job['partition']].append(job)
    q_bank = np.load(args.temporal / 'states.npy', mmap_mode='r')
    a_bank = np.load(args.temporal / 'actions.npy', mmap_mode='r')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(
        min(args.gpu_memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory, .9), device)
    model = load_dinov3(args.weights, device)
    contract = dict(schema=SCHEMA, part=part, jobs=jobs_meta, block_frames=args.block_frames, codec='zstd1',
        tokens=dict(file='block-<first row>.v21z', dtype='int8', shape=['rows', 2, PATCHES, DIM]),
        scales=dict(file='scale-<first row>.v21z', dtype='float16', shape=['rows', 2, PATCHES],
                    rule='per token: max |value| / 127 rounded to fp16, value = int8 * scale'),
        meta='meta-<first row>.npz: global_frame_index, valid, reasons, capture_times, capture_q, video_pts',
        encoder=dict(timm=MODEL, weights=str(args.weights), weights_sha256=b.sha256(args.weights), image_size=224,
                     patch_grid=[14, 14], prefix_tokens_dropped=PREFIX, final_norm=True,
                     compute_precision='fp32 weights, fp16 autocast',
                     normalization=dict(mean=MEAN, std=STD), torch=torch.__version__),
        decode=dict(recipe='fcpg.v21_cache_rebuild.episode_job', decoder=args.decode, resize='aspect-letterbox-cv2-INTER_AREA',
                    view_order='[slots[1], slots[0]] as the DINOv2 stores', motion_identity_checked=True),
        max_episodes=args.max_episodes, partitions=partitions, start_blocks=resume, stop_blocks=stops,
        source='stage 1 RGB cache ' + str(args.rgb_cache) if args.rgb_cache else 'decoded here')
    manifest_name = f'{part}/manifest-{args.tag}.json'
    sink.put(manifest_name, dumps(dict(contract, complete=False)))
    encoders = ThreadPoolExecutor(max_workers=args.encoders)
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                               initializer=os.nice, initargs=(5,))
    started, done_rows, pictures, extra_valid, stored = time.monotonic(), 0, 0, 0, 0
    timers = dict(wait_decode=0., dinov3=0., assemble=0., wait_encode=0., write=0.)
    pending, decode_errors = deque(), []

    def drain(limit):
        nonlocal stored
        while len(pending) > limit:
            clock = time.monotonic()
            names, future = pending.popleft()
            payloads = future.result()
            timers['wait_encode'] += time.monotonic() - clock
            clock = time.monotonic()
            for name, data in zip(names, payloads):
                sink.put(name, data)
                stored += len(data)
            timers['write'] += time.monotonic() - clock

    try:
        for partition in partitions:
            ids = np.load(args.jobs / partition / 'global-frame-index.npy', allow_pickle=False)
            expected_valid = np.load(args.jobs / partition / 'dinov2-view-valid.npy', allow_pickle=False)
            start = int(resume.get(partition, 0)) * args.block_frames
            stop = min(int(stops[partition]) * args.block_frames, len(ids)) if partition in stops else len(ids)
            jobs = [j for j in by_partition[partition] if j['positions'][-1] >= start and j['positions'][0] < stop]
            if args.max_episodes:
                jobs = jobs[:args.max_episodes]
            if not jobs:
                continue
            prefix = f'{part}/{partition}'
            chunks, buffer_start, buffered = [], start, 0
            stream_results = (b.ordered_rgb_cache(args.rgb_cache, jobs, 8) if args.rgb_cache
                              else b.ordered_decode(pool, jobs, 2 * args.workers))
            group, last_report = [], 0.

            def flush(final):
                nonlocal chunks, buffer_start, buffered, done_rows
                while buffered >= args.block_frames or (final and buffered):
                    n = min(args.block_frames, buffered)
                    joined = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
                    block = {k: v[:n] for k, v in joined.items()}
                    rest = {k: v[n:] for k, v in joined.items()}
                    chunks = [rest] if buffered > n else []
                    tokens, scales = np.ascontiguousarray(block.pop('tokens')), np.ascontiguousarray(block.pop('scales'))
                    names = [f'{prefix}/block-{buffer_start:09d}.v21z', f'{prefix}/scale-{buffer_start:09d}.v21z',
                             f'{prefix}/meta-{buffer_start:09d}.npz']
                    pending.append((names, encoders.submit(encode_block, tokens, scales, block)))
                    drain(2 * args.encoders)
                    buffer_start, buffered, done_rows = buffer_start + n, buffered - n, done_rows + n

            for number, job in enumerate(jobs):
                clock = time.monotonic()
                result = next(stream_results)
                timers['wait_decode'] += time.monotonic() - clock
                if result['identity'] != job['identity']:
                    raise ValueError('episode order mismatch')
                for c, view_detail in enumerate(result.get('details') or []):
                    if 'decode_error' in view_detail:           # a corrupt packet part-way: reason 7, on record
                        decode_errors.append(dict(identity=job['identity'], view=view_detail['view'],
                                                  path=view_detail['path'], after_pts=view_detail['decode_error_after_pts'],
                                                  views_lost=int((result['reasons'][:, c] == 7).sum()),
                                                  error=view_detail['decode_error']))
                        print(json.dumps(dict(decode_error=decode_errors[-1])), file=sys.stderr, flush=True)
                if not args.rgb_cache:                          # stage 1 checked it when it wrote the cache
                    b.check_motion(result, q_bank, a_bank, job)
                group.append((job, result))
                if sum(int(r['valid'].sum()) for _, r in group) < args.dino_images and number + 1 < len(jobs):
                    continue
                clock = time.monotonic()
                images = [r['rgb'][r['valid']] for _, r in group]
                count = sum(map(len, images))
                q, s = (dinov3_int8(model, np.concatenate(images), device, args.batch_size) if count else
                        (np.zeros((0, PATCHES, DIM), np.int8), np.zeros((0, PATCHES), np.float16)))
                timers['dinov3'] += time.monotonic() - clock
                clock = time.monotonic()
                cut = np.cumsum([0] + [len(x) for x in images])
                finished, group = group, []
                for k, (job_k, r) in enumerate(finished):
                    positions = np.asarray(job_k['positions'], np.int64)
                    valid = r['valid']
                    if not np.array_equal(r['endpoint_ids'], ids[positions]):
                        raise ValueError(f'episode {job_k["identity"]}: identities differ from the DINOv2 store')
                    lost = expected_valid[positions] & ~valid
                    if lost.any():
                        raise ValueError(f'episode {job_k["identity"]}: {int(lost.sum())} views valid in the DINOv2 '
                                         'store decode invalid here')
                    extra_valid += int((valid & ~expected_valid[positions]).sum())
                    tokens = np.zeros((len(positions), 2, PATCHES, DIM), np.int8)
                    scales = np.zeros((len(positions), 2, PATCHES), np.float16)
                    rows, views = np.nonzero(valid)
                    tokens[rows, views], scales[rows, views] = q[cut[k]:cut[k + 1]], s[cut[k]:cut[k + 1]]
                    keep = (positions >= buffer_start + buffered) & (positions < stop)   # only this row range
                    if keep.any() and not np.array_equal(positions[keep], np.arange(
                            buffer_start + buffered, buffer_start + buffered + int(keep.sum()))):
                        raise ValueError('rows must arrive in contiguous order')
                    chunks.append(dict(tokens=tokens[keep], scales=scales[keep],
                                       global_frame_index=r['endpoint_ids'][keep], valid=valid[keep],
                                       reasons=r['reasons'][keep], capture_times=r['capture_times'][keep],
                                       capture_q=r['capture_q'][keep], video_pts=r['video_pts'][keep]))
                    buffered += int(keep.sum())
                    pictures += int(valid[keep].sum())
                timers['assemble'] += time.monotonic() - clock
                flush(final=number + 1 == len(jobs))
                elapsed = time.monotonic() - started
                if elapsed - last_report > 30:
                    last_report = elapsed
                    print(json.dumps(dict(partition=partition, row=buffer_start, of=int(len(ids)), rows_done=done_rows,
                        rows_per_s=round(done_rows / elapsed, 1), pictures_per_s=round(pictures / elapsed, 1),
                        stored_gb=round(stored / 1e9, 2), extra_valid_views=extra_valid, elapsed_s=round(elapsed),
                        seconds={k: round(v) for k, v in timers.items()})), file=sys.stderr, flush=True)
            drain(0)
            if not args.max_episodes and buffer_start != stop:
                raise ValueError(f'{partition}: ended at row {buffer_start}, not {stop}')
        drain(0)
        summary = dict(rows=done_rows, pictures=pictures, extra_valid_views=extra_valid, stored_gb=stored / 1e9,
                       seconds=time.monotonic() - started, timers=timers, decode_errors=decode_errors)
        sink.put(manifest_name, dumps(dict(contract, complete=not args.max_episodes, summary=summary)))
        print(json.dumps(summary), file=sys.stderr, flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        encoders.shutdown(wait=True, cancel_futures=True)
        sink.close()


# ---------------------------------------------------------------- resume-point / verify (.102)

def block_starts(root):
    return sorted(int(p.name[6:15]) for p in root.glob('block-*.v21z'))


def read_block(root, start, codec):
    tokens = codec.decode((root / f'block-{start:09d}.v21z').read_bytes())
    scales = codec.decode((root / f'scale-{start:09d}.v21z').read_bytes())
    with np.load(root / f'meta-{start:09d}.npz', allow_pickle=False) as meta:
        meta = {k: meta[k] for k in meta.files}
    return tokens, scales, meta


def resume_point(args):
    codec, out = LosslessFeatures(level=1, shuffle=True), {}
    for partition in PARTITIONS:
        root, blocks = args.store / partition, 0
        while all((root / f'{kind}-{blocks * args.block_frames:09d}{ext}').exists()
                  for kind, ext in (('block', '.v21z'), ('scale', '.v21z'), ('meta', '.npz'))):
            blocks += 1
        while blocks:                                       # the last received block may be cut short
            try:
                read_block(root, (blocks - 1) * args.block_frames, codec)
                break
            except Exception:
                blocks -= 1
        out[partition] = blocks
    print(json.dumps(out))


def verify(args):
    codec = LosslessFeatures(level=1, shuffle=True)
    report = {}
    for partition in PARTITIONS:
        root = args.store / partition
        ids = np.load(args.jobs / partition / 'global-frame-index.npy', allow_pickle=False)
        expected_valid = np.load(args.jobs / partition / 'dinov2-view-valid.npy', allow_pickle=False)
        starts = block_starts(root)
        if starts != list(range(0, len(ids), args.block_frames)):
            raise ValueError(f'{partition}: blocks {len(starts)}, expected {-(-len(ids) // args.block_frames)}')
        rows = extra = pictures = stored = 0
        for start in starts:
            tokens, scales, meta = read_block(root, start, codec)
            n = min(args.block_frames, len(ids) - start)
            valid = meta['valid']
            if (tokens.dtype != np.int8 or tokens.shape != (n, 2, PATCHES, DIM) or scales.dtype != np.float16
                    or scales.shape != (n, 2, PATCHES) or valid.shape != (n, 2)):
                raise ValueError(f'{partition} block {start}: shapes')
            if not np.array_equal(meta['global_frame_index'], ids[start:start + n]):
                raise ValueError(f'{partition} block {start}: identities differ from the DINOv2 store')
            if (expected_valid[start:start + n] & ~valid).any():
                raise ValueError(f'{partition} block {start}: a view valid in the DINOv2 store is invalid here')
            if tokens[~valid].any() or scales[~valid].any() or not (scales[valid] > 0).all():
                raise ValueError(f'{partition} block {start}: invalid views not zero or valid scales not positive')
            rows += n
            pictures += int(valid.sum())
            extra += int((valid & ~expected_valid[start:start + n]).sum())
            stored += sum((root / f'{kind}-{start:09d}{ext}').stat().st_size
                          for kind, ext in (('block', '.v21z'), ('scale', '.v21z'), ('meta', '.npz')))
        report[partition] = dict(rows=rows, blocks=len(starts), pictures=pictures, extra_valid_views=extra,
                                 stored_gb=stored / 1e9, bytes_per_row=stored / max(rows, 1))
    if args.spot_rgb_cache:
        report['spot_check'] = spot_check(args, codec)
    print(json.dumps(report, indent=1))


def spot_check(args, codec):
    """DINOv3-L recomputed on this machine from the fresh RGB cache (stage 1 of the DINOv2 build) for a few episodes,
    against the received int8 tokens: cosine per picture."""
    import torch
    b = builder()
    device = torch.device('cuda:0')
    model = load_dinov3(args.weights, device)
    jobs = [json.loads(line) for line in open(args.jobs / 'jobs.jsonl')]
    pick = [jobs[i] for i in np.linspace(0, len(jobs) - 1, args.spot_episodes).round().astype(int)]
    cosines = []
    for job in pick:
        r = b.read_rgb_episode(args.spot_rgb_cache, job)
        positions = np.asarray(job['positions'])
        root = args.store / job['partition']
        q, s = dinov3_int8(model, r['rgb'][r['valid']], device, 64)
        here = (q.astype(np.float32) * s.astype(np.float32)[..., None])
        got = []
        for row, view in zip(*np.nonzero(r['valid'])):
            start = positions[row] // args.block_frames * args.block_frames
            tokens, scales, _ = read_block(root, int(start), codec)
            got.append(tokens[positions[row] - start, view].astype(np.float32) * scales[positions[row] - start, view, :, None])
        got = np.stack(got)
        cos = (here * got).sum(-1) / (np.linalg.norm(here, axis=-1) * np.linalg.norm(got, axis=-1) + 1e-8)
        cosines.append(cos.reshape(-1))
    cosines = np.concatenate(cosines)
    return dict(episodes=len(pick), tokens=int(len(cosines)), cosine_mean=float(cosines.mean()),
                cosine_p01=float(np.percentile(cosines, 1)), cosine_min=float(cosines.min()))


# ---------------------------------------------------------------- check-dinov2 (.102)

def check_dinov2(args):
    import torch
    from fcpg.prod_v18_1_training import load_frozen_dinov2
    from fcpg.v21_cache_rebuild import episode_job
    b = builder()
    meta = json.loads((args.jobs / 'jobs.json').read_text())
    jobs = [json.loads(line) for line in open(args.jobs / 'jobs.jsonl')]
    pick = [jobs[i] for i in np.linspace(0, len(jobs) - 1, args.episodes).round().astype(int)]
    for job in pick:                                         # back to this machine's copy of the raw data
        root = job['row']['dataset_root']
        job['row'] = dict(job['row'], dataset_root=meta['raw_prefix_from'] + root[len(meta['raw_prefix_to']):])
    device = torch.device('cuda:0')
    backbone = load_frozen_dinov2(args.repository, args.dino_weights, device=device)
    codec = LosslessFeatures(level=1, shuffle=True)
    store = Path(meta['dinov2_store'])
    same = total = valid_same = 0
    worst, cosines, time_shift, pts_shift = 0., [], [], []
    for job in pick:
        r = episode_job(job)
        positions = np.asarray(job['positions'])
        folder = store / job['partition']
        expected_valid = np.load(args.jobs / job['partition'] / 'dinov2-view-valid.npy', mmap_mode='r')[positions]
        valid_same += int(np.array_equal(r['valid'], expected_valid))
        both = r['valid'] & expected_valid
        # the store's own capture clock for these rows: same moments, or other frames?
        time_shift.append(np.abs(r['capture_times'][both] - np.load(folder / 'capture_times.npy', mmap_mode='r')[positions][both]))
        pts_shift.append(np.abs(r['video_pts'][both] - np.load(folder / 'video_pts.npy', mmap_mode='r')[positions][both]))
        tokens = b.gpu_fp16_tokens(backbone, r['rgb'][both], device, 64)
        stored = []
        for row, view in zip(*np.nonzero(both)):
            start = positions[row] // 64 * 64
            block = codec.decode((folder / f'block-{start:09d}.v21z').read_bytes())
            stored.append(block[positions[row] - start, view])
        stored = np.stack(stored) if stored else np.zeros_like(tokens)
        equal = (tokens.view(np.uint16) == stored.view(np.uint16)).all(axis=(1, 2))
        same += int(equal.sum())
        total += len(equal)
        if len(tokens):
            here, there = tokens.astype(np.float32).reshape(len(tokens), -1), stored.astype(np.float32).reshape(len(tokens), -1)
            worst = max(worst, float(np.abs(here - there).max()))
            cosines.append((here * there).sum(1) / (np.linalg.norm(here, axis=1) * np.linalg.norm(there, axis=1) + 1e-8))
    cosines, time_shift, pts_shift = (np.concatenate(x) if x else np.zeros(0) for x in (cosines, time_shift, pts_shift))
    print(json.dumps(dict(part=meta['part'], episodes=len(pick), episodes_with_same_validity=valid_same,
                          pictures=total, bit_equal_pictures=same, max_abs_difference=worst,
                          picture_cosine=dict(mean=float(cosines.mean()), min=float(cosines.min())),
                          capture_time_shift_s=dict(max=float(time_shift.max()), zero=float((time_shift == 0).mean())),
                          video_pts_shift_s=dict(max=float(np.nanmax(pts_shift)) if pts_shift.size else 0.)), indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    j = sub.add_parser('jobs')
    j.add_argument('--index', type=Path, required=True)
    j.add_argument('--part', choices=('old', 'fresh'), required=True)
    j.add_argument('--dinov2-store', type=Path, required=True, help='the part\'s DINOv2 store (partition folders)')
    j.add_argument('--raw-prefix-from', required=True)
    j.add_argument('--raw-prefix-to', required=True)
    j.add_argument('--out', type=Path, required=True)
    x = sub.add_parser('extra-jobs')
    x.add_argument('--index', type=Path, required=True, help='a half-second index built with --extra')
    x.add_argument('--out', type=Path, required=True)
    x.add_argument('--new-recordings-only', action='store_true',
                   help='17.90: only the recordings the extended temporal pool added')
    e = sub.add_parser('encode')
    e.add_argument('--jobs', type=Path, required=True)
    e.add_argument('--temporal', type=Path, required=True, help='folder with the temporal states.npy and actions.npy')
    e.add_argument('--weights', type=Path, required=True)
    e.add_argument('--output', type=Path, default=None)
    e.add_argument('--output-tar', action='store_true', help='write a tar stream to stdout instead of --output')
    e.add_argument('--workers', type=int, default=48)
    e.add_argument('--encoders', type=int, default=8)
    e.add_argument('--batch-size', type=int, default=128)
    e.add_argument('--dino-images', type=int, default=1024)
    e.add_argument('--block-frames', type=int, default=64)
    e.add_argument('--gpu-memory-gib', type=float, default=16.)
    e.add_argument('--max-episodes', type=int, default=0)
    e.add_argument('--resume-blocks', default='', help='JSON {partition: first block} (e.g. from resume-point)')
    e.add_argument('--stop-blocks', default='', help='JSON {partition: block to stop before}')
    e.add_argument('--partitions', default='', help='comma list (default both)')
    e.add_argument('--rgb-cache', type=Path, default=None, help='read the stage 1 RGB cache instead of decoding')
    e.add_argument('--decode', choices=('in-order', 'keyframe-skip'), default='in-order',
                   help='keyframe-skip seeks over frames between requests where a keyframe allows it (same pictures; '
                        'fcpg.v21_recorded_clock.decode_at_recorded_times_skipping)')
    e.add_argument('--tag', default='all', help='names this run\'s manifest (several runs may share a part)')
    r = sub.add_parser('resume-point')
    r.add_argument('--store', type=Path, required=True, help='the received part folder')
    r.add_argument('--block-frames', type=int, default=64)
    v = sub.add_parser('verify')
    v.add_argument('--store', type=Path, required=True)
    v.add_argument('--jobs', type=Path, required=True)
    v.add_argument('--block-frames', type=int, default=64)
    v.add_argument('--weights', type=Path, default=None)
    v.add_argument('--spot-rgb-cache', type=Path, default=None)
    v.add_argument('--spot-episodes', type=int, default=6)
    c = sub.add_parser('check-dinov2')
    c.add_argument('--jobs', type=Path, required=True)
    c.add_argument('--repository', type=Path, required=True)
    c.add_argument('--dino-weights', type=Path, required=True)
    c.add_argument('--episodes', type=int, default=6)
    args = p.parse_args()
    if args.command == 'encode' and bool(args.output) == bool(args.output_tar):
        raise ValueError('exactly one of --output and --output-tar')
    dict(jobs=make_jobs, encode=encode, verify=verify, **{'resume-point': resume_point, 'check-dinov2': check_dinov2,
                                                          'extra-jobs': make_extra_jobs})[args.command](args)


if __name__ == '__main__':
    main()
