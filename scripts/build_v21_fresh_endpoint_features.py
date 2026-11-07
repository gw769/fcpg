#!/usr/bin/env python3
"""Extract the 1.2M archive index's block-endpoint pictures (H0-H4, both camera slots) that the old 320k cache does
not have, with the old cache's exact recipe, into the lossless block format of the reused part.

James 2026-10-13: "你搞吧" - store only block endpoints (as the old 320k cache), both views ("柱子也要", "双视角"),
compute once (A100 for DINO), never at training time; "CPU解码完 缓存一下 ... 然后直接给咱们的A100".

Recipe = scripts/rebuild_v21_dino_cache.py (the old cache): fcpg.v21_cache_rebuild.episode_job (recorded-time PTS
decode, aspect letterbox 224 cv2 INTER_AREA, original camera masks, placeholder check), the same view order
[slots[1], slots[0]], DINOv2-S/14 from the archived frozen repository + weights, bf16 compute, fp16 storage, raw
q/action identity check against the temporal index. Storage = fcpg.v21_lossless_features (byte shuffle + zstd1),
blocks of 64 pictures in global-frame order per partition, every block decoded and compared before it is published,
plus per-picture metadata arrays and a metadata checkpoint. Never deletes, never reads fish data, resumable.

Three ways to run (same output format; a later mode resumes what an earlier one wrote):
  (default)          decode workers -> main process runs DINO and writes blocks (main process is the bottleneck)
  --decode-rgb-to D  stage 1: decode workers write each episode's 224 RGB (zstd1, lossless, ~36% of raw) and its
                     metadata straight to D; nothing large passes through the main process
  --rgb-cache D      stage 2: read stage 1's episodes in order, DINO on the GPU in large batches, encode blocks in
                     threads, write the features to --output
Usage on .102 (repo root): python scripts/build_v21_fresh_endpoint_features.py --index <1.2M index> --reference-endpoints
       <old 320k endpoint root> --repository <frozen-dinov2> --weights <dinov2_vits14_pretrain.pth> --output <feature dir>
       [--decode-rgb-to <rgb dir> | --rgb-cache <rgb dir>] [--workers 20] [--encoders 8] [--resume]"""
import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import faulthandler
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpg.v21_recorded_clock import CLOCK_VERSION, EXTRACTOR_VERSION
from fcpg.v21_cache_rebuild import episode_job
from fcpg.v21_lossless_features import LosslessFeatures

META = [('view_valid', (2,), bool), ('capture_times', (2,), np.float64), ('capture_q', (2, 6), np.float32),
        ('video_pts', (2,), np.float64), ('reason', (2,), np.uint8)]
RESULT_KEYS = (('view_valid', 'valid'), ('capture_times', 'capture_times'), ('capture_q', 'capture_q'),
               ('video_pts', 'video_pts'), ('reason', 'reasons'))
RGB_MAGIC = b'V21RGB01'
_WORKER = {}
_ZSTD = LosslessFeatures(level=1, shuffle=True)   # its system-libzstd handle; stage 2 reads the RGB cache with it


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


def plan(index, reference):
    """-> {partition: sorted fresh global frame ids}, episode records by temporal episode, offsets, temporal root."""
    manifest = json.loads((index / 'manifest.json').read_text())
    if not manifest.get('index_complete'):
        raise ValueError('complete identity index required')
    temporal = Path(manifest['temporal_root'])
    offsets = np.load(temporal / 'episode_offsets.npy', allow_pickle=False)
    endpoints = np.load(index / 'endpoint_global_frame_index.npy', allow_pickle=False).astype(np.int64)
    if not (np.diff(endpoints) > 0).all():
        raise ValueError('index endpoints must be sorted unique global frames')
    chain_ep = np.load(index / 'episode_index.npy', allow_pickle=False)
    split = np.load(index / 'split.npy', allow_pickle=False)
    episode_split = {}
    for ep, s in zip(chain_ep, split):
        if episode_split.setdefault(int(ep), int(s)) != int(s):
            raise ValueError('an episode crosses the TRAIN/DEV split')
    old_ep = np.load(reference / 'episode_index.npy', allow_pickle=False).astype(np.int64)
    old_local = np.load(reference / 'local_frame_index.npy', allow_pickle=False).astype(np.int64)
    reused = np.unique(offsets[old_ep] + old_local)
    if len(reused) != len(old_ep) or not np.isin(reused, endpoints).all():
        raise ValueError('old endpoints must be unique and inside the new index')
    fresh = np.setdiff1d(endpoints, reused, assume_unique=True)
    picture_ep = np.searchsorted(offsets, fresh, side='right') - 1
    picture_split = np.asarray([episode_split[int(e)] for e in picture_ep])
    records = {int(r['temporal_episode_index']): r for r in
               json.loads((index / 'episode-path-map.json').read_text())['episodes']}
    parts = {'temporal-train': fresh[picture_split == 0], 'temporal-dev': fresh[picture_split == 1]}
    return parts, records, offsets, temporal, dict(index_endpoints=len(endpoints), reused=len(reused), fresh=len(fresh))


def jobs_for(partition, ids, records, offsets, start_position, max_episodes):
    """Per-episode jobs from start_position's episode on. ids are sorted, so each episode is one contiguous run, found
    with searchsorted (one full-array scan per episode took ~13 min for the train partition while stage 1 ran)."""
    if not (np.diff(ids) > 0).all():
        raise ValueError('partition ids must be sorted unique global frames')
    episodes = np.searchsorted(offsets, ids, side='right') - 1
    first = int(episodes[start_position]) if start_position < len(ids) else None
    unique = np.unique(episodes)
    starts, stops = np.searchsorted(episodes, unique, side='left'), np.searchsorted(episodes, unique, side='right')
    count = 0
    for ep, start, stop in zip(unique, starts, stops):
        if first is None or ep < first:
            continue
        if max_episodes and count >= max_episodes:
            return
        at = np.arange(start, stop)
        record = records[int(ep)]
        if record['frame_start'] != offsets[ep] or record['frame_stop'] != offsets[ep + 1]:
            raise ValueError(f'episode frame range mismatch: {ep}')
        row = dict(record['manifest_row'])
        slots = row['lewm_v12_slot_views']
        count += 1
        yield dict(identity=int(ep), partition=partition, row=row, views=[slots[1], slots[0]], positions=at.tolist(),
                   endpoint_ids=ids[at].tolist(), local=(ids[at] - offsets[ep]).tolist(),
                   frame_start=int(record['frame_start']), frame_stop=int(record['frame_stop']))


def resume_position(root, block_frames):
    """First picture whose block or metadata may be missing: contiguous published blocks, capped by the checkpoint."""
    blocks = 0
    while (root / f'block-{blocks * block_frames:09d}.v21z').exists():
        blocks += 1
    checkpoint = root / 'metadata-checkpoint.json'
    saved = json.loads(checkpoint.read_text())['position'] if checkpoint.exists() else blocks * block_frames
    return min(blocks * block_frames, saved) // block_frames * block_frames


def check_motion(result, q_bank, a_bank, job):
    q, a = q_bank[job['frame_start']:job['frame_stop']], a_bank[job['frame_start']:job['frame_stop']]
    if not np.allclose(result['q'], q, atol=.02, rtol=0) or not np.allclose(result['actions'], a, atol=.02, rtol=0):
        raise ValueError(f'raw motion identity mismatch: episode {job["identity"]}')


def gpu_fp16_tokens(backbone, rgb, device, batch_size):
    """fcpg.prod_v18_1_training.extract_dinov2_tokens, but the float->fp16 rounding (round to nearest even, as numpy's
    astype) runs on the GPU and half the bytes cross PCIe; the original spent ~94% of its time in that copy and cast."""
    import torch
    from fcpg.prod_v18_1_training import _rgb_tensor
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    outputs = []
    for start in range(0, len(rgb), batch_size):
        pixels = _rgb_tensor(rgb[start:start + batch_size], device=device)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=dtype):
            features = backbone.forward_features(pixels)
        tokens = features['x_norm_patchtokens']
        if tokens.shape != (len(pixels), 256, 384) or not bool(torch.isfinite(tokens).all()):
            raise ValueError('DINOv2 patch tokens have the wrong shape or are not finite')
        outputs.append(tokens.float().to(torch.float16).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def encode_block(path, block):
    """Encode one block, check it decodes to the same bits, publish it atomically -> stored bytes (thread-safe)."""
    codec = LosslessFeatures(level=1, shuffle=True)
    payload = codec.encode(block)
    restored = codec.decode(payload)
    if restored.shape != block.shape or not np.array_equal(restored.view(np.uint16), block.view(np.uint16)):
        raise ValueError('lossless block does not decode to the same bits')
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(payload)
    temporary.replace(path)
    return len(payload)


# ---------------------------------------------------------------- stage 1: CPU decode -> lossless 224 RGB cache

def rgb_files(root, partition, ep):
    folder = Path(root) / partition
    return folder / f'ep-{ep:06d}.rgb', folder / f'ep-{ep:06d}.meta.npz'


def rgb_worker_init(temporal, root):
    os.nice(5)
    _WORKER.update(q=np.load(Path(temporal) / 'states.npy', mmap_mode='r'),
                   a=np.load(Path(temporal) / 'actions.npy', mmap_mode='r'), root=root)


def decode_rgb_episode(job):
    import pyarrow as pa
    rgb_path, meta_path = rgb_files(_WORKER['root'], job['partition'], job['identity'])
    if meta_path.exists():
        return dict(identity=job['identity'], skipped=True)
    result = episode_job(job)
    check_motion(result, _WORKER['q'], _WORKER['a'], job)
    rgb = np.ascontiguousarray(result['rgb'])
    payload = pa.Codec('zstd', compression_level=1).compress(rgb.reshape(-1))
    header = json.dumps(dict(shape=list(rgb.shape), dtype='|u1', raw_bytes=int(rgb.nbytes), codec='zstd1')).encode()
    temporary = rgb_path.with_suffix('.rgb-tmp')
    with open(temporary, 'wb') as handle:
        handle.write(RGB_MAGIC + struct.pack('<I', len(header)) + header)
        handle.write(memoryview(payload))
    temporary.replace(rgb_path)
    buffer = io.BytesIO()
    np.savez(buffer, positions=np.asarray(job['positions'], np.int64), endpoint_ids=np.asarray(job['endpoint_ids'], np.int64),
             valid=result['valid'], capture_times=result['capture_times'], capture_q=result['capture_q'],
             video_pts=result['video_pts'], reasons=result['reasons'], decode_s=np.float64(result['elapsed_s']))
    temporary = meta_path.with_suffix('.meta-tmp')
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(meta_path)                       # the episode counts as done only once this file exists
    return dict(identity=job['identity'], pictures=len(job['positions']), raw_bytes=int(rgb.nbytes),
                stored_bytes=int(payload.size), decode_s=float(result['elapsed_s']))


def read_rgb_episode(root, job):
    """Stage 1's episode back from the RGB cache. The zstd1 frames are read with the system libzstd: pyarrow's
    decompress segfaulted twice inside libarrow at the start of the dev partition (10/13 23:49 and 23:55)."""
    rgb_path, meta_path = rgb_files(root, job['partition'], job['identity'])
    with np.load(meta_path, allow_pickle=False) as meta:
        result = {k: meta[k] for k in ('positions', 'endpoint_ids', 'valid', 'capture_times', 'capture_q', 'video_pts', 'reasons')}
    if not np.array_equal(result['positions'], job['positions']) or not np.array_equal(result['endpoint_ids'], job['endpoint_ids']):
        raise ValueError(f'RGB cache identities differ for episode {job["identity"]}')
    data = rgb_path.read_bytes()
    if data[:8] != RGB_MAGIC:
        raise ValueError(f'not an RGB cache file: {rgb_path}')
    size = struct.unpack('<I', data[8:12])[0]
    header = json.loads(data[12:12 + size])
    rgb = np.empty(header['shape'], np.uint8)
    payload = data[12 + size:]
    if rgb.nbytes != header['raw_bytes'] or header.get('codec') != 'zstd1' or _ZSTD.checked(
            _ZSTD.lib.ZSTD_decompress(rgb.ctypes.data, rgb.nbytes, payload, len(payload))) != rgb.nbytes:
        raise ValueError(f'RGB cache payload does not decode to its declared size: {rgb_path}')
    result['rgb'] = rgb
    result['identity'] = job['identity']
    return result


def decode_stage(args, parts, records, offsets, temporal):
    root = args.decode_rgb_to
    for partition in parts:
        (root / partition).mkdir(parents=True, exist_ok=True)
    jobs = []
    for partition, ids in parts.items():
        start = resume_position(args.output / partition, args.block_frames) if (args.output / partition).exists() else 0
        jobs += [j for j in jobs_for(partition, ids, records, offsets, start, args.max_episodes)
                 if not rgb_files(root, partition, j['identity'])[1].exists()]
    write_json(root / 'manifest.json', dict(schema='v21-1p2m-fresh-endpoint-rgb-cache-v1', index=str(args.index.resolve()),
        format='per episode: V21RGB01 header + zstd1 of uint8 [pictures, 2, 224, 224, 3] (letterboxed, invalid views zero)'
               ' and .meta.npz (positions, endpoint_ids, valid, capture_times, capture_q, video_pts, reasons)',
        view_order='[slots[1], slots[0]] as the old cache', decode='fcpg.v21_cache_rebuild.episode_job',
        motion_identity_checked=True, pending_episodes_at_start=len(jobs)))
    started, done, pictures, raw, stored, decode_s, last = time.monotonic(), 0, 0, 0, 0, 0., 0.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=rgb_worker_init, initargs=(str(temporal), str(root))) as pool:
        stream, running = iter(jobs), set()
        for job in stream:
            running.add(pool.submit(decode_rgb_episode, job))
            if len(running) < 2 * args.workers:
                continue
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                r = future.result()
                done += 1
                if not r.get('skipped'):
                    pictures += r['pictures']; raw += r['raw_bytes']; stored += r['stored_bytes']; decode_s += r['decode_s']
            elapsed = time.monotonic() - started
            if elapsed - last > 10:
                last = elapsed
                write_json(root / 'progress.json', dict(episodes_done=done, episodes_total=len(jobs), pictures=pictures,
                    raw_gb=raw / 1e9, stored_gb=stored / 1e9, elapsed_s=elapsed, pictures_per_s=pictures / elapsed,
                    worker_decode_s=decode_s, workers=args.workers))
        for future in running:
            r = future.result()
            done += 1
            if not r.get('skipped'):
                pictures += r['pictures']; raw += r['raw_bytes']; stored += r['stored_bytes']; decode_s += r['decode_s']
    elapsed = time.monotonic() - started
    write_json(root / 'progress.json', dict(episodes_done=done, episodes_total=len(jobs), pictures=pictures, raw_gb=raw / 1e9,
        stored_gb=stored / 1e9, elapsed_s=elapsed, pictures_per_s=pictures / max(elapsed, 1e-6), worker_decode_s=decode_s,
        workers=args.workers, complete=done == len(jobs)))
    print(json.dumps(dict(episodes=done, pictures=pictures, stored_gb=stored / 1e9, seconds=elapsed)), flush=True)


# ---------------------------------------------------------------- feature stage (default, or from the RGB cache)

def ordered_decode(pool, jobs, depth):
    """In-order results of fcpg.v21_cache_rebuild.episode_job with a deep window (one slow episode stalls less)."""
    stream, pending = iter(jobs), deque()
    for _ in range(depth):
        job = next(stream, None)
        if job is not None:
            pending.append(pool.submit(episode_job, job))
    while pending:
        result = pending.popleft().result(timeout=1800)
        job = next(stream, None)
        if job is not None:
            pending.append(pool.submit(episode_job, job))
        yield result


def ordered_rgb_cache(root, jobs, depth, wait_s=7200):
    """In-order episodes from the stage 1 RGB cache, read ahead in threads; waits for episodes stage 1 has not written."""
    def load(job):
        meta_path = rgb_files(root, job['partition'], job['identity'])[1]
        deadline = time.monotonic() + wait_s
        while not meta_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f'stage 1 never wrote episode {job["identity"]}')
            time.sleep(2)
        return read_rgb_episode(root, job)
    with ThreadPoolExecutor(max_workers=3) as readers:
        stream, pending = iter(jobs), deque()
        for _ in range(depth):
            job = next(stream, None)
            if job is not None:
                pending.append(readers.submit(load, job))
        while pending:
            result = pending.popleft().result()
            job = next(stream, None)
            if job is not None:
                pending.append(readers.submit(load, job))
            yield result


def main():
    faulthandler.enable()                          # a native crash now prints the Python stacks
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--reference-endpoints', type=Path, required=True)
    p.add_argument('--repository', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--decode-rgb-to', type=Path, default=None)
    p.add_argument('--rgb-cache', type=Path, default=None)
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--encoders', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--dino-images', type=int, default=512, help='images per DINO call (grouped across episodes)')
    p.add_argument('--block-frames', type=int, default=64)
    p.add_argument('--gpu-memory-gib', type=float, default=12.)
    p.add_argument('--max-episodes', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not 1 <= args.workers <= 24 or not 1 <= args.encoders <= 16 or (args.decode_rgb_to and args.rgb_cache):
        raise ValueError('1..24 decode workers, 1..16 encoders, and at most one of --decode-rgb-to / --rgb-cache')
    parts, records, offsets, temporal, counts = plan(args.index, args.reference_endpoints)
    if args.decode_rgb_to:
        return decode_stage(args, parts, records, offsets, temporal)
    extraction = dict(clock_version=CLOCK_VERSION, extractor_version=EXTRACTOR_VERSION,
        repository=str(args.repository), weights=str(args.weights), weights_sha256=sha256(args.weights),
        input_rgb=True, image_size=224, resize='aspect-letterbox-cv2-INTER_AREA',
        normalization=dict(mean=[.485, .456, .406], std=[.229, .224, .225]),
        compute_precision='bf16', storage_precision='float16', view_order='[slots[1], slots[0]] as the old cache',
        camera_role_policy='preserve historical slots; archive physical roles unverified, no automatic swaps',
        reason_codes={'0': 'valid', '1': 'missing-or-outside-video-interval', '2': 'no-causal-capture-readback',
                      '3': 'original-camera-mask', '4': 'placeholder'})
    contract = dict(schema='v21-1p2m-fresh-endpoint-lossless-v1', index=str(args.index.resolve()),
        reference_endpoints=str(args.reference_endpoints.resolve()), pictures='block endpoints H0-H4, both slots',
        codec='byte-shuffle-zstd1', storage_dtype='float16', block_frames=args.block_frames,
        patch_shape=[2, 256, 384], partitions={k: int(len(v)) for k, v in parts.items()}, counts=counts,
        extraction=extraction, max_episodes=args.max_episodes)
    manifest_path = args.output / 'manifest.json'
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(k) != v for k, v in contract.items()):
            raise ValueError('resume contract changed')
    else:
        need = sum(map(len, parts.values())) * 2 * 256 * 384 * 2 * (1 if not args.max_episodes else 0) + 16 * 2**30
        if shutil.disk_usage(args.output.parent).free < need:
            raise RuntimeError('not enough HDD space for the uncompressed worst case plus 16 GiB')
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(manifest_path, dict(contract, complete=False))
    import torch
    from fcpg.prod_v18_1_training import load_frozen_dinov2, extract_dinov2_tokens
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(
        min(args.gpu_memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory, .9), device)
    backbone = load_frozen_dinov2(args.repository, args.weights, device=device)
    probe = np.random.default_rng(0).integers(0, 256, (48, 224, 224, 3), dtype=np.uint8)
    if not np.array_equal(gpu_fp16_tokens(backbone, probe, device, 16).view(np.uint16),
                          extract_dinov2_tokens(backbone, probe, device=device, batch_size=16).view(np.uint16)):
        raise ValueError('GPU fp16 conversion is not bit-identical to the original extraction helper')
    q_bank, a_bank = np.load(temporal / 'states.npy', mmap_mode='r'), np.load(temporal / 'actions.npy', mmap_mode='r')
    started, done_pictures, images, stored, last_report = time.monotonic(), 0, 0, 0, 0.
    timers = dict(wait_input=0., dino=0., bookkeeping=0., wait_encode=0.)
    encoders = ThreadPoolExecutor(max_workers=args.encoders)
    pending = deque()

    def drain(limit):
        nonlocal stored
        while len(pending) > limit:
            clock = time.monotonic()
            stored += pending.popleft().result()
            timers['wait_encode'] += time.monotonic() - clock
    try:
        for partition, ids in parts.items():
            root = args.output / partition
            root.mkdir(exist_ok=True)
            gpath = root / 'global-frame-index.npy'
            if gpath.exists():
                if not np.array_equal(np.load(gpath), ids):
                    raise ValueError('resume identities changed')
            else:
                np.save(gpath, ids, allow_pickle=False)
            meta = {}
            for name, trailing, dtype in META:
                path = root / f'{name}.npy'
                meta[name] = (np.load(path, mmap_mode='r+') if path.exists() else
                              np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=(len(ids), *trailing)))
            checkpoint_path = root / 'metadata-checkpoint.json'
            position = resume_position(root, args.block_frames)
            buffer, buffer_start, blocks_since_flush = [], position, 0
            jobs = list(jobs_for(partition, ids, records, offsets, position, args.max_episodes))
            pool = (None if args.rgb_cache else
                    ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                                        initializer=os.nice, initargs=(5,)))
            stream = (ordered_rgb_cache(args.rgb_cache, jobs, 8) if args.rgb_cache
                      else ordered_decode(pool, jobs, 2 * args.workers))
            group = []
            try:
                for number, job in enumerate(jobs):
                    clock = time.monotonic()
                    result = next(stream)
                    timers['wait_input'] += time.monotonic() - clock
                    if result['identity'] != job['identity']:
                        raise ValueError('episode order mismatch')
                    if not args.rgb_cache:                       # stage 1 already checked the motion identity
                        check_motion(result, q_bank, a_bank, job)
                    group.append((job, result))
                    if sum(int(r['valid'].sum()) for _, r in group) < args.dino_images and number + 1 < len(jobs):
                        continue
                    clock = time.monotonic()
                    images_in = [r['rgb'][r['valid']] for _, r in group]
                    encoded = (gpu_fp16_tokens(backbone, np.concatenate(images_in), device, args.batch_size)
                               if sum(map(len, images_in)) else np.zeros((0, 256, 384), np.float16))
                    timers['dino'] += time.monotonic() - clock
                    clock = time.monotonic()
                    cut = np.cumsum([0] + [len(x) for x in images_in])
                    finished, group = group, []
                    for k, (job_k, result_k) in enumerate(finished):
                        valid = result_k['valid']
                        tokens = np.zeros((len(job_k['positions']), 2, 256, 384), np.float16)
                        if valid.any():
                            rows, views = np.nonzero(valid)
                            tokens[rows, views] = encoded[cut[k]:cut[k + 1]]
                        positions = np.asarray(job_k['positions'])
                        keep = positions >= buffer_start + len(buffer)   # drops only already-published pictures on resume
                        if keep.any() and not np.array_equal(positions[keep], np.arange(
                                buffer_start + len(buffer), buffer_start + len(buffer) + keep.sum())):
                            raise ValueError('pictures must arrive in contiguous global-frame order')
                        for name, key in RESULT_KEYS:
                            meta[name][positions[keep]] = result_k[key][keep]
                        buffer.extend(tokens[keep])
                        images += int(valid[keep].sum())
                        while len(buffer) >= args.block_frames or (buffer and buffer_start + len(buffer) == len(ids)):
                            n = min(args.block_frames, len(buffer))
                            block = np.stack(buffer[:n])
                            pending.append(encoders.submit(encode_block, root / f'block-{buffer_start:09d}.v21z', block))
                            blocks_since_flush += 1
                            if blocks_since_flush >= 32:
                                for array in meta.values():
                                    array.flush()
                                write_json(checkpoint_path, dict(position=int(buffer_start + n)))
                                blocks_since_flush = 0
                            drain(2 * args.encoders)
                            done_pictures += n
                            buffer, buffer_start = buffer[n:], buffer_start + n
                    timers['bookkeeping'] += time.monotonic() - clock
                    elapsed = time.monotonic() - started
                    if elapsed - last_report > 10:
                        last_report = elapsed
                        write_json(args.output / 'progress.json', dict(partition=partition, episode=job['identity'],
                            partition_pictures_submitted=buffer_start, partition_pictures=int(len(ids)),
                            pictures_submitted_this_run=done_pictures, valid_images_this_run=images,
                            compressed_bytes_published_this_run=stored, elapsed_s=elapsed,
                            pictures_per_s=done_pictures / max(elapsed, 1e-6), mode='rgb-cache' if args.rgb_cache else 'decode',
                            main_thread_seconds={k: round(v, 1) for k, v in timers.items()}, gpu_used=True, fish_read=False))
            finally:
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
            drain(0)                                      # every block of this partition is published
            for array in meta.values():
                array.flush()
            write_json(checkpoint_path, dict(position=int(buffer_start)))
            if args.max_episodes:
                break
        final = dict(contract, complete=not args.max_episodes and all(
            (args.output / k / f'block-{((len(v) - 1) // args.block_frames) * args.block_frames:09d}.v21z').exists()
            for k, v in parts.items()), elapsed_s=time.monotonic() - started)
        write_json(manifest_path, final)
        print(json.dumps(dict(published=done_pictures, images=images, compressed_gb=stored / 1e9,
                              seconds=time.monotonic() - started, complete=final['complete'])), flush=True)
    except BaseException as error:
        write_json(args.output / 'error.json', dict(error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        encoders.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    main()
