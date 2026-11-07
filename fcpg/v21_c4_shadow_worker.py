"""C4 shadow worker (docs/V21_LEWM_C4_STEP2_20261010.md sections 3-4): a separate interpreter fed by the tap.

It reads tap messages from stdin and writes only its own records under --output. It imports no robot,
serial, camera, service, Actor or CEM code, and the tap never reads anything back from it.

Per camera frame pair (the 10 Hz grid): RGB (as lerobot's cameras deliver) -> BGR -> the cache's letterbox
(OpenCV INTER_AREA, fcpg.lerobot_archive.letterbox_rgb) -> the cache's automatic admission of that camera's
exposure (camera_check: empty placeholder, raw neutral fill) -> DINOv2-S/14 patch tokens of the admitted
cameras (extract_dinov2_tokens: fp16 autocast on a V100, stored fp16 like the cache). A camera that fails the
check is masked like a missing one; the tick's joints and command and the other camera are kept.

The factual command of a tick is what the follower reported as actually sent (`sent`, after its own limits)
when the tap has it, else the requested command; the two are compared and their difference reported.

Per window t of the offline test's population (10 Hz grid, t from 1.0 s after the segment starts to 31
frames before its last frame), once the ticks it needs are in: the offline test's eight candidates
(fcpg.v21_lewm_c4_scorer.offline_candidates; partner = the next window of the same block of 32 whose
commands are complete, cyclically, as batch_partners) scored by C4 against the frame actually captured at
t+4.0 s (the hindsight test), and the factual rollout's per-camera squared error at t+0.5 ... t+4.0 s set
against persistence (scripts/eval_v21_lewm_standalone.py rule). Frame clock = command index (frame k at
k/30 s), as LeRobot records it; the actual loop times are kept and their offsets reported per window.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fcpg.v21_camera_quality import CAMERA_QUALITY_VERSION, raw_camera_quality

REPO = Path(__file__).resolve().parents[1]
STEP = 3              # frames per 10 Hz command at 30 fps
HALF = 15             # frames per 0.5 s
PAST = 30             # t-1.0 s
GOAL = 120            # t+4.0 s
TAIL = 31             # the offline population: windows t with t + 31 <= the segment's last frame
BLOCK = 32            # the offline evaluation batch; partners never leave it
COMMANDS = 50
NEAR_TIE = .01
OFFSETS = [-PAST, -HALF, 0] + [HALF * k for k in range(1, 9)]   # t-1.0, t-0.5, t, t+0.5 ... t+4.0
VIEWS = ('wrist', 'column')
PATCHES, DINO_DIM = 256, 384
FPS = 30.


def _six(values):
    out = np.asarray(values, np.float32)
    if out.shape != (6,) or not np.isfinite(out).all():
        raise ValueError('six finite joints expected')
    return out


def letterbox(rgb):
    """lerobot's RGB frame -> the cache's 224 input (fcpg.lerobot_archive.letterbox_rgb takes BGR, returns RGB)."""
    from fcpg.lerobot_archive import letterbox_rgb
    return letterbox_rgb(np.ascontiguousarray(rgb[..., ::-1]), image_size=224)


def camera_check(rgb, boxed):
    """The cache's admission of one camera's exposure that did arrive (fcpg/v21_cache_rebuild.py episode_job),
    -> (reason, detail): 0 admitted; 4 an empty placeholder (the 224 input's max or range <= 2); 5 the raw
    neutral-fill check (fcpg.v21_camera_quality.raw_camera_quality on the raw RGB, before the letterbox).
    Reason 6 (an exposure withdrawn by the offline review) has no live counterpart."""
    if int(boxed.max()) <= 2 or int(np.ptp(boxed)) <= 2:
        return 4, 'placeholder'
    quality = raw_camera_quality(rgb)
    return (0, None) if quality['valid'] else (5, quality['reason'])


class Recording:
    """One segment's ticks and frame tokens, and the offline test's window geometry over them."""

    def __init__(self, segment, meta):
        self.segment, self.meta = int(segment), dict(meta)
        self.ticks = {}        # index -> (state [6] float32, action [6] float32, t_obs)
        self.tokens = {}       # index -> (tokens [2, 256, 384] float16, valid [2] bool)
        self.frame_ready = {}  # index -> seconds from the control loop's observation to tokens ready
        self.last = -1
        self.ended = False
        self.next_window = 0
        self.row_base = int(self.meta.get('row_base', 1_000_000 * (self.segment + 1)))
        self.summary = dict(windows=0, ranked=0, unranked={}, scored_errors=0, ticks=0, masked_frames={},
                            sent_reported=0, sent_differs=0, sent_max_abs_difference=0.)

    @staticmethod
    def t_of(w):
        return PAST + STEP * w

    def add_tick(self, index, state, action, t_obs, requested=None):
        """`action` is the factual command (what the follower sent); `requested`, when the tap has both, is what
        was asked of it, kept only to report the difference."""
        action = _six(action)
        if requested is not None:
            difference = float(np.abs(action.astype(np.float64) - _six(requested)).max())
            self.summary['sent_reported'] += 1
            self.summary['sent_differs'] += int(difference > 0)
            self.summary['sent_max_abs_difference'] = max(self.summary['sent_max_abs_difference'], difference)
        self.ticks[int(index)] = (_six(state), action, t_obs)
        self.summary['ticks'] += 1
        self.last = max(self.last, int(index))

    def settled(self, index):
        """Every tick up to `index` has either arrived or never will (ticks arrive in order)."""
        return self.ended or self.last >= index

    def command_frames(self, t):
        return [t - PAST + STEP * k for k in range(COMMANDS)]

    def command_valid(self, t):
        """Offline rule (v21_recorded_clock.future_grid): a command counts when its frame and the next 10 Hz
        readback both exist; here a frame exists when its tick arrived (a dropped tick never does)."""
        return np.array([f in self.ticks and f + STEP in self.ticks for f in self.command_frames(t)])

    def commands(self, t):
        out = np.zeros((COMMANDS, 6), np.float32)
        for k, f in enumerate(self.command_frames(t)):
            if f in self.ticks:
                out[k] = self.ticks[f][1]
        return out

    def complete_known(self, w):
        t = self.t_of(w)
        if not self.settled(t + GOAL):        # the last command (t+117) needs the readback at t+120
            return False, False
        return True, bool(self.command_valid(t).all())

    def member_known(self, w):
        t = self.t_of(w)
        if self.last >= t + TAIL:
            return True, True
        return (True, False) if self.ended else (False, False)

    def _next_slot(self, j, b0):
        """The block slot after j, cyclically; None = not known yet."""
        if j + 1 == b0 + BLOCK:
            return b0
        known, member = self.member_known(j + 1)
        if not known:
            return None
        return j + 1 if member else b0

    def partner(self, w):
        """fcpg.v21_lewm_c4_scorer.batch_partners inside w's block of 32: the next window (cyclically) whose 50
        commands are all valid, else the next window. None = not known yet."""
        b0 = w - w % BLOCK
        first = j = self._next_slot(w, b0)
        while j is not None and j != w:
            known, complete = self.complete_known(j)
            if not known:
                return None
            if complete:
                return j
            j = self._next_slot(j, b0)
        return None if j is None else first

    def settle_next(self):
        """The next population window with everything it needs, else None (wait for more ticks)."""
        w = self.next_window
        t = self.t_of(w)
        known, member = self.member_known(w)
        if not known or not member or not self.settled(t + GOAL):
            return None
        valid = self.command_valid(t)
        frames = [t + d for d in OFFSETS]
        view_valid = np.array([self.tokens[f][1] if f in self.tokens else np.zeros(2, bool) for f in frames])
        history_ok = bool(view_valid[:3].any(-1).all())
        goal_ok = view_valid[-1] & history_ok & bool(valid.all())        # the offline test's goal_ok
        item = dict(w=w, t=t, row=self.row_base + w, frames=frames, view_valid=view_valid,
                    command_valid=valid, history_ok=history_ok, goal_ok=goal_ok)
        if goal_ok.any():
            p = self.partner(w)
            if p is None:
                return None
            item['partner'] = p
        self.next_window += 1
        return item

    def prune(self):
        """Keep what unsettled windows and their block's partners can still need."""
        b0 = self.next_window - self.next_window % BLOCK
        keep = self.t_of(b0) - PAST - STEP
        for book in (self.ticks, self.tokens, self.frame_ready):
            for k in [k for k in book if k < keep]:
                del book[k]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_model(path, device):
    import torch
    from fcpg.v21_lewm_standalone import StandaloneLeWM
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = StandaloneLeWM(**payload['model_config'])
    model.load_state_dict(payload['state_dict'])
    return model.to(device).eval().requires_grad_(False), payload


def make_dino(repo, weights, device):
    """The cache's DINO for 224 inputs made by `letterbox` -> [N, 256, 384] float16 tokens."""
    from fcpg.prod_v18_1_training import extract_dinov2_tokens, load_frozen_dinov2
    backbone = load_frozen_dinov2(repo, weights, device=device)

    def tokens(boxed_frames):
        return extract_dinov2_tokens(backbone, np.stack(boxed_frames), device=device)
    return tokens


class Worker:
    def __init__(self, model, dino, output, *, device, dtype, clock=time.monotonic):
        import torch
        self.torch, self.model, self.dino = torch, model, dino
        self.device, self.dtype, self.clock = device, dtype, clock
        self.output = Path(output)
        self.logs = {name: open(self.output / f'{name}.jsonl', 'a', buffering=1)
                     for name in ('windows', 'frames', 'segments', 'errors')}
        self.recordings = {}
        self.tap_counts = None
        self.messages = 0

    def write(self, name, record):
        self.logs[name].write(json.dumps(record, allow_nan=False, default=float) + '\n')

    # ---------------------------------------------------------------- messages
    def handle(self, header, arrays):
        self.messages += 1
        kind = header.get('type')
        try:
            if kind == 'tick':
                self.tap_counts = header.get('counts')
                r = self.recording(header['segment'], header.get('meta', {}))
                sent = header.get('sent')
                r.add_tick(header['index'], header['state'], header['action'] if sent is None else sent,
                           header.get('t_obs'), requested=None if sent is None else header['action'])
                if any(view in arrays for view in VIEWS):
                    self.frame(r, header, arrays)
            elif kind == 'segment_begin':
                self.recording(header['segment'], header.get('meta', {}))
            elif kind == 'segment_end':
                r = self.recording(header['segment'], {})
                r.summary['flags'] = header.get('flags', {})
                self.end(r)
        except Exception as error:                              # one message lost, the worker goes on
            self.write('errors', dict(message=self.messages, type=kind, segment=header.get('segment'),
                                      index=header.get('index'), error=f'{type(error).__name__}: {error}'))
        self.settle()

    def recording(self, segment, meta):
        if segment not in self.recordings:
            for r in self.recordings.values():      # segments never interleave: a new one ends the others
                if not r.ended:
                    self.end(r)
            self.recordings[segment] = Recording(segment, meta)
            fps = self.recordings[segment].meta.get('fps')
            if fps is not None and float(fps) != FPS:   # the window geometry is frames at 30 fps
                self.write('errors', dict(segment=segment, error=f'segment fps {fps} is not {FPS:g}'))
        return self.recordings[segment]

    def frame(self, r, header, arrays):
        received = self.clock()
        present = [view in arrays for view in VIEWS]
        reasons, boxed = [None] * len(VIEWS), {}
        for c, view in enumerate(VIEWS):
            if present[c]:
                b = letterbox(arrays[view])
                reasons[c] = camera_check(arrays[view], b)
                if reasons[c][0] == 0:
                    boxed[c] = b
                else:
                    key = f'{view}:{reasons[c][0]}'
                    r.summary['masked_frames'][key] = r.summary['masked_frames'].get(key, 0) + 1
        valid = np.array([c in boxed for c in range(len(VIEWS))])
        tokens = np.zeros((len(VIEWS), PATCHES, DINO_DIM), np.float16)
        if boxed:
            tokens[sorted(boxed)] = self.dino([boxed[c] for c in sorted(boxed)])
        ready = self.clock()
        index = int(header['index'])
        r.tokens[index] = (tokens, valid)
        t_obs = header.get('t_obs')
        r.frame_ready[index] = None if t_obs is None else ready - t_obs
        self.write('frames', dict(segment=r.segment, index=index, views=present, valid=valid.tolist(),
                                  reasons=[None if x is None else x[0] for x in reasons],
                                  quality=[None if x is None else x[1] for x in reasons], t_obs=t_obs,
                                  queue_s=None if t_obs is None else received - t_obs, dino_s=ready - received,
                                  ready_s=r.frame_ready[index]))

    def end(self, r):
        r.ended = True
        self.settle()

    # ---------------------------------------------------------------- windows
    def settle(self):
        for r in self.recordings.values():
            if r.summary.get('closed'):
                continue
            while True:
                item = r.settle_next()
                if item is None:
                    break
                try:
                    self.window(r, item)
                except Exception as error:                      # one window lost, the worker goes on
                    r.summary['scored_errors'] += 1
                    self.write('errors', dict(segment=r.segment, w=item['w'], t=item['t'],
                                              error=f'{type(error).__name__}: {error}'))
            r.prune()
            if r.ended:
                r.summary['closed'] = True
                self.write('segments', dict(segment=r.segment, meta=r.meta, last_index=r.last, **r.summary))

    def window(self, r, item):
        torch = self.torch
        from fcpg.v21_lewm_c4_scorer import offline_candidates, rank_of_first, score_embeddings
        start = self.clock()
        t, w = item['t'], item['w']
        r.summary['windows'] += 1
        record = dict(type='window', segment=r.segment, episode=r.meta.get('episode'), phase=r.meta.get('phase'),
                      w=w, t=t, row=item['row'], view_valid=item['view_valid'].tolist(),
                      commands_valid=int(item['command_valid'].sum()), ranked=bool(item['goal_ok'].any()))
        if not item['history_ok']:
            r.summary['unranked']['history'] = r.summary['unranked'].get('history', 0) + 1
            record.update(reason='history')
            self.write('windows', record)
            return
        tokens = np.stack([r.tokens[f][0] if f in r.tokens else np.zeros((2, PATCHES, DINO_DIM), np.float16)
                           for f in item['frames']])
        patches = torch.from_numpy(tokens).to(self.device).float()[None]   # fp16 values; autocast casts back exactly
        valid = torch.from_numpy(item['view_valid']).to(self.device)[None]
        commands = torch.from_numpy(r.commands(t))
        command_valid = torch.from_numpy(item['command_valid'])
        if record['ranked']:
            p = item['partner']
            q_now = torch.from_numpy(r.ticks[t][0])
            candidates = offline_candidates(commands, command_valid, q_now, item['row'],
                                            torch.from_numpy(r.commands(r.t_of(p))))
            record.update(partner=p, partner_t=r.t_of(p), partner_row=r.row_base + p)
        else:
            candidates = commands[None]
            missing = 'goal' if not item['view_valid'][-1].any() else 'commands'
            r.summary['unranked'][missing] = r.summary['unranked'].get(missing, 0) + 1
            record.update(reason=missing)
        with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None):
            emb = self.model.encode(patches, valid)[0]
            result = score_embeddings(self.model, emb[:3], valid[0, :3], candidates.to(self.device),
                                      command_valid.to(self.device), emb[-1], valid[0, -1])
        # factual rollout vs persistence, the eval's admission rule
        blocks_ok = command_valid.reshape(10, 5).all(-1).int().cumprod(0).bool()[2:].to(self.device)   # [8]
        admitted = valid[0, 3:] & valid[0, 2:3] & blocks_ok[:, None]
        target, current = emb[3:].float(), emb[2:3].float()
        error = ((result['predicted'][0].float() - target) ** 2).mean(-1)
        persistence = ((current - target) ** 2).mean(-1)
        record.update(error=torch.where(admitted, error, torch.zeros_like(error)).tolist(),
                      persistence=torch.where(admitted, persistence, torch.zeros_like(persistence)).tolist(),
                      admitted=admitted.tolist())
        if record['ranked']:
            c = result['costs'].float().cpu().numpy().astype(np.float64)
            gap = np.abs(np.delete(c, 0) - c[0]).min() / max(abs(c[0]), 1e-12)
            q = r.ticks[t][0]
            record.update(costs=c.tolist(), rank=rank_of_first(c), cheapest=int(c.argmin()), near_tie=bool(gap < NEAR_TIE),
                          goal_cameras=result['goal_cameras'],
                          moving=bool(np.abs(r.commands(t)[-1] - q).max() > 10.))
            r.summary['ranked'] += 1
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        done = self.clock()
        anchors = {'t-1.0': t - PAST, 't-0.5': t - HALF, 't': t, 't+4.0': t + GOAL}
        base = r.ticks[t][2] if t in r.ticks else None
        record['offsets_s'] = {name: (None if base is None or f not in r.ticks or r.ticks[f][2] is None
                                      else r.ticks[f][2] - base - (f - t) / FPS) for name, f in anchors.items()}
        goal_obs = r.ticks[t + GOAL][2] if t + GOAL in r.ticks else None
        record['latency_s'] = dict(score=done - start, frame_t=r.frame_ready.get(t),
                                   planning_equivalent=None if r.frame_ready.get(t) is None
                                   else r.frame_ready[t] + (done - start),
                                   since_goal_observed=None if goal_obs is None else done - goal_obs)
        self.write('windows', record)

    def warm_up(self, height=480, width=640):
        """CUDA kernels load on first use; do it before the tap is told the worker is ready."""
        torch = self.torch
        from fcpg.v21_lewm_c4_scorer import score_embeddings
        frames = [np.random.default_rng(k).integers(0, 256, (height, width, 3), dtype=np.uint8) for k in range(2)]
        boxed = [letterbox(f) for f in frames]
        if [camera_check(f, b)[0] for f, b in zip(frames, boxed)] != [0, 0]:
            raise RuntimeError('the camera check refused random warm-up frames')
        for _ in range(3):
            self.dino(boxed)
        patches = torch.zeros((1, len(OFFSETS), 2, PATCHES, DINO_DIM), device=self.device)
        valid = torch.ones((1, len(OFFSETS), 2), dtype=torch.bool, device=self.device)
        commands = torch.zeros((8, COMMANDS, 6), device=self.device)
        command_valid = torch.ones(COMMANDS, dtype=torch.bool, device=self.device)
        for _ in range(3):
            with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None):
                emb = self.model.encode(patches, valid)[0]
                score_embeddings(self.model, emb[:3], valid[0, :3], commands, command_valid, emb[-1], valid[0, -1])
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def finish(self):
        for r in self.recordings.values():
            if not r.ended:
                self.end(r)
        self.settle()
        summary = dict(schema='v21-c4-shadow-worker-summary-v2', messages=self.messages,
                       ticks_received=sum(r.summary['ticks'] for r in self.recordings.values()),
                       tap_counts=self.tap_counts,
                       segments={r.segment: {k: v for k, v in r.summary.items()} for r in self.recordings.values()})
        (self.output / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
        for f in self.logs.values():
            f.close()


def main(argv=None):
    from fcpg import v21_c4_shadow_protocol as protocol
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dino-repo', type=Path, required=True)
    p.add_argument('--dino-weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--precision', choices=('fp16', 'fp32'), default='fp16')
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'windows.jsonl').exists():
        raise SystemExit('this output already has windows.jsonl')
    import torch
    torch.set_num_threads(2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, payload = load_model(args.checkpoint, device)
    dino = make_dino(args.dino_repo, args.dino_weights, device)
    try:
        commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    code = {name: sha256(REPO / name) for name in (
        'fcpg/v21_c4_shadow_worker.py', 'fcpg/v21_c4_shadow_tap.py', 'fcpg/v21_c4_shadow_protocol.py',
        'fcpg/v21_lewm_c4_scorer.py', 'fcpg/v21_lewm_standalone.py', 'fcpg/lerobot_archive.py',
        'fcpg/prod_v18_1_training.py', 'fcpg/v21_camera_quality.py')}
    manifest = dict(schema='v21-c4-shadow-worker-v2', args={k: str(v) for k, v in vars(args).items()},
                    checkpoint_sha256=sha256(args.checkpoint), dino_weights_sha256=sha256(args.dino_weights),
                    model_config=payload.get('model_config'), commit=commit, code_sha256=code,
                    torch=torch.__version__, device=str(device),
                    device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                    camera_check=f'placeholder (reason 4) + {CAMERA_QUALITY_VERSION} (reason 5)',
                    geometry=dict(step=STEP, half=HALF, past=PAST, goal=GOAL, tail=TAIL, block=BLOCK, fps=FPS))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    worker = Worker(model, dino, args.output, device=device,
                    dtype=torch.float16 if args.precision == 'fp16' else None)
    worker.warm_up()
    (args.output / 'ready.json').write_text(json.dumps(dict(ready=time.strftime('%Y-%m-%dT%H:%M:%S%z'))) + '\n')
    print('worker ready', flush=True)
    stream = sys.stdin.buffer
    try:
        while True:
            message = protocol.decode(stream)
            if message is None:
                break
            worker.handle(*message)
    finally:
        worker.finish()
        print('worker done', flush=True)


if __name__ == '__main__':
    main()
