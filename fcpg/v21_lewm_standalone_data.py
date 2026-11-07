"""Batches for the standalone LeWM, built only from the existing V21 readers.

Frames: the three chain endpoints t-1.0, t-0.5, t (ScratchDinoFeatureBank, 0.5 s apart)
and the sparse8 futures t+0.5 ... t+4.0 (FutureSceneTargets). Commands: horizon_targets
at chain position 0 (its first 10 commands are the two past blocks) and position 2 (the
40 future commands). Every validity, causality and episode check of those readers applies
unchanged; this module only stacks their outputs.
"""
import numpy as np
import torch

from .prod_v20_policy import DEFAULT_LOWER, DEFAULT_UPPER
from .v21_horizon_training import horizon_metadata, horizon_targets
from .v21_lewm_standalone import BLOCK, CONTEXT_FRAMES, FRAMES, JOINTS
from .v21_scene_targets import FutureSceneTargets
from .v21_scratch_features import ScratchDinoFeatureBank

SPARSE8 = [4, 9, 14, 19, 24, 29, 34, 39]
CENTER = torch.tensor([(hi + lo) / 2 for lo, hi in zip(DEFAULT_LOWER, DEFAULT_UPPER)])
HALF_RANGE = torch.tensor([(hi - lo) / 2 for lo, hi in zip(DEFAULT_LOWER, DEFAULT_UPPER)])
PAST_COMMANDS = (CONTEXT_FRAMES - 1) * BLOCK                # 10
COMMANDS = (FRAMES - 1) * BLOCK                            # 50


def normalize_commands(commands):
    return (commands - CENTER.to(commands)) / HALF_RANGE.to(commands)


def action_blocks(commands, command_valid):
    """[B, 50, 6] commands -> ([B, 10, 30] normalized blocks, [B, 10] block complete)."""
    b = commands.shape[0]
    if commands.shape != (b, COMMANDS, JOINTS) or command_valid.shape != (b, COMMANDS):
        raise ValueError('expected 50 commands per row')
    blocks = normalize_commands(commands.float()).reshape(b, FRAMES - 1, BLOCK * JOINTS)
    return blocks, command_valid.reshape(b, FRAMES - 1, BLOCK).all(-1)


DELTA_SCALE = 20.                                          # calibrated units; a 0.5 s block rarely moves further


def delta_blocks(commands, command_valid):
    """[B, 50, 6] commands -> ([B, 10, 30] change of each block's five commands relative to the command
    just before the block, divided by DELTA_SCALE; [B, 10] block scored). Block 0 has no earlier command
    in the window and is measured from its own first command. A block is scored when its five commands
    and its reference command are all real (not padding)."""
    b = commands.shape[0]
    if commands.shape != (b, COMMANDS, JOINTS) or command_valid.shape != (b, COMMANDS):
        raise ValueError('expected 50 commands per row')
    c = commands.float().reshape(b, FRAMES - 1, BLOCK, JOINTS)
    v = command_valid.reshape(b, FRAMES - 1, BLOCK)
    reference = torch.cat((c[:, :1, :1], c[:, :-1, -1:]), 1)             # [B, 10, 1, 6]
    reference_valid = torch.cat((v[:, :1, 0], v[:, :-1, -1]), 1)          # [B, 10]
    return ((c - reference) / DELTA_SCALE).reshape(b, FRAMES - 1, BLOCK * JOINTS), v.all(-1) & reference_valid


def stutter(batch, generator, probability, hold_share=.5):
    """Stop-and-go copies of demonstration windows (docs/V21_LEWM_JEPA_RETRAIN_20261012.md section 7).

    A demonstration moves smoothly, so its next frame can be guessed from the frames before it and the
    commands add little. The live service moves for one block and then stands. Each chosen row is rebuilt
    from its own frames: every step either ADVANCES to the row's next real frame with that step's real
    command block, or HOLDS - the same frame again, with the last transmitted command repeated five times.
    Nothing is invented: every frame is a real frame of the row and every held command is a command that
    was really sent. `batch` holds frames [B, 11, C, P, D], valid [B, 11, C], commands [B, 50, 6],
    command_valid [B, 50]; other keys pass through. Returns (new batch, [B] rows rebuilt).
    """
    frames, valid, commands, command_valid = (batch[k] for k in ('frames', 'valid', 'commands', 'command_valid'))
    b = frames.shape[0]
    steps = FRAMES - 1
    if commands.shape != (b, COMMANDS, JOINTS) or frames.shape[1] != FRAMES or not 0. <= probability <= 1.:
        raise ValueError('stutter needs 11 frames, 50 commands and a probability')
    chosen = torch.rand(b, generator=generator) < probability
    out = {k: (v.clone() if k in ('frames', 'valid', 'commands', 'command_valid') else v) for k, v in batch.items()}
    for row in torch.nonzero(chosen).flatten().tolist():
        hold = torch.rand(steps, generator=generator) < hold_share
        advances = int((~hold).sum())
        start = int(torch.randint(0, FRAMES - advances, (1,), generator=generator))
        index = start
        last = commands[row, BLOCK * index - 1] if index else commands[row, 0]
        last_valid = command_valid[row, BLOCK * index - 1] if index else command_valid[row, 0]
        out['frames'][row, 0], out['valid'][row, 0] = frames[row, index], valid[row, index]
        for k in range(steps):
            block = slice(BLOCK * k, BLOCK * k + BLOCK)
            if bool(hold[k]):
                out['commands'][row, block] = last
                out['command_valid'][row, block] = last_valid
            else:
                source = slice(BLOCK * index, BLOCK * index + BLOCK)
                out['commands'][row, block] = commands[row, source]
                out['command_valid'][row, block] = command_valid[row, source]
                last, last_valid = commands[row, BLOCK * index + BLOCK - 1], command_valid[row, BLOCK * index + BLOCK - 1]
                index += 1
            out['frames'][row, k + 1], out['valid'][row, k + 1] = frames[row, index], valid[row, index]
    return out, chosen


def split_fish_rows(rows, begins, episodes, *, fps=30., validation_episode=1, validation_fraction=.25):
    """Few-shot split of fish TRAIN rows: the last quarter of ep1 (by start) validates.

    A training window spans [t-1.0, t+4.0] s and so does a validation window; every training
    window of the validation episode must end before the first validation window begins, so
    no frame is shared (gap of 5 s of starts). Other episodes are entirely training.
    """
    rows, begins, episodes = (np.asarray(x) for x in (rows, begins, episodes))
    if not (rows.shape == begins.shape == episodes.shape) or rows.ndim != 1:
        raise ValueError('rows, start frames and episodes must align')
    if np.isin(episodes, (2, 3)).any():
        raise ValueError('sealed or DEV fish episode in the few-shot TRAIN pool')
    own = np.flatnonzero(episodes == validation_episode)
    if len(own) < 8:
        raise ValueError('validation episode has too few windows')
    ordered = own[np.argsort(begins[own], kind='stable')]
    first = ordered[int(np.floor(len(ordered) * (1 - validation_fraction)))]
    start = begins[first]
    span = int(round(5 * fps))                      # 1 s history + 4 s future of a window
    validation = own[begins[own] >= start]
    train = np.concatenate((np.flatnonzero(episodes != validation_episode),
                            own[begins[own] < start - span]))
    info = dict(validation_episode=int(validation_episode), validation_start_frame=int(start),
                gap_frames=span, train_windows=int(len(train)), validation_windows=int(len(validation)),
                dropped_for_gap=int(len(own) - len(validation) - (len(train) - int((episodes != validation_episode).sum()))))
    return rows[np.sort(train)], rows[np.sort(validation)], info


def fish_split(source, rows):
    begins = horizon_metadata(source, rows, 'fish', CONTEXT_FRAMES - 1)[0]
    return split_fish_rows(rows, begins, np.asarray(source.fish.episodes[begins]))


def fish_episodes(source, rows):
    """Recording (episode id) of each fish row, read like fish_split reads it."""
    begins = horizon_metadata(source, np.asarray(rows, np.int64), 'fish', CONTEXT_FRAMES - 1)[0]
    return np.asarray(source.fish.episodes[begins])


def check_fish_fold(adapt, test, adapt_episodes, test_episodes, allowed):
    """Fish leave-one-recording-out fold (docs/V21_LEWM_FISH_CV_20261008.md): one test recording,
    adaptation rows from other recordings only, every row an admitted fish row (ep0/ep1 TRAIN,
    ep3 DEV; the sealed ep2 is not a fish row and is refused again here)."""
    adapt, test = np.asarray(adapt, np.int64), np.asarray(test, np.int64)
    if not len(adapt) or not len(test) or len(adapt_episodes) != len(adapt) or len(test_episodes) != len(test):
        raise ValueError('a fish fold needs adaptation and test rows with their recordings')
    if not (np.isin(adapt, allowed).all() and np.isin(test, allowed).all()):
        raise ValueError('fish fold rows outside the fish TRAIN/DEV rows')
    if np.isin(np.concatenate((adapt_episodes, test_episodes)), [2]).any():
        raise ValueError('sealed fish episode 2 in a fold')
    if len(np.unique(test_episodes)) != 1:
        raise ValueError('a fish fold tests exactly one recording')
    if np.intersect1d(adapt, test).size or np.intersect1d(adapt_episodes, test_episodes).size:
        raise ValueError('fish adaptation and test recordings overlap')


def fish_fold(source, splits, path, fold):
    """(adaptation rows, test rows, info) of fold `fold` in the .npz from build_v21_fish_cv.py.
    Fold 'all' (sections 4 and 9, the model wired back): every unsealed fish row, i.e. the union of
    the three test recordings ep0, ep1, ep3; it has no test rows."""
    folds = np.load(path, allow_pickle=False)
    allowed = np.concatenate([np.asarray(splits[role]['fish'], np.int64) for role in ('train', 'development')])
    if fold == 'all':
        parts = [np.asarray(folds[k], np.int64) for k in sorted(folds.files) if k.startswith('test/')]
        adapt = np.sort(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        if len(np.unique(adapt)) != len(adapt) or sorted(adapt.tolist()) != sorted(allowed.tolist()):
            raise ValueError('fish fold all must be every fish TRAIN/DEV row, once')
        if sorted(set(fish_episodes(source, adapt).tolist())) != [0, 1, 3]:
            raise ValueError('fish fold all must hold exactly ep0, ep1 and ep3 (ep2 stays sealed)')
        info = dict(path=str(path), fold='all', rows=int(len(adapt)), episodes=[0, 1, 3], test_rows=0, test_episodes=[])
        return adapt, np.zeros(0, np.int64), info
    if f'adapt/{fold}' not in folds.files or f'test/{fold}' not in folds.files:
        raise ValueError(f'unknown fish fold {fold}')
    adapt, test = (np.asarray(folds[f'{part}/{fold}'], np.int64) for part in ('adapt', 'test'))
    adapt_episodes, test_episodes = fish_episodes(source, adapt), fish_episodes(source, test)
    check_fish_fold(adapt, test, adapt_episodes, test_episodes, allowed)
    if fold != f'f{int(test_episodes[0])}':
        raise ValueError('fold name and its test recording differ')
    info = dict(path=str(path), fold=fold, rows=int(len(adapt)), episodes=sorted(set(adapt_episodes.tolist())),
                test_rows=int(len(test)), test_episodes=sorted(set(test_episodes.tolist())))
    return adapt, test, info


class WindowLoader:
    """CPU loader; safe to fork into DataLoader workers (read-only sources)."""

    def __init__(self, source, *, bank_bytes=16 * 1024 ** 2):
        self.source = source
        self.bank = ScratchDinoFeatureBank(source, 'cpu', bank_bytes)
        self.future = FutureSceneTargets(source, self.bank)
        if list(self.future.query_indices) != SPARSE8:
            raise ValueError('standalone LeWM frames are the sparse8 futures t+0.5 ... t+4.0 s')

    def load(self, rows, kind='archive'):
        rows = np.asarray(rows, np.int64)
        context, context_valid = self.bank.load(rows, kind, context_frames=CONTEXT_FRAMES)
        future = self.future.load(rows, kind, CONTEXT_FRAMES - 1, 'cpu')
        past = horizon_targets(self.source, rows, kind, 0)
        now = horizon_targets(self.source, rows, kind, CONTEXT_FRAMES - 1)
        begin = horizon_metadata(self.source, rows, kind, CONTEXT_FRAMES - 1)[0]
        q_bank = self.source.fish.q if kind == 'fish' else self.source.raw_q
        frames = torch.cat((context.half(), future.features.half()), 1)
        valid = torch.cat((torch.as_tensor(np.asarray(context_valid)), future.valid), 1)
        if frames.shape != (len(rows), FRAMES, 2, 256, 384) or valid.shape != (len(rows), FRAMES, 2):
            raise ValueError('standalone LeWM sequence shape differs')
        return dict(
            rows=torch.as_tensor(rows),
            frames=frames, valid=valid,
            commands=torch.cat((past.commands[:, :PAST_COMMANDS], now.commands), 1),
            command_valid=torch.cat((past.valid[:, :PAST_COMMANDS], now.valid), 1),
            q_now=torch.as_tensor(np.asarray(q_bank[begin], np.float32)),
            episodes=torch.as_tensor(np.asarray(now.episode_ids, np.int64)))


class HalfBatches(torch.utils.data.Dataset):
    """Item i = the i-th scheduled half batch (rows fixed before training starts)."""

    def __init__(self, loader, halves, kind='archive'):
        self.loader, self.halves, self.kind = loader, halves, kind

    def __len__(self):
        return len(self.halves)

    def __getitem__(self, index):
        return self.loader.load(self.halves[index], self.kind)


class Items(torch.utils.data.Dataset):
    """Item i = (kind, rows) of the schedule; a training step joins several items.

    kind 'target' = few-shot rows of a held-out Archive source (the few-shot benchmark);
    they are Archive rows and load exactly like them.
    """

    def __init__(self, loader, items):
        self.loader, self.items = loader, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        kind, rows = self.items[index]
        return self.loader.load(rows, 'archive' if kind == 'target' else kind)


def join(*parts):
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
