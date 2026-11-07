#!/usr/bin/env python3
"""Camera layout table that trains only on recordings from chosen sources (docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md
17.103; James 10/25 「可以 先设计 推吧」).

17.98 item 3 measured per source how the wrist picture moves when only the base turns (the image shift of the scene
per positive pan turn) and sorted the sources against the main direction: 'same' (within 45 degrees), 'sideways',
'other way', 'inconsistent' (the source's own pairs disagree), 'not measured'. A verdict file names the kind of each
source (manifest_row.source_group); a source missing from it counts as 'not measured'. Here every training recording
whose source's kind is not kept becomes undecided, so the trainers (which admit exactly the recordings whose layout is
decided, fcpg/v21_archive_camera_layout.py) leave it out. As in scripts/dedupe_v21_camera_layout.py the index and the
training roots stay as they are, the development recordings stay as they are (development errors stay comparable with
every earlier run on the same index), and an undecided entry keeps the layout contract (order [0, 1], no wrist,
external present). Entries undecided before stay undecided. The table also stores `source_dropped` (and carries
`duplicate_dropped` over when the input has it); the trainers read only the four layout arrays.
--others external-only (11/01, 17.162; James 「所有的全部用上」「好做的做，不好做的刪除了」): a training recording of a
source not kept whose layout is decided and that has an external camera keeps it and only its wrist picture is left out
(wrist_present False, as a recording without a wrist camera: the trainers give it no slot-0 view); one without an
external camera is still left out. The table then also stores `wrist_dropped`.
Writes OUT/{layout.npz, report.json}; OUT must not exist.
Usage: python scripts/select_v21_camera_layout_sources.py --layout <layout.npz> --index <half-second index>
       --kinds <verdicts.json> [--keep same] [--others drop|external-only] --out <dir>
verdicts.json: {"sources": {source_group: {"kind": ...}}} (the measurement's summary) or {source_group: kind}.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

TRAIN, DEVELOPMENT = 0, 1        # split.npy of the half-second index
KINDS = ('same', 'sideways', 'other way', 'inconsistent', 'not measured')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_kinds(path):
    """{source_group: kind} from a measurement summary or a plain mapping; unknown kinds stop."""
    data = json.loads(Path(path).read_text())
    kinds = ({name: entry.get('kind', 'not measured') for name, entry in data['sources'].items()}
             if isinstance(data.get('sources'), dict) else dict(data))
    unknown = sorted(set(kinds.values()) - set(KINDS))
    if unknown:
        raise ValueError(f'{path}: unknown kinds {unknown}')
    return kinds


def source_drops(source, in_development, kinds, keep):
    """{pool entry: kind} of the training recordings to leave out: their source's kind is not kept."""
    drop = {}
    for entry, name in source.items():
        kind = kinds.get(name, 'not measured')
        if kind not in keep and not in_development[entry]:
            drop[entry] = kind
    return drop


def external_only(layout, drop):
    """17.162: (wrist_off, drop) - of the recordings to leave out, those with a decided layout and an external camera
    keep it and lose only their wrist picture; the others stay in drop. Both {entry: kind}."""
    wrist_off = {e: k for e, k in drop.items() if layout['decided'][e] and layout['external_present'][e]}
    return wrist_off, {e: k for e, k in drop.items() if e not in wrist_off}


def select(layout, drop, wrist_off=None):
    """The layout arrays with the dropped entries undecided (and the wrist_off entries without their wrist picture),
    and the per-entry record of the dropped ones."""
    o, w = layout['order'].copy(), layout['wrist_present'].copy()
    d, x = layout['decided'].copy(), layout['external_present'].copy()
    dropped = np.zeros(len(d), bool)
    for e in drop:
        o[e], w[e], d[e], x[e], dropped[e] = [0, 1], False, False, True, True
    for e in wrist_off or {}:
        if not d[e] or not x[e]:
            raise ValueError('only a decided recording with an external camera can lose its wrist picture')
        w[e] = False
    if (o[~d] != [0, 1]).any() or w[~d].any() or not x[~d].all() or (d & ~w & ~x).any():
        raise ValueError('the new table breaks the layout contract')
    return dict(order=o, wrist_present=w, decided=d, external_present=x), dropped


def build(layout_path, index, kinds_path, keep, out, others='drop'):
    keep = tuple(keep)
    if not keep or set(keep) - set(KINDS):
        raise ValueError(f'keep must name kinds among {KINDS}')
    if others not in ('drop', 'external-only'):
        raise ValueError("others must be 'drop' or 'external-only'")
    kinds = read_kinds(kinds_path)
    with np.load(layout_path) as data:
        layout = {k: np.asarray(data[k]) for k in ('order', 'wrist_present', 'decided', 'external_present')}
        extra = {k: np.asarray(data[k]) for k in ('duplicate_dropped',) if k in data.files}
    episodes = json.loads((index / 'episode-path-map.json').read_text())['episodes']
    source = {int(r['temporal_episode_index']): r['manifest_row']['source_group'] for r in episodes}
    if len(layout['decided']) <= max(source):
        raise ValueError('the layout table is shorter than the index\'s pool')
    episode = np.load(index / 'episode_index.npy')
    split = np.load(index / 'split.npy')
    in_development = np.zeros(len(layout['decided']), bool)
    in_development[episode[split == DEVELOPMENT]] = True
    if in_development[episode[split == TRAIN]].any():
        raise ValueError('a recording has rows in both splits')
    if set(np.unique(episode)) - set(source):
        raise ValueError('the index has rows of recordings its episode-path-map does not name')
    drop = source_drops(source, in_development, kinds, keep)
    wrist_off = {}
    if others == 'external-only':            # 17.162
        wrist_off, drop = external_only(layout, drop)
    new, dropped = select(layout, drop, wrist_off)
    wrist_dropped = np.zeros(len(dropped), bool)
    wrist_dropped[list(wrist_off)] = True
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / 'layout.npz', **new, **extra, source_dropped=dropped,
             **(dict(wrist_dropped=wrist_dropped) if others == 'external-only' else {}))

    def admitted(table, s, wrist):
        rows = episode[split == s]
        return int((table['decided'] & (table['wrist_present'] | (not wrist)))[rows].sum())
    rows = {name: dict(before=admitted(layout, s, False), after=admitted(new, s, False),
                       with_wrist_before=admitted(layout, s, True), with_wrist_after=admitted(new, s, True))
            for name, s in (('train', TRAIN), ('development', DEVELOPMENT))}
    if rows['development']['before'] != rows['development']['after']:
        raise ValueError('the development rows changed')
    train_rows = np.bincount(episode[split == TRAIN], minlength=len(dropped))
    wrist_rows = train_rows * (layout['decided'] & layout['wrist_present'])
    by_kind, by_source = Counter(), Counter()
    for e, name in source.items():
        by_kind[kinds.get(name, 'not measured')] += int(wrist_rows[e])
        if e in drop:
            by_source[name] += int(wrist_rows[e])
    report = dict(layout=str(layout_path), layout_sha256=sha256(layout_path), index=str(index.resolve()),
                  kinds=str(kinds_path), kinds_sha256=sha256(kinds_path), keep=list(keep),
                  out_sha256=sha256(out / 'layout.npz'), entries=len(dropped),
                  dropped=dict(recordings=len(drop), by_kind=dict(Counter(drop.values())),
                               were_decided=int(sum(bool(layout['decided'][e]) for e in drop))),
                  rows=rows, wrist_train_rows_before_by_kind=dict(by_kind),
                  kept_sources=len({name for e, name in source.items()
                                    if wrist_rows[e] and e not in drop and e not in wrist_off}),
                  **(dict(others=others, wrist_left_out=dict(
                      recordings=len(wrist_off), by_kind=dict(Counter(wrist_off.values())),
                      train_rows=int(train_rows[list(wrist_off)].sum()) if wrist_off else 0,
                      sources=len({source[e] for e in wrist_off}))) if others == 'external-only' else {}),
                  wrist_train_rows_dropped_by_source=dict(by_source.most_common(20)))
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--layout', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--kinds', type=Path, required=True)
    p.add_argument('--keep', nargs='+', default=['same'])
    p.add_argument('--others', choices=('drop', 'external-only'), default='drop',
                   help="17.162: the recordings of sources not kept are left out (drop) or keep their external camera")
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    report = build(args.layout, args.index, args.kinds, args.keep, args.out, others=args.others)
    print(json.dumps(dict(dropped=report['dropped'], rows=report['rows'], kept_sources=report['kept_sources'],
                          wrist_left_out=report.get('wrist_left_out'),
                          wrist_train_rows_before_by_kind=report['wrist_train_rows_before_by_kind'])), flush=True)


if __name__ == '__main__':
    main()
