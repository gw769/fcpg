"""10/31 (James 「大哥你不是有很多真实的 自主抓取成功的记录马？ 没必要 直接重新训练就行阿」「我要的是全部都这么训练」): the
robot's own completed blocks as LeWM training rows, with their pictures.

One row per completed block of an autonomous robot trial: the alignment-input snapshot the service planned it from (the
visual memory of both cameras in the service's own DINOv3-L tokens, the joints, the completed history, the time), the
five commands it sent (offsets from the block's source time) and the joints read back after them (measured_q_path[1:6]
at their read-back times) - the fields of the 575-record response pack (v21-h40-local-response-retention-v1), plus the
pictures and the task text. A block counts when it completed, has its snapshot, its shapes are the 5-command block's,
and the snapshot's joints are the block's source joints (otherwise the snapshot shows another moment). The task text is
the snapshot's own task ids read with the vocabulary of the package that ran the session.
"""
import json
import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

SCHEMA = 'v21-h40-robot-block-pack-v1'
RESERVED = ('<pad>', '<unk>', '<bos>', '<eos>')


def session_model(sessions, session):
    launch = Path(sessions) / f'{session}.launch.txt'
    if not launch.exists():
        return None
    for line in launch.read_text().splitlines():
        if line.startswith('started '):
            words = line.split()
            return words[words.index('model') + 1] if 'model' in words else None
    return None


def task_text(task_ids, words):
    tokens = [words[i] for i in task_ids.reshape(-1).tolist() if i < len(words) and words[i] not in RESERVED]
    if not tokens or len(tokens) != sum(1 for i in task_ids.reshape(-1).tolist() if i > 3):
        raise ValueError('a task id outside the vocabulary of the package that ran the session')
    return ' '.join(tokens)


def collect(sessions, days, *, log=print):
    """The blocks of the sessions whose names start with one of days (YYYYMMDD), in session and block order:
    (records, skipped counts)."""
    sessions = Path(sessions)
    rows, skipped = [], dict(not_completed=0, no_snapshot=0, shapes=0, source_q=0, times=0, no_model=0)
    vocabularies = {}
    for directory in sorted(p for p in sessions.iterdir() if p.is_dir() and p.name[:8] in days):
        model = session_model(sessions, directory.name)
        if model is None or not (Path(model) / 'config.json').exists():
            skipped['no_model'] += len(list(directory.glob('execution-*.json')))
            continue
        if model not in vocabularies:
            vocabularies[model] = tuple(json.loads((Path(model) / 'config.json').read_text())['vocabulary'])
        words = vocabularies[model]
        count = 0
        for path in sorted(directory.glob('execution-*.json')):
            record = json.loads(path.read_text())
            align = record.get('alignment_cem') or {}
            if not record.get('completed'):
                skipped['not_completed'] += 1
                continue
            snapshot = align.get('input_snapshot')
            if not snapshot or not (directory / snapshot).exists():
                skipped['no_snapshot'] += 1
                continue
            sent = np.asarray(record.get('commands_position'), np.float64)
            measured = np.asarray(record.get('measured_q_path'), np.float64)
            command_times = np.asarray(record.get('command_times_s') or [], np.float64)
            readback_times = np.asarray(record.get('readback_times_s') or [], np.float64)
            if (sent.shape != (5, 6) or measured.shape != (6, 6) or command_times.shape != (5,)
                    or readback_times.shape != (6,)):
                skipped['shapes'] += 1
                continue
            source_time = float(record['source_time_s'])
            offsets, query = command_times - source_time, readback_times[1:] - source_time
            if not (np.isfinite(offsets).all() and np.isfinite(query).all() and offsets[0] >= -1e-6
                    and (np.diff(offsets) > 0).all() and (query > 0).all() and (np.diff(query) > 0).all()):
                skipped['times'] += 1
                continue
            state = torch.load(directory / snapshot, map_location='cpu', weights_only=False)['state']
            if not np.allclose(state.q[0].double().numpy(), np.asarray(record['source_q_position'], np.float64),
                               atol=1e-4, rtol=0):
                skipped['source_q'] += 1
                continue
            rows.append(dict(session=directory.name, record=path.name, model=model, state=state,
                             commands=sent, offsets=np.clip(offsets, 0., None), query=query, measured=measured[1:],
                             task=task_text(state.task_ids, words),
                             holding=bool((align.get('grip_latch') or {}).get('holding'))))
            count += 1
        log(f'{directory.name}: {count} blocks')
    return rows, skipped


def build(rows, *, split, days):
    """The pack of these rows (CPU tensors)."""
    if not rows:
        raise ValueError('no robot blocks')
    states = [r['state'] for r in rows]
    stack = lambda get: torch.cat([get(s) for s in states])
    memory_fields = [f.name for f in fields(type(states[0].memory)) if f.name != 'camera_types']
    if any(s.memory.camera_types is not None for s in states):
        raise ValueError('typed camera memories are not packed')
    history_fields = [f.name for f in fields(type(states[0].history))]
    tasks = sorted({r['task'] for r in rows})
    sessions = sorted({r['session'] for r in rows})
    data = dict(
        q=stack(lambda s: s.q.float()),
        now=stack(lambda s: s.time.double()),
        trace=dict(positions=torch.tensor(np.stack([r['commands'] for r in rows]), dtype=torch.float32),
                   offsets=torch.tensor(np.stack([r['offsets'] for r in rows]), dtype=torch.float64),
                   valid=torch.ones(len(rows), 5, dtype=torch.bool), source='recorded', complete=True),
        history={name: stack(lambda s, n=name: getattr(s.history, n)) for name in history_fields},
        memory={name: stack(lambda s, n=name: getattr(s.memory, n)) for name in memory_fields},
        query_offsets=torch.tensor(np.stack([r['query'] for r in rows]), dtype=torch.float64),
        actual=torch.tensor(np.stack([r['measured'] for r in rows]), dtype=torch.float32),
        task=torch.tensor([tasks.index(r['task']) for r in rows], dtype=torch.long),
        session=torch.tensor([sessions.index(r['session']) for r in rows], dtype=torch.long),
        holding=torch.tensor([r['holding'] for r in rows], dtype=torch.bool))
    if data['memory']['features'].dtype != torch.float16:
        raise ValueError('the service keeps its DINO tokens in float16')
    for name in ('q', 'actual'):
        if not torch.isfinite(data[name]).all():
            raise ValueError(f'robot blocks: {name} not finite')
    return dict(schema=SCHEMA, split=split, days=sorted(days), tasks=tasks, sessions=sessions,
                records=[f"{r['session']}/{r['record']}" for r in rows],
                models=sorted({r['model'] for r in rows}), blocks=len(rows),
                labels='measured_q_path[1:6] at readback_times_s[1:6] - source_time_s; commands at command_times_s - '
                       'source_time_s; the alignment-input snapshot as the state (its time, memory and history)',
                data=data)


ARM_P_SCHEMA = 'v21-robot-arm-p-v1'


def session_arm_p(sessions, session):
    """(P of shoulder_lift and elbow_flex in a robot session, the launch record's evidence): 'arm P 32 (17.92)' gives
    32, 'arm P 16 (lerobot)' or no arm P (sessions before 17.92) LeRobot's configure() 16 (fcpg/v21_arm_gain.py)."""
    launch = Path(sessions) / f'{session}.launch.txt'
    if not launch.exists():
        raise ValueError(f'{session}: no launch record')
    found = sorted({int(m) for m in re.findall(r'\barm P (\d+)\b', launch.read_text())})
    if len(found) > 1 or (found and found[0] not in (16, 32)):
        raise ValueError(f'{session}: arm P {found} in its launch record')
    return (found[0], f'arm P {found[0]}') if found else (16, "no arm P: LeRobot's configure() 16")


class RobotBlocks:
    """A loaded pack on the training device; batch(ids) gives the LeWM's inputs, the read-backs and the task ids
    (encoded with vocabulary, when one is given); with an arm P table (17.153), controller(ids) gives the LeWM's
    control-setting inputs [P 32, 1]."""

    def __init__(self, pack, device, vocabulary=None, *, split='train', arm_p=None):
        if pack.get('schema') != SCHEMA or pack.get('split') != split:
            raise ValueError(f'robot block pack must be a {split} split of {SCHEMA}')
        from fcpg.prod_v20_policy import ActualHistory
        from fcpg.v21_memory import VisualMemory
        from fcpg.v21_models import CommandTrace
        self._history, self._memory, self._trace = ActualHistory, VisualMemory, CommandTrace
        to = lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
        data = pack['data']
        self.data = {k: ({n: to(v) for n, v in x.items()} if isinstance(x, dict) else to(x)) for k, x in data.items()}
        self.size = len(self.data['q'])
        self.task_ids = None if vocabulary is None else vocabulary.encode(pack['tasks'], device=device)
        self.record = dict(schema=SCHEMA, blocks=self.size, sessions=len(pack['sessions']), days=pack['days'],
                           tasks=pack['tasks'], carried=int(self.data['holding'].sum()))
        if arm_p is not None:
            if arm_p.get('schema') != ARM_P_SCHEMA or not set(pack['sessions']) <= set(arm_p['sessions']):
                raise ValueError(f'the arm P table must be a {ARM_P_SCHEMA} table of every session of the pack')
            p32 = torch.tensor([arm_p['sessions'][s] == 32 for s in pack['sessions']], device=self.data['q'].device)
            self.data['arm_p32'] = p32[self.data['session']]
            self.record['arm_p32_blocks'] = int(self.data['arm_p32'].sum())

    def controller(self, ids):
        """17.153: [B, 2] float = (shoulder_lift/elbow_flex at P 32 in the block's session, the service's planner)."""
        if 'arm_p32' not in self.data:
            raise ValueError('no arm P table was given')
        p32 = self.data['arm_p32'][torch.as_tensor(ids, device=self.data['q'].device)].float()
        return torch.stack((p32, torch.ones_like(p32)), -1)

    def batch(self, ids):
        ids = torch.as_tensor(ids, device=self.data['q'].device)
        d = self.data
        take = lambda x: x[ids]
        trace = self._trace(**{k: take(v) if isinstance(v, torch.Tensor) else v for k, v in d['trace'].items()})
        history = self._history(**{k: take(v) for k, v in d['history'].items()})
        memory = self._memory(**{k: take(v) for k, v in d['memory'].items()}, camera_types=None)
        task_ids = None if self.task_ids is None else self.task_ids[take(d['task'])]
        return take(d['q']), trace, history, take(d['query_offsets']), take(d['actual']), memory, take(d['now']), task_ids
