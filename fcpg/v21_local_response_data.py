"""Read factual local LeWM responses without a runtime, cameras, or training.

The default split contains only the inventory's eligible TRAIN records. Saved
policy snapshots supply ActualHistory; predictions, follow samples, Goal reads,
and STOP/Hold observations are never supervised position labels. Native H0
diagnostics use their saved before/after runtime witnesses, not invented policy
snapshots. Existing hash fields are provenance only, not admission rules.
"""
from collections import Counter
from dataclasses import dataclass, fields
import json
from numbers import Integral
from pathlib import Path
import re
from typing import Sequence

import torch

from .prod_v20_policy import ActualHistory, _validate_history, empty_history
from .v21_models import CommandTrace


MANIFEST_SCHEMA = 'v21-local-real-response-train-candidates-v1'
SPLITS = ('train', 'validation', 'role_conflict', 'holdout')
# This observed failure remains a diagnostic holdout even if its manifest role
# is accidentally changed while preparing a later training manifest.
BLOCK01_RECORD_ID = 'execution-1789883132353020151.json'
_ROLE_EXCLUSIONS = {'prior_validation_or_test_role_preserved',
                    'no_explicit_fitting_declaration', 'current_block01_locked_diagnostic_holdout'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, shape, name):
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64, device='cpu')
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f'{name} is not numeric') from error
    _require(tuple(tensor.shape) == shape and torch.isfinite(tensor).all(),
             f'{name} must be finite with shape {shape}')
    return tensor


def _group(row):
    declarations = row.get('provenance_declarations', [])
    # A sealed TEST identity is not opened through a development split.
    if any(re.search(r'(^|[_\s-])test($|[_\s-])', str(d.get(key, '')).lower())
           for d in declarations for key in ('group', 'role', 'split')):
        return 'excluded'
    if (row.get('record_id') == BLOCK01_RECORD_ID
            or row.get('role') == 'additional_diagnostic_holdout'
            or 'current_block01_locked_diagnostic_holdout' in row.get('exclusion_reasons', [])):
        return 'holdout'
    if any(reason not in _ROLE_EXCLUSIONS for reason in row.get('exclusion_reasons', [])):
        return 'excluded'
    fit = any(d.get('declares_fit') is True for d in declarations)
    validation = any(d.get('declares_validation_or_test') is True for d in declarations)
    if validation:
        return 'role_conflict' if fit else 'validation'
    if fit and row.get('eligible_for_real_response_adapter') is True and not row.get('exclusion_reasons'):
        return 'train'
    return 'excluded'


@dataclass(frozen=True)
class LocalResponseBatch:
    q_position: torch.Tensor       # [B,6], measured dispatch q0
    trace: CommandTrace            # Five factual transmitted commands, no densification
    history: ActualHistory         # Saved actual input history, never post-execution history
    query_offsets: torch.Tensor    # [B,5], float64 actual q1..q5 host clocks relative to q0
    y: torch.Tensor                # [B,5,6], original measured_q_path[1:6]
    record_ids: tuple[str, ...]
    goal_status: tuple[str, ...]    # 'verified' or 'unknown'; never a motion-success label
    split: str

    def __len__(self):
        return len(self.q_position)

    def to(self, device):
        return LocalResponseBatch(self.q_position.to(device),
            CommandTrace(self.trace.positions.to(device), self.trace.offsets.to(device),
                         self.trace.valid.to(device), 'recorded', True),
            self.history.to(device), self.query_offsets.to(device), self.y.to(device),
            self.record_ids, self.goal_status, self.split)


class LocalResponseDataset:
    """Lazy CPU loader for the reviewed local response inventory.

    ``batch = LocalResponseDataset(path).batch([0, 1])`` can be passed directly
    to ``lewm(batch.q_position, batch.trace, batch.history, batch.query_offsets)``.
    Only ``split='train'`` may enter optimization. ``validation`` contains rows
    never declared fitting; ``role_conflict`` keeps earlier validation later
    used for fitting separate; ``holdout`` contains additional diagnostics.
    These reviewed development groups are not blind performance evidence.

    Torch snapshots are trusted local artifacts and are loaded on CPU. Only the
    small numeric response sample is cached; visual memory is released.
    """

    def __init__(self, manifest_path, *, split='train'):
        _require(split in SPLITS, f'unknown local response split: {split}')
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        document = json.loads(self.manifest_path.read_text())
        _require(document.get('schema') == MANIFEST_SCHEMA, 'unsupported local response manifest')
        records = document.get('records')
        _require(isinstance(records, list), 'manifest records must be a list')
        ids = [row['record_id'] for row in records]
        paths = [str(self._path(row['record_path'])) for row in records]
        _require(len(set(ids)) == len(ids) and len(set(paths)) == len(paths),
                 'duplicate local record identity or original path')
        self.split = split
        self.records = tuple(row for row in records if _group(row) == split)
        self.record_ids = tuple(row['record_id'] for row in self.records)
        self.run_ids = tuple(str(self._path(row['record_path']).parent) for row in self.records)
        self._groups = Counter(_group(row) for row in records)
        self._cache = {}

    def _path(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.manifest_path.parent / path

    def __len__(self):
        return len(self.records)

    def summary(self):
        return dict(split=self.split, records=len(self),
            group_counts={name: self._groups[name] for name in (*SPLITS, 'excluded')},
            goal_status=dict(Counter(row.get('goal_verification', {}).get('status', 'unknown')
                                     for row in self.records)),
            history_counts=dict(Counter(str(row.get('history', {}).get('valid_count'))
                                        for row in self.records)),
            train_only=self.split == 'train', labels='original measured_q_path[1:6]',
            normalization='unchanged calibrated positions; checkpoint retains its own scales',
            command_representation='five original writes and actual relative host clocks',
            reviewed_development_not_blind=True)

    def batch(self, indices: Sequence[int] | torch.Tensor | None = None, *, device='cpu'):
        if indices is None:
            indices = range(len(self))
        if isinstance(indices, torch.Tensor):
            _require(indices.ndim == 1 and indices.dtype in (torch.int32, torch.int64),
                     'batch indices must be one-dimensional integers')
            indices = indices.cpu().tolist()
        indices = list(indices)
        _require(bool(indices), 'local response batch cannot be empty')
        samples = []
        for index in indices:
            _require(isinstance(index, Integral) and not isinstance(index, bool) and 0 <= index < len(self),
                     'local response index out of range')
            index = int(index)
            if index not in self._cache:
                self._cache[index] = self._load(self.records[index])
            samples.append(self._cache[index])
        cat = lambda name: torch.cat([getattr(sample, name) for sample in samples])
        history = ActualHistory(*(torch.cat([getattr(s.history, field.name) for s in samples])
                                  for field in fields(ActualHistory)))
        trace = CommandTrace(torch.cat([s.trace.positions for s in samples]),
                             torch.cat([s.trace.offsets for s in samples]),
                             torch.cat([s.trace.valid for s in samples]), 'recorded', True)
        return LocalResponseBatch(cat('q_position'), trace, history, cat('query_offsets'), cat('y'),
            tuple(s.record_ids[0] for s in samples), tuple(s.goal_status[0] for s in samples), self.split).to(device)

    def _policy_history(self, row, source, q, commands, epoch):
        _require(row.get('snapshot_path'), 'policy response requires its actual input snapshot')
        snapshot = torch.load(self._path(row['snapshot_path']), map_location='cpu', weights_only=False)
        _require(snapshot.get('kind') == 'actual-actor-input' and snapshot.get('schema') == 2
                 and snapshot.get('executable') is False, 'not an actual policy input snapshot')
        _require(snapshot.get('action_source') == source.get('action_source'), 'snapshot action source differs')
        _require(abs(float(snapshot['dispatch_source_time_s'])-epoch) <= 1e-8,
                 'snapshot dispatch epoch differs')
        saved_q = _number(snapshot.get('dispatch_source_q_position'), (6,), 'snapshot dispatch q')
        saved_u = _number(snapshot.get('commands_position'), (5, 6), 'snapshot commands')
        _require(torch.allclose(saved_q, q, atol=1e-4, rtol=0)
                 and torch.allclose(saved_u, commands, atol=1e-4, rtol=0), 'snapshot dispatch q or commands differ')
        state = snapshot.get('state')
        _require(state is not None and isinstance(getattr(state, 'history', None), ActualHistory),
                 'snapshot actual history missing')
        snapshot_time = float(_number(state.time, (1,), 'snapshot time')[0])
        _require(snapshot_time <= epoch+1e-8, 'snapshot is after dispatch')
        history = state.history.to(device='cpu', dtype=torch.float32)
        _validate_history(history, 1)
        ends = _number(state.history_ends, (1, 3), 'actual history end times')[0]
        valid = history.valid[0]
        _require((ends <= epoch+1e-8).all() and (ends[valid] <= snapshot_time+1e-8).all()
                 and (ends[valid].diff() > 0).all(), 'future or unordered actual history')
        _require((ends[valid] >= history.time_deltas[0, valid].double().sum(-1)).all(),
                 'history starts before its clock origin')
        return ActualHistory(*(getattr(history, f.name).detach().clone() for f in fields(ActualHistory)))

    def _native_history(self, row, report, source):
        legacy_pid = report.get('schema') == 'v21-j3-pid-phase-result-v2'
        _require(source.get('not_model_transition') is True and source.get('model_feedback_label') is False
                 and report.get('model_history_imported', False if legacy_pid else None) is False,
                 'native diagnostic must preserve the recorded model-history boundary')
        witnesses = row.get('history', {}).get('h0_witnesses', [])
        _require(len(witnesses) == 2 and row.get('history', {}).get('valid_count') == 0
                 and row.get('runtime_session_id'), 'native response requires two saved H0 witnesses')
        times = []
        for witness in witnesses:
            status = json.loads(self._path(witness['path']).read_text())
            runtime = status['service']['runtime']
            progress = runtime['readback_progress']
            _require(runtime['history_chunks'] == [0] and progress['current_index'] == [0]
                     and progress['session_id'] == row['runtime_session_id']
                     and not runtime['pending'] and not runtime['executing'], 'native saved H0 witness differs')
            times.append(float(status['published_monotonic_s']))
        times.sort()
        # The older PID receipt records drive/STOP boundaries rather than the
        # range diagnostic's outer started_s/finished_s fields. The two saved
        # runtime witnesses still prove H0 across the entire observed action.
        started = report.get('drive_started_s') if legacy_pid else report.get('started_s')
        finished = report.get('stop_returned_s') if legacy_pid else report.get('finished_s')
        _require(started is not None and finished is not None
                 and times[0] <= started < finished <= times[1],
                 'saved H0 witnesses do not bracket native diagnostic')
        return empty_history(1, device='cpu')

    def _load(self, row):
        document = json.loads(self._path(row['record_path']).read_text())
        report = document.get('protocol_report', document)
        native = row.get('source_class') == 'native_diagnostic'
        source = report['drive_result'] if native else report
        count = len(source.get('commands_position', []))
        _require(source.get('completed') is True and count >= 5 and (native or count == 5)
                 and source.get('policy_commands_attempted') == count
                 and source.get('policy_commands_returned') == count, 'incomplete actual command transaction')
        # Native range10 contributes only its original first five writes and
        # corresponding six sampled positions, never its second half or follow.
        commands = _number(source['commands_position'], (count, 6), 'actual commands')[:5]
        measured = _number(source.get('measured_q_path'), (count+1, 6), 'actual measured q path')[:6]
        q = _number(source.get('source_q_position'), (6,), 'actual dispatch q')
        ct = _number(source.get('command_times_s'), (count,), 'actual command times')[:5]
        qt = _number(source.get('readback_times_s'), (count+1,), 'actual readback times')[:6]
        write_ends = _number(source.get('write_completed_times_s'), (count,), 'actual write completion times')[:5]
        epoch = float(source['source_time_s'])
        _require(abs(float(qt[0])-epoch) <= 1e-8 and torch.allclose(q, measured[0], atol=1e-4, rtol=0),
                 'actual source q or epoch differs from q0 readback')
        _require((ct.diff() > 0).all() and (qt.diff() > 0).all() and (ct >= qt[:-1]).all()
                 and (write_ends >= ct).all() and (write_ends < qt[1:]).all(),
                 'actual command/readback clocks are not causal')
        for event in source.get('actuator_events', []):
            if event.get('motor_register_writes', event.get('goal_register_writes', 0)):
                _require(not epoch <= event['started_s'] <= float(qt[-1]),
                         'auxiliary Hold/write occurred inside supervised path')
        goal_status = 'unknown'
        proofs = source.get('policy_goal_readbacks')
        if proofs is not None:
            _require(isinstance(proofs, list) and len(proofs) == count
                     and all(source.get(key) == count for key in ('policy_goal_reads_attempted',
                         'policy_goal_reads_returned', 'policy_goal_registers_verified')),
                     'incomplete per-command Goal evidence')
            for i, proof in enumerate(proofs[:5]):
                _require(proof.get('command_index') == i and proof.get('goal_register_verified') is True
                         and proof.get('read_attempted') is True and proof.get('read_returned') is True
                         and proof.get('motor_register_writes') == 0
                         and isinstance(proof.get('goal_raw'), dict)
                         and proof.get('goal_raw') == proof.get('expected_goal_raw')
                         and write_ends[i] <= proof['read_started_s'] <= proof['read_returned_s'] < qt[i+1],
                         'per-command Goal readback contradicts its recorded write')
            goal_status = 'verified'
        _require(row.get('goal_verification', {}).get('status', goal_status) == goal_status,
                 'manifest Goal status differs from original response record')
        history = (self._native_history(row, report, source) if native
                   else self._policy_history(row, source, q, commands, epoch))
        if row.get('history', {}).get('valid_count') is not None:
            _require(int(history.valid.sum()) == row['history']['valid_count'], 'saved history count differs')
        trace = CommandTrace(commands.float()[None], (ct-epoch)[None],
                             torch.ones(1, 5, dtype=torch.bool), 'recorded', True)
        trace.validate(1)
        queries = (qt[1:]-epoch)[None]
        _require(torch.equal(trace.active_at(queries, q.float()[None]), trace.positions),
                 'actual readback/action-index pairing differs')
        return LocalResponseBatch(q.float()[None], trace, history, queries, measured[1:].float()[None],
                                  (row['record_id'],), (goal_status,), self.split)
