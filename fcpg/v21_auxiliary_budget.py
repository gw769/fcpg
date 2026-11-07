"""A shared, fail-closed budget for one independent auxiliary experiment."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time


class AuxiliaryMotionBudget:
    def __init__(self, *, identity_floor=0, maximum_blocks=10):
        if type(identity_floor) is not int or identity_floor<0:
            raise ValueError('invalid auxiliary trial identity floor')
        if type(maximum_blocks) is not int or maximum_blocks not in (10,30):
            raise ValueError('auxiliary cap must be 10 or explicitly authorized 30')
        self.maximum_blocks=maximum_blocks
        self.active=False; self.block_attempts=0; self.returned=0; self.attempted=0
        self.unknown=False; self.records=[]; self.trial_id=None
        self.pending=False; self.requires_home=False
        self._pending_home=False; self._restored=False
        self._lock=threading.RLock()
        self._identity_floor=identity_floor

    @property
    def identity_floor(self):
        return self._identity_floor

    @classmethod
    def restore_latest(cls, output):
        """Restore accounting, never pretend to restore the model/fit state.

        This supports ordinary process restarts using the existing atomic JSON
        writer. It does not claim crash durability after a storage/power failure.
        Invalid ledgers reject startup before the caller acquires hardware.
        """
        paths=list(Path(output).glob('auxiliary-budget-*.json'))
        if not paths:
            return cls()
        for path in paths:
            identity=path.stem.removeprefix('auxiliary-budget-')
            if not identity.isascii() or not identity.isdecimal():
                raise ValueError('invalid auxiliary budget filename; inspect accounting before startup')
        latest=max(paths,key=lambda p:int(p.stem.removeprefix('auxiliary-budget-')))
        data=None
        try:
            for path in paths:
                record=json.loads(path.read_text(encoding='utf-8'))
                cls._validate_ledger(record,path.stem.removeprefix('auxiliary-budget-'))
                if path!=latest and record['active']:
                    raise ValueError('older auxiliary trial is still open; accounting review required')
                if path==latest:
                    data=record
        except (OSError,ValueError,TypeError,KeyError) as exc:
            raise ValueError(f'invalid auxiliary budget ledger {path.name}; HOME/accounting review required') from exc
        budget=cls(identity_floor=int(data['trial_id']),maximum_blocks=data['maximum_blocks'])
        if not data['active']:  # Only an explicitly completed HOME can close it.
            return budget
        budget.active=True; budget.trial_id=data['trial_id']
        budget.block_attempts=data['block_attempts']
        budget.attempted=data['policy_commands_attempted']; budget.returned=data['policy_commands_returned']
        budget.unknown=data['accounting_unknown']; budget.records=deepcopy(data['records'])
        budget.pending=data.get('pending_reservation',False)
        budget.requires_home=True  # Local response and actual neural memory were not restored.
        budget._restored=True
        return budget

    @staticmethod
    def _validate_ledger(data,identity):
        if not isinstance(data,dict) or data.get('trial_id')!=identity:
            raise ValueError('budget identity mismatch')
        for name in ('active','accounting_unknown'):
            if type(data.get(name)) is not bool:
                raise ValueError('budget flags must be boolean')
        pending=data.get('pending_reservation',False)  # Legacy completed ledgers.
        closed=data.get('closed_by_home',False)
        if type(pending) is not bool or type(closed) is not bool:
            raise ValueError('budget lifecycle flags must be boolean')
        if data['active']==closed or (closed and pending):
            raise ValueError('only completed HOME closes an auxiliary trial')
        blocks=data.get('block_attempts'); attempted=data.get('policy_commands_attempted')
        returned=data.get('policy_commands_returned')
        if (type(data.get('maximum_blocks')) is not int or data['maximum_blocks'] not in (10,30)
                or type(blocks) is not int or not 1<=blocks<=data['maximum_blocks']
                or type(attempted) is not int or type(returned) is not int
                or not 0<=returned<=attempted<=5*blocks):
            raise ValueError('invalid budget counters')
        records=data.get('records')
        if not isinstance(records,list) or len(records)+int(pending)!=blocks:
            raise ValueError('each block must have one result or a pending reservation')
        counted_attempted=counted_returned=0; unknown=False
        for record in records:
            if not isinstance(record,dict) or type(record.get('accounting_available')) is not bool:
                raise ValueError('invalid block accounting')
            if not record['accounting_available']:
                unknown=True
                continue
            a=record.get('attempted'); r=record.get('returned')
            if (type(a) is not int or type(r) is not int or not 0<=r<=a<=5
                    or type(record.get('completed')) is not bool
                    or (record['completed'] and (a,r)!=(5,5))):
                raise ValueError('invalid per-block command counters')
            counted_attempted+=a; counted_returned+=r
        if (attempted,returned,unknown)!=(counted_attempted,counted_returned,data['accounting_unknown']):
            raise ValueError('budget totals disagree with per-block records')

    def reserve(self):
        with self._lock:
            if self.requires_home:
                raise RuntimeError('auxiliary trial restored after restart; explicit HOME required')
            if self.pending:
                raise RuntimeError('an auxiliary block reservation is already pending')
            if self.unknown or self.block_attempts >= self.maximum_blocks:
                raise RuntimeError('auxiliary block budget exhausted or command count unknown; HOME required')
            if not self.active:
                self.active=True
                # A wall-clock correction must not put a new active trial behind
                # an older closed ledger or reuse its output filename.
                self.trial_id=str(max(time.time_ns(),self._identity_floor+1))
                self._identity_floor=int(self.trial_id)
            self.block_attempts+=1; self.pending=True

    def finish(self,attempt):
        # The runner clears last_attempt before planning. None proves execution
        # was never entered; otherwise the bridge's explicit counters are used.
        with self._lock:
            if not self.pending or (self.requires_home and not self._pending_home):
                raise RuntimeError('no current auxiliary reservation to finish')
            self.pending=False; self._pending_home=False
            record=attempt if isinstance(attempt,dict) else {}
            attempted=0 if attempt is None else record.get('policy_commands_attempted')
            returned=0 if attempt is None else record.get('policy_commands_returned')
            completed=record.get('completed',False)
            if (type(attempted) is not int or type(returned) is not int
                    or not 0 <= returned <= attempted <= 5 or type(completed) is not bool
                    or (completed and (attempted,returned)!=(5,5))):
                self.unknown=True
                self.records.append(dict(accounting_available=False))
                return
            self.attempted+=attempted;self.returned+=returned
            self.records.append(dict(accounting_available=True,attempted=attempted,returned=returned,
                completed=completed,action_source=record.get('action_source')))

    def reserve_home(self):
        """Charge one protected manual HOME block to this trial's remaining cap.

        This never permits normal policy motion. An interrupted pre-restart
        reservation remains an unknown, fully consumed block, not a zero write.
        """
        with self._lock:
            if not self.active or self.block_attempts>=self.maximum_blocks:
                raise RuntimeError('no auxiliary budget remains for a HOME motion block')
            if self.pending:
                if not self._restored:
                    raise RuntimeError('cannot HOME while an auxiliary block is pending')
                self.records.append(dict(accounting_available=False))
                self.unknown=True; self.pending=False
            self.requires_home=True
            self.block_attempts+=1; self.pending=True; self._pending_home=True

    def extend_to_thirty(self):
        """Explicit operator extension only; never clear counters or HOME interlocks."""
        with self._lock:
            if self.pending or self.unknown:
                raise RuntimeError('cannot extend pending or unknown accounting')
            self.maximum_blocks=30

    def accept_actual_restore(self,verified_ledger):
        """Called only after full actual-state and fresh physical-input validation."""
        with self._lock:
            current=self.status()
            required={'active','trial_id','maximum_blocks','block_attempts',
                'policy_commands_attempted','policy_commands_returned','accounting_unknown',
                'pending_reservation','records'}
            if (not isinstance(verified_ledger,dict) or set(verified_ledger)!=required
                    or not self._restored or not self.requires_home or self.pending or self.unknown
                    or any(current.get(key)!=value for key,value in verified_ledger.items())):
                raise RuntimeError('actual-state restore does not match the protected budget')
            self.requires_home=False;self._restored=False

    def closed_after_home_status(self):
        """Return a closure to persist AFTER protected HOME, without clearing state."""
        value=self.status()
        if value['pending_reservation']:
            # A process may have died after a write but before recording its result.
            value['records'].append(dict(accounting_available=False))
            value['accounting_unknown']=True
        value.update(active=False,closed_by_home=True,pending_reservation=False,
                     requires_home_after_restart=False,requires_home_completion=False)
        return value

    def status(self):
        with self._lock:
            return dict(active=self.active,trial_id=self.trial_id,maximum_blocks=self.maximum_blocks,
                block_attempts=self.block_attempts,policy_commands_attempted=self.attempted,
                policy_commands_returned=self.returned,accounting_unknown=self.unknown,
                pending_reservation=self.pending,requires_home_after_restart=self._restored and self.requires_home,
                requires_home_completion=self.requires_home,
                command_counters_complete=not self.pending and not self.unknown,
                command_counters_semantics='known-result-records-only',
                closed_by_home=False,physical_grasp_verified=False,records=deepcopy(self.records))
