"""Bounded host-side dispatch; no policy, pose, force or success heuristics.

Trial JSON stores experiment identity and operator scores. Completing motion
never sets task_success_verified. A block whose alignment-CEM record reports
released_over_box (fcpg/v21_alignment_cem.py) ends the trial with that outcome
(James 10/16: home once the task is done); it is the planner's report, the
operator still scores the trial.
"""
from copy import deepcopy
import math
import threading
import time


ACTION_SOURCE = 'v21_actor'
EXECUTION_GEOMETRY_SOURCE = 'v21_actor_execution_geometry'
LEWM_CEM_SOURCE = 'v21_lewm_cem'
C4_SHADOW_SOURCE = 'v21_actor_c4_shadow'
C4_SOURCE = 'v21_c4_cem'
ALIGN_SHADOW_SOURCE = 'v21_actor_align_shadow'     # fcpg/v21_alignment_cem.py
ALIGN_SOURCE = 'v21_lewm_align_cem'
EXPLORE_SOURCE = 'v21_actor_explore'              # fcpg/v21_explore.py
CUE_SHADOW_SOURCE = 'v21_actor_cue_shadow'        # fcpg/v21_cue_cem.py
CUE_SOURCE = 'v21_lewm_cue_cem'
ADMITTED_SOURCES = (ACTION_SOURCE, EXECUTION_GEOMETRY_SOURCE, LEWM_CEM_SOURCE, C4_SHADOW_SOURCE, C4_SOURCE,
                    ALIGN_SHADOW_SOURCE, ALIGN_SOURCE, EXPLORE_SOURCE, CUE_SHADOW_SOURCE, CUE_SOURCE)
# a trial never mixes in another source
EXCLUSIVE_SOURCES = (LEWM_CEM_SOURCE, C4_SHADOW_SOURCE, C4_SOURCE, ALIGN_SHADOW_SOURCE, ALIGN_SOURCE,
                     EXPLORE_SOURCE, CUE_SHADOW_SOURCE, CUE_SOURCE)
CAMERA_CONDITIONS = ('both_fresh', 'wrist_h0', 'wrist_cold', 'external_h0', 'external_cold')
ADDRESS_MODES = ('factual', 'zero')
HUMAN_LABELS = ('unverified', 'success', 'fail')
FAIL_TYPES = ('unverified', 'miss', 'push', 'tail-only', 'empty-close',
              'human-abort', 'collision-stop')
_UNSET = object()


def _fallback_message(source):
    if source == LEWM_CEM_SOURCE:
        return 'LeWM CEM cannot fall back to Actor'
    return f'{source} cannot fall back to another action source'


def _as_action_source(value):
    if value is None:
        return ACTION_SOURCE
    if value not in ADMITTED_SOURCES:
        raise ValueError('capped trial action_source must be v21_actor, v21_actor_execution_geometry, '
                         'v21_lewm_cem, v21_actor_c4_shadow or v21_c4_cem '
                         '(or v21_actor_align_shadow, v21_lewm_align_cem, v21_actor_explore, '
                         'v21_actor_cue_shadow, v21_lewm_cue_cem)')
    return value


def _as_camera_condition(value):
    if value is None:
        return None
    if value not in CAMERA_CONDITIONS:
        raise ValueError('unknown camera_condition')
    return value


def _as_address_mode(value):
    if value is None:
        return None
    if value not in ADDRESS_MODES:
        raise ValueError('address_mode must be factual or zero')
    return value


def _as_reset_pose(value):
    if value is None:
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError('reset pose must be six finite joint values') from exc
    if len(values) != 6 or not all(math.isfinite(item) for item in values):
        raise ValueError('reset pose must be six finite joint values')
    return values


def _as_human_label(name, value):
    aliases = {'yes': 'success', 'no': 'fail'}
    value = aliases.get(value, value)
    if value not in HUMAN_LABELS:
        raise ValueError(f'{name} must be unverified, success, or fail')
    return value


def _as_fail_type(value):
    if value in (None, ''):
        value = 'unverified'
    if value not in FAIL_TYPES:
        raise ValueError('unknown fail_type')
    return value


class CappedTrial:
    """Keep chat latency out of a maximum-sixteen-block trial.

    Its cancellation latch is distinct from the normal per-block HOLD/STOP.
    Only explicitly starting a new trial clears this latch.
    live_service may call bind_experiment(...) or start(..., camera_condition=)
    before dispatch, and record_human_scores(...) after the run.
    """
    def __init__(self, plan, execute, halt, save, *, max_blocks=16, clock=time.monotonic,
                 last_attempt=None, action_source=ACTION_SOURCE, camera_condition=None,
                 address_mode=None, reset_pose=None):
        if not 1 <= max_blocks <= 32:
            raise ValueError('trial budget must be between 1 and 32 blocks')
        self.plan, self.execute, self.halt, self.save = plan, execute, halt, save
        self.max_blocks, self.clock = max_blocks, clock
        self.last_attempt = last_attempt
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.gate = threading.Event()
        self.thread = None
        self._action_source = _as_action_source(action_source)
        self._camera_condition = _as_camera_condition(camera_condition)
        self._address_mode = _as_address_mode(address_mode)
        self._reset_pose = _as_reset_pose(reset_pose)
        self.data = dict(running=False, outcome='not_started', blocks_started=0,
                         blocks_completed=0, policy_commands_returned=0, max_blocks=max_blocks,
                         task_success_verified=False, records=[], **self._logging_fields())

    def _logging_fields(self, *, reset_human=True):
        fields = dict(action_source=self._action_source,
                      camera_condition=self._camera_condition,
                      address_mode=self._address_mode,
                      reset_pose=deepcopy(self._reset_pose))
        if reset_human:
            fields.update(grasp='unverified', place='unverified', fail_type='unverified')
        return fields

    def bind_experiment(self, *, action_source=_UNSET, camera_condition=_UNSET,
                        address_mode=_UNSET, reset_pose=_UNSET):
        """Attach identity for a later start(); does not plan or dispatch."""
        with self.lock:
            if self.data.get('running') or (self.thread is not None and self.thread.is_alive()):
                raise RuntimeError('cannot rebind a running trial')
            if action_source is not _UNSET:
                self._action_source = _as_action_source(action_source)
            if camera_condition is not _UNSET:
                self._camera_condition = _as_camera_condition(camera_condition)
            if address_mode is not _UNSET:
                self._address_mode = _as_address_mode(address_mode)
            if reset_pose is not _UNSET:
                self._reset_pose = _as_reset_pose(reset_pose)
            if self.data.get('outcome') == 'not_started':
                self.data.update(self._logging_fields())
        return self.status()

    def record_human_scores(self, *, grasp='unverified', place='unverified',
                            fail_type='unverified'):
        """Store operator grasp/place/fail labels. Never infers task success."""
        scores = dict(grasp=_as_human_label('grasp', grasp),
                      place=_as_human_label('place', place),
                      fail_type=_as_fail_type(fail_type))
        with self.lock:
            self.data.update(scores)
            self.data['task_success_verified'] = False
            started = 'trial_id' in self.data
        if started:
            try:
                self.save(self.status())
            except Exception as exc:
                with self.lock:
                    self.data['save_error'] = str(exc)
        return self.status()

    @property
    def running(self):
        with self.lock: return self.data['running']

    def owns_current_thread(self):
        return threading.current_thread() is self.thread

    def status(self):
        with self.lock: return deepcopy(self.data)

    def start(self, *, paused=False, action_source=_UNSET, camera_condition=_UNSET,
              address_mode=_UNSET, reset_pose=_UNSET):
        with self.lock:
            if self.data['running'] or (self.thread is not None and self.thread.is_alive()):
                raise RuntimeError('a capped trial is already running')
        self.bind_experiment(action_source=action_source, camera_condition=camera_condition,
                             address_mode=address_mode, reset_pose=reset_pose)
        with self.lock:
            if self.data['running'] or (self.thread is not None and self.thread.is_alive()):
                raise RuntimeError('a capped trial is already running')
            self.cancelled.clear(); self.gate.clear()
            self.data = dict(running=True, outcome='running', trial_id=str(time.time_ns()),
                started_s=self.clock(), blocks_started=0, blocks_completed=0,
                policy_commands_returned=0, max_blocks=self.max_blocks,
                task_success_verified=False, records=[], **self._logging_fields())
            # Persist the trial intent before any worker can plan or dispatch.
            # A process exit during the first block must leave an unresolved
            # started trial, not silently remove it from the experiment count.
            try:
                self.save(self.status())
            except BaseException as exc:
                self.data.update(running=False, outcome='start_log_failed',
                    error=f'{type(exc).__name__}: {exc}', ended_s=self.clock())
                self.cancelled.set()
                raise
            if not paused: self.gate.set()
            self.thread = threading.Thread(target=self._run, name='v21-capped-trial', daemon=True)
            try:
                self.thread.start()
            except BaseException as exc:
                self.data.update(running=False, outcome='worker_start_failed',
                    error=f'{type(exc).__name__}: {exc}', ended_s=self.clock())
                self.cancelled.set()
                if not self.thread.is_alive(): self.thread=None
                try: self.save(self.status())
                except Exception as save_exc: self.data['save_error']=str(save_exc)
                raise

    def release_start(self):
        self.gate.set()

    def cancel(self):
        self.cancelled.set()
        self.gate.set()  # also cancel a trial that has not begun dispatch

    def join(self, timeout=5.):
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive(): raise RuntimeError('trial worker has not stopped')

    def _run(self):
        outcome, error = 'budget_exhausted', None
        previous_end = None
        try:
            self.gate.wait()
            for block in range(1, self.max_blocks+1):
                if self.cancelled.is_set():
                    outcome='operator_stopped'; break
                planned = self.plan()
                if self.cancelled.is_set():
                    outcome='operator_stopped'; break
                planned_source = (planned.get('plan') or {}).get('action_source')
                if self._action_source in EXCLUSIVE_SOURCES and planned_source != self._action_source:
                    raise RuntimeError(_fallback_message(self._action_source))
                if (not planned.get('ready')
                        or planned_source not in ADMITTED_SOURCES):
                    raise RuntimeError('trial requires an admitted Actor, execution-geometry, or LeWM CEM plan')
                with self.lock: self.data['blocks_started']=block
                if self.cancelled.is_set():
                    outcome='operator_stopped'; break
                try:
                    result = self.execute()
                except BaseException:
                    self._record_failed_attempt()
                    raise
                actual = result.get('last_execution') or {}
                if self._action_source in EXCLUSIVE_SOURCES and actual.get('action_source') != self._action_source:
                    raise RuntimeError(_fallback_message(self._action_source))
                stop_err=str(result.get('stop_error') or '')
                chunk_done=(bool(actual.get('completed'))
                    and len(actual.get('commands_position') or [])==5
                    and actual.get('policy_commands_returned')==5)
                packet_latch='no status packet' in stop_err.lower()
                # #region agent log
                try:
                    import json as _json, time as _time
                    with open('/home/james/feedback-corrected-grounding/.cursor/debug-55ff20.log','a') as _f:
                        _f.write(_json.dumps({'sessionId':'55ff20','runId':'post-fix','hypothesisId':'C','location':'v21_capped_trial.py:_run','message':'chunk contract','data':{'block':block,'chunk_done':chunk_done,'packet_latch':packet_latch,'error':str(result.get('error') or '')[:160],'stop_error':stop_err[:160]},'timestamp':int(_time.time()*1000)})+'\n')
                except Exception:
                    pass
                # #endregion
                if (result.get('error') or (stop_err and not (packet_latch and chunk_done)) or not actual.get('completed')
                        or actual.get('action_source') not in ADMITTED_SOURCES
                        or len(actual.get('commands_position') or [])!=5
                        or actual.get('policy_commands_returned')!=5):
                    raise RuntimeError('five-command transaction did not complete as contracted')
                record = {k:deepcopy(actual.get(k)) for k in (
                    'source_time_s','endpoint_time_s','commands_position','command_times_s',
                    'readback_times_s','measured_q_path','feedback_q_path','normal_follow',
                    'following_q_path','following_readback_times_s','model_feedback_path','policy_input_snapshot',
                    'planning_observation_source','action_source','block_io','lewm_cem','c4_planner','alignment_cem','explore','cue_cem')}
                record['block']=block
                record['inter_block_gap_s']=None if previous_end is None else actual['source_time_s']-previous_end
                previous_end=actual['endpoint_time_s']
                with self.lock:
                    self.data['records'].append(record)
                    self.data['blocks_completed']=block
                    self.data['policy_commands_returned']+=5
                self.save(self.status())
                if actual.get('four_stage_sequence_complete') is True:
                    outcome = 'four_stage_sequence_complete'
                    break
                if (actual.get('alignment_cem') or {}).get('released_over_box') is True:
                    outcome = 'released_over_box'
                    break
            if self.cancelled.is_set(): outcome='operator_stopped'
        except BaseException as exc:
            outcome='operator_stopped' if self.cancelled.is_set() else 'execution_error'
            error=f'{type(exc).__name__}: {exc}'
        finally:
            try: self.halt()
            except Exception as exc:
                outcome='stop_error'; error=f'{type(exc).__name__}: {exc}'
            with self.lock:
                self.data.update(running=False,outcome=outcome,error=error,ended_s=self.clock())
            try: self.save(self.status())
            except Exception as exc:
                with self.lock: self.data['save_error']=str(exc)

    def _record_failed_attempt(self):
        """Count known returned commands even when a five-step block aborts."""
        attempt = None if self.last_attempt is None else self.last_attempt()
        if not isinstance(attempt, dict):
            with self.lock: self.data['partial_attempt_accounting_available'] = False
            return
        attempted = attempt.get('policy_commands_attempted')
        returned = attempt.get('policy_commands_returned')
        if (type(attempted) is not int or type(returned) is not int
                or not 0 <= returned <= attempted <= 5):
            with self.lock: self.data['partial_attempt_accounting_available'] = False
            return
        with self.lock:
            self.data['partial_attempt_accounting_available'] = True
            self.data['partial_attempt'] = deepcopy(attempt)
            self.data['partial_policy_commands_attempted'] = attempted
            self.data['partial_policy_commands_returned'] = returned
            self.data['policy_commands_returned'] += returned
