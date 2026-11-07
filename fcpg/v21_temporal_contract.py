"""Explicit temporal compatibility for the audited V21 Actor checkpoint.

Times remain factual float64 clocks. Only consumer masks and explicitly declared
unsupported projection columns change. This does not train long-age behavior or
validate LeWM's use of old actuator history.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import MethodType

import torch

from .v21_checks import checking, deferring, fail_if, later
from .v21_memory import clock

WALL_AGE_PROFILE = 'audited-3k-fish-age-compatibility-v1'
PROGRESS_PROFILE = 'actual-readback-progress-compatibility-v1'
PROFILE = PROGRESS_PROFILE
AUDITED_CHECKPOINT_SHA256 = '9525fd08ee01b335509ffc962dc77185004d6a6aa7402281045bbe3c4eeec875'
SUPPORTED_PROFILE = PROFILE
SUPPORTED_PROFILES = (PROGRESS_PROFILE, WALL_AGE_PROFILE)
EXPECTED_PARENT_SHA256 = AUDITED_CHECKPOINT_SHA256
FISH_TRAINING_MAX_VISUAL_AGE_S = 1.5
TRAINING_POLICY = 'zero_unsupported_actor_ages_v1'
PROGRESS_TRAINING_POLICY = 'actual_readback_progress_v1'
TRAINING_POLICIES = ('legacy', TRAINING_POLICY, PROGRESS_TRAINING_POLICY)
READBACKS_PER_COMPLETED_BLOCK = 5
MAX_VISUAL_PROGRESS_SAMPLES = 15
PROGRESS_FEATURE_SCALE = .1  # dimensionless compatibility scale; never a measured duration
AGE_COLUMNS = (('actor.q_error.0.weight', 31), ('actor.flags.weight', 4))


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def validate_temporal_checkpoint(checkpoint_path, profile=PROFILE):
    """Preflight file identity without model construction or physical IO."""
    if profile not in SUPPORTED_PROFILES:
        raise ValueError('unknown temporal compatibility profile')
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checksum = file_hash(checkpoint_path)
    if checksum != AUDITED_CHECKPOINT_SHA256:
        raise ValueError('temporal compatibility is restricted to the audited checkpoint SHA256')
    return dict(profile=profile, parent_checkpoint=str(checkpoint_path), parent_sha256=checksum)


def visual_actor_hash(core):
    result = hashlib.sha256()
    for name, value in sorted(core.state_dict().items()):
        if name.startswith(('reader.', 'actor.')):
            result.update(name.encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def prepare_progress_state(state):
    """Bind legacy snapshots to an explicitly local, capped progress origin.

    Only the three retained completed blocks can be reconstructed from an old
    snapshot. This is not a claim to know its global historical sample count.
    """
    memory = state.memory
    if (getattr(memory, 'readback_indices', None) is not None
            and getattr(memory, 'current_readback_index', None) is not None
            and getattr(state, 'feedback_readback_index', None) is not None
            and getattr(state, 'visual_feedback_readback_indices', None) is not None):
        return state
    ends = state.history_ends
    valid = state.history.valid
    current = valid.sum(-1).long()*READBACKS_PER_COMPLETED_BLOCK
    def at_time(times):
        expanded_ends = ends.reshape(len(ends), *([1]*(times.ndim-1)), 3)
        expanded_valid = valid.reshape(len(valid), *([1]*(times.ndim-1)), 3)
        return ((expanded_ends <= times[..., None]+1e-9) & expanded_valid).sum(-1).long()*READBACKS_PER_COMPLETED_BLOCK
    bound = replace(memory, readback_indices=at_time(memory.image_times), current_readback_index=current)
    local_identity = hashlib.sha256((tensor_hash(ends)+tensor_hash(valid)).encode()).hexdigest()[:16]
    return replace(state, memory=bound, feedback_readback_index=at_time(state.feedback_time),
        visual_feedback_readback_indices=at_time(state.visual_feedback_time),
        progress_session_id='legacy-local-'+local_identity,
        progress_origin='reconstructed_from_up_to_three_retained_actual_blocks; local capped distance, not global sample count')


def progress_distance(memory):
    indices = getattr(memory, 'readback_indices', None)
    current = getattr(memory, 'current_readback_index', None)
    if indices is None or current is None:
        raise ValueError('progress profile requires bound indices; call prepare_progress_state(saved_state) before direct reader replay')
    if (indices.shape != memory.valid.shape or current.shape != (len(memory.features),)
            or indices.dtype != torch.long or current.dtype != torch.long):
        raise ValueError('invalid completed readback progress shape or dtype')
    if checking():
        fail_if((current < 0).any() | (memory.valid & (indices < 0)).any(), 'invalid completed readback progress shape or dtype')
    distance = current[:, None, None]-indices
    if checking():
        fail_if((memory.valid & (distance < 0)).any(), 'image progress index is after the current actual readback index')
    return distance


def _consumer_contract(core):
    contract = getattr(core, 'temporal_contract_identity', None)
    if contract is not None:
        return contract
    training = getattr(core, 'temporal_training_contract', None)
    if training is not None and training['policy'] == PROGRESS_TRAINING_POLICY:
        return dict(profile=PROGRESS_PROFILE, max_visual_progress_samples=MAX_VISUAL_PROGRESS_SAMPLES,
                    progress_feature_scale=PROGRESS_FEATURE_SCALE)
    return None


def uses_progress(core):
    contract = _consumer_contract(core)
    return contract is not None and contract['profile'] == PROGRESS_PROFILE


def actor_temporal_features(core, state):
    """Return model compatibility features, leaving source clocks untouched."""
    if uses_progress(core):
        current = state.memory.current_readback_index
        execution = current-state.feedback_readback_index
        visual = current[:, None]-state.visual_feedback_readback_indices
        if checking():
            fail_if((state.feedback_available & (execution < 0)).any() | (state.visual_feedback_available & (visual < 0)).any(),
                    'feedback progress cannot be from a later completion')
        return (torch.where(state.feedback_available, execution, torch.zeros_like(execution)).float()*PROGRESS_FEATURE_SCALE,
                torch.where(state.visual_feedback_available, visual, torch.zeros_like(visual)).float()*PROGRESS_FEATURE_SCALE)
    return (torch.where(state.feedback_available, state.time-state.feedback_time, torch.zeros_like(state.time)).float(),
            torch.where(state.visual_feedback_available, state.time[:, None]-state.visual_feedback_time,
                        torch.zeros_like(state.visual_feedback_time)).float())


def mask_memory_by_age(memory, now, maximum_age_s):
    clock('temporal consumer time', now, (len(memory.features),))
    age = now[:, None, None]-memory.image_times
    if checking():
        fail_if((memory.valid & (age < -1e-9)).any(), 'temporal consumer cannot use future visual evidence')
    valid = memory.valid & (age <= maximum_age_s+1e-9)
    return _with_valid(memory, valid)


def _with_valid(memory, valid):
    """The memory with these valid flags. Section 17.80: while checks are deferred, the copy is made even when nothing
    changed (the same values; asking whether anything changed would make the CPU wait for the GPU)."""
    if deferring():
        return replace(memory, valid=valid)
    return memory if torch.equal(valid, memory.valid) else replace(memory, valid=valid)


def effective_memory(core, memory, now):
    contract = _consumer_contract(core)
    if contract is None:
        return memory
    if contract['profile'] == PROGRESS_PROFILE:
        distance = progress_distance(memory)
        valid = memory.valid & (distance <= contract['max_visual_progress_samples'])
        return _with_valid(memory, valid)
    return mask_memory_by_age(memory, now, contract['max_visual_age_s'])


def temporal_actor_state(core, state):
    """One ephemeral view supplies all Actor visual inputs; raw state survives."""
    contract = _consumer_contract(core)
    if contract is None:
        return state
    if uses_progress(core):
        state = prepare_progress_state(state)
        distance = state.memory.current_readback_index[:, None]-state.visual_feedback_readback_indices
        within = distance <= contract['max_visual_progress_samples']
    else:
        age = state.time[:, None]-state.visual_feedback_time
        if checking():
            fail_if((state.visual_feedback_available & (age < -1e-9)).any(), 'visual feedback event is in the future')
        within = age <= contract['max_visual_age_s']+1e-9
    memory = effective_memory(core, state.memory, state.time)
    visual_valid = state.visual_feedback_available & memory.available & within
    relative = state.object_relative
    if relative is not None:
        relative = torch.where(memory.valid[:, :1, -1], relative, torch.zeros_like(relative))
    place = {}
    if getattr(state, 'place_relative', None) is not None:   # the place token, same wrist-frame rule (section 17)
        place = dict(place_relative=torch.where(memory.valid[:, :1, -1], state.place_relative,
                                            torch.zeros_like(state.place_relative)))
    return replace(state, memory=memory,
        grounding=core.reader(memory, state.q, state.time, state.history, task_ids=getattr(state, 'task_ids', None)),
        visual_error=torch.where(visual_valid[..., None, None], state.visual_error, torch.zeros_like(state.visual_error)),
        visual_feedback_available=visual_valid,
        measurement_pending=state.measurement_pending & memory.available,
        object_relative=relative, **place)


def _install_consumer_hooks(core):
    if getattr(core, '_temporal_reader_hook', None) is not None:
        return
    original_metadata = core.reader.frame_metadata
    def metadata(reader, memory, actual_q, now, history):
        values = original_metadata(memory, actual_q, now, history)
        if uses_progress(core):
            values = values.clone()
            feature = torch.log1p(progress_distance(memory).clamp_min(0).to(values)*PROGRESS_FEATURE_SCALE)
            values[..., 24] = torch.where(memory.valid, feature, torch.zeros_like(feature))
        return values
    core.reader.frame_metadata = MethodType(metadata, core.reader)
    def filter_reader(_module, args, kwargs):
        memory = args[0] if args else kwargs['memory']
        now = args[2] if len(args) >= 3 else kwargs['now']
        filtered = effective_memory(core, memory, now)
        if args:
            return (filtered, *args[1:]), kwargs
        return args, dict(kwargs, memory=filtered)
    core._temporal_reader_hook = core.reader.register_forward_pre_hook(filter_reader, with_kwargs=True)
    core.reader._projection_cache = None


def install_temporal_contract(core, checkpoint_path, *, profile=PROFILE,
                              max_visual_age_s=FISH_TRAINING_MAX_VISUAL_AGE_S):
    """Install once on a loaded core; return a replayable model identity.

    The parent checkpoint and current Actor/reader must match exactly. An adapted
    LeWM is allowed because neither its parameters nor actual history is changed.
    """
    if (isinstance(max_visual_age_s, bool) or not math.isfinite(max_visual_age_s)
            or not 0 < max_visual_age_s <= FISH_TRAINING_MAX_VISUAL_AGE_S):
        raise ValueError('visual age limit exceeds the verified fish-chain support')
    preflight = validate_temporal_checkpoint(checkpoint_path, profile)
    checkpoint_path = Path(preflight['parent_checkpoint'])
    checksum = preflight['parent_sha256']
    existing = getattr(core, 'temporal_contract_identity', None)
    if existing is not None:
        expected_limit = max_visual_age_s if profile == WALL_AGE_PROFILE else None
        if (existing['profile'] != profile or existing['max_visual_age_s'] != expected_limit
                or existing['reader_actor_state_sha256_after'] != visual_actor_hash(core)):
            raise ValueError('temporal profile already installed with a different configuration or changed weights')
        return deepcopy(existing)
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    source = payload['model']
    for name, value in core.state_dict().items():
        if name.startswith(('reader.', 'actor.')) and (name not in source or not torch.equal(value.cpu(), source[name])):
            raise ValueError('loaded Actor/reader differs from the audited checkpoint: '+name)
    before = visual_actor_hash(core)
    parameters = dict(core.named_parameters())
    columns = []
    if profile == WALL_AGE_PROFILE:
        columns = [dict(parameter=name, column=index, original_sha256=tensor_hash(parameters[name][:, index]))
                   for name, index in AGE_COLUMNS]
        with torch.no_grad():
            for name, index in AGE_COLUMNS:
                parameters[name][:, index].zero_()
    identity = dict(schema='v21-temporal-compatibility-v1', profile=profile,
        parent_checkpoint=str(checkpoint_path), parent_sha256=checksum,
        max_visual_age_s=float(max_visual_age_s) if profile == WALL_AGE_PROFILE else None,
        max_visual_progress_samples=MAX_VISUAL_PROGRESS_SAMPLES if profile == PROGRESS_PROFILE else None,
        progress_feature_scale=PROGRESS_FEATURE_SCALE if profile == PROGRESS_PROFILE else None,
        modified_columns=columns,
        reader_actor_state_sha256_before=before, reader_actor_state_sha256_after=visual_actor_hash(core),
        true_timestamps_preserved=True,
        unsupported_age_projection=('neutralized_explicitly_not_retrained' if profile == WALL_AGE_PROFILE
            else 'real-clock age is not fed to Actor; explicitly scaled completed-readback distance uses the existing projection'),
        visual_evidence_policy=('mask_age_exceeding_fish_training_support_in_one_consumer_view' if profile == WALL_AGE_PROFILE
            else 'mask only after more than 15 completed readback samples; waiting never advances this distance'),
        raw_visual_memory_preserved=True, actual_q_error_and_history_preserved=True,
        execution_history_age_policy='unchanged; LeWM old-history support is outside this compatibility profile',
        support_scope='fish Actor H0..H3 at 15/30 s; archive population age support not claimed',
        progress_semantics=('q1..q5 add five only on a completed actual policy block; fresh exposure binds current index; HOME/new state resets session'
                            if profile == PROGRESS_PROFILE else None),
        feature_encoding=('log1p(0.1 * completed_readback_sample_distance); dimensionless compatibility feature, not measured seconds'
                          if profile == PROGRESS_PROFILE else None),
        old_snapshot_policy=('local capped reconstruction from up to three factual history_ends; never asserted to be global sequence'
                             if profile == PROGRESS_PROFILE else None),
        control_influence=True, experimental=True, independently_validated_task_accuracy=False)
    identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    core.temporal_contract_identity = identity
    _install_consumer_hooks(core)
    return deepcopy(identity)


def begin_training_temporal_contract(core, policy):
    if policy not in TRAINING_POLICIES:
        raise ValueError('unknown temporal training feature policy')
    if policy == 'legacy':
        if getattr(core, 'temporal_training_contract', None) is not None:
            raise ValueError('checkpoint with an explicit temporal training policy requires that same policy')
        return
    if getattr(core, 'temporal_contract_identity', None) is not None:
        raise ValueError('deployment compatibility is not a training input policy')
    core.temporal_training_contract = dict(schema='v21-temporal-training-feature-contract-v1',
        policy=policy, columns=[dict(parameter=name, column=index) for name, index in AGE_COLUMNS],
        clock_mutation=False, generated_labels=False,
        action='zero and freeze unsupported columns; reject positive ages until an explicit supported training recipe exists')
    core._temporal_collect_training_coverage = True
    if policy == PROGRESS_TRAINING_POLICY:
        core.temporal_training_contract.update(feature_encoding='log1p(0.1 * completed_readback_sample_distance)',
            feature_unit='dimensionless, not measured seconds', max_visual_progress_samples=MAX_VISUAL_PROGRESS_SAMPLES,
            action='train on completed-readback progress; zero unsupported latest-feedback projection columns')
        _install_consumer_hooks(core)
    core.temporal_feature_coverage = dict(actor_calls=0, rows=0, available_execution_feedback=0,
        available_visual_feedback=0, execution_age_max_s=0., visual_age_max_s=0.,
        execution_age_nonzero_count=0, visual_age_nonzero_count=0,
        consumed_visual_progress_max_samples=0, consumed_visual_progress_nonzero_count=0,
        counting_semantics='observed forward calls including retries; no generated labels')
    constrain_training_age_columns(core)


def record_training_temporal_features(core, state):
    coverage = getattr(core, 'temporal_feature_coverage', None)
    if coverage is None or not getattr(core, '_temporal_collect_training_coverage', False):
        return
    age = torch.where(state.feedback_available, state.time-state.feedback_time, torch.zeros_like(state.time))
    visual_age = torch.where(state.visual_feedback_available,
        state.time[:, None]-state.visual_feedback_time, torch.zeros_like(state.visual_feedback_time))
    if checking():
        fail_if(~torch.isfinite(age).all() | ~torch.isfinite(visual_age).all() | (age < 0).any() | (visual_age < 0).any(),
                'training temporal feature has invalid factual age')
    coverage['actor_calls'] += 1; coverage['rows'] += len(state.q)
    # section 17.80: the counts are read in one go (later(): at once, or when a deferred update's checks are read)
    counts = [state.feedback_available.sum(), state.visual_feedback_available.sum(), age.max(), visual_age.max(),
              (age.abs()>1e-9).sum(), (visual_age.abs()>1e-9).sum()]
    progress = uses_progress(core)
    if progress:
        bound = prepare_progress_state(state)
        memory = effective_memory(core, bound.memory, bound.time)
        distance = progress_distance(memory)       # its valid entries, counted without selecting them
        counts += [memory.valid.sum(), torch.where(memory.valid, distance, torch.full_like(distance, -1)).max(),
                   ((distance > 0) & memory.valid).sum()]

    def count(values):
        coverage['available_execution_feedback'] += int(values[0])
        coverage['available_visual_feedback'] += int(values[1])
        coverage['execution_age_max_s'] = max(coverage['execution_age_max_s'], values[2])
        coverage['visual_age_max_s'] = max(coverage['visual_age_max_s'], values[3])
        coverage['execution_age_nonzero_count'] += int(values[4])
        coverage['visual_age_nonzero_count'] += int(values[5])
        if progress and values[6] > 0:
            coverage['consumed_visual_progress_max_samples'] = max(
                coverage['consumed_visual_progress_max_samples'], int(values[7]))
            coverage['consumed_visual_progress_nonzero_count'] += int(values[8])

    later(torch.stack([c.double() for c in counts]), count)
    if core.temporal_training_contract['policy'] == TRAINING_POLICY:
        if checking():
            fail_if((age.abs()>1e-9).any() | (visual_age.abs()>1e-9).any(),
                    'zero-unsupported-age training policy requires zero supported ages; declare a new recipe for nonzero ages')


def constrain_training_age_columns(core, optimizer=None):
    if getattr(core, 'temporal_training_contract', None) is None:
        return
    parameters = dict(core.named_parameters())
    with torch.no_grad():
        for name, index in AGE_COLUMNS:
            parameter = parameters[name]
            parameter[:, index].zero_()
            if parameter.grad is not None:
                parameter.grad[:, index].zero_()
            if optimizer is not None:
                for value in optimizer.state.get(parameter, {}).values():
                    if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                        value[:, index].zero_()


def restore_training_temporal_consumer(core):
    """Honor a new checkpoint's declared progress features when loading it."""
    if uses_progress(core):
        _install_consumer_hooks(core)


def temporal_training_configuration(policy):
    """Keep legacy run configs unchanged; new semantics are immutable metadata."""
    if policy not in TRAINING_POLICIES:
        raise ValueError('unknown temporal training feature policy')
    if policy == 'legacy':
        return {}
    return dict(temporal_feature_policy=policy, temporal_input_contract=dict(
        scope='Actor and FCPG only; LeWM uses factual command/readback time intervals',
        feature_encoding=('log1p(0.1 * completed_readback_sample_distance)' if policy == PROGRESS_TRAINING_POLICY
                          else 'legacy relative clock ages with explicitly zero unsupported Actor age columns'),
        feature_unit='dimensionless, not measured seconds' if policy == PROGRESS_TRAINING_POLICY else 'legacy',
        max_visual_progress_samples=MAX_VISUAL_PROGRESS_SAMPLES if policy == PROGRESS_TRAINING_POLICY else None,
        clock_mutation=False, generated_labels=False,
        unsupported_actor_age_columns='zero-initialize and constrain; save feature coverage with checkpoint'))
