"""Native 40-query Actor/LeWM preview, separate from five-step deployment."""
from copy import deepcopy
from dataclasses import replace
import hashlib

import torch
from torch import nn

from .v21_core import V21Config, V21ResearchCore
from .v21_models import CommandTrace, admit_untrained_scene_projections
from .v21_temporal_contract import restore_training_temporal_consumer
from .v21_command_memory import command_mode_from_checkpoint
from .v21_camera_types import LEGACY_SLOTS, camera_mode_from_checkpoint

SCHEMA = 'v21-native-horizon40-preview-v1'


def core_from_parent(payload):
    core = V21ResearchCore(V21Config(**payload['config']))
    if payload.get('scene_lewm_architecture') is not None:
        from .v21_scene_lewm import (V21SceneLeWM, scene_normalization_from_checkpoint,
                                    scene_prediction_from_checkpoint)
        core.lewm = V21SceneLeWM(horizon=40,
            normalization=scene_normalization_from_checkpoint(payload),
            scene_prediction_mode=scene_prediction_from_checkpoint(payload),
            command_conditioning=command_mode_from_checkpoint(payload))
        from .v21_scene_fg import fg_checkpoint_mode, enable_fg_readout
        if fg_checkpoint_mode(payload) is not None:
            enable_fg_readout(core.lewm)
        from .v21_scene_q_grounding import q_checkpoint_mode, enable_q_readout
        if q_checkpoint_mode(payload) is not None:
            enable_q_readout(core.lewm)
    elif any(key.startswith(('lewm.film.', 'lewm.concat_inputs')) for key in payload['model']):
        from .v21_full_patch_lewm import V21FullPatchLeWM, architecture_from_state
        core.lewm = V21FullPatchLeWM(horizon=40,
            command_conditioning=command_mode_from_checkpoint(payload),
            **architecture_from_state(payload['model'], 'lewm.'))
    camera_mode = camera_mode_from_checkpoint(payload)
    if camera_mode != LEGACY_SLOTS:
        if not hasattr(core.lewm, 'enable_camera_conditioning'):
            raise ValueError('typed camera checkpoint requires full-patch LeWM')
        core.lewm.enable_camera_conditioning(camera_mode)
    admit_untrained_scene_projections(payload['model'], core.state_dict())
    core.load_state_dict(payload['model'], strict=True)
    if payload.get('temporal_training_contract'):
        core.temporal_training_contract = deepcopy(payload['temporal_training_contract'])
        core.temporal_feature_coverage = deepcopy(payload.get('temporal_feature_coverage'))
        restore_training_temporal_consumer(core)
    return core


def initialize_native40(core):
    """Warm-start query slots; this is initialization, never trained evidence."""
    for module in (core.actor, core.lewm):
        if module.query.shape != (1, 5, 192):
            raise ValueError('native horizon initialization requires five learned query slots')
        module.query = nn.Parameter(module.query.detach().repeat(1, 8, 1).clone())
        module.horizon = 40
    core.config = replace(core.config, prediction_horizon=40)
    return core


@torch.inference_mode()
def native_outlook(core, state, text_ids):
    if core.actor.horizon != 40 or core.lewm.horizon != 40:
        raise ValueError('native outlook requires 40-query Actor and LeWM')
    offsets = torch.arange(40, device=state.q.device, dtype=torch.float64)[None] * .1
    commands = core.act(state, text_ids, offsets).commands_position
    trace = CommandTrace(commands, offsets, torch.ones_like(offsets, dtype=torch.bool), 'planned', True)
    predicted = core.lewm(state.q, trace, state.history, offsets + .1,
                          patches=state.memory.features, patch_valid=state.memory.valid).predicted_q_path
    return dict(available=True, executable=False, control_input=False,
        kind='native_actor40_lewm40', steps=40, native_horizon=40,
        actor_forward_calls=1, lewm_forward_calls=1, actual_future_labels_available=False,
        source_time_s=float(state.time[0]), source_q_position=state.q[0].tolist(),
        commands_position=commands[0].tolist(), predicted_q_path=predicted[0].tolist(),
        command_offsets_s=offsets[0].tolist(), readback_offsets_s=[i * .1 for i in range(41)])


def load_native_preview(path, parent_checkpoint):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('schema') != SCHEMA or payload.get('training_complete') is not True
            or payload.get('parent_sha256') != hashlib.sha256(parent_checkpoint.read_bytes()).hexdigest()
            or payload.get('optimizer_steps', 0) <= 0):
        raise ValueError('native40 preview checkpoint is incomplete or has a different parent')
    core = initialize_native40(core_from_parent(torch.load(parent_checkpoint, map_location='cpu', weights_only=False)))
    admit_untrained_scene_projections(payload['model'], core.state_dict())
    core.load_state_dict(payload['model'], strict=True)
    return core.eval(), payload['validation']
