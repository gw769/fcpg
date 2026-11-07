"""Bound the live 40-step rollout near the object before LeWM prediction.

This is an explicit execution heuristic in calibrated joint units. Image
proximity is not a depth measurement or a verified grasp/contact condition.
"""
import torch


NEAR_DISTANCE_PX = 80.


def select_approach_commands(q, full, object_relative):
    if q.shape != (1, 6) or full.shape != (1, 40, 6):
        raise ValueError('approach scheduling requires one actual q and 40 commands')
    if not torch.isfinite(q).all() or not torch.isfinite(full).all():
        raise ValueError('approach scheduling requires finite positions')
    seq = full[0]
    actual = q[0].to(seq)
    previous = torch.cat((actual[None], seq[:-1]), 0)
    mean_step = float((seq - previous).abs().mean())
    cue_valid = (isinstance(object_relative, torch.Tensor)
                 and object_relative.shape == (1, 8)
                 and bool(torch.isfinite(object_relative).all())
                 and float(object_relative[0, 0]) > 0)
    distance = (float(torch.linalg.vector_norm(object_relative[0, 2:4]) * 224.)
                if cue_valid else None)
    conservative = distance is None or distance <= NEAR_DISTANCE_PX
    # Five writes from the measured pose to the fifth Actor command.
    # The same 0.2..1 ramp used by the place/home chunks.
    count = 5
    actor_index = count - 1
    delta = seq[actor_index] - actual
    alphas = torch.linspace(1 / count, 1, count, device=seq.device, dtype=seq.dtype)
    commands = actual[None] + alphas[:, None] * delta[None]
    note = dict(inferred_steps=40, execute_steps=count, mean_abs_step=mean_step,
        small_step_extension_enabled=False, path='uniform_chunk_start_to_end',
        execution_policy='uniform_five_from_start_to_end_v1',
        proximity_source='current_actor_object_relative', wrist_target_distance_px=distance,
        near_distance_px=NEAR_DISTANCE_PX, near_target=distance is not None and conservative,
        conservative_approach=conservative, actor_target_index=actor_index,
        command_scale=1.0, maximum_arm_delta=None,
        close_deferred_until_centered=False,
        engineering_heuristic=True, physical_alignment_verified=False)
    return commands, note
