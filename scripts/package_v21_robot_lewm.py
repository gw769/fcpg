#!/usr/bin/env python3
"""The robot package: an Actor package (scripts/package_v21_fcpg_actor_candidate.py) with its LeWM replaced by a
picture-memory LeWM (the last N pictures per camera, memory_pictures / memory_tokens, with the balanced and the
control-setting inputs), then loaded the way the robot service loads it (fcpg.v21_runtime.load_runtime on the GPU) and
checked:
- every Actor/FCPG tensor equal to the package's, every LeWM tensor equal to the run's; the runtime keeps N pictures
  per camera in its states, empty at every new state, and gives the LeWM the service's setting [0, 1];
- on the first 64 robot exam blocks, with their memories from the exam streams, the loaded LeWM forecasts as the run's
  LeWM (within 1e-5);
- the service's rule (fcpg/v21_picture_memory.py write_pictures: each new exposure in, the oldest out, empty at every
  new state) rebuilds every exam block's memory exactly: each state's blocks in order, their new pictures pushed one by
  one, equal to the exam stream's memory of that block (tokens, joints, times, validity).
Usage (V21_ENCODER=dinov3_vitl16): python scripts/package_v21_robot_lewm.py PACKAGE LEWM_RUN OUTPUT
       --exam EXAM.pt --memory EXAM-MEMORY.pt --arm-p ARM-P.json
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

os.environ.setdefault('V21_ENCODER', 'dinov3_vitl16')
os.environ.setdefault('V21_DINOV2', 'off')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fcpg.v21_camera_types import LEGACY_SLOTS, REVIEWED_TYPES  # noqa: E402
from fcpg.v21_command_memory import FULL_COMMANDS, LEGACY_COMMANDS  # noqa: E402
from fcpg.v21_full_patch_lewm import V21FullPatchLeWM, architecture_from_state  # noqa: E402
from fcpg.v21_native_horizon import core_from_parent  # noqa: E402
from fcpg.v21_picture_memory import RobotPictureMemory, empty_pictures, write_pictures  # noqa: E402
from fcpg.v21_robot_block_pack import RobotBlocks  # noqa: E402

LIMITS = ('position_lower', 'position_upper', 'position_center', 'position_half_range')
CONDITIONING = ('lewm_command_conditioning', 'lewm_camera_conditioning')
TASK = 'Pick up the red fish toy and place it in the cardboard box'


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def same(a, b):
    return a is not None and b is not None and a.shape == b.shape and torch.equal(a.float(), b.float())


def service_rule_check(pack, stream, block_state, pictures):
    """Every exam block's memory by the service's rule against the exam stream's (robot_memory_index): each state starts
    empty; block by block, each camera's pictures newer than the last one taken in enter one by one, oldest first."""
    memory, now = pack['data']['memory'], pack['data']['now']
    times, valid = memory['image_times'], memory['valid']
    differing, state, live, last = 0, None, None, None
    for r in range(len(block_state)):
        if block_state[r] != state:
            state, live, last = block_state[r], empty_pictures(1, pictures, dtype=torch.float16), [-np.inf, -np.inf]
        for c in range(2):
            for j in sorted((j for j in range(times.shape[2]) if bool(valid[r, c, j]) and float(times[r, c, j]) > last[c]),
                            key=lambda j: float(times[r, c, j])):
                inserted = torch.zeros(1, 2, dtype=torch.bool)
                inserted[0, c] = True
                exposure = SimpleNamespace(features=memory['features'][r, :, j][None], q_at_image=memory['q_at_image'][r, :, j][None],
                                           image_times=times[r, :, j][None])
                live = write_pictures(live, exposure, inserted)
                last[c] = float(times[r, c, j])
        expected = stream.memory(np.array([r]), now[r:r + 1], pictures)
        differing += not (torch.equal(live.valid, expected.valid) and torch.equal(live.image_times, expected.image_times)
                          and torch.equal(live.q_at_image[live.valid], expected.q_at_image[expected.valid])
                          and torch.equal(live.features[live.valid], expected.features[expected.valid]))
    return dict(blocks=len(block_state), states=int(len(set(block_state.tolist()))), differing=differing)


def forbidden(*args, **kwargs):
    raise RuntimeError('no robot here')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('package', type=Path, help='the Actor package')
    p.add_argument('run', type=Path, help='the picture-memory LeWM training run')
    p.add_argument('output', type=Path)
    p.add_argument('--exam', type=Path, required=True, help='the robot exam pack (exam.pt)')
    p.add_argument('--memory', type=Path, required=True, help="the exam pack's picture streams (exam-memory50.pt)")
    p.add_argument('--arm-p', type=Path, required=True, help='the arm P table of the robot sessions')
    args = p.parse_args()
    package, run, output = args.package, args.run, args.output
    if output.exists():
        raise FileExistsError(f'{output} exists')
    source = run / 'fish' / 'final.pt'
    protocol = json.loads((run / 'protocol.json').read_text())
    config = json.loads((package / 'config.json').read_text())
    payload = torch.load(package / 'fish' / 'final.pt', map_location='cpu', weights_only=False)
    donor = torch.load(source, map_location='cpu', weights_only=False)
    if donor.get('complete') is not True or donor.get('stage') != 'fish' or donor.get('smoke') is not False:
        raise ValueError("the LeWM checkpoint is not a completed formal run's final weights")
    new = {k: v for k, v in donor['model'].items() if k.startswith('lewm.')}
    old = {k: v for k, v in payload['model'].items() if k.startswith('lewm.')}
    size = architecture_from_state(new, 'lewm.')
    if not size.get('memory_pictures') or not size.get('controller_input') or not size.get('balanced_inputs'):
        raise ValueError('not a picture-memory LeWM with the balanced and control-setting inputs')
    recorded = protocol.get('lewm_picture_memory') or {}
    if recorded.get('pictures') != size['memory_pictures'] or (
            'memory_tokens' in size and recorded.get('tokens') != size['memory_tokens']):   # concat_inputs: no pool
        raise ValueError("the run's protocol and its weights differ on the picture memory")
    expected = {f'lewm.{k}': tuple(v.shape) for k, v in V21FullPatchLeWM(horizon=40, **size).state_dict().items()}
    if {k: tuple(v.shape) for k, v in new.items()} != expected:
        raise ValueError("the checkpoint's lewm.* weights are not a full-patch LeWM of their size")
    added = set(new) - set(old)
    allowed = {'lewm.controller_weight', 'lewm.command_symlog'} | {
        f'lewm.layers.{i}.norm_m.{p}' for i in range(size['layers']) for p in ('weight', 'bias')}
    # a LeWM with concatenated inputs (concat_inputs) has its own layout: its shapes were checked above
    if not size.get('concat_inputs') and (set(old) - set(new)
                                         or any(k not in allowed and not k.startswith('lewm.memory_') for k in added)):
        raise ValueError("the new LeWM's tensors are not the replaced ones plus the controller, bank-norm, symlog and "
                         'memory tensors')
    ranges = {}
    for name in LIMITS:
        mine, actor = new.get(f'lewm.{name}'), payload['model'].get(f'actor.{name}')
        if not same(mine, actor):
            raise ValueError(f"the new LeWM's {name} differs from the Actor's")
        ranges[name] = [round(float(x), 6) for x in mine]
    if not same(new['lewm.increment_scale'], donor['model'].get('q_loss_scale')):
        raise ValueError('the LeWM was trained with another q_loss_scale')
    ranges['increment_scale'] = [round(float(x), 6) for x in new['lewm.increment_scale']]
    ranges['replaced_lewm_increment_scale'] = [round(float(x), 6) for x in old['lewm.increment_scale']]
    modes = dict(lewm_command_conditioning=(FULL_COMMANDS if any(k.startswith('lewm.command_projection.') for k in old)
                                            else LEGACY_COMMANDS),
                 lewm_camera_conditioning=REVIEWED_TYPES if 'lewm.camera_type_embedding' in old else LEGACY_SLOTS)
    for key in CONDITIONING:
        if donor.get(key, modes[key]) != modes[key]:
            raise ValueError(f'{key} differs: {donor.get(key)!r} (new) against {modes[key]!r} (package)')
    if donor['config'].get('encoder') != config['model'].get('encoder'):
        raise ValueError('the LeWM was trained on another encoder')
    record = dict(package=str(package), package_checkpoint_sha256=sha(package / 'fish' / 'final.pt'),
                  lewm_checkpoint=str(source), lewm_checkpoint_sha256=sha(source), lewm_architecture=size,
                  lewm_parameters=sum(v.numel() for v in new.values()), replaced_lewm_tensors=len(old),
                  added_lewm_tensors=sorted(added), camera_layout=protocol.get('archive_camera_layout'),
                  picture_memory=recorded,
                  controller=dict(inputs=['arm_p32', 'planner'],
                                  weight_norm=None if 'lewm.controller_weight' not in new else
                                  [round(float(x), 4) for x in new['lewm.controller_weight'].float().norm(dim=1)],
                                  robot_setting='load_runtime(arm_p_coefficient=None) -> [0, 1]',
                                  training=protocol.get('lewm_controller')),
                  **modes, joint_ranges=ranges,
                  note='Actor/FCPG tensors and fields unchanged; the runtime states keep the last N pictures per '
                       'camera, empty at every new task')
    payload['model'] = {**{k: v for k, v in payload['model'].items() if not k.startswith('lewm.')}, **new}
    payload['lewm_swap'] = record
    (output / 'fish').mkdir(parents=True)
    torch.save(payload, output / 'fish' / 'final.pt.tmp')
    (output / 'fish' / 'final.pt.tmp').replace(output / 'fish' / 'final.pt')
    (output / 'config.json').write_text(json.dumps(dict(config, lewm_swap=record), indent=2))

    from fcpg.v21_runtime import load_runtime
    torch.backends.mha.set_fastpath_enabled(False)
    runtime = load_runtime(output, TASK, device='cuda', physical_check=forbidden, lease_check=forbidden)
    core = runtime.core
    lewm = core.lewm
    n = int(size['memory_pictures'])
    original = torch.load(package / 'fish' / 'final.pt', map_location='cpu', weights_only=False)['model']
    state = {k: v.detach().cpu() for k, v in core.state_dict().items()}
    kept = [k for k in state if not k.startswith('lewm.')]
    assert set(state) == set(kept) | set(new), 'loaded keys differ'
    assert all(torch.equal(state[k], original[k]) for k in kept), 'an Actor/FCPG tensor changed'
    assert all(torch.equal(state[k], new[k]) for k in new), 'a LeWM tensor differs from the LeWM checkpoint'
    assert all(torch.equal(state[f'lewm.{n_}'], state[f'actor.{n_}']) for n_ in LIMITS), 'loaded limits differ'
    assert int(lewm.memory_pictures) == n and core.picture_slots() == n, 'the runtime keeps no picture memory'
    assert lewm.controller_input and torch.equal(core.arm_setting.cpu(), torch.tensor([0., 1.])), 'arm setting'
    assert getattr(lewm, 'balanced_inputs', False) and hasattr(lewm, 'command_symlog') == bool(size.get('command_symlog'))
    start = core.initial_state(torch.zeros(1, 6, device='cuda'), torch.zeros(1, dtype=torch.float64, device='cuda'))
    assert start.pictures.valid.shape == (1, 2, n) and not start.pictures.valid.any(), 'a new state must start empty'
    trained = core_from_parent(donor).to('cuda').eval().lewm.requires_grad_(False)
    pack = torch.load(args.exam, map_location='cpu', weights_only=False)
    stream_index = torch.load(args.memory, map_location='cpu', weights_only=False)
    blocks = RobotBlocks(pack, 'cuda', None, split='exam', arm_p=json.loads(args.arm_p.read_text()))
    stream = RobotPictureMemory(stream_index, 'cuda')
    ids = np.arange(64)
    q, trace, history, qt, actual, _, now, _ = blocks.batch(ids)
    memory = stream.memory(ids, now, n)
    pictures = dict(patches=memory.features, patch_valid=memory.valid, memory=memory, now_time=now)
    service = core.arm_setting.to(q).expand(len(q), 2)
    with torch.no_grad():
        loaded = lewm(q, trace, history, qt, controller=service, **pictures).predicted_q_path
        from_run = trained(q, trace, history, qt, controller=service, **pictures).predicted_q_path
        own_setting = lewm(q, trace, history, qt, controller=blocks.controller(ids), **pictures).predicted_q_path
    checks = dict(service_setting_loaded_vs_run=float((loaded - from_run).abs().max()),
                  error_first5_service_setting=float((loaded[:, 1:] - actual).abs().mean()),
                  error_first5_own_setting=float((own_setting[:, 1:] - actual).abs().mean()), blocks=len(ids),
                  service_rule=service_rule_check(pack, RobotPictureMemory(stream_index, 'cpu'),
                                                  stream_index['block_state'].numpy(), n))
    assert checks['service_setting_loaded_vs_run'] <= 1e-5, checks
    assert checks['service_rule']['differing'] == 0, checks
    (output / 'swap-check.json').write_text(json.dumps(dict(record, output=str(output), checks=checks,
                                                            actor_fcpg_tensors_unchanged=len(kept),
                                                            lewm_tensors_from_checkpoint=len(new)), indent=1))
    print(json.dumps(dict(output=str(output), loaded_lewm=lewm.architecture(), checks=checks,
                          actor_fcpg_tensors_unchanged=len(kept), lewm_tensors_from_checkpoint=len(new)), indent=1))


if __name__ == '__main__':
    main()
