#!/usr/bin/env python3
"""Make a kept fish-stage step of a scripts/train_v21_fcpg_actor.py run loadable by fcpg.v21_runtime.load_runtime.

The training run keeps intermediate fish steps (actor_fish/step-XXXXX.pt); the registered rule in
docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md picks one of them. This writes that step as a model directory
(config.json + fish/final.pt) with the weights untouched and its origin recorded. No training, no robot.
Usage: python scripts/package_v21_fcpg_actor_candidate.py <run dir> <fish step> <new model dir>
"""
import hashlib
import json
from pathlib import Path
import sys

import torch

KIND = 'v21-fcpg-actor-overhaul-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    run, step, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    if out.exists():
        raise SystemExit(f'{out} exists')
    protocol = json.loads((run / 'protocol.json').read_text())
    if protocol.get('kind') != KIND or protocol.get('smoke') is not False:
        raise SystemExit('not a formal FCPG/Actor overhaul run')
    final = step == protocol['actor_fish']['updates']
    source = run / 'fish' / 'final.pt' if final else run / 'actor_fish' / f'step-{step:05d}.pt'
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if (payload.get('step') != step or payload.get('smoke') is not False
            or payload.get('stage') != ('fish' if final else 'actor_fish') or payload.get('complete') is not final):
        raise SystemExit('checkpoint is not the requested fish step of this run')
    origin = dict(run=str(run), step=step, checkpoint=str(source), checkpoint_sha256=sha256(source),
                  final_step_of_run=final)
    payload.update(stage='fish', complete=True, packaged_from=origin)
    (out / 'fish').mkdir(parents=True)
    torch.save(payload, out / 'fish' / 'final.pt')
    (out / 'config.json').write_text(json.dumps(dict(
        kind=KIND, contract=payload['contract'], smoke=False, complete=True, model=payload['config'],
        vocabulary=payload['vocabulary'], fish_updates=step, protocol=protocol, packaged_from=origin,
        deployment_approved=False, physical_success_verified=False), indent=1) + '\n')
    print(json.dumps(dict(model=str(out), **origin)))


if __name__ == '__main__':
    main()
