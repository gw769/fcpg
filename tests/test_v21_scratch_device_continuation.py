"""A device continuation changes resources, never the scratch experiment."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_training import V21Schedule, save_checkpoint, load_checkpoint
from scripts import train_v21_from_scratch as fresh


def configs():
    previous = dict(recipe=fresh.RECIPE, device='cpu', threads=8, microbatch=8,
        precision='fp32', external_weights_loaded=False, smoke=False,
        initialization='module-default-random-all-components-created-before-first-stage',
        schedule=asdict(V21Schedule()), q_loss_scale=[2.]*6,
        training_data_contract=fresh.DATA_CONTRACT, source_paths={'archive':'same-root'})
    current = dict(previous, recipe=fresh.GPU_RECIPE, device='cuda:0', microbatch=64,
        cuda_visible_devices='0', gpu_memory_limit_gib=15., source_preparation_device='cpu',
        training_chain_transfer='preserve-dtype-nonblocking-v1')
    return previous, current


def populations():
    return dict(train=dict(archive=np.array([0,2]), dense=np.array([2]), fish=np.array([0])),
        development=dict(archive=np.array([1]), dense=np.array([1]), fish=np.array([1])))


def test_resource_migration_preserves_real_checkpoint_and_split(tmp_path):
    previous, current = configs()
    splits = populations()
    fresh.configure_output(tmp_path, previous, splits, resume=False)
    vocabulary = TaskVocabulary.build(['pick red block'])
    core = V21ResearchCore(V21Config(len(vocabulary.words)))
    schedule = V21Schedule()
    path = tmp_path/'lewm/latest.pt'
    save_checkpoint(path, core, vocabulary, schedule, stage='lewm', step=500,
                    complete=False, smoke=False, metrics=dict(updates=500))
    original_bytes = path.read_bytes()
    fresh.configure_output(tmp_path, current, splits, resume=True, resume_on_gpu=True)
    assert path.read_bytes() == original_bytes
    assert json.loads((tmp_path/'config-before-gpu.json').read_text()) == previous
    assert json.loads((tmp_path/'config.json').read_text()) == current
    assert json.loads((tmp_path/'device-continuation.json').read_text())['same_random_start_run'] is True
    fresh.configure_output(tmp_path, current, splits, resume=True)
    payload = load_checkpoint(path, core, vocabulary, schedule, stage='lewm')
    assert payload['step'] == 500 and payload['metrics']['updates'] == 500


@pytest.mark.parametrize('change', [
    {'q_loss_scale':[3.]*6}, {'precision':'bf16'}, {'source_paths':{'archive':'other-root'}},
    {'schedule':asdict(V21Schedule(archive_passes=1))}, {'external_weights_loaded':True},
    {'training_data_contract':{}}, {'initialization':'pretrained'},
])
def test_resource_migration_cannot_change_training_experiment(change):
    previous, current = configs()
    current.update(change)
    with pytest.raises(ValueError, match='cannot change'):
        fresh.validate_gpu_continuation(previous, current)


def test_no_checkpoint_and_changed_rows_cannot_reconfigure_output(tmp_path):
    previous, current = configs()
    splits = populations()
    fresh.configure_output(tmp_path, previous, splits, resume=False)
    with pytest.raises(ValueError, match='saved checkpoint'):
        fresh.configure_output(tmp_path, current, splits, resume=True, resume_on_gpu=True)
    (tmp_path/'lewm').mkdir(); (tmp_path/'lewm/latest.pt').write_bytes(b'identity-only fixture')
    altered = deepcopy(splits); altered['train']['archive'] = np.array([2,0])
    with pytest.raises(ValueError, match='identities changed'):
        fresh.configure_output(tmp_path, current, altered, resume=True, resume_on_gpu=True)
    assert json.loads((tmp_path/'config.json').read_text()) == previous


def test_device_change_requires_explicit_resume_option(tmp_path):
    previous, current = configs()
    splits = populations()
    fresh.configure_output(tmp_path, previous, splits, resume=False)
    with pytest.raises(ValueError, match='exact scratch run configuration'):
        fresh.configure_output(tmp_path, current, splits, resume=True)
    with pytest.raises(ValueError, match='requires --resume'):
        fresh.configure_output(tmp_path, current, splits, resume=False, resume_on_gpu=True)
