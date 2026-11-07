"""CPU-only contract checks for changing GPU resources within one scratch run."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from fcpg.v21_training import V21Schedule
from scripts import train_v21_from_scratch as fresh


RESOURCE_FIELDS = {'gpu_memory_limit_gib', 'gpu_memory_fraction', 'feature_bank_gib'}


def original_config():
    total = 85651*1024**2
    return dict(recipe=fresh.GPU_RECIPE, device='cuda:0', smoke=False,
        external_weights_loaded=False, external_checkpoint_reads=0,
        initialization='module-default-random-all-components-created-before-first-stage',
        cuda_visible_devices='GPU-476b3a2e-008f-5dd3-fa77-99063d068e51',
        gpu_memory_limit_gib=15., gpu_memory_fraction=15*1024**3/total,
        gpu_total_memory_bytes=total, feature_bank_gib=0.,
        precision='fp32', tf32_enabled=False, threads=8, microbatch=64,
        source_preparation_device='cpu', training_chain_transfer='preserve-dtype-nonblocking-v1',
        source_paths={'archive': '/original/recorded-source'},
        body_table='/original/object-relative.npy', body_manifest='/original/body-manifest.json',
        model={'seed': 20000, 'object_relative': True}, schedule=asdict(V21Schedule()),
        vocabulary=['<pad>', '<unk>', '<bos>', '<eos>', 'pick'],
        q_loss_scale=[1., 2., 3., 4., 5., 6.], training_data_contract={'train': 272000},
        measurement_dropout={'protocol': 'burst-cold-v1'},
        temporal_feature_policy='completed-actual-readbacks')


def splits():
    return dict(train=dict(archive=np.array([0, 2, 4]), dense=np.array([2, 4]), fish=np.array([0, 1])),
        development=dict(archive=np.array([1, 3]), dense=np.array([3]), fish=np.array([2])))


def larger_resources(previous):
    result = deepcopy(previous)
    result.update(gpu_memory_limit_gib=72.,
        gpu_memory_fraction=72*1024**3/result['gpu_total_memory_bytes'], feature_bank_gib=48.)
    return result


def files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def history_files(root):
    return sorted(set(root.glob('resources-history*.json'))
        | set((root/'resources-history').glob('*.json')))


def tree_scalars(value):
    if isinstance(value, dict):
        return [item for child in value.values() for item in tree_scalars(child)]
    if isinstance(value, list):
        return [item for child in value for item in tree_scalars(child)]
    return [value]


@pytest.fixture
def recorded_run(tmp_path):
    root = tmp_path/'scratch'
    previous, populations = original_config(), splits()
    fresh.configure_output(root, previous, populations, resume=False)
    (root/'actor').mkdir()
    (root/'actor/latest.pt').write_bytes(b'configuration-only checkpoint placeholder')
    # These belong to the earlier CPU -> GPU migration and must not be rewritten.
    (root/'config-before-gpu.json').write_text('{"recipe": "original-cpu-config"}\n')
    (root/'device-continuation.json').write_text('{"source_device": "cpu", "marker": 197}\n')
    return root, previous, populations


def migrate(root, config, populations, **kwargs):
    return fresh.configure_output(root, config, populations,
        resume=True, resume_with_resources=True, **kwargs)


def test_larger_cap_and_feature_cache_preserve_original_run_and_record_each_change(recorded_run):
    root, previous, populations = recorded_run
    protected = {name: value for name, value in files(root).items() if name != 'config.json'}
    changed = larger_resources(previous)
    assert {key for key in previous if previous[key] != changed[key]} == RESOURCE_FIELDS
    migrate(root, changed, populations)
    assert json.loads((root/'config.json').read_text()) == changed
    for name, contents in protected.items():
        assert (root/name).read_bytes() == contents, name
    first_history = {str(path): path.read_bytes() for path in history_files(root)}
    assert len(first_history) == 1
    receipt = json.loads(next(iter(first_history.values())))
    assert isinstance(receipt, dict)
    values = tree_scalars(receipt)
    assert all(value in values for value in (15., 72., 0., 48.))
    assert any(('time' in key.lower() or 'timestamp' in key.lower())
        for key in receipt), 'resource migration must record when it happened'

    # Consecutive resource changes must append records even within one second.
    changed_again = deepcopy(changed)
    changed_again['feature_bank_gib'] = 40.
    migrate(root, changed_again, populations)
    assert json.loads((root/'config.json').read_text()) == changed_again
    assert len(history_files(root)) == 2
    for path, contents in first_history.items():
        assert Path(path).read_bytes() == contents
    for name, contents in protected.items():
        assert (root/name).read_bytes() == contents, name


def test_legacy_zero_cache_plain_resume_remains_byte_exact(recorded_run):
    root, previous, populations = recorded_run
    before = files(root)
    fresh.configure_output(root, previous, populations, resume=True)
    assert files(root) == before


@pytest.mark.parametrize('field,value', [
    ('cuda_visible_devices', 'GPU-96e31ee1-51e0-bd4a-45b9-3ca04da88de0'),
    ('threads', 16), ('microbatch', 32), ('precision', 'bf16'), ('tf32_enabled', True),
    ('gpu_total_memory_bytes', 123456789), ('source_preparation_device', 'cuda:0'),
    ('training_chain_transfer', 'another-transfer'), ('source_paths', {'archive': '/other'}),
    ('body_table', '/other.npy'), ('body_manifest', '/other.json'),
    ('model', {'seed': 20001, 'object_relative': True}), ('schedule', {'chains_per_batch': 128}),
    ('vocabulary', ['other']), ('q_loss_scale', [2.]*6),
    ('training_data_contract', {'train': 320000}), ('measurement_dropout', {'protocol': 'legacy'}),
    ('temporal_feature_policy', 'clock-time'), ('initialization', 'pretrained'),
    ('external_checkpoint_reads', 1), ('unrecognized_execution_option', True),
])
def test_every_non_resource_change_is_rejected_without_writing(recorded_run, field, value):
    root, previous, populations = recorded_run
    changed = larger_resources(previous)
    changed[field] = value
    before = files(root)
    with pytest.raises(ValueError):
        migrate(root, changed, populations)
    assert files(root) == before


@pytest.mark.parametrize('field,value', [
    ('recipe', fresh.RECIPE), ('recipe', 'other-recipe'), ('device', 'cpu'),
    ('device', 'cuda:1'), ('smoke', True), ('external_weights_loaded', True),
    ('external_weights_loaded', None),
])
@pytest.mark.parametrize('which', ['previous', 'current', 'both'])
def test_resource_migration_requires_formal_gpu_scratch_on_both_sides(recorded_run, field, value, which):
    root, previous, populations = recorded_run
    changed = larger_resources(previous)
    if which in ('previous', 'both'):
        previous[field] = value
        (root/'config.json').write_text(json.dumps(previous))
    if which in ('current', 'both'):
        changed[field] = value
    before = files(root)
    with pytest.raises(ValueError):
        migrate(root, changed, populations)
    assert files(root) == before


@pytest.mark.parametrize('fault', ['reorder', 'replace', 'remove', 'add'])
def test_resource_change_cannot_mask_recorded_split_drift(recorded_run, fault):
    root, previous, populations = recorded_run
    with np.load(root/'training-rows.npz') as recorded:
        rows = {key: recorded[key] for key in recorded.files}
    if fault == 'reorder':
        rows['train_archive'] = rows['train_archive'][::-1]
    elif fault == 'replace':
        rows['development_fish'] = np.array([3])
    elif fault == 'remove':
        rows.pop('development_dense')
    else:
        rows['undeclared_rows'] = np.array([123])
    np.savez(root/'training-rows.npz', **rows)
    before = files(root)
    with pytest.raises(ValueError):
        migrate(root, larger_resources(previous), populations)
    assert files(root) == before


def test_resource_migration_requires_an_existing_stage_checkpoint(recorded_run):
    root, previous, populations = recorded_run
    (root/'actor/latest.pt').unlink()
    before = files(root)
    with pytest.raises(ValueError):
        migrate(root, larger_resources(previous), populations)
    assert files(root) == before


def test_final_checkpoint_also_admits_resource_migration(recorded_run):
    root, previous, populations = recorded_run
    (root/'actor/latest.pt').rename(root/'actor/final.pt')
    changed = larger_resources(previous)
    migrate(root, changed, populations)
    assert json.loads((root/'config.json').read_text()) == changed
    assert (root/'actor/final.pt').read_bytes() == b'configuration-only checkpoint placeholder'


def test_resource_changes_require_explicit_resource_flag(recorded_run):
    root, previous, populations = recorded_run
    before = files(root)
    with pytest.raises(ValueError):
        fresh.configure_output(root, larger_resources(previous), populations, resume=True)
    assert files(root) == before


def test_resource_flag_requires_resume_before_output_creation(tmp_path):
    root = tmp_path/'absent'
    with pytest.raises(ValueError):
        fresh.configure_output(root, original_config(), splits(), resume=False, resume_with_resources=True)
    assert not root.exists()


def test_gpu_continuation_and_resource_flags_are_mutually_exclusive(recorded_run):
    root, previous, populations = recorded_run
    before = files(root)
    with pytest.raises(ValueError):
        migrate(root, larger_resources(previous), populations, resume_on_gpu=True)
    assert files(root) == before
