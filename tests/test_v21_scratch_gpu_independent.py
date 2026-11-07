"""Shared-GPU admission tests using CPU-only doubles; no GPU is queried or used."""
from types import SimpleNamespace
from dataclasses import fields, is_dataclass, replace
import json
import random

import numpy as np
import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary, V21Sources
from fcpg.v21_training import synthetic_chain
from scripts import train_v21_from_scratch as fresh


@pytest.fixture(autouse=True)
def no_real_cuda(monkeypatch):
    """Any accidental lazy CUDA initialization fails this CPU-only test file."""
    previous_threads = torch.get_num_threads()
    previous_precision = torch.get_float32_matmul_precision()
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    python_rng, numpy_rng, tensor_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('real CUDA initialization forbidden'))
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda _seed: None)
    torch.set_num_threads(2)
    yield
    random.setstate(python_rng); np.random.set_state(numpy_rng); torch.set_rng_state(tensor_rng)
    torch.set_num_threads(previous_threads)
    torch.set_float32_matmul_precision(previous_precision)
    torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32


def virtual_gpu(monkeypatch, events, *, visible='0', count=1, total_gib=86):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: count)
    def properties(device):
        assert device == torch.device('cuda:0')
        events.append(('properties', device))
        return SimpleNamespace(total_memory=total_gib*2**30)
    def cap(fraction, device):
        assert device == torch.device('cuda:0')
        events.append(('cap', fraction, device))
    def select(device):
        assert device == torch.device('cuda:0')
        events.append(('select', device))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', properties)
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', cap)
    monkeypatch.setattr(torch.cuda, 'set_device', select)


def test_cpu_default_does_not_query_cuda_or_change_its_settings(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('CPU default queried CUDA')
    for name in ('is_available', 'device_count', 'get_device_properties', 'set_device',
                 'set_per_process_memory_fraction'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    before = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    device, config = fresh.configure_device()
    assert device == torch.device('cpu') and config == {}
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before


@pytest.mark.parametrize('visible', ['0', 'GPU-12345678-1234-5678-90ab-123456789abc'])
def test_single_explicit_gpu_sets_exact_eight_gib_allocator_cap(monkeypatch, visible):
    events = []
    virtual_gpu(monkeypatch, events, visible=visible)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device, config = fresh.configure_device('cuda:0', 8.)
    assert device == torch.device('cuda:0')
    assert [event[0] for event in events] == ['properties', 'cap', 'select']
    assert events[1][1] * (86*2**30) == pytest.approx(8*2**30)
    assert config['gpu_memory_limit_gib'] == 8.
    assert config['gpu_memory_fraction'] == events[1][1]
    assert config['gpu_total_memory_bytes'] == 86*2**30
    assert config['cuda_visible_devices'] == visible
    assert config['tf32_enabled'] is False
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32


def test_ambiguous_masks_multiple_devices_and_invalid_caps_are_rejected_before_cap(monkeypatch):
    events = []
    virtual_gpu(monkeypatch, events)
    for visible in ('', '0,1', '-1', 'GPU-short', ' 0'):
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
        with pytest.raises(ValueError, match='explicit GPU'):
            fresh.configure_device('cuda:0', 8.)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    with pytest.raises(ValueError, match='exactly one'):
        fresh.configure_device('cuda:0', 8.)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    for cap in (0., -1., float('nan'), float('inf'), 87.):
        with pytest.raises(ValueError, match='limit'):
            fresh.configure_device('cuda:0', cap)
    for device in ('cuda', 'cuda:1', 'mps'):
        with pytest.raises(ValueError, match='logical cuda:0'):
            fresh.configure_device(device, 8.)
    assert not any(event[0] in ('cap', 'select') for event in events)


def test_source_upload_targets_selected_device_and_keeps_factual_dtypes(monkeypatch):
    source = object.__new__(V21Sources)
    source.device = torch.device('cuda:0')
    observed = []
    def intercepted_to(value, *args, **kwargs):
        observed.append((kwargs['device'], kwargs['dtype']))
        return value
    monkeypatch.setattr(torch.Tensor, 'to', intercepted_to)
    for value in (np.zeros((1, 6), np.float32), np.zeros((1, 6), np.float64),
                  np.ones((1, 2), bool), np.ones((1,), np.int64)):
        tensor = source._tensor(value)
        assert observed[-1] == (torch.device('cuda:0'), tensor.dtype)
        assert tensor.numpy().dtype == value.dtype


def test_gpu_main_caps_before_source_and_migrates_the_same_cpu_random_core(tmp_path, monkeypatch):
    events = []
    virtual_gpu(monkeypatch, events)
    vocabulary = TaskVocabulary.build(['pick red block'])
    scale = np.asarray([2., 3., 4., 5., 6., 7.], np.float32)
    torch.manual_seed(20000)
    reference = V21ResearchCore(V21Config(len(vocabulary.words), seed=20000, object_relative=True))
    reference.set_loss_scale(scale)
    expected = {name: value.clone() for name, value in reference.state_dict().items()}
    del reference
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('fresh GPU run read weights'))
    class DeviceView:
        def __init__(self, core, device):
            self.core, self.device = core, device
        def __getattr__(self, name):
            return getattr(self.core, name)
        def parameters(self):
            return (SimpleNamespace(device=self.device, dtype=p.dtype) for p in self.core.parameters())
    class InitializationProbe(V21ResearchCore):
        def to(self, *, device, dtype):
            assert device == torch.device('cuda:0') and dtype == torch.float32
            assert all(p.device.type == 'cpu' for p in self.parameters())
            for name, value in self.state_dict().items():
                assert torch.equal(value, expected[name]), name
            events.append(('transfer', device))
            return DeviceView(self, device)
    monkeypatch.setattr(fresh, 'V21ResearchCore', InitializationProbe)
    source = SimpleNamespace(vocabulary=vocabulary, fish=object())
    def source_factory(paths, *, device, object_relative_table):
        assert any(event[0] == 'cap' for event in events)
        assert torch.device(device) == torch.device('cpu')
        source.device = torch.device(device)
        source.device_cache, source.compact_host_features = object(), True
        events.append(('source', device))
        return source
    monkeypatch.setattr(fresh, 'V21Sources', source_factory)
    splits = dict(train=dict(archive=np.arange(48000, 320000), dense=np.arange(48000, 65000), fish=np.arange(381)),
        development=dict(archive=np.arange(48000), dense=np.arange(3000), fish=np.arange(381, 486)),
        fish_required_endpoints=np.asarray([0, 15, 30]))
    original = {'fish_wrist_root': '/fixture/fish'}
    mapping = tmp_path/'source-map.json'
    mapping.write_text(json.dumps(dict(source_paths_original=original)))
    monkeypatch.setattr(fresh, 'resolve_source_paths', lambda *a: (original, {}))
    monkeypatch.setattr(fresh, 'derive_splits', lambda s: splits)
    monkeypatch.setattr(fresh, 'validate_manifest', lambda *a, **kw: {})
    monkeypatch.setattr(fresh, 'training_loss_scale', lambda s, rows: scale)
    monkeypatch.setattr(fresh, 'pin_source_map', lambda *a: None)
    class TrainerProbe:
        def __init__(self, core, vocab, output, schedule, **options):
            assert core.device == torch.device('cuda:0') and source.device == torch.device('cpu')
            assert options['precision'] == 'fp32' and options['smoke'] is True
            self.core = core
            events.append(('trainer', source.device))
    monkeypatch.setattr(fresh, 'ScratchTrainer', TrainerProbe)
    def train(trainer, actual_source, populations, **options):
        assert isinstance(actual_source, fresh.DeviceStagedSources) and actual_source is not source
        assert actual_source.device == trainer.core.device == torch.device('cuda:0')
        assert actual_source.vocabulary is source.vocabulary and populations is splits['train']
        assert source.device_cache is None and source.compact_host_features is False
        assert options['canary'] is True
        events.append(('train', source.device))
    monkeypatch.setattr(fresh, 'run_stages', train)
    output = tmp_path/'new-gpu'
    assert fresh.main(['--source-map', str(mapping), '--body-table', str(tmp_path/'body.npy'),
        '--body-manifest', str(tmp_path/'body.json'), '--output', str(output), '--device', 'cuda:0',
        '--gpu-memory-limit-gib', '8', '--two-step-canary', '--threads', '2', '--microbatch', '1']) == 0
    assert [event[0] for event in events] == ['properties', 'cap', 'select', 'source', 'transfer', 'trainer', 'train']
    config = json.loads((output/'config.json').read_text())
    assert config['recipe'] == fresh.GPU_RECIPE and config['device'] == 'cuda:0'
    assert config['precision'] == 'fp32' and config['feature_bank_gib'] == 0.
    assert config['gpu_memory_limit_gib'] == 8. and config['cuda_visible_devices'] == '0'
    assert config['external_weights_loaded'] is False and config['external_checkpoint_reads'] == 0
    assert config['smoke'] is True and config['training_data_contract'] == fresh.DATA_CONTRACT


def chain_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from chain_tensors(getattr(value, field.name))
    elif isinstance(value, tuple):
        for member in value:
            yield from chain_tensors(member)


def assert_same_payload(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif is_dataclass(actual):
        for field in fields(actual):
            assert_same_payload(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_same_payload(a, b)
    else:
        assert actual == expected


def factual_chain(*, images=True):
    vocabulary = TaskVocabulary.build(['pick red block'])
    batch = synthetic_chain(vocabulary, batch=2, images=images)
    return replace(batch, object_relative=torch.arange(80, dtype=torch.float32).reshape(2, 5, 8),
        actor_visual_drop=torch.tensor([False, True]), source_index=torch.tensor([-1, 3]))


@pytest.mark.parametrize('images', [False, True])
def test_complete_chain_staging_preserves_every_dtype_value_none_and_metadata(images):
    original = factual_chain(images=images)
    if not images:
        original = replace(original, object_relative=None, actor_visual_drop=None)
    before = [(tensor, tensor.clone()) for tensor in chain_tensors(original)]
    moved = fresh.stage_chain_to_device(original, torch.device('cpu'))
    assert_same_payload(moved, original)
    assert all(tensor.device.type == 'cpu' for tensor in chain_tensors(moved))
    assert moved.endpoint_times.dtype == moved.readback_offsets.dtype == torch.float64
    assert moved.q.dtype == moved.commands.dtype == moved.paths.dtype == torch.float32
    assert moved.view_valid.dtype == torch.bool
    assert moved.text_ids.dtype == moved.source_index.dtype == torch.int64
    for actual, expected in zip(moved.traces, original.traces):
        assert actual.offsets.dtype == torch.float64 and actual.valid.dtype == torch.bool
        assert actual.source == expected.source == 'recorded' and actual.complete is expected.complete is True
    for tensor, saved in before:
        assert torch.equal(tensor, saved)


def test_every_chain_tensor_is_copied_to_one_cuda_device_without_cast_or_source_mutation(monkeypatch):
    original = factual_chain()
    # Moving data cannot upgrade an incomplete/proposed trace into factual data.
    proposed = replace(original.traces[0], source='planned', complete=False)
    original = replace(original, traces=(proposed, *original.traces[1:]))
    before = [(tensor, tensor.clone()) for tensor in chain_tensors(original)]
    calls = []
    def fake_copy(value, *args, **kwargs):
        device = torch.device(kwargs.get('device', args[0] if args else 'cpu'))
        assert device == torch.device('cuda:0') and kwargs.get('non_blocking') is True
        assert kwargs.get('dtype', value.dtype) == value.dtype
        calls.append(id(value))
        return value.clone()  # CPU value double; never allocate on CUDA.
    monkeypatch.setattr(torch.Tensor, 'to', fake_copy)
    moved = fresh.stage_chain_to_device(original, torch.device('cuda:0'))
    assert sorted(calls) == sorted(id(tensor) for tensor in chain_tensors(original))
    assert_same_payload(moved, original)
    assert moved.traces[0].source == 'planned' and moved.traces[0].complete is False
    for tensor, saved in before:
        assert torch.equal(tensor, saved)
    for old, new in zip(chain_tensors(original), chain_tensors(moved)):
        assert old is not new


def test_staged_source_covers_training_preparation_and_direct_development_reads(monkeypatch):
    batch = factual_chain()
    saved = [(tensor, tensor.clone()) for tensor in chain_tensors(batch)]
    seen = []
    host = SimpleNamespace(device=torch.device('cpu'), device_cache=None, compact_host_features=False,
        counts=dict(archive=6, fish=3),
        archive=SimpleNamespace(archive=SimpleNamespace(split=np.asarray([1, 1, 0, 1, 0, 1]))))
    def host_chain(rows, kind='archive', *, images=True):
        assert host.device == torch.device('cpu')
        seen.append((np.asarray(rows).copy(), kind, images))
        for trace in batch.traces:
            trace.validate(2)
        return batch if images else replace(batch, features=None)
    host.chain = host_chain
    transfers = []
    actual_transfer = fresh.stage_chain_to_device
    def observed_transfer(chain, device):
        assert device == torch.device('cuda:0')
        transfers.append(chain)
        return actual_transfer(chain, torch.device('cpu'))
    monkeypatch.setattr(fresh, 'stage_chain_to_device', observed_transfer)
    source = fresh.DeviceStagedSources(host, torch.device('cuda:0'))
    assert source.device == torch.device('cuda:0') and source.counts is host.counts
    train_rows = np.asarray([2, 4])
    prepared = fresh.prepare_request(source,
        dict(stage='actor', variant=1, groups=[('archive', train_rows)]))[0]
    expected = fresh.measurement_dropout(replace(batch, features=batch.features.float()), train_rows,
        protocol='burst-cold-v1', variant=1)
    assert_same_payload(prepared, expected)
    # The evaluator calls source.chain directly and reads original DEV rows.
    development_rows = np.asarray([0, 1])
    development = source.chain(development_rows, 'archive', images=True)
    assert_same_payload(development, batch)
    motion = source.chain(train_rows, 'archive', images=False)
    assert motion.features is None
    assert len(transfers) == 3 and [entry[2] for entry in seen] == [True, True, False]
    np.testing.assert_array_equal(seen[0][0], train_rows)
    np.testing.assert_array_equal(seen[1][0], development_rows)
    for tensor, saved_tensor in saved:
        assert torch.equal(tensor, saved_tensor)
