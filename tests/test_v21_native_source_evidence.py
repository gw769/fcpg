"""Native diagnostics freeze existing exact view sources without new IO."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.prod_v20_policy import empty_history
from fcpg.v21_core import ActualState
from fcpg.v21_j3_native_service import J3NativeResponseDiagnostic
from fcpg.v21_memory import VisualMemory, VisualObservation
from fcpg.v21_source_evidence import SourceEvidenceStore, object_token_provenance


def source_fixture(tmp_path):
    q = torch.tensor([[0., -90., 95., 60., -50., 30.]])
    now = torch.tensor([10.2], dtype=torch.float64)
    token = torch.tensor([[1., .3, .4, .1, .2, 0., .2, .1]])
    observation = VisualObservation(torch.ones(1, 2, 256, 384, dtype=torch.float16),
        q[:, None].expand(1, 2, 6).clone(), torch.tensor([[10., 10.1]], dtype=torch.float64),
        torch.ones(1, 2, dtype=torch.bool), object_relative=token)
    memory, _ = VisualMemory.empty(1).write(observation, now, readback_index=torch.zeros(1, dtype=torch.long))
    mask = torch.zeros(1, 2, dtype=torch.bool)
    state = ActualState(memory=memory, q=q, time=now, history=empty_history(1),
        history_ends=torch.zeros(1, 3, dtype=torch.float64), grounding=torch.zeros(1, 2, 8, 192),
        visual_error=torch.zeros(1, 2, 8, 192), measurement_available=observation.measurement_available,
        measurement_pending=observation.measurement_available.clone(), visual_feedback_available=mask,
        visual_feedback_time=torch.zeros(1, 2, dtype=torch.float64), q_error=torch.zeros(1, 5, 6),
        feedback_available=mask[:, 0].clone(), feedback_time=torch.zeros(1, dtype=torch.float64),
        feedback_offsets=torch.zeros(1, 5, dtype=torch.float64), feedback_retimed=mask[:, 0].clone(),
        object_relative=token, feedback_readback_index=torch.zeros(1, dtype=torch.long),
        visual_feedback_readback_indices=torch.zeros(1, 2, dtype=torch.long))
    rgb = np.stack((np.full((224, 224, 3), 30, np.uint8), np.full((224, 224, 3), 70, np.uint8)))
    store = SourceEvidenceStore()
    provenance = object_token_provenance(token, wrist_measurement_available=True,
        policy_record={'source': 'operator_colour_body', 'cog_px': [67., 89.]},
        image_time_s=10., q_at_image=q[0])
    store.remember(rgb, observation, object_provenance=provenance)
    def forbidden(*args, **kwargs):
        pytest.fail('saving a native source may not capture, infer, or read hardware')
    runtime = SimpleNamespace(state=state, text_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))
    owner = SimpleNamespace(source_evidence_store=store, capture=forbidden, _read_q=forbidden)
    service = SimpleNamespace(owner=owner, record_policy_inputs=True, root=tmp_path / 'model-output',
                              clock=lambda: 20., temporal_contract=None, lewm_controller=None)
    diagnostic = J3NativeResponseDiagnostic(service, output=tmp_path)
    return diagnostic, runtime, rgb


def saved_source(diagnostic, runtime):
    reference = diagnostic._save_source_state(runtime)
    payload = torch.load(reference['path'], map_location='cpu', weights_only=False)
    visual = payload['source_visual_evidence']
    root = diagnostic.path.parent
    manifest = json.loads((root / visual['json']).read_text())
    with np.load(root / visual['npz'], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return reference, payload, manifest, arrays


def test_native_source_binds_exact_two_views_and_original_wrist_token(tmp_path):
    diagnostic, runtime, rgb = source_fixture(tmp_path)
    reference, payload, manifest, arrays = saved_source(diagnostic, runtime)
    assert reference['source_visual_evidence'] == payload['source_visual_evidence']
    assert payload['kind'] == 'native-response-source' and payload['model_transition'] is False
    assert manifest['view_order'] == ['wrist', 'column']
    for index, view in enumerate(('wrist', 'column')):
        np.testing.assert_array_equal(arrays[view + '_rgb'], rgb[index])
        np.testing.assert_array_equal(arrays[view + '_features'], runtime.state.memory.features[0, index, -1])
        assert arrays[view + '_image_time_s'] == runtime.state.memory.image_times[0, index, -1]
        assert manifest['views'][index]['raw_rgb_available']
    assert payload['object_target_source'] == 'registered-fish-body-cog'
    assert payload['object_token_provenance'] == manifest['object_token']
    assert payload['object_token_provenance']['image_time_s'] == 10.
    assert torch.equal(payload['state'].time, runtime.state.time)
    assert payload['state'].q.data_ptr() != runtime.state.q.data_ptr()
    assert payload['source_visual_evidence_missing_reason'] is None


def test_absent_capture_store_saves_truthful_missing_views_without_fallback(tmp_path):
    diagnostic, runtime, _ = source_fixture(tmp_path)
    del diagnostic.service.owner.source_evidence_store
    _, payload, manifest, arrays = saved_source(diagnostic, runtime)
    assert not arrays
    assert all(row['memory_available'] and not row['raw_rgb_available']
               and row['reason'] == 'raw_source_not_retained' for row in manifest['views'])
    assert payload['source_visual_evidence']['complete_raw_views'] == []
    assert payload['object_target_source'] is None
    assert payload['object_token_provenance']['reason'] == 'available_token_source_evidence_missing'
    assert torch.equal(payload['state'].memory.features, runtime.state.memory.features)


def test_same_time_wrong_column_features_cannot_borrow_source_pixels(tmp_path):
    diagnostic, runtime, _ = source_fixture(tmp_path)
    features = runtime.state.memory.features.clone()
    features[:, 1, -1] += 1
    runtime.state = replace(runtime.state, memory=replace(runtime.state.memory, features=features))
    _, payload, manifest, arrays = saved_source(diagnostic, runtime)
    assert manifest['views'][0]['raw_rgb_available']
    assert manifest['views'][1]['reason'] == 'source_feature_pairing_mismatch'
    assert not manifest['views'][1]['raw_rgb_available'] and 'column_rgb' not in arrays
    assert payload['source_visual_evidence']['complete_raw_views'] == ['wrist']


def test_existing_native_snapshot_is_not_replaced_or_given_new_sidecars(tmp_path):
    diagnostic, runtime, _ = source_fixture(tmp_path)
    path = tmp_path / 'native-response-source-3.pt'
    path.write_bytes(b'existing snapshot')
    with pytest.raises(FileExistsError):
        diagnostic._save_source_state(runtime)
    assert path.read_bytes() == b'existing snapshot'
    assert list(tmp_path.iterdir()) == [path]
