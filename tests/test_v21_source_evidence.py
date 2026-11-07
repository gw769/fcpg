"""Source evidence tests use injected arrays, never camera/robot IO."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_memory import VisualMemory, VisualObservation
from fcpg.v21_object_relative import encode_vector
from fcpg.v21_source_evidence import (
    SourceEvidenceStore, object_token_provenance, snapshot_source_evidence, save_source_evidence,
)
from test_v21_capture_integration import owner_case  # noqa: F401 -- injected camera/readback fixture


def exposure(stamp=10., marker=37, valid=(True, True), cue=None):
    rgb = np.full((2, 224, 224, 3), marker, np.uint8)
    rgb[1] += 1
    features = torch.full((1, 2, 256, 384), float(marker), dtype=torch.float16)
    q = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    times = torch.tensor([[stamp, stamp + .01]], dtype=torch.float64)
    mask = torch.tensor([valid])
    obs = VisualObservation(features, q, times, mask,
                            object_relative=None if cue is None else torch.tensor([cue], dtype=torch.float32))
    memory, _ = VisualMemory.empty(1, dtype=torch.float16).write(obs, torch.tensor([stamp + .1], dtype=torch.float64))
    state = SimpleNamespace(memory=memory, time=torch.tensor([stamp + .1], dtype=torch.float64),
                            q=q[:, 0].clone(), measurement_available=mask,
                            object_relative=torch.zeros(1, 8) if cue is None else obs.object_relative.clone())
    return rgb, obs, state


def provenance(obs, **kw):
    return object_token_provenance(
        obs.object_relative, wrist_measurement_available=bool(obs.measurement_available[0, 0]),
        image_time_s=float(obs.image_times[0, 0]), q_at_image=obs.q_at_image[0, 0], **kw)


def test_source_rgb_survives_endpoint_capture_and_cache_eviction(tmp_path):
    rgb, obs, state = exposure()
    original = rgb.copy()
    store = SourceEvidenceStore(maximum_frames_per_view=1)
    store.remember(rgb, obs, object_provenance=provenance(obs))
    bundle = snapshot_source_evidence(state, store)
    # Mutate caller-owned capture buffers and capture a completely different endpoint.
    rgb[:] = 255
    obs.features[:] = 999
    later_rgb, later_obs, _ = exposure(20., 87)
    store.remember(later_rgb, later_obs)
    np.testing.assert_array_equal(bundle.arrays["wrist_rgb"], original[0])
    assert bundle.arrays["wrist_features"][0, 0] == 37
    ref = save_source_evidence(tmp_path, "plan-1", bundle)
    saved = json.loads((tmp_path / ref["json"]).read_text())
    assert saved["executable"] is False
    assert saved["views"][0]["image_time_s"] == 10.
    with np.load(tmp_path / ref["npz"], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["wrist_rgb"], original[0])
        assert arrays["wrist_image_time_s"].dtype == np.float64
    assert ref["json_sha256"] == hashlib.sha256((tmp_path / ref["json"]).read_bytes()).hexdigest()
    assert ref["npz_sha256"] == hashlib.sha256((tmp_path / ref["npz"]).read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        save_source_evidence(tmp_path, "plan-1", bundle)


def test_missing_raw_source_and_absent_view_are_reported_without_replacement():
    _, _, state = exposure()
    later_rgb, later_obs, _ = exposure(20., 90)
    store = SourceEvidenceStore()
    store.remember(later_rgb, later_obs)
    bundle = snapshot_source_evidence(state, store)
    assert not bundle.arrays
    assert all(row["reason"] == "raw_source_not_retained" for row in bundle.manifest["views"])
    state.memory = replace(state.memory, valid=torch.tensor([[[False, False, True], [False, False, False]]]))
    assert snapshot_source_evidence(state).manifest["views"][1]["reason"] == "memory_view_unavailable"


@pytest.mark.parametrize("field,reason", [("q_at_image", "source_q_pairing_mismatch"),
                                         ("features", "source_feature_pairing_mismatch")])
def test_same_timestamp_with_different_model_q_or_features_cannot_claim_rgb(field, reason):
    rgb, obs, state = exposure()
    store = SourceEvidenceStore(); store.remember(rgb, obs)
    changed = getattr(state.memory, field).clone()
    changed[0, 0, -1] += 1
    state.memory = replace(state.memory, **{field: changed})
    bundle = snapshot_source_evidence(state, store)
    assert bundle.manifest["views"][0]["reason"] == reason
    assert "wrist_rgb" not in bundle.arrays
    assert bundle.manifest["views"][1]["raw_rgb_available"]


def test_duplicate_exposure_with_changed_pixels_is_rejected():
    rgb, obs, _ = exposure()
    store = SourceEvidenceStore(); store.remember(rgb, obs)
    store.remember(rgb.copy(), obs)
    rgb[0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="same source exposure"):
        store.remember(rgb, obs)


def test_real_capture_path_retains_exact_encoder_input_pair(owner_case):
    owner = owner_case.owner
    endpoint = owner.capture()
    memory, _ = VisualMemory.empty(1, dtype=endpoint.observation.features.dtype).write(
        endpoint.observation, endpoint.time)
    state = SimpleNamespace(memory=memory, time=endpoint.time, q=endpoint.q,
                            measurement_available=endpoint.observation.measurement_available,
                            object_relative=endpoint.observation.object_relative)
    bundle = snapshot_source_evidence(state, owner.source_evidence_store)
    for i, view in enumerate(("wrist", "column")):
        np.testing.assert_array_equal(bundle.arrays[view + "_rgb"], owner.last_rgb[i])
        np.testing.assert_array_equal(bundle.arrays[view + "_features"],
                                      endpoint.observation.features[0, i].numpy())
        np.testing.assert_array_equal(bundle.arrays[view + "_q_at_image"],
                                      endpoint.observation.q_at_image[0, i].numpy())
        assert bundle.arrays[view + "_image_time_s"] == endpoint.observation.image_times[0, i].item()
    assert len(owner_case.camera_calls) == 2  # evidence persistence performs no extra capture
    assert bundle.manifest["object_token"]["source"] is None  # synthetic fixture has no source record


def test_current_token_unavailable_never_inherits_registered_historical_source():
    cue = [1., .8, .1, .2, .3, .1, 1., 0.]
    rgb, obs, state = exposure(cue=cue)
    store = SourceEvidenceStore()
    store.remember(rgb, obs, object_provenance=provenance(
        obs, policy_record=dict(source="operator_colour_body", cog_px=[100., 100.])))
    old = snapshot_source_evidence(state, store).manifest["object_token"]
    assert old["available"] and old["source"] == "registered-fish-body-cog"
    state.measurement_available = torch.tensor([[False, True]])
    state.object_relative = torch.zeros(1, 8)
    current = snapshot_source_evidence(state, store).manifest["object_token"]
    assert not current["available"] and current["source"] is None
    assert current["reason"] == "current_wrist_observation_unavailable"


def test_duplicate_insertion_mask_does_not_erase_exact_paired_current_token():
    cue = [1., .8, .1, .2, .3, .1, 1., 0.]
    rgb, obs, state = exposure(cue=cue)
    store = SourceEvidenceStore()
    store.remember(rgb, obs, object_provenance=provenance(
        obs, policy_record=dict(source="operator_colour_body", cog_px=[100., 100.])))
    state.measurement_available = torch.tensor([[False, False]])
    assert snapshot_source_evidence(state, store).manifest["object_token"]["available"]


def test_token_missing_clipped_unverified_and_changed_source_are_truthful():
    absent = object_token_provenance(np.zeros(8), wrist_measurement_available=True,
                                     diagnostic_record=dict(source="dino", support_clipped=True))
    assert absent["source"] is None and absent["reason"] == "registered_body_target_unavailable"
    clipped = object_token_provenance(np.zeros(8), wrist_measurement_available=True,
                                      policy_record=dict(source="operator_colour_body", support_clipped=True))
    assert clipped["reason"] == "policy_support_clipped" and clipped["source"] is None
    cue = [1., .8, .1, .2, .3, .1, 1., 0.]
    rgb, obs, state = exposure(cue=cue)
    assert snapshot_source_evidence(state).manifest["object_token"]["source"] is None
    store = SourceEvidenceStore()
    store.remember(rgb, obs, object_provenance=provenance(
        obs, policy_record=dict(source="operator_colour_body")))
    state.object_relative[0, 2] += .1
    changed = snapshot_source_evidence(state, store).manifest["object_token"]
    assert changed["source"] is None and changed["reason"] == "state_token_differs_from_paired_wrist_capture"


def test_provenance_cannot_use_another_exposure_clock_or_token():
    rgb, obs, _ = exposure()
    store = SourceEvidenceStore()
    wrong = provenance(obs); wrong["image_time_s"] += 1
    with pytest.raises(ValueError, match="paired to the wrist"):
        store.remember(rgb, obs, object_provenance=wrong)
    wrong = provenance(obs); wrong["vector"][2] = .1
    with pytest.raises(ValueError, match="differs from"):
        store.remember(rgb, obs, object_provenance=wrong)


@pytest.mark.parametrize("part", ["array", "manifest"])
def test_frozen_bundle_mutation_is_detected_before_persistence(tmp_path, part):
    rgb, obs, state = exposure()
    store = SourceEvidenceStore(); store.remember(rgb, obs)
    bundle = snapshot_source_evidence(state, store)
    if part == "array":
        bundle.arrays["wrist_rgb"][0, 0, 0] += 1
    else:
        bundle.manifest["source_time_s"] += 1
    with pytest.raises(ValueError, match="changed"):
        save_source_evidence(tmp_path, "bad", bundle)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field,value", [("axis_px", [float("nan"), 1.]),
                                         ("axis_px", [1.]), ("axis_px", [1., 0., 1.]),
                                         ("width_px", float("nan")), ("length_px", float("inf")),
                                         ("score", float("nan"))])
def test_nonfinite_or_malformed_geometry_cannot_publish_available_token(field, value):
    record = dict(cog_px=[110., 120.], axis_px=[1., 0.], length_px=40., width_px=20., score=.8)
    record[field] = value
    vector = encode_vector(record, [[90., 150.], [130., 150.]])
    assert vector.shape == (8,) and not vector.any() and np.isfinite(vector).all()
