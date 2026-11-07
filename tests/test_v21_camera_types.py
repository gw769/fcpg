"""Opt-in camera metadata must not change pixels, labels or legacy behavior."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_camera_types import (ReviewedCameraTypes, SCHEMA, REVIEWED_TYPES,
    LEGACY_SLOTS, camera_mode_from_checkpoint)
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_horizon_training import factual_state
from fcpg.v21_memory import VisualMemory, VisualObservation
from fcpg.v21_native_horizon import core_from_parent
from fcpg.v21_scene_lewm import V21SceneLeWM
from fcpg.v21_training import synthetic_chain
from scripts.train_v21_from_scratch import stage_chain_to_device


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)


def registry():
    paths = dict(fish_wrist_root='/fish/wrist', fish_column_root='/fish/column')
    archive = SimpleNamespace(endpoint_episode_index=np.array([0, 0, 1, 1, 2, 2]),
        episodes=[dict(manifest_row=dict(dataset_root='/recorded', episode_index=100+i,
                  lewm_v12_slot_views=['base', 'left'])) for i in range(3)])
    contract = dict(schema=SCHEMA, purpose='research-only', production_enabled=False,
        unknown_policy='explicit-zero-type-retain-all-data', slot_policy='original-order-no-permutation',
        source_wide_mapping_verified=False,
        episodes=[dict(canonical_episode=i, source_episode=100+i, dataset_root='/recorded',
            historical_views=['left', 'base'], physical_roles=roles,
            reviewed_frames=[1, 5], evidence='original paired images')
            for i, roles in enumerate((['external', 'wrist'], ['external', 'external']))],
        fish=dict(**paths, physical_roles=['wrist', 'external'], allowed_episodes=[0, 1, 3],
                  evidence='paired original capture'))
    return archive, paths, contract


def test_types_keep_raw_slot_identity_and_unknown_is_not_a_missing_camera():
    archive, paths, contract = registry()
    roles = ReviewedCameraTypes(contract, archive, paths)
    np.testing.assert_array_equal(roles.for_endpoints([0, 2, 4]), [[2, 1], [2, 2], [0, 0]])
    np.testing.assert_array_equal(roles.for_chain(np.array([[0, 1], [2, 3]]), 'archive', None),
                                  [[2, 1], [2, 2]])
    assert roles.coverage([0, 2, 4]) == dict(windows=3, reviewed_windows=2, episodes=3,
        unknown_slots=2, wrist_slots=1, external_slots=3, two_external_windows=1)
    with pytest.raises(ValueError, match='crosses'):
        roles.for_chain(np.array([[0, 2]]), 'archive', None)


@pytest.mark.parametrize('field,value', [('dataset_root', '/wrong'), ('source_episode', 42),
    ('historical_views', ['base', 'left']), ('physical_roles', ['guess', 'external']), ('evidence', '')])
def test_review_is_bound_to_exact_episode_and_order(field, value):
    archive, paths, contract = registry()
    contract['episodes'][0][field] = value
    with pytest.raises(ValueError, match='identity'):
        ReviewedCameraTypes(contract, archive, paths)


def test_no_source_wide_or_wrong_fish_source_promotion():
    archive, paths, contract = registry()
    with pytest.raises(ValueError, match='episode-scoped'):
        ReviewedCameraTypes(dict(contract, source_wide_mapping_verified=True), archive, paths)
    with pytest.raises(ValueError, match='fish camera'):
        ReviewedCameraTypes(contract, archive, dict(paths, fish_wrist_root='/other'))


def model_and_chain():
    vocabulary = TaskVocabulary.build(['pick red block'])
    chain = synthetic_chain(vocabulary, batch=2)
    core = V21ResearchCore(V21Config(len(vocabulary.words), prediction_horizon=40)).eval()
    core.lewm = V21SceneLeWM(normalization='layernorm').eval()
    return core, chain


def test_zero_init_keeps_history_feedback_actor_scene_and_forecast_bit_exact():
    core, chain = model_and_chain()
    original = factual_state(core, chain, 2, images=True)
    with torch.no_grad():
        baseline = core.predict(original, chain.traces[2], chain.readback_offsets[:, 2, 1:])
        actor_before = core.act(original, chain.text_ids, chain.readback_offsets[:, 2, :-1])
    state_before = {k: v.clone() for k, v in core.state_dict().items()}
    rng = torch.get_rng_state().clone()
    core.lewm.enable_camera_conditioning(REVIEWED_TYPES)
    assert torch.equal(rng, torch.get_rng_state())
    typed = replace(chain, camera_types=torch.tensor([[2, 2], [0, 0]]))
    assert torch.equal(typed.slice(1, 2).camera_types, typed.camera_types[1:2])
    assert torch.equal(stage_chain_to_device(typed, 'cpu').camera_types, typed.camera_types)
    seen = []
    handle = core.lewm.register_forward_pre_hook(lambda m, a, kw:
        seen.append(kw['memory'].camera_types.clone()), with_kwargs=True)
    current = factual_state(core, typed, 2, images=True)
    with torch.no_grad():
        after = core.predict(current, chain.traces[2], chain.readback_offsets[:, 2, 1:])
        actor_after = core.act(current, chain.text_ids, chain.readback_offsets[:, 2, :-1])
        # Exercise actual-time re-evaluation, not only the primary forecast.
        core.complete(current, replace(after, query_offsets=after.query_offsets+.001), chain.traces[2],
                      chain.paths[:, 2], current.time[:, None]+chain.readback_offsets[:, 2])
    handle.remove()
    assert len(seen) == 4 and all(torch.equal(x, typed.camera_types) for x in seen)
    assert torch.equal(current.q_error, original.q_error)
    assert torch.equal(current.grounding, original.grounding)
    assert torch.equal(actor_before.commands_position, actor_after.commands_position)
    for field in ('predicted_q_path', 'predicted_scene', 'observation_scene'):
        assert torch.equal(getattr(baseline.response, field), getattr(after.response, field))
    assert all(torch.equal(v, core.state_dict()[k]) for k, v in state_before.items())


def test_cold_memory_keeps_explicit_types_without_inventing_valid_images():
    core, chain = model_and_chain()
    core.lewm.enable_camera_conditioning(REVIEWED_TYPES)
    typed = replace(chain, camera_types=torch.tensor([[2, 2], [0, 0]]),
                    view_valid=torch.zeros_like(chain.view_valid))
    state = factual_state(core, typed, 2, images=True)
    assert not state.memory.valid.any()
    assert not state.measurement_available.any()
    assert not state.memory.features.any()
    assert torch.equal(state.memory.camera_types, typed.camera_types)
    with torch.no_grad():
        assert torch.isfinite(core.predict(state, chain.traces[2], chain.readback_offsets[:, 2, 1:])
                              .response.predicted_scene).all()


def test_metadata_cannot_disappear_change_or_relabel_existing_memory():
    features = torch.randn(2, 2, 256, 384)
    now = torch.ones(2, dtype=torch.float64)
    observation = VisualObservation(features, torch.zeros(2, 2, 6), now[:, None].expand(-1, 2),
                                    torch.ones(2, 2, dtype=torch.bool))
    typed = replace(observation, view_order=('slot0', 'slot1'), camera_types=torch.tensor([[2, 2], [1, 2]]))
    legacy, _ = VisualMemory.empty(2).write(observation, now)
    with pytest.raises(ValueError, match='relabel'):
        legacy.write(typed, now)
    memory, _ = VisualMemory.empty(2).write(typed, now)
    with pytest.raises(ValueError, match='stream changed'):
        memory.write(observation, now)
    with pytest.raises(ValueError, match='cannot change'):
        memory.write(replace(typed, camera_types=torch.zeros(2, 2, dtype=torch.long)), now)


def test_type_branch_gets_gradients_unknown_has_no_parameter_scene_target_stays_image_only():
    core, chain = model_and_chain()
    core.lewm.enable_camera_conditioning(REVIEWED_TYPES)
    core.lewm.train()
    typed = replace(chain, camera_types=torch.tensor([[1, 2], [2, 2]]))
    # Dropout does not enter the frozen factual context construction.
    core.lewm.eval()
    state = factual_state(core, typed, 2, images=True)
    response = core.predict(state, chain.traces[2], chain.readback_offsets[:, 2, 1:]).response
    future = torch.randn(2, 2, 2, 256, 384, requires_grad=True)
    valid = torch.ones(2, 2, 2, dtype=torch.bool)
    target = core.lewm.scene_encoder(future, valid)
    (response.predicted_scene[:, :2]-target).square().mean().backward()
    gradient = core.lewm.camera_type_embedding.grad
    assert gradient.shape == (2, 384) and (gradient.abs().sum(-1) > 0).all()
    assert future.grad is not None and future.grad.abs().sum() > 0
    with torch.no_grad():
        core.lewm.camera_type_embedding.add_(torch.randn_like(gradient))
        assert torch.equal(target.detach(), core.lewm.scene_encoder(future, valid))
    core.zero_grad(set_to_none=True)
    state = replace(state, memory=replace(state.memory, camera_types=torch.zeros(2, 2, dtype=torch.long)))
    core.predict(state, chain.traces[2], chain.readback_offsets[:, 2, 1:]).response.predicted_scene.square().mean().backward()
    assert torch.count_nonzero(core.lewm.camera_type_embedding.grad) == 0


def test_typed_checkpoint_requires_metadata_and_explicit_runtime_types():
    core, chain = model_and_chain()
    legacy = dict(config=asdict(core.config), model=core.state_dict(),
                  scene_lewm_architecture=core.lewm.architecture, scene_normalization='layernorm')
    assert camera_mode_from_checkpoint(legacy) == LEGACY_SLOTS
    core_from_parent(legacy)
    core.lewm.enable_camera_conditioning(REVIEWED_TYPES)
    payload = dict(legacy, model=core.state_dict(), lewm_camera_conditioning=REVIEWED_TYPES)
    restored = core_from_parent(payload)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in core.state_dict().items())
    with pytest.raises(ValueError, match='metadata'):
        core_from_parent(dict(payload, lewm_camera_conditioning=LEGACY_SLOTS))
    with pytest.raises(ValueError, match='metadata'):
        core_from_parent(dict(legacy, lewm_camera_conditioning=REVIEWED_TYPES))
    with pytest.raises(ValueError, match='camera types'):
        factual_state(restored, chain, 1, images=True)
    # Original motion-only response retention is still a legal separate path.
    state = factual_state(restored, chain, 0, images=False)
    with torch.no_grad():
        restored.lewm(state.q, chain.traces[0], state.history, chain.readback_offsets[:, 0, 1:])


def test_actual_resume_keeps_old_serialization_and_roundtrips_optional_types():
    from fcpg.v21_actual_resume import pack, unpack_state
    core, chain = model_and_chain()
    state = factual_state(core, chain, 0, images=True)
    packed = pack(state)
    assert 'camera_types' not in packed['memory']  # old consumers see the old schema
    assert unpack_state(packed, 'cpu').memory.camera_types is None
    typed = replace(state, memory=replace(state.memory, camera_types=torch.tensor([[2, 2], [1, 2]])))
    packed = pack(typed)
    assert torch.equal(unpack_state(packed, 'cpu').memory.camera_types, typed.memory.camera_types)
    packed['memory']['camera_types'] = torch.tensor([[3, 2], [1, 2]])
    with pytest.raises(ValueError, match='camera types'):
        unpack_state(packed, 'cpu')
