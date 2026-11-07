from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_live_service import V21LiveService
from fcpg.v21_source_evidence import SourceEvidenceStore
from test_v21_live_service import service  # noqa: F401


def enable_completed_input_recording(service):
    service.record_policy_inputs = True
    original = service.owner.execute
    def execute():
        offsets = service.owner.runtime.pending.trace.offsets.cpu().tolist()
        result = original()
        service.owner.bridge.last_actor_input = service.owner.runtime.last_actor_input
        result['protective_hold_after_q5'] = True
        result['commanded_offsets_s'] = offsets
        return result
    service.owner.bridge.execute = execute


def test_source_pixels_are_bound_before_execution_and_not_replaced_by_endpoint(service):
    enable_completed_input_recording(service)
    service.replan_on_execute = False
    store = service.owner.source_evidence_store = SourceEvidenceStore()
    capture = service.owner.capture
    colour = [17]
    def remembered_capture():
        endpoint = capture()
        rgb = np.full((2, 224, 224, 3), colour[0], np.uint8)
        store.remember(rgb, endpoint.observation)
        return endpoint
    service.owner.capture = remembered_capture
    plan = service.handle('plan')['plan']
    evidence = plan['source_visual_evidence']
    assert evidence['complete_raw_views'] == ['wrist', 'column']
    original_pixels = np.load(service.output/evidence['npz'])['wrist_rgb'].copy()
    colour[0] = 91
    result = service.handle('execute')['last_execution']
    assert result['source_visual_evidence'] == evidence
    saved = np.load(service.output/evidence['npz'])
    assert np.array_equal(saved['wrist_rgb'], original_pixels) and (saved['wrist_rgb'] == 17).all()
    payload = torch.load(service.output/result['policy_input_snapshot'], weights_only=False)
    assert payload['source_visual_evidence'] == evidence
    assert payload['object_target_source'] is None
    assert payload['object_token_provenance']['available'] is False
    assert service.owner.writes == 5


def test_service_persists_successful_cem_rng_trace_without_model_or_motor_replay(service, monkeypatch):
    from fcpg import v21_lewm_cem as cem
    service.record_policy_inputs = True
    service.lewm_cem_objective = 'actor_response_tracking'
    service.lewm_cem_candidate_parameterization = 'scaled_actor_sequence'
    state = service.owner.runtime.state
    calls = []
    def planner(model, state, proposal, dt, gate, *, trace, **kwargs):
        calls.append(kwargs)
        trace.begin(proposal, q=state.q, actor_commands=proposal)
        selected = proposal.clone()
        trace.record('selected', commands=selected)
        trace.finish(proposal)
        return selected, dict(action_source='v21_lewm_cem')
    monkeypatch.setattr(cem, 'plan_commands', planner)
    before = torch.get_rng_state().clone()
    service._lewm_cem_builder(state, torch.arange(5)[None].double()*.1)
    reference = service.lewm_cem_trace_record
    raw = (service.output/reference['path']).read_bytes()
    assert reference['saved'] and reference['complete'] and reference['sha256'] == sha256(raw).hexdigest()
    payload = torch.load(service.output/reference['path'], weights_only=False)
    assert payload['source_time_s'] == float(state.time[0])
    assert payload['source_q_position'] == state.q[0].tolist()
    assert torch.equal(payload['rng_before']['cpu_global_state'], before)
    assert torch.equal(torch.get_rng_state(), before)
    assert len(calls) == 1 and service.owner.writes == 0


def test_failed_cem_keeps_original_failure_and_incomplete_trace(service, monkeypatch):
    from fcpg import v21_lewm_cem as cem
    service.record_policy_inputs = True
    service.lewm_cem_objective = 'actor_response_tracking'
    def fail(model, state, proposal, dt, gate, *, trace, **kwargs):
        trace.begin(proposal, q=state.q)
        trace.finish(proposal, error='no admissible fixture')
        raise ValueError('no admissible fixture')
    monkeypatch.setattr(cem, 'plan_commands', fail)
    with pytest.raises(ValueError, match='no admissible fixture'):
        service._lewm_cem_builder(service.owner.runtime.state, torch.arange(5)[None].double()*.1)
    reference = service.lewm_cem_trace_record
    assert reference['saved'] and not reference['complete']
    assert service.owner.writes == 0


def test_unknown_temporal_profile_rejects_before_owner_construction(tmp_path):
    calls = []
    with pytest.raises(ValueError, match='temporal.*profile'):
        V21LiveService(tmp_path/'model', tmp_path/'out', temporal_profile='unknown',
                       owner_factory=lambda *a, **kw: calls.append(kw))
    assert not calls


def test_plan_evidence_write_failure_cancels_armed_plan_without_motor_write(service, monkeypatch):
    from fcpg import v21_source_evidence
    service.record_policy_inputs = True
    def fail(*args, **kwargs):
        raise OSError('source evidence disk unavailable')
    monkeypatch.setattr(v21_source_evidence, 'save_source_evidence', fail)
    with monkeypatch.context() as guard:
        guard.setattr(service.owner, '_halt', lambda: pytest.fail('failed planning must not dispatch a hardware HOLD'))
        with pytest.raises(OSError, match='source evidence disk unavailable'):
            service.handle('plan')
    assert service.owner.runtime.stopped and service.owner.runtime.pending is None
    assert service.owner.stop_event.is_set() and service.owner.writes == 0
    assert not service.status()['ready']


def test_fresh_core_reinstalls_the_declared_progress_profile(service, monkeypatch):
    from fcpg import v21_temporal_contract
    identity = dict(schema='test-profile-identity', profile='actual-readback-progress-compatibility-v1')
    service.temporal_contract = identity
    calls = []
    def install(core, path, *, profile):
        calls.append((core, path, profile))
        core.temporal_contract_identity = identity.copy()
        return identity.copy()
    monkeypatch.setattr(v21_temporal_contract, 'install_temporal_contract', install)
    cloned = service._fresh_eval_core()
    assert cloned is not service.owner.runtime.core
    assert cloned.temporal_contract_identity == identity
    assert calls == [(cloned, service.root/'fish/final.pt', identity['profile'])]
    assert service.owner.writes == 0


def test_non_cem_plan_does_not_inherit_an_earlier_preview_trace(service):
    service.lewm_cem_trace_record = dict(saved=True, complete=True, path='unrelated-preview.pt')
    result = service.handle('plan')
    assert result['plan']['action_source'] != 'v21_lewm_cem'
    assert 'cem_trace' not in result['plan']
    assert service.owner.writes == 0
