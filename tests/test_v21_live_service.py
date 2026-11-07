"""Resident dashboard contracts with fake IO; never open cameras or serial ports."""
from copy import deepcopy
from dataclasses import replace
import json
import shutil
import threading
from types import SimpleNamespace
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import numpy as np
import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_memory import VisualObservation
from fcpg.v21_runtime import V21ResearchRuntime, EncodedEndpoint
from fcpg.v21_live_service import V21LiveService
from scripts.serve_so101_v21_research import make_server, HTML


class GainBus:
    """Follower registers the LeWM CEM execute path uses: it raises the elbow and
    wrist P gains for one chunk and writes 16 back afterwards."""
    def __init__(self):
        self.values={('P_Coefficient',m):16 for m in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')}
        self.writes=[]
    def read(self,register,motor,*,normalize=True,num_retry=0):return self.values[(register,motor)]
    def write(self,register,motor,value,*,normalize=True,num_retry=0):
        self.writes.append((register,motor,int(value)));self.values[(register,motor)]=int(value)


@pytest.fixture
def service(tmp_path, monkeypatch):
    from fcpg import v21_j3_pid_service
    monkeypatch.setattr(v21_j3_pid_service, 'DEFAULT_SEAL_PATH', tmp_path/'j3-seal.json')
    torch.set_num_threads(2); torch.manual_seed(21)
    now = [10.]
    monkeypatch.setattr(V21LiveService,'_monitor_q',lambda self:None)
    class Owner:
        def __init__(self, root, output, task, device, open_hardware, max_plan_age_s=300.):
            self.stop_event=threading.Event(); self.q=np.zeros(6,np.float32);self.writes=0;self.holds=0;self.closed=False
            self.advance_clock=lambda seconds: now.__setitem__(0, now[0]+float(seconds))
            vocab=TaskVocabulary.build([task]); core=V21ResearchCore(V21Config(len(vocab.words)))
            self.runtime=V21ResearchRuntime(core,vocab.encode([task]),physical_check=lambda q,u:self._physical_check(q,u),
                lease_check=lambda:None,clock_fn=lambda:now[0],max_plan_age_s=max_plan_age_s)
            self.frames={v:np.zeros((480,640,3),np.uint8) for v in ('wrist','column')}
            self.bus_lock=threading.RLock()
            self.io=SimpleNamespace(camera_lock=threading.Lock(),_last_raw_frames=self.frames,
                                     settle_step_tolerance=.25,settle_position_tolerance=3.,
                                     _read_camera=lambda view:np.zeros((224,224,3),np.uint8),
                                     follower=SimpleNamespace(bus=GainBus()))
            self.bridge=SimpleNamespace(plan=self.plan,execute=self.execute,stop=self.stop,last_attempt=None)
        def _read_q(self):return self.q.copy()
        def read_checked(self):return self._read_q()
        def _physical_check(self,q,commands):pass
        def _halt(self):self.holds+=1
        def capture(self):
            now[0]+=.01; self.last_rgb=np.zeros((2,224,224,3),np.uint8)
            q=torch.from_numpy(self.q.copy())[None]; t=torch.tensor([now[0]],dtype=torch.float64)
            obs=VisualObservation(torch.ones(1,2,256,384),q[:,None].expand(-1,2,-1),t[:,None].expand(-1,2),torch.ones(1,2,dtype=torch.bool))
            self.last_endpoint=EncodedEndpoint(q,t,obs);return self.last_endpoint
        def plan(self,**kwargs):
            self.runtime.stop();return self.runtime.plan(endpoint=self.capture(),**kwargs)
        def execute(self):
            pending=self.runtime.pending;epoch=pending.source_time.item()
            action_source=self.runtime.pending_action_source
            q=pending.source_q;u=pending.trace.positions.detach();path=torch.cat((q[:,None],u),1)
            times=torch.tensor([[epoch+i*.1 for i in range(6)]],dtype=torch.float64)
            self.runtime.begin_execution(q,pending.source_time)
            pending=self.runtime.pending;now[0]=epoch+.5;self.q=u[0,-1].numpy();self.writes+=5
            self.runtime.complete(replace(pending.trace,source='recorded'),path,times,self.capture())
            self.runtime.stop()
            result=dict(completed=True,source_time_s=epoch,source_q_position=q[0].tolist(),commands_position=u[0].tolist(),
                endpoint_time_s=float(self.runtime.state.time[0]),
                action_source=action_source,
                policy_commands_attempted=5,policy_commands_returned=5,
                measured_q_path=path[0].tolist(),readback_times_s=times[0].tolist(),
                command_times_s=[epoch+i*.1 for i in range(5)])
            self.bridge.last_attempt=deepcopy(result);return result
        def stop(self):self.runtime.stop();self._halt()
        def images(self,name):pass
        def close(self):self.closed=True
    value=V21LiveService(tmp_path/'model',tmp_path/'out',device='cpu',owner_factory=Owner,
                        target_diagnostics=False,clock=lambda:now[0])
    yield value
    value.close()


def test_startup_has_images_and_stop_without_policy_actions(service):
    d=service.status()
    assert d['model']['version']=='21.0.2' and not d['model']['deployment_revision_only']
    assert d['model']['planning_observations']=='candidate-until-execution'
    assert d['snapshot']['sequence']==1 and d['runtime']['stopped'] and not d['ready']
    assert service.owner.writes==0 and not d['model']['pixel_keypoint_head']
    assert d['camera_condition']=='both_fresh' and d['camera_address']=='factual' and d['camera_h0_committed']


def test_startup_only_retries_discarded_50ms_bracket_samples(service, tmp_path, monkeypatch):
    owner_type=type(service.owner); original=owner_type.capture; calls=[]
    def delayed(self):
        calls.append(len(calls)+1)
        if len(calls)<3:
            raise RuntimeError('camera/readback acquisition bracket exceeds 50ms')
        return original(self)
    monkeypatch.setattr(owner_type,'capture',delayed)
    configured=V21LiveService(tmp_path/'model',tmp_path/'startup-retry',device='cpu',
        owner_factory=owner_type,target_diagnostics=False,clock=service.clock)
    try:
        assert calls==[1,2,3]
        assert configured.startup_observation['attempts']==3
        assert len(configured.startup_observation['bracket_timeouts'])==2
        assert configured.snapshot['sequence']==1 and configured.owner.writes==0
    finally:
        configured.close()


def test_startup_does_not_retry_unclassified_capture_error(service, tmp_path, monkeypatch):
    owner_type=type(service.owner);calls=[]
    def broken(self):
        calls.append(1)
        raise RuntimeError('camera identity changed')
    monkeypatch.setattr(owner_type,'capture',broken)
    with pytest.raises(RuntimeError,match='camera identity changed'):
        V21LiveService(tmp_path/'model',tmp_path/'startup-fail',device='cpu',
            owner_factory=owner_type,target_diagnostics=False,clock=service.clock)
    assert calls==[1]


def test_explicit_live_plan_age_is_exposed_without_sending_motion(service, tmp_path):
    configured = V21LiveService(tmp_path/'model', tmp_path/'age', device='cpu',
        owner_factory=type(service.owner), target_diagnostics=False, clock=service.clock,
        max_plan_age_s=1.)
    try:
        assert configured.owner.runtime.max_plan_age_s == 1.
        assert configured.status()['model']['max_plan_age_s'] == 1.
        assert configured.owner.writes == 0
    finally:
        configured.close()


def test_lewm_preview_retains_the_cem_and_controller_receipts(service):
    service.lewm_cem = True
    service.lewm_cem_record = {'physical_gate_cache': {'success_hits': 7}}
    service.lewm_controller = {'candidate_sha256': 'b' * 64}
    def builder(state, offsets):
        del offsets
        return state.q[:, None].expand(-1, 5, -1).clone()
    builder.wants_command_offsets = True
    service._lewm_cem_builder = builder
    result = service.handle('preview')
    assert result['preview']['lewm_cem'] == {'physical_gate_cache': {'success_hits': 7}}
    assert result['preview']['lewm_controller'] == {'candidate_sha256': 'b' * 64}
    assert service.owner.writes == 0


def test_stale_interactive_plan_is_recaptured_before_execute(service, monkeypatch):
    service.owner.runtime.max_plan_age_s = 1.
    calls = []
    original = service._plan_with_policy
    def observed(*args, **kwargs):
        calls.append(service.clock())
        return original(*args, **kwargs)
    monkeypatch.setattr(service, '_plan_with_policy', observed)
    service.handle('plan')
    service.owner.advance_clock(2.)
    result = service.handle('execute')
    assert len(calls) == 2
    assert result['last_execution']['completed'] is True
    assert service.owner.writes == 5


def test_empty_gripper_diagnostic_is_disabled_by_default(service):
    assert service.status()['model']['gripper_follow_check_enabled'] is False


def test_explicit_raw_actor_plan_has_no_geometry_rewrite(service):
    service.raw_actor = True
    assert not service._uses_vertical_pinch_geometry()
    service.handle('plan')
    assert service.owner.runtime.pending_action_source == 'v21_actor'
    assert service.execution_geometry_record is None
    assert service.owner.writes == 0


def test_candidate_preview_missing_visual_goal_never_commits_or_moves(service):
    previous = service.owner.runtime.state
    result = service.handle('candidate-preview')
    assert result['candidate_preview']['available'] is False
    assert result['candidate_preview']['selected'] is None
    assert service.owner.runtime.state is previous
    assert service.owner.runtime.pending is None and service.owner.runtime.stopped
    assert service.owner.writes == 0
    with pytest.raises(RuntimeError,match='not enabled'):
        service.handle('gripper-follow-check')
    assert service.owner.writes==0 and service.owner.runtime.stopped


def test_follow_record_keeps_nominal_prediction_separate(service):
    nominal=[[0.]*6]+[[1.]*6 for _ in range(5)]
    prediction=[[0.]*6 for _ in range(6)]
    feedback=deepcopy(nominal);feedback[-1]=[2.]*6
    result=dict(measured_q_path=nominal,feedback_q_path=feedback,
                retimed_predicted_q_path=deepcopy(prediction),q_path_residual=[[1.]*6 for _ in range(5)])
    service._save_execution(result)
    assert result['retimed_predicted_q_path']==prediction
    assert result['measured_q_path']==nominal
    assert result['q_path_residual']==[[1.]*6 for _ in range(5)]
    assert result['feedback_predicted_q_path'][-1]==[2.]*6
    assert service.owner.writes==0


def test_block_io_keeps_dispatch_object_relative_after_post_hold_recapture(service):
    service.handle('plan')
    runtime = service.owner.runtime
    source = replace(runtime._pending_state, object_relative=torch.tensor(
        [[1., .4581687, -.6716852, -.38994795, .0415577, .0378181, .71531355, .69880366]],
        dtype=torch.float32))
    runtime._pending_state = source
    service.owner.last_object_token_provenance = {
        'source': 'registered-fish-body-cog', 'available': True,
        'vector': source.object_relative[0].tolist(),
    }
    service._record_plan(runtime.pending.trace.positions)
    expected_object_relative = source.object_relative[0].tolist()
    service.owner.bridge.last_actor_input = source
    source.object_relative.zero_()  # defensive check: source object was refreshed in place
    service.owner.last_object_relative = [0.] * 8  # post-HOLD display refresh
    result = dict(
        completed=True, action_source='v21_actor',
        source_time_s=float(source.time[0]),
        source_q_position=source.q[0].detach().cpu().tolist(),
        commands_position=runtime.pending.trace.positions[0].detach().cpu().tolist(),
        measured_q_path=[[0.] * 6 for _ in range(6)],
    )
    service._save_execution(result)
    block = result['block_io']
    assert block['object_relative'] == pytest.approx(expected_object_relative)
    assert block['object_relative_source'] == 'dispatch_actor_input'
    assert block['object_relative_snapshot_valid'] is True
    assert block['object_relative_finite'] is True
    assert block['object_relative_available'] is True
    assert block['object_relative_source_time_s'] == pytest.approx(float(source.time[0]))
    assert block['object_relative_source_q_position'] == pytest.approx(source.q[0].tolist())
    assert block['object_token_provenance']['source'] == 'registered-fish-body-cog'


def test_missing_column_is_not_rendered_as_a_current_black_frame(service):
    service.owner.active_views = ('wrist',)
    service.owner.last_endpoint.observation.measurement_available[0,1] = False
    service._publish()
    status = service.status()
    assert status['input_mode'] == 'live-wrist-only'
    assert status['snapshot']['view_sources']['column'] == 'unavailable'
    with pytest.raises(ValueError, match='no current or restored'):
        service.frame('column', 'endpoint')
    assert service.owner.writes == 0


def test_tip_overlay_is_same_snapshot_only_and_never_changes_policy(service):
    import cv2
    from fcpg.v21_tip_display import tip_record
    service.target_diagnostics=True
    service.target_tracker=SimpleNamespace(locate_by_frozen_features=lambda *a:dict(visible=False))
    service.tip_display=SimpleNamespace(observe=lambda rgb,endpoint,raw,history:{
        v:tip_record([[80,100],[130,100]],
                     'wrist-fk-projection' if v == 'wrist' else 'column-rgb-jaw-contour',
                     float(endpoint.observation.image_times[0,i]),np.zeros(6))
        for i,v in enumerate(('wrist','column'))})
    before=service.owner.runtime.state.memory.features.clone()
    service._publish()
    status=service.status()
    assert not status['model']['pixel_keypoint_head']
    assert status['fingertips']['wrist']['available']
    status['fingertips']['wrist']['fm_px'][0][0]=10  # response is not mutable state
    jpeg=service.frame('wrist','endpoint',status['snapshot']['sequence'])
    image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
    assert image[100,80,1] > image[100,80,0] + 60
    live=cv2.imdecode(np.frombuffer(service.frame('wrist','live'),np.uint8),cv2.IMREAD_COLOR)
    assert int(live[100,80].max()) == 0
    assert service.owner.writes == 0
    assert torch.equal(service.owner.runtime.state.memory.features,before)


def test_status_and_video_do_not_update_model_history(service):
    state=service.owner.runtime.state;sequence=service.sequence
    for _ in range(3):
        service.status();assert service.frame('wrist').startswith(b'\xff\xd8')
    assert service.owner.runtime.state is state and service.sequence==sequence
    assert service.owner.writes==0


def test_transport_status_omits_bulk_position_servo_samples(service):
    servo = dict(
        schema='v21-arm-position-servo-v1', completed=True, applied=True,
        axes={'elbow_flex': dict(axis=2, original_target=12., applied=True,
                                 completed=True, stages={'i1': {'samples': list(range(1000))}})},
        readbacks=[{'q': list(range(100))} for _ in range(100)],
        parameter_writes=[{'register': 'P_Coefficient'} for _ in range(7)],
        original_goal_hold=dict(completed=True, sample_count=22,
                                readbacks=list(range(1000)), snapshot_attempts=list(range(1000))),
    )
    service.last_execution = {'completed': True, 'position_servo': servo}
    service.last_attempt = deepcopy(service.last_execution)

    full = service.status()
    compact = service.transport_status()
    assert len(full['last_execution']['position_servo']['readbacks']) == 100
    for key in ('last_execution', 'last_attempt'):
        summary = compact[key]['position_servo']
        assert summary['telemetry_omitted_from_status'] is True
        assert summary['telemetry_counts'] == {'axes': 1, 'readbacks': 100,
                                                'parameter_writes': 7}
        assert 'stages' not in summary['axes']['elbow_flex']
        assert 'readbacks' not in summary['original_goal_hold']
        assert summary['original_goal_hold']['readback_count'] == 1000
        assert summary['original_goal_hold']['snapshot_attempt_count'] == 1000
    assert len(json.dumps(compact)) < len(json.dumps(full)) / 10


def test_live_video_never_draws_the_last_endpoint_object_overlay(service,monkeypatch):
    import cv2
    service.owner.last_object_diagnostic=np.array([1.,.8,0.,0.,.4,.2,1.,0.],np.float32)
    service.owner.object_relative=SimpleNamespace(last_record={'centroid_px':[100.,100.]})
    monkeypatch.setattr(service,'_draw_object_geometry',lambda *a,**k:pytest.fail('no stale live overlay'))
    image=cv2.imdecode(np.frombuffer(service.frame('wrist','live'),np.uint8),cv2.IMREAD_COLOR)
    assert image.max()==0 and service.owner.writes==0


@pytest.mark.parametrize('truncated', [True, False])
def test_endpoint_object_overlay_uses_its_snapshot_not_later_encoder_state(service,monkeypatch,truncated):
    # Overlay prefers whole-object colour body (last_record), not the DINO tail CoG.
    record={'centroid_px':np.array([100.,100.]),'cog_px':np.array([100.,100.]),
            'axis_px':np.array([0.,1.]),'length_px':30.,'width_px':10.,
            'source':'operator_colour_body','body_truncated':truncated,
            'contour_px':[[90.,90.],[110.,90.],[110.,110.],[90.,110.]]}
    service.owner.object_relative=SimpleNamespace(last_record=record,last_policy_record=None,
        jaw_pair=lambda q:np.array([[90.,140.],[130.,140.]]))
    service.owner.last_object_relative=np.array([1.,.8,0.,-.2,.2,.1,-1.,0.],np.float32)
    service._publish();sequence=service.sequence
    overlay=service.status()['snapshot']['object_overlay']
    json.dumps(overlay,allow_nan=False)
    assert overlay['source']=='operator_colour_body'
    assert overlay['grasp_kind']=='colour_body_width_run'
    assert overlay['view']=='wrist'
    assert overlay['cog_px']==[100.,100.]
    assert overlay['body_truncated'] is truncated
    assert overlay['support_clipped'] is truncated
    assert overlay['aimable'] is None
    assert overlay['policy_token_available'] is True
    assert overlay['target_geometry']=='visible_colour_support_width_run'
    # Overlay blue-line contract: grasp_axis_px is always ⊥ axis_px.
    ax=np.asarray(overlay['axis_px'],float); ga=np.asarray(overlay['grasp_axis_px'],float)
    np.testing.assert_allclose(ga, [-ax[1]/np.linalg.norm(ax), ax[0]/np.linalg.norm(ax)], atol=1e-9)
    assert abs(float(np.dot(ax/np.linalg.norm(ax), ga))) < 1e-9
    record['centroid_px'][:]=[180.,180.]
    record['cog_px'][:]=[180.,180.]
    service.owner.last_object_relative[2]=.9
    seen=[]
    def draw(image,*args,overlay=None):seen.append(overlay);return image
    monkeypatch.setattr(service,'_draw_object_geometry',draw)
    service.frame('wrist','endpoint',sequence)
    assert seen[0]['centroid_px']==[100.,100.]
    assert seen[0]['vector'][2]==0.
    assert service.owner.writes==0


def test_column_grasp_target_prefers_cog_px(service, monkeypatch):
    called = {}

    def fake_backproj(fit, pixel, z_base):
        called['pixel'] = np.asarray(pixel, np.float64).tolist()
        return np.array([0.20, 0.10, z_base])

    monkeypatch.setattr(
        'fcpg.v21_live_service.approximate_column_backprojection_on_plane', fake_backproj)
    service._column_fit = object()
    service.targets = {'targets': {'pickup': {'column': {
        'visible': True,
        'cog_px': [80.0, 90.0],
        'centroid_px': [10.0, 10.0],
    }}}}
    xyz = service._column_grasp_target_xyz()
    assert called['pixel'] == [80.0, 90.0]
    assert xyz is not None


def _jaw_pair_linear(q):
    q = np.asarray(q, np.float64).reshape(-1)
    mid = np.array([112. + 3. * q[0], 140. - 2. * q[1] + 1. * q[2]])
    return np.stack([mid - [20., 0.], mid + [20., 0.]])


def _run_pinch_builder(service, monkeypatch, *, closing, vector, cog_px, truncated=False):
    rt = service.owner.runtime
    service.owner.q = np.array([0., -30., 40., 60., -52., 31.], np.float32)  # unfolded, jaws open
    service.owner.last_object_relative = list(vector)
    service.owner.object_relative = SimpleNamespace(
        last_policy_record={'cog_px': cog_px, 'source': 'frozen_vit_features'},
        last_record={'cog_px': cog_px, 'source': 'operator_colour_body', 'body_truncated': truncated},
        jaw_pair=_jaw_pair_linear)
    original = rt.core.act
    def act(state, text, offsets):
        out = original(state, text, offsets)
        cmds = state.q[0].detach().clone()[None].repeat(5, 1)
        cmds[:, 5] = torch.linspace(31., 6., 5) if closing else 31.
        out.commands_position.data.copy_(cmds[None])
        return out
    monkeypatch.setattr(rt.core, 'act', act)
    rt.stop()
    rt.plan(endpoint=service.owner.capture(), command_builder=service._vertical_pinch_builder,
            action_source='v21_actor_execution_geometry')
    return rt.pending.trace.positions[0].detach().cpu().numpy(), service.execution_geometry_record


def test_pinch_builder_holds_jaws_and_servos_wrist_r_when_object_far(service, monkeypatch):
    far = [1., .45, -.5, .1, .5, .16, 1., 0.]          # r≈0.51, like trial 1789358513759997162
    cmds, diag = _run_pinch_builder(service, monkeypatch, closing=True, vector=far, cog_px=[40., 150.])
    assert diag['wrist_r']['close_veto'] == 'wrist_r_far'
    np.testing.assert_allclose(cmds[:, 5], 31.)            # Actor close vetoed
    assert diag['approach_override'] == 'wrist_r'
    assert diag['wrist_r']['servo']['applied'] is True
    assert cmds[-1, 0] < 0.                                 # J1 pans toward the left CoG
    assert diag['aim']['reason'] == 'wrist_r_owns_approach'
    assert service.owner.writes == 0


@pytest.mark.parametrize(('kinematics_available', 'column_available'), [(True, True), (False, True), (True, False)])
def test_pinch_builder_falls_back_to_column_m_on_wrist_stall(service, monkeypatch, kinematics_available, column_available):
    import fcpg.v21_live_service as ls

    kinematics = object()
    pulls = []
    if kinematics_available:
        service.owner.kinematics = kinematics

    def stall(q, cmds, *args, **kwargs):
        return cmds.copy(), dict(kind='execution_geometry_wrist_r', applied=False,
                                 reason='no_progress_within_limits')

    def pull(q, cmds, kin, target, lower, upper):
        assert kin is kinematics
        pulls.append(kin)
        out = np.asarray(cmds, np.float64).copy()
        out[:, 0] = q[0] - 3.0
        return out, dict(kind='execution_geometry_align_M', applied=True, error_m=0.04)

    monkeypatch.setattr(ls, 'project_wrist_m_toward_pixel', stall)
    monkeypatch.setattr(ls, 'project_fk_m_toward_xyz', pull)
    column_targets = {'targets': {'pickup': {'column': {
        'visible': True, 'source': 'operator_colour_body',
        'cog_px': [90.0, 130.0], 'centroid_px': [90.0, 130.0],
    }}}}
    # Supply this test's observation when the builder refreshes diagnostics;
    # the shared fixture has target diagnostics disabled and publishes no target.
    monkeypatch.setattr(service, '_publish', lambda: setattr(
        service, 'targets', deepcopy(column_targets) if column_available else {'targets': {}}))
    service._column_grasp_target_xyz = lambda: np.array([0.22, 0.12, 0.02])
    far = [1., .45, -.5, .1, .5, .16, 1., 0.]
    cmds, diag = _run_pinch_builder(service, monkeypatch, closing=True, vector=far, cog_px=[40., 150.])
    if kinematics_available and column_available:
        assert diag['approach_override'] == 'column_m_after_wrist_stall'
        assert diag['aim']['applied'] is True
        assert cmds[-1, 0] < 0.
        assert pulls == [kinematics]
    else:
        assert diag['approach_override'] == 'wrist_r_hold'
        assert diag['aim']['applied'] is False
        np.testing.assert_allclose(cmds[:, :3], np.broadcast_to(service.owner.q[:3], (5, 3)))
        assert pulls == []
    np.testing.assert_allclose(cmds[:, 5], 31.)
    assert service.owner.writes == 0


def test_pinch_builder_lets_actor_close_when_wrist_r_near(service, monkeypatch):
    near = [1., .45, .04, .05, .5, .16, -1., 0.]        # r≈0.06 < WRIST_R_CLOSE
    cmds, diag = _run_pinch_builder(service, monkeypatch, closing=True, vector=near, cog_px=[120., 150.])
    assert diag['wrist_r']['close_veto'] is None
    assert cmds[-1, 5] < 10.                                # Actor's close passes
    assert diag['wrist_r']['servo'] is None                 # no far-phase servo


def test_pinch_builder_vetoes_close_when_wrist_cue_unavailable(service, monkeypatch):
    none = [0.] * 8
    cmds, diag = _run_pinch_builder(service, monkeypatch, closing=True, vector=none, cog_px=[120., 150.])
    assert diag['wrist_r']['close_veto'] == 'wrist_cue_unavailable'
    np.testing.assert_allclose(cmds[:, 5], 31.)


def test_partial_body_permits_approach_and_preserves_existing_far_close_gate(service, monkeypatch):
    far = [1., .45, -.5, .1, .5, .16, 1., 0.]
    cmds, diag = _run_pinch_builder(service, monkeypatch, closing=True,
        vector=far, cog_px=[40., 150.], truncated=True)
    assert diag['wrist_r']['cue']['valid']
    assert not diag['wrist_r']['cue']['body_complete']
    assert diag['wrist_r']['close_veto'] == 'wrist_r_far'
    assert diag['approach_override'] == 'wrist_r'
    np.testing.assert_allclose(cmds[:, 5], 31.)


def test_column_grasp_pixel_prefers_colour_body_cog_over_dino_mask(service):
    # DINO mask station would land on a left-edge blob (tube); colour cog is the fish.
    mask = np.zeros((224, 224), dtype=bool)
    mask[100:160, 20:50] = True
    column = dict(
        source='operator_colour_body',
        support_mask=mask,
        contour_px=[[20., 100.], [50., 100.], [50., 160.], [20., 160.]],
        cog_px=[120.0, 140.0],
        centroid_px=[120.0, 140.0],
    )
    pixel = service._column_grasp_pixel(column)
    np.testing.assert_allclose(pixel, [120.0, 140.0])


def test_column_grasp_pixel_prefers_body_station_over_tail_cog(service):
    # DINO-only records: never rebuild support_mask station (tube lock). Use CoG.
    mask = np.zeros((224, 224), dtype=bool)
    mask[100:124, 30:90] = True   # body slab
    mask[96:128, 90:110] = True   # short wide tail
    column = {
        'visible': True,
        'source': 'frozen_vit_features',
        'cog_px': [100.0, 112.0],
        'centroid_px': [100.0, 112.0],
        'support_mask': mask,
    }
    pixel = service._column_grasp_pixel(column)
    np.testing.assert_allclose(pixel, [100.0, 112.0])


def test_wrist_r_cue_uses_colour_diagnostic_not_thin_dino(service):
    # Thin DINO Actor 8-D would shape_reject; colour diagnostic is a real fish body.
    service.owner.last_object_relative = [1., .9, -.5, .05, .08, .02, 1., 0.]  # thin → reject
    service.owner.last_object_diagnostic = [1., .9, -.2, .05, .45, .16, 1., 0.]  # fish
    service.owner.object_relative = SimpleNamespace(
        last_record={'cog_px': [80., 140.], 'source': 'operator_colour_body'},
        last_policy_record={'cog_px': [40., 140.], 'source': 'frozen_vit_features',
                            'width_px': 4., 'length_px': 18.},
        last_diagnostic_vector=np.array([1., .9, -.2, .05, .45, .16, 1., 0.], np.float32),
    )
    cue = service._wrist_r_cue()
    assert cue['valid'] is True
    assert cue['source'] == 'operator_colour_body'
    assert abs(cue['r'] - float(np.hypot(.2, .05))) < 1e-6
    assert cue['cog_px'] == [80., 140.]


def test_historical_column_is_labelled_and_not_saved_as_current(service):
    from fcpg.v21_saved_observation import load_saved_column
    owner=service.owner
    old_time=owner.last_endpoint.time.item()-5
    old=replace(owner.last_endpoint.observation,
                image_times=torch.tensor([[0.,old_time]],dtype=torch.float64))
    owner.saved_column=SimpleNamespace(rgb=np.full((224,224,3),80,np.uint8),observation=old)
    owner.active_views=('wrist',)
    owner.last_endpoint=replace(owner.last_endpoint,observation=replace(owner.last_endpoint.observation,
        measurement_available=torch.tensor([[True,False]])))
    current_times=owner.last_endpoint.observation.image_times.clone()
    service._publish()
    data=service.status()
    assert data['input_mode']=='live-wrist-plus-historical-column'
    assert data['snapshot']['view_sources']['column']=='historical_observation_not_live'
    assert data['snapshot']['image_times'][0][1]==old_time
    torch.testing.assert_close(owner.last_endpoint.observation.image_times,current_times)
    with pytest.raises(ValueError,match='real Column'):
        load_saved_column(service.output/'current-observation.npz',now=owner.last_endpoint.time)
    with np.load(service.output/'current-observation.npz', allow_pickle=False) as saved:
        assert 'wrist_raw_rgb' in saved and 'column_raw_rgb' not in saved
    with pytest.raises(ValueError,match='no current'):
        service.frame('column','live')
    assert service.frame('column','endpoint').startswith(b'\xff\xd8')


def test_saved_raw_images_remain_paired_with_current_exposures(service):
    owner = service.owner
    owner.frames['wrist'][31, 73] = [17, 63, 219]
    owner.frames['column'][39, 86] = [231, 12, 77]
    before = owner.last_endpoint.observation
    history = owner.runtime.state.history.valid.clone()
    writes = owner.writes
    service._publish()
    with np.load(service.output/'current-observation.npz', allow_pickle=False) as saved:
        for view in ('wrist', 'column'):
            np.testing.assert_array_equal(saved[view + '_raw_rgb'], owner.frames[view])
        np.testing.assert_array_equal(saved['rgb'], owner.last_rgb)
        np.testing.assert_array_equal(saved['q_at_image'], before.q_at_image.numpy())
        np.testing.assert_array_equal(saved['image_times_s'], before.image_times.numpy())
    torch.testing.assert_close(owner.runtime.state.history.valid, history)
    assert owner.writes == writes and owner.runtime.stopped


def test_observe_preserves_stop_and_execution_preserves_actual_history(service):
    service.handle('observe');assert service.owner.runtime.stopped
    service.handle('plan');assert service.status()['ready']
    service.handle('execute')
    d=service.status();assert d['runtime']['stopped'] and d['runtime']['history_chunks']==[1]
    assert service.owner.writes==5 and len(d['last_execution']['retimed_predicted_q_path'])==6
    service.handle('observe')
    assert service.status()['runtime']['history_chunks']==[1] and service.owner.writes==5


def test_preview_is_visible_but_never_arms_or_executes(service):
    before=service.owner.runtime.state
    result=service.handle('preview')
    assert result['preview']['executable'] is False
    assert result['preview']['memory_committed'] is False
    assert service.owner.runtime.state is before
    assert result['preview']['observation_sequence']==result['snapshot']['sequence']
    assert len(result['preview']['commands_position'][0])==5
    assert result['runtime']['stopped'] and not result['runtime']['pending'] and not result['ready']
    assert service.owner.writes==0 and result['runtime']['history_chunks']==[0]


def test_repeated_previews_and_cancelled_plans_do_not_evict_actual_frames(service):
    service.handle('plan');service.handle('execute')
    before=service.owner.runtime.state
    sequence=service.sequence
    for _ in range(4):
        service.handle('preview')
        service.handle('plan')
        service.handle('stop')
    assert service.sequence>sequence
    assert service.owner.runtime.state is before
    assert service.owner.writes==5
    assert service.status()['runtime']['history_chunks']==[1]
    service.handle('plan');service.handle('execute')
    assert service.owner.writes==10
    assert service.status()['runtime']['history_chunks']==[2]


def test_execution_refreshes_the_plan_before_committing_source_frame(service):
    before=service.owner.runtime.state
    planned=service.handle('plan')
    first_time=planned['plan']['source_time_s']
    assert service.owner.runtime.state is before
    executed=service.handle('execute')
    assert executed['last_execution']['source_time_s']>first_time
    assert service.owner.runtime.state is not before
    assert service.owner.writes==5 and service.owner.runtime.stopped


def test_home_uses_readbacks_then_resets_unmodelled_history(service,monkeypatch):
    import fcpg.v21_live_service as module
    service.handle('plan');service.handle('execute')
    assert service.owner.runtime.state.history.valid.sum()==1
    target=np.full(6,20.,np.float32);owner=service.owner;home_calls=[]
    owner.read_checked=lambda:owner.q.copy()
    owner.io.settle_position_tolerance=3.
    (service.output/'pending-home-receipt.json').write_text('{}')
    owner.io.execute_home_chunk=lambda *a,**k:pytest.fail('legacy dual-camera HOME called')
    monkeypatch.setattr(module.established,'_load_v177_home_state',lambda *a,**k:target.copy())
    monkeypatch.setattr(module,'plan_home_chunk',lambda *a:np.repeat(target[None],5,axis=0))
    def home_readbacks(current,commands,**kwargs):
        home_calls.append(commands.copy());current.q=target.copy()
        return dict(kind='manual-home-readback-only',model_transition=False,completed=True)
    monkeypatch.setattr(module,'execute_home_readback_chunk',home_readbacks)
    result=service.handle('home')
    assert len(home_calls)==1 and result['runtime']['history_chunks']==[0]
    assert result['runtime']['stopped'] and owner.writes==5
    assert 'explicit HOME' in result['memory_reset_reason']
    assert not (service.output/'pending-home-receipt.json').exists()


def test_home_retries_only_expired_camera_readback_bracket(service, monkeypatch):
    import fcpg.v21_live_service as module
    owner=service.owner
    target=np.full(6,20.,np.float32)
    owner.read_checked=lambda:owner.q.copy()
    owner.io.settle_position_tolerance=3.
    monkeypatch.setattr(module.established,'_load_v177_home_state',lambda *a,**k:target.copy())
    monkeypatch.setattr(module,'plan_home_chunk',lambda *a:np.repeat(target[None],5,axis=0))
    monkeypatch.setattr(module,'execute_home_readback_chunk',
                        lambda current,commands,**kwargs: (setattr(owner,'q',target.copy()) or
                                                            dict(kind='manual-home-readback-only',
                                                                 model_transition=False,completed=True)))
    original=service._observe
    calls=[]
    def delayed_observe():
        calls.append(len(calls)+1)
        if len(calls)<3:
            raise RuntimeError('camera/readback acquisition bracket exceeds 50ms')
        return original()
    monkeypatch.setattr(service,'_observe',delayed_observe)

    service.handle('home')

    assert calls==[1,2,3]
    receipt=next(service.output.glob('home-*.json'))
    assert json.loads(receipt.read_text())['observation']=={
        'attempts':3,
        'bracket_timeouts':['camera/readback acquisition bracket exceeds 50ms']*2,
    }


def test_failed_home_does_not_discard_actual_history(service,monkeypatch):
    import fcpg.v21_live_service as module
    service.handle('plan');service.handle('execute')
    owner=service.owner;before=owner.runtime.state
    target=np.full(6,20.,np.float32)
    owner.read_checked=lambda:owner.q.copy();owner.io.settle_position_tolerance=3.
    monkeypatch.setattr(module.established,'_load_v177_home_state',lambda *a,**k:target.copy())
    monkeypatch.setattr(module,'plan_home_chunk',lambda *a:np.repeat(target[None],5,axis=0))
    monkeypatch.setattr(module,'execute_home_readback_chunk',
        lambda *a,**k:(_ for _ in ()).throw(RuntimeError('HOME readback failure')))
    with pytest.raises(RuntimeError,match='readback failure'):service.handle('home')
    assert owner.runtime.state is before and owner.runtime.stopped and owner.stop_event.is_set()


def test_home_failed_servo_restoration_never_clears_stop_or_failure_state(service,monkeypatch):
    import fcpg.v21_position_servo_recovery as recovery
    owner=service.owner
    owner.runtime.requires_home_after_incomplete_execution=True
    owner.stop_event.set()
    state=owner.runtime.state

    def rejected(current):
        assert current is service and service.operation_lock.locked()
        assert owner.stop_event.is_set() and owner.runtime.stopped
        raise RuntimeError('failed profile lacks actual restoration proof')

    monkeypatch.setattr(recovery,'restore_failed_position_servo_baseline',rejected)
    monkeypatch.setattr(service,'_clear_stop_for_current_operation',
        lambda:pytest.fail('HOME must not arm after failed profile restoration'))
    with pytest.raises(RuntimeError,match='lacks actual restoration proof'):
        service.handle('home')
    assert owner.stop_event.is_set() and owner.runtime.stopped
    assert owner.runtime.requires_home_after_incomplete_execution
    assert owner.runtime.state is state and owner.writes==0


def test_home_profile_restoration_precedes_arming_and_preserves_home_requirement(service,monkeypatch):
    import fcpg.v21_live_service as module
    import fcpg.v21_position_servo_recovery as recovery
    owner=service.owner
    owner.runtime.requires_home_after_incomplete_execution=True
    owner.stop_event.set()
    owner.read_checked=lambda:owner.q.copy()
    owner.io.settle_position_tolerance=3.
    calls=[]

    def restored(current):
        assert current is service and service.operation_lock.locked()
        assert owner.stop_event.is_set() and owner.runtime.requires_home_after_incomplete_execution
        calls.append('restored')
        return dict(completed=True,model_history_committed=False)

    clear=service._clear_stop_for_current_operation
    def arm():
        assert calls==['restored'] and owner.runtime.requires_home_after_incomplete_execution
        calls.append('home_armed')
        clear()

    monkeypatch.setattr(recovery,'restore_failed_position_servo_baseline',restored)
    monkeypatch.setattr(service,'_clear_stop_for_current_operation',arm)
    monkeypatch.setattr(module.established,'_load_v177_home_state',lambda *a,**k:owner.q.copy())
    result=service.handle('home')
    assert calls==['restored','home_armed']
    assert not result['runtime']['requires_home_after_incomplete_execution']
    assert owner.stop_event.is_set() and owner.runtime.stopped and owner.writes==0


def test_fixed_goal_compensation_dispatch_keeps_stop_and_model_feedback_separate(service,monkeypatch):
    import fcpg.v21_fixed_goal_service as compensation
    owner=service.owner
    owner.runtime.requires_home_after_incomplete_execution=True
    owner.stop_event.set()
    state=owner.runtime.state
    calls=[]
    def run(current):
        assert current is service and service.operation_lock.locked()
        assert owner.stop_event.is_set() and owner.runtime.stopped
        calls.append('compensate')
        service.fixed_goal_compensation=dict(completed=True,configuration_restored=True)
    monkeypatch.setattr(compensation,'handle_fixed_goal_compensation',run)
    result=service.handle('fixed-goal-compensate')
    assert calls==['compensate'] and owner.runtime.state is state and owner.writes==0
    assert result['runtime']['stopped'] and result['runtime']['requires_home_after_incomplete_execution']
    assert result['fixed_goal_compensation']['configuration_restored'] is True
    with pytest.raises(ValueError,match='accepts no parameters'):
        service.handle('fixed-goal-compensate',{'target':1})
    assert calls==['compensate']


def test_home_freshly_verifies_completed_compensation_before_skipping_cap_restoration(service,monkeypatch):
    import fcpg.v21_fixed_goal_service as compensation
    import fcpg.v21_position_servo_recovery as recovery
    import fcpg.v21_live_service as module
    owner=service.owner
    owner.runtime.requires_home_after_incomplete_execution=True
    owner.stop_event.set()
    owner.read_checked=lambda:owner.q.copy()
    owner.io.settle_position_tolerance=3.
    calls=[]
    def verified(current):
        assert current is service and service.operation_lock.locked()
        assert owner.stop_event.is_set() and owner.runtime.requires_home_after_incomplete_execution
        calls.append('fresh_compensation_readback')
        return dict(completed=True,configuration_restored=True)
    monkeypatch.setattr(compensation,'verify_fixed_goal_compensation_before_home',verified)
    monkeypatch.setattr(recovery,'restore_failed_position_servo_baseline',
        lambda *a:pytest.fail('Already restored cap must not be treated as cap200'))
    clear=service._clear_stop_for_current_operation
    def arm():
        assert calls==['fresh_compensation_readback'] and owner.runtime.requires_home_after_incomplete_execution
        calls.append('home_armed');clear()
    monkeypatch.setattr(service,'_clear_stop_for_current_operation',arm)
    monkeypatch.setattr(module.established,'_load_v177_home_state',lambda *a,**k:owner.q.copy())
    result=service.handle('home')
    assert calls==['fresh_compensation_readback','home_armed']
    assert not result['runtime']['requires_home_after_incomplete_execution']
    assert owner.stop_event.is_set() and owner.runtime.stopped and owner.writes==0


def test_compensation_compact_status_does_not_apply_old_success_to_a_new_execution(service):
    class NoCopy:
        def __deepcopy__(self, memo):
            pytest.fail('compact compensation status must not copy full readbacks')
    service.fixed_goal_compensation = dict(source_time_s=10., completed=True,
        configuration_restored=True, readbacks=NoCopy())
    service.last_attempt = dict(source_time_s=10., completed=False)
    matching = service.status()['fixed_goal_compensation']
    assert matching['source_matches_current_execution'] is True
    assert matching['source_time_s'] == 10. and matching['completed'] is True
    assert 'readbacks' not in matching
    service.last_attempt = dict(source_time_s=20., completed=False)
    newer = service.status()['fixed_goal_compensation']
    assert newer['source_matches_current_execution'] is False
    assert newer['completed'] is True  # Retain the historical result truthfully.
    service.last_attempt = None  # The coordinator uses the saved source after restart.
    assert service.status()['fixed_goal_compensation']['source_matches_current_execution'] is True
    service.fixed_goal_compensation = dict(completed=True)
    assert service.status()['fixed_goal_compensation']['source_matches_current_execution'] is False
    assert service.owner.writes == 0


def test_alignment_motion_is_disabled_by_default(service):
    before=service.owner.runtime.state
    with pytest.raises(RuntimeError,match='not enabled'):
        service.handle('alignment-probe',{'joint':0,'direction':1})
    assert service.owner.writes==0 and service.owner.runtime.state is before
    assert service.owner.runtime.stopped


def test_alignment_probe_uses_shared_runner_and_explicit_source(service,monkeypatch):
    from fcpg.v21_visual_alignment import alignment_observation
    owner=service.owner;service.alignment_motion_enabled=True
    import fcpg.v21_live_service as live
    monkeypatch.setattr(live.established,'_load_v177_home_state',lambda *a,**k:owner.q.copy())
    owner.read_checked=owner._read_q
    owner.io.settle_position_tolerance=3.
    def measured():
        endpoint=owner.last_endpoint;t=endpoint.time.item()
        q=endpoint.q[0].numpy();record=dict(role='pickup',view='wrist',visible=True,
            view_source='current_observation',source='frozen_dinov2_profile_prototype',
            width_px=20.,length_px=40.,
            image_time_s=t,centroid_px=[110+q[0],112+q[1]],axis_px=[0,1])
        return alignment_observation(record,q,[[90,112],[130,112]],
            image_time_s=t,q_time_s=t,now_s=t,max_age_s=.5)
    monkeypatch.setattr(service,'_alignment_measure',measured)
    result=service.handle('alignment-probe',{'joint':0,'direction':1})
    assert owner.writes==5 and result['runtime']['stopped']
    assert result['last_execution']['action_source']=='visual_alignment_probe'
    assert result['alignment']['response_pairs']==1
    result=service.handle('alignment-return')
    assert owner.writes==10 and result['last_execution']['action_source']=='visual_alignment_return'
    np.testing.assert_allclose(owner.q,np.zeros(6),atol=1e-6)
    assert result['alignment']['response_pairs']==2


def test_invalid_probe_cannot_accept_raw_commands(service):
    service.alignment_motion_enabled=True
    with pytest.raises(ValueError,match='only joint'):
        service.handle('alignment-probe',{'commands_position':[[0]*6]*5})
    assert service.owner.writes==0 and service.owner.runtime.stopped


def test_unsafe_preview_shows_reason_without_executable_plan(service):
    service.owner._physical_check=lambda q,c:(_ for _ in ()).throw(RuntimeError('physical limit'))
    result=service.handle('preview')
    assert not result['preview']['mechanical_admission']['allowed']
    assert result['preview']['mechanical_admission']['reason']=='physical limit'
    assert result['runtime']['stopped'] and not result['ready'] and service.owner.writes==0


def test_stop_during_plan_publication_does_not_leave_ready_plan(service,monkeypatch):
    original=service._publish
    def stop_first():service.stop();original()
    monkeypatch.setattr(service,'_publish',stop_first)
    result=service.handle('plan')
    assert result['runtime']['stopped'] and not result['ready'] and service.owner.writes==0


def test_busy_control_is_not_queued_and_stop_remains_available(service):
    service.operation_lock.acquire()
    try:
        with pytest.raises(RuntimeError,match='active'):service.handle('execute')
        service.handle('stop');assert service.owner.stop_event.is_set()
    finally:service.operation_lock.release()


def test_file_failure_does_not_leave_operation_lock_held(service,monkeypatch):
    import fcpg.v21_live_service as module
    service.handle('plan')
    monkeypatch.setattr(module,'write_json',lambda *a:(_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):service.handle('execute')
    assert not service.operation_lock.locked() and service.owner.runtime.stopped


def test_http_explicit_diagnostics_and_cross_origin_control_rejected(service):
    server=make_server(('127.0.0.1',0),{'service':service})
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        html=urlopen(base).read().decode()
        assert 'V21.0.2' in html and 'Wrist 参考参与 Actor 的目标相对量' in html and 'AF/AM' not in html
        data=json.load(urlopen(base+'/api/status'));assert data['runtime']['stopped']
        assert data['camera_condition']=='both_fresh'
        req=Request(base+'/api/execute',data=b'{}',headers={'Origin':'https://unrelated.example','Content-Type':'application/json'})
        with pytest.raises(HTTPError) as error:urlopen(req)
        assert error.value.code==403 and service.owner.writes==0
        bad=Request(base+'/api/camera-condition',data=b'{"camera_condition":"wrist_h0"}',
                    headers={'Content-Type':'application/json'})
        with pytest.raises(HTTPError) as mid:urlopen(bad)
        assert mid.value.code==409 and service.owner.writes==0
    finally:
        server.shutdown();server.server_close();worker.join(timeout=2)


def test_failed_hardware_startup_keeps_page_and_rejects_control():
    server=make_server(('127.0.0.1',0),{'startup_error':'Column camera read failed'})
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        html=urlopen(base).read().decode()
        assert '更新定位 + 预览' in html
        with pytest.raises(HTTPError) as status:urlopen(base+'/api/status')
        assert status.value.code==503
        assert json.loads(status.value.read())['error']=='Column camera read failed'
        with pytest.raises(HTTPError) as control:urlopen(Request(base+'/api/execute',data=b'{}'))
        assert control.value.code==423
    finally:
        server.shutdown();server.server_close();worker.join(timeout=2)


def test_browser_shows_tip_sources_without_sending_commands(service):
    from fcpg.v21_tip_display import tip_record
    playwright=pytest.importorskip('playwright.sync_api')
    chrome=shutil.which('google-chrome')
    if chrome is None:pytest.skip('system Chrome is unavailable')
    service.target_diagnostics=True
    service.target_tracker=SimpleNamespace(locate_by_frozen_features=lambda *a:dict(visible=False))
    def same_frame_tips(rgb, endpoint, raw, history):
        tips = {
            view: tip_record([[80,100],[130,100]],
                'wrist-fk-projection' if view == 'wrist' else 'column-rgb-jaw-contour',
                float(endpoint.observation.image_times[0,index]), np.zeros(6))
            for index, view in enumerate(('wrist','column'))}
        tips['column']['ordered_identity_available'] = False
        return tips
    service.tip_display=SimpleNamespace(observe=same_frame_tips)
    service._publish()
    server=make_server(('127.0.0.1',0),{'service':service})
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    try:
        with playwright.sync_playwright() as p:
            browser=p.chromium.launch(headless=True,executable_path=chrome)
            try:
                page=browser.new_page()
                page.goto(f'http://127.0.0.1:{server.server_port}/',wait_until='domcontentloaded')
                try:page.wait_for_load_state('networkidle',timeout=1500)
                except playwright.TimeoutError:pass  # continuous status/image polling
                playwright.expect(page.locator('.no-point')).to_contain_text('Wrist 机械参考')
                playwright.expect(page.locator('.no-point')).to_contain_text('Column 图像轮廓端点')
                playwright.expect(page.locator('.no-point')).to_contain_text('两端身份未确认')
                page.wait_for_function("['wristImage','columnImage'].every(id=>document.getElementById(id).naturalWidth>0)")
                page.get_by_label('图像来源').select_option('live')
                playwright.expect(page.locator('#sourceNote')).to_contain_text('不叠加旧时刻的位置')
                assert service.owner.writes == 0 and service.owner.runtime.stopped
            finally:browser.close()
    finally:
        server.shutdown();server.server_close();worker.join(timeout=2)


def test_browser_preview_and_cancel_keep_committed_history(service):
    playwright=pytest.importorskip('playwright.sync_api')
    chrome=shutil.which('google-chrome')
    if chrome is None:pytest.skip('system Chrome is unavailable')
    server=make_server(('127.0.0.1',0),{'service':service})
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    try:
        with playwright.sync_playwright() as p:
            browser=p.chromium.launch(headless=True,executable_path=chrome)
            try:
                page=browser.new_page();errors=[]
                page.on('pageerror',lambda error:errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}/',wait_until='domcontentloaded')
                try:page.wait_for_load_state('networkidle',timeout=1500)
                except playwright.TimeoutError:pass  # Intentional live image/status polling.
                playwright.expect(page.locator('#previewMemoryPolicy')).to_contain_text('临时观测')
                points=page.evaluate('commandStepPoints([[30],[20],[10]],[0,.1,.2],0,.3)')
                assert points == [[0,30],[.1,30],[.1,20],[.2,20],[.2,10],[.3,10]]
                assert page.evaluate('commandStepPoints([[30],[20]],[.2,.1],0,.3)') == []
                before=service.owner.runtime.state
                for _ in range(3):
                    page.get_by_role('button',name='更新定位 + 预览 · 不执行').click()
                    playwright.expect(page.locator('#operation')).to_contain_text('动作预览已更新')
                    assert service.owner.runtime.state is before and service.owner.writes==0
                page.get_by_role('button',name='生成可执行计划').click()
                playwright.expect(page.locator('#execute')).to_be_enabled()
                assert service.owner.runtime.state is before
                page.get_by_role('button',name='STOP',exact=True).click()
                playwright.expect(page.locator('#execute')).to_be_disabled()
                assert service.owner.runtime.state is before and service.owner.writes==0
                page.evaluate("""() => {
                    window.ageEvents=[];
                    new MutationObserver(()=>window.ageEvents.push(document.getElementById('wristAge').textContent))
                      .observe(document.getElementById('wristAge'),{childList:true,subtree:true,characterData:true});
                }""")
                stale=[False]
                def stale_once(route):
                    if stale[0]:return route.continue_()
                    stale[0]=True
                    route.fulfill(status=409,content_type='application/json',
                        body=json.dumps({'error':'observation snapshot changed; refresh status'}))
                page.route('**/api/frame/wrist.jpg?**',stale_once)
                page.wait_for_function("window.ageEvents.includes('同步新快照…')")
                page.wait_for_function("document.getElementById('wristAge').textContent.startsWith('定位图')")
                assert '读取失败' not in page.evaluate('window.ageEvents')
                assert not errors
            finally:browser.close()
    finally:
        server.shutdown();server.server_close();worker.join(timeout=2)


def test_draw_object_geometry_blue_grasp_is_perpendicular_to_green_axis(service):
    """Blue ink through C must be ⊥ green long-axis, not a parallel CM lookalike."""
    bgr = np.zeros((224, 224, 3), np.uint8)
    overlay = dict(
        cog_px=[112.0, 112.0], centroid_px=[112.0, 112.0],
        axis_px=[1.0, 0.0], grasp_axis_px=[1.0, 0.0],  # deliberately wrong/parallel
        length_px=80.0, width_px=24.0, contour_px=[],
        jaw_pair_px=[[90.0, 160.0], [134.0, 160.0]],
        vector=[1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # available body target
    )
    out = service._draw_object_geometry(bgr, overlay=overlay)
    green = out[112, 140]   # horizontal long axis
    blue = out[100, 112]    # vertical grasp ⊥ (above C; CM is below)
    assert int(green[1]) > 150 and int(green[1]) >= int(green[0]) and int(green[1]) >= int(green[2]), green
    assert int(blue[0]) > 150 and int(blue[0]) > int(blue[1]) and int(blue[0]) > int(blue[2]), blue


def test_wrist_drop_zone_registration_reaches_the_service_tracker(service, tmp_path, monkeypatch):
    """Training overhaul doc section 17.6: --wrist-drop-zone is checked before any hardware opens and handed to the
    service's V21TargetTracker; nothing moves."""
    from fcpg import v21_live_service
    from fcpg.v21_target_tracker import WRIST_DROP_ZONE_SCHEMA, load_wrist_drop_zone

    def registration(path, schema=WRIST_DROP_ZONE_SCHEMA):
        np.savez(path, schema=np.asarray(schema), template_rgb=np.full((56, 56, 3), 7, np.uint8),
                 centre_px=np.asarray((60., 100.)), radius_px=np.asarray(28.), source=np.asarray('demo 0 frame 449'))
        return path

    made = []

    class Tracker:
        def __init__(self, profile_root, *, wrist_drop_zone=None):
            made.append(wrist_drop_zone)

        def locate_by_frozen_features(self, role, view, frame, **kwargs):
            return dict(visible=False, reason='test')

    monkeypatch.setattr(v21_live_service, 'V21TargetTracker', Tracker)
    owner_type = type(service.owner)
    with pytest.raises(ValueError, match='is not a'):
        V21LiveService(tmp_path/'model', tmp_path/'bad-box', device='cpu', owner_factory=owner_type,
                       target_diagnostics=True, clock=service.clock,
                       wrist_drop_zone=registration(tmp_path/'other.npz', 'something-else'))
    path = registration(tmp_path/'box.npz')
    configured = V21LiveService(tmp_path/'model', tmp_path/'box', device='cpu', owner_factory=owner_type,
                                target_diagnostics=True, clock=service.clock, wrist_drop_zone=path)
    try:
        assert configured.wrist_drop_zone['sha256'] == load_wrist_drop_zone(path)['sha256']
        configured._publish()
        assert made and set(made) == {str(path.resolve())}
        assert configured.targets['targets']['drop_zone']['wrist']['visible'] is False
        assert configured.owner.writes == 0
    finally:
        configured.close()
