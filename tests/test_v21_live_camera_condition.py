"""Live 8765 camera-condition wiring; CPU fake observations, no robot or cameras."""
from dataclasses import replace
import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch

from fcpg.v21_camera_condition import CONDITIONS, V21CameraCondition
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_memory import VisualObservation
from fcpg.v21_runtime import EncodedEndpoint, V21ResearchRuntime
from fcpg.v21_live_service import V21LiveService
from scripts.serve_so101_v21_research import build_parser, make_server


EXPECTED = {
    "both_fresh": ((True, True), (True, True)),
    "wrist_h0": ((True, True), (False, True)),
    "wrist_cold": ((False, True), (False, True)),
    "external_h0": ((True, True), (True, False)),
    "external_cold": ((True, False), (True, False)),
}


class Owner:
    def __init__(self, root, output, task, device, open_hardware, **_extra):
        self.stop_event = threading.Event()
        self.q = np.zeros(6, np.float32)
        self.writes = 0
        self.holds = 0
        self.closed = False
        self._valid = (True, True)
        vocab = TaskVocabulary.build([task])
        core = V21ResearchCore(V21Config(len(vocab.words), object_relative=True))
        self.runtime = V21ResearchRuntime(
            core, vocab.encode([task]), physical_check=lambda q, u: self._physical_check(q, u),
            lease_check=lambda: None, clock_fn=lambda: now[0], max_plan_age_s=300)
        self.frames = {v: np.zeros((480, 640, 3), np.uint8) for v in ('wrist', 'column')}
        self.io = SimpleNamespace(camera_lock=threading.Lock(), _last_raw_frames=self.frames,
                                  settle_step_tolerance=.25, settle_position_tolerance=3.,
                                  _read_camera=lambda view: np.zeros((224, 224, 3), np.uint8))
        self.bridge = SimpleNamespace(plan=self.plan, execute=self.execute, stop=self.stop, last_attempt=None)

    def _read_q(self):
        return self.q.copy()

    def read_checked(self):
        return self._read_q()

    def _physical_check(self, q, commands):
        pass

    def _halt(self):
        self.holds += 1

    def capture(self):
        now[0] += .01
        self.last_rgb = np.zeros((2, 224, 224, 3), np.uint8)
        q = torch.from_numpy(self.q.copy())[None]
        t = torch.tensor([now[0]], dtype=torch.float64)
        cue = torch.tensor([[1., .8, .3, -.2, .4, .15, -.8, .6]])
        obs = VisualObservation(
            torch.ones(1, 2, 256, 384), q[:, None].expand(-1, 2, -1),
            t[:, None].expand(-1, 2), torch.tensor([self._valid], dtype=torch.bool),
            object_relative=cue)
        self.last_endpoint = EncodedEndpoint(q, t, obs)
        return self.last_endpoint

    def plan(self, **kwargs):
        self.runtime.stop()
        return self.runtime.plan(endpoint=self.capture(), **kwargs)

    def execute(self):
        pending = self.runtime.pending
        epoch = pending.source_time.item()
        action_source = self.runtime.pending_action_source
        q = pending.source_q
        u = pending.trace.positions.detach()
        path = torch.cat((q[:, None], u), 1)
        times = torch.tensor([[epoch + i * .1 for i in range(6)]], dtype=torch.float64)
        self.runtime.begin_execution(q, pending.source_time)
        pending = self.runtime.pending
        now[0] = epoch + .5
        self.q = u[0, -1].numpy()
        self.writes += 5
        self.runtime.complete(replace(pending.trace, source='recorded'), path, times, self.capture())
        self.runtime.stop()
        result = dict(completed=True, source_time_s=epoch, source_q_position=q[0].tolist(),
                      commands_position=u[0].tolist(), endpoint_time_s=float(self.runtime.state.time[0]),
                      action_source=action_source, policy_commands_attempted=5, policy_commands_returned=5,
                      measured_q_path=path[0].tolist(), readback_times_s=times[0].tolist(),
                      command_times_s=[epoch + i * .1 for i in range(5)])
        self.bridge.last_attempt = result
        return result

    def stop(self):
        self.runtime.stop()
        self._halt()

    def images(self, name):
        pass

    def close(self):
        self.closed = True


now = [10.]


def make_service(tmp_path, monkeypatch, **kwargs):
    now[0] = 10.
    monkeypatch.setattr(V21LiveService, '_monitor_q', lambda self: None)
    service = V21LiveService(tmp_path / 'model', tmp_path / 'out', device='cpu',
                             owner_factory=Owner, target_diagnostics=False,
                             clock=lambda: now[0], **kwargs)
    return service


def ready_at_home(service, monkeypatch):
    import fcpg.v21_live_service as module
    target = service.owner.q.copy()
    monkeypatch.setattr(module.established, '_load_v177_home_state', lambda *a, **k: target.copy())
    service.owner.io.settle_position_tolerance = 3.


@pytest.fixture
def service(tmp_path, monkeypatch):
    value = make_service(tmp_path, monkeypatch)
    yield value
    value.close()


def _mask(state):
    return tuple(bool(v) for v in state.measurement_available[0].tolist())


def test_cli_defaults_and_choices():
    args = build_parser().parse_args([])
    assert args.camera_condition == 'both_fresh' and args.camera_address == 'factual'
    parsed = build_parser().parse_args(['--camera-condition', 'wrist_cold', '--camera-address', 'zero'])
    assert parsed.camera_condition == 'wrist_cold' and parsed.camera_address == 'zero'
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--camera-condition', 'wrist_only'])


@pytest.mark.parametrize('name', CONDITIONS)
def test_startup_filters_h0_then_refresh(tmp_path, monkeypatch, name):
    service = make_service(tmp_path, monkeypatch, camera_condition=name)
    try:
        status = service.status()
        assert status['camera_condition'] == name
        assert status['camera_address'] == 'factual'
        assert status['camera_h0_committed']
        assert _mask(service.owner.runtime.state) == EXPECTED[name][0]
        if not EXPECTED[name][0][0]:
            assert not service.owner.runtime.state.object_relative.count_nonzero()
            assert not service.owner.last_endpoint.observation.object_relative.count_nonzero()
        service.handle('observe')
        assert _mask(service.owner.runtime.state) == EXPECTED[name][1]
        if not EXPECTED[name][1][0]:
            assert not service.owner.runtime.state.object_relative.count_nonzero()
        if name != 'both_fresh':
            missing = 0 if name.startswith('wrist_') else 1
            assert int(service.owner.runtime.state.memory.valid[0, missing].sum()) == int(name.endswith('_h0'))
    finally:
        service.close()


def test_zero_address_configures_unused_eval_core(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch, camera_condition='wrist_h0', camera_address='zero')
    try:
        assert service.status()['camera_address'] == 'zero'
        assert service.owner.runtime.core.reader.mode == 'temporal'
        assert service.owner.runtime.core.training is False
        ready_at_home(service, monkeypatch)
        before = service.owner.runtime.core
        service.handle('home')
        assert service.owner.runtime.core is not before
        assert service.owner.runtime.core.reader.mode == 'temporal'
        assert service.camera_condition.h0_committed
        assert _mask(service.owner.runtime.state) == EXPECTED['wrist_h0'][0]
    finally:
        service.close()


def test_preview_does_not_commit_h0(service):
    service.camera_condition = V21CameraCondition('wrist_h0')
    result = service.handle('preview')
    assert not service.camera_condition.h0_committed
    assert result['preview']['memory_committed'] is False
    assert service.owner.writes == 0


def test_h0_seed_requires_both_actual_views(service):
    service.camera_condition = V21CameraCondition('wrist_cold')
    service.owner.runtime.state = None
    service.owner._valid = (False, True)
    with pytest.raises(ValueError, match='H0 seed'):
        service._observe()
    assert not service.camera_condition.h0_committed
    assert service.owner.runtime.state is None


def test_home_payload_reseeds_condition(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    try:
        ready_at_home(service, monkeypatch)
        result = service.handle('home', {'camera_condition': 'external_cold', 'camera_address': 'factual'})
        assert result['camera_condition'] == 'external_cold'
        assert result['camera_h0_committed']
        assert _mask(service.owner.runtime.state) == EXPECTED['external_cold'][0]
        assert 'explicit HOME' in result['memory_reset_reason']
        assert not bool(service.owner.runtime.state.history.valid.any())
    finally:
        service.close()


def test_dedicated_post_only_when_idle_at_home(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    try:
        ready_at_home(service, monkeypatch)
        result = service.handle('camera-condition', {'camera_condition': 'wrist_h0'})
        assert result['camera_condition'] == 'wrist_h0'
        assert result['camera_h0_committed']
        assert _mask(service.owner.runtime.state) == EXPECTED['wrist_h0'][0]
        assert 'empty memory reseeds H0' in result['memory_reset_reason']
        service.handle('plan')
        with pytest.raises(RuntimeError, match='active live runtime'):
            service.handle('camera-condition', {'camera_condition': 'wrist_cold'})
        service.handle('execute')
        with pytest.raises(RuntimeError, match='populated live runtime'):
            service.handle('camera-condition', {'camera_condition': 'wrist_cold'})
    finally:
        service.close()


def test_completion_refresh_uses_committed_condition(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch, camera_condition='wrist_h0')
    try:
        service.handle('plan')
        service.handle('execute')
        assert _mask(service.owner.runtime.state) == EXPECTED['wrist_h0'][1]
        assert not service.owner.runtime.state.object_relative.count_nonzero()
        assert service.owner.writes == 5
    finally:
        service.close()


def test_an_owner_with_an_empty_encoder_slot_needs_an_object_finder(tmp_path, monkeypatch):
    # 10/31: the TrialOwner keeps its object encoder in object_relative (None until a finder installs one); a model that
    # reads the wrist object cue does not start without a finder (the colour fallback is gone)
    class EmptySlot(Owner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.object_relative = None
    monkeypatch.setattr(V21LiveService, '_monitor_q', lambda self: None)
    with pytest.raises(ValueError, match='object finder'):
        V21LiveService(tmp_path / 'model', tmp_path / 'out', device='cpu', owner_factory=EmptySlot,
                       target_diagnostics=False, clock=lambda: now[0])


def test_rejects_wrist_only_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(V21LiveService, '_monitor_q', lambda self: None)
    with pytest.raises(ValueError, match='empty dual-view'):
        V21LiveService(tmp_path / 'model', tmp_path / 'out', device='cpu', owner_factory=Owner,
                       target_diagnostics=False, wrist_only=True, camera_condition='wrist_cold')


def test_http_home_and_camera_condition(tmp_path, monkeypatch):
    service = make_service(tmp_path, monkeypatch)
    ready_at_home(service, monkeypatch)
    server = make_server(('127.0.0.1', 0), {'service': service})
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        home = Request(base + '/api/home',
                       data=b'{"camera_condition":"external_h0"}',
                       headers={'Content-Type': 'application/json'})
        data = json.loads(urlopen(home).read())
        assert data['camera_condition'] == 'external_h0' and data['camera_h0_committed']
        change = Request(base + '/api/camera-condition',
                         data=b'{"camera_condition":"wrist_cold","camera_address":"factual"}',
                         headers={'Content-Type': 'application/json'})
        data = json.loads(urlopen(change).read())
        assert data['camera_condition'] == 'wrist_cold'
        assert _mask(service.owner.runtime.state) == EXPECTED['wrist_cold'][0]
        extra = Request(base + '/api/camera-condition',
                        data=b'{"camera_condition":"wrist_h0","assist":true}',
                        headers={'Content-Type': 'application/json'})
        with pytest.raises(HTTPError) as err:
            urlopen(extra)
        assert err.value.code == 409
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        service.close()
