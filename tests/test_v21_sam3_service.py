"""The colour-free live finder inside the live service (training overhaul doc section 17.32): the owner's object encoder
is the SAM 3 one in the convention the model learned, the colour code is gone (10/31), the column fish for the live grasp
target comes from SAM 3 and the place goal takes only that. A fake finder answers on a Unix socket; no hardware.
Section 17.51: with the learned finder, the column fish comes from the learned column finder instead."""
import json
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg import v21_sam3_finder as finder
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_hardware_bridge import V21TransactionBridge
from fcpg.v21_live_service import TASK, V21LiveService
from fcpg.v21_memory import VisualObservation
from fcpg.v21_place_goal import HoldTracker, LivePlaceGoal
from fcpg.v21_runtime import EncodedEndpoint, V21ResearchRuntime


class FakeDino:
    def locate_by_frozen_features(self, role, view, frame, **kwargs):
        return dict(visible=True, score=.43)


def fish_mask(h, w):
    ys, xs = np.mgrid[:h, :w]
    return ((xs - .55 * w) / (.12 * w)) ** 2 + ((ys - .5 * h) / (.05 * h)) ** 2 <= 1.


@pytest.fixture
def fake_finder(tmp_path):
    path = tmp_path / 'finder.sock'
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(4)
    seen = []

    def serve(conn):
        with conn:
            while True:
                try:
                    header, payload = finder.receive_message(conn)
                except (ConnectionError, OSError):
                    return
                seen.append(header)
                if header['op'] == 'describe':
                    finder.send_message(conn, dict(ok=True, schema='fake'))
                elif header['op'] == 'track_reset':
                    finder.send_message(conn, dict(ok=True))
                else:
                    h, w, _ = header['shape']
                    mask = fish_mask(h, w)[None]
                    reply = dict(ok=True, mask_shape=[1, h, w], model_s=.01)
                    if header['op'] == 'track':
                        reply.update(track=dict(source='anchored_on_detection'), word_scores={'fish': .8})
                    else:
                        reply.update(scores=[.8], prompt_index=[0])
                    finder.send_message(conn, reply, finder.pack_masks(mask))

    def accept():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            threading.Thread(target=serve, args=(conn,), daemon=True).start()
    threading.Thread(target=accept, daemon=True).start()
    yield SimpleNamespace(path=path, seen=seen)
    server.close()


@pytest.fixture
def build(tmp_path, monkeypatch, fake_finder):
    from fcpg import v21_j3_pid_service
    import scripts.serve_so101_prod_v17_7 as established
    monkeypatch.setattr(v21_j3_pid_service, 'DEFAULT_SEAL_PATH', tmp_path / 'j3-seal.json')
    monkeypatch.setattr(V21LiveService, '_monitor_q', lambda self: None)
    kinematics = established.SO101MuJoCoKinematics(established.DEFAULT_KINEMATICS_MODEL)
    report = tmp_path / 'fish-cues.npz'
    np.savez(report, report=json.dumps({str(d): dict(reference_px=[300. + d, 250.], picture_hw=[448, 592])
                                        for d in (0, 1, 3)}))
    services = []

    def make(object_relative=True, object_cues=False, object_finder=None, place_goal=False, fish_token=None,
             wrist_drop_zone=None):
        class Owner:
            def __init__(self, root, output, task, device, open_hardware, **kwargs):
                self.stop_event = threading.Event()
                self.q = np.asarray([0., 0., 0., 0., 0., 30.], np.float32)
                vocab = TaskVocabulary.build([task])
                core = V21ResearchCore(V21Config(len(vocab.words), prediction_horizon=40,
                                                 object_relative=object_relative, place_goal=place_goal))
                self.runtime = V21ResearchRuntime(core, vocab.encode([task]), physical_check=lambda q, u: None,
                                                  lease_check=lambda: None)
                self.kinematics = kinematics
                self.frames = {v: np.zeros((480, 640, 3), np.uint8) for v in ('wrist', 'column')}
                self.object_relative = SimpleNamespace(tracker=FakeDino())   # a frozen-feature DINO tracker the owner offers
                self.bus_lock = threading.RLock()
                self.io = SimpleNamespace(camera_lock=threading.Lock(), _last_raw_frames=self.frames)
                self.bridge = V21TransactionBridge(
                    self.runtime, read_q=lambda: self.q.copy(), write_q=lambda c, p: c, capture_endpoint=lambda: None,
                    physical_check=lambda q, u: None, halt=lambda: None, request_stop=self.stop_event.set,
                    cancelled=self.stop_event.is_set)

            def capture(self):
                self.last_rgb = np.zeros((2, 224, 224, 3), np.uint8)
                q = torch.from_numpy(self.q.copy())[None]
                t = torch.tensor([time.monotonic()], dtype=torch.float64)
                obs = VisualObservation(torch.ones(1, 2, 256, 384), q[:, None].expand(-1, 2, -1),
                                        t[:, None].expand(-1, 2), torch.ones(1, 2, dtype=torch.bool),
                                        object_relative=torch.zeros(1, 8))
                self.last_endpoint = EncodedEndpoint(q, t, obs)
                return self.last_endpoint

            def read_checked(self): return self.q.copy()
            def _read_q(self): return self.q.copy()
            def _halt(self): pass
            def images(self, name): pass
            def stop(self): pass
            def close(self): pass
        root = tmp_path / f'model-{len(services)}'
        root.mkdir()
        protocol = dict(object_cues=dict(section='17.30')) if object_cues else {}
        if fish_token is not None:
            protocol['fish_token'] = fish_token
        (root / 'config.json').write_text(json.dumps(dict(protocol=protocol)))
        service = V21LiveService(root, tmp_path / f'out-{len(services)}', device='cpu', owner_factory=Owner,
                                 target_diagnostics=False, wrist_drop_zone=wrist_drop_zone,
                                 object_finder=object_finder or dict(socket=fake_finder.path, word='fish',
                                                                     anchor_score=.5, hold_reference=report))
        services.append(service)
        return service, report
    yield make
    for service in services:
        try:
            service.close()
        except Exception:
            pass


def test_the_jaw_layout_for_a_model_without_the_outline_cue_and_colour_is_gone(build, fake_finder):
    service, _ = build()
    encoder = service.owner.object_relative
    assert isinstance(encoder, finder.Sam3ObjectRelativeEncoder) and encoder.convention == 'jaw'
    assert encoder.mode == 'track' and encoder.prompts == ['fish'] and not hasattr(encoder, '_colour_model')
    record = json.loads((service.output / 'object-finder.json').read_text())
    assert record['encoder']['colour_used'] is False and record['colour_functions'] == 'removed from the code (10/31)'
    import fcpg.v21_object_relative as relative
    assert not hasattr(relative, 'colour_body_geometry') and not hasattr(relative, 'overwrite_pickup_with_colour_body')
    from fcpg.letterbox_geometry import letterbox_rgb_image
    vector = encoder.encode(letterbox_rgb_image(service.owner.frames['wrist']), service.owner.q)
    assert vector[0] == 1. and encoder.last_record['picture_source'] == 'raw_same_exposure'
    assert encoder.score_source == 'dino_pickup' and vector[1] == pytest.approx(.43)   # the colour-era token's score
    assert [h['op'] for h in fake_finder.seen][-1] == 'track'


def test_the_hold_layout_for_a_model_that_learned_the_outline_cue(build):
    service, report = build(object_cues=True)
    encoder = service.owner.object_relative
    assert encoder.convention == 'hold'
    assert np.allclose(encoder.hold_px, finder.demo_hold_reference(report))
    from fcpg.letterbox_geometry import letterbox_rgb_image
    vector = encoder.encode(letterbox_rgb_image(service.owner.frames['wrist']), service.owner.q)
    assert encoder.score_source == 'detection' and vector[1] == pytest.approx(.8)       # the demonstrations' cue score


def test_a_model_without_the_object_token_is_refused(build):
    with pytest.raises(ValueError):
        build(object_relative=False)


def test_the_column_fish_for_the_grasp_target_comes_from_sam3_only(build):
    service, _ = build()
    goal = LivePlaceGoal.__new__(LivePlaceGoal)
    goal.grasp_target, goal.hold, goal.fish_source = True, HoldTracker(), finder.SOURCE
    goal.fish_xy, goal.fish_pixels, goal.fish_times = None, [], []
    goal.camera = SimpleNamespace(fish_point=lambda pixel: np.r_[pixel / 100., 0.])
    service.owner.place_goal = goal
    from fcpg.letterbox_geometry import letterbox_rgb_image
    targets = dict(targets=dict(pickup=dict(column=dict(visible=True, source='operator_colour_body',
                                                         cog_px=[1., 2.]))))
    service._sam3_column_pickup(targets, letterbox_rgb_image(service.owner.frames['column']))
    record = targets['targets']['pickup']['column']
    assert record['source'] == finder.SOURCE and record['visible'] and record['picture_source'] == 'raw_same_exposure'
    for stamp in range(6):
        goal.observe_fish(dict(record, view_source='current_observation', image_time_s=float(stamp)))
    assert goal.fish_xy is not None
    goal.fish_xy, goal.fish_pixels, goal.fish_times = None, [], []
    for stamp in range(6):                                     # a colour record is never taken
        goal.observe_fish(dict(visible=True, source='operator_colour_body', cog_px=[5., 5.],
                               view_source='current_observation', image_time_s=float(stamp)))
    assert goal.fish_xy is None and not goal.fish_pixels


def test_the_learned_finder_reads_the_exposure_tokens_and_never_asks_sam(build, fake_finder, tmp_path):
    """training overhaul doc section 17.34: --object-finder learned, no SAM process, colour still forbidden"""
    from fcpg.v21_object_finder import LearnedObjectRelativeEncoder, ObjectFinder, save
    model = ObjectFinder(conditional=False)
    with torch.no_grad():                               # every cell certain: the whole picture, centre (112, 112)
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(10.)
    path = tmp_path / 'finder.pt'
    save(path, model, fish_reference=np.zeros(384, np.float32))
    before = len(fake_finder.seen)
    service, _ = build(object_finder=dict(kind='learned', model=path, hold_reference=None))
    encoder = service.owner.object_relative
    assert isinstance(encoder, LearnedObjectRelativeEncoder) and encoder.convention == 'jaw'
    service.owner.last_dino_tokens = np.zeros((2, 256, 384), np.float16)
    vector = encoder.encode(np.zeros((224, 224, 3), np.uint8), service.owner.q)
    assert vector[0] == 1. and vector[1] == pytest.approx(.43)          # the colour-era token's score field
    assert encoder.last_record['cog_px'] == pytest.approx([112., 112.]) and encoder.last_record['source'] == \
        'learned_dino_finder'
    assert len(fake_finder.seen) == before                               # no SAM call at all
    record = json.loads((service.output / 'object-finder.json').read_text())
    assert record['kind'] == 'learned' and record['encoder']['sam_used'] is False
    assert record['colour_functions'] == 'removed from the code (10/31)'


def test_a_model_with_the_17_57_fish_token_gets_score_one_and_no_dino_tracker(build, fake_finder, tmp_path):
    """training overhaul doc section 17.57: a model trained on the fish token table (protocol fish_token: 'jaw' layout,
    score 'one') gets the learned finder's token with the score field 1; the DINO tracker is never asked"""
    from fcpg.v21_object_finder import LearnedObjectRelativeEncoder, save

    class NoDino:
        def locate_by_frozen_features(self, *args, **kwargs):
            raise AssertionError('the 17.57 fish token never asks the DINO tracker')
    path = tmp_path / 'finder.pt'
    save(path, certain_finder(False), fish_reference=np.zeros(384, np.float32))
    service, _ = build(object_finder=dict(kind='learned', model=path, hold_reference=None),
                       fish_token=dict(section='17.57', convention='jaw', score_source='one'))
    encoder = service.owner.object_relative
    assert isinstance(encoder, LearnedObjectRelativeEncoder) and encoder.convention == 'jaw'
    assert encoder.score_source == 'one' and encoder.dino_tracker is None
    service.owner.last_dino_tokens = np.zeros((2, 256, 384), np.float16)
    encoder.dino_tracker = NoDino()
    vector = encoder.encode(np.zeros((224, 224, 3), np.uint8), service.owner.q)
    assert vector[0] == 1. and vector[1] == 1. and encoder.last_record['score_source'] == 'one'
    record = json.loads((service.output / 'object-finder.json').read_text())
    assert record['model_convention_source'] == 'config.json protocol.fish_token (section 17.57)'


def test_a_fish_token_table_of_another_kind_is_refused(build, tmp_path):
    from fcpg.v21_object_finder import save
    path = tmp_path / 'finder.pt'
    save(path, certain_finder(False), fish_reference=np.zeros(384, np.float32))
    with pytest.raises(ValueError, match='unknown fish token table'):
        build(object_finder=dict(kind='learned', model=path, hold_reference=None),
              fish_token=dict(convention='jaw', score_source='dino_pickup'))


def test_a_model_with_the_17_57_fish_token_is_not_run_with_sam3(build):
    with pytest.raises(ValueError, match='run with the learned finder'):
        build(fish_token=dict(convention='jaw', score_source='one'))


def certain_finder(conditional):
    """a finder whose every cell is certain: the whole picture, centre (112, 112)"""
    from fcpg.v21_object_finder import ObjectFinder
    model = ObjectFinder(conditional=conditional)
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(10.)
    return model


def test_the_learned_column_finder_gives_the_grasp_target_fish(build, fake_finder, tmp_path):
    """training overhaul doc section 17.51: --object-finder learned with --column-finder-model; the column fish of the
    grasp target comes from the learned column finder on the exposure's own column tokens, never from SAM or colour"""
    from fcpg.v21_object_finder import COLUMN_SOURCE, save
    wrist, column = tmp_path / 'finder.pt', tmp_path / 'column.pt'
    save(wrist, certain_finder(False), fish_reference=np.zeros(384, np.float32))
    save(column, certain_finder(True), fish_reference=np.zeros(384, np.float32),
         column_reference=np.ones(384, np.float32))
    before = len(fake_finder.seen)
    service, _ = build(object_finder=dict(kind='learned', model=wrist, hold_reference=None, column_model=column))
    described = json.loads((service.output / 'object-finder.json').read_text())
    assert described['finder']['column']['source'] == COLUMN_SOURCE
    goal = LivePlaceGoal.__new__(LivePlaceGoal)
    goal.grasp_target, goal.hold, goal.fish_source = True, HoldTracker(), COLUMN_SOURCE
    goal.fish_xy, goal.fish_pixels, goal.fish_times = None, [], []
    goal.camera = SimpleNamespace(fish_point=lambda pixel: np.r_[pixel / 100., 0.])
    service.owner.place_goal = goal
    service.owner.last_dino_tokens = np.ones((2, 256, 384), np.float16)
    targets = dict(targets=dict(pickup=dict(column=dict(visible=True, source='operator_colour_body',
                                                         cog_px=[1., 2.]))))
    service._sam3_column_pickup(targets, np.zeros((224, 224, 3), np.uint8))
    record = targets['targets']['pickup']['column']
    assert record['source'] == COLUMN_SOURCE and record['visible'] and record['cog_px'] == pytest.approx([112., 112.])
    assert record['sam_used'] is False and record['colour_used'] is False
    for stamp in range(6):
        goal.observe_fish(dict(record, view_source='current_observation', image_time_s=float(stamp)))
    assert goal.fish_xy == pytest.approx([1.12, 1.12])
    assert len(fake_finder.seen) == before                               # no SAM call at all
    goal.fish_xy, goal.fish_pixels, goal.fish_times = None, [], []
    service.owner.last_dino_tokens = np.zeros((2, 256, 384), np.float16)  # no column tokens for this exposure
    targets = dict(targets=dict(pickup=dict()))
    service._sam3_column_pickup(targets, np.zeros((224, 224, 3), np.uint8))
    assert targets['targets']['pickup']['column']['visible'] is False


def test_a_column_finder_without_its_column_reference_is_refused(build, tmp_path):
    from fcpg.v21_object_finder import save
    wrist, plain = tmp_path / 'finder.pt', tmp_path / 'plain.pt'
    save(wrist, certain_finder(False), fish_reference=np.zeros(384, np.float32))
    save(plain, certain_finder(True), fish_reference=np.zeros(384, np.float32))        # no column reference
    with pytest.raises(ValueError, match='column-pairs'):
        build(object_finder=dict(kind='learned', model=wrist, hold_reference=None, column_model=plain))


def test_the_column_finder_comes_only_with_the_learned_finder(build, fake_finder, tmp_path):
    from fcpg.v21_object_finder import save
    column = tmp_path / 'column.pt'
    save(column, certain_finder(True), fish_reference=np.zeros(384, np.float32),
         column_reference=np.ones(384, np.float32))
    with pytest.raises(ValueError, match='comes with the learned finder'):           # SAM 3 with the column finder
        build(object_finder=dict(socket=fake_finder.path, word='fish', anchor_score=.5,
                                 hold_reference=tmp_path / 'fish-cues.npz', column_model=column))


def test_a_model_that_reads_the_place_goal_needs_the_column_finder(build, tmp_path):
    from fcpg.v21_object_finder import save
    wrist = tmp_path / 'finder.pt'
    save(wrist, certain_finder(False), fish_reference=np.zeros(384, np.float32))
    with pytest.raises(ValueError, match='column finder'):
        build(object_finder=dict(kind='learned', model=wrist, hold_reference=None), place_goal=True)


def l_finders(tmp_path):
    """a certain wrist finder, column finder and box finder on DINOv3-L tokens (the whole picture, centre (112, 112))"""
    from fcpg.v21_object_finder import ObjectFinder, save

    def certain(conditional):
        model = ObjectFinder(conditional=conditional, encoder='dinov3_vitl16')
        with torch.no_grad():
            model.head[-1].weight.zero_()
            model.head[-1].bias.fill_(10.)
        return model
    wrist, column, box = tmp_path / 'a-l.pt', tmp_path / 'cf-l.pt', tmp_path / 'box-l.pt'
    save(wrist, certain(False), fish_reference=np.zeros(1024, np.float32))
    save(column, certain(True), fish_reference=np.zeros(1024, np.float32), column_reference=np.ones(1024, np.float32))
    save(box, certain(True), wrist_box_reference=np.ones(1024, np.float32),
         column_box_reference=np.ones(1024, np.float32))
    return wrist, column, box


@pytest.fixture
def dinov2_off(monkeypatch):
    """a V21_DINOV2=off, V21_ENCODER=dinov3_vitl16 service process, the forbidden entry points restored afterwards"""
    import fcpg.prod_v18_1_training as backbone
    import fcpg.so101_frozen_vit_grounding as grounding
    import fcpg.v21_live_encoder as live_encoder
    import fcpg.v21_live_service as live
    for name in ('load_frozen_dinov2', 'extract_dinov2_tokens'):
        monkeypatch.setattr(backbone, name, getattr(backbone, name))
    monkeypatch.setattr(grounding.FrozenViTGrounder, '__init__', grounding.FrozenViTGrounder.__init__)
    monkeypatch.setattr(live_encoder, 'DINOV2_OFF', True)
    monkeypatch.setattr(live, 'DINOV2_OFF', True)
    monkeypatch.setattr(live, 'ENCODER', 'dinov3_vitl16')
    return live_encoder


def test_an_all_l_service_reads_only_l_and_every_dinov2_entry_point_raises(build, tmp_path, dinov2_off):
    """training overhaul doc section 17.57: V21_DINOV2=off; the fish token (score 1), the column fish and the box all
    come from learned finders on this exposure's own DINOv3-L tokens; reaching DINOv2 anywhere raises"""
    from fcpg.v21_live_service import exposure_tokens
    from fcpg.v21_object_finder import BOX_SOURCE
    wrist, column, box = l_finders(tmp_path)
    service, _ = build(object_finder=dict(kind='learned', model=wrist, hold_reference=None, column_model=column,
                                          box_model=box),
                       fish_token=dict(section='17.57', convention='jaw', score_source='one'))
    record = json.loads((service.output / 'object-finder.json').read_text())
    assert record['dinov2_off'] is True and len(record['dinov2_functions_forbidden']) == 3
    owner = service.owner
    owner.last_dino_tokens = None
    owner.last_model_tokens = np.ones((2, 196, 1024), np.float16)
    vector = owner.object_relative.encode(np.zeros((224, 224, 3), np.uint8), owner.q)
    assert vector[0] == 1. and vector[1] == 1.
    with pytest.raises(ValueError, match='V21_DINOV2=off'):
        exposure_tokens(owner, 'dinov2_vits14')
    import fcpg.prod_v18_1_training as backbone
    from fcpg.so101_frozen_vit_grounding import FrozenViTGrounder
    with pytest.raises(dinov2_off.DinoV2Off):
        backbone.load_frozen_dinov2('x', 'y', device='cpu')
    with pytest.raises(dinov2_off.DinoV2Off):
        FrozenViTGrounder(device='cpu')
    targets = service._learned_targets(dict(wrist='current_observation', column='current_observation'))
    drop, pickup = targets['targets']['drop_zone'], targets['targets']['pickup']
    for view in ('wrist', 'column'):
        assert drop[view]['visible'] is True and drop[view]['source'] == BOX_SOURCE
        assert drop[view]['cog_px'] == pytest.approx([112., 112.]) and drop[view]['role'] == 'drop_zone'
    assert pickup['wrist']['visible'] is True and pickup['wrist']['cog_px'] == pytest.approx([112., 112.])
    assert pickup['column']['visible'] is True
    hidden = service._learned_targets(dict(wrist='unavailable', column='historical_observation_not_live'))
    assert not hidden['targets']['drop_zone']['wrist']['visible'] and not hidden['targets']['pickup']['column']['visible']


def test_an_all_l_service_needs_the_box_finder_and_the_17_57_fish_token(build, tmp_path, dinov2_off):
    wrist, column, box = l_finders(tmp_path)
    with pytest.raises(ValueError, match='learned box finder'):
        build(object_finder=dict(kind='learned', model=wrist, hold_reference=None),
              fish_token=dict(section='17.57', convention='jaw', score_source='one'))


def test_an_all_l_service_refuses_the_colour_era_score(build, tmp_path, dinov2_off):
    wrist, column, box = l_finders(tmp_path)
    with pytest.raises(ValueError, match='frozen DINOv2'):
        build(object_finder=dict(kind='learned', model=wrist, hold_reference=None, box_model=box))


def test_the_box_finder_replaces_the_wrist_registration(build, tmp_path):
    wrist, column, box = l_finders(tmp_path)
    with pytest.raises(ValueError, match='replaces the wrist drop-zone registration'):
        build(object_finder=dict(kind='learned', model=wrist, hold_reference=None, box_model=box),
              fish_token=dict(section='17.57', convention='jaw', score_source='one'),
              wrist_drop_zone=tmp_path / 'any.npz')
