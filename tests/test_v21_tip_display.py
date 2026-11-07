from types import SimpleNamespace

import numpy as np
import torch

from fcpg.v21_tip_display import (
    TipDisplay, column_pair_hits_sensor_edge, order_column_pair_from_q_geometry,
    tip_record,
)
from fcpg.v21_runtime import EncodedEndpoint
from fcpg.v21_memory import VisualObservation
from fcpg.so101_camera_calibration import ProjectionFit


def test_tip_record_has_explicit_non_policy_source_and_no_offscreen_clipping():
    good = tip_record([[50,90],[80,90]], 'test-reference', 12., np.zeros(6))
    assert good['available'] and good['display_only'] and not good['control_input']
    assert not good['accuracy_verified']
    assert good['coordinate_space'] == 'letterbox224'
    for points in ([[50,5],[80,90]], [[50,90],[250,90]], [[50,90],[50,90]], [[float('nan'),90],[80,90]]):
        bad = tip_record(points,'test-reference',12.,np.zeros(6))
        assert not bad['available'] and bad['fm_px'] is None


def fixture():
    display = TipDisplay.__new__(TipDisplay)
    display.device = torch.device('cpu'); display.raw_shape = (480,640)
    display.legacy_root = '/unused-test-model'
    from pathlib import Path
    display.legacy_root = Path(display.legacy_root)
    calls = []
    def frames(q):
        calls.append(('wrist',q.copy()))
        world=np.zeros((1,3,3)); local=world.copy()
        local[0,1,0]=.6; local[0,2,0]=1.
        return world,local
    display.wrist_kinematics=SimpleNamespace(forward_anchor_frames=frames)
    coefficients=np.zeros((5,2));coefficients[0,0]=100.;coefficients[4,1]=110.
    display.wrist_fit=ProjectionFit('wrist',('x_gripper','y_gripper','z_gripper','opening','bias'),coefficients,0.)
    display._fk_m_px=lambda view,q: np.array([100.,110.],dtype=np.float64)
    def prior(q):
        calls.append(('column',q.clone()));return object()
    display.geometry=SimpleNamespace(make_prior=prior)
    display._load_column=lambda:None
    class Observer:
        evidence_threshold=.5
        def __call__(self,rgb,dino,valid,prior,*,q_position,actual_history,mechanical_prior):
            assert rgb.shape == (1,2,3,224,224)
            weights=torch.ones(1,2,4);weights[0,1,2]=2
            pairs=torch.tensor([[[[40.,60.],[50.,60.]],[[41.,60.],[51.,60.]],
                                 [[42.,60.],[52.,60.]],[[43.,60.],[53.,60.]]]])
            return SimpleNamespace(candidate_belief=SimpleNamespace(weights=weights,
                visible_probability=torch.ones(1,2,4,2)),raw_column_fm_px=pairs)
    display.observer=Observer()
    q_at=torch.stack((torch.arange(6),torch.arange(6)+10)).float()[None]
    obs=VisualObservation(torch.ones(1,2,256,384),q_at,
                          torch.tensor([[1.,1.02]],dtype=torch.float64),torch.ones(1,2,dtype=torch.bool))
    endpoint=EncodedEndpoint(torch.full((1,6),99.),torch.tensor([1.03],dtype=torch.float64),obs)
    return display,calls,endpoint


def test_each_view_uses_its_image_time_q_not_the_later_endpoint_q(monkeypatch):
    d,calls,e=fixture()
    pixels=np.zeros((2,224,224,3),np.uint8)
    raw={v:np.zeros((480,640,3),np.uint8) for v in ('wrist','column')}
    visual_calls = []
    def visual(image, proposal, *, image_space):
        visual_calls.append((image, proposal.copy(), image_space))
        return dict(available=True, fm_px=[[44.,65.],[56.,66.]], reason=None)
    monkeypatch.setattr('fcpg.v21_tip_display.locate_column_jaw_tips', visual)
    before=e.observation.features.clone()
    tips=d.observe(pixels,e,raw,None)
    # Wrist F/G/M come from FK+calibration at the exposure q (display only, unverified).
    w=tips['wrist']
    assert w['source'] == 'wrist-fk-projection' and w['display_only'] and not w['control_input']
    assert w['fk_fm_px'] is not None and w['fk_m_px'] is not None and w['fk_based']
    assert not w['accuracy_verified']
    if w['available']:
        assert w['fm_px'] == w['fk_fm_px']
        np.testing.assert_allclose(w['m_px'], np.mean(w['fm_px'], axis=0))
    else:
        assert 'out-of-frame' in (w['reason'] or '')
    assert tips['column']['available']
    np.testing.assert_array_equal(calls[0][1][0],np.arange(6))
    torch.testing.assert_close(calls[1][1],torch.arange(6)[None].float()+10)
    assert tips['wrist']['image_time_s'] == 1.
    assert tips['column']['image_time_s'] == 1.02
    assert tips['column']['source'] == 'column-rgb-jaw-contour'
    assert tips['column']['legacy_proposal']['component'] == 2
    assert tips['column']['legacy_proposal']['pair_px'] == [[42.,60.],[52.,60.]]
    assert tips['column']['fm_px'] == [[42.,60.],[52.,60.]]
    assert tips['column']['m_px'] == [47.,60.]
    assert tips['column']['visual_diagnostics']['dino_fg_trusted'] is True
    assert tips['column']['unordered'] and not tips['column']['ordered_identity_available']
    assert not tips['column']['physical_identity_verified']
    assert not tips['column']['fk_based']
    assert len(visual_calls) == 1 and visual_calls[0][0] is raw['column']
    assert visual_calls[0][2] == 'raw'
    np.testing.assert_array_equal(visual_calls[0][1], [[42.,60.],[52.,60.]])
    assert torch.equal(e.observation.features,before)


def test_blank_column_image_rejects_high_visibility_old_pair():
    d,_,e=fixture()
    raw={v:np.zeros((480,640,3),np.uint8) for v in ('wrist','column')}
    tips=d.observe(np.zeros((2,224,224,3),np.uint8),e,raw,None)
    c=tips['column']
    assert c['source'] == 'column-rgb-jaw-contour'
    assert c['legacy_proposal']['visibility'] == [1.,1.]
    assert c['available'] and c['fm_px'] == [[42.,60.],[52.,60.]]
    assert c['visual_diagnostics']['dino_fg_trusted'] is True
    assert c['fk_m_px'] is None and not c['fk_based']
    assert c['unordered'] and not c['ordered_identity_available']


def test_column_rgb_endpoints_are_ordered_by_column_q_geometry_only(monkeypatch):
    d,_,e=fixture()
    pixels=np.zeros((2,224,224,3),np.uint8)
    raw={v:np.zeros((480,640,3),np.uint8) for v in ('wrist','column')}
    rgb_pair=np.asarray([[22.4,104.3],[44.1,107.1]])
    projected_fg=np.asarray([[113.3,96.6],[74.3,97.7]])
    monkeypatch.setattr('fcpg.v21_tip_display.locate_column_jaw_tips',
        lambda *args,**kwargs:dict(available=True,fm_px=rgb_pair.tolist(),reason=None))
    d._column_identity_reference=lambda q:(projected_fg.copy(),
        dict(available=True,source='column-q-fk-projection'))

    column=d.observe(pixels,e,raw,None)['column']

    # The projected pair only orders F and G. The coordinates are DINO's pair.
    dino_pair=np.asarray([[42.,60.],[52.,60.]])
    got=np.asarray(column['fm_px'],np.float64)
    assert got.shape==(2,2) and (np.allclose(got,dino_pair) or np.allclose(got,dino_pair[::-1]))
    assert column['ordered_identity_available'] and not column['unordered']
    assert column['identity_order_fk_based'] and not column['pixel_position_fk_based']
    assert not column['fk_based'] and not column['physical_identity_verified']
    assert column['identity_assignment']['swapped']
    assert column['identity_assignment']['projected_coordinates_used'] is False
    np.testing.assert_array_equal(column['visual_diagnostics']['contour_fm_px'],rgb_pair)
    np.testing.assert_array_equal(column['visual_diagnostics']['fm_px'],[[42.,60.],[52.,60.]])


def test_column_q_geometry_fails_closed_when_axis_is_orthogonal():
    pair=np.asarray([[22.4,104.3],[44.1,107.1]])
    midpoint=pair.mean(0)
    observed=pair[1]-pair[0]
    orthogonal=np.asarray([-observed[1],observed[0]])
    reference=np.stack((midpoint-.5*orthogonal,midpoint+.5*orthogonal))

    ordered,diagnostic=order_column_pair_from_q_geometry(pair,reference)

    np.testing.assert_array_equal(ordered,pair)
    assert not diagnostic['available']
    assert abs(diagnostic['alignment']) < 1e-12
    assert 'orthogonal' in diagnostic['reason']


def test_column_q_geometry_rejects_current_mount_pair_but_accepts_jaw_pair():
    # Current scene: the mounting fragments have only 0.608 alignment with the
    # Column FK axis, while the actual jaw-tip pair has 0.984 alignment.
    reference=np.asarray([[50.,100.],[20.,102.05]])
    mount_pair=np.asarray([[22.75,66.85],[12.95,55.65]])
    jaw_pair=np.asarray([[44.39,104.33],[16.3,101.2]])

    _,mount_diagnostic=order_column_pair_from_q_geometry(mount_pair,reference)
    ordered_jaw,jaw_diagnostic=order_column_pair_from_q_geometry(jaw_pair,reference)

    assert 0.60 < abs(mount_diagnostic['alignment']) < 0.62
    assert not mount_diagnostic['available']
    assert jaw_diagnostic['alignment'] > 0.98 and jaw_diagnostic['available']
    np.testing.assert_array_equal(ordered_jaw,jaw_pair)


def test_complete_column_fk_prior_uses_exposure_readback_and_never_creates_rgb_tips(monkeypatch):
    d,_,e=fixture()
    frames=d.wrist_kinematics.forward_anchor_frames
    def world_frames(q):
        world,local=frames(q)
        world[0,1]=[.06 + .001*q[0,0],.10,.0]
        world[0,2]=[.09,.10 + .001*q[0,4],.001*q[0,5]]
        world[0,0]=world[0,1:3].mean(0)
        return world,local
    d.wrist_kinematics=SimpleNamespace(forward_anchor_frames=world_frames)
    d.column_fit=ProjectionFit('column',('x_base','y_base','z_base','bias'),
        np.asarray([[500.,0.],[0.,500.],[0.,-200.],[50.,70.]]),0.)
    captured=[]
    def missing_visual(image,proposal,*,image_space,pose_prior):
        captured.append(pose_prior)
        return dict(available=False,fm_px=None,reason='one unresolved silhouette')
    monkeypatch.setattr('fcpg.v21_tip_display.locate_column_jaw_tips',missing_visual)
    raw={v:np.zeros((480,640,3),np.uint8) for v in ('wrist','column')}
    result=d.observe(np.zeros((2,224,224,3),np.uint8),e,raw,None)
    column=result['column']
    assert len(captured)==1
    np.testing.assert_array_equal(captured[0].q_at_image,e.observation.q_at_image[0,1])
    assert column['pose_prior']['full_joint_fk']
    assert column['pose_prior']['image_time_s']==1.02
    assert column['pose_prior']['inferred'] and not column['pose_prior']['observed']
    assert column['available'] and column['fm_px'] == [[42.,60.],[52.,60.]]
    assert result['wrist']['source']=='wrist-fk-projection'


def test_tip_record_m_is_fg_midpoint():
    rec = tip_record([[40.,80.],[60.,80.]], 'test', 1., np.zeros(6), fk_m_px=[51.,79.])
    # On-screen M follows the drawn F–G segment, not a separate FK pixel.
    assert rec['m_px'] == [50.,80.] and rec['fk_m_px'] == [51.,79.]
    assert rec['fm_px'][1] == [60.,80.]  # moving tip stays G, not M
    assert rec['fk_disagreement_px'] is not None and rec['fk_disagreement_px'] < 2.0
    assert rec['accuracy_verified'] is True
    only = tip_record(None, 'test', 1., np.zeros(6), fk_m_px=[90.,100.])
    # FK alone must never become available M (not ground truth).
    assert not only['available'] and only['m_px'] is None and only['fm_px'] is None
    assert only['fk_m_px'] == [90.,100.]
    far = tip_record([[40.,80.],[60.,80.]], 'test', 1., np.zeros(6), fk_m_px=[100.,80.])
    assert far['available'] and far['m_px'] == [50.,80.]
    assert far['fk_disagreement_px'] == 50.0 and far['accuracy_verified'] is False


def test_missing_view_never_reuses_old_tip_or_nominal_fallback():
    d,calls,e=fixture();e.observation.measurement_available.zero_()
    tips=d.observe(np.zeros((2,224,224,3),np.uint8),e,{},None)
    assert calls == []
    assert all(not record['available'] and record['fm_px'] is None for record in tips.values())
    # FK diagnostic may remain, but must not fabricate M.
    assert all(record.get('m_px') is None for record in tips.values())


def test_missing_dino_target_uses_fcpg_retained_fg(monkeypatch):
    from fcpg.v21_column_temporal_prior import ColumnTemporalPrior
    prior = ColumnTemporalPrior(dict(source='test-completed-forecast',
        predicted_pair_px=[[90., 143.], [124., 143.]], predicted_midpoint_px=[107., 143.],
        progress_session_id='one', target_readback_index=5,
        method_label='previous actual RGB + LeWM/FK candidate prior'))
    calls = []
    def absent(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs['temporal_prior'] is prior
        return dict(available=False, fm_px=None, reason='no supported current pair')
    monkeypatch.setattr('fcpg.v21_tip_display.locate_column_jaw_tips', absent)
    for has_frame in (True, False):
        display, _, endpoint = fixture()
        endpoint.observation.measurement_available[0, 1] = has_frame
        features = endpoint.observation.features.clone()
        raw = {v: np.zeros((480, 640, 3), np.uint8) for v in ('wrist', 'column')}
        records = display.observe(np.zeros((2, 224, 224, 3), np.uint8), endpoint, raw, None,
                                  column_temporal_prior=prior)
        column = records['column']
        if has_frame:
            assert column['available'] and column['observed']
            assert column['fm_px'] == [[42., 60.], [52., 60.]]
            assert column['visual_diagnostics']['dino_fg_trusted'] is True
            continue
        assert column['available'] and not column['observed']
        assert column['source'] == 'fcpg-retained-column-fg'
        assert column['fm_px'] == [[90., 143.], [124., 143.]]
        assert column['m_px'] == [107., 143.]
        assert column['display_only'] and not column['control_input']
        assert not column['actual_visual_memory_write']
        suggestion = column['prediction_suggestion']
        assert not suggestion['available'] and not suggestion['observed']
        assert suggestion['current_frame_available'] is has_frame
        assert suggestion['display_only'] and not suggestion['control_input']
        assert not suggestion['actual_visual_memory_write']
        assert not suggestion['fcpg_pixel_prediction_available']
        assert suggestion['target_readback_index'] == 5
        assert records['wrist']['available'] and records['wrist']['source'] == 'wrist-fk-projection'
        torch.testing.assert_close(endpoint.observation.features, features)
    assert len(calls) == 1  # no new frame cannot be treated as a visual measurement


def test_current_rgb_pair_is_kept_even_when_temporal_suggestion_disagrees(monkeypatch):
    from fcpg.v21_column_temporal_prior import ColumnTemporalPrior
    display, _, endpoint = fixture()
    prior = ColumnTemporalPrior(dict(source='test-completed-forecast',
        predicted_pair_px=[[90., 143.], [124., 143.]], predicted_midpoint_px=[107., 143.]))
    monkeypatch.setattr('fcpg.v21_tip_display.locate_column_jaw_tips',
        lambda *args, **kwargs: dict(available=True, fm_px=[[44., 80.], [63., 80.]], reason=None))
    raw = {v: np.zeros((480, 640, 3), np.uint8) for v in ('wrist', 'column')}
    column = display.observe(np.zeros((2, 224, 224, 3), np.uint8), endpoint, raw, None,
                             column_temporal_prior=prior)['column']
    assert column['available'] and column['observed']
    assert column['fm_px'] == [[42., 60.], [52., 60.]]
    assert column['suggested_fm_px'] is None and column['prediction_suggestion'] is None


def test_sensor_edge_column_pair_is_not_an_observation(monkeypatch):
    visual = dict(available=True, fm_px=[[28.35, 129.15], [0.5, 123.55]], reason=None,
        near_sensor_edge=True, selected_component_indices=[1, 0],
        candidates=[
            dict(tip_px=[0.5, 123.55], sensor_edge_distance_px=0.5, near_sensor_edge=True),
            dict(tip_px=[28.35, 129.15], sensor_edge_distance_px=28.35),
        ])
    assert column_pair_hits_sensor_edge(visual)
    assert not column_pair_hits_sensor_edge(dict(available=True, fm_px=[[44., 65.], [56., 66.]]))
    display, _, endpoint = fixture()
    monkeypatch.setattr('fcpg.v21_tip_display.locate_column_jaw_tips',
        lambda *args, **kwargs: visual)
    raw = {v: np.zeros((480, 640, 3), np.uint8) for v in ('wrist', 'column')}
    column = display.observe(np.zeros((2, 224, 224, 3), np.uint8), endpoint, raw, None)['column']
    assert column['available'] and column['fm_px'] == [[42., 60.], [52., 60.]]
    assert column['visual_diagnostics']['dino_fg_trusted'] is True


def test_visible_column_tips_near_edge_survive_display_admission(monkeypatch):
    """Real stopped-camera counterexample to the extra 12px exclusion band.

    This checks admission of already supported image endpoints, not a labeled
    accuracy benchmark or proof of physical grasping.
    """
    from pathlib import Path
    import cv2
    from fcpg.v21_column_tip_visual import locate_column_jaw_tips

    image = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / 'fixtures' /
        'column_visible_tips_near_edge_20260922.png')), cv2.COLOR_BGR2RGB)
    proposal = [[29.197704315185547, 93.77286529541016],
                [4.382122993469238, 89.11112213134766]]
    visual = locate_column_jaw_tips(image, proposal, image_space='raw')
    assert visual['available'] and not visual['near_sensor_edge']
    edge = min(visual['candidates'][i]['sensor_edge_distance_px']
               for i in visual['selected_component_indices'])
    assert 5. < edge < 12.
    assert not column_pair_hits_sensor_edge(visual)

    display, _, endpoint = fixture()
    monkeypatch.setattr('fcpg.v21_tip_display.locate_column_jaw_tips',
                        lambda *args, **kwargs: visual)
    raw = {'wrist': np.zeros_like(image), 'column': image}
    column = display.observe(np.zeros((2, 224, 224, 3), np.uint8), endpoint, raw, None)['column']
    assert column['available'] and column['observed']
    np.testing.assert_array_equal(column['fm_px'], [[42., 60.], [52., 60.]])
    assert column['accuracy_verified'] is False
