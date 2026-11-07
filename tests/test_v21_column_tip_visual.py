import cv2
import numpy as np
import pytest
from pathlib import Path

from fcpg.v21_column_tip_visual import locate_column_jaw_tips
from fcpg.letterbox_geometry import letterbox_rgb_image


PROPOSAL = np.array([[94.,141.],[120.,141.]])


def scene(*, jaws=(90,124), blade_color=(60,90,130), pale=True, blade_halfwidth=5):
    image=np.full((224,224,3),40,np.uint8)
    image[:28]=0;image[196:]=0
    for x in jaws:
        if pale:
            cv2.rectangle(image,(x-8,89),(x+8,104),(180,210,225),-1)
        cv2.fillConvexPoly(image,np.array([[x-blade_halfwidth,102],
            [x+blade_halfwidth,102],[x,143]]),blade_color)
    return image


def detect(image, proposal=PROPOSAL):
    return locate_column_jaw_tips(image,proposal,image_space='letterbox224')


def test_pair_is_supported_by_image_not_the_proposal_and_does_not_mutate_inputs():
    image=scene(); before=image.copy(); proposal=PROPOSAL.copy()
    result=detect(image,proposal)
    assert result['available']
    np.testing.assert_allclose(result['fm_px'],[[90,143],[124,143]],atol=1.)
    assert result['fm_px'] != proposal.tolist()
    assert result['unordered'] and not result['ordered_identity_available']
    assert not result['physical_identity_verified'] and not result['accuracy_verified']
    assert not result['fk_used'] and not result['proposal_used_as_endpoint']
    np.testing.assert_array_equal(image,before)
    np.testing.assert_array_equal(proposal,PROPOSAL)


@pytest.mark.parametrize('angle,dx,dy', [(0,12,-8),(25,8,-4),(-35,-5,-8),(165,0,-9)])
def test_rotation_translation_follow_the_image(angle,dx,dy):
    affine=cv2.getRotationMatrix2D((107,120),angle,1.)
    affine[:,2]+=[dx,dy]
    move=lambda points: np.asarray(points)@affine[:,:2].T+affine[:,2]
    image=cv2.warpAffine(scene(),affine,(224,224),flags=cv2.INTER_NEAREST)
    result=detect(image,move(PROPOSAL))
    assert result['available'], result['reason']
    np.testing.assert_allclose(result['fm_px'],move([[90,143],[124,143]]),atol=2.)
    assert result['unordered']


def test_pair_order_is_only_proposal_association_not_image_left_right_identity():
    forward=detect(scene())
    reverse=detect(scene(),PROPOSAL[::-1])
    np.testing.assert_array_equal(reverse['fm_px'],forward['fm_px'][::-1])
    assert reverse['unordered'] and not reverse['ordered_identity_available']


@pytest.mark.parametrize('image', [np.zeros((224,224,3),np.uint8),scene(jaws=(90,)),
    scene(blade_color=(200,80,40)),scene(blade_color=(110,85,60)),scene(pale=False)])
def test_absent_or_wrong_appearance_jaws_do_not_return_proposal(image):
    result=detect(image)
    assert not result['available'] and result['fm_px'] is None


def test_three_equally_supported_jaws_are_ambiguous():
    result=detect(scene(jaws=(80,107,134)),[[80,141],[134,141]])
    assert not result['available']
    assert 'ambiguous' in result['reason']


def test_raw_and_explicit_letterbox_use_the_same_coordinates():
    image=scene()
    expected=detect(image)
    raw=image[28:196].copy()
    actual=locate_column_jaw_tips(raw,PROPOSAL)
    np.testing.assert_array_equal(actual['fm_px'],expected['fm_px'])
    raw640=cv2.resize(raw,(640,480),interpolation=cv2.INTER_NEAREST)
    actual640=locate_column_jaw_tips(raw640,PROPOSAL)
    assert actual640['available']
    np.testing.assert_allclose(actual640['fm_px'],expected['fm_px'],atol=1.1)


@pytest.mark.parametrize('image,proposal,space', [
    (np.zeros((31,40,3),np.uint8),PROPOSAL,'raw'),
    (np.zeros((224,224),np.uint8),PROPOSAL,'letterbox224'),
    (scene().astype(float),PROPOSAL,'letterbox224'),
    (np.zeros((480,800,3),np.uint8),PROPOSAL,'raw'),
    (scene(),[[10,10],[20,20]],'letterbox224'),
    (scene(),[[80,140],[80,140]],'letterbox224'),
    (scene(),[[np.nan,140],[90,140]],'letterbox224'),
    (scene(),[[80,140]],'letterbox224'),
    (scene(),PROPOSAL,'unknown')])
def test_invalid_image_or_proposal_is_unavailable(image,proposal,space):
    result=locate_column_jaw_tips(image,proposal,image_space=space)
    assert not result['available'] and result['fm_px'] is None


def test_internal_roi_cut_is_not_an_endpoint():
    result=detect(scene(),[[98,98],[116,98]])
    assert not result['available']


def test_sensor_edge_proximity_is_explicit():
    affine=np.array([[1.,0.,-89.],[0.,1.,0.]])
    image=cv2.warpAffine(scene(),affine,(224,224),flags=cv2.INTER_NEAREST)
    result=detect(image,PROPOSAL-[89,0])
    assert result['available']
    assert result['near_sensor_edge']
    assert not result['accuracy_verified']


@pytest.mark.parametrize('dx,dy', [(0,-6),(-1,-6),(0,-4),(2,-5)])
def test_high_contrast_interior_rib_does_not_pull_tip_toward_root(dx,dy):
    # A dark 2x2 texture lies inside the still connected distal blade. It has
    # a stronger image corner than the silhouette but is not the blade end.
    raw=cv2.resize(scene(blade_halfwidth=7)[28:196],(640,480),
                   interpolation=cv2.INTER_NEAREST)
    clean=locate_column_jaw_tips(raw,PROPOSAL)
    for x in (90,124):
        rx,ry=round(x/.35),round((143-28)/.35)
        cv2.rectangle(raw,(rx+dx,ry+dy),(rx+dx+1,ry+dy+1),(0,0,0),-1)
    textured=locate_column_jaw_tips(raw,PROPOSAL)
    assert clean['available'] and textured['available']
    np.testing.assert_allclose(textured['fm_px'],clean['fm_px'],atol=.75)


@pytest.mark.parametrize('root_color', [(165,196,217),(170,198,215),(180,210,220)])
def test_blue_tinted_shared_pale_housing_does_not_merge_the_two_blades(root_color):
    image=scene()
    # Both the blade and root colour predicates accept this shared plastic
    # housing. It must provide root support without joining the two blades.
    cv2.rectangle(image,(82,89),(132,101),root_color,-1)
    result=detect(image)
    assert result['available'], result['reason']
    np.testing.assert_allclose(result['fm_px'],[[90,143],[124,143]],atol=1.)
    assert not result['fk_used'] and not result['proposal_used_as_endpoint']


def test_recorded_column_frame_with_shared_pale_housing_recovers_visible_tips():
    # Exact raw Column exposure, image_time_s=166515.110347475, captured in
    # scratch-live-20260920/control/current-observation.npz. The old detector
    # joined the visible blades into one component and returned candidates=[].
    path=Path(__file__).parent/'fixtures'/'column_shared_pale_housing_20260920.png'
    raw=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    proposal=np.array([[47.88665771484375,103.39340209960938],
                       [20.952238082885742,97.25822448730469]])
    result=locate_column_jaw_tips(raw,proposal)
    assert result['available'], result['reason']
    # Manually reviewed distal silhouettes in the raw image, in letterbox224.
    # The tolerance reflects visual annotation rather than metric accuracy.
    expected=np.array([[36.,109.],[14.,104.]])
    np.testing.assert_allclose(result['fm_px'],expected,atol=1.2)
    assert len(result['selected_component_indices'])==2
    assert not result['fk_used'] and not result['proposal_used_as_endpoint']
    assert result['unordered'] and not result['physical_identity_verified']


@pytest.mark.parametrize('angle', [0, 30, 165])
def test_pale_side_background_does_not_contaminate_attached_root(angle):
    image=scene()
    for x in (90,124):
        cv2.rectangle(image,(x+4,117),(x+12,134),(180,210,225),-1)
    affine=cv2.getRotationMatrix2D((107,120),angle,1.)
    move=lambda points: np.asarray(points)@affine[:,:2].T+affine[:,2]
    image=cv2.warpAffine(image,affine,(224,224),flags=cv2.INTER_NEAREST)
    result=detect(image,move(PROPOSAL))
    assert result['available'], result['reason']
    np.testing.assert_allclose(result['fm_px'],move([[90,143],[124,143]]),atol=2.)
    assert all(len(result['candidates'][i]['root_pale_component_labels'])==1
               for i in result['selected_component_indices'])


def test_pale_side_background_alone_is_not_a_gripper_housing():
    image=scene(pale=False)
    for x in (90,124):
        cv2.rectangle(image,(x+4,117),(x+12,134),(180,210,225),-1)
    result=detect(image)
    assert not result['available'] and result['fm_px'] is None


def test_opposite_end_housings_are_ambiguous_not_an_oriented_tip():
    image=scene()
    for x in (90,124):
        cv2.rectangle(image,(x-8,144),(x+8,159),(180,210,225),-1)
    result=detect(image)
    assert not result['available'] and result['fm_px'] is None


def test_one_ambiguous_root_is_resolved_by_an_unambiguous_parallel_blade():
    image=scene()
    # Only the left blade has pale support at both ends. The right blade is an
    # unambiguous anchor with the same distal direction and terminal height.
    cv2.rectangle(image,(82,144),(98,159),(180,210,225),-1)
    result=detect(image)
    assert result['available'], result['reason']
    np.testing.assert_allclose(result['fm_px'],[[90,143],[124,143]],atol=1.)
    selected=[result['candidates'][i] for i in result['selected_component_indices']]
    assert sorted(item['root_support_ambiguous'] for item in selected)==[False,True]
    assert result['pair_candidates'][0]['root_ambiguous_count']==1


def test_recorded_column_bright_table_preserves_two_visible_jaw_endpoints():
    # Exact post-execution raw Column exposure at 170603.479571202, saved
    # from expanded-action-deployment/control/current-observation.npz. Pale
    # tabletop pixels previously moved one blade's root centroid to mid-shaft.
    path=Path(__file__).parent/'fixtures'/'column_bright_table_root_20260920.png'
    raw=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    before=raw.copy()
    proposal=np.array([[49.79882049560547,105.2317123413086],
                       [23.638015747070312,97.1611099243164]])
    result=locate_column_jaw_tips(raw,proposal)
    assert result['available'], result['reason']
    # Manual image annotation; this tolerance is not a metric accuracy claim.
    np.testing.assert_allclose(result['fm_px'],[[37.5,107.5],[16.,103.]],atol=1.2)
    selected=[result['candidates'][i] for i in result['selected_component_indices']]
    assert len({item['label'] for item in selected})==2
    for item in selected:
        away=np.asarray(item['tip_px'])-item['root_px']
        assert float(away@item['distal_axis'])>15.
        assert len(item['root_pale_component_labels'])==1
    assert not result['fk_used'] and not result['proposal_used_as_endpoint']
    assert result['unordered'] and not result['physical_identity_verified']
    np.testing.assert_array_equal(raw,before)
    # The same background plus only one real blade cannot supply a pair.
    cv2.rectangle(raw,(100,152),(160,232),(180,150,120),-1)
    missing=locate_column_jaw_tips(raw,proposal)
    assert not missing['available'] and missing['fm_px'] is None


def test_pale_side_pixels_connected_to_housing_do_not_become_root_support():
    image=scene()
    cv2.rectangle(image,(82,65),(132,101),(180,210,225),-1)
    for x in (90,124):
        # The root and side background are one connected pale component.
        cv2.rectangle(image,(x+5,98),(x+7,144),(180,210,225),-1)
    result=detect(image,[[94,130],[120,130]])
    assert result['available'], result['reason']
    np.testing.assert_allclose(result['fm_px'],[[90,143],[124,143]],atol=1.)
    selected=[result['candidates'][i] for i in result['selected_component_indices']]
    assert all(item['tip_root_distance_px']>35 for item in selected)


def test_recorded_connected_pale_background_requires_same_frame_association():
    # Exact raw Column exposure image_time_s=9439.176371708 from
    # continue-arm-partial-position-servo-deployment/control/current-observation.npz.
    # The right blade's pale surroundings were connected to its housing;
    # a thin cool tabletop streak also joined the left blade's colour mask.
    path=Path(__file__).parent/'fixtures'/'column_connected_pale_background_20260920.png'
    raw=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    before=raw.copy()
    proposal=[[50.239906311035156,97.67747497558594],
              [26.893829345703125,94.22413635253906]]
    stale=locate_column_jaw_tips(raw,proposal)
    assert not stale['available'] and stale['fm_px'] is None
    assert stale['proposal_association_rejections']
    assert all(row['endpoint_association_px']>row['maximum_endpoint_association_px']
               for row in stale['proposal_association_rejections'])

    nearby=[[37.,105.],[13.,101.]]
    result=locate_column_jaw_tips(raw,nearby)
    assert result['available'], result['reason']
    # Manual distal silhouette annotation in letterbox224, not metric truth.
    np.testing.assert_allclose(result['fm_px'],[[34.3,108.5],[10.,103.]],atol=1.)
    assert not result['fk_used'] and not result['proposal_used_as_endpoint']
    assert result['unordered'] and not result['physical_identity_verified']
    np.testing.assert_array_equal(raw,before)
    # Separate inner and outer colour regions on one blade must not become
    # an invented pair when either real blade is absent.
    for x0,y0,x1,y1 in ((0,146,92,238),(92,149,160,242)):
        missing=raw.copy()
        missing[y0:y1,x0:x1]=[180,150,120]
        absent=locate_column_jaw_tips(missing,nearby)
        assert not absent['available'] and absent['fm_px'] is None


def test_recorded_open_gripper_rejects_two_fragments_from_one_blade():
    # Exact raw Column exposure at image_time_s=13571.603770622 with the
    # gripper read back at J6=31.1432.  Before the transverse-pair check,
    # components 2 and 3 on the right blade were returned as both jaws.
    path=Path(__file__).parent/'fixtures'/'column_open_gripper_fragments_20260920.png'
    raw=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    proposal=[[47.71640396118164,96.54998016357422],
              [19.859256744384766,92.40576934814453]]
    result=locate_column_jaw_tips(raw,proposal)
    assert result['available'], result['reason']
    np.testing.assert_allclose(result['fm_px'],
                               [[33.95,106.40],[11.55,101.50]],atol=.8)
    selected=[result['candidates'][i] for i in result['selected_component_indices']]
    assert len({candidate['label'] for candidate in selected})==2
    selected_pair=result['pair_candidates'][0]
    assert selected_pair['transverse_separation_px']>20.
    assert not result['fk_used'] and result['unordered']
    assert not result['physical_identity_verified']


def test_current_observation_overexposure_ignores_stale_proposal_coordinates():
    """The live overexposed frame must use current RGB evidence for both tips.

    This fixture is the frame that previously reported ``candidates=[]`` while
    retaining a high-visibility legacy proposal over the tabletop.  The two
    proposals below are deliberately different stale priors; neither may
    become an endpoint or move the image-supported pair.
    """
    path = Path(__file__).parent/'fixtures'/'column_current_observation_20260921_033215.png'
    raw = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    old = np.array([[67.89, 96.57], [48.17, 90.80]])
    alternate = np.array([[47.8866577, 103.3934], [20.9522, 97.2582]])

    first = locate_column_jaw_tips(raw, old)
    second = locate_column_jaw_tips(raw, alternate)
    assert first['available'], first['reason']
    assert second['available'], second['reason']
    np.testing.assert_allclose(first['fm_px'], second['fm_px'], atol=1e-6)
    np.testing.assert_allclose(first['fm_px'], [[34.6, 108.5], [11.3, 103.0]], atol=2.0)
    assert first['fallback_mode'] == 'overexposed-full-frame-cool-component-consensus'
    assert first['proposal_used_for_ranking_only']
    assert not first['proposal_used_as_endpoint']
    assert not first['fk_used']
    assert first['unordered'] and not first['physical_identity_verified']


def test_latest_live_overexposure_resolves_geometric_tie_without_promoting_proposal():
    """A cable-like component must not hide the two visible Column jaws.

    This is the sequence-6 Column exposure captured at image time
    50197.985254933. It produces two nearly tied geometric pairs. The current
    visual proposal is allowed to associate the pair, but its points must
    remain metadata only; the returned endpoints come from RGB contours.
    """
    path = Path(__file__).parent/'fixtures/'/'column_current_observation_20260921_050198.png'
    raw = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    proposal = np.array([[37.18922424316406, 100.79000854492188],
                         [20.669452667236328, 97.53515625]])
    stale = np.array([[67.8865, 96.5743], [48.1692, 90.8018]])

    result = locate_column_jaw_tips(raw, proposal)
    stale_result = locate_column_jaw_tips(raw, stale)
    for item in (result, stale_result):
        assert item['available'], item['reason']
        np.testing.assert_allclose(item['fm_px'],
                                   [[37.3, 108.2], [18.9, 105.06]], atol=1.2)
        assert item['fallback_mode'] == 'overexposed-full-frame-cool-component-consensus'
        assert item['proposal_used_for_ranking_only']
        assert item['pair_tie_break']['kind'] == 'proposal_association_only'
        assert not item['proposal_used_as_endpoint']
        assert not item['fk_used']
        assert item['unordered'] and not item['ordered_identity_available']
        assert not item['accuracy_verified'] and not item['physical_identity_verified']
    np.testing.assert_allclose(result['fm_px'], stale_result['fm_px'], atol=1e-6)


def test_current_normal_exposure_uses_ambiguous_blade_only_with_anchor():
    """Distal tabletop support must not erase the real left Column blade."""
    path=Path(__file__).parent/'fixtures'/'column_ambiguous_distal_support_20260921.png'
    raw=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    before=raw.copy()
    proposal=np.array([[52.860958099365234,97.24279022216797],
                       [27.942066192626953,91.4612045288086]])

    result=locate_column_jaw_tips(raw,proposal)
    assert result['available'], result['reason']
    np.testing.assert_allclose(result['fm_px'],
                               [[44.39,104.33],[16.3,101.2]],atol=1.)
    selected=[result['candidates'][i] for i in result['selected_component_indices']]
    assert {item['label'] for item in selected}=={30,36}
    assert sorted(item['root_support_ambiguous'] for item in selected)==[False,True]
    assert result['pair_candidates'][0]['root_ambiguous_count']==1
    assert result['pair_candidates'][0]['midpoint_distance_px']<=result['proposal_span_px']
    rejected=result['proposal_association_rejections']
    assert any(row['midpoint_distance_px']>row['maximum_midpoint_distance_px']
               and row['endpoint_association_px']>row['maximum_endpoint_association_px']
               for row in rejected)
    assert not result['fk_used'] and not result['proposal_used_as_endpoint']
    np.testing.assert_array_equal(raw,before)


@pytest.mark.parametrize('angle,scale', [(0,1.),(30,.75),(-25,1.5)])
def test_nearby_fragments_cannot_replace_a_wider_same_frame_proposal(angle, scale):
    # Both fragments pass the existing shape, housing and 4 px transverse
    # checks, but represent less than half the proposed open-jaw width.
    image=scene(jaws=(103,111),blade_halfwidth=2)
    affine=cv2.getRotationMatrix2D((107,120),angle,scale)
    move=lambda points: np.asarray(points)@affine[:,:2].T+affine[:,2]
    image=cv2.warpAffine(image,affine,(224,224),flags=cv2.INTER_NEAREST)
    result=detect(image,move(PROPOSAL))
    assert not result['available'] and result['fm_px'] is None
    assert 'less than half' in result['reason']
    assert result['span_consistency_rejections']
    assert all(row['span_ratio_to_proposal']<.5
               for row in result['span_consistency_rejections'])
    assert result['proposal_span_used_for_rejection_only']
    assert not result['fk_used'] and not result['proposal_used_as_endpoint']


def test_small_rgb_opening_with_matching_proposal_remains_small_and_visual():
    image=scene(jaws=(103,111),blade_halfwidth=2)
    proposal=[[103.,132.],[111.,132.]]
    result=detect(image,proposal)
    assert result['available'], result['reason']
    np.testing.assert_allclose(result['fm_px'],[[103,143],[111,143]],atol=1.)
    assert result['proposal_span_px']==8. and result['minimum_pair_span_px']==4.
    assert result['fm_px']!=proposal
    assert not result['fk_used'] and not result['proposal_used_as_endpoint']


def test_full_geometry_prior_can_select_a_narrow_rgb_pair_despite_wide_legacy_proposal():
    from fcpg.v21_column_pose_prior import ColumnPosePrior
    image = scene(jaws=(103,111), blade_halfwidth=2)
    prior = ColumnPosePrior.from_world_pair(
        [[-.004,0.,0.],[.004,0.,0.]],
        [[1000.,0.,0.,107.],[0.,1000.,0.,143.]],
        proposal_pair=PROPOSAL)
    result = locate_column_jaw_tips(
        image, PROPOSAL, image_space='letterbox224', pose_prior=prior)
    assert result['available'], result['reason']
    np.testing.assert_allclose(result['fm_px'], [[103,143],[111,143]], atol=1.)
    assert result['geometry_ranking_source'] == 'full-q-camera-distribution'
    assert not result['proposal_span_used_for_rejection_only']
    assert result['pair_candidates'][0]['pose_match']['log_likelihood'] < 0.


def test_full_geometry_prior_ranks_true_current_jaws_above_mount_fragments():
    from fcpg.v21_column_pose_prior import ColumnPosePrior
    path=Path(__file__).parent/'fixtures'/'column_ambiguous_distal_support_20260921.png'
    raw=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    proposal=np.asarray([[52.8609581,97.2427902],[27.9420662,91.4612045]])
    prior=ColumnPosePrior.from_world_pair(
        [[.03,0.,0.],[0.,0.,0.]],
        [[1000.,0.,0.,15.],[0.,1000.,0.,103.]],
        proposal_pair=proposal)
    result=locate_column_jaw_tips(raw,proposal,pose_prior=prior)
    assert result['available'], result['reason']
    np.testing.assert_allclose(result['fm_px'],[[45.15,104.65],[16.8,101.15]],atol=1.)
    # The mechanical prior ranks only image pairs associated with this frame.
    # Distant upper mounts must not bypass that visual association.
    assert result['proposal_association_rejections']
    assert result['pair_candidates'][0]['pose_match']['center_mahalanobis_sq'] < (
        result['pose_center_mahalanobis_sq_limit'])


def test_full_geometry_prior_accepts_collapsed_proposal_without_inventing_pixels():
    from fcpg.v21_column_pose_prior import ColumnPosePrior
    proposal=np.asarray([[107.,141.],[107.,141.]])
    prior=ColumnPosePrior.from_world_pair(
        [[-.017,0.,0.],[.017,0.,0.]],
        [[1000.,0.,0.,107.],[0.,1000.,0.,143.]],proposal_pair=proposal)
    result=locate_column_jaw_tips(scene(),proposal,image_space='letterbox224',pose_prior=prior)
    assert result['available'], result['reason']
    np.testing.assert_allclose(result['fm_px'],[[90,143],[124,143]],atol=1.)
    absent=locate_column_jaw_tips(
        scene(jaws=(90,)),proposal,image_space='letterbox224',pose_prior=prior)
    assert not absent['available'] and absent['fm_px'] is None


def test_consistent_proposal_width_cannot_rescale_the_measured_rgb_pair():
    image=scene()
    narrow=detect(image,[[94,141],[120,141]])
    wider=detect(image,[[83,141],[131,141]])
    assert narrow['available'] and wider['available']
    np.testing.assert_array_equal(narrow['fm_px'],wider['fm_px'])
    assert wider['minimum_pair_span_px']==24.
    assert wider['proposal_span_px']==48.
    assert not wider['proposal_used_as_endpoint']


def _latest_overexposed_column_fixture():
    path = Path(__file__).parent/'fixtures/'/'column_current_observation_20260921_033215.png'
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


LATEST_STALE_PROPOSAL = np.array([
    [67.88652801513672, 96.57430267333984],
    [48.169189453125, 90.80176544189453],
])


def test_latest_overexposed_observation_uses_image_supported_pair_outside_stale_roi():
    raw = _latest_overexposed_column_fixture()
    result = locate_column_jaw_tips(raw, LATEST_STALE_PROPOSAL)
    assert result['available'], result['reason']
    # Same-frame visual annotation only; this is not a metric or physical GT.
    np.testing.assert_allclose(result['fm_px'], [[34., 109.], [11., 103.]], atol=1.5)
    assert result['fallback_mode'] == 'overexposed-full-frame-cool-component-consensus'
    assert result['proposal_used_for_ranking_only']
    assert not result['proposal_used_as_endpoint']
    assert result['unordered'] and not result['ordered_identity_available']
    assert not result['accuracy_verified'] and not result['physical_identity_verified']
    assert not result['fk_used']


def test_latest_overexposed_observation_224_canvas_fails_closed():
    raw = _latest_overexposed_column_fixture()
    canvas = letterbox_rgb_image(raw, 224)
    result = locate_column_jaw_tips(canvas, LATEST_STALE_PROPOSAL,
                                     image_space='letterbox224')
    # The diagnostic fallback is raw-only; a display canvas must not silently
    # change the detector contract.
    assert not result['available'] and result['fm_px'] is None


@pytest.mark.parametrize('polygon', [
    np.array([[20, 140], [90, 140], [90, 245], [15, 245]], np.int32),
    np.array([[90, 95], [175, 95], [175, 250], [90, 250]], np.int32),
])
def test_latest_overexposed_observation_missing_one_blade_fails_closed(polygon):
    raw = _latest_overexposed_column_fixture()
    cv2.fillPoly(raw, [polygon], (180, 150, 120))
    result = locate_column_jaw_tips(raw, LATEST_STALE_PROPOSAL)
    assert not result['available'] and result['fm_px'] is None
    assert not result['proposal_used_as_endpoint']


def _full_q_connected_housing_fixture():
    from fcpg.v21_column_pose_prior import ColumnPosePrior
    path=Path(__file__).parent/'fixtures'/'column_full_q_connected_housing_20260921.png'
    raw=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    proposal=np.array([[40.03851318359375,99.56428527832031],
                       [12.658121109008789,91.93400573730469]])
    world=[[.17526582521202783,.01070557764836832,-.002049455321714258],
           [.17872284963403182,-.04400419313788463,.0021089410148222992]]
    matrix=[[-50.590089607430706,745.225792598815,-9.109569008450272,107.2949405074015],
            [505.88653410428714,7.956898879146009,-366.5720595365771,14.15297540586495]]
    q=[.072254337,-97.282829,98.846497,62.751678,-52.935066,31.406044]
    prior=ColumnPosePrior.from_world_pair(world,matrix,q_at_image=q,proposal_pair=proposal)
    return raw,proposal,prior


def test_recorded_full_q_roi_uses_local_housing_cap_not_connected_region_centroid():
    # Exact raw Column RGB from the 2026-09-21 current observation. Broadening
    # the ROI via full FK used to move a pale-table component's global centre
    # into blade 71's axial interval, erase that real blade, and select mounts.
    import hashlib
    raw,proposal,prior=_full_q_connected_housing_fixture()
    assert hashlib.sha256(raw.tobytes()).hexdigest()==(
        'b3b4492775859d603423997fe9b3165978d6c1c52c49eb6485ec013638821169')
    for pose in (None,prior):
        result=locate_column_jaw_tips(raw,proposal,pose_prior=pose)
        assert result['available'],result['reason']
        np.testing.assert_allclose(result['fm_px'],[[34.3,108.5],[10.85,103.25]],atol=.6)
        assert not result['fk_used'] and not result['proposal_used_as_endpoint']
        assert not result['physical_identity_verified']
    wrong_mount=np.array([[33.25,47.60],[3.85,54.95]])
    assert prior.score_pair(wrong_mount)['center_mahalanobis_sq']>(
        result['pose_center_mahalanobis_sq_limit'])
    assert result['pair_candidates'][0]['pose_match']['center_mahalanobis_sq']<(
        result['pose_center_mahalanobis_sq_limit'])
    # Re-associating the returned order (as tip_display does) is idempotent.
    assert not prior.score_pair(result['fm_px'])['swapped']
    reversed_proposal=locate_column_jaw_tips(raw,proposal[::-1],pose_prior=prior)
    np.testing.assert_array_equal(reversed_proposal['fm_px'],result['fm_px'])


def test_wrong_upper_mount_pose_prior_cannot_override_same_frame_rgb_evidence():
    from fcpg.v21_column_pose_prior import ColumnCameraUncertainty,ColumnPosePrior
    raw,proposal,_=_full_q_connected_housing_fixture()
    wrong_mount=np.array([[33.25,47.60],[3.85,54.95]])
    wrong_prior=ColumnPosePrior.from_world_pair(
        np.column_stack((wrong_mount,np.zeros(2))),[[1.,0.,0.,0.],[0.,1.,0.,0.]],
        proposal_pair=proposal,
        uncertainty=ColumnCameraUncertainty(translation_std_fraction=(0.,0.)))
    result=locate_column_jaw_tips(raw,proposal,pose_prior=wrong_prior)
    assert not result['available'] and result['fm_px'] is None
    assert result['pose_center_consistency_rejections']
    assert result['proposal_association_rejections']
    assert not result['fk_used'] and not result['proposal_used_as_endpoint']


def test_full_q_relative_range_rejects_nearby_pose_inconsistent_contours():
    from fcpg.v21_column_pose_prior import ColumnCameraUncertainty,ColumnPosePrior
    raw,proposal,_=_full_q_connected_housing_fixture()
    vertical_prior=ColumnPosePrior.from_world_pair(
        [[0.,-.05,0.],[0.,.05,0.]],
        [[1000.,0.,0.,0.],[0.,1000.,0.,0.]],proposal_pair=proposal,
        uncertainty=ColumnCameraUncertainty(
            rotation_std_deg=(0.,0.,0.),log_scale_std=0.))
    result=locate_column_jaw_tips(raw,proposal,pose_prior=vertical_prior)
    assert not result['available'] and result['fm_px'] is None
    assert result['pose_relative_consistency_rejections']
    assert all(row['relative_mahalanobis_sq']
               > row['maximum_relative_mahalanobis_sq']
               for row in result['pose_relative_consistency_rejections'])
    assert not result['fk_used'] and not result['proposal_used_as_endpoint']


def test_projected_overlap_keeps_one_visible_contour_unresolved():
    from fcpg.v21_column_pose_prior import ColumnCameraUncertainty,ColumnPosePrior
    proposal=[[107.,141.],[107.,141.]]
    prior=ColumnPosePrior.from_world_pair(
        [[0.,0.,-.02],[0.,0.,.02]],
        [[1000.,0.,0.,107.],[0.,1000.,0.,143.]],proposal_pair=proposal,
        uncertainty=ColumnCameraUncertainty(rotation_std_deg=(0.,0.,0.),log_scale_std=0.))
    result=locate_column_jaw_tips(
        scene(jaws=(107,)),proposal,image_space='letterbox224',pose_prior=prior)
    assert result['pose_overlap_affinity']==pytest.approx(1.)
    assert not result['available'] and result['fm_px'] is None
    assert len({candidate['label'] for candidate in result['candidates']})==1


def test_overexposed_near_tie_can_select_the_second_scored_pair():
    from fcpg.v21_column_tip_visual import _overexposed_full_frame_pair
    path=Path(__file__).parent/'fixtures'/'column_current_observation_20260921_050198.png'
    raw=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    proposal=np.array([[37.18922424316406,100.79000854492188],
                       [20.669452667236328,97.53515625]])
    class SlightPreferenceForCable:
        def score_pair(self,pair):
            # Change only which of two already-near-tied geometric rows sorts
            # first. The same-frame association still clearly favors jaws.
            return dict(log_likelihood=.05 if min(point[0] for point in pair)<5 else 0.,
                        center_mahalanobis_sq=0.,relative_mahalanobis_sq=0.,
                        swapped=True,identity_probability=.5)
    result=_overexposed_full_frame_pair(
        raw,proposal,image_space='raw',scale=.35,pad=np.array([0.,28.]),
        content=np.array([0.,28.,224.,196.]),pose_prior=SlightPreferenceForCable())
    assert result is not None
    np.testing.assert_allclose(result['fm_px'],[[37.3,108.2],[18.9,105.06]],atol=.1)
    assert result['pair_candidates'][0]['score']<result['pair_candidates'][1]['score']
    assert result['pair_tie_break']['association_gap_px']>4.
