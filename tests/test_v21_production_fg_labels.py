import numpy as np
import pytest

from fcpg.v21_production_fg_labels import history_samples, label_record, ProductionFGTable, SCHEMA, label_json_default


def test_production_numpy_diagnostics_are_json_numbers_not_strings():
    import json
    value = dict(component=np.int64(2), trusted=np.bool_(True), xy=np.array([1.,2.],np.float32))
    assert json.loads(json.dumps(value, default=label_json_default, allow_nan=False)) == dict(
        component=2, trusted=True, xy=[1.,2.])
    with pytest.raises(ValueError):
        json.dumps(dict(bad=np.float32(np.nan)), default=label_json_default, allow_nan=False)


def test_wrist_fk_out_of_frame_is_not_clipped_or_claimed_visual():
    points = [[-10, 25], [300, 150]]
    record = label_record({'wrist': dict(source='wrist-fk-projection', fk_fm_px=points)}, [True, True])['wrist']
    assert record['pair_available'] and not record['in_view']
    assert record['fg_px'] == points and not record['current_visual_supported']
    assert not record['human_verified']


def test_column_retained_fk_proposal_cannot_become_current_visual_label():
    for source in ('fcpg-retained-column-fg', 'column-q-fk-projection'):
        rec = dict(available=True, source=source, fm_px=[[50, 60], [80, 60]], ordered_identity_available=True)
        value = label_record({'column':rec}, [True, True])['column']
        assert not value['pair_available'] and value['fg_px'] is None
    rec['source'] = 'column-rgb-jaw-contour'
    rec['ordered_identity_available'] = False
    value = label_record({'column':rec}, [True, True])['column']
    assert value['pair_available'] and not value['ordered_identity_available']
    assert not label_record({'column':rec}, [True, False])['column']['pair_available']


@pytest.mark.parametrize('invalid', [None, 'observed', 'current_image_valid', 'spatial_update_valid', 'nominal_conflict'])
def test_feedback_posterior_retains_original_evidence_decision(invalid):
    rec=dict(source='v2032-current-image-actual-readback-posterior',available=True,
        observed=True,current_image_valid=True,spatial_update_valid=True,nominal_conflict=False,
        fm_px=[[50,60],[80,60]],raw_pair_px=[[180,120],[185,120]],ordered_identity_available=False)
    if invalid is not None: rec[invalid]=invalid=='nominal_conflict'
    result=label_record({'column':rec},[True,True])['column']
    assert result['pair_available']==(invalid is None)
    assert result['fg_px']==([[50,60],[80,60]] if invalid is None else None)
    assert not result['ordered_identity_available']
    assert not result['human_verified']


def test_recorded_history_is_causal_and_reset_at_each_episode():
    times = np.arange(100)/30
    q = np.repeat(np.arange(100)[:, None], 6, axis=1).astype(float)
    values = history_samples(times, q, q+10, 1.)
    assert values[3].tolist() == [[False, True, True]]
    assert max(max(v) for v in values[4] if v is not None) == 30
    changed = q.copy(); changed[31:] = -9999
    again = history_samples(times, changed, changed+10, 1.)
    for a,b in zip(values[:4], again[:4]): np.testing.assert_array_equal(a,b)
    assert not history_samples(times, q, q+10, 0.)[3].any()


def test_loader_fails_on_incomplete_generation_before_arrays(tmp_path):
    import json
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema=SCHEMA, complete=False)))
    with pytest.raises(ValueError, match='complete same-clock'):
        ProductionFGTable(tmp_path, clock_version='clock', extraction_id='version')


def test_completed_export_and_canary_are_not_label_quality_acceptance(tmp_path):
    import json
    table = ProductionFGTable.__new__(ProductionFGTable)
    table.root, table.ids, table.manifest = tmp_path, np.arange(3), dict(source_snapshot='frozen-source')
    with pytest.raises(ValueError, match='not reviewed'):
        table.require_training_admission()
    path = tmp_path/'quality-review.json'
    review = dict(accepted_for_training=False, rows=3, source_snapshot='frozen-source')
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='failed quality review'):
        table.require_training_admission()
    review['accepted_for_training'] = True
    path.write_text(json.dumps(review))
    table.require_training_admission()
    review['source_snapshot'] = 'other-label-generation'
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='failed quality review'):
        table.require_training_admission()
