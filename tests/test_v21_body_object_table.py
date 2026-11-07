import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import build_v21_body_object_table as table


class GuardedValues:
    """A memmap stand-in that cannot reveal any sealed or whole-source value."""
    def __init__(self, episodes, frames, rgb=False):
        self.episodes, self.frames, self.rgb = episodes, frames, rgb
        self.shape = (len(episodes), 224, 224, 3) if rgb else (len(episodes), 6)
        self.dtype = np.dtype('uint8' if rgb else 'float32')
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError('whole source materialization forbidden')

    def __getitem__(self, index):
        assert isinstance(index, int), 'only admitted scalar row access is allowed'
        assert self.episodes[index] != 2, 'sealed episode value read'
        self.reads.append(index)
        if self.rgb:
            value = np.full((224, 224, 3), index % 256, np.uint8)
            value[:28] = 0; value[196:] = 0
            return value
        return np.array([index, self.frames[index], 0, 0, 0, 0], np.float32)


class StatefulEncoder:
    def __init__(self, failure_at=None):
        self.calls = []; self.resets = []; self.failure_at = failure_at
        self._sticky_colour_cog = 900
        self._last_body_area = 900

    def describe(self):
        return dict(policy_source=table.TARGET)

    def clear_observation(self):
        self.resets.append(len(self.calls))
        self.last_policy_record = self.last_record = None
        self._sticky_colour_cog = self._last_body_area = None

    def encode(self, rgb, q):
        if len(self.calls) == self.failure_at:
            raise RuntimeError('encoder interrupted')
        index, local = int(q[0]), int(q[1])
        assert rgb[28, 0, 0] == index % 256
        self.calls.append(index)
        if self._sticky_colour_cog is None: self._sticky_colour_cog = index
        self._last_body_area = (self._last_body_area or 0) + 1
        self.last_policy_record = dict(reason='test actual absence', body_truncated=False)
        if local == 1:
            return np.zeros(8, np.float32)
        return np.array([1, .9, self._sticky_colour_cog/100, index/100,
                         self._last_body_area, .1, 1, 0], np.float32)


def make_source(tmp_path, monkeypatch, *, length=4):
    source = tmp_path/'source'; source.mkdir()
    episodes = np.repeat(np.arange(4, dtype=np.int64), length)
    frames = np.tile(np.arange(length, dtype=np.int64), 4)
    np.save(source/'episodes.npy', episodes)
    np.save(source/'frame_indices.npy', frames)
    (source/'manifest.json').write_text(json.dumps(dict(camera='observation.images.wrist',
        fps=30, image_size=224, dataset_id='guarded-unit-fixture', source_root='/actual-fixture', frames=len(episodes))))
    rgb, q = GuardedValues(episodes, frames, True), GuardedValues(episodes, frames)
    original_load = np.load
    opened = []
    def guarded_load(path, *args, **kwargs):
        name = str(path)
        assert not name.endswith('actions.npy'), 'builder must not open action labels'
        if name == str(source/'frames.npy'): value = rgb
        elif name == str(source/'states.npy'): value = q
        else: return original_load(path, *args, **kwargs)
        assert kwargs == dict(mmap_mode='r', allow_pickle=False)
        opened.append(name)
        return value
    monkeypatch.setattr(table.np, 'load', guarded_load)
    fish = SimpleNamespace(episodes=episodes, frame_indices=frames)
    return source, fish, rgb, q, opened


def built(tmp_path, monkeypatch, selected=(0, 1, 3), failure_at=None):
    source, fish, rgb, q, opened = make_source(tmp_path, monkeypatch)
    encoder = StatefulEncoder(failure_at)
    output = tmp_path/'body'
    report = table.build_table(source, selected, output, encoder_factory=lambda: encoder, source_geometry='letterbox_v1')
    return source, fish, encoder, rgb, q, output, report


def validate(source, fish, output, required=(0, 3, 4, 7, 12, 15)):
    return table.validate_manifest(output/table.TABLE_NAME,
        table.manifest_path_for(output/table.TABLE_NAME), fish, source, required)


def test_only_admitted_rgb_q_are_read_in_order_and_episode_state_resets(tmp_path, monkeypatch):
    source, fish, encoder, rgb, q, output, report = built(tmp_path, monkeypatch)
    expected = [*range(8), *range(12, 16)]
    assert encoder.calls == rgb.reads == q.reads == expected
    assert encoder.resets == [0, 4, 8]
    values = np.load(output/table.TABLE_NAME)
    np.testing.assert_array_equal(values[[0, 4, 12], 4], [1, 1, 1])
    np.testing.assert_allclose(values[[0, 4, 12], 2], [0, .04, .12])
    assert report['computed_rows'] == 12 and report['available_rows'] == 9
    assert report['action_files_opened'] is False and report['episode_2_values_read'] is False
    contract = validate(source, fish, output)
    assert len(contract['computed_cues']) == 12 and contract['required_rows_verified'] == 6
    # An actual encoded absence and an unprocessed zero have different admission.
    assert report['rows'][1]['computed'] and not report['rows'][1]['available']
    validate(source, fish, output, required=[1])
    with pytest.raises(ValueError, match='uncomputed'):
        validate(source, fish, output, required=[8])


@pytest.mark.parametrize('episodes', ([2], [0, 0], [], [0.0], [True], [4]))
def test_bad_episode_requests_fail_before_any_io(tmp_path, monkeypatch, episodes):
    monkeypatch.setattr(table, 'metadata', lambda *_: pytest.fail('metadata opened before rejection'))
    with pytest.raises(ValueError, match='sealed'):
        table.build_table(tmp_path/'missing', episodes, tmp_path/'out',
            encoder_factory=lambda: pytest.fail('encoder instantiated'), source_geometry='letterbox_v1')


def test_missing_episode_or_interleaved_source_fails_before_value_access(tmp_path, monkeypatch):
    source, fish, rgb, q, opened = make_source(tmp_path, monkeypatch)
    fish.episodes[4:] = 2
    np.save(source/'episodes.npy', fish.episodes)
    with pytest.raises(ValueError, match='frame identities'):
        table.build_table(source, [0, 1], tmp_path/'out', encoder_factory=lambda: pytest.fail('encoder instantiated'), source_geometry='letterbox_v1')
    assert not opened and not rgb.reads and not q.reads
    with pytest.raises(ValueError, match='frame identities'):
        table.selected_rows(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), [0])


def test_partial_failure_is_retained_as_incomplete_and_refused(tmp_path, monkeypatch):
    source, fish, rgb, q, opened = make_source(tmp_path, monkeypatch)
    output = tmp_path/'body'
    with pytest.raises(RuntimeError, match='interrupted'):
        table.build_table(source, [0, 1, 3], output, encoder_factory=lambda: StatefulEncoder(failure_at=2), source_geometry='letterbox_v1')
    report = json.loads(table.manifest_path_for(output/table.TABLE_NAME).read_text())
    assert report['complete'] is False and report['computed_rows'] == 2
    assert 'interrupted' in report['error']
    with pytest.raises(ValueError, match='complete identity-bound'):
        validate(source, fish, output)


@pytest.mark.parametrize('change', ['drop_endpoint', 'reorder', 'wrong_episode', 'not_computed',
    'fake_available', 'source_mismatch', 'summary', 'missing_vector'])
def test_changed_identity_or_computation_evidence_fails(tmp_path, monkeypatch, change):
    source, fish, encoder, rgb, q, output, report = built(tmp_path, monkeypatch)
    if change == 'drop_endpoint': report['rows'].pop(3)
    elif change == 'reorder': report['rows'][2:4] = report['rows'][2:4][::-1]
    elif change == 'wrong_episode': report['rows'][0]['episode'] = 3
    elif change == 'not_computed': report['rows'][0]['computed'] = False
    elif change == 'fake_available': report['rows'][0]['available'] = False
    elif change == 'source_mismatch': report['source']['manifest']['dataset_id'] = 'other'
    elif change == 'summary': report['available_rows'] = 100
    elif change == 'missing_vector': report['rows'][0]['vector'] = [0]*8
    table.manifest_path_for(output/table.TABLE_NAME).write_text(json.dumps(report))
    with pytest.raises(ValueError): validate(source, fish, output)


@pytest.mark.parametrize('row,column,value', [(0, 2, 99), (1, 2, 5), (8, 0, 1), (0, 1, float('nan'))])
def test_table_edits_cannot_be_accepted_as_original_encoder_output(tmp_path, monkeypatch, row, column, value):
    source, fish, encoder, rgb, q, output, report = built(tmp_path, monkeypatch)
    values = np.load(output/table.TABLE_NAME)
    values[row, column] = value
    np.save(output/table.TABLE_NAME, values)
    with pytest.raises(ValueError): validate(source, fish, output)


def wrist_reference():
    return dict(geometry_protocol='letterbox_v1', image_size=[224, 224],
        content_rect=[0, 28, 224, 196], source_width=640, source_height=480,
        fits={'wrist': dict(view='wrist', feature_names=list(table.WRIST_FEATURES),
            coefficients=np.arange(10).reshape(5, 2).tolist(), ridge=1e-8)})


@pytest.mark.parametrize('change', ['squash', 'size', 'rect', 'raw_size', 'column', 'features', 'nan'])
def test_same_shape_wrong_wrist_reference_is_rejected(tmp_path, change):
    data = wrist_reference()
    if change == 'squash': data['geometry_protocol'] = 'squash_v0'
    elif change == 'size': data['image_size'] = [480, 640]
    elif change == 'rect': data['content_rect'] = [0, 0, 224, 224]
    elif change == 'raw_size': data['source_width'] = 1920
    elif change == 'column': data['fits']['wrist']['view'] = 'column'
    elif change == 'features': data['fits']['wrist']['feature_names'][0] = 'x_base'
    elif change == 'nan': data['fits']['wrist']['coefficients'][0][0] = float('nan')
    path = tmp_path/'reference.json'; path.write_text(json.dumps(data))
    with pytest.raises(ValueError): table.read_wrist_reference(path)


def test_readable_calibration_and_asset_identity_recorded_without_hashes(tmp_path):
    profile = tmp_path/'profile'; profile.mkdir()
    (profile/'pickup.png').write_bytes(b'fixture-template')
    (profile/'targets.json').write_text(json.dumps(dict(version=1,
        targets=[dict(role='pickup', view='wrist', template_file='pickup.png')])))
    reference = tmp_path/'reference.json'; reference.write_text(json.dumps(wrist_reference()))
    model = tmp_path/'model.xml'; model.write_text('<mujoco model="test"/>')
    result = table.dependency_description(profile, reference, model)
    assert result['wrist_projection']['fit'] == wrist_reference()['fits']['wrist']
    assert result['kinematics_model_xml'] == model.read_text()
    assert result['templates'][0]['size_bytes'] == 16
    assert 'hash' not in json.dumps(result)


def test_progress_uses_completed_real_rows_and_episode_boundaries(tmp_path, monkeypatch):
    source, fish, rgb, q, opened = make_source(tmp_path, monkeypatch, length=35)
    progress = []
    report = table.build_table(source, [0, 1, 3], tmp_path/'body',
        encoder_factory=StatefulEncoder, source_geometry='letterbox_v1', progress=progress.append)
    assert [r['computed'] for r in progress] == [35, 70, 100, 105]
    assert progress[-1]['available'] == report['available_rows']
    assert all(row['elapsed_s'] >= 0 for row in progress)


def test_source_geometry_must_be_explicit_even_for_same_224_shape(tmp_path, monkeypatch):
    source, fish, rgb, q, opened = make_source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='explicit verified'):
        table.build_table(source, [0], tmp_path/'body', encoder_factory=StatefulEncoder)
    assert not opened


def test_declared_squash_source_cannot_be_overridden_as_letterbox(tmp_path, monkeypatch):
    source, fish, rgb, q, opened = make_source(tmp_path, monkeypatch)
    path = source/'manifest.json'; manifest = json.loads(path.read_text())
    manifest['geometry_protocol'] = 'squash_v0'; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='contradicts'):
        table.build_table(source, [0], tmp_path/'body', encoder_factory=StatefulEncoder, source_geometry='letterbox_v1')
    assert not opened


def test_every_selected_frame_padding_is_checked_and_failed_table_is_incomplete(tmp_path, monkeypatch):
    source, fish, rgb, q, opened = make_source(tmp_path, monkeypatch)
    original = GuardedValues.__getitem__
    def wrong_padding(self, index):
        value = original(self, index)
        if self.rgb and index == 3: value[223, 0, 0] = 1
        return value
    monkeypatch.setattr(GuardedValues, '__getitem__', wrong_padding)
    output = tmp_path/'body'
    with pytest.raises(ValueError, match='padding geometry'):
        table.build_table(source, [0, 1, 3], output, encoder_factory=StatefulEncoder, source_geometry='letterbox_v1')
    report = json.loads(table.manifest_path_for(output/table.TABLE_NAME).read_text())
    assert report['complete'] is False and report['computed_rows'] == 3
    assert report['source_geometry']['padding_checked_frames'] == 3
    assert q.reads == [0, 1, 2]


def test_legacy_missing_geometry_remains_honestly_undeclared_after_explicit_verified_use(tmp_path, monkeypatch):
    source, fish, encoder, rgb, q, output, report = built(tmp_path, monkeypatch)
    assert 'geometry_protocol' not in json.loads((source/'manifest.json').read_text())
    assert report['source_geometry']['original_manifest_declaration'] is None
    assert report['source_geometry']['configured_explicitly'] is True
    assert report['source_geometry']['padding_checked_frames'] == 12
    validate(source, fish, output)


def test_relocated_metadata_needs_explicit_original_provenance_root(tmp_path, monkeypatch):
    source, fish, encoder, rgb, q, output, report = built(tmp_path, monkeypatch)
    local = tmp_path/'relocated'
    shutil.copytree(source, local)
    with pytest.raises(ValueError, match='source manifest differs'):
        validate(local, fish, output)
    args = (output/table.TABLE_NAME, table.manifest_path_for(output/table.TABLE_NAME), fish, local, [0, 4, 12])
    accepted = table.validate_manifest(*args, source_wrist_root=source)
    assert accepted['source'] == report['source']
    # The original path is provenance only; the moved actual metadata is what gets checked.
    shutil.rmtree(source)
    table.validate_manifest(*args, source_wrist_root=source)
    actual = json.loads((local/'manifest.json').read_text())
    actual['dataset_id'] = 'other-source'
    (local/'manifest.json').write_text(json.dumps(actual))
    with pytest.raises(ValueError, match='source manifest differs'):
        table.validate_manifest(*args, source_wrist_root=source)


def test_relocation_cannot_relax_frame_or_computed_vector_checks(tmp_path, monkeypatch):
    source, fish, encoder, rgb, q, output, report = built(tmp_path, monkeypatch)
    local = tmp_path/'relocated'; shutil.copytree(source, local)
    args = (output/table.TABLE_NAME, table.manifest_path_for(output/table.TABLE_NAME), fish, local, [0, 4, 12])
    frames = np.load(local/'frame_indices.npy'); frames[0] += 1
    np.save(local/'frame_indices.npy', frames)
    with pytest.raises(ValueError, match='frame identities differ'):
        table.validate_manifest(*args, source_wrist_root=source)
    np.save(local/'frame_indices.npy', fish.frame_indices)
    values = np.load(output/table.TABLE_NAME); values[0, 2] += 1
    np.save(output/table.TABLE_NAME, values)
    with pytest.raises(ValueError, match='cue values differ'):
        table.validate_manifest(*args, source_wrist_root=source)
