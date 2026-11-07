import json

import numpy as np
import pytest

from fcpg.v21_archive_camera_layout import ArchiveCameraLayout
from fcpg.v21_encoder import DIM, PATCHES
from scripts.select_v21_camera_layout_sources import build, read_kinds, select, source_drops


def _layout(n):
    return {"order": np.tile([0, 1], (n, 1)), "wrist_present": np.ones(n, bool), "decided": np.ones(n, bool),
            "external_present": np.ones(n, bool)}


def test_only_training_recordings_of_sources_not_kept_are_left_out():
    # 10/25 (17.103): the wrist picture moves another way in some sources when the base turns
    source = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "b"}
    kinds = {"a": "same", "b": "sideways", "c": "other way", "d": "inconsistent"}      # "e" was never measured
    development = np.zeros(6, bool)
    development[5] = True
    drop = source_drops(source, development, kinds, ("same",))
    assert drop == {1: "sideways", 2: "other way", 3: "inconsistent", 4: "not measured"}      # 5 is development
    assert source_drops(source, development, kinds, ("same", "not measured")) == {
        1: "sideways", 2: "other way", 3: "inconsistent"}
    new, dropped = select(_layout(6), drop)
    assert dropped.tolist() == [False, True, True, True, True, False]
    assert new["decided"].tolist() == [True, False, False, False, False, True]
    assert not new["wrist_present"][dropped].any() and new["external_present"].all()
    assert (new["order"][dropped] == [0, 1]).all()


def test_verdicts_are_read_from_a_summary_or_a_mapping(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"main_angle": -160, "sources": {
        "a": {"kind": "same", "v3": 4}, "b": {"kind": "sideways"}, "c": {"recordings": 2}}}))
    assert read_kinds(tmp_path / "summary.json") == {"a": "same", "b": "sideways", "c": "not measured"}
    (tmp_path / "plain.json").write_text(json.dumps({"a": "other way"}))
    assert read_kinds(tmp_path / "plain.json") == {"a": "other way"}
    (tmp_path / "bad.json").write_text(json.dumps({"a": "upside down"}))
    with pytest.raises(ValueError):
        read_kinds(tmp_path / "bad.json")


def test_the_table_is_rebuilt_from_an_index_and_loads_as_a_layout(tmp_path):
    index = tmp_path / "index"
    index.mkdir()
    rows = [  # pool entry, source, split, rows
        (1, "a", "train", 3), (2, "b", "train", 2), (3, "b", "development", 2), (4, "c", "train", 4), (6, "a", "train", 1)]
    episodes = [{"temporal_episode_index": e, "manifest_row": {"source_group": s, "split": sp}} for e, s, sp, _ in rows]
    (index / "episode-path-map.json").write_text(json.dumps({"episodes": episodes}))
    np.save(index / "episode_index.npy", np.repeat([e for e, *_ in rows], [n for *_, n in rows]).astype(np.int32))
    np.save(index / "split.npy", np.repeat([int(sp == "development") for _, _, sp, _ in rows],
                                           [n for *_, n in rows]).astype(np.uint8))
    layout = _layout(8)
    layout["decided"][6] = False                       # undecided before (a duplicate copy): stays undecided
    layout["wrist_present"][6] = False
    duplicate = np.zeros(8, bool)
    duplicate[6] = True
    np.savez(tmp_path / "layout.npz", **layout, duplicate_dropped=duplicate)
    (tmp_path / "kinds.json").write_text(json.dumps({"sources": {"a": {"kind": "same"}, "b": {"kind": "sideways"}}}))
    report = build(tmp_path / "layout.npz", index, tmp_path / "kinds.json", ["same"], tmp_path / "out")
    assert report["dropped"]["by_kind"] == {"sideways": 1, "not measured": 1}
    assert report["rows"]["train"] == dict(before=9, after=3, with_wrist_before=9, with_wrist_after=3)
    assert report["rows"]["development"]["after"] == report["rows"]["development"]["before"] == 2
    assert report["wrist_train_rows_before_by_kind"] == {"same": 3, "sideways": 2, "not measured": 4}
    assert report["wrist_train_rows_dropped_by_source"] == {"c": 4, "b": 2}
    assert report["kept_sources"] == 1
    with np.load(tmp_path / "out" / "layout.npz") as data:
        assert data["duplicate_dropped"].tolist() == duplicate.tolist()
        assert data["source_dropped"].tolist() == [False, False, True, False, True, False, False, False]

    class Archive:                                    # what ArchiveCameraLayout reads of the Archive
        episodes = [None] * 8
        endpoint_episode_index = np.arange(8)
    table = ArchiveCameraLayout(tmp_path / "out" / "layout.npz", Archive)
    assert table.decided.tolist() == [True, True, False, True, False, True, False, True]
    with pytest.raises(FileExistsError):
        build(tmp_path / "layout.npz", index, tmp_path / "kinds.json", ["same"], tmp_path / "out")
    with pytest.raises(ValueError):
        build(tmp_path / "layout.npz", index, tmp_path / "kinds.json", ["upside down"], tmp_path / "out2")


def test_others_can_keep_their_external_camera_and_lose_only_the_wrist_picture(tmp_path):
    # 11/01 (17.162): a recording of a source whose wrist direction is not kept keeps its external camera
    index = tmp_path / "index"
    index.mkdir()
    rows = [  # pool entry, source, split, rows
        (0, "a", "train", 3), (1, "b", "train", 2), (2, "b", "development", 2), (3, "c", "train", 4), (4, "d", "train", 5)]
    episodes = [{"temporal_episode_index": e, "manifest_row": {"source_group": s, "split": sp}} for e, s, sp, _ in rows]
    (index / "episode-path-map.json").write_text(json.dumps({"episodes": episodes}))
    np.save(index / "episode_index.npy", np.repeat([e for e, *_ in rows], [n for *_, n in rows]).astype(np.int32))
    np.save(index / "split.npy", np.repeat([int(sp == "development") for _, _, sp, _ in rows],
                                           [n for *_, n in rows]).astype(np.uint8))
    layout = _layout(5)
    layout["order"][1] = [1, 0]                        # b's cameras the other way round: its external view is raw slot 0
    layout["external_present"][3] = False             # c: a wrist camera only
    layout["decided"][4] = False                      # d: undecided before
    layout["wrist_present"][4] = False
    np.savez(tmp_path / "layout.npz", **layout)
    (tmp_path / "kinds.json").write_text(json.dumps({"sources": {"a": {"kind": "same"}, "b": {"kind": "sideways"},
                                                                 "c": {"kind": "other way"}}}))
    report = build(tmp_path / "layout.npz", index, tmp_path / "kinds.json", ["same"], tmp_path / "out",
                   others="external-only")
    assert report["wrist_left_out"] == dict(recordings=1, by_kind={"sideways": 1}, train_rows=2, sources=1)
    assert report["dropped"]["by_kind"] == {"other way": 1, "not measured": 1}
    assert report["rows"]["train"] == dict(before=9, after=5, with_wrist_before=9, with_wrist_after=3)
    assert report["rows"]["development"]["after"] == report["rows"]["development"]["before"] == 2

    class Archive:
        episodes = [None] * 5
        endpoint_episode_index = np.arange(5)
    table = ArchiveCameraLayout(tmp_path / "out" / "layout.npz", Archive)
    assert table.decided.tolist() == [True, True, True, False, False]
    assert table.wrist_present.tolist() == [True, False, True, False, False]
    with np.load(tmp_path / "out" / "layout.npz") as data:
        assert data["wrist_dropped"].tolist() == [False, True, False, False, False]
    tokens = np.arange(5 * 2, dtype=np.float32).reshape(5, 2, 1, 1) + 1       # raw view v of entry e = 2e + v + 1
    tokens = np.broadcast_to(tokens, (5, 2, PATCHES, DIM)).copy()
    out = table.remap(tokens, np.arange(5))
    assert (out[1, 0] == 0).all() and (out[1, 1] == 3).all()       # b: no wrist view; its external view (raw slot 0)
    assert (out[0, 0] == 1).all() and (out[0, 1] == 2).all()       # a: as before
    plain = build(tmp_path / "layout.npz", index, tmp_path / "kinds.json", ["same"], tmp_path / "plain")
    assert "wrist_left_out" not in plain and plain["rows"]["train"]["after"] == 3
    with pytest.raises(ValueError):
        build(tmp_path / "layout.npz", index, tmp_path / "kinds.json", ["same"], tmp_path / "bad", others="rotate")
