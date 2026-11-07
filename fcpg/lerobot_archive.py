"""Read immutable LeRobot episode rows and archived video frames.

The reader supports both per-episode LeRobot v2 media and chunk/file-sharded
v3 media.  It deliberately exposes only archived files named by a canonical
manifest; it contains no live camera or robot device path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


_CV2_RESIZE_THREADS_CONFIGURED = False

from fcpg.so101_archive_units import canonicalize_joint_values, row_joint_transform


@dataclass(frozen=True)
class EpisodeColumns:
    state: np.ndarray
    action: np.ndarray
    timestamp_s: np.ndarray
    task_index: np.ndarray


def _read_parquet_slice(path: Path, start: int, stop: int, columns: list[str]):
    """Read only row groups intersecting one absolute parquet row slice."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if start < 0 or stop <= start or stop > parquet.metadata.num_rows:
        raise ValueError(f"invalid parquet segment [{start},{stop}) for {path}")
    cursor = 0
    row_groups = []
    first_start = None
    for index in range(parquet.metadata.num_row_groups):
        length = parquet.metadata.row_group(index).num_rows
        group_stop = cursor + length
        if group_stop > start and cursor < stop:
            row_groups.append(index)
            if first_start is None:
                first_start = cursor
        cursor = group_stop
    if not row_groups or first_start is None:
        raise ValueError(f"no parquet row group covers [{start},{stop}) in {path}")
    table = parquet.read_row_groups(row_groups, columns=columns)
    return table.slice(start - first_start, stop - start)


def read_episode_columns(row: dict) -> EpisodeColumns:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("PyArrow is required to read LeRobot episode columns") from exc

    root = Path(row["dataset_root"])
    episode = int(row["episode_index"])
    pieces: dict[str, list[np.ndarray]] = {
        "observation.state": [],
        "action": [],
        "timestamp": [],
        "task_index": [],
    }
    for segment in row["segments"]:
        path = root / segment["data_path"]
        start = int(segment["row_start"])
        stop = int(segment["row_stop"])
        table = _read_parquet_slice(
            path,
            start,
            stop,
            [
                "episode_index",
                "observation.state",
                "action",
                "timestamp",
                "task_index",
            ],
        )
        episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        if np.any(episodes != episode):
            raise ValueError(f"segment {path}[{start}:{stop}] crosses episode identity")
        for name in pieces:
            values = np.asarray(table[name].to_pylist())
            pieces[name].append(values)

    if not pieces["timestamp"]:
        raise ValueError("canonical episode has no parquet segments")
    state = np.concatenate(pieces["observation.state"]).astype(np.float32, copy=False)
    action = np.concatenate(pieces["action"]).astype(np.float32, copy=False)
    timestamp = np.concatenate(pieces["timestamp"]).astype(np.float64, copy=False)
    task = np.concatenate(pieces["task_index"]).astype(np.int64, copy=False)
    expected = int(row["length"])
    if any(len(value) != expected for value in (state, action, timestamp, task)):
        raise ValueError("episode column lengths differ from canonical manifest")
    if state.shape != (expected, 6) or action.shape != (expected, 6):
        raise ValueError(
            f"forward pipeline requires state/action [N,6], got {state.shape}/{action.shape}"
        )
    state_order, scale, offset = row_joint_transform(row, "state")
    action_order, action_scale, action_offset = row_joint_transform(row, "action")
    if action_scale != scale or action_offset != offset:
        raise ValueError("canonical manifest must use one physical affine for state/action")
    state = canonicalize_joint_values(
        state, order=state_order, scale=scale, offset=offset
    ).astype(np.float32)
    action = canonicalize_joint_values(
        action, order=action_order, scale=scale, offset=offset
    ).astype(np.float32)
    if not all(np.all(np.isfinite(value)) for value in (state, action, timestamp)):
        raise ValueError("episode contains non-finite state/action/timestamp values")
    if not np.all(np.diff(timestamp) > 0):
        raise ValueError("episode timestamps are not strictly increasing")
    return EpisodeColumns(state=state, action=action, timestamp_s=timestamp, task_index=task)


def _format_indices(segment_path: str, episode_index: int) -> dict[str, int]:
    chunk = re.search(r"chunk[-_](\d+)", segment_path)
    file_match = re.search(r"file[-_](\d+)", segment_path)
    return {
        "episode_index": int(episode_index),
        "episode_chunk": int(chunk.group(1)) if chunk else 0,
        "chunk_index": int(chunk.group(1)) if chunk else 0,
        "file_index": int(file_match.group(1)) if file_match else 0,
    }


def _segment_for_offset(row: dict, episode_offset: int) -> tuple[dict, int]:
    if not 0 <= episode_offset < int(row["length"]):
        raise IndexError(f"episode offset {episode_offset} is outside episode")
    cursor = 0
    for segment in row["segments"]:
        length = int(segment["row_stop"]) - int(segment["row_start"])
        if cursor <= episode_offset < cursor + length:
            return segment, episode_offset - cursor
        cursor += length
    raise IndexError(f"episode offset {episode_offset} is absent from segments")


def video_path_for_segment(
    row: dict,
    role: str,
    segment: dict,
    info: dict,
) -> tuple[Path, bool]:
    """Resolve one manifest segment to its archived video without decoding."""

    if role not in {"external", "wrist"}:
        raise ValueError("role must be external or wrist")
    return video_path_for_view_segment(
        row,
        str(row[f"{role}_view"]),
        segment,
        info,
    )


def video_path_for_view_segment(
    row: dict,
    view_key: str,
    segment: dict,
    info: dict,
) -> tuple[Path, bool]:
    """Resolve an arbitrary roleless FCPG-H view without assigning semantics."""

    if not view_key:
        raise ValueError("view_key cannot be empty")
    pattern = info.get("video_path")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(f"{row['dataset_root']} has no video_path template")
    values = _format_indices(segment["data_path"], int(row["episode_index"]))
    values["video_key"] = str(view_key)
    path = Path(row["dataset_root"]) / pattern.format(**values)
    return path, "{episode_index" in pattern


@lru_cache(maxsize=256)
def _episode_video_metadata(dataset_root: str) -> dict[int, dict]:
    """Load only authoritative v3 video-shard columns once per dataset root."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("PyArrow is required to read LeRobot video metadata") from exc
    root = Path(dataset_root)
    paths = sorted(root.glob("meta/episodes/chunk-*/*.parquet"))
    if not paths:
        return {}
    result: dict[int, dict] = {}
    suffixes = {"chunk_index", "file_index", "from_timestamp", "to_timestamp"}
    for path in paths:
        schema_names = pq.ParquetFile(path).schema_arrow.names
        columns = ["episode_index"] + [
            name
            for name in schema_names
            if name.startswith("videos/") and name.rsplit("/", 1)[-1] in suffixes
        ]
        if len(columns) == 1:
            continue
        for row in pq.read_table(path, columns=columns).to_pylist():
            episode = int(row.pop("episode_index"))
            if episode in result:
                raise ValueError(f"duplicate episode {episode} in {root}/meta/episodes")
            result[episode] = row
    return result


def video_frame_request(row: dict, role: str, episode_offset: int) -> tuple[Path, int]:
    if role not in {"external", "wrist"}:
        raise ValueError("role must be external or wrist")
    return video_frame_request_view(
        row,
        str(row[f"{role}_view"]),
        episode_offset,
    )


def video_frame_request_view(
    row: dict,
    view_key: str,
    episode_offset: int,
) -> tuple[Path, int]:
    """Resolve one arbitrary archived view without a camera-role lookup."""

    if not view_key:
        raise ValueError("view_key cannot be empty")
    root = Path(row["dataset_root"])
    info = json.loads((root / "meta" / "info.json").read_text())
    feature = info.get("features", {}).get(view_key)
    if not isinstance(feature, dict):
        raise ValueError(f"{root} has no feature metadata for {view_key}")
    data_fps = float(info.get("fps", 0))
    video_fps = float(feature.get("info", {}).get("video.fps", data_fps))
    if data_fps <= 0 or video_fps <= 0:
        raise ValueError("data and video FPS must be positive")
    segment, within = _segment_for_offset(row, int(episode_offset))
    path, per_episode = video_path_for_view_segment(row, view_key, segment, info)
    if not per_episode:
        episode_meta = _episode_video_metadata(str(root)).get(int(row["episode_index"]))
        prefix = f"videos/{view_key}/"
        required = tuple(prefix + name for name in (
            "chunk_index",
            "file_index",
            "from_timestamp",
            "to_timestamp",
        ))
        if episode_meta is not None and all(name in episode_meta for name in required):
            values = _format_indices(segment["data_path"], int(row["episode_index"]))
            values["video_key"] = view_key
            values["chunk_index"] = int(episode_meta[prefix + "chunk_index"])
            values["episode_chunk"] = values["chunk_index"]
            values["file_index"] = int(episode_meta[prefix + "file_index"])
            path = root / info["video_path"].format(**values)
            if not path.is_file():
                raise FileNotFoundError(path)
            start_s = float(episode_meta[prefix + "from_timestamp"])
            stop_s = float(episode_meta[prefix + "to_timestamp"])
            frame_index = int(round((start_s + int(episode_offset) / data_fps) * video_fps))
            stop_index = int(round(stop_s * video_fps))
            if frame_index >= stop_index:
                raise IndexError(
                    f"episode offset {episode_offset} exceeds video interval [{start_s},{stop_s})"
                )
            return path, frame_index
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_index = int(episode_offset) if per_episode else int(segment["row_start"]) + within
    return path, int(round(raw_index * video_fps / data_fps))


def letterbox_rgb(frame_bgr: np.ndarray, *, image_size: int = 224) -> np.ndarray:
    import cv2

    global _CV2_RESIZE_THREADS_CONFIGURED
    if not _CV2_RESIZE_THREADS_CONFIGURED:
        # Hundreds of tiny 224px resize jobs are driven by the episode-block
        # loop. Letting each one fan out over every host core causes severe
        # oversubscription alongside PyAV and PyTorch workers.
        cv2.setNumThreads(1)
        _CV2_RESIZE_THREADS_CONFIGURED = True

    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3 or image_size <= 0:
        raise ValueError("frame must be BGR [H,W,3] and image_size positive")
    height, width = frame.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _decode_video_frames_opencv(path: Path, indices: tuple[int, ...]) -> dict[int, np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open archived video {path}")
    decoded: dict[int, np.ndarray] = {}
    try:
        current = -1
        for frame_index in indices:
            if current != frame_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to decode {path} frame {frame_index}")
            current = frame_index + 1
            decoded[frame_index] = frame
    finally:
        capture.release()
    return decoded


def _sequential_decode_groups(
    indices: tuple[int, ...], *, max_gap: int = 128
) -> tuple[tuple[int, ...], ...]:
    """Group nearby exact frames so a codec keyframe is decoded only once."""

    if max_gap <= 0:
        raise ValueError("decode group max_gap must be positive")
    groups: list[list[int]] = []
    for index in indices:
        if not groups or index - groups[-1][-1] > max_gap:
            groups.append([index])
        else:
            groups[-1].append(index)
    return tuple(tuple(group) for group in groups)


def _decode_video_frames_pyav(
    path: Path,
    indices: tuple[int, ...],
    *,
    decoder_threads: int = 4,
) -> dict[int, np.ndarray]:
    import av

    if decoder_threads <= 0:
        raise ValueError("PyAV decoder thread count must be positive")
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # AUTO can create hundreds of decoder workers across a batch and has
        # shown minute-scale tail stalls on mixed AV1/H.264 archives.  A small
        # fixed frame-thread pool is slightly less aggressive but gives stable
        # p95 latency while independent episode blocks provide outer parallelism.
        stream.thread_type = "FRAME"
        stream.codec_context.thread_count = int(decoder_threads)
        rate_value = (
            stream.average_rate
            or getattr(stream, "guessed_rate", None)
            or getattr(stream, "base_rate", None)
        )
        if rate_value is None or stream.time_base is None:
            raise RuntimeError(f"video has no CFR timing metadata: {path}")
        rate = float(rate_value)
        time_base = float(stream.time_base)
        start_pts = int(stream.start_time or 0)
        if rate <= 0 or time_base <= 0:
            raise RuntimeError(f"video has invalid timing metadata: {path}")
        for group in _sequential_decode_groups(indices):
            first, last = group[0], group[-1]
            target_pts = start_pts + int(round((first / rate) / time_base))
            # Seek to the preceding keyframe.  Frame PTS is then converted back
            # to a zero-based CFR index, avoiding dependence on codec-specific
            # packet counts or hardware acceleration.  Nearby requested frames
            # share this seek and sequential decode; the old per-frame seek was
            # the dominant cost in batch-128 temporal training.
            container.seek(max(start_pts, target_pts), stream=stream, backward=True)
            pending = set(group)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                decoded_index = int(
                    round((int(frame.pts) - start_pts) * time_base * rate)
                )
                if decoded_index in pending:
                    decoded[decoded_index] = frame.to_ndarray(format="bgr24")
                    pending.remove(decoded_index)
                if decoded_index >= last:
                    break
            if pending:
                raise RuntimeError(
                    f"failed exact PyAV decode of {path} frames {sorted(pending)}"
                )
    return decoded


def decode_video_frames_bgr(
    path: str | Path,
    indices: Iterable[int],
    *,
    decoder_threads: int = 4,
) -> dict[int, np.ndarray]:
    """Decode exact archived frame indices, preferring CPU PyAV for AV1."""

    resolved = Path(path)
    requested = tuple(sorted(set(int(index) for index in indices)))
    if not requested or requested[0] < 0:
        raise ValueError("video frame indices must be non-empty and non-negative")
    if decoder_threads <= 0:
        raise ValueError("decoder thread count must be positive")
    try:
        import av  # noqa: F401
    except ImportError:
        return _decode_video_frames_opencv(resolved, requested)
    return _decode_video_frames_pyav(
        resolved,
        requested,
        decoder_threads=decoder_threads,
    )


def decode_video_frames_letterboxed_job(
    payload: tuple[str, tuple[int, ...], int, int],
) -> tuple[str, np.ndarray, np.ndarray]:
    """Process-pool worker for exact decode plus 224-style letterboxing.

    Returning one contiguous frame tensor is substantially cheaper to pickle
    than a dictionary of individual arrays.  The worker lives in this
    lightweight archive module so video decoder processes do not construct any
    model or CUDA state.
    """

    path_text, requested, image_size, decoder_threads = payload
    decoded = decode_video_frames_bgr(
        path_text,
        requested,
        decoder_threads=decoder_threads,
    )
    ordered = np.asarray(requested, dtype=np.int64)
    frames = np.stack(
        [letterbox_rgb(decoded[int(index)], image_size=image_size) for index in ordered]
    )
    return path_text, ordered, frames


def decode_episode_frames(
    row: dict,
    roles: Iterable[str],
    episode_offsets: Iterable[int],
    *,
    image_size: int = 224,
) -> dict[str, np.ndarray]:
    """Decode selected archived offsets once per backing video file."""

    offsets = tuple(int(value) for value in episode_offsets)
    if not offsets:
        raise ValueError("at least one episode offset is required")
    role_list = tuple(roles)
    requests: dict[Path, set[int]] = {}
    locations: dict[tuple[str, int], tuple[Path, int]] = {}
    for role in role_list:
        for offset in offsets:
            path, frame_index = video_frame_request(row, role, offset)
            requests.setdefault(path, set()).add(frame_index)
            locations[(role, offset)] = (path, frame_index)

    decoded: dict[tuple[Path, int], np.ndarray] = {}
    for path, indices in requests.items():
        frames = decode_video_frames_bgr(path, indices)
        for frame_index, frame in frames.items():
            decoded[(path, frame_index)] = letterbox_rgb(frame, image_size=image_size)
    return {
        role: np.stack(
            [decoded[locations[(role, offset)]] for offset in offsets], axis=0
        )
        for role in role_list
    }


def decode_episode_views(
    row: dict,
    view_keys: Iterable[str],
    episode_offsets: Iterable[int],
    *,
    image_size: int = 224,
    decoder_threads: int = 4,
) -> dict[str, np.ndarray]:
    """Decode roleless FCPG-H streams in episode-block order."""

    offsets = tuple(int(value) for value in episode_offsets)
    views = tuple(str(value) for value in view_keys)
    if not offsets or not views:
        raise ValueError("at least one offset and view are required")
    requests: dict[Path, set[int]] = {}
    locations: dict[tuple[str, int], tuple[Path, int]] = {}
    for view in views:
        for offset in offsets:
            path, frame_index = video_frame_request_view(row, view, offset)
            requests.setdefault(path, set()).add(frame_index)
            locations[(view, offset)] = (path, frame_index)
    decoded: dict[tuple[Path, int], np.ndarray] = {}
    for path, indices in requests.items():
        for frame_index, frame in decode_video_frames_bgr(
            path,
            indices,
            decoder_threads=decoder_threads,
        ).items():
            decoded[(path, frame_index)] = letterbox_rgb(frame, image_size=image_size)
    return {
        view: np.stack(
            [decoded[locations[(view, offset)]] for offset in offsets], axis=0
        )
        for view in views
    }


__all__ = [
    "EpisodeColumns",
    "decode_episode_frames",
    "decode_episode_views",
    "decode_video_frames_bgr",
    "decode_video_frames_letterboxed_job",
    "letterbox_rgb",
    "read_episode_columns",
    "video_frame_request",
    "video_frame_request_view",
    "video_path_for_segment",
    "video_path_for_view_segment",
]
