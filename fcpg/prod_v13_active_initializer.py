"""Offline active fingertip initialization for the Prod-V13 observer.

The initializer needs no pixel click, camera calibration, optical flow, mask
model, camera identity, or network access.  While the five arm joints remain
fixed, it uses one already-supported gripper close/open/restore pulse as a
known excitation.  A visual point is accepted as the moving fingertip only
when its trajectory is both large and repeatable on the closing and opening
branches at the same measured aperture.  The closed endpoint supplies the
fixed fingertip; the restored endpoint supplies the current moving fingertip.

CoTracker remains a frozen local point-measurement tool.  The command-locked
loop response only ranks its candidates; that ranking is not independent
physical proof that the selected point is a fingertip.

This initializer-specific tracker heuristic is not the authoritative
zero-training evidence screen.  Archived aperture-response evidence is
evaluated by :mod:`fcpg.aperture_lock_oracle`, whose support cannot by itself
be interpreted as a jaw or fingertip location.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch


VIEW_ORDER = ("wrist", "column")
MINIMUM_BRANCH_FRAMES = 8
APERTURE_LEVELS = 32
MAXIMUM_CANDIDATES = 192
MINIMUM_VISIBILITY = 0.90
MINIMUM_MONOTONICITY = 0.90
MAXIMUM_LOOP_RMSE_PX = 3.0
COMPONENT_QUANTILES = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97)
# The command-response component describes the pixels swept during the pulse,
# not the complete jaw silhouette in the first open frame.  One V12 patch plus
# a small sub-patch allowance reaches the distal open tip on the saved wrist
# pulses without opening the search to the full image.
COMMAND_LOCK_SUPPORT_RADIUS_PX = 18


@dataclass(frozen=True)
class ActiveViewInitialization:
    fixed_tip_px: np.ndarray
    moving_tip_px: np.ndarray
    selected_track_px: np.ndarray
    selected_visible: np.ndarray
    displacement_px: float
    loop_rmse_px: float
    monotonicity: float
    visibility_fraction: float
    runner_up_displacement_px: float
    response_bbox_xyxy: tuple[int, int, int, int]
    candidate_count: int
    selected_candidate_index: int

    @property
    def tip_px(self) -> np.ndarray:
        return np.stack((self.fixed_tip_px, self.moving_tip_px)).astype(
            np.float32
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "slot_semantics": ["fixed_closed_endpoint", "moving_restored_endpoint"],
            "fixed_tip_px": self.fixed_tip_px.tolist(),
            "moving_tip_px": self.moving_tip_px.tolist(),
            "displacement_px": float(self.displacement_px),
            "loop_rmse_px": float(self.loop_rmse_px),
            "monotonicity": float(self.monotonicity),
            "visibility_fraction": float(self.visibility_fraction),
            "runner_up_displacement_px": float(
                self.runner_up_displacement_px
            ),
            "response_bbox_xyxy": list(self.response_bbox_xyxy),
            "candidate_count": int(self.candidate_count),
            "selected_candidate_index": int(self.selected_candidate_index),
        }


@dataclass(frozen=True)
class ActiveFingertipInitialization:
    tip_px: np.ndarray
    views: dict[str, ActiveViewInitialization]
    final_frames: dict[str, np.ndarray]
    final_state: np.ndarray
    closed_frame_index: int
    restored_frame_index: int

    def __post_init__(self) -> None:
        points = np.asarray(self.tip_px, dtype=np.float32)
        state = np.asarray(self.final_state, dtype=np.float32)
        if points.shape != (2, 2, 2) or not np.all(np.isfinite(points)):
            raise ValueError("automatic fingertip points must be finite [V,2,2]")
        if np.any(points < 0.0) or np.any(points > 223.0):
            raise ValueError("automatic fingertip points must be inside the image")
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("automatic fingertip state must be finite [6]")
        if set(self.views) != set(VIEW_ORDER) or set(self.final_frames) != set(
            VIEW_ORDER
        ):
            raise ValueError("automatic fingertip initialization needs both views")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": "command_locked_close_open_loop_v1",
            "requires_mouse_query": False,
            "network_access": False,
            "optical_flow": False,
            "camera_calibration": False,
            "closed_frame_index": int(self.closed_frame_index),
            "restored_frame_index": int(self.restored_frame_index),
            "views": {
                view: self.views[view].diagnostics() for view in VIEW_ORDER
            },
        }


def _validate_pulse(
    frames_by_view: dict[str, np.ndarray], states: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if set(frames_by_view) != set(VIEW_ORDER):
        raise ValueError("active initialization requires wrist and column RGB")
    checked: dict[str, np.ndarray] = {}
    length: int | None = None
    for view in VIEW_ORDER:
        frames = np.asarray(frames_by_view[view])
        if (
            frames.ndim != 4
            or frames.shape[1:] != (224, 224, 3)
            or frames.dtype != np.uint8
        ):
            raise ValueError(f"{view} pulse must be uint8 [T,224,224,3]")
        if length is None:
            length = len(frames)
        elif len(frames) != length:
            raise ValueError("wrist and column pulse lengths differ")
        checked[view] = frames
    value = np.asarray(states, dtype=np.float32)
    if length is None or value.shape != (length, 6) or not np.all(
        np.isfinite(value)
    ):
        raise ValueError("pulse states must be finite [T,6] and align with RGB")
    if length < 2 * MINIMUM_BRANCH_FRAMES + 1:
        raise ValueError("pulse is too short for close/open loop identification")
    arm_drift = float(np.max(np.ptp(value[:, :5], axis=0)))
    if arm_drift > 2.0:
        raise ValueError(
            f"arm moved {arm_drift:.3f} during active fingertip initialization"
        )
    return checked, value


def _branch_interpolate(
    values: np.ndarray, aperture: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float32)
    command = np.asarray(aperture, dtype=np.float32)
    levels = np.asarray(grid, dtype=np.float32)
    if samples.shape[0] != len(command) or samples.shape[0] < 2:
        raise ValueError("branch samples and aperture do not align")
    order = np.argsort(command, kind="stable")
    command = command[order]
    samples = samples[order]
    unique, inverse = np.unique(command, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("pulse branch has fewer than two aperture levels")
    flat = samples.reshape(len(samples), -1).astype(np.float64)
    sums = np.zeros((len(unique), flat.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, flat)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    means = sums / counts[:, None]
    upper = np.searchsorted(unique, levels, side="right")
    upper = np.clip(upper, 1, len(unique) - 1)
    lower = upper - 1
    span = unique[upper] - unique[lower]
    alpha = (levels - unique[lower]) / np.maximum(span, 1.0e-8)
    result = means[lower] * (1.0 - alpha[:, None]) + means[upper] * alpha[:, None]
    return result.reshape((len(levels),) + samples.shape[1:]).astype(np.float32)


def command_lock_map(
    frames: np.ndarray, aperture: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Return a close/open repeatability map without optical flow."""

    images = np.asarray(frames)
    command = np.asarray(aperture, dtype=np.float32)
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) != len(command):
        raise ValueError("RGB and aperture do not align")
    closed = int(np.argmin(command))
    restored = closed + int(np.argmax(command[closed:]))
    if closed < MINIMUM_BRANCH_FRAMES or restored - closed < MINIMUM_BRANCH_FRAMES:
        raise ValueError("pulse lacks complete closing and opening branches")
    gray = np.stack(
        [
            cv2.GaussianBlur(
                cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (3, 3), 0
            )
            for frame in images
        ]
    ).astype(np.float32)
    centered = gray - np.median(gray, axis=(1, 2), keepdims=True)
    low = max(
        float(np.min(command[: closed + 1])),
        float(np.min(command[closed : restored + 1])),
    )
    high = min(
        float(np.max(command[: closed + 1])),
        float(np.max(command[closed : restored + 1])),
    )
    if high <= low:
        raise ValueError("closing and opening branches have no aperture overlap")
    grid = np.linspace(low, high, APERTURE_LEVELS, dtype=np.float32)
    closing = _branch_interpolate(
        centered[: closed + 1], command[: closed + 1], grid
    )
    opening = _branch_interpolate(
        centered[closed : restored + 1],
        command[closed : restored + 1],
        grid,
    )
    paired = 0.5 * (closing + opening)
    command_signal = np.var(paired, axis=0)
    loop_disagreement = np.mean((closing - opening) ** 2, axis=0)
    score = command_signal / (loop_disagreement + 64.0)
    score = cv2.GaussianBlur(score.astype(np.float32), (7, 7), 0)
    if not np.all(np.isfinite(score)):
        raise ValueError("command-lock response is non-finite")
    return score, closed, restored


def command_lock_support(
    score: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    response = np.asarray(score, dtype=np.float32)
    if response.shape != (224, 224) or not np.all(np.isfinite(response)):
        raise ValueError("command-lock map must be finite [224,224]")
    image_area = int(response.size)
    candidates: list[
        tuple[float, np.ndarray, tuple[int, int, int, int]]
    ] = []
    for quantile in COMPONENT_QUANTILES:
        threshold = float(np.quantile(response, quantile))
        mask = (response >= threshold).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for label in range(1, count):
            x, y, width, height, area = map(int, stats[label])
            if area < 32 or area > int(round(0.22 * image_area)):
                continue
            component = labels == label
            component_score = float(area) * float(np.mean(response[component]))
            candidates.append(
                (
                    component_score,
                    component,
                    (x, y, x + width, y + height),
                )
            )
    if not candidates:
        raise ValueError("no command-locked image component was found")
    _, component, bbox = max(candidates, key=lambda row: row[0])
    # The response is measured over the whole close/open sweep.  At the open
    # endpoint the distal metal edge can sit just outside that response blob:
    # on three saved same-pose wrist pulses its repeatable tip was at y~=123,
    # while the response bbox began at y=130--133.  Using the exact blob made
    # the first-frame query land on the table at y=136--145 and produced an
    # 18.6 px repeated-pulse error.  A circular dilation restores the visible
    # open-tip neighbourhood while excluding the unrelated corners admitted by
    # the historical padded rectangle.
    radius = int(COMMAND_LOCK_SUPPORT_RADIUS_PX)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    support = cv2.dilate(
        component.astype(np.uint8, copy=False),
        kernel,
        iterations=1,
    )
    return support, bbox


def image_candidates(
    frame: np.ndarray, support: np.ndarray, *, maximum: int = MAXIMUM_CANDIDATES
) -> np.ndarray:
    image = np.asarray(frame)
    mask = np.asarray(support, dtype=np.uint8)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("candidate image must be uint8 [224,224,3]")
    if mask.shape != (224, 224) or maximum < 2:
        raise ValueError("candidate support/maximum is invalid")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=maximum,
        qualityLevel=0.005,
        minDistance=3,
        mask=(mask * 255).astype(np.uint8),
        blockSize=5,
        useHarrisDetector=False,
    )
    rows = [] if corners is None else [point[0] for point in corners]
    ys, xs = np.nonzero(mask)
    if len(xs):
        for y in range(int(ys.min()), int(ys.max()) + 1, 6):
            for x in range(int(xs.min()), int(xs.max()) + 1, 6):
                if mask[y, x]:
                    rows.append(np.asarray((x, y), dtype=np.float32))
    if not rows:
        raise ValueError("command-locked support contains no RGB candidates")
    points = np.asarray(rows, dtype=np.float32)
    keys = np.round(points * 2.0).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)][:maximum]


@torch.inference_mode()
def track_candidates(
    tracker: torch.nn.Module,
    frames: np.ndarray,
    points_px: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(frames)
    points = np.asarray(points_px, dtype=np.float32)
    step = int(getattr(tracker, "step", -1))
    if images.ndim != 4 or images.shape[1:] != (224, 224, 3):
        raise ValueError("CoTracker RGB must be [T,224,224,3]")
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("CoTracker candidates must be non-empty [N,2]")
    if step != 8 or len(images) <= step:
        raise ValueError("active initializer requires CoTracker3-online step=8")
    video = torch.from_numpy(images[None]).to(device)
    video = video.permute(0, 1, 4, 2, 3).float().div_(255.0)
    queries = torch.cat(
        (
            torch.zeros((1, len(points), 1), device=device),
            torch.from_numpy(points)[None].to(device),
        ),
        dim=-1,
    )
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        initialized = tracker(
            video_chunk=video[:, :step],
            is_first_step=True,
            queries=queries,
            grid_size=0,
            add_support_grid=True,
        )
        if initialized != (None, None):
            raise RuntimeError("CoTracker initialization returned a prediction")
        begin = 0
        tracks = visible = None
        while len(images) - begin > step:
            chunk = video[:, begin : min(begin + 2 * step, len(images))]
            tracks, visible = tracker(
                video_chunk=chunk,
                is_first_step=False,
                grid_size=0,
                add_support_grid=True,
            )
            begin += step
    if tracks is None or visible is None:
        raise RuntimeError("CoTracker produced no active-initialization output")
    track_array = tracks[0, :, : len(points)].float().cpu().numpy()
    visible_array = visible[0, :, : len(points)].bool().cpu().numpy()
    if visible_array.ndim == 3 and visible_array.shape[-1] == 1:
        visible_array = visible_array[..., 0]
    expected_tracks = (len(images), len(points), 2)
    expected_visible = (len(images), len(points))
    if track_array.shape != expected_tracks or visible_array.shape != expected_visible:
        raise ValueError("CoTracker active-initialization shape drifted")
    return track_array, visible_array


def select_command_locked_track(
    tracks_px: np.ndarray,
    visible: np.ndarray,
    aperture: np.ndarray,
    *,
    closed_frame_index: int,
    restored_frame_index: int,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    tracks = np.asarray(tracks_px, dtype=np.float32)
    mask = np.asarray(visible, dtype=bool)
    command = np.asarray(aperture, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError("candidate tracks must be [T,N,2]")
    if mask.shape != tracks.shape[:2] or command.shape != (len(tracks),):
        raise ValueError("track visibility/aperture do not align")
    closed = int(closed_frame_index)
    restored = int(restored_frame_index)
    if not 0 < closed < restored < len(tracks):
        raise ValueError("pulse branch indices are invalid")
    low = max(
        float(np.min(command[: closed + 1])),
        float(np.min(command[closed : restored + 1])),
    )
    high = min(
        float(np.max(command[: closed + 1])),
        float(np.max(command[closed : restored + 1])),
    )
    grid = np.linspace(low, high, APERTURE_LEVELS, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for index in range(tracks.shape[1]):
        finite = np.all(np.isfinite(tracks[:, index]), axis=-1)
        visibility = float(np.mean(mask[:, index] & finite))
        if visibility < MINIMUM_VISIBILITY:
            continue
        tail_start = max(0, len(tracks) - 3)
        tail_valid = mask[tail_start:, index] & finite[tail_start:]
        if int(np.count_nonzero(tail_valid)) < 2:
            continue
        closing = _branch_interpolate(
            tracks[: closed + 1, index], command[: closed + 1], grid
        )
        opening = _branch_interpolate(
            tracks[closed : restored + 1, index],
            command[closed : restored + 1],
            grid,
        )
        mean_track = 0.5 * (closing + opening)
        loop_rmse = float(
            np.sqrt(np.mean(np.sum((closing - opening) ** 2, axis=-1)))
        )
        displacement = float(np.linalg.norm(mean_track[-1] - mean_track[0]))
        path_length = float(np.sum(np.linalg.norm(np.diff(mean_track, axis=0), axis=-1)))
        monotonicity = displacement / max(path_length, 1.0e-8)
        if (
            loop_rmse > MAXIMUM_LOOP_RMSE_PX
            or monotonicity < MINIMUM_MONOTONICITY
        ):
            continue
        rows.append(
            {
                "index": int(index),
                "displacement_px": displacement,
                "loop_rmse_px": loop_rmse,
                "monotonicity": monotonicity,
                "visibility_fraction": visibility,
                "fixed_tip_px": mean_track[0].astype(np.float32),
                "moving_tip_px": np.median(
                    tracks[tail_start:, index][tail_valid], axis=0
                ).astype(np.float32),
            }
        )
    if not rows:
        raise ValueError("no repeatable command-locked candidate track")
    rows.sort(key=lambda row: float(row["displacement_px"]), reverse=True)
    return int(rows[0]["index"]), rows[0], rows


def initialize_fingertips_from_pulse(
    *,
    tracker: torch.nn.Module,
    frames_by_view: dict[str, np.ndarray],
    states: np.ndarray,
    device: torch.device,
) -> ActiveFingertipInitialization:
    """Produce both-view fixed/moving tip queries from one completed pulse."""

    frames, value = _validate_pulse(frames_by_view, states)
    aperture = value[:, 5]
    view_results: dict[str, ActiveViewInitialization] = {}
    closed_indices: list[int] = []
    restored_indices: list[int] = []
    points_by_view: list[np.ndarray] = []
    for view in VIEW_ORDER:
        response, closed, restored = command_lock_map(frames[view], aperture)
        support, bbox = command_lock_support(response)
        points = image_candidates(frames[view][0], support)
        tracks, visible = track_candidates(
            tracker, frames[view], points, device=device
        )
        selected_index, selected, rows = select_command_locked_track(
            tracks,
            visible,
            aperture,
            closed_frame_index=closed,
            restored_frame_index=restored,
        )
        runner_up = (
            float(rows[1]["displacement_px"]) if len(rows) > 1 else 0.0
        )
        result = ActiveViewInitialization(
            fixed_tip_px=np.asarray(selected["fixed_tip_px"], dtype=np.float32),
            moving_tip_px=np.asarray(selected["moving_tip_px"], dtype=np.float32),
            selected_track_px=tracks[:, selected_index].astype(np.float32),
            selected_visible=visible[:, selected_index].astype(bool),
            displacement_px=float(selected["displacement_px"]),
            loop_rmse_px=float(selected["loop_rmse_px"]),
            monotonicity=float(selected["monotonicity"]),
            visibility_fraction=float(selected["visibility_fraction"]),
            runner_up_displacement_px=runner_up,
            response_bbox_xyxy=bbox,
            candidate_count=len(points),
            selected_candidate_index=selected_index,
        )
        view_results[view] = result
        points_by_view.append(result.tip_px)
        closed_indices.append(closed)
        restored_indices.append(restored)
    if len(set(closed_indices)) != 1 or len(set(restored_indices)) != 1:
        raise ValueError("dual-view pulse branch indices disagree")
    return ActiveFingertipInitialization(
        tip_px=np.stack(points_by_view).astype(np.float32),
        views=view_results,
        final_frames={view: frames[view][-1].copy() for view in VIEW_ORDER},
        final_state=value[-1].copy(),
        closed_frame_index=closed_indices[0],
        restored_frame_index=len(value) - 1,
    )


__all__ = [
    "ActiveFingertipInitialization",
    "ActiveViewInitialization",
    "COMMAND_LOCK_SUPPORT_RADIUS_PX",
    "command_lock_map",
    "command_lock_support",
    "image_candidates",
    "initialize_fingertips_from_pulse",
    "select_command_locked_track",
    "track_candidates",
]
