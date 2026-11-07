"""Offline dataset for the FCPG recursive observer (milestone M0).

An archived live-deploy round carries far more than the one planning frame the
replay harness walks.  ``plan.json`` pins the joint state the planner saw plus
every grounding channel it produced, and ``execution/receipt.json`` pins the
``observed_states`` trajectory alongside the ``fresh_*``/``post_command_NN_*``
frames captured at those states.  One round therefore yields up to seven
(image, joint state) pairs at *distinct* arm poses, which is exactly the
excitation an observer needs.

Every extracted frame carries:

* the image path and the ``letterbox_v1`` geometry it was written in,
* the six normalized LeRobot joint positions at capture time,
* FK anchor coordinates in the base frame and in the gripper frame
  (:meth:`~fcpg.so101_kinematics.SO101MuJoCoKinematics.forward_anchor_frames`),
* for planning frames, every measured/predicted pixel channel the planner
  recorded, kept **separate** rather than collapsed into one "measurement".

Keeping the channels separate is the point.  ``grounding[view].corrected`` is
not an independent observation of the tips: it is the FCPG local head's
refinement of ``correction_seed``, which is itself the affine ``ProjectionFit``
prediction plus the live-anchor offset.  Fitting a camera model to it would
partly re-fit the affine model that is being replaced.  ``global_locator``
(a learned detector) and, later, motion-consistency measurements (M3) are the
channels that carry information the projection model did not already contain,
so the consumer has to be able to choose, and the audit has to be able to
report which channel it used.

The archives store the 224-px letterboxed planning canvas, never the 640x480
raw frame.  Under ``letterbox_v1`` that is a *uniform* rescale (0.35) plus a
pad offset, so pixels stay square and a pinhole model fitted in canvas pixels
is still a pinhole model; the native-resolution intrinsics are recovered by
dividing focal lengths and principal point by the scale.  The legacy
anisotropic ``(224/640, 224/480)`` map used by the ``error_field`` CLI would
break that and is rejected here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from fcpg.dicg_test_seal import refuse_associated_session_extract
from fcpg.letterbox_geometry import (
    LETTERBOX_V1,
    PLANNING_IMAGE_SIZE,
    PlanningGeometry,
)
from fcpg.so101_kinematics import SO101_ANCHOR_NAMES, SO101MuJoCoKinematics


__all__ = [
    "ANCHOR_COUNT",
    "DATASET_SCHEMA_VERSION",
    "PLAN_FRAME_KEY",
    "FRESH_FRAME_KEY",
    "ObserverFrame",
    "PixelChannel",
    "RoundExtraction",
    "SessionExtraction",
    "iter_round_dirs",
    "round_timestamp_ns",
    "extract_round",
    "extract_session",
    "frame_to_json",
    "frame_from_json",
    "write_frames_jsonl",
    "read_frames_jsonl",
    "measurement_matrix",
]


ANCHOR_COUNT = len(SO101_ANCHOR_NAMES)

#: Bumped whenever the on-disk frame record changes shape.
DATASET_SCHEMA_VERSION = 2

#: ``model_contract`` fields that together pin which static pixel map produced
#: the archived predictions.  Refitting a fresh affine map to the archived
#: ``nominal`` channel leaves 8-19 px of median residual per session, which is
#: impossible for a genuinely static map -- the live-anchor transform is
#: recalibrated and toggled mid-session.  Any camera model fitted across an
#: epoch boundary is therefore fitting a moving target, so the boundary has to
#: travel with the data.
_EPOCH_CONTRACT_KEYS = (
    "projection_calibration_sha256",
    "live_anchor_calibration_sha256",
    "live_anchor_transforms_px",
    "live_anchor_offsets_px",
    "live_anchor_support",
)

PLAN_FRAME_KEY = "plan"
FRESH_FRAME_KEY = "fresh"

_ROUND_DIR_PATTERN = re.compile(r"^r-(\d+)-[0-9a-f]+$")
_POST_COMMAND_IMAGE = "post_command_{index:02d}_{view}.jpg"

#: Pixel channels lifted out of ``plan.json``/``snapshots.json``.  ``nominal``
#: and ``correction_seed`` are model *predictions* and are kept only so the
#: audit can quantify how much a channel merely echoes them.
_GROUNDING_CHANNELS = ("nominal", "correction_seed", "corrected")


@dataclass(frozen=True)
class PixelChannel:
    """One named [3, 2] pixel observation with its per-anchor validity."""

    name: str
    points_px: np.ndarray
    valid: np.ndarray
    accepted: bool | None = None
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points_px, dtype=np.float64)
        if points.shape != (ANCHOR_COUNT, 2):
            raise ValueError(
                f"channel {self.name!r} points must have shape "
                f"[{ANCHOR_COUNT}, 2], got {points.shape}"
            )
        valid = np.asarray(self.valid, dtype=bool)
        if valid.shape != (ANCHOR_COUNT,):
            raise ValueError(
                f"channel {self.name!r} valid must have shape [{ANCHOR_COUNT}]"
            )
        # A non-finite coordinate is not a measurement, whatever the flag says.
        valid = valid & np.all(np.isfinite(points), axis=1)
        object.__setattr__(self, "points_px", points)
        object.__setattr__(self, "valid", valid)
        if self.confidence is not None:
            confidence = np.asarray(self.confidence, dtype=np.float64)
            object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class ObserverFrame:
    """One (image, joint state, FK anchors, measured pixels) observation."""

    session: str
    round_id: str
    round_dir: Path
    view: str
    frame_key: str
    frame_index: int
    image_path: Path
    joints: np.ndarray
    anchors_base_xyz: np.ndarray
    anchors_gripper_xyz: np.ndarray
    geometry: PlanningGeometry
    phase: str | None = None
    timestamp_s: float | None = None
    round_timestamp_ns: int | None = None
    channels: tuple[PixelChannel, ...] = ()
    on_tips: bool | None = None
    tip_separation_px: float | None = None
    pickup_visible: bool | None = None
    three_point_verdict: str | None = None
    #: Identifies the calibration epoch this frame's archived pixel channels
    #: were produced under.  Frames from different epochs must not share one
    #: camera extrinsic.
    epoch_key: str = "unknown"

    def __post_init__(self) -> None:
        joints = np.asarray(self.joints, dtype=np.float64)
        if joints.shape != (6,):
            raise ValueError(f"joints must have shape [6], got {joints.shape}")
        if not np.all(np.isfinite(joints)):
            raise ValueError("joints must be finite")
        for name in ("anchors_base_xyz", "anchors_gripper_xyz"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (ANCHOR_COUNT, 3):
                raise ValueError(
                    f"{name} must have shape [{ANCHOR_COUNT}, 3], got {values.shape}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "joints", joints)
        if self.geometry.protocol != LETTERBOX_V1:
            raise ValueError(
                f"observer frames require {LETTERBOX_V1!r} geometry, got "
                f"{self.geometry.protocol!r}; a squash_v0 archive must be "
                "re-exported, never reinterpreted"
            )
        names = [channel.name for channel in self.channels]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate pixel channel names: {names}")

    def channel(self, name: str) -> PixelChannel | None:
        for entry in self.channels:
            if entry.name == name:
                return entry
        return None

    @property
    def gripper(self) -> float:
        return float(self.joints[5])


@dataclass(frozen=True)
class RoundExtraction:
    frames: tuple[ObserverFrame, ...]
    skipped: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SessionExtraction:
    session: str
    frames: tuple[ObserverFrame, ...]
    skipped: tuple[tuple[str, str], ...]

    @property
    def rounds_used(self) -> int:
        return len({frame.round_id for frame in self.frames})


def round_timestamp_ns(round_dir_name: str) -> int:
    """Parse the nanosecond stamp out of an ``r-<ns>-<hash>`` directory name."""

    match = _ROUND_DIR_PATTERN.match(round_dir_name)
    if match is None:
        raise ValueError(f"not a round directory name: {round_dir_name!r}")
    return int(match.group(1))


def iter_round_dirs(session_dir: Path) -> list[Path]:
    """Return the session's round directories in monotonic capture order.

    Directory names sort lexicographically by their nanosecond stamp only
    while the digit count is constant, so the stamp is parsed and sorted
    numerically instead.  Unparseable names are dropped rather than ordered
    arbitrarily.
    """

    session_dir = Path(session_dir)
    rounds_dir = session_dir / "rounds"
    if not rounds_dir.is_dir():
        raise ValueError(f"no rounds/ directory under {session_dir}")
    entries: list[tuple[int, Path]] = []
    for child in sorted(rounds_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            entries.append((round_timestamp_ns(child.name), child))
        except ValueError:
            continue
    entries.sort(key=lambda item: item[0])
    return [path for _, path in entries]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _geometry_from_contract(contract: Any) -> PlanningGeometry:
    """Build the planning geometry a round was archived in, fail-closed."""

    if not isinstance(contract, dict):
        raise ValueError("plan.json has no model_contract")
    protocol = contract.get("geometry_protocol")
    if protocol != LETTERBOX_V1:
        raise ValueError(
            f"geometry_protocol must be {LETTERBOX_V1!r}, got {protocol!r}"
        )
    source = contract.get("source_image_size")
    if not isinstance(source, (list, tuple)) or len(source) != 2:
        raise ValueError("model_contract.source_image_size must be [width, height]")
    image_size = int(contract.get("planning_image_size", PLANNING_IMAGE_SIZE))
    geometry = PlanningGeometry.letterbox_v1(
        int(source[0]), int(source[1]), image_size=image_size
    )
    recorded = contract.get("content_rect")
    if recorded is not None:
        expected = list(geometry.content_rect)
        if [int(v) for v in recorded] != expected:
            raise ValueError(
                f"model_contract.content_rect {list(recorded)} disagrees with the "
                f"{LETTERBOX_V1} geometry for {source[0]}x{source[1]} ({expected})"
            )
    return geometry


def _epoch_key(
    session: str, contract: Any, view_grounding: Any, view: str
) -> str:
    """Build the calibration-epoch identifier for one round.

    Includes the per-round ``live_anchor_transform.applied`` flag: the same
    contract yields two different pixel maps depending on whether the live
    anchor transform was inside its support radius that round, so the flag is
    part of the epoch, not an attribute of it.
    """

    payload: dict[str, Any] = {"session": session, "view": view}
    if isinstance(contract, dict):
        for key in _EPOCH_CONTRACT_KEYS:
            payload[key] = contract.get(key)
    applied = None
    if isinstance(view_grounding, dict):
        transform = view_grounding.get("live_anchor_transform")
        if isinstance(transform, dict):
            applied = bool(transform.get("applied", False))
        prior = view_grounding.get("prior_source")
        if prior is not None:
            payload["prior_source"] = prior
    payload["live_anchor_applied"] = applied
    return json.dumps(payload, sort_keys=True, default=str)


def _as_points(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    points = np.asarray(value, dtype=np.float64)
    if points.shape != (ANCHOR_COUNT, 2):
        return None
    return points


def _valid_mask(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) == ANCHOR_COUNT:
        return np.asarray([bool(v) for v in value], dtype=bool)
    return np.ones(ANCHOR_COUNT, dtype=bool)


def _locator_top_candidate(
    locator: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Reduce the locator's ``[anchor, candidate, 2]`` heatmap peaks to top-1.

    The learned tip locator emits several ranked candidates per anchor with a
    confidence each.  Only the highest-confidence one is a point measurement;
    the rest are alternatives the locator did not commit to.
    """

    raw = locator.get("points_px")
    if raw is None:
        return None, None
    points = np.asarray(raw, dtype=np.float64)
    confidence_raw = locator.get("confidence")
    confidence = (
        np.asarray(confidence_raw, dtype=np.float64)
        if confidence_raw is not None
        else None
    )
    if points.shape == (ANCHOR_COUNT, 2):
        return points, confidence
    if points.ndim == 3 and points.shape[0] == ANCHOR_COUNT and points.shape[2] == 2:
        if confidence is not None and confidence.shape == points.shape[:2]:
            best = np.argmax(confidence, axis=1)
            selected = points[np.arange(ANCHOR_COUNT), best]
            return selected, confidence[np.arange(ANCHOR_COUNT), best]
        return points[:, 0, :], None
    return None, None


def _grounding_channels(
    view_grounding: Any, snapshot_anchors: Any
) -> tuple[PixelChannel, ...]:
    """Lift every recorded pixel channel for one view, keeping them distinct."""

    channels: list[PixelChannel] = []
    if isinstance(view_grounding, dict):
        base_valid = _valid_mask(view_grounding.get("valid"))
        for name in _GROUNDING_CHANNELS:
            points = _as_points(view_grounding.get(name))
            if points is None:
                continue
            channels.append(PixelChannel(name=name, points_px=points, valid=base_valid))
        locator = view_grounding.get("global_locator")
        if isinstance(locator, dict):
            points, confidence = _locator_top_candidate(locator)
            if points is not None:
                channels.append(
                    PixelChannel(
                        name="global_locator",
                        points_px=points,
                        valid=base_valid,
                        accepted=bool(locator.get("accepted", False)),
                        confidence=confidence,
                    )
                )
    if isinstance(snapshot_anchors, dict):
        points = _as_points(snapshot_anchors.get("points_px"))
        if points is not None:
            source = snapshot_anchors.get("source")
            # The source goes in the channel name so the audit can tell a
            # snapshot that merely echoes ``corrected`` from one that does not.
            name = f"snapshot:{source}" if isinstance(source, str) else "snapshot"
            channels.append(
                PixelChannel(
                    name=name,
                    points_px=points,
                    valid=np.ones(ANCHOR_COUNT, dtype=bool),
                    accepted=bool(snapshot_anchors.get("on_tips", False)),
                )
            )
    return tuple(channels)


def _states_and_frames(
    round_dir: Path,
    view: str,
    plan_state: np.ndarray,
    receipt: Any,
) -> list[tuple[str, int, Path, np.ndarray, float | None]]:
    """Enumerate ``(frame_key, index, image_path, joints, timestamp)`` rows.

    The pairing between ``post_command_NN`` frames and ``observed_states[NN-1]``
    is the one the live archive itself uses
    (``fcpg/so101_live_attempt_archive.py``), so both consumers agree.
    """

    rows: list[tuple[str, int, Path, np.ndarray, float | None]] = []
    plan_image = round_dir / f"{view}_raw.jpg"
    if plan_image.is_file():
        rows.append((PLAN_FRAME_KEY, -1, plan_image, plan_state, None))
    if not isinstance(receipt, dict):
        return rows

    execution = round_dir / "execution"
    before_state = receipt.get("before_state")
    fresh_image = execution / f"fresh_{view}.jpg"
    if fresh_image.is_file() and isinstance(before_state, (list, tuple)):
        values = np.asarray(before_state, dtype=np.float64)
        if values.shape == (6,) and np.all(np.isfinite(values)):
            timestamp = receipt.get("before_timestamp_s")
            rows.append(
                (
                    FRESH_FRAME_KEY,
                    0,
                    fresh_image,
                    values,
                    float(timestamp) if isinstance(timestamp, (int, float)) else None,
                )
            )

    observed = receipt.get("observed_states")
    timestamps = receipt.get("observed_timestamps_s")
    if isinstance(observed, (list, tuple)):
        for index, state in enumerate(observed):
            image = execution / _POST_COMMAND_IMAGE.format(index=index + 1, view=view)
            if not image.is_file():
                continue
            values = np.asarray(state, dtype=np.float64)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                continue
            timestamp = None
            if isinstance(timestamps, (list, tuple)) and index < len(timestamps):
                candidate = timestamps[index]
                if isinstance(candidate, (int, float)):
                    timestamp = float(candidate)
            rows.append(
                (
                    f"post_command_{index + 1:02d}",
                    index + 1,
                    image,
                    values,
                    timestamp,
                )
            )
    return rows


def extract_round(
    round_dir: Path,
    *,
    view: str,
    kinematics: SO101MuJoCoKinematics,
    session: str | None = None,
    include_execution_frames: bool = True,
) -> RoundExtraction:
    """Extract every usable observer frame from one archived round."""

    round_dir = Path(round_dir)
    refuse_associated_session_extract(round_dir)
    session_name = session or round_dir.parent.parent.name
    refuse_associated_session_extract(Path(session_name))
    plan_path = round_dir / "plan.json"
    if not plan_path.is_file():
        return RoundExtraction((), ((round_dir.name, "missing plan.json"),))
    try:
        plan = _load_json(plan_path)
    except json.JSONDecodeError as exc:
        return RoundExtraction((), ((round_dir.name, f"bad plan.json: {exc}"),))
    if not isinstance(plan, dict):
        return RoundExtraction((), ((round_dir.name, "plan.json is not an object"),))

    try:
        geometry = _geometry_from_contract(plan.get("model_contract"))
    except ValueError as exc:
        return RoundExtraction((), ((round_dir.name, f"geometry: {exc}"),))

    plan_state = np.asarray(plan.get("current_state"), dtype=np.float64)
    if plan_state.shape != (6,) or not np.all(np.isfinite(plan_state)):
        return RoundExtraction((), ((round_dir.name, "bad current_state"),))

    receipt: Any = None
    receipt_path = round_dir / "execution" / "receipt.json"
    if include_execution_frames and receipt_path.is_file():
        try:
            receipt = _load_json(receipt_path)
        except json.JSONDecodeError:
            receipt = None

    rows = _states_and_frames(round_dir, view, plan_state, receipt)
    if not rows:
        return RoundExtraction((), ((round_dir.name, f"no {view} frames"),))

    states = np.stack([row[3] for row in rows], axis=0)
    base_xyz, gripper_xyz = kinematics.forward_anchor_frames(states)

    grounding = plan.get("grounding")
    view_grounding = grounding.get(view) if isinstance(grounding, dict) else None
    snapshots_path = round_dir / "snapshots.json"
    snapshot_anchors = None
    snapshot_extras: dict[str, Any] = {}
    if snapshots_path.is_file():
        try:
            snapshots = _load_json(snapshots_path)
        except json.JSONDecodeError:
            snapshots = None
        if isinstance(snapshots, dict):
            anchors = snapshots.get("anchors")
            if isinstance(anchors, dict):
                snapshot_anchors = anchors.get(view)
            snapshot_extras = snapshots
    channels = _grounding_channels(view_grounding, snapshot_anchors)
    epoch_key = _epoch_key(
        session_name, plan.get("model_contract"), view_grounding, view
    )

    phase = plan.get("phase")
    phase_status = plan.get("phase_status")
    verdict = None
    if isinstance(phase_status, dict):
        candidate = phase_status.get("three_point_verdict")
        verdict = str(candidate) if candidate is not None else None
    on_tips = None
    if isinstance(snapshot_anchors, dict) and "on_tips" in snapshot_anchors:
        on_tips = bool(snapshot_anchors["on_tips"])
    separation = None
    tip_sep = snapshot_extras.get("tip_separation_px")
    if isinstance(tip_sep, dict):
        candidate = tip_sep.get(view)
        if isinstance(candidate, (int, float)):
            separation = float(candidate)
    elif isinstance(tip_sep, (int, float)):
        separation = float(tip_sep)
    pickup_visible = None
    visible = snapshot_extras.get("pickup_visible")
    if isinstance(visible, bool):
        pickup_visible = visible
    elif isinstance(visible, dict):
        candidate = visible.get(view)
        if isinstance(candidate, bool):
            pickup_visible = candidate

    round_id = str(plan.get("round_id", round_dir.name))
    try:
        stamp_ns: int | None = round_timestamp_ns(round_dir.name)
    except ValueError:
        stamp_ns = None

    frames: list[ObserverFrame] = []
    for position, (frame_key, frame_index, image_path, joints, timestamp) in enumerate(
        rows
    ):
        frames.append(
            ObserverFrame(
                session=session_name,
                round_id=round_id,
                round_dir=round_dir,
                view=view,
                frame_key=frame_key,
                frame_index=frame_index,
                image_path=image_path,
                joints=joints,
                anchors_base_xyz=base_xyz[position],
                anchors_gripper_xyz=gripper_xyz[position],
                geometry=geometry,
                phase=str(phase) if phase is not None else None,
                timestamp_s=timestamp,
                round_timestamp_ns=stamp_ns,
                # Only the planning frame's pixels were measured at that pose.
                channels=channels if frame_key == PLAN_FRAME_KEY else (),
                on_tips=on_tips if frame_key == PLAN_FRAME_KEY else None,
                tip_separation_px=(
                    separation if frame_key == PLAN_FRAME_KEY else None
                ),
                pickup_visible=(
                    pickup_visible if frame_key == PLAN_FRAME_KEY else None
                ),
                three_point_verdict=(
                    verdict if frame_key == PLAN_FRAME_KEY else None
                ),
                epoch_key=epoch_key,
            )
        )
    return RoundExtraction(tuple(frames), ())


def extract_session(
    session_dir: Path,
    *,
    view: str = "column",
    kinematics: SO101MuJoCoKinematics,
    include_execution_frames: bool = True,
    limit: int | None = None,
) -> SessionExtraction:
    """Extract observer frames from every round of one archived session."""

    session_dir = Path(session_dir)
    refuse_associated_session_extract(session_dir)
    rounds = iter_round_dirs(session_dir)
    if limit is not None:
        rounds = rounds[: max(0, int(limit))]
    frames: list[ObserverFrame] = []
    skipped: list[tuple[str, str]] = []
    for round_dir in rounds:
        result = extract_round(
            round_dir,
            view=view,
            kinematics=kinematics,
            session=session_dir.name,
            include_execution_frames=include_execution_frames,
        )
        frames.extend(result.frames)
        skipped.extend(result.skipped)
    return SessionExtraction(
        session=session_dir.name, frames=tuple(frames), skipped=tuple(skipped)
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _channel_to_json(channel: PixelChannel) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": channel.name,
        "points_px": channel.points_px.tolist(),
        "valid": channel.valid.tolist(),
    }
    if channel.accepted is not None:
        payload["accepted"] = bool(channel.accepted)
    if channel.confidence is not None:
        payload["confidence"] = np.asarray(channel.confidence).tolist()
    return payload


def frame_to_json(frame: ObserverFrame) -> dict[str, Any]:
    """Serialize one frame; paths stay absolute so records are self-contained."""

    payload: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "session": frame.session,
        "round_id": frame.round_id,
        "round_dir": str(frame.round_dir),
        "view": frame.view,
        "frame_key": frame.frame_key,
        "frame_index": frame.frame_index,
        "image_path": str(frame.image_path),
        "joints": frame.joints.tolist(),
        "anchors_base_xyz": frame.anchors_base_xyz.tolist(),
        "anchors_gripper_xyz": frame.anchors_gripper_xyz.tolist(),
        "geometry": frame.geometry.to_dict(),
        "channels": [_channel_to_json(channel) for channel in frame.channels],
        "epoch_key": frame.epoch_key,
    }
    for key, value in (
        ("phase", frame.phase),
        ("timestamp_s", frame.timestamp_s),
        ("round_timestamp_ns", frame.round_timestamp_ns),
        ("on_tips", frame.on_tips),
        ("tip_separation_px", frame.tip_separation_px),
        ("pickup_visible", frame.pickup_visible),
        ("three_point_verdict", frame.three_point_verdict),
    ):
        if value is not None:
            payload[key] = value
    return payload


def frame_from_json(payload: dict[str, Any]) -> ObserverFrame:
    """Rebuild a frame, rejecting records written by a different schema."""

    version = int(payload.get("schema_version", -1))
    if version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"observer frame schema_version must be {DATASET_SCHEMA_VERSION}, "
            f"got {version}"
        )
    geometry_payload = payload.get("geometry")
    if not isinstance(geometry_payload, dict):
        raise ValueError("frame record is missing geometry")
    protocol = geometry_payload.get("geometry_protocol")
    if protocol != LETTERBOX_V1:
        raise ValueError(
            f"frame geometry must be {LETTERBOX_V1!r}, got {protocol!r}"
        )
    geometry = PlanningGeometry.letterbox_v1(
        int(geometry_payload["source_width"]),
        int(geometry_payload["source_height"]),
        image_size=int(geometry_payload.get("planning_image_size", PLANNING_IMAGE_SIZE)),
    )
    channels = tuple(
        PixelChannel(
            name=str(entry["name"]),
            points_px=np.asarray(entry["points_px"], dtype=np.float64),
            valid=np.asarray(entry["valid"], dtype=bool),
            accepted=entry.get("accepted"),
            confidence=(
                np.asarray(entry["confidence"], dtype=np.float64)
                if entry.get("confidence") is not None
                else None
            ),
        )
        for entry in payload.get("channels", [])
    )
    return ObserverFrame(
        session=str(payload["session"]),
        round_id=str(payload["round_id"]),
        round_dir=Path(payload["round_dir"]),
        view=str(payload["view"]),
        frame_key=str(payload["frame_key"]),
        frame_index=int(payload["frame_index"]),
        image_path=Path(payload["image_path"]),
        joints=np.asarray(payload["joints"], dtype=np.float64),
        anchors_base_xyz=np.asarray(payload["anchors_base_xyz"], dtype=np.float64),
        anchors_gripper_xyz=np.asarray(
            payload["anchors_gripper_xyz"], dtype=np.float64
        ),
        geometry=geometry,
        phase=payload.get("phase"),
        timestamp_s=payload.get("timestamp_s"),
        round_timestamp_ns=payload.get("round_timestamp_ns"),
        channels=channels,
        on_tips=payload.get("on_tips"),
        tip_separation_px=payload.get("tip_separation_px"),
        pickup_visible=payload.get("pickup_visible"),
        three_point_verdict=payload.get("three_point_verdict"),
        epoch_key=str(payload.get("epoch_key", "unknown")),
    )


def write_frames_jsonl(path: Path, frames: Iterable[ObserverFrame]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(frame_to_json(frame)) + "\n")
            count += 1
    return count


def read_frames_jsonl(path: Path) -> Iterator[ObserverFrame]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield frame_from_json(json.loads(line))


def measurement_matrix(
    frames: Sequence[ObserverFrame], channel: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack one channel into ``(pixels, anchors_base_xyz, valid)`` arrays.

    Frames without the requested channel are dropped, so a caller asking for
    ``motion_flow`` gets only the frames that actually carry it rather than a
    silently zero-filled block.
    """

    pixels: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    for frame in frames:
        entry = frame.channel(channel)
        if entry is None:
            continue
        pixels.append(entry.points_px)
        anchors.append(frame.anchors_base_xyz)
        valid.append(entry.valid)
    if not pixels:
        empty_px = np.zeros((0, ANCHOR_COUNT, 2), dtype=np.float64)
        empty_xyz = np.zeros((0, ANCHOR_COUNT, 3), dtype=np.float64)
        empty_valid = np.zeros((0, ANCHOR_COUNT), dtype=bool)
        return empty_px, empty_xyz, empty_valid
    return (
        np.stack(pixels, axis=0),
        np.stack(anchors, axis=0),
        np.stack(valid, axis=0),
    )
