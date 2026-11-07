"""Portable recording contract for read-only SO-101 shadow inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from fcpg.letterbox_geometry import LETTERBOX_V1, SQUASH_V0
from fcpg.so101_kinematics import SO101_JOINT_NAMES


SESSION_FORMAT = "so101-shadow-session-v1"


def observation_state(observation: Mapping[str, object]) -> np.ndarray:
    """Extract normalized SO-101 joints in the repository's fixed order."""

    missing = [name for name in SO101_JOINT_NAMES if name not in observation]
    if missing:
        raise KeyError(f"SO-101 observation is missing joints: {missing}")
    state = np.asarray([observation[name] for name in SO101_JOINT_NAMES], dtype=np.float32)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("SO-101 observation joints must be six finite scalars")
    return state


def observation_rgb(observation: Mapping[str, object], camera: str) -> np.ndarray:
    """Extract one RGB camera frame and enforce the model-facing image contract."""

    if camera not in observation:
        raise KeyError(f"SO-101 observation is missing camera {camera!r}")
    frame = np.asarray(observation[camera])
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"camera frame must have shape [H, W, 3], got {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"camera frame must be uint8 RGB, got {frame.dtype}")
    return np.ascontiguousarray(frame)


def ordered_action(action: Mapping[str, object]) -> np.ndarray:
    """Extract a leader/policy command in the same six-joint order as state."""

    missing = [name for name in SO101_JOINT_NAMES if name not in action]
    if missing:
        raise KeyError(f"SO-101 action is missing joints: {missing}")
    values = np.asarray([action[name] for name in SO101_JOINT_NAMES], dtype=np.float32)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("SO-101 action must contain six finite scalars")
    return values


def resize_rgb(frame: np.ndarray, image_size: int) -> np.ndarray:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    values = np.asarray(frame)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError("resize_rgb expects uint8 RGB [H, W, 3]")
    if values.shape[:2] == (image_size, image_size):
        return np.ascontiguousarray(values)
    from PIL import Image

    resized = Image.fromarray(values, mode="RGB").resize(
        (image_size, image_size), Image.Resampling.LANCZOS
    )
    return np.asarray(resized, dtype=np.uint8)


@dataclass(frozen=True)
class SO101ShadowSession:
    root: Path
    manifest: dict
    frames: np.ndarray
    states: np.ndarray
    timestamps_s: np.ndarray
    actions: np.ndarray | None

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def _validate_arrays(
    frames: np.ndarray,
    states: np.ndarray,
    timestamps_s: np.ndarray,
    actions: np.ndarray | None,
) -> None:
    count = len(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("session frames must be uint8 [N, H, W, 3]")
    if states.shape != (count, 6) or not np.all(np.isfinite(states)):
        raise ValueError("session states must be finite [N, 6]")
    if timestamps_s.shape != (count,) or not np.all(np.isfinite(timestamps_s)):
        raise ValueError("session timestamps must be finite [N]")
    if count > 1 and np.any(np.diff(timestamps_s) <= 0):
        raise ValueError("session timestamps must be strictly increasing")
    if actions is not None and (
        actions.shape != (count, 6) or not np.all(np.isfinite(actions))
    ):
        raise ValueError("session actions must be finite [N, 6]")


def save_shadow_session(
    root: str | Path,
    *,
    frames: np.ndarray,
    states: np.ndarray,
    timestamps_s: np.ndarray,
    actions: np.ndarray | None,
    source: str,
    camera: str,
    fps: float,
    action_source: str,
    geometry_protocol: str,
    control_sent_by_recorder: bool = False,
    extra_manifest: Mapping[str, object] | None = None,
) -> SO101ShadowSession:
    """Write one compact session without any robot-control capability.

    ``geometry_protocol`` is keyword-only and has NO default on purpose.  Every
    recorder resizes raw camera frames onto a square planning canvas, and both
    supported protocols produce the identical ``224x224`` shape -- so a reader
    cannot recover which one ran from the pixels, the manifest's
    ``image_shape``, or anything else on disk.  A default would therefore
    declare a geometry nobody verified, which is precisely the fail-open this
    parameter closes: a session recorded under one protocol and replayed under
    the other looks completely normal while every derived pixel carries the pad
    affine.  Forcing the recorder to state it means a future change to the
    capture preprocessing cannot silently keep an now-untrue declaration.
    """

    root = Path(root).expanduser().resolve()
    frames = np.asarray(frames)
    states = np.asarray(states, dtype=np.float32)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    actions = None if actions is None else np.asarray(actions, dtype=np.float32)
    _validate_arrays(frames, states, timestamps_s, actions)
    if not source or not camera or not action_source:
        raise ValueError("source, camera and action_source must be non-empty")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if geometry_protocol not in (SQUASH_V0, LETTERBOX_V1):
        raise ValueError(
            f"geometry_protocol must be {SQUASH_V0!r} or {LETTERBOX_V1!r}, got "
            f"{geometry_protocol!r}; a square planning frame is consistent with "
            "both, so the recorder must declare which one preprocessed these "
            "pixels"
        )
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"shadow-session output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    np.save(root / "frames.npy", frames)
    np.save(root / "states.npy", states)
    np.save(root / "timestamps_s.npy", timestamps_s)
    if actions is not None:
        np.save(root / "actions.npy", actions)
    manifest = {
        "format": SESSION_FORMAT,
        "source": source,
        "camera": camera,
        "fps": float(fps),
        "frames": len(frames),
        "image_shape": list(frames.shape[1:]),
        "joint_names": list(SO101_JOINT_NAMES),
        "joint_units": "LeRobot normalized positions; arm [-100,100], gripper [0,100]",
        "action_source": action_source,
        "contains_actions": actions is not None,
        "control_sent_by_recorder": bool(control_sent_by_recorder),
        # Top-level, so the reserved-field overlap guard below covers it for
        # free: `extra_manifest` can no longer shadow the declaration with an
        # unverified one.
        "geometry_protocol": str(geometry_protocol),
    }
    if extra_manifest:
        overlap = set(manifest) & set(extra_manifest)
        if overlap:
            raise ValueError(f"extra manifest cannot replace reserved fields: {sorted(overlap)}")
        manifest.update(extra_manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return load_shadow_session(root, allow_control=control_sent_by_recorder)


def load_shadow_session(
    root: str | Path,
    *,
    allow_control: bool = False,
) -> SO101ShadowSession:
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != SESSION_FORMAT:
        raise ValueError(f"unsupported shadow-session format: {manifest.get('format')!r}")
    if tuple(manifest.get("joint_names", ())) != SO101_JOINT_NAMES:
        raise ValueError("shadow-session joint order differs from SO-101 contract")
    frames = np.load(root / "frames.npy", mmap_mode="r")
    states = np.load(root / "states.npy", mmap_mode="r")
    timestamps_s = np.load(root / "timestamps_s.npy", mmap_mode="r")
    action_path = root / "actions.npy"
    actions = np.load(action_path, mmap_mode="r") if action_path.is_file() else None
    _validate_arrays(frames, states, timestamps_s, actions)
    if int(manifest.get("frames", -1)) != len(frames):
        raise ValueError("shadow-session manifest frame count does not match arrays")
    if bool(manifest.get("contains_actions")) != (actions is not None):
        raise ValueError("shadow-session action manifest does not match arrays")
    control_sent = manifest.get("control_sent_by_recorder")
    if control_sent not in (False, True):
        raise ValueError("shadow-session must declare whether control was sent")
    if control_sent and not allow_control:
        raise ValueError("shadow-session must explicitly state that no control was sent")
    return SO101ShadowSession(root, manifest, frames, states, timestamps_s, actions)
