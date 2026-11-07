"""Non-actuating live adapter for the fixed Prod-V13 multi-rate endpoint.

The adapter keeps the separation used during training:

* CoTracker owns the current pixel measurement;
* the frozen V12.4 ViT supplies tokens at those measured pixels;
* ``MultiRatePartTokenLeWM`` sees only the previous token and the completed
  joint/CAD displacement;
* the LeWM prior may only re-rank a small current-image measurement support.

This module never emits a robot command. After automatic H0, the runtime may
consume its current fixed/moving posterior as the fingertip measurement used
by planning; action proposal and command supervision remain separate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import threading
import time
import types
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fcpg.modules import sample_feature_map_at_pixels
from fcpg.prod_v13_active_initializer import initialize_fingertips_from_pulse
from fcpg.prod_v13_multirate import (
    GRID,
    PATCHES,
    TIP_DEGENERATE_RADIUS_PX,
    TOKEN_DIM,
    MultiRatePartTokenLeWM,
    tracker_candidate_feedback,
)
from v10.model import FrozenV9ViT


FORMAT = "so101-prod-v13-multirate-live-shadow-v1"
CHECKPOINT_FORMAT = "so101-prod-v13-cotracker-part-token-lewm-v1"
BEST_SEED = 13_161
VIEW_ORDER = ("wrist", "column")
TIP_ORDER = ("tip_a", "tip_b")
TRACKER_WINDOW = 16
TRACKER_STEP = 8
ROLLOUT_HORIZONS = 4
LOCAL_CANDIDATE_OFFSETS_PX = (
    (0.0, 0.0),
    (-0.75, 0.0),
    (0.75, 0.0),
    (0.0, -0.75),
    (0.0, 0.75),
    (-1.5, 0.0),
    (1.5, 0.0),
    (0.0, -1.5),
    (0.0, 1.5),
)
MEASUREMENT_SIGMA_PX = 1.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def parse_unordered_tip_queries(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Parse two query-owned tip slots per view without assigning jaw identity."""

    views = payload.get("views")
    if not isinstance(views, Mapping) or set(views) != set(VIEW_ORDER):
        raise ValueError("query init requires exactly wrist and column views")
    parsed: dict[str, np.ndarray] = {}
    for view in VIEW_ORDER:
        row = views.get(view)
        if not isinstance(row, Mapping) or not set(TIP_ORDER).issubset(row):
            raise ValueError(f"{view} query requires tip_a and tip_b")
        points = np.asarray([row[name] for name in TIP_ORDER], dtype=np.float32)
        if (
            points.shape != (2, 2)
            or not np.all(np.isfinite(points))
            or np.any(points < 0.0)
            or np.any(points > 223.0)
        ):
            raise ValueError(f"{view} tip queries must be finite [0,223] pixels")
        # Closed jaws may produce coincident/subpixel query coordinates.  A/B
        # remain independent query banks even in that case.
        parsed[view] = points
    return parsed


def load_part_token_lewm(
    checkpoint: str | Path, device: torch.device
) -> tuple[MultiRatePartTokenLeWM, dict[str, Any]]:
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not the Prod-V13 PartTokenLeWM format")
    if int(payload.get("step", -1)) != 20_000:
        raise ValueError("live shadow requires the fixed step-20000 endpoint")
    training = payload.get("training")
    if (
        not isinstance(training, Mapping)
        or int(training.get("seed", -1)) != BEST_SEED
    ):
        raise ValueError("live shadow requires engineering-best seed13161")
    model = MultiRatePartTokenLeWM(
        joint_rms=torch.as_tensor(payload["joint_rms"]),
        cad_rms=torch.as_tensor(payload["cad_rms"]),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    if int(payload.get("parameter_count", -1)) != model.parameter_count:
        raise ValueError("PartTokenLeWM parameter count differs from checkpoint")
    model.requires_grad_(False).eval().to(device)
    return model, payload


def _enable_cotracker_batch_compatibility(model: torch.nn.Module) -> None:
    """Materialize expanded coordinates for the pinned two-view batch."""

    if bool(getattr(model, "_fcpg_batch_compatibility_enabled", False)):
        return
    original = model.model.forward_window

    def contiguous_forward_window(
        _self: torch.nn.Module,
        fmaps_pyramid: object,
        coords: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> object:
        return original(fmaps_pyramid, coords.contiguous(), *args, **kwargs)

    model.model.forward_window = types.MethodType(
        contiguous_forward_window, model.model
    )
    model._fcpg_batch_compatibility_enabled = True


def load_cotracker_online(
    *, repo: str | Path, checkpoint: str | Path, device: torch.device
) -> torch.nn.Module:
    root = Path(repo).expanduser().resolve()
    weights = Path(checkpoint).expanduser().resolve()
    if not (root / "cotracker" / "predictor.py").is_file():
        raise FileNotFoundError(f"CoTracker repository is missing: {root}")
    if not weights.is_file():
        raise FileNotFoundError(f"CoTracker checkpoint is missing: {weights}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from cotracker.predictor import CoTrackerOnlinePredictor

    tracker = CoTrackerOnlinePredictor(checkpoint=str(weights)).to(device).eval()
    if int(getattr(tracker, "step", -1)) != TRACKER_STEP:
        raise ValueError("live adapter requires CoTracker3-online step=8")
    tracker.requires_grad_(False)
    _enable_cotracker_batch_compatibility(tracker)
    return tracker


class ProdV13MultirateShadowObserver:
    """Query-initialized, measurement-dominant non-actuating observer."""

    requires_observation_sequence = True

    def __init__(
        self,
        *,
        lewm: MultiRatePartTokenLeWM,
        vit: torch.nn.Module,
        tracker: torch.nn.Module,
        device: torch.device,
        artifact_root: Path,
        checkpoint_path: Path,
        checkpoint_seed: int,
        checkpoint_step: int,
        git_sha: str,
    ) -> None:
        self.lewm = lewm.eval()
        self.vit = vit.eval()
        self.tracker = tracker.eval()
        self.device = torch.device(device)
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.checkpoint_seed = int(checkpoint_seed)
        self.checkpoint_step = int(checkpoint_step)
        self.git_sha = str(git_sha)
        self.lock = threading.RLock()
        self.generation = 0
        self.reset("service_start")

    @classmethod
    def from_checkpoints(
        cls,
        *,
        checkpoint: str | Path,
        v12_checkpoint: str | Path,
        cotracker_repo: str | Path,
        cotracker_checkpoint: str | Path,
        device: str | torch.device,
        artifact_root: Path,
        git_sha: str,
    ) -> "ProdV13MultirateShadowObserver":
        resolved = torch.device(device)
        lewm, payload = load_part_token_lewm(checkpoint, resolved)
        training = payload.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("PartTokenLeWM checkpoint lacks training metadata")
        vit = FrozenV9ViT(v12_checkpoint).to(resolved).eval()
        tracker = load_cotracker_online(
            repo=cotracker_repo,
            checkpoint=cotracker_checkpoint,
            device=resolved,
        )
        return cls(
            lewm=lewm,
            vit=vit,
            tracker=tracker,
            device=resolved,
            artifact_root=artifact_root,
            checkpoint_path=Path(checkpoint),
            checkpoint_seed=int(training.get("seed", -1)),
            checkpoint_step=int(payload["step"]),
            git_sha=git_sha,
        )

    def reset(self, reason: str) -> None:
        with getattr(self, "lock", threading.RLock()):
            self.generation = int(getattr(self, "generation", 0)) + 1
            self.reset_reason = str(reason)
            self.initialized = False
            self.query_px: np.ndarray | None = None
            self.tracker_px: np.ndarray | None = None
            self.last_frames: dict[str, np.ndarray] | None = None
            self.latest_state: np.ndarray | None = None
            self.current_tokens: torch.Tensor | None = None
            self.latest_measurement: dict[str, Any] | None = None
            self.latest_prior: dict[str, Any] | None = None
            self.latest_posterior: dict[str, Any] | None = None
            self.tip_trajectory: list[dict[str, Any]] = []
            self.transition_artifacts: list[str] = []
            self.rollout_index = 0
            self.latest_latency_ms: float | None = None
            self.initialization_source: str | None = None
            self.initialization_diagnostics: dict[str, Any] | None = None
            self.rollout_active = False
            self.rollout_horizon = 0
            self.kinematics: Any | None = None
            self.session_root: Path | None = None
            self.v14_current_patches: torch.Tensor | None = None
            self.v14_previous_global: torch.Tensor | None = None
            self.v14_current_global: torch.Tensor | None = None

    @staticmethod
    def _validate_frames(frames: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if set(frames) != set(VIEW_ORDER):
            raise ValueError("multi-rate shadow requires wrist and column frames")
        result: dict[str, np.ndarray] = {}
        for view in VIEW_ORDER:
            frame = np.asarray(frames[view])
            if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
                raise ValueError(f"{view} frame must be uint8 [224,224,3]")
            result[view] = frame.copy()
        return result

    @torch.inference_mode()
    def _encode_patches(self, frames: Mapping[str, np.ndarray]) -> torch.Tensor:
        checked = self._validate_frames(frames)
        image = np.stack([checked[view] for view in VIEW_ORDER])
        tensor = torch.from_numpy(image).to(self.device)
        tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
        hidden = self.vit(tensor)
        if hidden.shape != (2, 257, TOKEN_DIM):
            raise ValueError("frozen V12.4 ViT returned an unexpected token shape")
        return hidden[:, 1:].reshape(1, 2, PATCHES, TOKEN_DIM)

    @staticmethod
    def _sample_tokens(patches: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        feature = patches[0].reshape(2, GRID, GRID, TOKEN_DIM).permute(0, 3, 1, 2)
        sampled = sample_feature_map_at_pixels(
            feature.float(), points.float(), image_size=(224, 224), padding_mode="border"
        )
        return F.normalize(sampled[None].float(), dim=-1, eps=1.0e-8)

    def _record_v14_visual_state(self, patches: torch.Tensor) -> None:
        """Cache the same patch/global state consumed by the V14 trainer."""

        value = patches.detach()
        if value.shape != (1, 2, PATCHES, TOKEN_DIM):
            raise ValueError("V14 visual cache requires [1,2,256,192] patches")
        current = F.layer_norm(value.float().mean(dim=2), (TOKEN_DIM,)).detach()
        if self.v14_current_global is None:
            self.v14_previous_global = current.clone()
        else:
            self.v14_previous_global = self.v14_current_global.clone()
        self.v14_current_global = current
        self.v14_current_patches = value

    def v14_visual_snapshot(self) -> dict[str, Any]:
        """Return current frozen visual tensors without actor or legacy anchors."""

        with self.lock:
            if (
                not self.initialized
                or self.tracker_px is None
                or self.v14_current_patches is None
                or self.v14_current_global is None
                or self.v14_previous_global is None
            ):
                raise RuntimeError("V14 snapshot requires initialized visual state")
            visible = np.ones((2, 2), dtype=bool)
            if isinstance(self.latest_measurement, Mapping):
                raw = self.latest_measurement.get("tracker_visible")
                if raw is not None:
                    visible = np.asarray(raw, dtype=bool)
            if visible.shape != (2, 2):
                raise ValueError("V14 snapshot tip visibility must be [2,2]")
            return {
                "patch_tokens": self.v14_current_patches.clone(),
                "global_history": torch.stack(
                    (self.v14_previous_global, self.v14_current_global), dim=1
                ).clone(),
                "tip_px": self.tracker_px.copy(),
                "tip_visible": visible.copy(),
                "generation": self.generation,
                "measurement_key": (
                    self.generation,
                    self.rollout_index,
                    self.rollout_horizon,
                    len(self.tip_trajectory),
                ),
                "view_order": VIEW_ORDER,
            }

    @torch.inference_mode()
    def _track_sequence(
        self,
        observation_sequence: Sequence[Mapping[str, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.last_frames is None or self.tracker_px is None:
            raise RuntimeError("tracker query has not been initialized")
        checked = [self._validate_frames(row) for row in observation_sequence]
        raw: dict[str, list[np.ndarray]] = {
            view: [self.last_frames[view], *[row[view] for row in checked]]
            for view in VIEW_ORDER
        }
        length = len(raw[VIEW_ORDER[0]])
        if length > TRACKER_WINDOW:
            for view in VIEW_ORDER:
                raw[view] = raw[view][-TRACKER_WINDOW:]
            length = TRACKER_WINDOW
        pad = TRACKER_WINDOW - length
        video_rows = []
        for view in VIEW_ORDER:
            frames = [raw[view][0]] * pad + raw[view]
            video_rows.append(np.stack(frames))
        video = torch.from_numpy(np.stack(video_rows)).to(self.device)
        video = video.permute(0, 1, 4, 2, 3).float().div_(255.0)
        query_time = float(pad)
        points = torch.from_numpy(self.tracker_px).to(
            self.device, dtype=torch.float32
        )
        queries = torch.cat(
            (
                torch.full((2, 2, 1), query_time, device=self.device),
                points,
            ),
            dim=-1,
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            initialized = self.tracker(
                video_chunk=video[:, :TRACKER_STEP],
                is_first_step=True,
                queries=queries,
                grid_size=0,
                add_support_grid=False,
            )
            if initialized != (None, None):
                raise RuntimeError("CoTracker initialization returned a prediction")
            tracks, visible = self.tracker(
                video_chunk=video,
                is_first_step=False,
                grid_size=0,
                add_support_grid=False,
            )
        mask = visible.bool()
        if mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if tracks.shape[:3] != (2, TRACKER_WINDOW, 2) or mask.shape != (
            2,
            TRACKER_WINDOW,
            2,
        ):
            raise ValueError("CoTracker output shape differs from the live contract")
        return tracks[:, -1].float().cpu().numpy(), mask[:, -1].cpu().numpy()

    def initialize_query(
        self,
        payload: Mapping[str, Any],
        *,
        frames: Mapping[str, np.ndarray],
        state: np.ndarray,
        timestamp_s: float,
    ) -> dict[str, Any]:
        del timestamp_s
        with self.lock:
            clicks = parse_unordered_tip_queries(payload)
            checked = self._validate_frames(frames)
            value = np.asarray(state, dtype=np.float32)
            if value.shape != (6,) or not np.all(np.isfinite(value)):
                raise ValueError("query state must be finite [6]")
            started = time.perf_counter()
            points = np.stack([clicks[view] for view in VIEW_ORDER])
            return self._initialize_points(
                points=points,
                frames=checked,
                state=value,
                source="operator_query",
                trajectory_kind="operator_query",
                diagnostics={"requires_mouse_query": True},
                started=started,
            )

    def _initialize_points(
        self,
        *,
        points: np.ndarray,
        frames: Mapping[str, np.ndarray],
        state: np.ndarray,
        source: str,
        trajectory_kind: str,
        diagnostics: Mapping[str, Any],
        started: float,
    ) -> dict[str, Any]:
        checked = self._validate_frames(frames)
        value = np.asarray(state, dtype=np.float32)
        query = np.asarray(points, dtype=np.float32)
        if query.shape != (2, 2, 2) or not np.all(np.isfinite(query)):
            raise ValueError("observer initialization points must be finite [2,2,2]")
        if np.any(query < 0.0) or np.any(query > 223.0):
            raise ValueError("observer initialization points must be inside the image")
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("observer initialization state must be finite [6]")
        patches = self._encode_patches(checked)
        tokens = self._sample_tokens(
            patches, torch.from_numpy(query).to(self.device)
        )
        self.reset(f"{source}_initialized")
        self._record_v14_visual_state(patches)
        self.query_px = query.copy()
        self.tracker_px = query.copy()
        self.last_frames = checked
        self.latest_state = value.copy()
        self.current_tokens = tokens.detach()
        self.initialized = True
        self.initialization_source = str(source)
        self.initialization_diagnostics = dict(diagnostics)
        self.latest_measurement = {
            "slot_semantics": list(TIP_ORDER),
            "tracker_tip_px": query,
            "tracker_visible": np.ones((2, 2), dtype=bool),
            "source": str(source),
        }
        self.tip_trajectory = [
            {
                "rollout_index": 0,
                "horizon": 0,
                "kind": str(trajectory_kind),
                "measurement_tip_px": query.copy(),
                "measurement_visible": np.ones((2, 2), dtype=bool),
                "posterior_tip_px": None,
                "prior_changed_selection": None,
            }
        ]
        self.latest_latency_ms = (time.perf_counter() - started) * 1000.0
        prefix = "auto-init" if source == "fcpgh_active_measurement" else "query"
        self.session_root = self.artifact_root / (
            f"{prefix}-{time.time_ns()}-g{self.generation}"
        )
        self.session_root.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            self.session_root / "initialization.npz",
            wrist_frame=checked["wrist"],
            column_frame=checked["column"],
            state=value,
            query_px=query,
        )
        self._write_status()
        return self.status()

    @torch.inference_mode()
    def initialize_from_pulse(
        self,
        *,
        frames_by_view: Mapping[str, np.ndarray],
        states: np.ndarray,
    ) -> dict[str, Any]:
        """Automatically seed FCPG-H from one completed local gripper pulse."""

        with self.lock:
            started = time.perf_counter()
            result = initialize_fingertips_from_pulse(
                tracker=self.tracker,
                frames_by_view={
                    view: np.asarray(frames_by_view[view]) for view in VIEW_ORDER
                },
                states=np.asarray(states, dtype=np.float32),
                device=self.device,
            )
            status = self._initialize_points(
                points=result.tip_px,
                frames=result.final_frames,
                state=result.final_state,
                source="fcpgh_active_measurement",
                trajectory_kind="automatic_fcpgh_query",
                diagnostics=result.diagnostics(),
                started=started,
            )
            if self.session_root is not None:
                np.savez_compressed(
                    self.session_root / "active_fingertip_tracks.npz",
                    wrist_track_px=result.views["wrist"].selected_track_px,
                    wrist_visible=result.views["wrist"].selected_visible,
                    column_track_px=result.views["column"].selected_track_px,
                    column_visible=result.views["column"].selected_visible,
                    states=np.asarray(states, dtype=np.float32),
                )
                self._write_status()
            return status

    def observe_measurement(
        self,
        *,
        frames: Mapping[str, np.ndarray],
        state: np.ndarray,
        timestamp_s: float,
    ) -> dict[str, Any]:
        with self.lock:
            if not self.initialized or self.tracker_px is None:
                raise RuntimeError("Prod-V13 query is not initialized")
            checked = self._validate_frames(frames)
            value = np.asarray(state, dtype=np.float32)
            if value.shape != (6,) or not np.all(np.isfinite(value)):
                raise ValueError("measurement state must be finite [6]")
            frame_changed = self.last_frames is None or any(
                not np.array_equal(checked[view], self.last_frames[view])
                for view in VIEW_ORDER
            )
            visible = np.ones((2, 2), dtype=bool)
            if frame_changed:
                # A replan may follow a camera perturbation without a robot
                # command. Track that visual change before exposing pixels to
                # the planner; copying the previous coordinates would recreate
                # the exact stale-anchor failure Prod-V13 is meant to remove.
                tracked_px, visible = self._track_sequence((checked,))
                self.tracker_px = tracked_px.copy()
            # During H1--H4, ``current_tokens`` is the recursive posterior from
            # the previous completed transition.  Replanning captures a fresh
            # idle frame before every next chunk; replacing the token here
            # would silently erase the feedback state before H2/H3/H4.  The
            # tracker coordinates still refresh for current-image planning,
            # while the recurrent token is preserved until the next completed
            # action consumes it.  Outside an active rollout, the fresh visual
            # measurement is the authoritative state for the next H1.
            tokens: torch.Tensor | None = None
            if not self.rollout_active:
                patches = self._encode_patches(checked)
                if frame_changed or self.v14_current_patches is None:
                    self._record_v14_visual_state(patches)
                tokens = self._sample_tokens(
                    patches, torch.from_numpy(self.tracker_px).to(self.device)
                )
            if frame_changed:
                recurrent_token_preserved = bool(self.rollout_active)
                if tokens is not None:
                    self.current_tokens = tokens.detach()
                self.latest_prior = None
                self.latest_posterior = None
                self.tip_trajectory.append(
                    {
                        "rollout_index": self.rollout_index,
                        "horizon": self.rollout_horizon,
                        "kind": "current_measurement_refresh",
                        "timestamp_s": float(timestamp_s),
                        "measurement_tip_px": self.tracker_px.copy(),
                        "measurement_visible": visible.copy(),
                        "posterior_tip_px": None,
                        "prior_changed_selection": None,
                        "recurrent_token_preserved": recurrent_token_preserved,
                    }
                )
            elif not self.rollout_active:
                if tokens is None:
                    raise RuntimeError("idle measurement token was not sampled")
                self.current_tokens = tokens.detach()
            self.last_frames = checked
            self.latest_state = value.copy()
            self.latest_measurement = {
                "slot_semantics": list(TIP_ORDER),
                "tracker_tip_px": self.tracker_px.copy(),
                "tracker_visible": visible,
                "source": (
                    "frozen_cotracker3_current_frame_refresh"
                    if frame_changed
                    else "current_frame_at_last_visual_measurement"
                ),
            }
            self._write_status()
            return self.status()

    def begin_rollout(
        self,
        *,
        planned_actions: np.ndarray,
        past_chunks: Sequence[np.ndarray],
        kinematics: Any,
    ) -> dict[str, Any]:
        del past_chunks
        with self.lock:
            if not self.initialized or self.current_tokens is None:
                raise RuntimeError("multi-rate rollout requires query initialization")
            actions = np.asarray(planned_actions, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 6 or not len(actions):
                raise ValueError("planned actions must be non-empty [N,6]")
            if not np.all(np.isfinite(actions)):
                raise ValueError("planned actions must be finite")
            self.kinematics = kinematics
            self.rollout_index += 1
            self.rollout_active = True
            self.rollout_horizon = 0
            self.latest_prior = None
            self.latest_posterior = None
            self._write_status()
            return self.status()

    @staticmethod
    def _completed_joint_increments(
        previous_state: np.ndarray, executed_actions: np.ndarray
    ) -> np.ndarray:
        action = np.asarray(executed_actions, dtype=np.float32)
        if action.shape != (5, 6) or not np.all(np.isfinite(action)):
            raise ValueError("completed shadow chunk must be finite [5,6]")
        sequence = np.concatenate((previous_state[None], action), axis=0)
        return np.diff(sequence, axis=0).astype(np.float32)

    @staticmethod
    def _cad_displacement(
        kinematics: Any, previous_state: np.ndarray, final_state: np.ndarray
    ) -> np.ndarray:
        anchors = np.asarray(
            kinematics.forward_anchors(
                np.stack((previous_state, final_state)).astype(np.float32)
            ),
            dtype=np.float32,
        )
        if anchors.shape != (2, 3, 3) or not np.all(np.isfinite(anchors)):
            raise ValueError("live FK anchors must be finite [2,3,3]")
        return (anchors[1, 1:] - anchors[0, 1:]).astype(np.float32)

    @staticmethod
    def _candidate_bank(
        track_px: np.ndarray, visible: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offsets = np.asarray(LOCAL_CANDIDATE_OFFSETS_PX, dtype=np.float32)
        points = track_px[..., None, :] + offsets[None, None]
        distance2 = np.sum(offsets * offsets, axis=-1)
        logits = -0.5 * distance2 / (MEASUREMENT_SIGMA_PX**2)
        logits = np.broadcast_to(logits, points.shape[:-1]).copy()
        in_frame = (
            (points[..., 0] >= 0.0)
            & (points[..., 0] <= 223.0)
            & (points[..., 1] >= 0.0)
            & (points[..., 1] <= 223.0)
        )
        support = in_frame & visible[..., None]
        return (
            torch.from_numpy(points),
            torch.from_numpy(logits),
            torch.from_numpy(support),
        )

    @torch.inference_mode()
    def advance_after_chunk(
        self,
        *,
        frames: Mapping[str, np.ndarray],
        state: np.ndarray,
        timestamp_s: float,
        executed_actions: np.ndarray,
        observation_sequence: Sequence[Mapping[str, np.ndarray]],
    ) -> dict[str, Any]:
        with self.lock:
            if (
                not self.rollout_active
                or self.current_tokens is None
                or self.latest_state is None
                or self.kinematics is None
            ):
                raise RuntimeError("multi-rate rollout has not been started")
            if self.rollout_horizon >= ROLLOUT_HORIZONS:
                raise RuntimeError("multi-rate H1-H4 rollout is complete")
            started = time.perf_counter()
            current_state = np.asarray(state, dtype=np.float32)
            if current_state.shape != (6,) or not np.all(np.isfinite(current_state)):
                raise ValueError("post-chunk state must be finite [6]")
            previous_state = self.latest_state.copy()
            previous_tokens = self.current_tokens.detach().clone()
            previous_tracker_px = self.tracker_px.copy()
            previous_frames = {
                view: self.last_frames[view].copy() for view in VIEW_ORDER
            }
            tracked_px, visible = self._track_sequence(observation_sequence)
            checked = self._validate_frames(frames)
            patches = self._encode_patches(checked)
            self._record_v14_visual_state(patches)
            joint = self._completed_joint_increments(
                self.latest_state, executed_actions
            )
            cad = self._cad_displacement(
                self.kinematics, self.latest_state, current_state
            )
            prior = self.lewm.predict_token_prior(
                previous_tokens=self.current_tokens,
                joint_increment_chunk=torch.from_numpy(joint)[None].to(self.device),
                cad_tip_displacement=torch.from_numpy(cad)[None].to(self.device),
            )
            candidate_px, measurement_logits, candidate_valid = self._candidate_bank(
                tracked_px, visible
            )
            feedback = tracker_candidate_feedback(
                previous_tokens=self.current_tokens,
                prior_tokens=prior.prior_tokens,
                current_patch_tokens=patches,
                candidate_px=candidate_px[None].to(self.device),
                measurement_logits=measurement_logits[None].to(self.device),
                candidate_valid=candidate_valid[None].to(self.device),
                view_valid=torch.ones(1, 2, dtype=torch.bool, device=self.device),
            )
            self.current_tokens = torch.where(
                feedback.view_valid[:, :, None, None],
                feedback.posterior_tokens,
                self.current_tokens,
            ).detach()
            self.tracker_px = tracked_px.copy()
            self.last_frames = checked
            self.latest_state = current_state.copy()
            self.rollout_horizon += 1
            if self.rollout_horizon >= ROLLOUT_HORIZONS:
                self.rollout_active = False
            self.latest_measurement = {
                "slot_semantics": list(TIP_ORDER),
                "tracker_tip_px": tracked_px,
                "tracker_visible": visible,
                "candidate_count": len(LOCAL_CANDIDATE_OFFSETS_PX),
                "source": "frozen_cotracker3_online",
            }
            self.latest_prior = {
                "horizon": self.rollout_horizon,
                "token_norm": prior.prior_tokens.float().norm(dim=-1)[0],
                "dynamics_norm": prior.dynamics_latent.float().norm(dim=-1)[0],
                "completed_joint_increment_max": float(np.max(np.abs(joint))),
                "cad_tip_displacement_m": cad,
            }
            self.latest_posterior = {
                "horizon": self.rollout_horizon,
                "tip_px": feedback.posterior_tip_px[0],
                "center_px": feedback.posterior_center_px[0],
                "measurement_candidate_index": feedback.measurement_candidate_index[0],
                "selected_candidate_index": feedback.selected_candidate_index[0],
                "prior_changed_selection": feedback.prior_changed_selection[0],
                "view_valid": feedback.view_valid[0],
                "measurement_missing": feedback.measurement_missing[0],
                "tip_pair_degenerate": feedback.tip_pair_degenerate[0],
            }
            self.tip_trajectory.append(
                {
                    "rollout_index": self.rollout_index,
                    "horizon": self.rollout_horizon,
                    "kind": "observed_feedback",
                    "timestamp_s": float(timestamp_s),
                    "measurement_tip_px": tracked_px.copy(),
                    "measurement_visible": visible.copy(),
                    "posterior_tip_px": (
                        feedback.posterior_tip_px[0].float().cpu().numpy()
                    ),
                    "prior_changed_selection": (
                        feedback.prior_changed_selection[0].cpu().numpy()
                    ),
                    "recurrent_token_preserved": True,
                }
            )
            self.latest_latency_ms = (time.perf_counter() - started) * 1000.0
            if self.session_root is not None:
                artifact_name = (
                    f"transition_r{self.rollout_index:04d}_"
                    f"h{self.rollout_horizon:02d}.npz"
                )
                artifact_path = self.session_root / artifact_name
                sequences = {
                    view: np.stack(
                        [np.asarray(row[view], dtype=np.uint8) for row in observation_sequence]
                    )
                    for view in VIEW_ORDER
                }
                with artifact_path.open("xb") as stream:
                    np.savez_compressed(
                        stream,
                        rollout_index=np.asarray(self.rollout_index, dtype=np.int64),
                        horizon=np.asarray(self.rollout_horizon, dtype=np.int64),
                        timestamp_s=np.asarray(timestamp_s, dtype=np.float64),
                        previous_wrist_frame=previous_frames["wrist"],
                        previous_column_frame=previous_frames["column"],
                        wrist_observation_sequence=sequences["wrist"],
                        column_observation_sequence=sequences["column"],
                        wrist_frame=checked["wrist"],
                        column_frame=checked["column"],
                        previous_state=previous_state,
                        state=current_state,
                        executed_actions=np.asarray(executed_actions, dtype=np.float32),
                        completed_joint_increments=joint,
                        cad_tip_displacement=cad,
                        previous_tracker_px=previous_tracker_px,
                        tracker_px=tracked_px,
                        tracker_visible=visible,
                        candidate_px=candidate_px.numpy(),
                        measurement_logits=measurement_logits.numpy(),
                        candidate_valid=candidate_valid.numpy(),
                        previous_tokens=previous_tokens.cpu().numpy().astype(np.float16),
                        prior_tokens=prior.prior_tokens.cpu().numpy().astype(np.float16),
                        dynamics_latent=prior.dynamics_latent.cpu().numpy().astype(np.float16),
                        posterior_tokens=self.current_tokens.cpu().numpy().astype(np.float16),
                        posterior_tip_px=feedback.posterior_tip_px.cpu().numpy(),
                        measurement_candidate_index=(
                            feedback.measurement_candidate_index.cpu().numpy()
                        ),
                        selected_candidate_index=(
                            feedback.selected_candidate_index.cpu().numpy()
                        ),
                        prior_changed_selection=(
                            feedback.prior_changed_selection.cpu().numpy()
                        ),
                        view_valid=feedback.view_valid.cpu().numpy(),
                        measurement_missing=feedback.measurement_missing.cpu().numpy(),
                        tip_pair_degenerate=feedback.tip_pair_degenerate.cpu().numpy(),
                    )
                self.transition_artifacts.append(artifact_name)
                self.tip_trajectory[-1]["artifact_file"] = artifact_name
            self._write_status()
            return self.status()

    def _write_status(self) -> None:
        if self.session_root is None:
            return
        (self.session_root / "status.json").write_text(
            json.dumps(self.status(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def status(self) -> dict[str, Any]:
        with self.lock:
            return _jsonable(
                {
                    "format": FORMAT,
                    "enabled": True,
                    "non_actuating": True,
                    "initialized": self.initialized,
                    "generation": self.generation,
                    "view_order": list(VIEW_ORDER),
                    "reset_reason": self.reset_reason,
                    "identity": {
                        "status": (
                            "automatic_fixed_moving_slots"
                            if self.initialization_source
                            == "fcpgh_active_measurement"
                            else (
                                "unordered_query_slots"
                                if self.initialized
                                else "automatic_initialization_required"
                            )
                        ),
                        "mapping": (
                            {"tip_a": "fixed", "tip_b": "moving"}
                            if self.initialization_source
                            == "fcpgh_active_measurement"
                            else None
                        ),
                        "fixed_moving_assignment_required": False,
                    },
                    "initialization_source": self.initialization_source,
                    "initialization_diagnostics": self.initialization_diagnostics,
                    "rollout_ready": bool(
                        self.initialized and self.current_tokens is not None
                    ),
                    "rollout_active": self.rollout_active,
                    "rollout_horizon_completed": self.rollout_horizon,
                    "measurement": self.latest_measurement,
                    "prior": self.latest_prior,
                    "posterior": self.latest_posterior,
                    "tip_trajectory": self.tip_trajectory,
                    "recorded_transition_count": len(self.transition_artifacts),
                    "transition_artifacts": self.transition_artifacts,
                    "pixel_prediction_contract": (
                        (
                            "automatic_fcpgh_h0_pending; no_pixel_query; "
                            if not self.initialized
                            else (
                                "automatic_fcpgh_tracker_posterior_only; "
                                if self.initialization_source
                                == "fcpgh_active_measurement"
                                else "observed_query_tracker_posterior_only; "
                            )
                        )
                        + "token_prior_has_no_future_pixel_coordinate"
                    ),
                    "latency_ms": self.latest_latency_ms,
                    "checkpoint": self.checkpoint_path,
                    "checkpoint_seed": self.checkpoint_seed,
                    "checkpoint_step": self.checkpoint_step,
                    "git_sha": self.git_sha,
                    "artifact_root": None if self.session_root is None else str(self.session_root),
                    "model_revision": "multirate_part_token_lewm_measurement_dominant",
                    "measurement_source": "frozen_cotracker3_online",
                    "candidate_offsets_px": LOCAL_CANDIDATE_OFFSETS_PX,
                    "tip_degenerate_radius_px": TIP_DEGENERATE_RADIUS_PX,
                    "action_influence": (
                        "current_fingertip_measurement_consumed_by_planner; "
                        "observer_emits_no_command"
                    ),
                }
            )


__all__ = [
    "FORMAT",
    "BEST_SEED",
    "LOCAL_CANDIDATE_OFFSETS_PX",
    "ProdV13MultirateShadowObserver",
    "load_part_token_lewm",
    "parse_unordered_tip_queries",
]
