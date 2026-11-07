"""Historical default-off shadow state for fixed Prod-V13 recursive-v3.2.

This loader is not compatible with the maintained multi-rate token-LeWM
endpoint.  It remains importable only for the frozen recursive-v3.2 archive;
the next ``:8765`` adapter must use ``fcpg.prod_v13_multirate`` instead.

The class never writes robot commands.  It owns only immutable query tokens,
pure visual FCPG history, one action-conditioned LeWM rollout state, and
append-only shadow artifacts.  Hardware motion remains in the existing
controller/I/O safety path.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from fcpg.offline_pipeline_manifest import sha256_file
from fcpg.prod_v13_action_innovation import (
    IIR_WRITEBACK_GAIN,
    SOFT_IIR_METHOD_REVISION,
    ProdV13ActionInnovationSoftIIR025,
    load_soft_iir_checkpoint,
)
from fcpg.prod_v13_closed_loop import (
    GRID,
    HORIZONS,
    PATCHES,
    TOKEN_DIM,
)
from fcpg.prod_v13_recursive_v2 import SUPPORT_SIZE
from fcpg.transformer_grounding import sample_feature_map_at_pixels
from v10.data import JOINT_SCALE


FORMAT = "so101-prod-v13-live-shadow-v2-soft-iir025"
VIEW_ORDER = ("wrist", "column")
POINT_ORDER = ("tip_a", "tip_b", "gripper_body")
IDENTITY_MIN_RELATIVE_MOTION_PX = 3.0
IDENTITY_MIN_GAP_PX = 2.0
IDENTITY_MAX_RETURN_ERROR_PX = 6.0
PATCH_SIZE = 14.0
PATCH_CENTER_OFFSET = 6.5


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


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parse_query_clicks(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    views = payload.get("views")
    if not isinstance(views, Mapping) or set(views) != set(VIEW_ORDER):
        raise ValueError("query init requires exactly wrist and column views")
    parsed: dict[str, np.ndarray] = {}
    for view in VIEW_ORDER:
        row = views.get(view)
        if not isinstance(row, Mapping) or set(row) != set(POINT_ORDER):
            raise ValueError(
                f"{view} query requires tip_a, tip_b and gripper_body"
            )
        points = np.asarray([row[name] for name in POINT_ORDER], dtype=np.float32)
        if (
            points.shape != (3, 2)
            or not np.all(np.isfinite(points))
            or np.any(points < 0.0)
            or np.any(points >= 224.0)
        ):
            raise ValueError(f"{view} query clicks must be finite [0,224) pixels")
        if np.linalg.norm(points[0] - points[1]) < 2.0:
            raise ValueError(f"{view} tip-A and tip-B clicks must be distinct")
        parsed[view] = points
    return parsed


def resolve_probe_identity(
    initial_px: np.ndarray,
    moved_px: np.ndarray,
    restored_px: np.ndarray,
) -> dict[str, Any]:
    """Resolve unordered A/B identity from motion relative to gripper body."""

    initial = np.asarray(initial_px, dtype=np.float32)
    moved = np.asarray(moved_px, dtype=np.float32)
    restored = np.asarray(restored_px, dtype=np.float32)
    if any(value.shape != (2, 3, 2) for value in (initial, moved, restored)):
        raise ValueError("identity probe points must be [wrist,column,3,2]")
    if not all(np.all(np.isfinite(value)) for value in (initial, moved, restored)):
        raise ValueError("identity probe points must be finite")
    per_view: dict[str, Any] = {}
    implied: dict[str, str | None] = {}
    for view_index, view in enumerate(VIEW_ORDER):
        initial_relative = initial[view_index, :2] - initial[view_index, 2]
        moved_relative = moved[view_index, :2] - moved[view_index, 2]
        restored_relative = restored[view_index, :2] - restored[view_index, 2]
        relative_motion = np.linalg.norm(
            moved_relative - initial_relative, axis=-1
        )
        return_error = np.linalg.norm(
            restored_relative - initial_relative, axis=-1
        )
        gap = float(abs(relative_motion[0] - relative_motion[1]))
        strongest = float(relative_motion.max())
        return_ok = bool(return_error.max() <= IDENTITY_MAX_RETURN_ERROR_PX)
        informative = bool(
            strongest >= IDENTITY_MIN_RELATIVE_MOTION_PX
            and gap >= IDENTITY_MIN_GAP_PX
            and return_ok
        )
        mapping = None
        if gap >= IDENTITY_MIN_GAP_PX and return_ok:
            mapping = "a_fixed" if relative_motion[0] < relative_motion[1] else "b_fixed"
        implied[view] = mapping
        per_view[view] = {
            "relative_motion_px": relative_motion.tolist(),
            "return_error_px": return_error.tolist(),
            "motion_gap_px": gap,
            "strongest_relative_motion_px": strongest,
            "informative": informative,
            "implied_mapping": mapping,
        }
    wrist_mapping = implied["wrist"]
    wrist_valid = bool(per_view["wrist"]["informative"] and wrist_mapping)
    column_mapping = implied["column"]
    column_conflict = bool(
        column_mapping is not None
        and wrist_mapping is not None
        and column_mapping != wrist_mapping
    )
    resolved = wrist_valid and not column_conflict
    if resolved and wrist_mapping == "a_fixed":
        mapping = {"tip_a": "fixed", "tip_b": "moving"}
    elif resolved:
        mapping = {"tip_a": "moving", "tip_b": "fixed"}
    else:
        mapping = None
    return {
        "status": "identity_resolved" if resolved else "identity_unresolved",
        "mapping": mapping,
        "wrist_identity_signal_valid": wrist_valid,
        "column_informative": bool(per_view["column"]["informative"]),
        "column_conflict": column_conflict,
        "per_view": per_view,
        "thresholds": {
            "minimum_relative_motion_px": IDENTITY_MIN_RELATIVE_MOTION_PX,
            "minimum_motion_gap_px": IDENTITY_MIN_GAP_PX,
            "maximum_return_error_px": IDENTITY_MAX_RETURN_ERROR_PX,
        },
    }


def _soft_argmax(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != (GRID, GRID):
        raise ValueError("point logits must end in 16x16")
    flat = logits.float().flatten(-2)
    probability = flat.softmax(dim=-1)
    index = torch.arange(PATCHES, device=logits.device)
    x = index.remainder(GRID).float() * PATCH_SIZE + PATCH_CENTER_OFFSET
    y = torch.div(index, GRID, rounding_mode="floor").float()
    y = y * PATCH_SIZE + PATCH_CENTER_OFFSET
    centres = torch.stack((x, y), dim=-1)
    return torch.einsum("...p,pd->...d", probability, centres)


class ProdV13ShadowObserver:
    """Non-actuating online measurement/prior/posterior state machine."""

    def __init__(
        self,
        model: ProdV13ActionInnovationSoftIIR025,
        *,
        device: torch.device,
        artifact_root: Path,
        checkpoint_sha256: str,
        git_sha: str,
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device)
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.git_sha = str(git_sha)
        self.lock = threading.RLock()
        self.generation = 0
        self.reset("service_start")

    @classmethod
    def from_checkpoints(
        cls,
        *,
        checkpoint: Path,
        actor_checkpoint: Path,
        v12_checkpoint: Path,
        fcpg_checkpoint: Path,
        device: str | torch.device,
        artifact_root: Path,
        git_sha: str,
    ) -> "ProdV13ShadowObserver":
        resolved_device = torch.device(device)
        model, _ = load_soft_iir_checkpoint(
            checkpoint,
            production_actor_checkpoint=actor_checkpoint,
            v12_checkpoint=v12_checkpoint,
            fcpg_checkpoint=fcpg_checkpoint,
            device=resolved_device,
        )
        return cls(
            model,
            device=resolved_device,
            artifact_root=artifact_root,
            checkpoint_sha256=sha256_file(checkpoint),
            git_sha=git_sha,
        )

    def reset(self, reason: str) -> None:
        with getattr(self, "lock", threading.RLock()):
            self.generation = int(getattr(self, "generation", 0)) + 1
            self.reset_reason = str(reason)
            self.initialized = False
            self.query_clicks: dict[str, np.ndarray] | None = None
            self.initial_point_tokens: torch.Tensor | None = None
            self.initial_tip_tokens: torch.Tensor | None = None
            self.visual_history: deque[dict[str, Any]] = deque(maxlen=16)
            self.identity = {
                "status": "identity_unresolved",
                "mapping": None,
                "reason": "query_or_probe_required",
            }
            self.current_posterior: torch.Tensor | None = None
            self.latest_measurement: dict[str, Any] | None = None
            self.latest_prior: dict[str, Any] | None = None
            self.latest_posterior: dict[str, Any] | None = None
            self.latest_latency_ms: float | None = None
            self.rollout_dynamics: torch.Tensor | None = None
            self.rollout_reference_dynamics: torch.Tensor | None = None
            self.rollout_horizon = 0
            self.rollout_action: np.ndarray | None = None
            self.rollout_executed: list[np.ndarray] = []
            self.session_root: Path | None = None

    @staticmethod
    def _validate_frames(frames: Mapping[str, np.ndarray]) -> np.ndarray:
        if set(frames) != set(VIEW_ORDER):
            raise ValueError("Prod-V13 shadow requires fresh wrist and column frames")
        values = []
        for view in VIEW_ORDER:
            frame = np.asarray(frames[view])
            if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
                raise ValueError(f"{view} frame must be uint8 [224,224,3]")
            values.append(frame)
        return np.stack(values)

    @torch.inference_mode()
    def _encode(self, frames: Mapping[str, np.ndarray]) -> torch.Tensor:
        values = self._validate_frames(frames)
        image = torch.from_numpy(values).to(self.device)
        image = image.permute(0, 3, 1, 2).float().div_(255.0)
        return self.model.v12.vit(image).reshape(2, 257, TOKEN_DIM)

    @torch.inference_mode()
    def _sample_points(
        self, hidden: torch.Tensor, points: np.ndarray
    ) -> torch.Tensor:
        patch = hidden[:, 1:].reshape(2, GRID, GRID, TOKEN_DIM)
        patch_map = patch.permute(0, 3, 1, 2)
        px = torch.from_numpy(points).to(self.device).reshape(2, 3, 2)
        return sample_feature_map_at_pixels(
            patch_map.float(), px.float(), image_size=(224, 224)
        )

    @torch.inference_mode()
    def _measurement(self, hidden: torch.Tensor) -> tuple[Any, torch.Tensor]:
        if self.initial_tip_tokens is None:
            raise RuntimeError("Prod-V13 query is not initialized")
        history = list(self.visual_history)
        if not history:
            history = [{"hidden": hidden}]
        hidden_history = [row["hidden"] for row in history]
        if len(hidden_history) == 1:
            hidden_history.append(hidden_history[0])
        hidden_history = hidden_history[-16:]
        stack = torch.stack(hidden_history)
        maps = stack[:, :, 1:].reshape(
            len(stack), 2, GRID, GRID, TOKEN_DIM
        ).permute(0, 1, 4, 2, 3)
        pooled = F.avg_pool2d(
            maps.reshape(-1, TOKEN_DIM, GRID, GRID).float(), 2, 2
        ).reshape(1, len(stack), 2, TOKEN_DIM, 8, 8)
        current = hidden[:, 1:].reshape(2, GRID, GRID, TOKEN_DIM)
        current = current.permute(0, 3, 1, 2)[None]
        output = self.model.fcpg.forward_cached(
            pooled_history=pooled.to(hidden.dtype),
            current_patch_maps=current,
            initial_query_tokens=self.initial_tip_tokens[None],
            time_valid=torch.ones(
                1, len(stack), dtype=torch.bool, device=self.device
            ),
            view_valid=torch.ones(1, 2, dtype=torch.bool, device=self.device),
        )
        return output, current

    @staticmethod
    def _top_support(
        logits: torch.Tensor,
    ) -> list[list[list[dict[str, float | int]]]]:
        flat = logits.float().flatten(-2)
        values, indices = flat.topk(SUPPORT_SIZE, dim=-1)
        result = []
        for view in range(2):
            tips = []
            for tip in range(2):
                tips.append(
                    [
                        {"patch": int(index), "logit": float(value)}
                        for index, value in zip(
                            indices[0, view, tip].tolist(),
                            values[0, view, tip].tolist(),
                            strict=True,
                        )
                    ]
                )
            result.append(tips)
        return result

    def _append_hidden(
        self,
        hidden: torch.Tensor,
        *,
        state: np.ndarray,
        timestamp_s: float,
    ) -> None:
        value = np.asarray(state, dtype=np.float32)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("shadow observation state must be finite [6]")
        self.visual_history.append(
            {
                "hidden": hidden.detach(),
                "state": value.copy(),
                "timestamp_s": float(timestamp_s),
            }
        )

    def initialize_query(
        self,
        payload: Mapping[str, Any],
        *,
        frames: Mapping[str, np.ndarray],
        state: np.ndarray,
        timestamp_s: float,
    ) -> dict[str, Any]:
        with self.lock:
            clicks = parse_query_clicks(payload)
            started = time.perf_counter()
            hidden = self._encode(frames)
            points = np.stack([clicks[view] for view in VIEW_ORDER])
            point_tokens = self._sample_points(hidden, points)
            self.reset("query_initialized")
            self.query_clicks = clicks
            self.initial_point_tokens = point_tokens.detach()
            self.initial_tip_tokens = point_tokens[:, :2].detach()
            self.initialized = True
            self.session_root = self.artifact_root / (
                f"query-{time.time_ns()}-g{self.generation}"
            )
            self.session_root.mkdir(parents=True, exist_ok=False)
            self._append_hidden(hidden, state=state, timestamp_s=timestamp_s)
            measurement, _ = self._measurement(hidden)
            self._record_measurement(measurement)
            self.latest_latency_ms = (time.perf_counter() - started) * 1000.0
            np.savez_compressed(
                self.session_root / "query_init.npz",
                wrist_frame=np.asarray(frames["wrist"]),
                column_frame=np.asarray(frames["column"]),
                state=np.asarray(state, dtype=np.float32),
                clicks=points,
                initial_point_tokens=point_tokens.cpu().numpy().astype(np.float16),
            )
            self._write_status_artifact()
            return self.status()

    def _record_measurement(self, output: Any) -> None:
        logits = output.measurement.measurement_logits.detach()
        self.latest_measurement = {
            "slot_semantics": ["tip_a", "tip_b"],
            "tip_px": output.measurement.tip_px[0].float().cpu().numpy(),
            "top32": self._top_support(logits),
            "history_evidence_count": (
                output.endpoint_evidence_count[0].cpu().numpy()
            ),
        }

    def observe_measurement(
        self,
        *,
        frames: Mapping[str, np.ndarray],
        state: np.ndarray,
        timestamp_s: float,
    ) -> dict[str, Any]:
        """Refresh pure visual measurement without advancing LeWM state."""

        with self.lock:
            if not self.initialized:
                raise RuntimeError("Prod-V13 query is not initialized")
            started = time.perf_counter()
            hidden = self._encode(frames)
            self._append_hidden(hidden, state=state, timestamp_s=timestamp_s)
            measurement, _ = self._measurement(hidden)
            self._record_measurement(measurement)
            self.latest_latency_ms = (time.perf_counter() - started) * 1000.0
            self._write_status_artifact()
            return self.status()

    @torch.inference_mode()
    def _predict_all_points(self, hidden: torch.Tensor) -> np.ndarray:
        if self.initial_point_tokens is None:
            raise RuntimeError("identity probe requires initialized point tokens")
        patches = hidden[:, 1:].reshape(2, PATCHES, TOKEN_DIM)
        query = F.normalize(self.initial_point_tokens.float(), dim=-1)
        key = F.normalize(patches.float(), dim=-1)
        logits = torch.einsum("vkd,vpd->vkp", query, key) / 0.1
        return _soft_argmax(logits.reshape(2, 3, GRID, GRID)).cpu().numpy()

    def apply_identity_probe(
        self,
        *,
        moved_frames: Mapping[str, np.ndarray],
        restored_frames: Mapping[str, np.ndarray],
        moved_state: np.ndarray,
        restored_state: np.ndarray,
        command_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self.lock:
            if not self.initialized or self.query_clicks is None:
                raise RuntimeError("identity probe requires query initialization")
            moved_hidden = self._encode(moved_frames)
            restored_hidden = self._encode(restored_frames)
            initial = np.stack(
                [self.query_clicks[view] for view in VIEW_ORDER]
            )
            moved = self._predict_all_points(moved_hidden)
            restored = self._predict_all_points(restored_hidden)
            identity = resolve_probe_identity(initial, moved, restored)
            self.identity = identity
            self._append_hidden(
                restored_hidden,
                state=np.asarray(restored_state),
                timestamp_s=time.perf_counter(),
            )
            measurement, _ = self._measurement(restored_hidden)
            self._record_measurement(measurement)
            if identity["status"] == "identity_resolved":
                order = (
                    (0, 1)
                    if identity["mapping"]["tip_a"] == "fixed"
                    else (1, 0)
                )
                raw = measurement.measurement.part_tokens[:, :, list(order)]
                self.current_posterior = self.model.v12.patch_projector(raw).detach()
            else:
                self.current_posterior = None
                self.rollout_dynamics = None
                self.rollout_reference_dynamics = None
                self.rollout_horizon = 0
            if self.session_root is not None:
                np.savez_compressed(
                    self.session_root / "identity_probe.npz",
                    moved_wrist=np.asarray(moved_frames["wrist"]),
                    moved_column=np.asarray(moved_frames["column"]),
                    restored_wrist=np.asarray(restored_frames["wrist"]),
                    restored_column=np.asarray(restored_frames["column"]),
                    moved_state=np.asarray(moved_state, dtype=np.float32),
                    restored_state=np.asarray(restored_state, dtype=np.float32),
                    moved_points=moved,
                    restored_points=restored,
                )
                (self.session_root / "identity_probe.json").write_text(
                    json.dumps(
                        _jsonable(
                            {"identity": identity, "motion": command_record}
                        ),
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            self._write_status_artifact()
            return self.status()

    @staticmethod
    def _normalize_action_prefix(
        planned_actions: np.ndarray, current_state: np.ndarray
    ) -> np.ndarray:
        value = np.asarray(planned_actions, dtype=np.float32)
        state = np.asarray(current_state, dtype=np.float32)
        if value.ndim != 2 or value.shape[1:] != (6,) or not len(value):
            raise ValueError("planned shadow actions must be non-empty [N,6]")
        if not np.all(np.isfinite(value)) or state.shape != (6,):
            raise ValueError("planned shadow action/state must be finite")
        if len(value) >= 20:
            return value[:20].copy()
        tail = np.repeat(value[-1:, :], 20 - len(value), axis=0)
        return np.concatenate((value, tail), axis=0)

    @torch.inference_mode()
    def begin_rollout(
        self,
        *,
        planned_actions: np.ndarray,
        past_chunks: Sequence[np.ndarray],
        kinematics: Any,
    ) -> dict[str, Any]:
        with self.lock:
            if self.identity.get("status") != "identity_resolved":
                raise RuntimeError("Prod-V13 rollout requires resolved tip identity")
            if len(self.visual_history) < 2 or self.current_posterior is None:
                raise RuntimeError("Prod-V13 rollout needs two fresh visual states")
            history = list(self.visual_history)[-2:]
            cls = torch.stack([row["hidden"][:, 0] for row in history])
            cls = cls[None].to(self.device)
            states = np.stack([row["state"] for row in history]).astype(np.float32)
            anchors = np.asarray(kinematics.forward_anchors(states), dtype=np.float32)
            if anchors.shape != (2, 3, 3):
                raise RuntimeError("live FK anchors must be [2,3,3]")
            current = states[-1]
            action = self._normalize_action_prefix(planned_actions, current)
            if past_chunks:
                past = np.asarray(past_chunks[-1], dtype=np.float32)
                if past.shape != (5, 6):
                    raise ValueError("latest completed action chunk must be [5,6]")
            else:
                past = np.repeat(current[None], 5, axis=0)
            state_tensor = torch.from_numpy(states)[None].to(self.device)
            action_tensor = torch.from_numpy(action)[None].to(self.device)
            hold_tensor = state_tensor[:, -1, None].expand_as(action_tensor)
            common = {
                "vit_cls_history": cls,
                "history_view_valid": torch.ones(
                    1, 2, 2, dtype=torch.bool, device=self.device
                ),
                "anchor_xyz_history": torch.from_numpy(anchors)[None].to(
                    self.device
                ),
                "state_history": state_tensor / JOINT_SCALE,
                "past_action_chunks": torch.from_numpy(
                    past.reshape(1, 1, 30)
                ).to(self.device)
                / JOINT_SCALE,
            }
            dynamics = self.model.predict_dynamics_cached(
                **common,
                action=action_tensor / JOINT_SCALE,
            )
            reference = self.model.predict_dynamics_cached(
                **common,
                action=hold_tensor / JOINT_SCALE,
            )
            self.rollout_dynamics = dynamics.detach()
            self.rollout_reference_dynamics = reference.detach()
            self.rollout_horizon = 0
            self.rollout_action = action
            self.rollout_executed = []
            self.latest_prior = None
            self.latest_posterior = None
            self._write_status_artifact()
            return self.status()

    @torch.inference_mode()
    def advance_after_chunk(
        self,
        *,
        frames: Mapping[str, np.ndarray],
        state: np.ndarray,
        timestamp_s: float,
        executed_actions: np.ndarray,
    ) -> dict[str, Any]:
        with self.lock:
            if (
                self.rollout_dynamics is None
                or self.rollout_reference_dynamics is None
                or self.current_posterior is None
            ):
                raise RuntimeError("Prod-V13 rollout has not been started")
            if self.rollout_horizon >= HORIZONS:
                raise RuntimeError("Prod-V13 H1-H4 rollout is already complete")
            started = time.perf_counter()
            executed = np.asarray(executed_actions, dtype=np.float32)
            if executed.shape != (5, 6) or not np.all(np.isfinite(executed)):
                raise ValueError("completed shadow chunk must be finite [5,6]")
            hidden = self._encode(frames)
            self._append_hidden(hidden, state=state, timestamp_s=timestamp_s)
            measurement, current_map = self._measurement(hidden)
            self._record_measurement(measurement)
            mapping = self.identity["mapping"]
            order = (0, 1) if mapping["tip_a"] == "fixed" else (1, 0)
            raw_logits = measurement.measurement.measurement_logits[:, :, list(order)]
            flat = raw_logits.float().flatten(-2)
            support = torch.zeros_like(flat, dtype=torch.bool)
            support.scatter_(
                -1, flat.topk(SUPPORT_SIZE, dim=-1).indices, True
            )
            projected_patch = self.model.v12.patch_projector(
                current_map.permute(0, 1, 3, 4, 2).reshape(1, 2, PATCHES, TOKEN_DIM)
            )
            dynamics = self.rollout_dynamics[:, self.rollout_horizon]
            reference = self.rollout_reference_dynamics[
                :, self.rollout_horizon
            ]
            padded_dynamics = dynamics[:, None].expand(-1, HORIZONS, -1)
            padded_reference = reference[:, None].expand(-1, HORIZONS, -1)
            padded_patch = projected_patch[:, None].expand(
                -1, HORIZONS, -1, -1, -1
            )
            padded_logits = raw_logits[:, None].expand(
                -1, HORIZONS, -1, -1, -1, -1
            )
            padded_support = support.reshape(1, 2, 2, GRID, GRID)[:, None].expand(
                -1, HORIZONS, -1, -1, -1, -1
            )
            valid = torch.ones(
                1, HORIZONS, 2, dtype=torch.bool, device=self.device
            )
            feedback = self.model.feedback(
                padded_dynamics,
                self.current_posterior,
                padded_patch,
                padded_logits,
                valid,
                topk_support=padded_support,
                reference_dynamics_latent=padded_reference,
                center_prior_similarity=True,
                writeback_gain=IIR_WRITEBACK_GAIN,
                soft_writeback_forward=True,
            )
            horizon = self.rollout_horizon
            hard_posterior = feedback.posterior_tip_tokens[:, 0].detach()
            self.current_posterior = feedback.recurrent_tip_tokens[
                :, 0
            ].detach()
            self.latest_prior = {
                "horizon": horizon + 1,
                "token_sha256": _tensor_sha256(feedback.prior_tokens[:, 0]),
                "token_norm": feedback.prior_tokens[:, 0]
                .float()
                .norm(dim=-1)
                .cpu()
                .numpy(),
            }
            self.latest_posterior = {
                "horizon": horizon + 1,
                "tip_order": ["fixed", "moving"],
                "tip_px": feedback.posterior_tip_px[:, 0][0].cpu().numpy(),
                "center_px": feedback.posterior_anchor_px[:, 0, :, 0][0]
                .cpu()
                .numpy(),
                "patch_index": feedback.posterior_tip_patch_index[:, 0][0]
                .cpu()
                .numpy(),
                "hard_token_sha256": _tensor_sha256(hard_posterior),
                "belief_token_sha256": _tensor_sha256(self.current_posterior),
                "writeback_gain": IIR_WRITEBACK_GAIN,
                "writeback_value": "top32_posterior_expected_token",
            }
            self.rollout_executed.append(executed.copy())
            self.rollout_horizon += 1
            self.latest_latency_ms = (time.perf_counter() - started) * 1000.0
            if self.session_root is not None:
                np.savez_compressed(
                    self.session_root / f"shadow_h{horizon + 1}.npz",
                    wrist_frame=np.asarray(frames["wrist"]),
                    column_frame=np.asarray(frames["column"]),
                    state=np.asarray(state, dtype=np.float32),
                    executed_actions=executed,
                    planned_actions=self.rollout_action,
                    measurement_logits=raw_logits.cpu().numpy().astype(np.float16),
                    top32_support=support.cpu().numpy(),
                    prior_tokens=feedback.prior_tokens[:, 0]
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                    posterior_hard_tokens=hard_posterior.cpu()
                    .numpy()
                    .astype(np.float16),
                    posterior_belief_tokens=self.current_posterior.cpu()
                    .numpy()
                    .astype(np.float16),
                    posterior_tip_px=feedback.posterior_tip_px[:, 0]
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                )
            self._write_status_artifact()
            return self.status()

    def _write_status_artifact(self) -> None:
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
                    "generation": self.generation,
                    "reset_reason": self.reset_reason,
                    "initialized": self.initialized,
                    "identity": self.identity,
                    "visual_history_frames": len(self.visual_history),
                    "measurement": self.latest_measurement,
                    "prior": self.latest_prior,
                    "posterior": self.latest_posterior,
                    "rollout_horizon_completed": self.rollout_horizon,
                    "rollout_active": bool(
                        self.rollout_dynamics is not None
                        and self.rollout_reference_dynamics is not None
                        and self.rollout_horizon < HORIZONS
                    ),
                    "latency_ms": self.latest_latency_ms,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "git_sha": self.git_sha,
                    "artifact_root": (
                        None if self.session_root is None else str(self.session_root)
                    ),
                    "action_influence": "shadow_only_never_control",
                    "model_revision": SOFT_IIR_METHOD_REVISION,
                    "measurement_support_size": SUPPORT_SIZE,
                    "writeback_gain": IIR_WRITEBACK_GAIN,
                    "inference_dtype": "float32",
                }
            )


__all__ = [
    "FORMAT",
    "POINT_ORDER",
    "ProdV13ShadowObserver",
    "VIEW_ORDER",
    "parse_query_clicks",
    "resolve_probe_identity",
]
