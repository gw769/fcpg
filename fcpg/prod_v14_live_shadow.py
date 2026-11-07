"""Default-off live counterfactual ranker for the selected Prod-V14 endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import threading
from typing import Any

import numpy as np
import torch

from fcpg.modules import sample_feature_map_at_pixels
from fcpg.prod_v14_runtime import ProdV14PlannerAdapter


FORMAT = "so101-prod-v14-live-shadow-v1"
VIEW_ORDER = ("wrist", "column")
MEASUREMENT_HISTORY_LIMIT = 128
LIVE_SHADOW_BEST_STEP = 2_000
LIVE_SHADOW_ARCHIVE_STEP = 500
LIVE_SHADOW_ALLOWED_STEPS = (LIVE_SHADOW_BEST_STEP, LIVE_SHADOW_ARCHIVE_STEP)


def _finite_point(value: Any) -> np.ndarray | None:
    point = np.asarray(value, dtype=np.float32)
    if (
        point.shape != (2,)
        or not np.all(np.isfinite(point))
        or np.any(point < 0.0)
        or np.any(point > 223.0)
    ):
        return None
    return point


def selected_live_shadow_step(path: Path) -> int:
    """Resolve the maintained best or the explicit archived comparison."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("V14 live-shadow checkpoint must contain a mapping")
    step = int(checkpoint.get("step", -1))
    if step not in LIVE_SHADOW_ALLOWED_STEPS:
        raise ValueError("V14 live shadow requires selected step 2000 or archive step 500")
    return step


class ProdV14LiveShadow:
    """Compare the exact safe candidate bank without changing its selection."""

    def __init__(
        self,
        *,
        adapter: ProdV14PlannerAdapter,
        kinematics: Any,
        checkpoint_path: Path,
    ) -> None:
        self.adapter = adapter
        self.kinematics = kinematics
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.last_generation: int | None = None
        self.last_action_clock: tuple[int, int, int] | None = None
        self.last_query_px: np.ndarray | None = None
        self.last_delta_px: np.ndarray | None = None
        self.last_target_kind: str | None = None
        self.last_target_px: np.ndarray | None = None
        self.latest_result: dict[str, Any] | None = None
        self.measurement_history: list[dict[str, Any]] = []
        self.lock = threading.RLock()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device,
        kinematics: Any,
    ) -> "ProdV14LiveShadow":
        path = Path(checkpoint).expanduser().resolve()
        selected_step = selected_live_shadow_step(path)
        return cls(
            adapter=ProdV14PlannerAdapter.from_checkpoint(
                path,
                device=device,
                expected_step=selected_step,
            ),
            kinematics=kinematics,
            checkpoint_path=path,
        )

    def _target_queries(self, plan: Any) -> tuple[np.ndarray, np.ndarray]:
        target = getattr(plan, "target", None)
        descriptors = getattr(plan, "target_descriptors", None)
        if not isinstance(target, Mapping) or not isinstance(descriptors, Mapping):
            raise ValueError("V14 live shadow requires plan target descriptors")
        kind = str(target.get("kind") or "none")
        if kind != "pickup":
            raise ValueError("V14 live shadow currently ranks pickup candidates only")
        points: list[np.ndarray] = []
        valid: list[bool] = []
        for view in VIEW_ORDER:
            descriptor = descriptors.get(f"pickup_{view}")
            point = None
            if isinstance(descriptor, Mapping) and bool(
                descriptor.get("visible", False)
            ):
                point = _finite_point(descriptor.get("centroid_px"))
            points.append(
                np.full(2, np.nan, dtype=np.float32) if point is None else point
            )
            valid.append(point is not None)
        return np.stack(points), np.asarray(valid, dtype=bool)

    def _causal_previous_delta(
        self,
        *,
        query_px: np.ndarray,
        generation: int,
        measurement_key: tuple[int, ...],
        target_kind: str,
        target_px: np.ndarray,
    ) -> np.ndarray:
        target = np.asarray(target_px, dtype=np.float32)
        if target.shape != (2, 2) or not np.all(np.isfinite(target)):
            raise ValueError("V14 target query must be finite [2,2]")
        if len(measurement_key) < 3:
            raise ValueError("V14 measurement key must contain action clock fields")
        action_clock = tuple(int(value) for value in measurement_key[:3])
        with self.lock:
            reset = (
                self.last_generation != int(generation)
                or self.last_target_kind != str(target_kind)
                or self.last_query_px is None
                or self.last_target_px is None
                or not np.array_equal(self.last_target_px, target)
            )
            if reset:
                delta = np.zeros_like(query_px, dtype=np.float32)
                self.last_query_px = query_px.copy()
                self.last_delta_px = delta.copy()
                self.last_action_clock = action_clock
            elif self.last_action_clock != action_clock:
                delta = (query_px - self.last_query_px).astype(np.float32)
                self.last_query_px = query_px.copy()
                self.last_delta_px = delta.copy()
                self.last_action_clock = action_clock
            else:
                if self.last_delta_px is None:
                    raise RuntimeError("V14 previous displacement state is incomplete")
                delta = self.last_delta_px.copy()
            self.last_generation = int(generation)
            self.last_target_kind = str(target_kind)
            self.last_target_px = target.copy()
            return delta

    def _publish(
        self,
        result: dict[str, Any],
        *,
        measurement_key: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        """Store a bounded, current-measurement-only diagnostic history."""

        with self.lock:
            self.latest_result = deepcopy(result)
            if measurement_key is not None and "tip_px" in result:
                record = {
                    "measurement_key": list(measurement_key),
                    "displacement_clock_key": deepcopy(
                        result.get("displacement_clock_key")
                    ),
                    "generation": result.get("generation"),
                    "view_order": list(VIEW_ORDER),
                    "tip_px": deepcopy(result.get("tip_px")),
                    "tip_visible": deepcopy(result.get("tip_visible")),
                    "target_px": deepcopy(result.get("target_px")),
                    "target_valid": deepcopy(result.get("target_valid")),
                    "view_valid": deepcopy(result.get("view_valid")),
                    "measurement_source": result.get("measurement_source"),
                    "previous_delta_px": deepcopy(
                        result.get("previous_delta_px")
                    ),
                    "ranked": bool(result.get("ranked", False)),
                    "reason": result.get("reason"),
                    "selected_index": result.get("selected_index"),
                    "legacy_selected_index": result.get("legacy_selected_index"),
                    "changed_selection": result.get("changed_selection"),
                }
                if (
                    self.measurement_history
                    and self.measurement_history[-1]["measurement_key"]
                    == record["measurement_key"]
                ):
                    self.measurement_history[-1] = record
                else:
                    self.measurement_history.append(record)
                    del self.measurement_history[:-MEASUREMENT_HISTORY_LIMIT]
        return result

    def status(self) -> dict[str, Any]:
        """Expose actual input measurements separately from V14 forecasts."""

        with self.lock:
            return {
                "format": FORMAT,
                "enabled": True,
                "non_actuating": True,
                "checkpoint": self.checkpoint_path,
                "checkpoint_step": self.adapter.checkpoint_step,
                "measurement_contract": (
                    "current_fcpgh_tracker_measurements_only_no_legacy_v12_anchors"
                ),
                "view_order": list(VIEW_ORDER),
                "measurement_count": len(self.measurement_history),
                "measurement_history": deepcopy(self.measurement_history),
                "latest_result": deepcopy(self.latest_result),
            }

    def _candidate_control(
        self,
        *,
        current_state: np.ndarray,
        chunks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        current = np.asarray(current_state, dtype=np.float32)
        commands = np.asarray(chunks, dtype=np.float32)
        if current.shape != (6,) or commands.ndim != 3 or commands.shape[1:] != (
            5,
            6,
        ):
            raise ValueError("V14 candidate control requires state [6], chunks [K,5,6]")
        sequence = np.concatenate(
            (np.broadcast_to(current[None, None], (len(commands), 1, 6)), commands),
            axis=1,
        )
        joint = np.diff(sequence, axis=1).astype(np.float32)
        states = np.concatenate((current[None], commands[:, -1]), axis=0)
        anchors = np.asarray(
            self.kinematics.forward_anchors(states), dtype=np.float32
        )
        if anchors.shape != (len(commands) + 1, 3, 3) or not np.all(
            np.isfinite(anchors)
        ):
            raise ValueError("V14 candidate FK must return finite [K+1,3,3]")
        cad = anchors[1:, 1:] - anchors[0, 1:][None]
        return joint, cad.astype(np.float32)

    @torch.inference_mode()
    def rank_plan(self, plan: Any, observer: Any) -> dict[str, Any]:
        """Return a counterfactual record; never mutate ``plan.selected_chunk``."""

        result: dict[str, Any] = {
            "format": FORMAT,
            "enabled": True,
            "actuated": False,
            "checkpoint": self.checkpoint_path,
            "checkpoint_step": self.adapter.checkpoint_step,
        }
        if observer is None or not hasattr(observer, "v14_visual_snapshot"):
            return self._publish(
                {**result, "ranked": False, "reason": "observer_unavailable"}
            )
        chunks = getattr(plan, "candidate_chunks", None)
        rows = list(getattr(plan, "candidate_rows", None) or ())
        if chunks is None or not rows:
            return self._publish(
                {
                    **result,
                    "ranked": False,
                    "reason": "candidate_bank_unavailable",
                }
            )
        commands = np.asarray(chunks, dtype=np.float32)
        if commands.shape != (len(rows), 5, 6):
            raise ValueError("V14 plan candidate bank is not aligned with reports")
        safe_indices = np.asarray(
            [index for index, row in enumerate(rows) if bool(row.get("safe"))],
            dtype=np.int64,
        )
        if not len(safe_indices):
            return self._publish(
                {**result, "ranked": False, "reason": "no_safe_candidates"}
            )

        snapshot = observer.v14_visual_snapshot()
        if tuple(snapshot.get("view_order") or ()) != VIEW_ORDER:
            raise ValueError("V14 observer view order differs from live shadow")
        tips = np.asarray(snapshot["tip_px"], dtype=np.float32)
        tip_visible = np.asarray(snapshot["tip_visible"], dtype=bool)
        target, target_valid = self._target_queries(plan)
        if tips.shape != (2, 2, 2) or tip_visible.shape != (2, 2):
            raise ValueError("V14 observer tip snapshot has invalid shape")
        view_valid = tip_visible.all(axis=1) & target_valid
        measurement_key = tuple(int(v) for v in snapshot["measurement_key"])
        measurement = {
            **result,
            "generation": int(snapshot["generation"]),
            "measurement_key": list(measurement_key),
            "displacement_clock_key": list(measurement_key[:3]),
            "view_order": list(VIEW_ORDER),
            "tip_px": tips.tolist(),
            "tip_visible": tip_visible.tolist(),
            "target_px": target.tolist(),
            "target_valid": target_valid.tolist(),
            "view_valid": view_valid.tolist(),
            "measurement_source": "current_fcpgh_tracker",
        }
        if not bool(view_valid.all()):
            return self._publish(
                {
                    **measurement,
                    "ranked": False,
                    "reason": "dual_view_tip_or_target_missing",
                },
                measurement_key=measurement_key,
            )
        query_px = np.concatenate((tips, target[:, None]), axis=1)
        target_kind = str(getattr(plan, "target", {}).get("kind") or "none")
        previous_delta = self._causal_previous_delta(
            query_px=query_px,
            generation=int(snapshot["generation"]),
            measurement_key=measurement_key,
            target_kind=target_kind,
            target_px=target,
        )
        patches = torch.as_tensor(snapshot["patch_tokens"], device=self.adapter.device)
        feature = patches[0].reshape(2, 16, 16, 192).permute(0, 3, 1, 2)
        query_tensor = torch.as_tensor(
            query_px, device=self.adapter.device, dtype=torch.float32
        )
        query_tokens = sample_feature_map_at_pixels(
            feature.float(), query_tensor, image_size=(224, 224), padding_mode="border"
        )
        joint, cad = self._candidate_control(
            current_state=np.asarray(plan.current_state, dtype=np.float32),
            chunks=commands[safe_indices],
        )
        ranking = self.adapter.rank_candidates(
            global_history=torch.as_tensor(
                snapshot["global_history"], device=self.adapter.device
            ),
            query_tokens=query_tokens[None],
            query_px=query_tensor[None],
            previous_delta_px=torch.as_tensor(
                previous_delta[None], device=self.adapter.device
            ),
            candidate_joint_increments=torch.as_tensor(
                joint[None], device=self.adapter.device
            ),
            candidate_cad_tip_displacement=torch.as_tensor(
                cad[None], device=self.adapter.device
            ),
            view_valid=torch.as_tensor(
                view_valid[None], device=self.adapter.device
            ),
        )
        local = int(ranking.decision.selected_index.item())
        selected = int(safe_indices[local])
        legacy = getattr(plan, "selected_index", None)
        return self._publish(
            {
                **measurement,
                "ranked": True,
                "reason": "dual_view_counterfactual_only",
                "safe_candidate_indices": safe_indices.tolist(),
                "selected_index": selected,
                "legacy_selected_index": None if legacy is None else int(legacy),
                "changed_selection": (
                    None if legacy is None else bool(selected != legacy)
                ),
                "energy": ranking.decision.energy[0].float().cpu().tolist(),
                "per_view_distance_px": (
                    ranking.decision.per_view_distance_px[0].float().cpu().tolist()
                ),
                "query_px": query_px.tolist(),
                "previous_delta_px": previous_delta.tolist(),
                "action_gate_active_fraction": float(
                    ranking.action_gate_active.float().mean().item()
                ),
                "zero_shot_query_contract": (
                    "Q2_generic_train_to_Q3_tip_tip_target_live"
                ),
            },
            measurement_key=measurement_key,
        )


__all__ = [
    "FORMAT",
    "LIVE_SHADOW_BEST_STEP",
    "MEASUREMENT_HISTORY_LIMIT",
    "ProdV14LiveShadow",
    "VIEW_ORDER",
]
