"""Causal dual-view live secant candidate for the V16.1 diagnostic.

This is a small online MIMO system-identification block, not a learned pixel
head.  A transition enters the fit only after its command chunk completed and
the next dual-view fingertip measurement arrived.  The fitted response maps
realized arm motion (joint delta plus local CAD-tip delta) to the measured
midpoint displacement in wrist and column views.

The proposed arm delta is restricted to the span of completed joint motions.
It is emitted only when the linearized response improves both views.  The
class is shadow-only: it never mutates a planner command or writes to hardware.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import threading
from typing import Any, Mapping

import numpy as np

from fcpg.so101_retrieval_mpc import SO101_POSITION_LOWER, SO101_POSITION_UPPER


FORMAT = "so101-prod-v16-1-live-dual-view-secant-shadow-v1"
VIEW_ORDER = ("wrist", "column")
JOINT_SCALE = 10.0
CAD_SCALE_M = 0.02
RIDGE_FRACTION = 0.05
CONTROL_RIDGE_FRACTION = 0.05
HALF_STEP_GAIN = 0.5
MAX_COMMAND_STEP = 1.7
MIN_VIEW_IMPROVEMENT_PX = 0.25
MIN_AGGREGATE_IMPROVEMENT = 0.05
MIN_REALIZED_ARM_DELTA = 0.1
MAX_MEASURED_RESPONSE_PX = 96.0


@dataclass(frozen=True)
class LiveSecantProposal:
    """One non-actuating command candidate and its predicted response."""

    prefix: np.ndarray
    endpoint_delta: np.ndarray
    predicted_response_px: np.ndarray
    before_distance_px: np.ndarray
    after_distance_px: np.ndarray
    transition_count: int
    action_rank: int

    def __post_init__(self) -> None:
        prefix = np.asarray(self.prefix, dtype=np.float32)
        endpoint = np.asarray(self.endpoint_delta, dtype=np.float32)
        response = np.asarray(self.predicted_response_px, dtype=np.float32)
        before = np.asarray(self.before_distance_px, dtype=np.float32)
        after = np.asarray(self.after_distance_px, dtype=np.float32)
        if prefix.ndim != 2 or prefix.shape[1] != 6 or len(prefix) < 5:
            raise ValueError("live secant prefix must be [H>=5,6]")
        if endpoint.shape != (5,) or response.shape != (2, 2):
            raise ValueError("live secant endpoint/response shapes are invalid")
        if before.shape != (2,) or after.shape != (2,):
            raise ValueError("live secant distance shapes are invalid")
        if not all(
            np.all(np.isfinite(value))
            for value in (prefix, endpoint, response, before, after)
        ):
            raise ValueError("live secant proposal must be finite")
        if self.transition_count < 2 or self.action_rank < 2:
            raise ValueError("live secant proposal requires two causal directions")
        object.__setattr__(self, "prefix", prefix.copy())
        object.__setattr__(self, "endpoint_delta", endpoint.copy())
        object.__setattr__(self, "predicted_response_px", response.copy())
        object.__setattr__(self, "before_distance_px", before.copy())
        object.__setattr__(self, "after_distance_px", after.copy())


@dataclass(frozen=True)
class _Transition:
    joint_delta: np.ndarray
    control_feature: np.ndarray
    response_px: np.ndarray


@dataclass(frozen=True)
class _PendingExecution:
    round_id: str
    generation: int
    action_clock: tuple[int, ...]
    before_state: np.ndarray
    before_centres_px: np.ndarray
    sent_chunk: np.ndarray


def _finite_points(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.shape != shape or not np.all(np.isfinite(points)):
        raise ValueError(f"expected finite point array with shape {shape}")
    return points


def _tip_centres(tip_px: Any, tip_visible: Any) -> tuple[np.ndarray, np.ndarray]:
    points = _finite_points(tip_px, (2, 2, 2))
    visible = np.asarray(tip_visible, dtype=bool)
    if visible.shape != (2, 2):
        raise ValueError("tip visibility must be [2,2]")
    return points.mean(axis=1), visible.all(axis=1)


class ProdV16LiveSecantShadow:
    """Observe completed actions and propose, but never execute, one prefix."""

    def __init__(
        self,
        *,
        kinematics: Any,
        horizon_raw: int = 40,
        command_chunk: int = 5,
        history_limit: int = 16,
    ) -> None:
        if horizon_raw < command_chunk or command_chunk != 5:
            raise ValueError("live secant requires a five-command prefix")
        if history_limit < 2:
            raise ValueError("live secant history_limit must be at least two")
        self.kinematics = kinematics
        self.horizon_raw = int(horizon_raw)
        self.command_chunk = int(command_chunk)
        self.transitions: deque[_Transition] = deque(maxlen=int(history_limit))
        self.pending: _PendingExecution | None = None
        self.generation: int | None = None
        self.round_observations: dict[str, dict[str, Any]] = {}
        self.latest_result: dict[str, Any] | None = None
        self.last_commit: dict[str, Any] | None = None
        self.lock = threading.RLock()

    def _tip_anchors(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError("live secant FK states must be [N,6]")
        anchors = np.asarray(self.kinematics.forward_anchors(values), dtype=np.float32)
        if anchors.shape != (len(values), 3, 3) or not np.all(np.isfinite(anchors)):
            raise ValueError("live secant FK must return finite [N,3,3]")
        return anchors[:, 1:3]

    def _reset_for_generation(self, generation: int) -> None:
        if self.generation == int(generation):
            return
        self.transitions.clear()
        self.pending = None
        self.round_observations.clear()
        self.generation = int(generation)

    def _finish_pending(
        self,
        *,
        generation: int,
        action_clock: tuple[int, ...],
        current_state: np.ndarray,
        current_centres_px: np.ndarray,
        view_valid: np.ndarray,
    ) -> str | None:
        pending = self.pending
        if pending is None or action_clock == pending.action_clock:
            return None
        self.pending = None
        if generation != pending.generation:
            return "generation_changed"
        if not bool(view_valid.all()):
            return "post_action_dual_view_measurement_missing"
        state = np.asarray(current_state, dtype=np.float32)
        joint_delta = state[:5] - pending.before_state[:5]
        if float(np.linalg.norm(joint_delta)) < MIN_REALIZED_ARM_DELTA:
            return "completed_action_had_too_little_realized_motion"
        response = current_centres_px - pending.before_centres_px
        if float(np.max(np.linalg.norm(response, axis=-1))) > MAX_MEASURED_RESPONSE_PX:
            return "post_action_response_exceeds_live_bound"
        anchors = self._tip_anchors(
            np.stack((pending.before_state, state), axis=0)
        )
        cad_delta = (anchors[1] - anchors[0]).reshape(6)
        feature = np.concatenate(
            (joint_delta / JOINT_SCALE, cad_delta / CAD_SCALE_M), axis=0
        ).astype(np.float32)
        if not np.all(np.isfinite(feature)):
            return "completed_action_control_feature_nonfinite"
        self.transitions.append(
            _Transition(
                joint_delta=joint_delta.astype(np.float32),
                control_feature=feature,
                response_px=response.astype(np.float32),
            )
        )
        return "transition_accepted"

    def _cad_jacobian(self, current_state: np.ndarray) -> np.ndarray:
        epsilon = 0.25
        states: list[np.ndarray] = []
        current = np.asarray(current_state, dtype=np.float32)
        for joint in range(5):
            lower = current.copy()
            upper = current.copy()
            lower[joint] -= epsilon
            upper[joint] += epsilon
            states.extend((lower, upper))
        anchors = self._tip_anchors(np.stack(states, axis=0)).reshape(5, 2, 6)
        return ((anchors[:, 1] - anchors[:, 0]) / (2.0 * epsilon)).T

    def _proposal(
        self,
        *,
        current_state: np.ndarray,
        centres_px: np.ndarray,
        target_px: np.ndarray,
    ) -> tuple[LiveSecantProposal | None, str, dict[str, Any]]:
        count = len(self.transitions)
        if count < 2:
            return None, "need_two_completed_action_transitions", {}
        joint = np.stack([row.joint_delta for row in self.transitions]).astype(
            np.float64
        )
        feature = np.stack(
            [row.control_feature for row in self.transitions]
        ).astype(np.float64)
        response = np.stack([row.response_px for row in self.transitions]).reshape(
            count, 4
        ).astype(np.float64)
        _, singular, right = np.linalg.svd(joint, full_matrices=False)
        if not len(singular) or float(singular[0]) <= 1.0e-8:
            return None, "completed_action_subspace_is_zero", {}
        tolerance = max(joint.shape) * np.finfo(np.float64).eps * singular[0]
        rank = int(np.sum(singular > tolerance))
        if rank < 2:
            return None, "need_two_independent_completed_action_directions", {
                "action_rank": rank
            }
        basis = right[:rank].T
        normal = feature.T @ feature
        ridge = RIDGE_FRACTION * max(
            float(np.trace(normal)) / max(normal.shape[0], 1), 1.0e-6
        )
        coefficient = np.linalg.solve(
            normal + ridge * np.eye(normal.shape[0]), feature.T @ response
        )
        cad_jacobian = self._cad_jacobian(current_state)
        feature_jacobian = np.concatenate(
            (
                np.eye(5, dtype=np.float64) / JOINT_SCALE,
                cad_jacobian.astype(np.float64) / CAD_SCALE_M,
            ),
            axis=0,
        )
        pixel_jacobian = coefficient.T @ feature_jacobian
        reduced = pixel_jacobian @ basis
        desired = (target_px - centres_px).reshape(4).astype(np.float64)
        reduced_normal = reduced.T @ reduced
        control_ridge = CONTROL_RIDGE_FRACTION * max(
            float(np.trace(reduced_normal)) / max(rank, 1), 1.0e-6
        )
        amplitude = np.linalg.solve(
            reduced_normal + control_ridge * np.eye(rank), reduced.T @ desired
        )
        endpoint_delta = HALF_STEP_GAIN * (basis @ amplitude)
        maximum_endpoint = MAX_COMMAND_STEP * self.command_chunk
        peak = float(np.max(np.abs(endpoint_delta)))
        if peak > maximum_endpoint:
            endpoint_delta *= maximum_endpoint / peak
        current = np.asarray(current_state, dtype=np.float64)
        endpoint = current.copy()
        endpoint[:5] = np.clip(
            current[:5] + endpoint_delta,
            SO101_POSITION_LOWER[:5],
            SO101_POSITION_UPPER[:5],
        )
        endpoint[5] = current[5]
        endpoint_delta = endpoint[:5] - current[:5]
        if float(np.max(np.abs(endpoint_delta))) < 0.1:
            return None, "predicted_control_is_too_small", {"action_rank": rank}
        predicted = (pixel_jacobian @ endpoint_delta).reshape(2, 2)
        before = np.linalg.norm(target_px - centres_px, axis=-1)
        after = np.linalg.norm(target_px - (centres_px + predicted), axis=-1)
        aggregate_gain = 1.0 - float(after.mean() / max(before.mean(), 1.0e-6))
        diagnostics = {
            "action_rank": rank,
            "before_distance_px": before.astype(np.float32).tolist(),
            "after_distance_px": after.astype(np.float32).tolist(),
            "aggregate_improvement_fraction": aggregate_gain,
        }
        if not bool(np.all(after + MIN_VIEW_IMPROVEMENT_PX <= before)):
            return None, "no_joint_dual_view_improvement", diagnostics
        if aggregate_gain < MIN_AGGREGATE_IMPROVEMENT:
            return None, "dual_view_improvement_below_minimum", diagnostics
        prefix = np.broadcast_to(endpoint[None], (self.horizon_raw, 6)).copy()
        alpha = np.linspace(
            1.0 / self.command_chunk, 1.0, self.command_chunk, dtype=np.float64
        )
        prefix[: self.command_chunk, :5] = (
            current[None, :5] + alpha[:, None] * endpoint_delta[None]
        )
        prefix[: self.command_chunk, 5] = current[5]
        proposal = LiveSecantProposal(
            prefix=prefix.astype(np.float32),
            endpoint_delta=endpoint_delta.astype(np.float32),
            predicted_response_px=predicted.astype(np.float32),
            before_distance_px=before.astype(np.float32),
            after_distance_px=after.astype(np.float32),
            transition_count=count,
            action_rank=rank,
        )
        return proposal, "dual_view_candidate_available", diagnostics

    @staticmethod
    def _target_queries(plan: Any) -> tuple[np.ndarray, np.ndarray, str]:
        target = getattr(plan, "target", None)
        descriptors = getattr(plan, "target_descriptors", None)
        if not isinstance(target, Mapping) or not isinstance(descriptors, Mapping):
            raise ValueError("V16.1 live secant requires plan target descriptors")
        kind = str(target.get("kind") or "none")
        points: list[np.ndarray] = []
        valid: list[bool] = []
        for view in VIEW_ORDER:
            descriptor = descriptors.get(f"pickup_{view}")
            point: np.ndarray | None = None
            if kind == "pickup" and isinstance(descriptor, Mapping) and bool(
                descriptor.get("visible", False)
            ):
                candidate = np.asarray(descriptor.get("centroid_px"), dtype=np.float32)
                if (
                    candidate.shape == (2,)
                    and np.all(np.isfinite(candidate))
                    and np.all((candidate >= 0.0) & (candidate <= 223.0))
                ):
                    point = candidate
            points.append(
                np.full(2, np.nan, dtype=np.float32) if point is None else point
            )
            valid.append(point is not None)
        return np.stack(points), np.asarray(valid, dtype=bool), kind

    def rank_plan(self, plan: Any, observer: Any) -> dict[str, Any]:
        """Update the causal fit and publish one non-actuating proposal."""

        base: dict[str, Any] = {
            "format": FORMAT,
            "enabled": True,
            "non_actuating": True,
        }
        if observer is None or not hasattr(observer, "v14_visual_snapshot"):
            result = {**base, "proposed": False, "reason": "observer_unavailable"}
            with self.lock:
                self.latest_result = deepcopy(result)
            return result
        observer_status = getattr(observer, "status", None)
        if callable(observer_status):
            status = observer_status()
            if isinstance(status, Mapping) and not bool(status.get("initialized")):
                result = {
                    **base,
                    "proposed": False,
                    "reason": "observer_visual_state_uninitialized",
                }
                with self.lock:
                    self.latest_result = deepcopy(result)
                return result
        snapshot = observer.v14_visual_snapshot()
        if tuple(snapshot.get("view_order") or ()) != VIEW_ORDER:
            raise ValueError("V16.1 live secant observer view order mismatch")
        centres, tip_valid = _tip_centres(
            snapshot["tip_px"], snapshot["tip_visible"]
        )
        key = tuple(int(value) for value in snapshot["measurement_key"])
        if len(key) < 3:
            raise ValueError("V16.1 live secant measurement key is incomplete")
        action_clock = key[:3]
        generation = int(snapshot["generation"])
        state = _finite_points(getattr(plan, "current_state"), (6,))
        target, target_valid, target_kind = self._target_queries(plan)
        round_id = str(getattr(getattr(plan, "guard", None), "round_id", ""))
        if not round_id:
            raise ValueError("V16.1 live secant plan needs a round id")
        with self.lock:
            self._reset_for_generation(generation)
            transition_status = self._finish_pending(
                generation=generation,
                action_clock=action_clock,
                current_state=state,
                current_centres_px=centres,
                view_valid=tip_valid,
            )
            self.round_observations[round_id] = {
                "generation": generation,
                "action_clock": action_clock,
                "state": state.copy(),
                "centres_px": centres.copy(),
                "view_valid": tip_valid.copy(),
            }
            while len(self.round_observations) > 32:
                self.round_observations.pop(next(iter(self.round_observations)))
            view_valid = tip_valid & target_valid
            if not bool(view_valid.all()):
                proposal = None
                reason = "dual_view_tip_or_pickup_target_missing"
                diagnostics: dict[str, Any] = {}
            else:
                proposal, reason, diagnostics = self._proposal(
                    current_state=state,
                    centres_px=centres,
                    target_px=target,
                )
            result = {
                **base,
                "generation": generation,
                "measurement_key": list(key),
                "action_clock": list(action_clock),
                "target_kind": target_kind,
                "tip_centres_px": centres.tolist(),
                "target_px": target.tolist(),
                "view_valid": view_valid.tolist(),
                "transition_status": transition_status,
                "transition_count": len(self.transitions),
                "proposed": proposal is not None,
                "reason": reason,
                **diagnostics,
            }
            if proposal is not None:
                result["candidate"] = {
                    "source": "prod_v16_1_live_dual_view_secant",
                    "endpoint_delta": proposal.endpoint_delta.tolist(),
                    "command_chunk": proposal.prefix[: self.command_chunk].tolist(),
                    "predicted_response_px": proposal.predicted_response_px.tolist(),
                    "before_distance_px": proposal.before_distance_px.tolist(),
                    "after_distance_px": proposal.after_distance_px.tolist(),
                    "transition_count": proposal.transition_count,
                    "action_rank": proposal.action_rank,
                }
            self.latest_result = deepcopy(result)
            return result

    def commit_execution(self, *, plan: Any, sent_chunk: Any) -> dict[str, Any]:
        """Arm one transition only after the five commands actually completed."""

        round_id = str(getattr(getattr(plan, "guard", None), "round_id", ""))
        sent = np.asarray(sent_chunk, dtype=np.float32)
        if sent.shape != (self.command_chunk, 6) or not np.all(np.isfinite(sent)):
            raise ValueError("V16.1 live secant completed chunk must be finite [5,6]")
        with self.lock:
            observation = self.round_observations.get(round_id)
            if observation is None:
                raise ValueError("V16.1 live secant has no matching plan observation")
            if self.pending is not None:
                raise RuntimeError("V16.1 live secant pending transition was not observed")
            if not bool(np.asarray(observation["view_valid"], dtype=bool).all()):
                result = {
                    "committed": False,
                    "reason": "pre_action_dual_view_measurement_missing",
                    "round_id": round_id,
                }
                self.last_commit = deepcopy(result)
                return result
            self.pending = _PendingExecution(
                round_id=round_id,
                generation=int(observation["generation"]),
                action_clock=tuple(observation["action_clock"]),
                before_state=np.asarray(observation["state"], dtype=np.float32).copy(),
                before_centres_px=np.asarray(
                    observation["centres_px"], dtype=np.float32
                ).copy(),
                sent_chunk=sent.copy(),
            )
            result = {
                "committed": True,
                "reason": "awaiting_post_action_measurement",
                "round_id": round_id,
                "action_clock": list(self.pending.action_clock),
                "commanded_endpoint_delta": (
                    sent[-1, :5] - self.pending.before_state[:5]
                ).tolist(),
            }
            self.last_commit = deepcopy(result)
            return result

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "format": FORMAT,
                "enabled": True,
                "non_actuating": True,
                "generation": self.generation,
                "transition_count": len(self.transitions),
                "pending_transition": self.pending is not None,
                "last_commit": deepcopy(self.last_commit),
                "latest_result": deepcopy(self.latest_result),
            }


__all__ = [
    "FORMAT",
    "LiveSecantProposal",
    "ProdV16LiveSecantShadow",
    "VIEW_ORDER",
]
