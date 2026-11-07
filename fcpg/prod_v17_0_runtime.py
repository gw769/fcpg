"""Non-actuating runtime bridge for the Prod-V17.0 FCPG.

The bridge owns the recurrent F/M query state and consumes only settled
endpoint observations plus command/readback chunks reported by an external
runtime.  It never opens a camera or robot, ranks a candidate, selects a
candidate, or sends a command.

``V170PlainShadowStack`` runs the three V17 final models behind the same
no-influence boundary.  ``V170V168ShadowStack`` remains only as a comparison
adapter for the old V16.8 policy models.  Returned proposals and predicted
states are diagnostics only; the caller's candidate bank is never returned or
mutated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

import numpy as np
import torch

from fcpg.prod_v16_5 import StateFKLeWM
from fcpg.prod_v16_8 import FullHistoryFiveCommandActor
from fcpg.prod_v17_0 import (
    COMMAND_STEPS,
    IMAGE_SIZE,
    MODEL_DIMENSION,
    TIPS,
    VIEWS,
    V170RecurrentQueryFCPG,
)
from fcpg.prod_v17_1 import V171FactorizedQueryFCPG
from fcpg.prod_v17_4 import (
    V174FCPGMemory,
    V174ResidualBeliefFCPG,
    VIEW_NAMES as V174_VIEW_ORDER,
)
from fcpg.prod_v17_0_evaluation import (
    load_v170_final_model,
)


VIEW_ORDER = ("wrist", "column")
STATE_DIMENSION = 6
CANONICAL_TO_DEGREES = 100.0
WINDOW_TRANSITIONS = 4
WINDOW_COMMANDS = WINDOW_TRANSITIONS * COMMAND_STEPS
PROXY_GOAL_COMMAND_INDEX = WINDOW_COMMANDS - 1
ACTION_LOWER = np.asarray([-1, -1, -1, -1, -1, 0], dtype=np.float32)
ACTION_UPPER = np.ones(STATE_DIMENSION, dtype=np.float32)
ACTION_STEP_CAP = np.asarray([0.04, 0.04, 0.04, 0.04, 0.04, 0.08], dtype=np.float32)
V171_CHECKPOINT_FORMAT = "so101-prod-v17.1-fcpg-final-v2"
V174_CHECKPOINT_FORMAT = "prod-v17.4-fcpg-final-v1"
V174_CHECKPOINT_ARCHITECTURE = (
    "dual-view-local-measurement-residual-belief-observer"
)
# The maintained realtime planner predates V17.2 and exposes wrist first.
# V17.4 was trained column first.  These explicit permutations prevent a
# seemingly plausible but physically wrong cross-camera deployment.
V174_FROM_RUNTIME_VIEW = tuple(VIEW_ORDER.index(view) for view in V174_VIEW_ORDER)
RUNTIME_FROM_V174_VIEW = tuple(V174_VIEW_ORDER.index(view) for view in VIEW_ORDER)

RuntimeFCPG = (
    V170RecurrentQueryFCPG
    | V171FactorizedQueryFCPG
    | V174ResidualBeliefFCPG
)


class V170RuntimeError(RuntimeError):
    """A tensor, clock, or non-actuating runtime contract drifted."""


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _finite(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not bool(np.isfinite(result).all()):
        raise V170RuntimeError(f"{name} must be finite {list(shape)}")
    return result


def _mapping_keys(name: str, value: Mapping[str, Any]) -> None:
    if set(value) != set(VIEW_ORDER):
        raise V170RuntimeError(f"{name} must contain wrist and column")


def _rgb_chw(value: np.ndarray, *, view: str) -> np.ndarray:
    image = np.asarray(value)
    if image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3):
        image = np.moveaxis(image, -1, 0)
    if image.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise V170RuntimeError(f"{view} RGB must be [3,224,224] or [224,224,3]")
    if np.issubdtype(image.dtype, np.integer):
        if bool((image < 0).any()) or bool((image > 255).any()):
            raise V170RuntimeError(f"{view} integer RGB leaves [0,255]")
        result = image.astype(np.float32) / np.float32(255.0)
    else:
        result = image.astype(np.float32)
        if not bool(np.isfinite(result).all()) or bool(
            ((result < 0.0) | (result > 1.0)).any()
        ):
            raise V170RuntimeError(f"{view} floating RGB must be finite in [0,1]")
    return result


@dataclass(frozen=True)
class V170ShadowEndpoint:
    """One settled V17 measurement detached from model storage."""

    endpoint: str
    fm_px: np.ndarray
    midpoint_px: np.ndarray
    valid_fm: np.ndarray
    occlusion_probability: np.ndarray
    transition_consumed: bool
    window_completed: bool
    commands_sent: int = 0
    decision_influence: bool = False
    probability_semantics: str = "physical_occlusion"
    projection_invalid_probability: np.ndarray | None = None


def _load_runtime_fcpg(
    checkpoint: str | Path,
    *,
    device: str | torch.device,
) -> RuntimeFCPG:
    """Strict-load a supported ordinary V17 FCPG final.

    V17.1 intentionally keeps the same plain five-field checkpoint surface as
    V17.0.  Selection is by the declared format, never by a digest, receipt, or
    filename convention.
    """

    resolved = Path(checkpoint).expanduser().resolve()
    if not resolved.is_file():
        raise V170RuntimeError(f"final FCPG checkpoint is absent: {resolved}")
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - torch text varies by version
        raise V170RuntimeError("final FCPG checkpoint cannot be read") from exc
    if isinstance(payload, Mapping) and payload.get("format") == V174_CHECKPOINT_FORMAT:
        expected_fields = {"format", "architecture", "state_dict"}
        if set(payload) != expected_fields:
            raise V170RuntimeError("V17.4 final checkpoint fields differ")
        if payload.get("architecture") != V174_CHECKPOINT_ARCHITECTURE:
            raise V170RuntimeError("V17.4 final checkpoint architecture differs")
        received = payload.get("state_dict")
        if not isinstance(received, Mapping):
            raise V170RuntimeError("V17.4 final checkpoint state_dict must be a mapping")
        model = V174ResidualBeliefFCPG()
        expected = model.state_dict()
        if set(received) != set(expected):
            raise V170RuntimeError(
                "V17.4 FCPG state fields differ: "
                f"missing={sorted(set(expected) - set(received))} "
                f"unexpected={sorted(set(received) - set(expected))}"
            )
        checked: dict[str, torch.Tensor] = {}
        for name, reference in expected.items():
            value = received[name]
            if not isinstance(value, torch.Tensor):
                raise V170RuntimeError(f"V17.4 FCPG state[{name}] is not a tensor")
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise V170RuntimeError(
                    f"V17.4 FCPG state[{name}] shape or dtype differs"
                )
            if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all()
            ):
                raise V170RuntimeError(f"V17.4 FCPG state[{name}] is non-finite")
            checked[name] = value.detach().clone()
        model.load_state_dict(checked, strict=True)
        model.to(torch.device(device)).eval().requires_grad_(False)
        if model.training or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise V170RuntimeError("V17.4 final FCPG did not enter frozen eval mode")
        return model

    if not isinstance(payload, Mapping) or payload.get("format") != V171_CHECKPOINT_FORMAT:
        model = load_v170_final_model(resolved, stage="fcpg", device=device)
        if not isinstance(model, V170RecurrentQueryFCPG):
            raise V170RuntimeError("loaded V17.0 FCPG class differs")
        return model

    expected_fields = {"format", "stage", "global_step", "config", "model"}
    if set(payload) != expected_fields:
        raise V170RuntimeError("V17.1 final checkpoint fields differ")
    if payload.get("stage") != "fcpg":
        raise V170RuntimeError("V17.1 final checkpoint stage differs")
    step = payload.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise V170RuntimeError("V17.1 final checkpoint global_step is invalid")
    if not isinstance(payload.get("config"), Mapping):
        raise V170RuntimeError("V17.1 final checkpoint config must be a mapping")
    received = payload.get("model")
    if not isinstance(received, Mapping):
        raise V170RuntimeError("V17.1 final checkpoint model must be a mapping")

    model = V171FactorizedQueryFCPG()
    expected = model.state_dict()
    if set(received) != set(expected):
        raise V170RuntimeError(
            "V17.1 FCPG state fields differ: "
            f"missing={sorted(set(expected) - set(received))} "
            f"unexpected={sorted(set(received) - set(expected))}"
        )
    checked: dict[str, torch.Tensor] = {}
    for name, reference in expected.items():
        value = received[name]
        if not isinstance(value, torch.Tensor):
            raise V170RuntimeError(f"V17.1 FCPG state[{name}] is not a tensor")
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise V170RuntimeError(
                f"V17.1 FCPG state[{name}] shape or dtype differs"
            )
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise V170RuntimeError(f"V17.1 FCPG state[{name}] is non-finite")
        checked[name] = value.detach().clone()
    model.load_state_dict(checked, strict=True)
    model.to(torch.device(device)).eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise V170RuntimeError("V17.1 final FCPG did not enter frozen eval mode")
    return model


@dataclass(frozen=True)
class V170ShadowPolicyResult:
    """Counterfactual V16.8 Actor/LeWM output, never a candidate insertion."""

    generated: bool
    actor_chunk_deg: np.ndarray
    predicted_states_deg: np.ndarray
    projector_fixed_point: bool
    commands_sent: int = 0
    decision_influence: bool = False


@dataclass(frozen=True)
class V170ActorShadowResult:
    """One V17 Actor five-command counterfactual with its full-history clock."""

    transition: str
    action_deg: np.ndarray
    baseline_deg: np.ndarray
    residual_deg: np.ndarray
    history_valid: tuple[bool, bool, bool]
    commands_sent: int = 0
    decision_influence: bool = False


@dataclass(frozen=True)
class V170CandidateConsequences:
    """Frozen V17 LeWM outcomes for an unchanged offline candidate bank."""

    predicted_state_path_deg: np.ndarray
    baseline_state_path_deg: np.ndarray
    residual_state_path_deg: np.ndarray
    commands_sent: int = 0
    decision_influence: bool = False


@dataclass(frozen=True)
class _V170PolicyRound:
    full256: np.ndarray
    view_valid: np.ndarray
    current_state_deg: np.ndarray
    current_state: np.ndarray


class V170ShadowFCPGAdapter:
    """Stateful H0--H4 FCPG inference with no actuator surface."""

    mode = "shadow"
    decision_influence = False

    def __init__(
        self,
        *,
        model: RuntimeFCPG,
        device: str | torch.device = "cpu",
    ) -> None:
        resolved = torch.device(device)
        if resolved.type not in {"cpu", "cuda"}:
            raise V170RuntimeError("V17 runtime device must be cpu or cuda")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise V170RuntimeError("requested V17 CUDA device is unavailable")
        if not isinstance(
            model,
            (
                V170RecurrentQueryFCPG,
                V171FactorizedQueryFCPG,
                V174ResidualBeliefFCPG,
            ),
        ):
            raise V170RuntimeError("V17 runtime requires a supported V17 FCPG")
        self._model = model.to(resolved).eval().requires_grad_(False)
        self._device = resolved
        self._query: torch.Tensor | V174FCPGMemory | None = (
            None
            if isinstance(self._model, V174ResidualBeliefFCPG)
            else self._model.initial_query(
                1, device=resolved, dtype=torch.float32
            ).detach()
        )
        self._pending: tuple[np.ndarray, np.ndarray, str] | None = None
        # One causal decode is committed per endpoint clock.  Re-observing the
        # same stopped endpoint may use a fresher image, but must replay from
        # the original prior/transition rather than recurrently feeding the
        # endpoint's own decoded query back into itself.
        self._endpoint_context: tuple[
            torch.Tensor | V174FCPGMemory | None,
            np.ndarray,
            np.ndarray,
            bool,
            np.ndarray | None,
        ] | None = None
        self._transition = 0
        self._window = 0
        self._observations = 0
        self._external_completions = 0
        self._last_endpoint: str | None = None
        self._lock = threading.RLock()

    @classmethod
    def from_final_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "V170ShadowFCPGAdapter":
        model = _load_runtime_fcpg(checkpoint, device=device)
        return cls(model=model, device=device)

    @property
    def model_family(self) -> str:
        if isinstance(self._model, V174ResidualBeliefFCPG):
            return "v17.4"
        if isinstance(self._model, V171FactorizedQueryFCPG):
            return "v17.1"
        return "v17.0"

    @property
    def transition(self) -> int:
        return int(self._transition)

    @property
    def commands_sent(self) -> int:
        return 0

    @property
    def parameter_count(self) -> int:
        return int(self._model.parameter_count)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "format": "so101-prod-v17.0-shadow-runtime-v1",
                "model_family": self.model_family,
                "mode": self.mode,
                "decision_influence": False,
                "commands_sent": 0,
                "camera_opened": False,
                "robot_opened": False,
                "candidate_bank_mutated": False,
                "window": self._window,
                "next_endpoint": f"H{self._transition}",
                "observations": self._observations,
                "external_completions_observed": self._external_completions,
                "transition_pending": self._pending is not None,
                "last_endpoint": self._last_endpoint,
            }

    def reset_window(self, reason: str) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason.strip():
            raise V170RuntimeError("reset reason must be non-empty")
        with self._lock:
            before = f"H{self._transition}"
            self._query = (
                None
                if isinstance(self._model, V174ResidualBeliefFCPG)
                else self._model.initial_query(
                    1, device=self._device, dtype=torch.float32
                ).detach()
            )
            self._pending = None
            self._endpoint_context = None
            self._transition = 0
            self._window += 1
            self._last_endpoint = None
            return {
                "reason": reason.strip(),
                "before": before,
                "after": "H0",
                "window": self._window,
                "commands_sent": 0,
            }

    def observe_external_completion(
        self,
        *,
        completed_commands_deg: np.ndarray,
        measured_joint_path_deg: np.ndarray,
        round_id: str,
    ) -> None:
        """Queue a chunk that another runtime already confirmed as completed."""

        commands = _finite(
            "completed_commands_deg",
            completed_commands_deg,
            (COMMAND_STEPS, STATE_DIMENSION),
        )
        path = _finite(
            "measured_joint_path_deg",
            measured_joint_path_deg,
            (COMMAND_STEPS + 1, STATE_DIMENSION),
        )
        if not isinstance(round_id, str) or not round_id:
            raise V170RuntimeError("round_id must be non-empty")
        canonical_commands = commands / np.float32(CANONICAL_TO_DEGREES)
        canonical_path = path / np.float32(CANONICAL_TO_DEGREES)
        if bool(
            ((canonical_commands < ACTION_LOWER) | (canonical_commands > ACTION_UPPER)).any()
            or ((canonical_path < ACTION_LOWER) | (canonical_path > ACTION_UPPER)).any()
        ):
            raise V170RuntimeError("external completion leaves canonical joint bounds")
        with self._lock:
            if self._pending is not None:
                raise V170RuntimeError("previous external completion is still pending")
            if self._transition >= WINDOW_TRANSITIONS:
                raise V170RuntimeError("H4 endpoint must be observed before another chunk")
            # Keep transport-native degree values. V17.0 converts only at its
            # model boundary; V17.4 was explicitly trained in degrees.
            self._pending = (
                commands.astype(np.float32, copy=True),
                path.astype(np.float32, copy=True),
                round_id,
            )
            self._external_completions += 1

    @torch.inference_mode()
    def observe_endpoint(
        self,
        *,
        rgb_by_view: Mapping[str, np.ndarray],
        frozen_v12_tokens_by_view: Mapping[str, np.ndarray],
        nominal_fm_px_by_view: Mapping[str, np.ndarray],
        view_valid: Mapping[str, bool],
        current_state_deg: np.ndarray | None = None,
    ) -> V170ShadowEndpoint:
        _mapping_keys("rgb_by_view", rgb_by_view)
        _mapping_keys("frozen_v12_tokens_by_view", frozen_v12_tokens_by_view)
        _mapping_keys("nominal_fm_px_by_view", nominal_fm_px_by_view)
        _mapping_keys("view_valid", view_valid)
        rgb = np.stack([_rgb_chw(rgb_by_view[view], view=view) for view in VIEW_ORDER])
        tokens = np.stack(
            [
                _finite(
                    f"{view} frozen tokens",
                    frozen_v12_tokens_by_view[view],
                    (256, MODEL_DIMENSION),
                )
                for view in VIEW_ORDER
            ]
        )
        nominal = np.stack(
            [
                _finite(
                    f"{view} nominal F/M",
                    nominal_fm_px_by_view[view],
                    (TIPS, 2),
                )
                for view in VIEW_ORDER
            ]
        )
        if bool(((nominal < 0.0) | (nominal > IMAGE_SIZE - 1.0)).any()):
            raise V170RuntimeError("nominal F/M leaves the 224 canvas")
        valid = np.asarray([bool(view_valid[view]) for view in VIEW_ORDER], dtype=bool)
        if not bool(valid.any()):
            raise V170RuntimeError("at least one endpoint view must be valid")
        current_q_deg = (
            None
            if current_state_deg is None
            else _finite("current_state_deg", current_state_deg, (STATE_DIMENSION,))
        )
        if isinstance(self._model, V174ResidualBeliefFCPG) and current_q_deg is None:
            raise V170RuntimeError("V17.4 endpoint requires current_state_deg")

        with self._lock:
            pending = self._pending
            new_endpoint = self._endpoint_context is None or pending is not None
            if new_endpoint:
                prior = (
                    None
                    if self._query is None
                    else (
                        self._query.detach()
                        if isinstance(self._query, V174FCPGMemory)
                        else self._query.detach().clone()
                    )
                )
                if pending is None:
                    commands = np.zeros(
                        (COMMAND_STEPS, STATE_DIMENSION), dtype=np.float32
                    )
                    path = np.zeros(
                        (COMMAND_STEPS + 1, STATE_DIMENSION), dtype=np.float32
                    )
                    transition_valid = False
                else:
                    commands, path, _round_id = pending
                    transition_valid = True
                self._endpoint_context = (
                    prior,
                    commands.copy(),
                    path.copy(),
                    transition_valid,
                    None if current_q_deg is None else current_q_deg.copy(),
                )
            else:
                prior, commands, path, transition_valid, bound_q_deg = (
                    self._endpoint_context
                )
                if (
                    current_q_deg is not None
                    and bound_q_deg is not None
                    and not np.allclose(
                        current_q_deg, bound_q_deg, rtol=0.0, atol=1.0e-4
                    )
                ):
                    raise V170RuntimeError(
                        "endpoint state changed without a completed chunk"
                    )
            if isinstance(self._model, V171FactorizedQueryFCPG):
                output = self._model(
                    torch.from_numpy(rgb[None]).to(self._device),
                    torch.from_numpy(tokens[None]).to(self._device),
                    torch.from_numpy(nominal[None]).to(self._device),
                    torch.from_numpy(valid[None]).to(self._device),
                    previous_query=prior,
                )
                fm_px = output.fused_fm_px
                midpoint_px = output.fused_midpoint_px
                valid_fm = torch.from_numpy(valid[None, :, None]).to(
                    self._device
                ).expand(-1, -1, TIPS)
                occlusion_probability = torch.zeros_like(
                    output.projection_invalid_probability
                )
                projection_invalid_probability = (
                    output.projection_invalid_probability
                )
                probability_semantics = "projection_invalid"
            elif isinstance(self._model, V174ResidualBeliefFCPG):
                assert current_q_deg is not None
                model_order = np.asarray(V174_FROM_RUNTIME_VIEW, dtype=np.int64)
                runtime_order = np.asarray(RUNTIME_FROM_V174_VIEW, dtype=np.int64)
                output = self._model(
                    torch.from_numpy(rgb[model_order][None]).to(self._device),
                    torch.from_numpy(tokens[model_order][None]).to(self._device),
                    torch.from_numpy(nominal[model_order][None]).to(self._device),
                    torch.from_numpy(current_q_deg[None]).to(self._device),
                    torch.from_numpy(commands[None]).to(self._device),
                    torch.from_numpy(path[None]).to(self._device),
                    torch.tensor(
                        [transition_valid], dtype=torch.bool, device=self._device
                    ),
                    torch.from_numpy(valid[model_order][None]).to(self._device),
                    previous_memory=prior,
                )
                fm_px = output.fm_px[:, runtime_order]
                midpoint_px = output.midpoint_px[:, runtime_order]
                valid_fm = torch.from_numpy(valid[None, :, None]).to(
                    self._device
                ).expand(-1, -1, TIPS)
                occlusion_probability = torch.zeros(
                    (1, len(VIEW_ORDER), TIPS),
                    dtype=fm_px.dtype,
                    device=self._device,
                )
                projection_invalid_probability = None
                probability_semantics = "not_modeled"
            else:
                canonical_commands = commands / np.float32(CANONICAL_TO_DEGREES)
                canonical_path = path / np.float32(CANONICAL_TO_DEGREES)
                assert isinstance(prior, torch.Tensor)
                output = self._model(
                    torch.from_numpy(rgb[None]).to(self._device),
                    torch.from_numpy(tokens[None]).to(self._device),
                    torch.from_numpy(nominal[None]).to(self._device),
                    torch.from_numpy(canonical_commands[None]).to(self._device),
                    torch.from_numpy(canonical_path[None]).to(self._device),
                    prior,
                    torch.tensor(
                        [transition_valid], dtype=torch.bool, device=self._device
                    ),
                    torch.from_numpy(valid[None]).to(self._device),
                )
                fm_px = output.fm_px
                midpoint_px = output.midpoint_px
                valid_fm = output.valid_cfm[:, :, 1:]
                occlusion_probability = output.occluded_probability
                projection_invalid_probability = None
                probability_semantics = "physical_occlusion"
            if new_endpoint:
                self._query = (
                    output.next_memory.detach()
                    if isinstance(self._model, V174ResidualBeliefFCPG)
                    else output.next_query.detach()
                )
                if transition_valid:
                    self._transition += 1
                    self._pending = None
            endpoint = f"H{self._transition}"
            completed = self._transition == WINDOW_TRANSITIONS
            result = V170ShadowEndpoint(
                endpoint=endpoint,
                fm_px=_readonly(fm_px[0].float().cpu().numpy()),
                midpoint_px=_readonly(midpoint_px[0].float().cpu().numpy()),
                valid_fm=_readonly(valid_fm[0].cpu().numpy()),
                occlusion_probability=_readonly(
                    occlusion_probability[0].float().cpu().numpy()
                ),
                transition_consumed=bool(new_endpoint and transition_valid),
                window_completed=completed,
                probability_semantics=probability_semantics,
                projection_invalid_probability=(
                    None
                    if projection_invalid_probability is None
                    else _readonly(
                        projection_invalid_probability[0].float().cpu().numpy()
                    )
                ),
            )
            self._observations += 1
            self._last_endpoint = endpoint
            if completed:
                # H4 is returned to the caller, then the next causal window is
                # atomically prepared at H0.  No fifth action can be attached
                # to the just-finished training window.
                self._query = (
                    None
                    if isinstance(self._model, V174ResidualBeliefFCPG)
                    else self._model.initial_query(
                        1, device=self._device, dtype=torch.float32
                    ).detach()
                )
                self._endpoint_context = None
                self._transition = 0
                self._window += 1
            return result


class V170PlainShadowStack:
    """The complete plain-checkpoint V17 FCPG + LeWM + Actor shadow.

    FCPG owns its H0--H4 semantic-query clock.  The Actor owns the registered
    H0--H3 full-history window and resets only after the fourth externally
    completed chunk.  LeWM evaluates candidate consequences at the Actor's
    current history clock.  None of the outputs is inserted into a planner.
    """

    mode = "shadow"
    decision_influence = False

    def __init__(
        self,
        *,
        fcpg: V170ShadowFCPGAdapter,
        lewm: StateFKLeWM,
        actor: FullHistoryFiveCommandActor,
        device: str | torch.device = "cpu",
    ) -> None:
        resolved = torch.device(device)
        if resolved.type not in {"cpu", "cuda"}:
            raise V170RuntimeError("V17 plain stack device must be cpu or cuda")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise V170RuntimeError("requested V17 plain-stack CUDA device is unavailable")
        if not isinstance(fcpg, V170ShadowFCPGAdapter):
            raise V170RuntimeError("V17 plain stack requires its FCPG adapter")
        if not isinstance(lewm, StateFKLeWM) or not isinstance(
            actor, FullHistoryFiveCommandActor
        ):
            raise V170RuntimeError("V17 plain stack model classes differ")
        self.fcpg = fcpg
        self.lewm = lewm.to(resolved).eval().requires_grad_(False)
        self.actor = actor.to(resolved).eval().requires_grad_(False)
        self._device = resolved
        self._history = np.zeros(
            (3, COMMAND_STEPS, STATE_DIMENSION), dtype=np.float32
        )
        self._transition = 0
        self._policy_window = 0
        self._fixed_goal: np.ndarray | None = None
        self._round: _V170PolicyRound | None = None
        self._actor_calls = 0
        self._lewm_calls = 0
        self._external_completions = 0
        self._lock = threading.RLock()

    @classmethod
    def from_final_checkpoints(
        cls,
        *,
        fcpg_checkpoint: str | Path,
        lewm_checkpoint: str | Path,
        actor_checkpoint: str | Path,
        device: str | torch.device = "cpu",
    ) -> "V170PlainShadowStack":
        resolved = torch.device(device)
        fcpg = _load_runtime_fcpg(fcpg_checkpoint, device=resolved)
        lewm = load_v170_final_model(lewm_checkpoint, stage="lewm", device=resolved)
        actor = load_v170_final_model(actor_checkpoint, stage="actor", device=resolved)
        return cls(
            fcpg=V170ShadowFCPGAdapter(model=fcpg, device=resolved),
            lewm=lewm,  # type: ignore[arg-type]
            actor=actor,  # type: ignore[arg-type]
            device=resolved,
        )

    @property
    def transition(self) -> int:
        return int(self._transition)

    @property
    def commands_sent(self) -> int:
        return 0

    def _clear_policy_window(self) -> None:
        self._history.fill(0.0)
        self._transition = 0
        self._fixed_goal = None
        self._round = None
        self._policy_window += 1

    def reset_window(self, reason: str) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason.strip():
            raise V170RuntimeError("reset reason must be non-empty")
        with self._lock:
            self.fcpg.reset_window(reason)
            before = f"H{self._transition}"
            self._clear_policy_window()
            return {
                "reason": reason.strip(),
                "before": before,
                "after": "H0",
                "policy_window": self._policy_window,
                "commands_sent": 0,
            }

    @staticmethod
    def _canonical_state(name: str, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        degrees = _finite(name, value, (STATE_DIMENSION,))
        canonical = degrees / np.float32(CANONICAL_TO_DEGREES)
        if bool(((canonical < ACTION_LOWER) | (canonical > ACTION_UPPER)).any()):
            raise V170RuntimeError(f"{name} leaves canonical joint bounds")
        return degrees.astype(np.float32, copy=True), canonical.astype(
            np.float32, copy=True
        )

    def _past_commands(self, current: np.ndarray) -> np.ndarray:
        if self._transition == 0:
            return np.repeat(current[None], 2, axis=0).astype(np.float32)
        if self._transition == 1:
            return np.stack((current, self._history[0, -1])).astype(np.float32)
        return self._history[
            self._transition - 2 : self._transition, -1
        ].astype(np.float32, copy=True)

    def _require_round(self, current_state_deg: np.ndarray) -> _V170PolicyRound:
        if self._round is None:
            raise V170RuntimeError("V17 policy shadow needs a prepared endpoint")
        current = _finite(
            "current_state_deg", current_state_deg, (STATE_DIMENSION,)
        )
        if not np.array_equal(current, self._round.current_state_deg):
            raise V170RuntimeError("V17 policy state changed after endpoint capture")
        return self._round

    def prepare_endpoint(
        self,
        *,
        rgb_by_view: Mapping[str, np.ndarray],
        frozen_v12_tokens_by_view: Mapping[str, np.ndarray],
        nominal_fm_px_by_view: Mapping[str, np.ndarray],
        view_valid: Mapping[str, bool],
        current_state_deg: np.ndarray,
    ) -> V170ShadowEndpoint:
        _mapping_keys("frozen_v12_tokens_by_view", frozen_v12_tokens_by_view)
        _mapping_keys("view_valid", view_valid)
        scene = np.stack(
            [
                _finite(
                    f"{view} frozen tokens",
                    frozen_v12_tokens_by_view[view],
                    (256, MODEL_DIMENSION),
                )
                for view in VIEW_ORDER
            ]
        )
        views = np.asarray(
            [bool(view_valid[view]) for view in VIEW_ORDER], dtype=bool
        )
        if not bool(views.any()):
            raise V170RuntimeError("V17 policy needs at least one valid view")
        state_deg, state = self._canonical_state(
            "current_state_deg", current_state_deg
        )
        scene[~views] = 0.0
        with self._lock:
            measurement = self.fcpg.observe_endpoint(
                rgb_by_view=rgb_by_view,
                frozen_v12_tokens_by_view=frozen_v12_tokens_by_view,
                nominal_fm_px_by_view=nominal_fm_px_by_view,
                view_valid=view_valid,
                current_state_deg=state_deg,
            )
            self._round = _V170PolicyRound(
                full256=scene.astype(np.float32, copy=True),
                view_valid=views.copy(),
                current_state_deg=state_deg,
                current_state=state,
            )
        return measurement

    def observe_external_completion(
        self,
        *,
        completed_commands_deg: np.ndarray,
        measured_joint_path_deg: np.ndarray,
        round_id: str,
    ) -> None:
        commands = _finite(
            "completed_commands_deg",
            completed_commands_deg,
            (COMMAND_STEPS, STATE_DIMENSION),
        )
        path = _finite(
            "measured_joint_path_deg",
            measured_joint_path_deg,
            (COMMAND_STEPS + 1, STATE_DIMENSION),
        )
        canonical = commands / np.float32(CANONICAL_TO_DEGREES)
        if bool(((canonical < ACTION_LOWER) | (canonical > ACTION_UPPER)).any()):
            raise V170RuntimeError("completed policy chunk leaves canonical bounds")
        with self._lock:
            if self._round is None:
                raise V170RuntimeError(
                    "V17 policy completion requires a prepared endpoint"
                )
            if not np.allclose(
                path[0], self._round.current_state_deg, rtol=0.0, atol=1.0e-4
            ):
                raise V170RuntimeError(
                    "measured joint path root differs from prepared endpoint state"
                )
            # Queue FCPG before changing policy state.  A duplicate or FCPG
            # clock mismatch therefore leaves the Actor history unchanged.
            self.fcpg.observe_external_completion(
                completed_commands_deg=commands,
                measured_joint_path_deg=path,
                round_id=round_id,
            )
            if self._transition < 3:
                self._history[self._transition] = canonical
                self._transition += 1
                self._round = None
            else:
                self._clear_policy_window()
            self._external_completions += 1

    @torch.inference_mode()
    def propose_actor_shadow(
        self,
        *,
        current_state_deg: np.ndarray,
        desired_joint_goal_deg: np.ndarray,
    ) -> V170ActorShadowResult:
        with self._lock:
            prepared = self._require_round(current_state_deg)
            _goal_deg, goal = self._canonical_state(
                "desired_joint_goal_deg", desired_joint_goal_deg
            )
            if self._fixed_goal is None:
                self._fixed_goal = goal.copy()
            elif not np.array_equal(goal, self._fixed_goal):
                raise V170RuntimeError(
                    "V17 Actor goal must remain fixed for the H0-H3 window"
                )
            valid = np.arange(3) < self._transition
            future = ~valid[:, None, None]
            if not bool(
                np.all(
                    self._history[
                        np.broadcast_to(future, self._history.shape)
                    ]
                    == 0.0
                )
            ):
                raise V170RuntimeError("future Actor history must remain exact zero")
            output = self.actor(
                current_full256=torch.from_numpy(prepared.full256[None]).to(
                    self._device
                ),
                view_valid=torch.from_numpy(prepared.view_valid[None]).to(
                    self._device
                ),
                current_state=torch.from_numpy(prepared.current_state[None]).to(
                    self._device
                ),
                past_completed_commands=torch.from_numpy(
                    self._past_commands(prepared.current_state)[None]
                ).to(self._device),
                completed_command_history=torch.from_numpy(
                    self._history[None]
                ).to(self._device),
                history_valid=torch.from_numpy(valid[None]).to(self._device),
                transition=torch.tensor(
                    [self._transition], dtype=torch.int64, device=self._device
                ),
                desired_joint_goal=torch.from_numpy(goal[None]).to(self._device),
                baseline_kind="repeat",
                baseline_joint_goal=None,
                park_mask=None,
            )
            action = output.action[0].float().cpu().numpy()
            baseline = output.baseline[0].float().cpu().numpy()
            residual = output.residual[0].float().cpu().numpy()
            for name, value in (
                ("Actor action", action),
                ("Actor baseline", baseline),
                ("Actor residual", residual),
            ):
                if value.shape != (COMMAND_STEPS, STATE_DIMENSION) or not bool(
                    np.isfinite(value).all()
                ):
                    raise V170RuntimeError(f"{name} is not finite [5,6]")
            self._actor_calls += 1
            return V170ActorShadowResult(
                transition=f"H{self._transition}",
                action_deg=_readonly(action * np.float32(CANONICAL_TO_DEGREES)),
                baseline_deg=_readonly(
                    baseline * np.float32(CANONICAL_TO_DEGREES)
                ),
                residual_deg=_readonly(
                    residual * np.float32(CANONICAL_TO_DEGREES)
                ),
                history_valid=tuple(bool(item) for item in valid),
            )

    @torch.inference_mode()
    def predict_candidate_consequences(
        self,
        *,
        current_state_deg: np.ndarray,
        candidate_commands_deg: np.ndarray,
    ) -> V170CandidateConsequences:
        with self._lock:
            prepared = self._require_round(current_state_deg)
            candidates = np.asarray(candidate_commands_deg, dtype=np.float32)
            if (
                candidates.ndim != 3
                or candidates.shape[1:] != (COMMAND_STEPS, STATE_DIMENSION)
                or len(candidates) == 0
                or not bool(np.isfinite(candidates).all())
            ):
                raise V170RuntimeError("candidate_commands_deg must be finite [N,5,6]")
            canonical = candidates / np.float32(CANONICAL_TO_DEGREES)
            if bool(((canonical < ACTION_LOWER) | (canonical > ACTION_UPPER)).any()):
                raise V170RuntimeError("candidate bank leaves canonical joint bounds")
            output = self.lewm(
                current_state=torch.from_numpy(prepared.current_state[None]).to(
                    self._device
                ),
                past_completed_commands=torch.from_numpy(
                    self._past_commands(prepared.current_state)[None]
                ).to(self._device),
                candidate_commands=torch.from_numpy(canonical[None]).to(
                    self._device
                ),
            )
            predicted = output.predicted_state_path[0].float().cpu().numpy()
            baseline = output.baseline_state_path[0].float().cpu().numpy()
            residual = output.residual_state_path[0].float().cpu().numpy()
            expected_path = (len(candidates), COMMAND_STEPS + 1, STATE_DIMENSION)
            if predicted.shape != expected_path or baseline.shape != expected_path:
                raise V170RuntimeError("V17 LeWM candidate path shape drifted")
            if residual.shape != (
                len(candidates),
                COMMAND_STEPS,
                STATE_DIMENSION,
            ):
                raise V170RuntimeError("V17 LeWM residual path shape drifted")
            if not all(
                bool(np.isfinite(value).all())
                for value in (predicted, baseline, residual)
            ):
                raise V170RuntimeError("V17 LeWM candidate output is non-finite")
            root = np.broadcast_to(
                prepared.current_state,
                (len(candidates), STATE_DIMENSION),
            )
            if not np.allclose(predicted[:, 0], root, rtol=0.0, atol=1.0e-6):
                raise V170RuntimeError("V17 LeWM candidate q0 changed")
            self._lewm_calls += 1
            return V170CandidateConsequences(
                predicted_state_path_deg=_readonly(
                    predicted * np.float32(CANONICAL_TO_DEGREES)
                ),
                baseline_state_path_deg=_readonly(
                    baseline * np.float32(CANONICAL_TO_DEGREES)
                ),
                residual_state_path_deg=_readonly(
                    residual * np.float32(CANONICAL_TO_DEGREES)
                ),
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            valid = np.arange(3) < self._transition
            return {
                "format": "so101-prod-v17.0-plain-shadow-stack-v1",
                "mode": "shadow",
                "decision_influence": False,
                "commands_sent": 0,
                "candidate_bank_mutated": False,
                "candidate_selection_present": False,
                "policy_window": self._policy_window,
                "actor_transition": f"H{self._transition}",
                "history_valid": valid.tolist(),
                "future_history_exact_zero": bool(
                    np.all(self._history[~valid] == 0.0)
                ),
                "fixed_goal_present": self._fixed_goal is not None,
                "prepared_endpoint": self._round is not None,
                "actor_calls": self._actor_calls,
                "lewm_calls": self._lewm_calls,
                "external_completions_observed": self._external_completions,
                "fcpg": self.fcpg.status(),
                "model_parameters": {
                    "fcpg": self.fcpg.parameter_count,
                    "lewm": self.lewm.parameter_count,
                    "actor": self.actor.parameter_count,
                },
            }


class V170LiveStack(V170PlainShadowStack):
    """Plain-checkpoint V17 core for one externally supervised planner.

    Candidate mode allows the caller to insert the single Actor proposal into
    its existing bank.  Shadow mode runs the exact same FCPG, Actor and LeWM
    paths but marks the proposal as having no decision influence.  This object
    deliberately has no actuator or candidate-selection method; an external
    controller remains solely responsible for ranking, safety checks, command
    transport and readback.

    Model time advances only through :meth:`commit_completed_chunk`.  That
    method requires one complete five-command request, byte-for-value equal
    sent commands, and the initial state plus five finite measured readbacks.
    Calling the shadow completion API directly is rejected so a live caller
    cannot bypass this transport contract.
    """

    def __init__(
        self,
        *,
        fcpg: V170ShadowFCPGAdapter,
        lewm: StateFKLeWM,
        actor: FullHistoryFiveCommandActor,
        mode: str = "candidate",
        device: str | torch.device = "cpu",
    ) -> None:
        if mode not in {"shadow", "candidate"}:
            raise V170RuntimeError("V17 live-core mode must be shadow or candidate")
        super().__init__(fcpg=fcpg, lewm=lewm, actor=actor, device=device)
        self._mode = mode
        self._prepared_round_id: str | None = None
        self._fixed_goal_deg: np.ndarray | None = None
        self._fixed_goal_source: dict[str, Any] | None = None
        self._proposal_calls = 0
        self._confirmed_chunks = 0
        self._last_proposal: dict[str, Any] | None = None
        self._last_completion: dict[str, Any] | None = None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def decision_influence(self) -> bool:
        return self._mode == "candidate"

    @property
    def commands_sent(self) -> int:
        if self._mode == "shadow":
            return 0
        return self._confirmed_chunks * COMMAND_STEPS

    @classmethod
    def from_final_checkpoints(
        cls,
        *,
        fcpg_checkpoint: str | Path,
        lewm_checkpoint: str | Path,
        actor_checkpoint: str | Path,
        mode: str = "candidate",
        device: str | torch.device = "cpu",
    ) -> "V170LiveStack":
        """Strict-load the three ordinary V17 final checkpoint files."""

        resolved = torch.device(device)
        fcpg = _load_runtime_fcpg(fcpg_checkpoint, device=resolved)
        lewm = load_v170_final_model(lewm_checkpoint, stage="lewm", device=resolved)
        actor = load_v170_final_model(actor_checkpoint, stage="actor", device=resolved)
        return cls(
            fcpg=V170ShadowFCPGAdapter(model=fcpg, device=resolved),
            lewm=lewm,  # type: ignore[arg-type]
            actor=actor,  # type: ignore[arg-type]
            mode=mode,
            device=resolved,
        )

    def _clear_policy_window(self) -> None:
        super()._clear_policy_window()
        self._prepared_round_id = None
        self._fixed_goal_deg = None
        self._fixed_goal_source = None
        self._last_proposal = None

    def reset_window(self, reason: str) -> dict[str, Any]:
        result = super().reset_window(reason)
        with self._lock:
            self._last_completion = None
            return {
                **result,
                "mode": self.mode,
                "decision_influence": self.decision_influence,
            }

    def prepare_endpoint(
        self,
        *,
        rgb_by_view: Mapping[str, np.ndarray],
        frozen_v12_tokens_by_view: Mapping[str, np.ndarray],
        nominal_fm_px_by_view: Mapping[str, np.ndarray],
        view_valid: Mapping[str, bool],
        current_state_deg: np.ndarray,
        round_id: str,
    ) -> V170ShadowEndpoint:
        """Bind one settled endpoint and its immutable external round id."""

        if not isinstance(round_id, str) or not round_id.strip():
            raise V170RuntimeError("live round_id must be non-empty")
        with self._lock:
            measurement = super().prepare_endpoint(
                rgb_by_view=rgb_by_view,
                frozen_v12_tokens_by_view=frozen_v12_tokens_by_view,
                nominal_fm_px_by_view=nominal_fm_px_by_view,
                view_valid=view_valid,
                current_state_deg=current_state_deg,
            )
            self._prepared_round_id = round_id
            self._last_proposal = None
        return V170ShadowEndpoint(
            endpoint=measurement.endpoint,
            fm_px=measurement.fm_px,
            midpoint_px=measurement.midpoint_px,
            valid_fm=measurement.valid_fm,
            occlusion_probability=measurement.occlusion_probability,
            transition_consumed=measurement.transition_consumed,
            window_completed=measurement.window_completed,
            commands_sent=0,
            decision_influence=self.decision_influence,
            probability_semantics=measurement.probability_semantics,
            projection_invalid_probability=measurement.projection_invalid_probability,
        )

    def observe_external_completion(
        self,
        *,
        completed_commands_deg: np.ndarray,
        measured_joint_path_deg: np.ndarray,
        round_id: str,
    ) -> None:
        del completed_commands_deg, measured_joint_path_deg, round_id
        raise V170RuntimeError(
            "live V17 time advances only through commit_completed_chunk"
        )

    def _validated_planner_bank(
        self,
        *,
        baseline_prefixes_deg: np.ndarray,
        candidate_ik_safe: np.ndarray,
        selected_phase_distance: np.ndarray,
        horizon_raw: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(horizon_raw, bool) or not isinstance(horizon_raw, int):
            raise V170RuntimeError("planner horizon must be an integer")
        if horizon_raw < WINDOW_COMMANDS:
            raise V170RuntimeError(
                f"planner horizon must contain at least {WINDOW_COMMANDS} commands"
            )
        prefixes = np.asarray(baseline_prefixes_deg, dtype=np.float32)
        safe_raw = np.asarray(candidate_ik_safe)
        distance = np.asarray(selected_phase_distance, dtype=np.float32)
        if (
            prefixes.ndim != 3
            or prefixes.shape[1:] != (horizon_raw, STATE_DIMENSION)
            or len(prefixes) == 0
            or not bool(np.isfinite(prefixes).all())
        ):
            raise V170RuntimeError(
                f"baseline_prefixes_deg must be finite [N,{horizon_raw},6]"
            )
        if safe_raw.shape != (len(prefixes),) or safe_raw.dtype != np.bool_:
            raise V170RuntimeError("candidate_ik_safe must be boolean [N]")
        if distance.shape != (len(prefixes),):
            raise V170RuntimeError("selected_phase_distance must be [N]")
        return prefixes, safe_raw.astype(bool, copy=False), distance

    def planner_proxy_goal(
        self,
        *,
        baseline_prefixes_deg: np.ndarray,
        candidate_ik_safe: np.ndarray,
        selected_phase_distance: np.ndarray,
        horizon_raw: int,
        phase: Any,
    ) -> np.ndarray:
        """Choose once at H0, then return the same H4 proxy goal through H3."""

        prefixes, safe, distance = self._validated_planner_bank(
            baseline_prefixes_deg=baseline_prefixes_deg,
            candidate_ik_safe=candidate_ik_safe,
            selected_phase_distance=selected_phase_distance,
            horizon_raw=horizon_raw,
        )
        with self._lock:
            if self._round is None or self._prepared_round_id is None:
                raise V170RuntimeError("planner proxy goal needs a prepared live endpoint")
            if self._fixed_goal is None:
                if self._transition != 0:
                    raise V170RuntimeError("planner proxy goal is absent after H0")
                eligible = np.flatnonzero(safe & np.isfinite(distance))
                if len(eligible) == 0:
                    raise V170RuntimeError("no safe planner proxy goal exists")
                index = int(eligible[np.argmin(distance[eligible])])
                goal_deg, goal = self._canonical_state(
                    "planner_proxy_goal_deg",
                    prefixes[index, PROXY_GOAL_COMMAND_INDEX],
                )
                self._fixed_goal = goal
                self._fixed_goal_deg = goal_deg.copy()
                phase_value = phase.value if hasattr(phase, "value") else str(phase)
                self._fixed_goal_source = {
                    "baseline_index_at_h0": index,
                    "phase_at_h0": phase_value,
                    "proxy_command_index": PROXY_GOAL_COMMAND_INDEX,
                    "window_commands": WINDOW_COMMANDS,
                    "planner_horizon_raw": horizon_raw,
                }
                return _readonly(goal_deg)
            if self._fixed_goal_deg is None:
                raise V170RuntimeError("fixed planner proxy goal degrees are absent")
            return _readonly(self._fixed_goal_deg)

    def action_contract_fixed_point_mask(
        self,
        *,
        current_state: np.ndarray,
        five_command_candidates: np.ndarray,
    ) -> np.ndarray:
        """Return rows already satisfying the Actor's canonical projector."""

        state = np.asarray(current_state, dtype=np.float32)
        candidates = np.asarray(five_command_candidates, dtype=np.float32)
        if (
            state.shape != (STATE_DIMENSION,)
            or candidates.ndim != 3
            or candidates.shape[1:] != (COMMAND_STEPS, STATE_DIMENSION)
        ):
            raise V170RuntimeError("V17 action-contract inputs are misaligned")
        canonical_state = state / np.float32(CANONICAL_TO_DEGREES)
        canonical = candidates / np.float32(CANONICAL_TO_DEGREES)
        previous = np.concatenate(
            (
                np.broadcast_to(
                    canonical_state,
                    (len(canonical), 1, STATE_DIMENSION),
                ),
                canonical[:, :-1],
            ),
            axis=1,
        )
        finite = np.isfinite(canonical).all(axis=(1, 2)) & np.isfinite(
            canonical_state
        ).all()
        bounded = (
            (canonical >= ACTION_LOWER[None, None])
            & (canonical <= ACTION_UPPER[None, None])
        ).all(axis=(1, 2))
        stepped = (
            np.abs(canonical - previous)
            <= ACTION_STEP_CAP[None, None] + np.float32(1.0e-6)
        ).all(axis=(1, 2))
        return np.asarray(finite & bounded & stepped, dtype=bool)

    def propose_actor_prefix(
        self,
        *,
        current_state: np.ndarray,
        baseline_prefixes: np.ndarray,
        candidate_ik_safe: np.ndarray,
        selected_phase_distance: np.ndarray,
        horizon_raw: int,
        phase: Any,
    ) -> dict[str, Any]:
        """Return exactly one five-command Actor chunk followed by state hold."""

        with self._lock:
            prepared = self._require_round(current_state)
            prefixes, safe, distance = self._validated_planner_bank(
                baseline_prefixes_deg=baseline_prefixes,
                candidate_ik_safe=candidate_ik_safe,
                selected_phase_distance=selected_phase_distance,
                horizon_raw=horizon_raw,
            )
            if (
                self._transition == 0
                and self._fixed_goal is None
                and not bool((safe & np.isfinite(distance)).any())
            ):
                self._last_proposal = {
                    "round_id": self._prepared_round_id,
                    "generated": False,
                    "reason": "no_safe_planner_proxy_goal",
                    "transition": "H0",
                }
                return {
                    "prefixes": _readonly(
                        np.empty(
                            (0, horizon_raw, STATE_DIMENSION), dtype=np.float32
                        )
                    ),
                    "metadata": [],
                    "decision_influence": self.decision_influence,
                    "commands_sent": 0,
                }
            goal_deg = self.planner_proxy_goal(
                baseline_prefixes_deg=prefixes,
                candidate_ik_safe=safe,
                selected_phase_distance=distance,
                horizon_raw=horizon_raw,
                phase=phase,
            )
            actor_output = V170PlainShadowStack.propose_actor_shadow(
                self,
                current_state_deg=prepared.current_state_deg,
                desired_joint_goal_deg=goal_deg,
            )
            chunk = np.asarray(actor_output.action_deg, dtype=np.float32)
            fixed = self.action_contract_fixed_point_mask(
                current_state=prepared.current_state_deg,
                five_command_candidates=chunk[None],
            )
            if fixed.shape != (1,) or not bool(fixed[0]):
                raise V170RuntimeError(
                    "V17 Actor proposal is not an action-projector fixed point"
                )
            prefix = np.repeat(chunk[-1:], horizon_raw, axis=0)
            prefix[:COMMAND_STEPS] = chunk
            phase_value = phase.value if hasattr(phase, "value") else str(phase)
            metadata = {
                "source": "v17_0_full_history_actor",
                "variant": "five_command_then_hold",
                "actor_chunk_steps": COMMAND_STEPS,
                "phase": phase_value,
                "transition": actor_output.transition,
                "history_valid": list(actor_output.history_valid),
                "goal_fixed_window": True,
                "goal_source": dict(self._fixed_goal_source or {}),
                "lewm_scored": True,
                "mode": self.mode,
                "decision_influence": self.decision_influence,
                "commands_sent": 0,
            }
            self._proposal_calls += 1
            self._last_proposal = {
                "round_id": self._prepared_round_id,
                "transition": actor_output.transition,
                "five_command_deg": chunk.tolist(),
                "planner_horizon_raw": horizon_raw,
            }
            return {
                "prefixes": _readonly(prefix[None]),
                "metadata": [metadata],
                "decision_influence": self.decision_influence,
                "commands_sent": 0,
            }

    def predict_candidate_states(
        self,
        *,
        current_state: np.ndarray,
        five_command_candidates: np.ndarray,
    ) -> np.ndarray:
        """Return V17 LeWM q1--q5 for an unchanged five-command bank."""

        result = self.predict_candidate_consequences(
            current_state_deg=current_state,
            candidate_commands_deg=five_command_candidates,
        )
        return _readonly(result.predicted_state_path_deg[:, 1:])

    def commit_completed_chunk(
        self,
        *,
        requested_commands_deg: np.ndarray,
        sent_commands_deg: np.ndarray,
        measured_joint_path_deg: np.ndarray,
        round_id: str,
    ) -> dict[str, Any]:
        """Atomically advance both histories after complete confirmed transport."""

        if self._mode == "shadow":
            raise V170RuntimeError("shadow mode cannot commit physical transport")
        requested_raw = np.asarray(requested_commands_deg)
        sent_raw = np.asarray(sent_commands_deg)
        requested = _finite(
            "requested_commands_deg",
            requested_raw,
            (COMMAND_STEPS, STATE_DIMENSION),
        )
        sent = _finite(
            "sent_commands_deg",
            sent_raw,
            (COMMAND_STEPS, STATE_DIMENSION),
        )
        path = _finite(
            "measured_joint_path_deg",
            measured_joint_path_deg,
            (COMMAND_STEPS + 1, STATE_DIMENSION),
        )
        if not np.array_equal(requested_raw, sent_raw):
            raise V170RuntimeError("sent commands differ from requested commands")
        if not isinstance(round_id, str) or not round_id:
            raise V170RuntimeError("completion round_id must be non-empty")
        with self._lock:
            if self._prepared_round_id is None or self._round is None:
                raise V170RuntimeError("completion requires a prepared live endpoint")
            if round_id != self._prepared_round_id:
                raise V170RuntimeError("completion round differs from prepared live round")
            before = f"H{self._transition}"
            V170PlainShadowStack.observe_external_completion(
                self,
                completed_commands_deg=sent,
                measured_joint_path_deg=path,
                round_id=round_id,
            )
            self._prepared_round_id = None
            self._confirmed_chunks += 1
            completion = {
                "round_id": round_id,
                "before": before,
                "actor_after": f"H{self._transition}",
                "fcpg_transition_pending": True,
                "confirmed_command_count": COMMAND_STEPS,
            }
            self._last_completion = dict(completion)
            return completion

    def status(self) -> dict[str, Any]:
        with self._lock:
            valid = np.arange(3) < self._transition
            return {
                "format": "so101-prod-v17.0-plain-live-core-v1",
                "mode": self.mode,
                "decision_influence": self.decision_influence,
                "commands_sent": self.commands_sent,
                "candidate_bank_mutated": False,
                "actuator_surface_present": False,
                "candidate_selection_present": False,
                "policy_window": self._policy_window,
                "actor_transition": f"H{self._transition}",
                "history_valid": valid.tolist(),
                "future_history_exact_zero": bool(
                    np.all(self._history[~valid] == 0.0)
                ),
                "fixed_goal_present": self._fixed_goal is not None,
                "fixed_goal_source": (
                    None
                    if self._fixed_goal_source is None
                    else dict(self._fixed_goal_source)
                ),
                "prepared_round_id": self._prepared_round_id,
                "actor_calls": self._actor_calls,
                "proposal_calls": self._proposal_calls,
                "lewm_calls": self._lewm_calls,
                "confirmed_chunks": self._confirmed_chunks,
                "confirmed_commands": self._confirmed_chunks * COMMAND_STEPS,
                "last_proposal": self._last_proposal,
                "last_completion": self._last_completion,
                "fcpg": self.fcpg.status(),
                "model_parameters": {
                    "fcpg": self.fcpg.parameter_count,
                    "lewm": self.lewm.parameter_count,
                    "actor": self.actor.parameter_count,
                },
            }


class V170V168ShadowStack:
    """Run V17 FCPG and frozen V16.8 policy models without control influence."""

    mode = "shadow"
    decision_influence = False

    def __init__(self, *, fcpg: V170ShadowFCPGAdapter, foundation: Any, actor: Any) -> None:
        if getattr(actor, "mode", None) != "shadow" or bool(
            getattr(actor, "decision_influence", True)
        ):
            raise V170RuntimeError("V16.8 Actor must already be frozen in shadow mode")
        for name in ("action_contract_fixed_point_mask", "predict_candidate_states"):
            if not callable(getattr(foundation, name, None)):
                raise V170RuntimeError(f"V16.8 foundation omits {name}")
        for name in ("prepare_round", "propose_actor_prefix", "record_shadow_lewm"):
            if not callable(getattr(actor, name, None)):
                raise V170RuntimeError(f"V16.8 Actor overlay omits {name}")
        self.fcpg = fcpg
        self.foundation = foundation
        self.actor = actor

    def prepare_endpoint(
        self,
        *,
        rgb_by_view: Mapping[str, np.ndarray],
        frozen_v12_tokens_by_view: Mapping[str, np.ndarray],
        nominal_fm_px_by_view: Mapping[str, np.ndarray],
        view_valid: Mapping[str, bool],
        current_state_deg: np.ndarray,
    ) -> V170ShadowEndpoint:
        measurement = self.fcpg.observe_endpoint(
            rgb_by_view=rgb_by_view,
            frozen_v12_tokens_by_view=frozen_v12_tokens_by_view,
            nominal_fm_px_by_view=nominal_fm_px_by_view,
            view_valid=view_valid,
        )
        self.actor.prepare_round(
            full256_by_view=frozen_v12_tokens_by_view,
            view_valid=view_valid,
            current_state_deg=current_state_deg,
        )
        return measurement

    def run_policy_shadow(
        self,
        *,
        current_state_deg: np.ndarray,
        baseline_prefixes_deg: np.ndarray,
        candidate_ik_safe: np.ndarray,
        selected_phase_distance: np.ndarray,
        horizon_raw: int,
        phase: Any,
    ) -> V170ShadowPolicyResult:
        provided = self.actor.propose_actor_prefix(
            current_state=current_state_deg,
            baseline_prefixes=baseline_prefixes_deg,
            candidate_ik_safe=candidate_ik_safe,
            selected_phase_distance=selected_phase_distance,
            horizon_raw=horizon_raw,
            phase=phase,
        )
        prefixes = np.asarray(provided["prefixes"], dtype=np.float32)
        if prefixes.shape == (0, int(horizon_raw), STATE_DIMENSION):
            return V170ShadowPolicyResult(
                generated=False,
                actor_chunk_deg=_readonly(
                    np.empty((0, COMMAND_STEPS, STATE_DIMENSION), dtype=np.float32)
                ),
                predicted_states_deg=_readonly(
                    np.empty((0, COMMAND_STEPS, STATE_DIMENSION), dtype=np.float32)
                ),
                projector_fixed_point=True,
            )
        if prefixes.shape != (1, int(horizon_raw), STATE_DIMENSION):
            raise V170RuntimeError("shadow Actor must return zero or one prefix")
        chunks = prefixes[:, :COMMAND_STEPS]
        fixed = np.asarray(
            self.foundation.action_contract_fixed_point_mask(
                current_state=np.asarray(current_state_deg, dtype=np.float32),
                five_command_candidates=chunks,
            ),
            dtype=bool,
        )
        if fixed.shape != (1,) or not bool(fixed[0]):
            raise V170RuntimeError("shadow Actor proposal is not a projector fixed point")
        prediction = np.asarray(
            self.foundation.predict_candidate_states(
                current_state=np.asarray(current_state_deg, dtype=np.float32),
                five_command_candidates=chunks,
            ),
            dtype=np.float32,
        )
        if prediction.shape != (1, COMMAND_STEPS, STATE_DIMENSION) or not bool(
            np.isfinite(prediction).all()
        ):
            raise V170RuntimeError("shadow LeWM output must be finite [1,5,6]")
        self.actor.record_shadow_lewm(prediction)
        return V170ShadowPolicyResult(
            generated=True,
            actor_chunk_deg=_readonly(chunks),
            predicted_states_deg=_readonly(prediction),
            projector_fixed_point=True,
        )

    def status(self) -> dict[str, Any]:
        return {
            "format": "so101-prod-v17.0-v16.8-shadow-stack-v1",
            "mode": "shadow",
            "decision_influence": False,
            "commands_sent": 0,
            "candidate_bank_mutated": False,
            "fcpg": self.fcpg.status(),
            "actor_transition": f"H{int(getattr(self.actor, 'transition', 0))}",
            "lewm_ran": bool(
                getattr(self.actor, "status")().get("last_shadow_lewm")
                if callable(getattr(self.actor, "status", None))
                else False
            ),
        }


def assert_v170_shadow_bootstrap(*, execute_enabled: bool, mode: str) -> None:
    """Fail startup before I/O if a V17 shadow is paired with execution."""

    if mode != "shadow":
        raise V170RuntimeError("the current V17 adapter supports shadow mode only")
    if bool(execute_enabled):
        raise V170RuntimeError("V17 shadow bootstrap requires commands_sent=0")


__all__ = [
    "V170ActorShadowResult",
    "V170CandidateConsequences",
    "V170LiveStack",
    "V170PlainShadowStack",
    "V170RuntimeError",
    "V170ShadowEndpoint",
    "V170ShadowFCPGAdapter",
    "V170ShadowPolicyResult",
    "V170V168ShadowStack",
    "assert_v170_shadow_bootstrap",
]
