"""Receipt-bound, offline-only proposal adapter for Prod-V16.8.

This module loads exactly the frozen full/main/pass-8 V16.8 Actor and exposes
one deliberately small integration surface.  A caller starts an offline
planning window with one explicit planner-proxy joint goal, asks the Actor to
append one canonical-domain five-command proposal to an existing candidate
bank, and then explicitly records the chunk that an external replay says was
completed.  The adapter never ranks or selects a candidate and has no camera,
robot, serial, server, CEM, or LeWM surface.

The planner-proxy goal is fixed for the entire H0--H3 window.  It is not
claimed to be distributionally equivalent to the hindsight H4 goal used for
training.  Likewise, recording a completed chunk only advances causal replay
state; this module sends zero commands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch

from fcpg.prod_v16_2 import project_bounded_action_trajectory
from fcpg.prod_v16_5 import COMMAND_STEPS, PATCHES, STATE_DIM, TOKEN_DIM, VIEWS
from fcpg.prod_v16_6 import v166_state_dict_sha256
from fcpg.prod_v16_8 import FullHistoryFiveCommandActor


PROTOCOL_ID = "so101-prod-v16-8-offline-proposal-adapter-v1"
TRAINING_PROTOCOL_ID = "so101-prod-v16-8-train-source-held-ablation-v1"
TRAINING_RECEIPT_FORMAT = "so101-prod-v16-8-arm-receipt-v1"
TRAINING_CHECKPOINT_FORMAT = "so101-prod-v16-8-arm-checkpoint-v1"
REGISTERED_RECEIPT_SHA256 = (
    "5e847247e212787a7c90a9f957ed8b052198049387d3ef65198fe9df5781a6ec"
)
REGISTERED_CHECKPOINT_SHA256 = (
    "5a637e9a234284db9bd1a6fce0838a4f99700272344c8645ea4927e0a7319a5f"
)
REGISTERED_STATE_SHA256 = (
    "5d43daf136ae6e3c6daa32da3758f64867e59b8892989caa4472f6da7e917961"
)
REGISTERED_TRAIN_PAIR_SHA256 = (
    "5f0f86a60edbe2ed3701bb1c83432aa94d268ae880d50c65e17e75a6d5e8961b"
)
REGISTERED_TRAIN_PAIRS = 314_997
REGISTERED_COMPLETED_PASS = 8
REGISTERED_PARAMETER_COUNT = 288_672
HISTORY_CHUNKS = 3
ACTOR_SOURCE = "v16_8_full_main_pass8_actor"
COMPLETION_PROVENANCES = (
    "offline_factual",
    "offline_commands_sent_log",
)
CompletionProvenance = Literal[
    "offline_factual",
    "offline_commands_sent_log",
]

ACTION_CONTRACT = {
    "step_cap": [0.04, 0.04, 0.04, 0.04, 0.04, 0.08],
    "lower": [-1.0, -1.0, -1.0, -1.0, -1.0, 0.0],
    "upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}
MAIN_RECIPE = {
    "arm": "main",
    "fusion_mode": "query_residual",
    "history_mode": "full",
    "history": (
        "complete causal completed-command chunks with transition-exact prefix mask"
    ),
    "use_full_history": True,
    "residual_gate": True,
    "objective": (
        "normalized L1 action + 0.2 normalized L1 velocity + 1.0 repeat-regret"
    ),
    "objective_kind": "normalized_l1_with_repeat_regret",
    "repeat_regret_weight": 1.0,
    "dev_alpha_encoded": False,
}


class V168OfflineRuntimeContractError(RuntimeError):
    """The frozen artifact or offline causal-window contract drifted."""


_LOADER_TOKEN = object()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse_test_path(path: str | Path, *, label: str) -> Path:
    source = Path(path).expanduser()
    try:
        resolved = source.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise V168OfflineRuntimeContractError(
            f"{label} is not a readable file: {source}"
        ) from exc
    lowered = str(resolved).lower().replace("-", "_")
    components = {
        component.lower().replace("-", "_") for component in resolved.parts
    }
    if (
        "test_sealed" in lowered
        or "task_test" in lowered
        or "fish_box_test" in lowered
        or components.intersection(
            {"test", "tests", "test_sealed", "task_test", "fish_box_test"}
        )
    ):
        raise V168OfflineRuntimeContractError(
            f"TEST-like path is sealed from the offline adapter: {resolved}"
        )
    if not resolved.is_file():
        raise V168OfflineRuntimeContractError(
            f"{label} is not a readable file: {resolved}"
        )
    return resolved


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V168OfflineRuntimeContractError(f"cannot read {label}") from exc
    if not isinstance(value, Mapping):
        raise V168OfflineRuntimeContractError(f"{label} must contain an object")
    return value


def _contract_tensors(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "step_cap": torch.tensor(
            ACTION_CONTRACT["step_cap"], dtype=torch.float32, device=device
        ),
        "action_lower": torch.tensor(
            ACTION_CONTRACT["lower"], dtype=torch.float32, device=device
        ),
        "action_upper": torch.tensor(
            ACTION_CONTRACT["upper"], dtype=torch.float32, device=device
        ),
    }


def _tensor_sha256(value: np.ndarray) -> str:
    canonical = np.asarray(value, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _readonly_metadata(
    values: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Detach candidate metadata from caller mutation and freeze its top level."""

    snapshots: list[Mapping[str, Any]] = []
    for value in values:
        try:
            detached = copy.deepcopy(dict(value))
        except Exception as exc:
            raise V168OfflineRuntimeContractError(
                "candidate metadata could not be detached for offline audit"
            ) from exc
        snapshots.append(MappingProxyType(detached))
    return tuple(snapshots)


@dataclass(frozen=True)
class V168OfflineBinding:
    """Immutable identity and evidence boundary for the loaded Actor."""

    receipt_path: str
    receipt_sha256: str
    checkpoint_path: str
    training_milestone_path: str
    checkpoint_sha256: str
    state_sha256: str
    completed_pass: int
    parameters: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": PROTOCOL_ID,
            "training_protocol_id": TRAINING_PROTOCOL_ID,
            "receipt": {
                "path": self.receipt_path,
                "sha256": self.receipt_sha256,
            },
            "checkpoint": {
                "path": self.checkpoint_path,
                "training_milestone_path": self.training_milestone_path,
                "sha256": self.checkpoint_sha256,
                "state_sha256": self.state_sha256,
                "completed_pass": self.completed_pass,
            },
            "parameters": self.parameters,
            "actor_scope": "one_shared_H0_H1_H2_H3_actor",
            "candidate_domain": "canonical_joint_action",
            "candidate_shape": [COMMAND_STEPS, STATE_DIM],
            "goal_provenance": "planner_proxy_fixed_window",
            "training_h4_equivalence_verified": False,
            "formal_dev_interpretation": "descriptive_locked_model_transfer",
            "visual_scene_identity_dependence_established": False,
            "history_order_dependence_established": False,
            "ranking_authority": "external_frozen_lewm_or_registered_offline_ranker",
            "lewm_loaded": False,
            "cem_loaded": False,
            "performance_gate_present": False,
            "candidate_selection_present": False,
            "synthetic_or_offline_replay_only": True,
            "commands_sent": 0,
            "camera_opened": False,
            "hardware_opened": False,
            "task_test_rows_opened": 0,
            "deployment_authorized": False,
        }


@dataclass(frozen=True)
class V168ProposalBank:
    """One unranked Actor proposal appended to an existing offline bank."""

    candidates: np.ndarray
    metadata: tuple[Mapping[str, Any], ...]
    actor_proposal_index: int
    transition: int
    audit: Mapping[str, Any]


def load_registered_pass8_actor(
    receipt_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[FullHistoryFiveCommandActor, V168OfflineBinding]:
    """Load the exact pass-8 Actor by immutable content identity.

    The receipt retains the original training-host checkpoint path as
    provenance.  A byte-identical checkpoint may be relocated for offline
    benchmark or reviewed runtime use, so only its basename is compared to the
    milestone path; receipt/checkpoint/state SHA-256 identities remain exact.
    """

    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise V168OfflineRuntimeContractError(
            "V16.8 offline Actor device must be cpu or cuda"
        )
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise V168OfflineRuntimeContractError(
            "requested CUDA device is unavailable"
        )
    receipt_file = _refuse_test_path(receipt_path, label="V16.8 training receipt")
    checkpoint_file = _refuse_test_path(
        checkpoint_path, label="V16.8 pass8 checkpoint"
    )
    receipt_sha = _sha256_file(receipt_file)
    checkpoint_sha = _sha256_file(checkpoint_file)
    if receipt_sha != REGISTERED_RECEIPT_SHA256:
        raise V168OfflineRuntimeContractError(
            "offline adapter received the wrong V16.8 training receipt"
        )
    if checkpoint_sha != REGISTERED_CHECKPOINT_SHA256:
        raise V168OfflineRuntimeContractError(
            "offline adapter received the wrong V16.8 pass8 checkpoint"
        )
    receipt = _read_json(receipt_file, label="V16.8 training receipt")
    fixed_receipt = {
        "format": TRAINING_RECEIPT_FORMAT,
        "protocol_id": TRAINING_PROTOCOL_ID,
        "mode": "full",
        "stage": "arm",
        "arm": "main",
        "actor_scope": "shared_H0_H1_H2_H3",
        "status": "TRAINED_COMPLETE",
        "interpretation": (
            "TRAINED_COMPLETE_WITH_MILESTONE_CHECKPOINTS_NO_HELD_CURVES"
        ),
        "baseline": "repeat",
        "formal_dev_rows_opened": 0,
        "task_test_rows_opened": 0,
        "hardware_opened": False,
        "deployment_authorized": False,
        "winner_selected": False,
        "performance_gate_present": False,
        "early_stop_present": False,
        "formal_dev_or_test_metrics": 0,
    }
    if any(receipt.get(name) != expected for name, expected in fixed_receipt.items()):
        raise V168OfflineRuntimeContractError(
            "frozen V16.8 receipt identity or safety boundary drifted"
        )
    if receipt.get("recipe") != MAIN_RECIPE:
        raise V168OfflineRuntimeContractError("frozen V16.8 main recipe drifted")
    population = receipt.get("training_population")
    if not isinstance(population, Mapping) or (
        int(population.get("pairs", -1)) != REGISTERED_TRAIN_PAIRS
        or population.get("pair_sha256") != REGISTERED_TRAIN_PAIR_SHA256
        or int(population.get("formal_dev_pairs", -1)) != 0
        or int(population.get("task_test_pairs", -1)) != 0
    ):
        raise V168OfflineRuntimeContractError(
            "frozen V16.8 training population drifted"
        )
    milestones = receipt.get("checkpoint_milestones")
    pass8 = milestones.get("pass8") if isinstance(milestones, Mapping) else None
    if not isinstance(pass8, Mapping) or (
        Path(str(pass8.get("path"))).name != checkpoint_file.name
        or pass8.get("sha256") != REGISTERED_CHECKPOINT_SHA256
        or pass8.get("state_sha256") != REGISTERED_STATE_SHA256
        or int(pass8.get("completed_pass", -1)) != REGISTERED_COMPLETED_PASS
    ):
        raise V168OfflineRuntimeContractError(
            "receipt does not bind the exact V16.8 pass8 checkpoint"
        )
    try:
        checkpoint = torch.load(
            checkpoint_file, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise V168OfflineRuntimeContractError(
            "cannot safely load the V16.8 pass8 checkpoint"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise V168OfflineRuntimeContractError(
            "V16.8 pass8 checkpoint must contain an object"
        )
    fixed_checkpoint = {
        "format": TRAINING_CHECKPOINT_FORMAT,
        "protocol_id": TRAINING_PROTOCOL_ID,
        "mode": "full",
        "stage": "arm",
        "arm": "main",
        "actor_scope": "shared_H0_H1_H2_H3",
        "completed_pass": REGISTERED_COMPLETED_PASS,
        "model_kwargs": {
            "dimension": TOKEN_DIM,
            "fusion_mode": "query_residual",
            "residual_gate": True,
            "history_mode": "full",
        },
        "baseline": "repeat",
        "baseline_numeric_contract": (
            "Actor output.baseline must be bitwise equal to registered repeat"
        ),
        "transition_history_contract": (
            "history_valid[b,slot] iff slot < transition[b]"
        ),
        "action_contract": ACTION_CONTRACT,
        "population_pair_sha256": REGISTERED_TRAIN_PAIR_SHA256,
        "recipe": MAIN_RECIPE,
        "formal_dev_rows_opened": 0,
        "task_test_rows_opened": 0,
        "hardware_opened": False,
        "deployment_authorized": False,
        "optimizer_state_present": False,
        "automatic_resume_authorized": False,
        "lewm_model_present": False,
        "runtime_actor_count": 1,
        "runtime_actor_bank_present": False,
    }
    if any(
        checkpoint.get(name) != expected
        for name, expected in fixed_checkpoint.items()
    ):
        raise V168OfflineRuntimeContractError(
            "V16.8 pass8 checkpoint model or recipe identity drifted"
        )
    if checkpoint.get("state_sha256") != REGISTERED_STATE_SHA256:
        raise V168OfflineRuntimeContractError(
            "V16.8 pass8 checkpoint state identity drifted"
        )
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise V168OfflineRuntimeContractError(
            "V16.8 pass8 checkpoint has no state_dict"
        )
    cpu_contract = _contract_tensors(torch.device("cpu"))
    actor = FullHistoryFiveCommandActor(
        **cpu_contract,
        dimension=TOKEN_DIM,
        fusion_mode="query_residual",
        history_mode="full",
        residual_gate=True,
    )
    expected_state = actor.state_dict()
    if set(state) != set(expected_state):
        raise V168OfflineRuntimeContractError("V16.8 pass8 state keys drifted")
    for name, expected in expected_state.items():
        actual = state[name]
        if (
            not isinstance(actual, torch.Tensor)
            or actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not bool(torch.isfinite(actual).all())
        ):
            raise V168OfflineRuntimeContractError(
                f"V16.8 pass8 tensor {name} is invalid"
            )
    try:
        state_sha = v166_state_dict_sha256(state)
    except Exception as exc:
        raise V168OfflineRuntimeContractError(
            "V16.8 pass8 state hash could not be reproduced"
        ) from exc
    if state_sha != REGISTERED_STATE_SHA256:
        raise V168OfflineRuntimeContractError(
            "V16.8 pass8 state SHA does not reproduce"
        )
    result = actor.load_state_dict(dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise V168OfflineRuntimeContractError(
            "V16.8 pass8 failed strict state load"
        )
    actor.requires_grad_(False)
    actor = actor.to(resolved_device).eval()
    if any(parameter.requires_grad for parameter in actor.parameters()):
        raise V168OfflineRuntimeContractError("V16.8 offline Actor is not frozen")
    if sum(parameter.numel() for parameter in actor.parameters()) != REGISTERED_PARAMETER_COUNT:
        raise V168OfflineRuntimeContractError("V16.8 Actor parameter count drifted")
    if v166_state_dict_sha256(actor.state_dict()) != REGISTERED_STATE_SHA256:
        raise V168OfflineRuntimeContractError(
            "V16.8 Actor state changed after device load"
        )
    for name, expected in _contract_tensors(torch.device("cpu")).items():
        actual = getattr(actor, name, None)
        if (
            not isinstance(actual, torch.Tensor)
            or not torch.equal(actual.detach().cpu(), expected)
        ):
            raise V168OfflineRuntimeContractError(
                f"V16.8 Actor action buffer drifted: {name}"
            )
    binding = V168OfflineBinding(
        receipt_path=str(receipt_file),
        receipt_sha256=receipt_sha,
        checkpoint_path=str(checkpoint_file),
        training_milestone_path=str(pass8.get("path")),
        checkpoint_sha256=checkpoint_sha,
        state_sha256=state_sha,
        completed_pass=REGISTERED_COMPLETED_PASS,
        parameters=REGISTERED_PARAMETER_COUNT,
    )
    return actor, binding


# Compatibility for earlier internal callers.  New benchmark/runtime code must
# use the public content-addressed loader above.
_load_registered_actor = load_registered_pass8_actor


class V168OfflineProposalAdapter:
    """Stateful H0--H3 proposal source with no execution or ranking authority."""

    def __init__(
        self,
        *,
        actor: FullHistoryFiveCommandActor,
        binding: V168OfflineBinding,
        device: torch.device,
        _loader_token: object | None = None,
    ) -> None:
        if _loader_token is not _LOADER_TOKEN:
            raise V168OfflineRuntimeContractError(
                "construct V16.8 adapter only through from_registered_pass8"
            )
        self._actor = actor
        self._binding = binding
        self._device = torch.device(device)
        self._window_id: str | None = None
        self._goal: np.ndarray | None = None
        self._history = np.zeros(
            (HISTORY_CHUNKS, COMMAND_STEPS, STATE_DIM), dtype=np.float32
        )
        self._transition = 0
        self._proposal_generated = False
        self._proposal_current_state: np.ndarray | None = None
        self._proposal_candidate_bank: np.ndarray | None = None
        self._proposal_candidate_metadata: tuple[Mapping[str, Any], ...] | None = None
        self._proposal_candidate_bank_sha256: str | None = None
        self._completion_audit: list[dict[str, Any]] = []

    @classmethod
    def from_registered_pass8(
        cls,
        *,
        receipt_path: str | Path,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "V168OfflineProposalAdapter":
        actor, binding = load_registered_pass8_actor(
            receipt_path, checkpoint_path, device=device
        )
        resolved_device = next(actor.parameters()).device
        return cls(
            actor=actor,
            binding=binding,
            device=resolved_device,
            _loader_token=_LOADER_TOKEN,
        )

    @property
    def receipt(self) -> dict[str, Any]:
        return self._binding.as_dict()

    @property
    def transition(self) -> int:
        return int(self._transition)

    @property
    def window_active(self) -> bool:
        return self._goal is not None and self._transition < 4

    @property
    def fixed_planner_proxy_goal(self) -> np.ndarray | None:
        return None if self._goal is None else _readonly_copy(self._goal)

    @property
    def completed_command_history(self) -> np.ndarray:
        return _readonly_copy(self._history)

    @property
    def history_valid(self) -> np.ndarray:
        valid = np.arange(HISTORY_CHUNKS, dtype=np.int64) < min(
            self._transition, HISTORY_CHUNKS
        )
        return _readonly_copy(valid)

    def begin_window(
        self,
        *,
        window_id: str,
        planner_proxy_goal: np.ndarray,
    ) -> Mapping[str, Any]:
        """Start one fixed-goal H0--H3 offline replay window."""

        if self.window_active:
            raise V168OfflineRuntimeContractError(
                "cannot replace an active V16.8 planner-proxy goal"
            )
        if not isinstance(window_id, str) or not window_id.strip():
            raise V168OfflineRuntimeContractError("window_id must be non-empty")
        goal = np.asarray(planner_proxy_goal, dtype=np.float32)
        lower = np.asarray(ACTION_CONTRACT["lower"], dtype=np.float32)
        upper = np.asarray(ACTION_CONTRACT["upper"], dtype=np.float32)
        if goal.shape != (STATE_DIM,) or not bool(np.isfinite(goal).all()):
            raise V168OfflineRuntimeContractError(
                "planner_proxy_goal must be finite canonical [6]"
            )
        if bool(((goal < lower) | (goal > upper)).any()):
            raise V168OfflineRuntimeContractError(
                "planner_proxy_goal lies outside the V16.8 action contract"
            )
        self._window_id = window_id.strip()
        self._goal = goal.copy()
        self._history.fill(0.0)
        self._transition = 0
        self._proposal_generated = False
        self._proposal_current_state = None
        self._proposal_candidate_bank = None
        self._proposal_candidate_metadata = None
        self._proposal_candidate_bank_sha256 = None
        self._completion_audit = []
        return self.window_audit()

    def window_audit(self) -> dict[str, Any]:
        valid = self.history_valid
        invalid = ~valid[:, None, None]
        invalid_zero = bool(
            np.all(
                self._history[
                    np.broadcast_to(invalid, self._history.shape)
                ]
                == 0.0
            )
        )
        return {
            "format": PROTOCOL_ID,
            "window_id": self._window_id,
            "active": self.window_active,
            "transition": int(self._transition),
            "transition_name": (
                f"H{self._transition}" if self._transition < 4 else "COMPLETE"
            ),
            "completed_chunks": int(self._transition),
            "model_history_shape": list(self._history.shape),
            "history_valid": valid.tolist(),
            "history_rule": "history_valid[slot] iff slot < transition",
            "future_history_exact_zero": invalid_zero,
            "history_sha256": _tensor_sha256(self._history),
            "goal_provenance": "planner_proxy_fixed_window",
            "goal_sha256": (
                None if self._goal is None else _tensor_sha256(self._goal)
            ),
            "training_h4_equivalence_verified": False,
            "proposal_generated_for_transition": bool(self._proposal_generated),
            "pending_candidate_bank_sha256": self._proposal_candidate_bank_sha256,
            "completion_audit": [dict(item) for item in self._completion_audit],
            "commands_sent": 0,
            "deployment_authorized": False,
        }

    def _require_window(self) -> None:
        if not self.window_active or self._goal is None or self._window_id is None:
            raise V168OfflineRuntimeContractError(
                "begin_window must bind a fixed planner-proxy goal before proposal"
            )
        if self._transition < 0 or self._transition > 3:
            raise V168OfflineRuntimeContractError(
                "V16.8 offline transition leaves H0-H3"
            )
        expected = np.arange(HISTORY_CHUNKS) < self._transition
        if not np.array_equal(self.history_valid, expected):
            raise V168OfflineRuntimeContractError(
                "V16.8 history-valid mask drifted"
            )
        invalid = ~expected[:, None, None]
        if bool(
            np.any(
                self._history[np.broadcast_to(invalid, self._history.shape)]
                != 0.0
            )
        ):
            raise V168OfflineRuntimeContractError(
                "future completed-command history is not exact zero"
            )

    def _condensed_past(self, current_state: np.ndarray) -> np.ndarray:
        horizon = self._transition
        if horizon == 0:
            return np.repeat(current_state[None], 2, axis=0).astype(np.float32)
        endpoints = self._history[:horizon, -1]
        if horizon == 1:
            return np.stack((current_state, endpoints[-1])).astype(np.float32)
        return endpoints[-2:].astype(np.float32, copy=True)

    def _validate_existing_candidate_bank(
        self,
        candidate_bank: np.ndarray,
        *,
        current_state: np.ndarray,
        state_tensor: torch.Tensor,
    ) -> dict[str, Any]:
        """Require every pre-existing candidate to be executable as supplied."""

        if not isinstance(candidate_bank, np.ndarray):
            raise V168OfflineRuntimeContractError(
                "existing candidate_bank must be an np.ndarray without implicit conversion"
            )
        if candidate_bank.dtype != np.dtype(np.float32):
            raise V168OfflineRuntimeContractError(
                "existing candidate_bank dtype must be exactly float32"
            )
        if candidate_bank.ndim != 3 or candidate_bank.shape[1:] != (
            COMMAND_STEPS,
            STATE_DIM,
        ) or len(candidate_bank) == 0:
            raise V168OfflineRuntimeContractError(
                "existing candidate_bank must be canonical [K>0,5,6]"
            )
        finite = np.isfinite(candidate_bank).all(axis=(1, 2))
        lower = np.asarray(ACTION_CONTRACT["lower"], dtype=np.float32)
        upper = np.asarray(ACTION_CONTRACT["upper"], dtype=np.float32)
        bounded = (
            (candidate_bank >= lower[None, None])
            & (candidate_bank <= upper[None, None])
        ).all(axis=(1, 2)) & finite
        previous = np.concatenate(
            (
                np.broadcast_to(
                    current_state,
                    (len(candidate_bank), 1, STATE_DIM),
                ),
                candidate_bank[:, :-1],
            ),
            axis=1,
        )
        cap = np.asarray(ACTION_CONTRACT["step_cap"], dtype=np.float32)
        step_legal = (
            np.abs(candidate_bank - previous) <= cap[None, None] + 1.0e-6
        ).all(axis=(1, 2)) & finite
        roots = state_tensor.expand(len(candidate_bank), -1)
        commands = torch.from_numpy(np.ascontiguousarray(candidate_bank)).to(
            self._device
        )
        projected = project_bounded_action_trajectory(
            roots,
            commands,
            step_cap=self._actor.step_cap,
            action_lower=self._actor.action_lower,
            action_upper=self._actor.action_upper,
        )
        projector_fixed = (
            torch.isclose(commands, projected, rtol=0.0, atol=1.0e-6)
            .all(dim=(1, 2))
            .cpu()
            .numpy()
        ) & finite
        projected_twice = project_bounded_action_trajectory(
            roots,
            projected,
            step_cap=self._actor.step_cap,
            action_lower=self._actor.action_lower,
            action_upper=self._actor.action_upper,
        )
        projector_idempotent = (
            torch.isclose(projected, projected_twice, rtol=0.0, atol=1.0e-6)
            .all(dim=(1, 2))
            .cpu()
            .numpy()
        ) & finite
        checks = {
            "finite": finite,
            "absolute_bounds": bounded,
            "step_cap_legal": step_legal,
            "projector_fixed": projector_fixed,
            "projector_idempotent": projector_idempotent,
        }
        failed = {
            name: np.flatnonzero(~value).astype(int).tolist()
            for name, value in checks.items()
            if not bool(value.all())
        }
        if failed:
            raise V168OfflineRuntimeContractError(
                f"existing candidate_bank violates the canonical action contract: {failed}"
            )
        return {
            "existing_bank_all_finite": True,
            "existing_bank_absolute_bounds": True,
            "existing_bank_step_cap_legal": True,
            "existing_bank_projector_fixed": True,
            "existing_bank_projector_idempotent": True,
        }

    @torch.inference_mode()
    def append_actor_proposal(
        self,
        *,
        candidate_bank: np.ndarray,
        candidate_metadata: Sequence[Mapping[str, Any]],
        current_full256: np.ndarray,
        view_valid: np.ndarray,
        current_state: np.ndarray,
    ) -> V168ProposalBank:
        """Append exactly one unranked canonical ``[5,6]`` Actor proposal."""

        self._require_window()
        if self._proposal_generated:
            raise V168OfflineRuntimeContractError(
                "V16.8 already proposed once for this transition"
            )
        if not isinstance(candidate_bank, np.ndarray):
            raise V168OfflineRuntimeContractError(
                "existing candidate_bank must be an np.ndarray without implicit conversion"
            )
        if candidate_bank.ndim != 3 or candidate_bank.shape[1:] != (
            COMMAND_STEPS,
            STATE_DIM,
        ) or len(candidate_bank) == 0:
            raise V168OfflineRuntimeContractError(
                "existing candidate_bank must be canonical [K>0,5,6]"
            )
        try:
            metadata = tuple(candidate_metadata)
        except TypeError as exc:
            raise V168OfflineRuntimeContractError(
                "candidate metadata must be a sequence aligned with the existing bank"
            ) from exc
        if len(metadata) != len(candidate_bank) or any(
            not isinstance(item, Mapping) for item in metadata
        ):
            raise V168OfflineRuntimeContractError(
                "candidate metadata must align one-to-one with the existing bank"
            )
        scene = np.asarray(current_full256, dtype=np.float32)
        views = np.asarray(view_valid)
        state = np.asarray(current_state, dtype=np.float32)
        if scene.shape != (VIEWS, PATCHES, TOKEN_DIM) or not bool(
            np.isfinite(scene).all()
        ):
            raise V168OfflineRuntimeContractError(
                "current_full256 must be finite [2,256,192]"
            )
        if views.shape != (VIEWS,) or views.dtype != np.bool_:
            raise V168OfflineRuntimeContractError("view_valid must be boolean [2]")
        if not bool(views.any()):
            raise V168OfflineRuntimeContractError(
                "V16.8 Actor requires at least one valid view"
            )
        if bool(np.any(scene[~views] != 0.0)):
            raise V168OfflineRuntimeContractError(
                "invalid view tokens must be exact zero"
            )
        lower = np.asarray(ACTION_CONTRACT["lower"], dtype=np.float32)
        upper = np.asarray(ACTION_CONTRACT["upper"], dtype=np.float32)
        if state.shape != (STATE_DIM,) or not bool(np.isfinite(state).all()):
            raise V168OfflineRuntimeContractError(
                "current_state must be finite canonical [6]"
            )
        if bool(((state < lower) | (state > upper)).any()):
            raise V168OfflineRuntimeContractError(
                "current_state lies outside the V16.8 action contract"
            )
        state_tensor = torch.from_numpy(state[None]).to(self._device)
        existing_legality = self._validate_existing_candidate_bank(
            candidate_bank,
            current_state=state,
            state_tensor=state_tensor,
        )
        bank = candidate_bank
        history = self._history.copy()
        valid = np.arange(HISTORY_CHUNKS) < self._transition
        past = self._condensed_past(state)
        output = self._actor(
            current_full256=torch.from_numpy(scene[None]).to(self._device),
            view_valid=torch.from_numpy(views[None]).to(self._device),
            current_state=state_tensor,
            past_completed_commands=torch.from_numpy(past[None]).to(self._device),
            completed_command_history=torch.from_numpy(history[None]).to(
                self._device
            ),
            history_valid=torch.from_numpy(valid[None]).to(self._device),
            transition=torch.tensor(
                [self._transition], dtype=torch.int64, device=self._device
            ),
            desired_joint_goal=torch.from_numpy(self._goal[None]).to(self._device),
            baseline_kind="repeat",
            baseline_joint_goal=None,
        )
        raw = output.action.detach().float()
        if raw.shape != (1, COMMAND_STEPS, STATE_DIM) or not bool(
            torch.isfinite(raw).all()
        ):
            raise V168OfflineRuntimeContractError(
                "V16.8 Actor proposal must be finite [1,5,6]"
            )
        projected = project_bounded_action_trajectory(
            state_tensor,
            raw,
            step_cap=self._actor.step_cap,
            action_lower=self._actor.action_lower,
            action_upper=self._actor.action_upper,
        )
        if not bool(torch.isclose(raw, projected, rtol=0.0, atol=1.0e-6).all()):
            raise V168OfflineRuntimeContractError(
                "V16.8 Actor output is not a projector fixed point"
            )
        fixed = project_bounded_action_trajectory(
            state_tensor,
            projected,
            step_cap=self._actor.step_cap,
            action_lower=self._actor.action_lower,
            action_upper=self._actor.action_upper,
        )
        if not bool(torch.isclose(projected, fixed, rtol=0.0, atol=1.0e-6).all()):
            raise V168OfflineRuntimeContractError(
                "V16.8 projected proposal is not idempotent"
            )
        proposal = projected[0].cpu().numpy().astype(np.float32)
        previous = np.concatenate((state[None], proposal[:-1]), axis=0)
        cap = np.asarray(ACTION_CONTRACT["step_cap"], dtype=np.float32)
        finite = bool(np.isfinite(proposal).all())
        bounded = bool(((proposal >= lower) & (proposal <= upper)).all())
        step_legal = bool((np.abs(proposal - previous) <= cap + 1.0e-6).all())
        if not (finite and bounded and step_legal):
            raise V168OfflineRuntimeContractError(
                "V16.8 appended proposal is outside the action contract"
            )
        existing_bank_sha256 = _tensor_sha256(candidate_bank)
        merged = np.concatenate((candidate_bank, proposal[None]), axis=0)
        existing_bank_preserved = bool(
            merged[:-1].dtype == candidate_bank.dtype
            and merged[:-1].shape == candidate_bank.shape
            and np.array_equal(merged[:-1], candidate_bank)
            and _tensor_sha256(merged[:-1]) == existing_bank_sha256
        )
        if not existing_bank_preserved:
            raise V168OfflineRuntimeContractError(
                "existing candidate_bank changed while appending the Actor proposal"
            )
        proposal_index = len(bank)
        actor_metadata = {
            "source": ACTOR_SOURCE,
            "variant": "five_command",
            "actor_chunk_steps": COMMAND_STEPS,
            "candidate_domain": "canonical_joint_action",
            "transition": f"H{self._transition}",
            "window_id": self._window_id,
            "goal_provenance": "planner_proxy_fixed_window",
            "goal_sha256": _tensor_sha256(self._goal),
            "training_h4_equivalence_verified": False,
            "ranking_status": "UNRANKED",
            "ranking_authority": (
                "external_frozen_lewm_or_registered_offline_ranker"
            ),
            "lewm_loaded_by_adapter": False,
            "cem_loaded": False,
            "projector_fixed": True,
            "finite": finite,
            "absolute_bounds": bounded,
            "step_cap_legal": step_legal,
            "descriptive_only": True,
            "commands_sent": 0,
            "deployment_authorized": False,
            "checkpoint_sha256": self._binding.checkpoint_sha256,
            "state_sha256": self._binding.state_sha256,
        }
        internal_metadata = _readonly_metadata((*metadata, actor_metadata))
        result_metadata = _readonly_metadata((*metadata, actor_metadata))
        internal_bank = _readonly_copy(merged)
        result_bank = _readonly_copy(merged)
        merged_sha256 = _tensor_sha256(internal_bank)
        audit = {
            "format": PROTOCOL_ID,
            "window_id": self._window_id,
            "transition": int(self._transition),
            "history_valid": valid.tolist(),
            "history_sha256": _tensor_sha256(history),
            "condensed_past_sha256": _tensor_sha256(past),
            "goal_sha256": _tensor_sha256(self._goal),
            "existing_candidates": int(len(bank)),
            "appended_candidates": 1,
            "output_candidates": int(len(merged)),
            "actor_proposal_index": int(proposal_index),
            "actor_proposal_sha256": _tensor_sha256(proposal),
            "candidate_bank_sha256_after_append": merged_sha256,
            "candidate_metadata_snapshotted_read_only": True,
            "existing_bank_sha256_before": existing_bank_sha256,
            "existing_bank_sha256_after": _tensor_sha256(internal_bank[:-1]),
            "existing_bank_preserved": existing_bank_preserved,
            **existing_legality,
            "projector_fixed": True,
            "ranking_performed": False,
            "candidate_selected": False,
            "commands_sent": 0,
            "deployment_authorized": False,
        }
        self._proposal_generated = True
        self._proposal_current_state = state.copy()
        self._proposal_candidate_bank = internal_bank
        self._proposal_candidate_metadata = internal_metadata
        self._proposal_candidate_bank_sha256 = merged_sha256
        return V168ProposalBank(
            candidates=result_bank,
            metadata=result_metadata,
            actor_proposal_index=proposal_index,
            transition=int(self._transition),
            audit=audit,
        )

    def record_completed_chunk(
        self,
        completed_chunk: np.ndarray,
        *,
        selected_candidate_index: int,
        completion_confirmed: bool,
        provenance: CompletionProvenance,
    ) -> Mapping[str, Any]:
        """Advance only after the caller confirms an offline chunk completed.

        The caller must name the exact prior bank row via
        ``selected_candidate_index`` and provide its bitwise-identical command
        tensor.  This verifies external selection provenance; it does not make
        the adapter a selector. ``offline_commands_sent_log`` means the caller
        is replaying an existing log, not that this adapter sent it.
        """

        self._require_window()
        if completion_confirmed is not True:
            raise V168OfflineRuntimeContractError(
                "offline chunk completion must be explicitly confirmed"
            )
        if provenance not in COMPLETION_PROVENANCES:
            raise V168OfflineRuntimeContractError(
                "completed chunk provenance must be offline_factual or offline_commands_sent_log"
            )
        if (
            not self._proposal_generated
            or self._proposal_current_state is None
            or self._proposal_candidate_bank is None
            or self._proposal_candidate_metadata is None
            or self._proposal_candidate_bank_sha256 is None
        ):
            raise V168OfflineRuntimeContractError(
                "record_completed_chunk requires one proposal for the current transition"
            )
        if isinstance(selected_candidate_index, bool) or not isinstance(
            selected_candidate_index, (int, np.integer)
        ):
            raise V168OfflineRuntimeContractError(
                "selected_candidate_index must be an explicit integer"
            )
        selected_index = int(selected_candidate_index)
        if selected_index < 0 or selected_index >= len(self._proposal_candidate_bank):
            raise V168OfflineRuntimeContractError(
                "selected_candidate_index leaves the prior candidate bank"
            )
        if not isinstance(completed_chunk, np.ndarray):
            raise V168OfflineRuntimeContractError(
                "completed offline chunk must be an np.ndarray without implicit conversion"
            )
        if completed_chunk.dtype != np.dtype(np.float32):
            raise V168OfflineRuntimeContractError(
                "completed offline chunk dtype must be exactly float32"
            )
        chunk = completed_chunk
        if chunk.shape != (COMMAND_STEPS, STATE_DIM) or not bool(np.isfinite(chunk).all()):
            raise V168OfflineRuntimeContractError(
                "completed offline chunk must be finite canonical [5,6]"
            )
        expected = self._proposal_candidate_bank[selected_index]
        if not (
            np.array_equal(chunk, expected)
            and chunk.tobytes(order="C") == expected.tobytes(order="C")
        ):
            raise V168OfflineRuntimeContractError(
                "completed offline chunk does not exactly match selected_candidate_index"
            )
        if (
            _tensor_sha256(self._proposal_candidate_bank)
            != self._proposal_candidate_bank_sha256
        ):
            raise V168OfflineRuntimeContractError(
                "saved prior candidate bank changed after proposal"
            )
        state = torch.from_numpy(self._proposal_current_state[None]).to(self._device)
        command = torch.from_numpy(chunk[None]).to(self._device)
        projected = project_bounded_action_trajectory(
            state,
            command,
            step_cap=self._actor.step_cap,
            action_lower=self._actor.action_lower,
            action_upper=self._actor.action_upper,
        )
        if not bool(torch.isclose(command, projected, rtol=0.0, atol=1.0e-6).all()):
            raise V168OfflineRuntimeContractError(
                "completed offline chunk is not a projector fixed point"
            )
        completed_transition = self._transition
        if completed_transition < HISTORY_CHUNKS:
            self._history[completed_transition] = expected
        selected_metadata = self._proposal_candidate_metadata[selected_index]
        selected_source = selected_metadata.get("source")
        self._completion_audit.append(
            {
                "transition": f"H{completed_transition}",
                "provenance": str(provenance),
                "completion_confirmed": True,
                "selected_candidate_index": selected_index,
                "selected_candidate_source": (
                    None if selected_source is None else str(selected_source)
                ),
                "selected_candidate_sha256": _tensor_sha256(expected),
                "candidate_bank_sha256": self._proposal_candidate_bank_sha256,
                "external_selection_asserted_by_caller": True,
                "selection_performed_by_adapter": False,
                "adapter_sent_commands": 0,
            }
        )
        self._transition += 1
        self._proposal_generated = False
        self._proposal_current_state = None
        self._proposal_candidate_bank = None
        self._proposal_candidate_metadata = None
        self._proposal_candidate_bank_sha256 = None
        return self.window_audit()


__all__ = [
    "ACTION_CONTRACT",
    "ACTOR_SOURCE",
    "COMPLETION_PROVENANCES",
    "PROTOCOL_ID",
    "REGISTERED_CHECKPOINT_SHA256",
    "REGISTERED_RECEIPT_SHA256",
    "REGISTERED_STATE_SHA256",
    "load_registered_pass8_actor",
    "V168OfflineBinding",
    "V168OfflineProposalAdapter",
    "V168OfflineRuntimeContractError",
    "V168ProposalBank",
]
