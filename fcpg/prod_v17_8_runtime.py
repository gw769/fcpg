"""Pure model-clock runtime for the Prod-V17.8 controller.

This module owns no camera, HTTP or robot transport.  It enforces one causal
transaction only::

    actual observation -> five-command proposal -> completed readback
                       -> actual reobservation

The Actor always receives the frozen embedding associated with the supplied
task text, plus the last three *completed* transitions.  Deployments may
inject the precomputed training-cache embedding so MiniLM itself is not part
of the live process.  A proposal and its LeWM prior are transient.  Only
``complete_and_reobserve`` may append issued commands, measured readback and
the proposal's source-scene summary to actual history.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import time

import torch

from fcpg.prod_v17_8 import (
    ANCHORS,
    BELIEF_CLASSES,
    COMMAND_STEPS,
    GRID_CELLS,
    HISTORY_STEPS,
    IMAGE_SIZE,
    PATH_STEPS,
    STATE_DIM,
    TASK_ENCODER,
    TASK_EMBEDDING_DIM,
    VIEWS,
    VIT_DIM,
    VIT_PATCHES,
    V178ActorOutput,
    V178AssignedMeasurement,
    V178Core,
    V178FilterUpdate,
    V178GroundedState,
    V178Innovation,
    V178LeWMPrediction,
    V178Measurement,
    V178RawHistory,
    append_raw_history,
    empty_raw_history,
    pool_scene_tokens,
)


DEFAULT_PLAN_TTL_S = 5.0
MAX_SOURCE_STATE_ENCODER_CELL_DELTA = 4

AnalyticTransportProvider = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]
MonotonicClock = Callable[[], float]


class V178RuntimeError(RuntimeError):
    """Raised before an invalid transaction can change actual model state."""


class FrozenMiniLMTaskTextEncoder:
    """Encode runtime task text with the frozen training-side contract.

    The model is loaded lazily by construction so importing the pure runtime
    module remains possible in lightweight test environments.  A controller
    encodes its task once during construction and retains only that result.
    """

    model_name = TASK_ENCODER

    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise V178RuntimeError(
                "transformers is required to encode V17.8 runtime task text"
            ) from error
        self.device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.requires_grad_(False)
        self._model.eval()

    @torch.inference_mode()
    def __call__(self, text: str) -> torch.Tensor:
        normalized = " ".join(str(text).strip().split())
        if not normalized:
            raise V178RuntimeError("task text must be non-empty")
        encoded = self._tokenizer(
            [normalized],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        hidden = self._model(**encoded).last_hidden_state.float()
        mask = encoded["attention_mask"].to(hidden)[..., None]
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        if tuple(pooled.shape) != (1, TASK_EMBEDDING_DIM):
            raise V178RuntimeError(
                "MiniLM runtime embedding differs from the 384D training contract"
            )
        return pooled.cpu()


@dataclass(frozen=True)
class V178PendingChunk:
    """One transient proposal; none of its prediction is actual memory."""

    source_measurement: V178Measurement
    source_assigned_measurement: V178AssignedMeasurement
    source_update: V178FilterUpdate
    source_q: torch.Tensor
    actor_output: V178ActorOutput
    prediction: V178LeWMPrediction
    source_index: torch.Tensor
    analytic_shift_px: torch.Tensor
    analytic_valid: torch.Tensor
    generation: int
    created_monotonic_s: float
    expires_monotonic_s: float


@dataclass(frozen=True)
class V178RuntimeProposal:
    """A checked five-command proposal and its transient prediction."""

    measurement: V178Measurement
    assigned_measurement: V178AssignedMeasurement
    filter_update: V178FilterUpdate
    actor_output: V178ActorOutput
    prediction: V178LeWMPrediction
    source_index: torch.Tensor
    analytic_shift_px: torch.Tensor
    analytic_valid: torch.Tensor
    geometry_prior_applied: bool
    history_chunks: int
    generation: int
    created_monotonic_s: float
    expires_monotonic_s: float


def _batch_one(
    value: torch.Tensor,
    unbatched_shape: tuple[int, ...],
    *,
    device: torch.device,
    name: str,
    floating: bool = False,
    boolean: bool = False,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tuple(tensor.shape) == tuple(unbatched_shape):
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != (1, *unbatched_shape):
        raise V178RuntimeError(
            f"{name} must be {list(unbatched_shape)} or batch-one"
        )
    if boolean and tensor.dtype != torch.bool:
        raise V178RuntimeError(f"{name} must be bool")
    if floating and not tensor.is_floating_point():
        raise V178RuntimeError(f"{name} must be floating point")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise V178RuntimeError(f"{name} contains non-finite values")
    return tensor


def _finite_shape(
    value: torch.Tensor, shape: tuple[int, ...], *, name: str
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tuple(tensor.shape) != tuple(shape):
        raise V178RuntimeError(
            f"{name} shape differs: observed={tuple(tensor.shape)} "
            f"expected={tuple(shape)}"
        )
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise V178RuntimeError(f"{name} must be finite floating point")
    return tensor


def _normalized_distribution(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    expected = torch.zeros(tensor.shape[:-1], device=tensor.device)
    if not bool(
        torch.allclose(
            torch.logsumexp(tensor.float(), dim=-1),
            expected,
            atol=2.0e-4,
            rtol=0.0,
        )
    ):
        raise V178RuntimeError(f"{name} must be normalized")
    return tensor


def _clone_history(history: V178RawHistory) -> V178RawHistory:
    return V178RawHistory(
        commands=history.commands.detach().clone(),
        paths=history.paths.detach().clone(),
        scene_pooled=history.scene_pooled.detach().clone(),
        valid=history.valid.detach().clone(),
    )


def _clone_optional(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().clone()


def _clone_measurement(value: V178Measurement) -> V178Measurement:
    return V178Measurement(
        scene_tokens=value.scene_tokens.detach().clone(),
        dense_features=value.dense_features.detach().clone(),
        anchor_hypothesis_log_likelihood=(
            value.anchor_hypothesis_log_likelihood.detach().clone()
        ),
        view_valid=value.view_valid.detach().clone(),
        query_traces=_clone_optional(value.query_traces),
    )


def _clone_assigned(value: V178AssignedMeasurement) -> V178AssignedMeasurement:
    return V178AssignedMeasurement(
        scene_tokens=value.scene_tokens.detach().clone(),
        dense_features=value.dense_features.detach().clone(),
        ordered_log_likelihood=value.ordered_log_likelihood.detach().clone(),
        assignment=value.assignment.detach().clone(),
        view_valid=value.view_valid.detach().clone(),
        query_traces=_clone_optional(value.query_traces),
    )


def _clone_grounded(value: V178GroundedState) -> V178GroundedState:
    return V178GroundedState(
        log_posterior=value.log_posterior.detach().clone(),
        ordered_tokens=value.ordered_tokens.detach().clone(),
        dynamics_grounding=value.dynamics_grounding.detach().clone(),
        belief_statistics=value.belief_statistics.detach().clone(),
        view_valid=value.view_valid.detach().clone(),
        appearance_available=value.appearance_available.detach().clone(),
    )


def _clone_innovation(value: V178Innovation) -> V178Innovation:
    return V178Innovation(
        innovation=value.innovation.detach().clone(),
        belief_surprise=value.belief_surprise.detach().clone(),
        state_residual=value.state_residual.detach().clone(),
        view_valid=value.view_valid.detach().clone(),
        has_prediction=value.has_prediction.detach().clone(),
    )


def _clone_update(value: V178FilterUpdate) -> V178FilterUpdate:
    return V178FilterUpdate(
        grounded=_clone_grounded(value.grounded),
        innovation=_clone_innovation(value.innovation),
    )


def _clone_actor_output(value: V178ActorOutput) -> V178ActorOutput:
    return V178ActorOutput(
        commands=value.commands.detach().clone(),
        desired_commands=value.desired_commands.detach().clone(),
        increments=value.increments.detach().clone(),
        action_queries_by_layer=_clone_optional(value.action_queries_by_layer),
    )


def _clone_prediction(value: V178LeWMPrediction) -> V178LeWMPrediction:
    return V178LeWMPrediction(
        prior_log_belief=value.prior_log_belief.detach().clone(),
        predicted_q_path=value.predicted_q_path.detach().clone(),
        transport_shift_px=value.transport_shift_px.detach().clone(),
        transport_sigma_px=value.transport_sigma_px.detach().clone(),
        tokens_by_layer=_clone_optional(value.tokens_by_layer),
    )


class V178RuntimeController:
    """Strict batch-one state machine for a V17.8 core."""

    def __init__(
        self,
        model: V178Core,
        task_text: str,
        *,
        device: torch.device | str,
        source_index: int,
        task_embedding: torch.Tensor | None = None,
        analytic_transport_provider: AnalyticTransportProvider | None = None,
        source_state_atol_position: float = 0.25,
        command_atol_position: float = 0.25,
        plan_ttl_s: float = DEFAULT_PLAN_TTL_S,
        monotonic_clock: MonotonicClock = time.monotonic,
    ) -> None:
        self.device = torch.device(device)
        tolerance = float(source_state_atol_position)
        command_tolerance = float(command_atol_position)
        ttl = float(plan_ttl_s)
        for name, value, strictly_positive in (
            ("source_state_atol_position", tolerance, False),
            ("command_atol_position", command_tolerance, False),
            ("plan_ttl_s", ttl, True),
        ):
            if not math.isfinite(value) or value < 0.0 or (
                strictly_positive and value == 0.0
            ):
                raise V178RuntimeError(f"{name} is invalid")
        if not callable(monotonic_clock):
            raise V178RuntimeError("monotonic_clock must be callable")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise V178RuntimeError("source_index must be an integer")
        normalized_task_text = " ".join(str(task_text).strip().split())
        if not normalized_task_text:
            raise V178RuntimeError("task text must be non-empty")
        vocabulary = int(getattr(model.config, "source_vocabulary", 0))
        if not 0 <= source_index < vocabulary:
            raise V178RuntimeError("source_index is outside the model vocabulary")
        if getattr(model.config, "actor_mode", None) != "full":
            raise V178RuntimeError("runtime requires the V17.8 full Actor")
        if int(getattr(model.config, "raw_history_steps", 0)) != HISTORY_STEPS:
            raise V178RuntimeError("runtime requires exactly three history slots")
        if int(getattr(model.config, "task_embedding_dim", 0)) != TASK_EMBEDDING_DIM:
            raise V178RuntimeError("runtime task embedding contract differs")
        actor = getattr(model, "actor", None)
        if actor is None:
            raise V178RuntimeError("model has no Actor")
        if hasattr(actor, "step_cap"):
            raise V178RuntimeError("V17.8 Actor must not contain a step cap")
        if analytic_transport_provider is not None and not callable(
            analytic_transport_provider
        ):
            raise V178RuntimeError("analytic transport provider must be callable")

        if task_embedding is None:
            encoder = FrozenMiniLMTaskTextEncoder(device="cpu")
            if (
                not callable(encoder)
                or str(getattr(encoder, "model_name", "")) != TASK_ENCODER
            ):
                raise V178RuntimeError(
                    f"runtime task encoder must be {TASK_ENCODER}"
                )
            try:
                encoded_task = encoder(normalized_task_text)
            except V178RuntimeError:
                raise
            except Exception as error:
                raise V178RuntimeError("runtime task text encoding failed") from error
        else:
            encoded_task = torch.as_tensor(task_embedding).detach().clone()
        task = _batch_one(
            encoded_task,
            (TASK_EMBEDDING_DIM,),
            device=self.device,
            name="task embedding",
            floating=True,
        )
        task_norm = torch.linalg.vector_norm(task.float(), dim=-1)
        if not bool(
            torch.allclose(
                task_norm,
                torch.ones_like(task_norm),
                atol=1.0e-3,
                rtol=0.0,
            )
        ):
            raise V178RuntimeError(
                "task embedding must follow the unit-L2 training-cache contract"
            )
        self.model = model.to(self.device).eval()
        self._task_embedding = task.detach().clone()
        self._task_text = normalized_task_text
        self.source_index = int(source_index)
        self.source_state_atol_position = tolerance
        self.command_atol_position = command_tolerance
        self.plan_ttl_s = ttl
        self._clock = monotonic_clock
        self._source_index_tensor = torch.tensor(
            [source_index], dtype=torch.long, device=self.device
        )
        self._analytic_transport_provider = analytic_transport_provider
        self._source_state_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None
        self._next_proposal_generation = 1
        self._last_clock_s: float | None = None
        self._clear_actual_state()

    def _clear_actual_state(self) -> None:
        self._pending: V178PendingChunk | None = None
        self._latest_measurement: V178Measurement | None = None
        self._latest_assigned: V178AssignedMeasurement | None = None
        self._latest_update: V178FilterUpdate | None = None
        self._latest_q: torch.Tensor | None = None
        self._recent_readback_deltas: torch.Tensor | None = None
        self._recent_readback_errors: torch.Tensor | None = None
        self._history = empty_raw_history(1, device=self.device)
        self._completed_chunks = 0

    def reset(self, *, discard_pending: bool = False) -> None:
        """Clear all actual memory; never silently discard a live proposal."""

        if self._pending is not None and not bool(discard_pending):
            raise V178RuntimeError(
                "reset would discard a pending proposal; cancel it explicitly"
            )
        self._clear_actual_state()

    @property
    def has_pending_chunk(self) -> bool:
        return self._pending is not None

    @property
    def has_latest_actual(self) -> bool:
        return self._latest_update is not None

    @property
    def completed_chunks(self) -> int:
        return self._completed_chunks

    @property
    def pending_generation(self) -> int | None:
        return None if self._pending is None else int(self._pending.generation)

    @property
    def pending_commands(self) -> torch.Tensor | None:
        return (
            None
            if self._pending is None
            else self._pending.actor_output.commands.detach().clone()
        )

    @property
    def pending_source_q(self) -> torch.Tensor | None:
        return (
            None
            if self._pending is None
            else self._pending.source_q.detach().clone()
        )

    @property
    def latest_actual_q(self) -> torch.Tensor | None:
        return None if self._latest_q is None else self._latest_q.detach().clone()

    @property
    def task_embedding(self) -> torch.Tensor:
        """Return a copy; the task condition is immutable for this runtime."""

        return self._task_embedding.detach().clone()

    @property
    def task_text(self) -> str:
        return self._task_text

    @property
    def pending_expired(self) -> bool:
        if self._pending is None:
            return False
        return self._now() > self._pending.expires_monotonic_s

    def actual_history(self) -> V178RawHistory:
        return _clone_history(self._history)

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except Exception as exc:
            raise V178RuntimeError("monotonic clock failed") from exc
        if not math.isfinite(value):
            raise V178RuntimeError("monotonic clock returned a non-finite value")
        if self._last_clock_s is not None and value < self._last_clock_s:
            raise V178RuntimeError("monotonic clock moved backwards")
        self._last_clock_s = value
        return value

    def _require_pending(self, generation: int) -> V178PendingChunk:
        pending = self._pending
        if pending is None:
            raise V178RuntimeError("no proposed chunk is waiting for completion")
        if self._now() > pending.expires_monotonic_s:
            self._pending = None
            raise V178RuntimeError("pending proposal expired and was discarded")
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise V178RuntimeError("generation must be an integer")
        if generation != pending.generation:
            raise V178RuntimeError(
                "generation differs from the pending proposal"
            )
        return pending

    def bind_source_state_encoder(
        self, encoder: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        if not callable(encoder):
            raise V178RuntimeError("source state encoder must be callable")
        if self._source_state_encoder is not None:
            raise V178RuntimeError("source state encoder is already bound")
        self._require_pre_observation_binding("source state encoder")
        self._source_state_encoder = encoder

    def bind_analytic_transport_provider(
        self, provider: AnalyticTransportProvider
    ) -> None:
        if not callable(provider):
            raise V178RuntimeError("analytic transport provider must be callable")
        if self._analytic_transport_provider is not None:
            raise V178RuntimeError("analytic transport provider is already bound")
        self._require_pre_observation_binding("analytic transport provider")
        self._analytic_transport_provider = provider

    def _require_pre_observation_binding(self, name: str) -> None:
        if (
            self._pending is not None
            or self._latest_update is not None
            or self._completed_chunks != 0
        ):
            raise V178RuntimeError(f"{name} must be bound before observation")

    def _encoded_source_state(self, state: torch.Tensor) -> torch.Tensor:
        encoder = self._source_state_encoder
        if encoder is None:
            raise V178RuntimeError("source state encoder is not bound")
        try:
            cells = torch.as_tensor(
                encoder(state.detach().to("cpu").clone()), device="cpu"
            )
        except Exception as exc:
            raise V178RuntimeError("source state encoder failed") from exc
        if tuple(cells.shape) == (STATE_DIM,):
            cells = cells.unsqueeze(0)
        if tuple(cells.shape) != (1, STATE_DIM):
            raise V178RuntimeError(
                "source state encoder must return six batch-one cells"
            )
        if cells.dtype == torch.bool or cells.is_floating_point() or cells.is_complex():
            raise V178RuntimeError("source state encoder must return integer cells")
        return cells.to(dtype=torch.int64)

    def source_state_cell_delta(
        self, reference: torch.Tensor, measured: torch.Tensor
    ) -> torch.Tensor | None:
        expected = _batch_one(
            reference,
            (STATE_DIM,),
            device=self.device,
            name="reference source state",
            floating=True,
        )
        actual = _batch_one(
            measured,
            (STATE_DIM,),
            device=self.device,
            name="measured source state",
            floating=True,
        )
        if self._source_state_encoder is None:
            return None
        return torch.abs(
            self._encoded_source_state(actual) - self._encoded_source_state(expected)
        )

    def source_states_match(
        self, reference: torch.Tensor, measured: torch.Tensor
    ) -> bool:
        expected = _batch_one(
            reference,
            (STATE_DIM,),
            device=self.device,
            name="reference source state",
            floating=True,
        )
        actual = _batch_one(
            measured,
            (STATE_DIM,),
            device=self.device,
            name="measured source state",
            floating=True,
        )
        cell_delta = self.source_state_cell_delta(expected, actual)
        if cell_delta is not None:
            return bool(torch.all(cell_delta <= MAX_SOURCE_STATE_ENCODER_CELL_DELTA))
        return bool(
            torch.allclose(
                actual,
                expected,
                atol=self.source_state_atol_position,
                rtol=0.0,
            )
        )

    def _endpoint_inputs(
        self,
        dino_tokens: torch.Tensor,
        view_valid: torch.Tensor,
        current_q: torch.Tensor,
        rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = _batch_one(
            dino_tokens,
            (VIEWS, VIT_PATCHES, VIT_DIM),
            device=self.device,
            name="DINO tokens",
            floating=True,
        )
        valid = _batch_one(
            view_valid,
            (VIEWS,),
            device=self.device,
            name="view_valid",
            boolean=True,
        )
        state = _batch_one(
            current_q,
            (STATE_DIM,),
            device=self.device,
            name="current_q",
            floating=True,
        )
        image = _batch_one(
            rgb,
            (VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE),
            device=self.device,
            name="RGB",
        )
        if image.dtype != torch.uint8 and not image.is_floating_point():
            raise V178RuntimeError("RGB must be uint8 or floating point")
        if image.is_floating_point() and not bool(
            ((image >= 0.0) & (image <= 1.0)).all()
        ):
            raise V178RuntimeError("floating RGB must lie in [0,1]")
        return tokens, valid, state, image

    def _h0_geometry(self, geometry_log_prior: torch.Tensor) -> torch.Tensor:
        """Validate the calibrated physics prior required for H0 identity."""

        geometry = _batch_one(
            geometry_log_prior,
            (VIEWS, ANCHORS, BELIEF_CLASSES),
            device=self.device,
            name="geometry prior",
            floating=True,
        )
        _normalized_distribution(geometry, name="geometry prior")
        return geometry

    def _check_measurement(
        self, measurement: V178Measurement, expected_valid: torch.Tensor
    ) -> None:
        _finite_shape(
            measurement.scene_tokens,
            (1, VIEWS, VIT_PATCHES, VIT_DIM),
            name="measurement scene tokens",
        )
        _finite_shape(
            measurement.dense_features,
            (1, VIEWS, GRID_CELLS, self.model.config.feature_dim),
            name="measurement dense features",
        )
        likelihood = _finite_shape(
            measurement.anchor_hypothesis_log_likelihood,
            (1, VIEWS, ANCHORS, BELIEF_CLASSES),
            name="measurement hypothesis distribution",
        )
        _normalized_distribution(
            likelihood, name="measurement hypothesis distribution"
        )
        if measurement.view_valid.dtype != torch.bool or not torch.equal(
            measurement.view_valid, expected_valid
        ):
            raise V178RuntimeError("measurement changed view_valid")

    def _check_assigned(
        self, assigned: V178AssignedMeasurement, expected_valid: torch.Tensor
    ) -> None:
        ordered = _finite_shape(
            assigned.ordered_log_likelihood,
            (1, VIEWS, ANCHORS, BELIEF_CLASSES),
            name="ordered measurement distribution",
        )
        _normalized_distribution(ordered, name="ordered measurement distribution")
        assignment = _finite_shape(
            assigned.assignment,
            (1, VIEWS, ANCHORS, ANCHORS),
            name="identity assignment",
        )
        row = assignment.sum(-1)
        column = assignment.sum(-2)
        if not bool(
            ((assignment == 0.0) | (assignment == 1.0)).all()
            and torch.equal(row, torch.ones_like(row))
            and torch.equal(column, torch.ones_like(column))
        ):
            raise V178RuntimeError("identity assignment must be a permutation")
        if assigned.view_valid.dtype != torch.bool or not torch.equal(
            assigned.view_valid, expected_valid
        ):
            raise V178RuntimeError("assigned measurement changed view_valid")

    def _check_filter_update(
        self,
        update: V178FilterUpdate,
        *,
        expected_valid: torch.Tensor,
        expected_prediction: bool,
    ) -> None:
        posterior = _finite_shape(
            update.grounded.log_posterior,
            (1, VIEWS, ANCHORS, BELIEF_CLASSES),
            name="grounded posterior",
        )
        _normalized_distribution(posterior, name="grounded posterior")
        _finite_shape(
            update.grounded.dynamics_grounding,
            (1, VIEWS, ANCHORS, self.model.config.compact_dim),
            name="dynamics grounding",
        )
        _finite_shape(
            update.grounded.ordered_tokens,
            (1, VIEWS, ANCHORS, self.model.config.feature_dim),
            name="grounded appearance tokens",
        )
        _finite_shape(
            update.grounded.belief_statistics,
            (1, VIEWS, ANCHORS, 8),
            name="grounded belief statistics",
        )
        _finite_shape(
            update.innovation.innovation,
            (1, VIEWS, ANCHORS, self.model.config.compact_dim),
            name="innovation",
        )
        _finite_shape(
            update.innovation.belief_surprise,
            (1, VIEWS, ANCHORS),
            name="belief surprise",
        )
        _finite_shape(
            update.innovation.state_residual,
            (1, PATH_STEPS, STATE_DIM),
            name="state innovation",
        )
        if update.grounded.view_valid.dtype != torch.bool or not torch.equal(
            update.grounded.view_valid, expected_valid
        ):
            raise V178RuntimeError("filter update changed view_valid")
        if (
            update.grounded.appearance_available.dtype != torch.bool
            or not torch.equal(update.grounded.appearance_available, expected_valid)
        ):
            raise V178RuntimeError(
                "appearance availability must equal actual view validity"
            )
        if update.innovation.view_valid.dtype != torch.bool or not torch.equal(
            update.innovation.view_valid, expected_valid
        ):
            raise V178RuntimeError("innovation changed view_valid")
        present = update.innovation.has_prediction
        if (
            present.dtype != torch.bool
            or tuple(present.shape) != (1,)
            or bool(present.item()) != expected_prediction
        ):
            raise V178RuntimeError("innovation prediction clock is inconsistent")

    def _check_actor_output(
        self, output: V178ActorOutput, source_q: torch.Tensor
    ) -> None:
        commands = _finite_shape(
            output.commands,
            (1, COMMAND_STEPS, STATE_DIM),
            name="Actor commands",
        )
        desired = _finite_shape(
            output.desired_commands,
            (1, COMMAND_STEPS, STATE_DIM),
            name="Actor desired commands",
        )
        increments = _finite_shape(
            output.increments,
            (1, COMMAND_STEPS, STATE_DIM),
            name="Actor increments",
        )
        actor = self.model.actor
        if hasattr(actor, "step_cap"):
            raise V178RuntimeError("V17.8 Actor unexpectedly contains a step cap")
        lower = _finite_shape(
            actor.action_lower, (STATE_DIM,), name="Actor lower joint bounds"
        ).to(commands)
        upper = _finite_shape(
            actor.action_upper, (STATE_DIM,), name="Actor upper joint bounds"
        ).to(commands)
        if not bool((lower < upper).all()):
            raise V178RuntimeError("Actor joint bounds are invalid")
        expected_desired = source_q[:, None].to(increments) + torch.cumsum(
            increments, dim=1
        )
        if not bool(
            torch.allclose(desired, expected_desired.to(desired), atol=1.0e-4, rtol=0.0)
        ):
            raise V178RuntimeError("Actor desired commands do not accumulate deltas")
        expected_commands = desired.clamp(
            min=lower.reshape(1, 1, STATE_DIM),
            max=upper.reshape(1, 1, STATE_DIM),
        )
        if not bool(
            torch.allclose(commands, expected_commands, atol=1.0e-4, rtol=0.0)
        ):
            raise V178RuntimeError(
                "Actor commands are not the absolute-limit projection"
            )

    def _check_prediction(
        self,
        prediction: V178LeWMPrediction,
        source_q: torch.Tensor,
        analytic_shift_px: torch.Tensor,
        analytic_valid: torch.Tensor,
    ) -> None:
        prior = _finite_shape(
            prediction.prior_log_belief,
            (1, VIEWS, ANCHORS, BELIEF_CLASSES),
            name="predicted prior",
        )
        _normalized_distribution(prior, name="predicted prior")
        path = _finite_shape(
            prediction.predicted_q_path,
            (1, PATH_STEPS, STATE_DIM),
            name="predicted q path",
        )
        if not bool(
            torch.allclose(path[:, 0], source_q.to(path), atol=1.0e-5, rtol=0.0)
        ):
            raise V178RuntimeError("predicted q0 differs from source state")
        shift = _finite_shape(
            prediction.transport_shift_px,
            (1, VIEWS, ANCHORS, 2),
            name="transport shift",
        )
        sigma = _finite_shape(
            prediction.transport_sigma_px,
            (1, VIEWS, ANCHORS, 2),
            name="transport sigma",
        )
        valid_tracks = analytic_valid[:, :, None, None].expand_as(shift)
        residual = shift - analytic_shift_px.to(shift)
        learned_bound = float(self.model.config.max_transport_shift_px) + 1.0e-4
        analytic_bound = float(self.model.config.analytic_residual_max_px) + 1.0e-4
        within = torch.where(
            valid_tracks,
            residual.abs() <= analytic_bound,
            shift.abs() <= learned_bound,
        )
        if not bool(within.all()):
            raise V178RuntimeError("transport shift violates configured bounds")
        if not bool(
            (
                (sigma >= float(self.model.config.minimum_transport_sigma_px) - 1.0e-4)
                & (sigma <= float(self.model.config.maximum_transport_sigma_px) + 1.0e-4)
            ).all()
        ):
            raise V178RuntimeError("transport sigma exceeds configured bounds")

    def _analytic_inputs(
        self, source_q: torch.Tensor, commands: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(getattr(self.model.config, "analytic_transport", False)):
            return (
                torch.zeros(1, VIEWS, ANCHORS, 2, device=self.device),
                torch.zeros(1, VIEWS, dtype=torch.bool, device=self.device),
            )
        provider = self._analytic_transport_provider
        if provider is None:
            raise V178RuntimeError(
                "analytic transport is enabled but no provider is bound"
            )
        try:
            result = provider(
                source_q.detach().clone(), commands.detach().clone()
            )
        except Exception as exc:
            raise V178RuntimeError("analytic transport provider failed") from exc
        if not isinstance(result, tuple) or len(result) != 2:
            raise V178RuntimeError(
                "analytic transport provider must return (shift_px, valid)"
            )
        shift = _batch_one(
            result[0],
            (VIEWS, ANCHORS, 2),
            device=self.device,
            name="analytic_shift_px",
            floating=True,
        )
        valid = _batch_one(
            result[1],
            (VIEWS,),
            device=self.device,
            name="analytic_valid",
            boolean=True,
        )
        return torch.where(
            valid[:, :, None, None], shift, torch.zeros_like(shift)
        ), valid

    def _build_proposal(
        self,
        measurement: V178Measurement,
        assigned: V178AssignedMeasurement,
        update: V178FilterUpdate,
        state: torch.Tensor,
        *,
        geometry_prior_applied: bool,
    ) -> tuple[V178PendingChunk, V178RuntimeProposal]:
        history_snapshot = _clone_history(self._history)
        actor_output = self.model.act(
            measurement.scene_tokens,
            update.grounded,
            update.innovation.innovation,
            state,
            history=history_snapshot,
            task_embedding=self._task_embedding.detach().clone(),
        )
        self._check_actor_output(actor_output, state)
        analytic_shift, analytic_valid = self._analytic_inputs(
            state, actor_output.commands
        )
        prediction = self.model.predict(
            update.grounded,
            state,
            actor_output.commands,
            recent_readback_deltas=self._recent_readback_deltas,
            recent_readback_errors=self._recent_readback_errors,
            source_index=self._source_index_tensor,
            analytic_shift_px=analytic_shift,
            analytic_valid=analytic_valid,
        )
        self._check_prediction(prediction, state, analytic_shift, analytic_valid)
        now = self._now()
        generation = self._next_proposal_generation
        source_index = self._source_index_tensor.detach().clone()
        history_chunks = int(self._history.valid.sum().item())
        pending = V178PendingChunk(
            source_measurement=_clone_measurement(measurement),
            source_assigned_measurement=_clone_assigned(assigned),
            source_update=_clone_update(update),
            source_q=state.detach().clone(),
            actor_output=_clone_actor_output(actor_output),
            prediction=_clone_prediction(prediction),
            source_index=source_index,
            analytic_shift_px=analytic_shift.detach().clone(),
            analytic_valid=analytic_valid.detach().clone(),
            generation=generation,
            created_monotonic_s=now,
            expires_monotonic_s=now + self.plan_ttl_s,
        )
        proposal = V178RuntimeProposal(
            measurement=measurement,
            assigned_measurement=assigned,
            filter_update=update,
            actor_output=actor_output,
            prediction=prediction,
            source_index=source_index.detach().clone(),
            analytic_shift_px=analytic_shift.detach().clone(),
            analytic_valid=analytic_valid.detach().clone(),
            geometry_prior_applied=bool(geometry_prior_applied),
            history_chunks=history_chunks,
            generation=generation,
            created_monotonic_s=now,
            expires_monotonic_s=now + self.plan_ttl_s,
        )
        return pending, proposal

    def _reject_existing_pending(self) -> None:
        if self._pending is None:
            return
        if self._now() > self._pending.expires_monotonic_s:
            self._pending = None
            raise V178RuntimeError(
                "pending proposal expired and was discarded; propose again"
            )
        raise V178RuntimeError(
            "complete or cancel the pending chunk before proposing again"
        )

    def _commit_proposal(self, pending: V178PendingChunk) -> None:
        self._pending = pending
        self._next_proposal_generation += 1

    @torch.inference_mode()
    def observe_and_propose(
        self,
        dino_tokens: torch.Tensor,
        view_valid: torch.Tensor,
        current_q: torch.Tensor,
        *,
        rgb: torch.Tensor,
        geometry_log_prior: torch.Tensor,
    ) -> V178RuntimeProposal:
        """Consume the unique H0 observation and atomically propose H0."""

        self._reject_existing_pending()
        if self._latest_update is not None:
            raise V178RuntimeError(
                "an actual endpoint already exists; use propose_from_latest or reset"
            )
        if bool(getattr(self.model.config, "analytic_transport", False)) and (
            self._analytic_transport_provider is None
        ):
            raise V178RuntimeError(
                "analytic transport is enabled but no provider is bound"
            )
        tokens, valid, state, image = self._endpoint_inputs(
            dino_tokens, view_valid, current_q, rgb
        )
        geometry = self._h0_geometry(geometry_log_prior)
        measurement = self.model.measure(tokens, valid, image)
        self._check_measurement(measurement, valid)
        assigned = self.model.assign_identity(measurement, geometry)
        self._check_assigned(assigned, valid)
        update = self.model.update(assigned, geometry_log_prior=geometry)
        self._check_filter_update(
            update, expected_valid=valid, expected_prediction=False
        )
        pending, proposal = self._build_proposal(
            measurement,
            assigned,
            update,
            state,
            geometry_prior_applied=geometry is not None,
        )

        # Actual H0 and the proposal commit only after every contract check.
        self._latest_measurement = _clone_measurement(measurement)
        self._latest_assigned = _clone_assigned(assigned)
        self._latest_update = _clone_update(update)
        self._latest_q = state.detach().clone()
        self._commit_proposal(pending)
        return proposal

    @torch.inference_mode()
    def propose_from_latest(
        self, current_q: torch.Tensor | None = None
    ) -> V178RuntimeProposal:
        """Propose from the already consumed H1+ actual endpoint."""

        self._reject_existing_pending()
        if (
            self._latest_q is None
            or self._latest_measurement is None
            or self._latest_assigned is None
            or self._latest_update is None
        ):
            raise V178RuntimeError("no actual endpoint is available")
        state = self._latest_q
        if current_q is not None:
            supplied = _batch_one(
                current_q,
                (STATE_DIM,),
                device=self.device,
                name="current_q",
                floating=True,
            )
            if not self.source_states_match(self._latest_q, supplied):
                raise V178RuntimeError(
                    "current_q differs from latest reobserved endpoint"
                )
            state = supplied
        pending, proposal = self._build_proposal(
            self._latest_measurement,
            self._latest_assigned,
            self._latest_update,
            state,
            geometry_prior_applied=False,
        )
        self._commit_proposal(pending)
        return proposal

    def cancel_pending_proposal(self, *, generation: int) -> None:
        """Discard an unexecuted, unexpired proposal without advancing time."""

        self._require_pending(generation)
        self._pending = None

    @torch.inference_mode()
    def complete_and_reobserve(
        self,
        actual_dino_tokens: torch.Tensor,
        actual_view_valid: torch.Tensor,
        executed_commands: torch.Tensor,
        measured_q_path: torch.Tensor,
        *,
        generation: int,
        actual_rgb: torch.Tensor,
        actual_endpoint_q: torch.Tensor | None = None,
        path_valid: torch.Tensor | None = None,
    ) -> V178FilterUpdate:
        """Commit exactly five issued commands and one actual reobservation."""

        pending = self._require_pending(generation)
        executed = _batch_one(
            executed_commands,
            (COMMAND_STEPS, STATE_DIM),
            device=self.device,
            name="executed commands",
            floating=True,
        )
        if not bool(
            torch.allclose(
                executed,
                pending.actor_output.commands.to(executed),
                atol=self.command_atol_position,
                rtol=0.0,
            )
        ):
            raise V178RuntimeError(
                "executed commands differ from pending Actor proposal"
            )
        path = _batch_one(
            measured_q_path,
            (PATH_STEPS, STATE_DIM),
            device=self.device,
            name="measured q path",
            floating=True,
        )
        if path_valid is not None:
            valid_path = _batch_one(
                path_valid,
                (PATH_STEPS,),
                device=self.device,
                name="path_valid",
                boolean=True,
            )
            if not bool(valid_path.all()):
                raise V178RuntimeError(
                    "all six measured q path samples must be valid"
                )
        if not self.source_states_match(pending.source_q.to(path), path[:, 0]):
            raise V178RuntimeError(
                "measured q0 differs from proposed source state"
            )
        endpoint_q = path[:, -1]
        if actual_endpoint_q is not None:
            endpoint_q = _batch_one(
                actual_endpoint_q,
                (STATE_DIM,),
                device=self.device,
                name="actual endpoint q",
                floating=True,
            )
            if not self.source_states_match(path[:, -1], endpoint_q):
                raise V178RuntimeError(
                    "actual endpoint q differs from measured q path endpoint"
                )
        tokens, valid, final_q, image = self._endpoint_inputs(
            actual_dino_tokens,
            actual_view_valid,
            endpoint_q,
            actual_rgb,
        )
        measurement = self.model.measure(tokens, valid, image)
        self._check_measurement(measurement, valid)
        assigned = self.model.assign_identity(
            measurement, pending.prediction.prior_log_belief
        )
        self._check_assigned(assigned, valid)
        update = self.model.update(assigned, pending.prediction, path)
        self._check_filter_update(
            update, expected_valid=valid, expected_prediction=True
        )

        # Stage every actual-memory mutation before committing any of them.
        source_scene = pool_scene_tokens(
            pending.source_measurement.scene_tokens,
            pending.source_measurement.view_valid,
        )
        history = append_raw_history(
            self._history,
            executed,
            path,
            source_scene,
        )
        readback_delta = path[:, 1:] - path[:, :-1]
        readback_error = path[:, 1:] - executed.to(path)

        self._latest_measurement = _clone_measurement(measurement)
        self._latest_assigned = _clone_assigned(assigned)
        self._latest_update = _clone_update(update)
        self._latest_q = final_q.detach().clone()
        self._recent_readback_deltas = readback_delta.detach().clone()
        self._recent_readback_errors = readback_error.detach().clone()
        self._history = history
        self._completed_chunks += 1
        self._pending = None
        return update

    def actual_grounded_state(self) -> V178GroundedState:
        """Return the latest measurement-updated belief, never a LeWM prior."""

        if self._latest_update is None:
            raise V178RuntimeError("no actual endpoint has been observed")
        return _clone_grounded(self._latest_update.grounded)

    def latest_filter_update(self) -> V178FilterUpdate:
        if self._latest_update is None:
            raise V178RuntimeError("no actual endpoint has been observed")
        return _clone_update(self._latest_update)


__all__ = [
    "AnalyticTransportProvider",
    "DEFAULT_PLAN_TTL_S",
    "FrozenMiniLMTaskTextEncoder",
    "MAX_SOURCE_STATE_ENCODER_CELL_DELTA",
    "MonotonicClock",
    "V178PendingChunk",
    "V178RuntimeController",
    "V178RuntimeError",
    "V178RuntimeProposal",
]
