"""Batch-one causal runtime for the Prod-V17.7 full controller.

The runtime owns only the model clock.  Camera capture, DINO extraction,
calibrated FK projection, checkpoint loading and robot transport remain caller
responsibilities.  One transaction is:

``actual observation -> assignment/update -> Actor -> physical admission
-> analytic + learned GDM``

and, only after all five commands have completed:

``predicted prior -> new actual observation -> assignment/update -> innovation``.

The prediction is stored only in the pending transaction.  It is never written
to the latest actual state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from fcpg.prod_v17_7 import (
    ANCHORS,
    BELIEF_CLASSES,
    COMMAND_STEPS,
    GRID_CELLS,
    IMAGE_SIZE,
    PATH_STEPS,
    STATE_DIM,
    VIEWS,
    VIT_DIM,
    VIT_PATCHES,
    V177ActorOutput,
    V177AssignedMeasurement,
    V177Core,
    V177FilterUpdate,
    V177GroundedState,
    V177LeWMPrediction,
    V177Measurement,
)


class V177RuntimeError(RuntimeError):
    """Raised before an invalid endpoint transaction reaches V17.7."""


MAX_SOURCE_STATE_ENCODER_CELL_DELTA = 4
FISH_SOURCE_INDEX = 129

AnalyticTransportProvider = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]
CommandAdmissionProvider = Callable[[torch.Tensor, torch.Tensor], None]


@dataclass(frozen=True)
class V177PendingChunk:
    """Transient proposal state; none of it is an actual observation."""

    source_measurement: V177Measurement
    source_assigned_measurement: V177AssignedMeasurement
    source_update: V177FilterUpdate
    source_q: torch.Tensor
    actor_output: V177ActorOutput
    prediction: V177LeWMPrediction
    source_index: torch.Tensor
    analytic_shift_px: torch.Tensor
    analytic_valid: torch.Tensor
    generation: int


@dataclass(frozen=True)
class V177RuntimeProposal:
    """One legal five-command proposal and its transient GDM prediction."""

    measurement: V177Measurement
    assigned_measurement: V177AssignedMeasurement
    filter_update: V177FilterUpdate
    actor_output: V177ActorOutput
    prediction: V177LeWMPrediction
    source_index: torch.Tensor
    analytic_shift_px: torch.Tensor
    analytic_valid: torch.Tensor
    geometry_prior_applied: bool
    generation: int


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
    expected = (1, *unbatched_shape)
    if tuple(tensor.shape) != expected:
        raise V177RuntimeError(
            f"{name} must be {list(unbatched_shape)} or batch-one"
        )
    if boolean and tensor.dtype != torch.bool:
        raise V177RuntimeError(f"{name} must be bool")
    if floating and not tensor.is_floating_point():
        raise V177RuntimeError(f"{name} must be floating point")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise V177RuntimeError(f"{name} contains non-finite values")
    return tensor


def _finite_shape(
    value: torch.Tensor,
    shape: tuple[int, ...],
    *,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tuple(tensor.shape) != tuple(shape):
        raise V177RuntimeError(
            f"{name} shape differs: observed={tuple(tensor.shape)} "
            f"expected={tuple(shape)}"
        )
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise V177RuntimeError(f"{name} must be finite floating point")
    return tensor


def _normalized_distribution(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not bool(
        torch.allclose(
            torch.logsumexp(tensor.float(), dim=-1),
            torch.zeros(tensor.shape[:-1], device=tensor.device),
            atol=2.0e-4,
            rtol=0.0,
        )
    ):
        raise V177RuntimeError(f"{name} must be normalized")
    return tensor


class V177RuntimeController:
    """Strict causal state machine for a strictly loaded full V17.7 core."""

    def __init__(
        self,
        model: V177Core,
        *,
        device: torch.device | str,
        source_index: int = FISH_SOURCE_INDEX,
        analytic_transport_provider: AnalyticTransportProvider | None = None,
        command_admission_provider: CommandAdmissionProvider | None = None,
        source_state_atol_position: float = 0.25,
    ) -> None:
        tolerance = float(source_state_atol_position)
        if not bool(torch.isfinite(torch.tensor(tolerance))) or tolerance < 0.0:
            raise V177RuntimeError(
                "source_state_atol_position must be finite and non-negative"
            )
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise V177RuntimeError("source_index must be an integer")
        if source_index != FISH_SOURCE_INDEX:
            raise V177RuntimeError(
                f"live fish runtime requires source_index={FISH_SOURCE_INDEX}"
            )
        vocabulary = int(getattr(model.config, "source_vocabulary", 0))
        if not 0 <= source_index < vocabulary:
            raise V177RuntimeError("source_index is outside the model vocabulary")
        actor = getattr(model, "actor", None)
        actor_mode = getattr(actor, "mode", None)
        configured_actor_mode = getattr(model.config, "actor_mode", None)
        if actor_mode != "full" or configured_actor_mode != "full":
            raise V177RuntimeError("live runtime requires the V17.7 full Actor")
        if analytic_transport_provider is not None and not callable(
            analytic_transport_provider
        ):
            raise V177RuntimeError("analytic transport provider must be callable")
        if command_admission_provider is not None and not callable(
            command_admission_provider
        ):
            raise V177RuntimeError("command admission provider must be callable")

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.source_index = int(source_index)
        self.source_state_atol_position = tolerance
        self._source_index_tensor = torch.tensor(
            [self.source_index], dtype=torch.long, device=self.device
        )
        self._analytic_transport_provider = analytic_transport_provider
        self._command_admission_provider = command_admission_provider
        self._source_state_encoder: (
            Callable[[torch.Tensor], torch.Tensor] | None
        ) = None
        self._next_proposal_generation = 1
        self.reset()

    @property
    def has_bound_source_state_encoder(self) -> bool:
        return self._source_state_encoder is not None

    @property
    def has_bound_analytic_transport_provider(self) -> bool:
        return self._analytic_transport_provider is not None

    @property
    def has_bound_command_admission_provider(self) -> bool:
        return self._command_admission_provider is not None

    def bind_source_state_encoder(
        self, encoder: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """Bind the hardware adapter's calibrated motor-cell identity."""

        if not callable(encoder):
            raise V177RuntimeError("source state encoder must be callable")
        if self._source_state_encoder is not None:
            raise V177RuntimeError("source state encoder is already bound")
        self._require_pre_observation_binding("source state encoder")
        self._source_state_encoder = encoder

    def bind_analytic_transport_provider(
        self, provider: AnalyticTransportProvider
    ) -> None:
        """Bind calibrated FK pixel transport before the first observation."""

        if not callable(provider):
            raise V177RuntimeError("analytic transport provider must be callable")
        if self._analytic_transport_provider is not None:
            raise V177RuntimeError("analytic transport provider is already bound")
        self._require_pre_observation_binding("analytic transport provider")
        self._analytic_transport_provider = provider

    def bind_command_admission_provider(
        self, provider: CommandAdmissionProvider
    ) -> None:
        """Bind the physical reject-only gate before the first observation.

        The provider may either return ``None`` to admit the Actor's commands
        unchanged or raise to reject the complete action block.  It must not
        project or replace commands: the exact admitted block is what both the
        analytic transport and learned GDM receive.
        """

        if not callable(provider):
            raise V177RuntimeError("command admission provider must be callable")
        if self._command_admission_provider is not None:
            raise V177RuntimeError("command admission provider is already bound")
        self._require_pre_observation_binding("command admission provider")
        self._command_admission_provider = provider

    def _require_pre_observation_binding(self, name: str) -> None:
        if (
            self._pending is not None
            or self._latest_q is not None
            or self._latest_update is not None
            or self._completed_chunks != 0
        ):
            raise V177RuntimeError(f"{name} must be bound before observation")

    def _encoded_source_state(self, state: torch.Tensor) -> torch.Tensor:
        encoder = self._source_state_encoder
        if encoder is None:
            raise V177RuntimeError("source state encoder is not bound")
        try:
            cells = torch.as_tensor(
                encoder(state.detach().to("cpu").clone()), device="cpu"
            )
        except Exception as exc:
            raise V177RuntimeError("source state encoder failed") from exc
        if tuple(cells.shape) == (STATE_DIM,):
            cells = cells.unsqueeze(0)
        if tuple(cells.shape) != (1, STATE_DIM):
            raise V177RuntimeError(
                "source state encoder must return six batch-one cells"
            )
        if cells.dtype == torch.bool or cells.is_floating_point() or cells.is_complex():
            raise V177RuntimeError("source state encoder must return integer cells")
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
        cells = self.source_state_cell_delta(expected, actual)
        if cells is None:
            return bool(
                torch.allclose(
                    actual,
                    expected,
                    atol=self.source_state_atol_position,
                    rtol=0.0,
                )
            )
        return bool(torch.all(cells <= MAX_SOURCE_STATE_ENCODER_CELL_DELTA))

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

    def reset(self) -> None:
        self._pending: V177PendingChunk | None = None
        self._latest_measurement: V177Measurement | None = None
        self._latest_assigned: V177AssignedMeasurement | None = None
        self._latest_update: V177FilterUpdate | None = None
        self._latest_q: torch.Tensor | None = None
        self._recent_readback_deltas: torch.Tensor | None = None
        self._recent_readback_errors: torch.Tensor | None = None
        self._completed_chunks = 0

    def _endpoint_inputs(
        self,
        dino_tokens: torch.Tensor,
        view_valid: torch.Tensor,
        current_q: torch.Tensor,
        rgb: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        image = None
        if rgb is not None:
            image = _batch_one(
                rgb,
                (VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE),
                device=self.device,
                name="RGB",
            )
            if image.dtype != torch.uint8 and not image.is_floating_point():
                raise V177RuntimeError("RGB must be uint8 or floating point")
            if image.is_floating_point() and not bool(
                ((image >= 0.0) & (image <= 1.0)).all()
            ):
                raise V177RuntimeError("floating RGB must lie in [0,1]")
        return tokens, valid, state, image

    def _h0_inputs(
        self,
        geometry_log_prior: torch.Tensor | None,
        direct_assignment: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if geometry_log_prior is not None and direct_assignment is not None:
            raise V177RuntimeError(
                "supply either a geometry prior or direct assignment, not both"
            )
        geometry = None
        if geometry_log_prior is not None:
            geometry = _batch_one(
                geometry_log_prior,
                (VIEWS, ANCHORS, BELIEF_CLASSES),
                device=self.device,
                name="geometry prior",
                floating=True,
            )
            _normalized_distribution(geometry, name="geometry prior")
        direct = None
        if direct_assignment is not None:
            direct = _batch_one(
                direct_assignment,
                (VIEWS,),
                device=self.device,
                name="direct_assignment",
            )
            if direct.dtype != torch.bool and not bool(
                ((direct == 0) | (direct == 1)).all()
            ):
                raise V177RuntimeError(
                    "direct_assignment must contain only bool or exact 0/1"
                )
        return geometry, direct

    def _check_measurement(
        self, measurement: V177Measurement, expected_valid: torch.Tensor
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
        if measurement.view_valid.dtype != torch.bool or tuple(
            measurement.view_valid.shape
        ) != (1, VIEWS):
            raise V177RuntimeError("measurement view_valid must be bool [1,2]")
        if not torch.equal(measurement.view_valid, expected_valid):
            raise V177RuntimeError("measurement changed the supplied view_valid mask")

    def _check_assigned(
        self, assigned: V177AssignedMeasurement, expected_valid: torch.Tensor
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
        if not bool(
            (((assignment == 0.0) | (assignment == 1.0)).all())
            and torch.equal(assignment.sum(-1), torch.ones_like(assignment.sum(-1)))
            and torch.equal(assignment.sum(-2), torch.ones_like(assignment.sum(-2)))
        ):
            raise V177RuntimeError("identity assignment must be a permutation")
        if assigned.view_valid.dtype != torch.bool or not torch.equal(
            assigned.view_valid, expected_valid
        ):
            raise V177RuntimeError("assigned measurement changed view_valid")

    def _check_filter_update(
        self,
        update: V177FilterUpdate,
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
        if update.grounded.view_valid.dtype != torch.bool or not torch.equal(
            update.grounded.view_valid, expected_valid
        ):
            raise V177RuntimeError("filter update changed view_valid")
        if update.grounded.appearance_available.dtype != torch.bool or not torch.equal(
            update.grounded.appearance_available, expected_valid
        ):
            raise V177RuntimeError(
                "appearance availability must equal actual view validity"
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
        if update.innovation.view_valid.dtype != torch.bool or not torch.equal(
            update.innovation.view_valid, expected_valid
        ):
            raise V177RuntimeError("innovation changed view_valid")
        present = update.innovation.has_prediction
        if (
            present.dtype != torch.bool
            or tuple(present.shape) != (1,)
            or bool(present.item()) != bool(expected_prediction)
        ):
            raise V177RuntimeError("innovation prediction clock is inconsistent")

    def _check_prediction(
        self,
        prediction: V177LeWMPrediction,
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
            raise V177RuntimeError("predicted q0 differs from its source state")
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
            raise V177RuntimeError("transport shift violates analytic/learned bounds")
        if not bool(
            (
                (sigma >= float(self.model.config.minimum_transport_sigma_px) - 1.0e-4)
                & (sigma <= float(self.model.config.maximum_transport_sigma_px) + 1.0e-4)
            ).all()
        ):
            raise V177RuntimeError("transport sigma exceeds configured bounds")

    def _check_actor_output(
        self, output: V177ActorOutput, source_q: torch.Tensor
    ) -> None:
        commands = _finite_shape(
            output.commands,
            (1, COMMAND_STEPS, STATE_DIM),
            name="Actor commands",
        )
        _finite_shape(
            output.desired_commands,
            (1, COMMAND_STEPS, STATE_DIM),
            name="Actor desired commands",
        )
        _finite_shape(
            output.increments,
            (1, COMMAND_STEPS, STATE_DIM),
            name="Actor increments",
        )
        actor = getattr(self.model, "actor", None)
        cap = getattr(actor, "step_cap", None)
        lower = getattr(actor, "action_lower", None)
        upper = getattr(actor, "action_upper", None)
        if cap is not None:
            cap_tensor = torch.as_tensor(cap, device=commands.device).reshape(1, 1, -1)
            step = torch.diff(
                torch.cat((source_q[:, None].to(commands), commands), dim=1), dim=1
            )
            if tuple(cap_tensor.shape) != (1, 1, STATE_DIM) or not bool(
                (step.abs() <= cap_tensor + 1.0e-4).all()
            ):
                raise V177RuntimeError("Actor commands violate per-joint step caps")
        if lower is not None and upper is not None:
            low = torch.as_tensor(lower, device=commands.device).reshape(1, 1, -1)
            high = torch.as_tensor(upper, device=commands.device).reshape(1, 1, -1)
            if tuple(low.shape) != (1, 1, STATE_DIM) or tuple(high.shape) != (
                1,
                1,
                STATE_DIM,
            ):
                raise V177RuntimeError("Actor action bounds have invalid shape")
            if not bool(
                ((commands >= low - 1.0e-4) & (commands <= high + 1.0e-4)).all()
            ):
                raise V177RuntimeError("Actor commands violate joint bounds")

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
            raise V177RuntimeError(
                "analytic transport is enabled but no provider is bound"
            )
        try:
            result = provider(
                source_q.detach().clone(), commands.detach().clone()
            )
        except Exception as exc:
            raise V177RuntimeError("analytic transport provider failed") from exc
        if not isinstance(result, tuple) or len(result) != 2:
            raise V177RuntimeError(
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
        shift = torch.where(valid[:, :, None, None], shift, torch.zeros_like(shift))
        return shift, valid

    def _admit_actor_commands(
        self, source_q: torch.Tensor, commands: torch.Tensor
    ) -> None:
        provider = self._command_admission_provider
        if provider is None:
            return
        result = provider(
            source_q.detach().clone(), commands.detach().clone()
        )
        if result is not None:
            raise V177RuntimeError(
                "command admission provider must return None or raise"
            )

    def _build_proposal(
        self,
        measurement: V177Measurement,
        assigned: V177AssignedMeasurement,
        update: V177FilterUpdate,
        state: torch.Tensor,
        *,
        geometry_prior_applied: bool,
    ) -> tuple[V177PendingChunk, V177RuntimeProposal]:
        actor_output = self.model.act(
            measurement.scene_tokens,
            update.grounded,
            update.innovation.innovation,
            state,
        )
        self._check_actor_output(actor_output, state)
        # Physical legality belongs before every action-conditioned prediction.
        # A rejected block therefore reaches neither analytic FK transport nor
        # the learned GDM and can never become a pending transaction.
        self._admit_actor_commands(state, actor_output.commands)
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
        self._check_prediction(
            prediction, state, analytic_shift, analytic_valid
        )
        generation = self._next_proposal_generation
        source_index = self._source_index_tensor.detach().clone()
        pending = V177PendingChunk(
            source_measurement=measurement,
            source_assigned_measurement=assigned,
            source_update=update,
            source_q=state.detach().clone(),
            actor_output=actor_output,
            prediction=prediction,
            source_index=source_index,
            analytic_shift_px=analytic_shift.detach().clone(),
            analytic_valid=analytic_valid.detach().clone(),
            generation=generation,
        )
        proposal = V177RuntimeProposal(
            measurement=measurement,
            assigned_measurement=assigned,
            filter_update=update,
            actor_output=actor_output,
            prediction=prediction,
            source_index=source_index.detach().clone(),
            analytic_shift_px=analytic_shift.detach().clone(),
            analytic_valid=analytic_valid.detach().clone(),
            geometry_prior_applied=bool(geometry_prior_applied),
            generation=generation,
        )
        return pending, proposal

    def _commit_proposal(self, pending: V177PendingChunk) -> None:
        self._pending = pending
        self._next_proposal_generation += 1

    @torch.inference_mode()
    def observe_and_propose(
        self,
        dino_tokens: torch.Tensor,
        view_valid: torch.Tensor,
        current_q: torch.Tensor,
        *,
        rgb: torch.Tensor | None = None,
        geometry_log_prior: torch.Tensor | None = None,
        direct_assignment: torch.Tensor | None = None,
    ) -> V177RuntimeProposal:
        """Consume the unique H0 observation and atomically propose H0."""

        if self._pending is not None:
            raise V177RuntimeError(
                "complete or cancel the pending chunk before proposing again"
            )
        if self._latest_update is not None:
            raise V177RuntimeError(
                "an actual endpoint already exists; use propose_from_latest or reset"
            )
        if bool(getattr(self.model.config, "analytic_transport", False)) and (
            self._analytic_transport_provider is None
        ):
            raise V177RuntimeError(
                "analytic transport is enabled but no provider is bound"
            )
        tokens, valid, state, image = self._endpoint_inputs(
            dino_tokens, view_valid, current_q, rgb
        )
        geometry, direct = self._h0_inputs(geometry_log_prior, direct_assignment)
        measurement = self.model.measure(tokens, valid, image)
        self._check_measurement(measurement, valid)
        assigned = self.model.assign_identity(
            measurement, geometry, direct_assignment=direct
        )
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

        # Commit the actual observation and proposal only after every model,
        # geometry and analytic-transport check has succeeded.
        self._latest_measurement = measurement
        self._latest_assigned = assigned
        self._latest_update = update
        self._latest_q = state.detach().clone()
        self._commit_proposal(pending)
        return proposal

    @torch.inference_mode()
    def propose_from_latest(
        self, current_q: torch.Tensor | None = None
    ) -> V177RuntimeProposal:
        """Propose from the already consumed H1+ actual endpoint."""

        if self._pending is not None:
            raise V177RuntimeError(
                "complete or cancel the pending chunk before proposing again"
            )
        if (
            self._latest_q is None
            or self._latest_measurement is None
            or self._latest_assigned is None
            or self._latest_update is None
        ):
            raise V177RuntimeError("no actual endpoint is available")
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
                raise V177RuntimeError(
                    "current_q differs from the latest reobserved endpoint"
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

    def cancel_pending_proposal(self) -> None:
        """Discard an unexecuted action without advancing the actual clock."""

        self._pending = None

    @torch.inference_mode()
    def complete_and_reobserve(
        self,
        actual_dino_tokens: torch.Tensor,
        actual_view_valid: torch.Tensor,
        measured_q_path: torch.Tensor,
        *,
        actual_rgb: torch.Tensor | None = None,
        actual_endpoint_q: torch.Tensor | None = None,
        path_valid: torch.Tensor | None = None,
        generation: int | None = None,
    ) -> V177FilterUpdate:
        """Commit one completed chunk and exactly one actual endpoint update."""

        pending = self._pending
        if pending is None:
            raise V177RuntimeError("no proposed chunk is waiting for completion")
        if generation is not None:
            if isinstance(generation, bool) or not isinstance(generation, int):
                raise V177RuntimeError("completion generation must be an integer")
            if generation != pending.generation:
                raise V177RuntimeError(
                    "completion generation differs from the pending proposal"
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
                raise V177RuntimeError(
                    "all six measured q path samples must be valid"
                )
        if not self.source_states_match(pending.source_q.to(path), path[:, 0]):
            raise V177RuntimeError(
                "measured q0 differs from the proposed source state"
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

        # Only this measurement-updated state becomes actual memory.  The
        # prediction remains reachable solely through the completed proposal.
        self._latest_measurement = measurement
        self._latest_assigned = assigned
        self._latest_update = update
        self._latest_q = final_q.detach().clone()
        self._recent_readback_deltas = path[:, 1:] - path[:, :-1]
        self._recent_readback_errors = (
            path[:, 1:] - pending.actor_output.commands.to(path)
        )
        self._completed_chunks += 1
        self._pending = None
        return update

    def actual_grounded_state(self) -> V177GroundedState:
        """Return the latest measurement-updated state, never a prediction."""

        if self._latest_update is None:
            raise V177RuntimeError("no actual endpoint has been observed")
        return self._latest_update.grounded

    def latest_filter_update(self) -> V177FilterUpdate:
        if self._latest_update is None:
            raise V177RuntimeError("no actual endpoint has been observed")
        return self._latest_update


__all__ = [
    "AnalyticTransportProvider",
    "CommandAdmissionProvider",
    "FISH_SOURCE_INDEX",
    "MAX_SOURCE_STATE_ENCODER_CELL_DELTA",
    "V177PendingChunk",
    "V177RuntimeController",
    "V177RuntimeError",
    "V177RuntimeProposal",
]
