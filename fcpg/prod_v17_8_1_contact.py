"""Training-only contact-event utilities for Prod-V17.8.1.

This module does not participate in inference.  It converts factual five-command
chunks into auditable event labels for sampling and provides an auxiliary loss
that compares the tool orientation of a predicted chunk with the factual chunk.
The Actor still receives only its normal visual, belief, history, task and joint
inputs; event labels and FK callbacks are never model inputs.

All joint values use LeRobot calibrated-position units.  Every Cartesian value
returned by the event FK callback is in metres in a base frame whose +Z axis is
up.  No implicit ``/100`` conversion, degrees conversion, IK, or unit inference
is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


STATE_DIM = 6
COMMAND_STEPS = 5
JOINT_POSITION_UNIT = "lerobot_calibrated_position"
CARTESIAN_POSITION_UNIT = "metre"


class ContactContractError(ValueError):
    """Raised when contact-event training data violate the fixed contract."""


class ContactEvent(str, Enum):
    """Mutually exclusive training-only event labels."""

    VERTICAL_APPROACH = "vertical_approach"
    CLOSE = "close"
    LIFT = "lift"
    TRANSPORT = "transport"
    RELEASE = "release"


CONTACT_EVENTS: tuple[ContactEvent, ...] = tuple(ContactEvent)


class EventFKCallback(Protocol):
    """Vectorized, preprocessing-only FK callback.

    The input is ``[N, 6]`` in :data:`JOINT_POSITION_UNIT`.  The callback must
    return ``(position_m, tool_axis)`` with both arrays shaped ``[N, 3]``.
    ``position_m`` is expressed in :data:`CARTESIAN_POSITION_UNIT`; ``tool_axis``
    is a directed vector whose positive direction points out through the tool.
    """

    def __call__(self, joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


DifferentiableToolAxisCallback = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ContactEventThresholds:
    """Robot/task-independent numerical definition of the five event classes."""

    minimum_vertical_translation_m: float = 0.005
    minimum_transport_translation_m: float = 0.010
    minimum_gripper_change: float = 5.0
    open_gripper_minimum: float = 18.0
    closed_gripper_maximum: float = 18.0
    vertical_to_planar_ratio: float = 1.0
    minimum_downward_tool_cosine: float = 0.90

    def validate(self) -> None:
        values = np.asarray(
            (
                self.minimum_vertical_translation_m,
                self.minimum_transport_translation_m,
                self.minimum_gripper_change,
                self.open_gripper_minimum,
                self.closed_gripper_maximum,
                self.vertical_to_planar_ratio,
                self.minimum_downward_tool_cosine,
            ),
            dtype=np.float64,
        )
        if not bool(np.isfinite(values).all()):
            raise ContactContractError("contact-event thresholds must be finite")
        if self.minimum_vertical_translation_m <= 0.0:
            raise ContactContractError("minimum vertical translation must be positive")
        if self.minimum_transport_translation_m <= 0.0:
            raise ContactContractError("minimum transport translation must be positive")
        if self.minimum_gripper_change <= 0.0:
            raise ContactContractError("minimum gripper change must be positive")
        if self.vertical_to_planar_ratio <= 0.0:
            raise ContactContractError("vertical/planar ratio must be positive")
        if not -1.0 <= self.minimum_downward_tool_cosine <= 1.0:
            raise ContactContractError("minimum downward tool cosine must lie in [-1,1]")


@dataclass(frozen=True)
class ContactEventDiagnostic:
    """Auditable geometry used to assign one training event."""

    event: ContactEvent | None
    displacement_m: tuple[float, float, float]
    vertical_translation_m: float
    planar_translation_m: float
    gripper_start: float
    gripper_end: float
    gripper_change: float
    minimum_downward_tool_cosine: float


@dataclass(frozen=True)
class ContactEventTable:
    """Canonical, de-duplicated contact-event training rows.

    Rows are sorted by ``(episode, source_frame)``.  ``original_indices`` points
    to the first occurrence in the caller's input.  An unclassified factual
    chunk remains in the table with ``event=None`` but is not sampled by the
    balanced event sampler.
    """

    episodes: np.ndarray
    source_frames: np.ndarray
    current_q: np.ndarray
    commands: np.ndarray
    original_indices: np.ndarray
    events: tuple[ContactEvent | None, ...]
    diagnostics: tuple[ContactEventDiagnostic, ...]
    joint_position_unit: str = JOINT_POSITION_UNIT
    cartesian_position_unit: str = CARTESIAN_POSITION_UNIT

    def __len__(self) -> int:
        return int(len(self.episodes))


def _require_units(joint_position_unit: str, cartesian_position_unit: str) -> None:
    if joint_position_unit != JOINT_POSITION_UNIT:
        raise ContactContractError(
            f"joint unit must be {JOINT_POSITION_UNIT!r}, got {joint_position_unit!r}"
        )
    if cartesian_position_unit != CARTESIAN_POSITION_UNIT:
        raise ContactContractError(
            f"Cartesian unit must be {CARTESIAN_POSITION_UNIT!r}, "
            f"got {cartesian_position_unit!r}"
        )


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value).copy()
    result.setflags(write=False)
    return result


def _contact_arrays(
    episodes: Sequence[int] | np.ndarray,
    source_frames: Sequence[int] | np.ndarray,
    current_q: np.ndarray,
    commands: np.ndarray,
    *,
    joint_position_unit: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require_units(joint_position_unit, CARTESIAN_POSITION_UNIT)
    episode = np.asarray(episodes)
    frame = np.asarray(source_frames)
    state = np.asarray(current_q, dtype=np.float64)
    action = np.asarray(commands, dtype=np.float64)
    if episode.ndim != 1 or frame.shape != episode.shape or len(episode) == 0:
        raise ContactContractError("episode/source_frame must be non-empty matching [N]")
    if episode.dtype.kind not in "iu" or frame.dtype.kind not in "iu":
        raise ContactContractError("episode/source_frame must use integer values")
    if bool((episode < 0).any() or (frame < 0).any()):
        raise ContactContractError("episode/source_frame must be non-negative")
    if state.shape != (len(episode), STATE_DIM):
        raise ContactContractError("current_q must be [N,6]")
    if action.shape != (len(episode), COMMAND_STEPS, STATE_DIM):
        raise ContactContractError("commands must be [N,5,6]")
    if not bool(np.isfinite(state).all() and np.isfinite(action).all()):
        raise ContactContractError("current_q and commands must be finite")
    return (
        episode.astype(np.int64, copy=False),
        frame.astype(np.int64, copy=False),
        state,
        action,
    )


def deduplicate_episode_source(
    episodes: Sequence[int] | np.ndarray,
    source_frames: Sequence[int] | np.ndarray,
    current_q: np.ndarray,
    commands: np.ndarray,
    *,
    joint_position_unit: str = JOINT_POSITION_UNIT,
    equality_atol: float = 1.0e-6,
) -> np.ndarray:
    """Return canonical first-occurrence indices for unique episode/frame keys.

    Duplicate causal ordinals are allowed only when their factual current state
    and five-command label agree.  Inconsistent duplicates fail closed instead
    of letting input order choose a target silently.  Returned indices are sorted
    lexicographically by key, so the result is independent of caller row order.
    """

    episode, frame, state, action = _contact_arrays(
        episodes,
        source_frames,
        current_q,
        commands,
        joint_position_unit=joint_position_unit,
    )
    if not np.isfinite(equality_atol) or equality_atol < 0.0:
        raise ContactContractError("duplicate equality tolerance must be finite and non-negative")
    first: dict[tuple[int, int], int] = {}
    for index, raw_key in enumerate(zip(episode.tolist(), frame.tolist(), strict=True)):
        key = (int(raw_key[0]), int(raw_key[1]))
        reference = first.get(key)
        if reference is None:
            first[key] = index
            continue
        if not (
            np.allclose(state[index], state[reference], rtol=0.0, atol=equality_atol)
            and np.allclose(action[index], action[reference], rtol=0.0, atol=equality_atol)
        ):
            raise ContactContractError(
                f"duplicate episode/source_frame {key} has inconsistent factual labels"
            )
    return np.asarray([first[key] for key in sorted(first)], dtype=np.int64)


def _event_from_geometry(
    current_q: np.ndarray,
    commands: np.ndarray,
    positions_m: np.ndarray,
    tool_axes: np.ndarray,
    thresholds: ContactEventThresholds,
) -> ContactEventDiagnostic:
    displacement = positions_m[-1] - positions_m[0]
    vertical = float(displacement[2])
    planar = float(np.linalg.norm(displacement[:2]))
    gripper_start = float(current_q[-1])
    gripper_end = float(commands[-1, -1])
    gripper_change = gripper_end - gripper_start
    # Base +Z is up and the directed positive tool axis points out of the tool.
    downward_cosine = -tool_axes[:, 2]
    minimum_downward = float(np.min(downward_cosine))

    event: ContactEvent | None = None
    # Gripper transitions take precedence because a chunk may also begin lifting
    # or translating after contact within the same five-command window.
    if (
        gripper_start <= thresholds.closed_gripper_maximum
        and gripper_change >= thresholds.minimum_gripper_change
    ):
        event = ContactEvent.RELEASE
    elif (
        gripper_start >= thresholds.open_gripper_minimum
        and gripper_change <= -thresholds.minimum_gripper_change
    ):
        event = ContactEvent.CLOSE
    elif (
        gripper_end <= thresholds.closed_gripper_maximum
        and vertical >= thresholds.minimum_vertical_translation_m
        and vertical >= thresholds.vertical_to_planar_ratio * planar
    ):
        event = ContactEvent.LIFT
    elif (
        gripper_start >= thresholds.open_gripper_minimum
        and vertical <= -thresholds.minimum_vertical_translation_m
        and -vertical >= thresholds.vertical_to_planar_ratio * planar
        and minimum_downward >= thresholds.minimum_downward_tool_cosine
    ):
        event = ContactEvent.VERTICAL_APPROACH
    elif (
        gripper_end <= thresholds.closed_gripper_maximum
        and planar >= thresholds.minimum_transport_translation_m
        and planar >= thresholds.vertical_to_planar_ratio * abs(vertical)
    ):
        event = ContactEvent.TRANSPORT

    return ContactEventDiagnostic(
        event=event,
        displacement_m=tuple(float(value) for value in displacement),
        vertical_translation_m=vertical,
        planar_translation_m=planar,
        gripper_start=gripper_start,
        gripper_end=gripper_end,
        gripper_change=gripper_change,
        minimum_downward_tool_cosine=minimum_downward,
    )


def build_contact_event_table(
    episodes: Sequence[int] | np.ndarray,
    source_frames: Sequence[int] | np.ndarray,
    current_q: np.ndarray,
    commands: np.ndarray,
    fk_callback: EventFKCallback,
    *,
    thresholds: ContactEventThresholds = ContactEventThresholds(),
    joint_position_unit: str = JOINT_POSITION_UNIT,
    cartesian_position_unit: str = CARTESIAN_POSITION_UNIT,
    duplicate_equality_atol: float = 1.0e-6,
) -> ContactEventTable:
    """Deduplicate and classify factual chunks using training-time FK only."""

    _require_units(joint_position_unit, cartesian_position_unit)
    thresholds.validate()
    episode, frame, state, action = _contact_arrays(
        episodes,
        source_frames,
        current_q,
        commands,
        joint_position_unit=joint_position_unit,
    )
    selected = deduplicate_episode_source(
        episode,
        frame,
        state,
        action,
        joint_position_unit=joint_position_unit,
        equality_atol=duplicate_equality_atol,
    )
    unique_state = state[selected]
    unique_action = action[selected]
    trajectories = np.concatenate((unique_state[:, None], unique_action), axis=1)
    flat = np.ascontiguousarray(trajectories.reshape(-1, STATE_DIM))
    try:
        raw_position, raw_axis = fk_callback(flat)
    except Exception as error:  # make the data boundary explicit to callers
        raise ContactContractError("event FK callback failed") from error
    position = np.asarray(raw_position, dtype=np.float64)
    axis = np.asarray(raw_axis, dtype=np.float64)
    expected = (len(flat), 3)
    if position.shape != expected or axis.shape != expected:
        raise ContactContractError("event FK must return two [N,3] arrays")
    if not bool(np.isfinite(position).all() and np.isfinite(axis).all()):
        raise ContactContractError("event FK returned non-finite values")
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    if bool((norm <= 1.0e-12).any()):
        raise ContactContractError("event FK returned a zero tool axis")
    axis = axis / norm
    position = position.reshape(len(selected), COMMAND_STEPS + 1, 3)
    axis = axis.reshape(len(selected), COMMAND_STEPS + 1, 3)
    diagnostics = tuple(
        _event_from_geometry(
            unique_state[index],
            unique_action[index],
            position[index],
            axis[index],
            thresholds,
        )
        for index in range(len(selected))
    )
    return ContactEventTable(
        episodes=_readonly(episode[selected]),
        source_frames=_readonly(frame[selected]),
        current_q=_readonly(unique_state),
        commands=_readonly(unique_action),
        original_indices=_readonly(selected),
        events=tuple(value.event for value in diagnostics),
        diagnostics=diagnostics,
        joint_position_unit=joint_position_unit,
        cartesian_position_unit=cartesian_position_unit,
    )


def contact_event_metrics(table: ContactEventTable) -> dict[str, object]:
    """Return JSON-compatible unique-row coverage; never report exposures."""

    counts: dict[str, int] = {}
    episodes: dict[str, int] = {}
    for event in CONTACT_EVENTS:
        mask = np.asarray([value is event for value in table.events], dtype=bool)
        counts[event.value] = int(mask.sum())
        episodes[event.value] = int(len(np.unique(table.episodes[mask])))
    classified = int(sum(counts.values()))
    total = len(table)
    return {
        "unique_episode_source_rows": total,
        "classified_rows": classified,
        "unclassified_rows": int(total - classified),
        "event_unique_rows": counts,
        "event_unique_episodes": episodes,
        "event_fraction_of_unique_rows": {
            name: (float(count / total) if total else 0.0)
            for name, count in counts.items()
        },
        "joint_position_unit": table.joint_position_unit,
        "cartesian_position_unit": table.cartesian_position_unit,
    }


def event_balanced_epoch_indices(
    table: ContactEventTable,
    *,
    samples_per_event: int,
    seed: int,
    epoch: int = 0,
) -> np.ndarray:
    """Build a deterministic round-robin balanced epoch over five events.

    Every consecutive group of five contains one sample from each event.  Small
    event pools are reshuffled before each repeat.  Rows with ``event=None`` are
    intentionally excluded.  The table must already be key-de-duplicated.
    """

    if not isinstance(samples_per_event, int) or samples_per_event <= 0:
        raise ContactContractError("samples_per_event must be a positive integer")
    if not isinstance(seed, int) or not isinstance(epoch, int) or epoch < 0:
        raise ContactContractError("seed/epoch must be integers and epoch non-negative")
    columns: list[np.ndarray] = []
    for event_index, event in enumerate(CONTACT_EVENTS):
        pool = np.flatnonzero(
            np.asarray([value is event for value in table.events], dtype=bool)
        ).astype(np.int64, copy=False)
        if len(pool) == 0:
            raise ContactContractError(f"balanced sampler has no {event.value} rows")
        generator = np.random.default_rng(
            np.random.SeedSequence((seed, epoch, event_index))
        )
        pieces: list[np.ndarray] = []
        remaining = samples_per_event
        while remaining:
            shuffled = generator.permutation(pool)
            take = min(remaining, len(shuffled))
            pieces.append(shuffled[:take])
            remaining -= take
        columns.append(np.concatenate(pieces))
    matrix = np.stack(columns, axis=1)
    order_generator = np.random.default_rng(np.random.SeedSequence((seed, epoch, 991)))
    for row in range(len(matrix)):
        matrix[row] = matrix[row, order_generator.permutation(len(CONTACT_EVENTS))]
    return matrix.reshape(-1)


class DeterministicEventBalancedSampler(Sampler[int]):
    """PyTorch sampler wrapper around :func:`event_balanced_epoch_indices`."""

    def __init__(
        self,
        table: ContactEventTable,
        *,
        samples_per_event: int,
        seed: int,
    ) -> None:
        self.table = table
        self.samples_per_event = int(samples_per_event)
        self.seed = int(seed)
        self.epoch = 0
        # Validate eagerly so an empty rare event fails before optimization.
        event_balanced_epoch_indices(
            table,
            samples_per_event=self.samples_per_event,
            seed=self.seed,
            epoch=self.epoch,
        )

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ContactContractError("sampler epoch must be a non-negative integer")
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(
            event_balanced_epoch_indices(
                self.table,
                samples_per_event=self.samples_per_event,
                seed=self.seed,
                epoch=self.epoch,
            ).tolist()
        )

    def __len__(self) -> int:
        return self.samples_per_event * len(CONTACT_EVENTS)


@dataclass(frozen=True)
class ToolAxisCosineLoss:
    """Differentiable loss and detached reporting metrics."""

    loss: torch.Tensor
    cosine: torch.Tensor
    valid: torch.Tensor

    def metrics(self) -> dict[str, float | int | str]:
        selected = self.cosine.detach().float()[self.valid.detach()]
        if selected.numel() == 0:
            raise ContactContractError("tool-axis metrics have no valid commands")
        angle = torch.rad2deg(torch.acos(selected.clamp(-1.0, 1.0)))
        return {
            "tool_axis_cosine_loss": float(self.loss.detach().float().cpu()),
            "tool_axis_mean_cosine": float(selected.mean().cpu()),
            "tool_axis_median_cosine": float(selected.median().cpu()),
            "tool_axis_mean_error_deg": float(angle.mean().cpu()),
            "tool_axis_p90_error_deg": float(torch.quantile(angle, 0.9).cpu()),
            "tool_axis_valid_commands": int(selected.numel()),
            "joint_position_unit": JOINT_POSITION_UNIT,
        }


def training_tool_axis_cosine_loss(
    predicted_commands: torch.Tensor,
    factual_commands: torch.Tensor,
    tool_axis_callback: DifferentiableToolAxisCallback,
    *,
    valid: torch.Tensor | None = None,
    joint_position_unit: str = JOINT_POSITION_UNIT,
) -> ToolAxisCosineLoss:
    """Compare predicted and factual tool axes without exposing FK at runtime.

    ``tool_axis_callback`` is called only by this training utility.  It receives
    absolute ``[B,5,6]`` calibrated positions and must return differentiable
    directed axes ``[B,5,3]``.  The factual axes are detached, so gradients flow
    only through the predicted action and the caller's fixed differentiable FK.
    """

    _require_units(joint_position_unit, CARTESIAN_POSITION_UNIT)
    predicted = torch.as_tensor(predicted_commands)
    factual = torch.as_tensor(factual_commands, device=predicted.device)
    if predicted.ndim != 3 or tuple(predicted.shape[1:]) != (
        COMMAND_STEPS,
        STATE_DIM,
    ):
        raise ContactContractError("predicted_commands must be [B,5,6]")
    if factual.shape != predicted.shape:
        raise ContactContractError("factual_commands must match predicted [B,5,6]")
    if not bool(torch.isfinite(predicted).all() and torch.isfinite(factual).all()):
        raise ContactContractError("tool-axis loss commands must be finite")
    predicted_axis = tool_axis_callback(predicted)
    factual_axis = tool_axis_callback(factual.detach()).detach()
    if not isinstance(predicted_axis, torch.Tensor) or not isinstance(
        factual_axis, torch.Tensor
    ):
        raise ContactContractError("differentiable tool-axis callback must return tensors")
    expected = (*predicted.shape[:-1], 3)
    if tuple(predicted_axis.shape) != expected or tuple(factual_axis.shape) != expected:
        raise ContactContractError("tool-axis callback must return [B,5,3]")
    if predicted_axis.device != predicted.device or factual_axis.device != predicted.device:
        raise ContactContractError("tool-axis callback changed tensor device")
    if not bool(torch.isfinite(predicted_axis).all() and torch.isfinite(factual_axis).all()):
        raise ContactContractError("tool-axis callback returned non-finite tensors")
    predicted_norm = torch.linalg.vector_norm(predicted_axis, dim=-1)
    factual_norm = torch.linalg.vector_norm(factual_axis, dim=-1)
    if not bool((predicted_norm > 1.0e-12).all() and (factual_norm > 1.0e-12).all()):
        raise ContactContractError("tool-axis callback returned a zero vector")
    cosine = (
        (predicted_axis / predicted_norm[..., None])
        * (factual_axis / factual_norm[..., None])
    ).sum(dim=-1).clamp(-1.0, 1.0)
    if valid is None:
        active = torch.ones_like(cosine, dtype=torch.bool)
    else:
        active = torch.as_tensor(valid, device=predicted.device)
        if active.dtype is not torch.bool or active.shape != cosine.shape:
            raise ContactContractError("tool-axis valid mask must be bool [B,5]")
    if not bool(active.any()):
        raise ContactContractError("tool-axis valid mask selects no command")
    loss = (1.0 - cosine)[active].mean()
    return ToolAxisCosineLoss(loss=loss, cosine=cosine, valid=active)


__all__ = [
    "CARTESIAN_POSITION_UNIT",
    "COMMAND_STEPS",
    "CONTACT_EVENTS",
    "ContactContractError",
    "ContactEvent",
    "ContactEventDiagnostic",
    "ContactEventTable",
    "ContactEventThresholds",
    "DeterministicEventBalancedSampler",
    "EventFKCallback",
    "JOINT_POSITION_UNIT",
    "STATE_DIM",
    "ToolAxisCosineLoss",
    "build_contact_event_table",
    "contact_event_metrics",
    "deduplicate_episode_source",
    "event_balanced_epoch_indices",
    "training_tool_axis_cosine_loss",
]
