"""Settled-observation clock contract for Prod-V17.8.1.

The five command-time readbacks and the later settled observation are two
different measurements.  This module keeps them separate all the way through
history and innovation construction:

* ``timed_q_path`` is always ``q0 ... q5`` sampled on the command clock;
* ``settled_q`` is the post-wait endpoint;
* ``settle_delta`` is ``settled_q - timed_q_path[:, -1]``; and
* ``settle_duration_s`` covers only the interval after the timed ``q5``.

In particular, no helper in this module overwrites ``timed_q_path[:, -1]``
with the settled endpoint.  All stored observations are actual readbacks;
predicted paths never enter actual history.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, TypeAlias

import torch

from fcpg.prod_v17_8 import (
    COMMAND_STEPS,
    HISTORY_STEPS,
    PATH_STEPS,
    STATE_DIM,
    VIEWS,
    VIT_DIM,
    V178ContractError,
    V178LeWMPrediction,
    V178RawHistory,
    append_raw_history,
    empty_raw_history,
)


# Every q/path/delta/residual tensor in this module uses the same unit exposed
# by the LeRobot hardware boundary.  It is intentionally not named "degrees":
# different calibrated actuators need not share an angular interpretation.
POSITION_UNIT: Final[str] = "lerobot_calibrated_position"
DEFAULT_SOURCE_ATOL_POSITION: Final[float] = 1.0e-4


class V1781ClockError(V178ContractError):
    """Raised when the V17.8.1 observation clock contract is violated."""


def _floating_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise V1781ClockError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != tuple(shape):
        raise V1781ClockError(
            f"{name} must have shape {shape}, got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise V1781ClockError(f"{name} must have a floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise V1781ClockError(f"{name} must be finite")
    return value


def _bool_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise V1781ClockError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != tuple(shape):
        raise V1781ClockError(
            f"{name} must have shape {shape}, got {tuple(value.shape)}"
        )
    if value.dtype != torch.bool:
        raise V1781ClockError(f"{name} must have dtype torch.bool")
    return value


def _same_storage_contract(
    reference_name: str,
    reference: torch.Tensor,
    name: str,
    value: torch.Tensor,
) -> None:
    if value.device != reference.device:
        raise V1781ClockError(
            f"{name} and {reference_name} must be on the same device"
        )
    if value.dtype != reference.dtype:
        raise V1781ClockError(
            f"{name} and {reference_name} must have the same dtype"
        )


def _duration_vector(
    name: str,
    value: float | torch.Tensor,
    *,
    batch: int,
    like: torch.Tensor,
    strictly_positive: bool,
) -> torch.Tensor:
    if isinstance(value, bool):
        raise V1781ClockError(f"{name} must be a real duration")
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            duration = value.to(device=like.device, dtype=like.dtype).expand(batch)
        elif tuple(value.shape) == (batch,):
            duration = value.to(device=like.device, dtype=like.dtype)
        else:
            raise V1781ClockError(
                f"{name} must be scalar or shape ({batch},), got {tuple(value.shape)}"
            )
    else:
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise V1781ClockError(f"{name} must be a real duration") from exc
        duration = like.new_full((batch,), scalar)

    if not bool(torch.isfinite(duration).all().item()):
        raise V1781ClockError(f"{name} must be finite")
    if strictly_positive:
        valid = duration > 0.0
        requirement = "positive"
    else:
        valid = duration >= 0.0
        requirement = "non-negative"
    if not bool(valid.all().item()):
        raise V1781ClockError(f"{name} must be {requirement}")
    return duration


def _validate_atol(source_atol_position: float) -> float:
    tolerance = float(source_atol_position)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise V1781ClockError(
            "source_atol_position must be finite and non-negative"
        )
    return tolerance


@dataclass(frozen=True)
class V1781SettledObservation:
    """One completed transition with command-clock and settle-clock data.

    ``timed_duration_s`` is the duration from timed ``q0`` through timed
    ``q5``.  ``settle_duration_s`` starts at timed ``q5`` and ends when
    ``settled_q`` is sampled.
    """

    timed_q_path: torch.Tensor
    settled_q: torch.Tensor
    settle_delta: torch.Tensor
    timed_duration_s: torch.Tensor
    settle_duration_s: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.timed_q_path, torch.Tensor) or self.timed_q_path.ndim != 3:
            raise V1781ClockError(
                "timed_q_path must be a [B,6,6] floating tensor"
            )
        batch = int(self.timed_q_path.shape[0])
        if batch <= 0:
            raise V1781ClockError("observation batch must be positive")
        path = _floating_tensor(
            "timed_q_path",
            self.timed_q_path,
            (batch, PATH_STEPS, STATE_DIM),
        )
        settled = _floating_tensor(
            "settled_q", self.settled_q, (batch, STATE_DIM)
        )
        delta = _floating_tensor(
            "settle_delta", self.settle_delta, (batch, STATE_DIM)
        )
        timed_duration = _floating_tensor(
            "timed_duration_s", self.timed_duration_s, (batch,)
        )
        settle_duration = _floating_tensor(
            "settle_duration_s", self.settle_duration_s, (batch,)
        )
        for name, value in (
            ("settled_q", settled),
            ("settle_delta", delta),
            ("timed_duration_s", timed_duration),
            ("settle_duration_s", settle_duration),
        ):
            _same_storage_contract("timed_q_path", path, name, value)
        if not bool((timed_duration > 0.0).all().item()):
            raise V1781ClockError("timed_duration_s must be positive")
        if not bool((settle_duration >= 0.0).all().item()):
            raise V1781ClockError("settle_duration_s must be non-negative")
        expected_delta = settled - path[:, -1]
        if not torch.allclose(delta, expected_delta, atol=1.0e-6, rtol=0.0):
            raise V1781ClockError(
                "settle_delta must equal settled_q - timed_q_path[:, -1]"
            )

        # Actual observations are immutable, gradient-free clock records.
        object.__setattr__(self, "timed_q_path", path.detach().clone())
        object.__setattr__(self, "settled_q", settled.detach().clone())
        object.__setattr__(self, "settle_delta", delta.detach().clone())
        object.__setattr__(self, "timed_duration_s", timed_duration.detach().clone())
        object.__setattr__(
            self, "settle_duration_s", settle_duration.detach().clone()
        )

    @property
    def total_duration_s(self) -> torch.Tensor:
        """Total q0-to-settled duration, without conflating the two clocks."""

        return self.timed_duration_s + self.settle_duration_s


def make_settled_observation(
    timed_q_path: torch.Tensor,
    settled_q: torch.Tensor,
    *,
    timed_duration_s: float | torch.Tensor,
    settle_duration_s: float | torch.Tensor,
) -> V1781SettledObservation:
    """Build a validated observation without mutating the timed path."""

    if not isinstance(timed_q_path, torch.Tensor) or timed_q_path.ndim != 3:
        raise V1781ClockError("timed_q_path must be a [B,6,6] floating tensor")
    batch = int(timed_q_path.shape[0])
    path = _floating_tensor(
        "timed_q_path", timed_q_path, (batch, PATH_STEPS, STATE_DIM)
    )
    settled = _floating_tensor("settled_q", settled_q, (batch, STATE_DIM))
    _same_storage_contract("timed_q_path", path, "settled_q", settled)
    timed_duration = _duration_vector(
        "timed_duration_s",
        timed_duration_s,
        batch=batch,
        like=path,
        strictly_positive=True,
    )
    settle_duration = _duration_vector(
        "settle_duration_s",
        settle_duration_s,
        batch=batch,
        like=path,
        strictly_positive=False,
    )
    return V1781SettledObservation(
        timed_q_path=path,
        settled_q=settled,
        settle_delta=settled - path[:, -1],
        timed_duration_s=timed_duration,
        settle_duration_s=settle_duration,
    )


@dataclass(frozen=True)
class V1781StateInnovation:
    """Actual-minus-predicted state residuals on both observation clocks."""

    timed_path_residual: torch.Tensor
    timed_endpoint_residual: torch.Tensor
    settled_endpoint_residual: torch.Tensor
    settle_delta: torch.Tensor
    timed_duration_s: torch.Tensor
    settle_duration_s: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.timed_path_residual, torch.Tensor)
            or self.timed_path_residual.ndim != 3
        ):
            raise V1781ClockError(
                "timed_path_residual must be a [B,6,6] floating tensor"
            )
        batch = int(self.timed_path_residual.shape[0])
        path_residual = _floating_tensor(
            "timed_path_residual",
            self.timed_path_residual,
            (batch, PATH_STEPS, STATE_DIM),
        )
        timed_endpoint = _floating_tensor(
            "timed_endpoint_residual",
            self.timed_endpoint_residual,
            (batch, STATE_DIM),
        )
        settled_endpoint = _floating_tensor(
            "settled_endpoint_residual",
            self.settled_endpoint_residual,
            (batch, STATE_DIM),
        )
        delta = _floating_tensor(
            "settle_delta", self.settle_delta, (batch, STATE_DIM)
        )
        timed_duration = _floating_tensor(
            "timed_duration_s", self.timed_duration_s, (batch,)
        )
        settle_duration = _floating_tensor(
            "settle_duration_s", self.settle_duration_s, (batch,)
        )
        for name, value in (
            ("timed_endpoint_residual", timed_endpoint),
            ("settled_endpoint_residual", settled_endpoint),
            ("settle_delta", delta),
            ("timed_duration_s", timed_duration),
            ("settle_duration_s", settle_duration),
        ):
            _same_storage_contract(
                "timed_path_residual", path_residual, name, value
            )
        if not torch.allclose(
            timed_endpoint,
            path_residual[:, -1],
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise V1781ClockError(
                "timed_endpoint_residual must be timed_path_residual[:, -1]"
            )
        if not torch.allclose(
            settled_endpoint,
            timed_endpoint + delta,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise V1781ClockError(
                "settled endpoint residual must equal timed endpoint residual + settle delta"
            )
        if not bool((timed_duration > 0.0).all().item()):
            raise V1781ClockError("timed_duration_s must be positive")
        if not bool((settle_duration >= 0.0).all().item()):
            raise V1781ClockError("settle_duration_s must be non-negative")

        # Preserve gradients from residuals to a predicted path while still
        # returning tensors that do not alias caller-owned storage.
        object.__setattr__(self, "timed_path_residual", path_residual.clone())
        object.__setattr__(self, "timed_endpoint_residual", timed_endpoint.clone())
        object.__setattr__(
            self, "settled_endpoint_residual", settled_endpoint.clone()
        )
        object.__setattr__(self, "settle_delta", delta.clone())
        object.__setattr__(self, "timed_duration_s", timed_duration.clone())
        object.__setattr__(self, "settle_duration_s", settle_duration.clone())


PredictionPath: TypeAlias = torch.Tensor | V178LeWMPrediction


def state_innovation_from_observation(
    predicted: PredictionPath,
    observation: V1781SettledObservation,
    *,
    source_atol_position: float = DEFAULT_SOURCE_ATOL_POSITION,
) -> V1781StateInnovation:
    """Return actual-minus-predicted residuals without changing either input.

    The prediction and command-clock observation must share the same ``q0``.
    This check prevents an Hh prediction from being paired with an observation
    from a different transition.  The settled endpoint is compared with the
    predicted timed ``q5`` but remains a separate residual; it is never written
    into the timed path.
    """

    if isinstance(predicted, V178LeWMPrediction):
        predicted_q_path = predicted.predicted_q_path
    elif isinstance(predicted, torch.Tensor):
        predicted_q_path = predicted
    else:
        raise V1781ClockError(
            "predicted must be a q-path tensor or V178LeWMPrediction"
        )
    batch = int(observation.timed_q_path.shape[0])
    prediction = _floating_tensor(
        "predicted_q_path",
        predicted_q_path,
        (batch, PATH_STEPS, STATE_DIM),
    )
    _same_storage_contract(
        "timed_q_path", observation.timed_q_path, "predicted_q_path", prediction
    )
    tolerance = _validate_atol(source_atol_position)
    source_error = (
        prediction[:, 0] - observation.timed_q_path[:, 0]
    ).abs()
    if not bool((source_error <= tolerance).all().item()):
        raise V1781ClockError(
            "predicted_q_path q0 does not match the observation transition q0"
        )

    timed_residual = observation.timed_q_path - prediction
    timed_endpoint_residual = timed_residual[:, -1]
    settled_endpoint_residual = observation.settled_q - prediction[:, -1]
    return V1781StateInnovation(
        timed_path_residual=timed_residual,
        timed_endpoint_residual=timed_endpoint_residual,
        settled_endpoint_residual=settled_endpoint_residual,
        settle_delta=observation.settle_delta,
        timed_duration_s=observation.timed_duration_s,
        settle_duration_s=observation.settle_duration_s,
    )


def _clone_raw_history(raw: V178RawHistory, *, detach: bool) -> V178RawHistory:
    def copy(value: torch.Tensor) -> torch.Tensor:
        return value.detach().clone() if detach else value.clone()

    return V178RawHistory(
        commands=copy(raw.commands),
        paths=copy(raw.paths),
        scene_pooled=copy(raw.scene_pooled),
        valid=raw.valid.detach().clone(),
    )


@dataclass(frozen=True)
class V1781SettledHistory:
    """Three completed actual transitions plus their post-q5 settle records."""

    raw: V178RawHistory
    settled_q: torch.Tensor
    settle_delta: torch.Tensor
    timed_duration_s: torch.Tensor
    settle_duration_s: torch.Tensor
    source_atol_position: float = DEFAULT_SOURCE_ATOL_POSITION

    def __post_init__(self) -> None:
        if not isinstance(self.raw, V178RawHistory):
            raise V1781ClockError("raw must be a V178RawHistory")
        if not isinstance(self.raw.valid, torch.Tensor) or self.raw.valid.ndim != 2:
            raise V1781ClockError("raw.valid must be [B,3]")
        batch = int(self.raw.valid.shape[0])
        if batch <= 0:
            raise V1781ClockError("history batch must be positive")
        commands = _floating_tensor(
            "raw.commands",
            self.raw.commands,
            (batch, HISTORY_STEPS, COMMAND_STEPS, STATE_DIM),
        )
        paths = _floating_tensor(
            "raw.paths",
            self.raw.paths,
            (batch, HISTORY_STEPS, PATH_STEPS, STATE_DIM),
        )
        scene = _floating_tensor(
            "raw.scene_pooled",
            self.raw.scene_pooled,
            (batch, HISTORY_STEPS, VIEWS, VIT_DIM),
        )
        valid = _bool_tensor(
            "raw.valid", self.raw.valid, (batch, HISTORY_STEPS)
        )
        for name, value in (("raw.commands", commands), ("raw.scene_pooled", scene)):
            _same_storage_contract("raw.paths", paths, name, value)
        if valid.device != paths.device:
            raise V1781ClockError("raw.valid and raw.paths must be on the same device")
        # History is left-padded: FFF, FFT, FTT and TTT are the only masks.
        if bool((valid[:, :-1] & ~valid[:, 1:]).any().item()):
            raise V1781ClockError("raw.valid must be a left-padded valid suffix")

        settled = _floating_tensor(
            "settled_q",
            self.settled_q,
            (batch, HISTORY_STEPS, STATE_DIM),
        )
        delta = _floating_tensor(
            "settle_delta",
            self.settle_delta,
            (batch, HISTORY_STEPS, STATE_DIM),
        )
        timed_duration = _floating_tensor(
            "timed_duration_s",
            self.timed_duration_s,
            (batch, HISTORY_STEPS),
        )
        settle_duration = _floating_tensor(
            "settle_duration_s",
            self.settle_duration_s,
            (batch, HISTORY_STEPS),
        )
        for name, value in (
            ("settled_q", settled),
            ("settle_delta", delta),
            ("timed_duration_s", timed_duration),
            ("settle_duration_s", settle_duration),
        ):
            _same_storage_contract("raw.paths", paths, name, value)

        invalid = ~valid
        for name, value in (
            ("settled_q", settled),
            ("settle_delta", delta),
            ("timed_duration_s", timed_duration),
            ("settle_duration_s", settle_duration),
        ):
            expanded_invalid = invalid
            while expanded_invalid.ndim < value.ndim:
                expanded_invalid = expanded_invalid.unsqueeze(-1)
            if bool((value.masked_select(expanded_invalid) != 0.0).any().item()):
                raise V1781ClockError(f"padded {name} values must be zero")
        if bool((timed_duration.masked_select(valid) <= 0.0).any().item()):
            raise V1781ClockError("valid timed durations must be positive")
        if bool((settle_duration.masked_select(valid) < 0.0).any().item()):
            raise V1781ClockError("valid settle durations must be non-negative")

        expected_delta = settled - paths[:, :, -1]
        expanded_valid = valid[:, :, None]
        if not torch.allclose(
            delta.masked_select(expanded_valid),
            expected_delta.masked_select(expanded_valid),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise V1781ClockError(
                "valid settle_delta must equal settled_q - raw.paths q5"
            )
        source_tolerance = _validate_atol(self.source_atol_position)
        for index in range(1, HISTORY_STEPS):
            pair_valid = valid[:, index - 1] & valid[:, index]
            if bool(pair_valid.any().item()):
                causal_error = (
                    paths[:, index, 0] - settled[:, index - 1]
                ).abs()
                if not bool(
                    (causal_error[pair_valid] <= source_tolerance).all().item()
                ):
                    raise V1781ClockError(
                        "each timed q0 must follow the preceding settled endpoint"
                    )

        # History is an actual-data record and never carries gradients.
        object.__setattr__(self, "raw", _clone_raw_history(self.raw, detach=True))
        object.__setattr__(self, "settled_q", settled.detach().clone())
        object.__setattr__(self, "settle_delta", delta.detach().clone())
        object.__setattr__(self, "timed_duration_s", timed_duration.detach().clone())
        object.__setattr__(
            self, "settle_duration_s", settle_duration.detach().clone()
        )
        object.__setattr__(self, "source_atol_position", source_tolerance)


def empty_settled_history(
    batch: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    source_atol_position: float = DEFAULT_SOURCE_ATOL_POSITION,
) -> V1781SettledHistory:
    """Return an explicit H0 history with three empty settle slots."""

    if isinstance(batch, bool) or int(batch) != batch or int(batch) <= 0:
        raise V1781ClockError("batch must be a positive integer")
    if not dtype.is_floating_point:
        raise V1781ClockError("history dtype must be floating")
    count = int(batch)
    raw = empty_raw_history(count, device=device, dtype=dtype)
    return V1781SettledHistory(
        raw=raw,
        settled_q=torch.zeros(
            count, HISTORY_STEPS, STATE_DIM, device=device, dtype=dtype
        ),
        settle_delta=torch.zeros(
            count, HISTORY_STEPS, STATE_DIM, device=device, dtype=dtype
        ),
        timed_duration_s=torch.zeros(
            count, HISTORY_STEPS, device=device, dtype=dtype
        ),
        settle_duration_s=torch.zeros(
            count, HISTORY_STEPS, device=device, dtype=dtype
        ),
        source_atol_position=_validate_atol(source_atol_position),
    )


def append_settled_history(
    history: V1781SettledHistory,
    commands: torch.Tensor,
    observation: V1781SettledObservation,
    scene_pooled: torch.Tensor,
    *,
    source_atol_position: float = DEFAULT_SOURCE_ATOL_POSITION,
) -> V1781SettledHistory:
    """Append one completed transition while preserving both clocks.

    For H1 and later, the new timed ``q0`` must equal the newest prior
    ``settled_q``.  Matching the previous timed ``q5`` is insufficient when a
    non-zero settle delta was observed.
    """

    if not isinstance(history, V1781SettledHistory):
        raise V1781ClockError("history must be a V1781SettledHistory")
    if not isinstance(observation, V1781SettledObservation):
        raise V1781ClockError("observation must be a V1781SettledObservation")
    batch = int(history.raw.valid.shape[0])
    if int(observation.timed_q_path.shape[0]) != batch:
        raise V1781ClockError("observation batch does not match history batch")
    command_tensor = _floating_tensor(
        "commands", commands, (batch, COMMAND_STEPS, STATE_DIM)
    )
    scene_tensor = _floating_tensor(
        "scene_pooled", scene_pooled, (batch, VIEWS, VIT_DIM)
    )
    for name, value in (
        ("commands", command_tensor),
        ("scene_pooled", scene_tensor),
        ("observation.timed_q_path", observation.timed_q_path),
    ):
        _same_storage_contract("history.raw.paths", history.raw.paths, name, value)

    tolerance = _validate_atol(source_atol_position)
    has_previous = history.raw.valid.any(dim=1)
    if bool(has_previous.any().item()):
        source_error = (
            observation.timed_q_path[:, 0] - history.settled_q[:, -1]
        ).abs()
        if not bool((source_error[has_previous] <= tolerance).all().item()):
            raise V1781ClockError(
                "new timed q0 must match the newest previous settled endpoint"
            )

    raw = append_raw_history(
        history.raw,
        command_tensor,
        observation.timed_q_path,
        scene_tensor,
    )
    return V1781SettledHistory(
        raw=raw,
        settled_q=torch.cat(
            (history.settled_q[:, 1:], observation.settled_q[:, None]), dim=1
        ),
        settle_delta=torch.cat(
            (history.settle_delta[:, 1:], observation.settle_delta[:, None]),
            dim=1,
        ),
        timed_duration_s=torch.cat(
            (
                history.timed_duration_s[:, 1:],
                observation.timed_duration_s[:, None],
            ),
            dim=1,
        ),
        settle_duration_s=torch.cat(
            (
                history.settle_duration_s[:, 1:],
                observation.settle_duration_s[:, None],
            ),
            dim=1,
        ),
        source_atol_position=max(
            float(history.source_atol_position), tolerance
        ),
    )


def actor_history_from_settled(
    history: V1781SettledHistory,
) -> V178RawHistory:
    """Return V17.8 Actor history without altering its timed paths."""

    if not isinstance(history, V1781SettledHistory):
        raise V1781ClockError("history must be a V1781SettledHistory")
    return _clone_raw_history(history.raw, detach=True)


__all__ = [
    "DEFAULT_SOURCE_ATOL_POSITION",
    "POSITION_UNIT",
    "V1781ClockError",
    "V1781SettledHistory",
    "V1781SettledObservation",
    "V1781StateInnovation",
    "actor_history_from_settled",
    "append_settled_history",
    "empty_settled_history",
    "make_settled_observation",
    "state_innovation_from_observation",
]
