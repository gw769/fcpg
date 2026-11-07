"""Optional per-round Present_Load / Present_Current persistence.

Observability only.  Nothing in this module is a contact trigger, a gate, or
an input to planning.  The live control path already reads ``Present_Position``
(and, during settle traces, Goal / Velocity / Moving).  It does **not** already
sample these two registers, so the values cannot be reused: enabling the flag
costs two extra ``sync_read`` calls.

Those two reads are cheap next to a planning round and non-trivial next to the
30 Hz command period (~33 ms).  A Feetech group-read of six motors on the
1 Mbps bus is one serial transaction; the settle tracer already treats four
such transactions as enough to stretch the loop it measures.  This module
therefore samples **once per planning round**, never inside ``observe()``,
``execute_chunk``, or settle.

A failed or missing register becomes a null.  The read is not allowed to
raise out to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

from fcpg.servo_register_decode import (
    decode_present_current,
    decode_present_load,
    encoding_provenance,
)
from fcpg.so101_kinematics import SO101_MUJOCO_JOINT_NAMES


LOAD_REGISTER = "Present_Load"
CURRENT_REGISTER = "Present_Current"
SERVO_LOAD_REGISTERS = (LOAD_REGISTER, CURRENT_REGISTER)
#: Two group-reads: one for load, one for current.  Not per joint.
SERVO_LOAD_EXTRA_SYNC_READS = 2
SERVO_LOAD_READ_FREQUENCY = "once_per_planning_round"
#: Do not retry.  A missing register must not stall the round.
SERVO_LOAD_SYNC_READ_RETRIES = 0

__all__ = [
    "CURRENT_REGISTER",
    "LOAD_REGISTER",
    "SERVO_LOAD_EXTRA_SYNC_READS",
    "SERVO_LOAD_READ_FREQUENCY",
    "SERVO_LOAD_REGISTERS",
    "SERVO_LOAD_SYNC_READ_RETRIES",
    "ServoLoadTelemetry",
    "assess_physical_load_witness",
    "attach_servo_load_telemetry",
    "collect_servo_load_telemetry",
    "formal_screen_physical_witness_precheck",
    "make_servo_load_telemetry",
    "read_servo_load_registers",
    "servo_load_enabled",
    "servo_load_telemetry_contract",
]


def _decoded_magnitude(value: float, register: str) -> float:
    if register == LOAD_REGISTER:
        return float(decode_present_load(value).magnitude)
    if register == CURRENT_REGISTER:
        return float(decode_present_current(value).magnitude)
    raise ValueError(f"unsupported register {register!r}")


def _decoded_map(
    raw: Mapping[str, float | None], register: str
) -> dict[str, float | None]:
    decoded: dict[str, float | None] = {}
    for name, value in raw.items():
        if value is None:
            decoded[name] = None
            continue
        try:
            if register == LOAD_REGISTER:
                decoded[name] = float(decode_present_load(value).decoded)
            else:
                decoded[name] = float(decode_present_current(value).decoded)
        except (TypeError, ValueError):
            decoded[name] = None
    return decoded


def servo_load_enabled(planner: Any) -> bool:
    """True only when the caller opted in on ``PreparedPlannerConfig``."""

    config = getattr(planner, "config", None)
    return bool(getattr(config, "observer_servo_load", False))


def servo_load_telemetry_contract(*, enabled: bool) -> dict[str, Any]:
    """Self-describing archive field.  ``control_influence`` is always false."""

    return {
        "enabled": bool(enabled),
        "registers": list(SERVO_LOAD_REGISTERS),
        "frequency": SERVO_LOAD_READ_FREQUENCY,
        "extra_sync_reads": SERVO_LOAD_EXTRA_SYNC_READS,
        "control_influence": False,
        "register_encoding": encoding_provenance(),
        "safety_uses": "decoded_magnitude",
    }


def _phase_of(plan: Any) -> str | None:
    phase = getattr(plan, "phase", None)
    if phase is None:
        return None
    value = getattr(phase, "value", phase)
    if value is None:
        return None
    return str(value)


def _round_id_of(plan: Any) -> str:
    guard = getattr(plan, "guard", None)
    round_id = getattr(guard, "round_id", None)
    return "unknown" if round_id is None else str(round_id)


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _joint_map(raw: Any) -> dict[str, float | None]:
    values = raw if isinstance(raw, Mapping) else {}
    parsed: dict[str, float | None] = {}
    for name in SO101_MUJOCO_JOINT_NAMES:
        if name in values:
            parsed[name] = _finite_or_none(values[name])
            continue
        dotted = f"{name}.pos"
        if dotted in values:
            parsed[name] = _finite_or_none(values[dotted])
            continue
        parsed[name] = None
    return parsed


def _constant_register(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    first = values[0]
    return all(value == first for value in values)


def assess_physical_load_witness(
    present_load: Mapping[str, float | None],
    present_current: Mapping[str, float | None],
    *,
    load_error: str | None = None,
    current_error: str | None = None,
) -> tuple[bool, str]:
    """Mark whether a successful register read is usable as physical evidence.

    Observability only.  All-zero or stuck-constant maps stay a successful
    read; they are not a valid physical witness and must not invent an N.
    This verdict never arms, disarms, or bypasses a hardware safety gate.
    """

    if load_error or current_error:
        return False, "read_error"
    load_vals = [
        _decoded_magnitude(float(value), LOAD_REGISTER)
        for value in present_load.values()
        if value is not None
    ]
    current_vals = [
        _decoded_magnitude(float(value), CURRENT_REGISTER)
        for value in present_current.values()
        if value is not None
    ]
    expected = len(SO101_MUJOCO_JOINT_NAMES)
    if len(load_vals) != expected or len(current_vals) != expected:
        return False, "missing_registers"
    if all(value == 0.0 for value in load_vals) and all(
        value == 0.0 for value in current_vals
    ):
        return False, "all_zero"
    if _constant_register(load_vals) or _constant_register(current_vals):
        return False, "constant_register"
    return True, "ok"


def _read_register(bus: Any, register: str) -> tuple[dict[str, float | None], str | None]:
    try:
        raw = bus.sync_read(register, num_retry=SERVO_LOAD_SYNC_READ_RETRIES)
    except Exception as exc:
        return (
            {name: None for name in SO101_MUJOCO_JOINT_NAMES},
            f"{type(exc).__name__}: {exc}",
        )
    return _joint_map(raw), None


@dataclass(frozen=True)
class ServoLoadTelemetry:
    """One planning-round load/current snapshot.  Never used for control."""

    round_id: str
    phase: str | None
    timestamp_s: float
    wall_time_unix_s: float
    present_load: dict[str, float | None]
    present_current: dict[str, float | None]
    load_error: str | None
    current_error: str | None
    extra_sync_reads: int
    read_duration_ms: float | None
    frequency: str = SERVO_LOAD_READ_FREQUENCY
    control_influence: Literal[False] = False
    provenance: str = "bus.sync_read"
    physical_witness_valid: bool = False
    physical_witness_reason: str | None = "unassessed"
    present_load_decoded: dict[str, float | None] | None = None
    present_current_decoded: dict[str, float | None] | None = None
    register_encoding: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "phase": self.phase,
            "timestamp_s": self.timestamp_s,
            "wall_time_unix_s": self.wall_time_unix_s,
            "present_load": dict(self.present_load),
            "present_current": dict(self.present_current),
            "present_load_decoded": dict(self.present_load_decoded or {}),
            "present_current_decoded": dict(self.present_current_decoded or {}),
            "register_encoding": dict(
                self.register_encoding or encoding_provenance()
            ),
            "load_error": self.load_error,
            "current_error": self.current_error,
            "extra_sync_reads": int(self.extra_sync_reads),
            "read_duration_ms": self.read_duration_ms,
            "frequency": self.frequency,
            "control_influence": False,
            "provenance": self.provenance,
            "physical_witness_valid": bool(self.physical_witness_valid),
            "physical_witness_reason": self.physical_witness_reason,
        }


def make_servo_load_telemetry(
    *,
    present_load: Mapping[str, Any] | None,
    present_current: Mapping[str, Any] | None,
    timestamp_s: float,
    wall_time_unix_s: float,
    read_duration_ms: float | None,
    round_id: str,
    phase: str | None,
    load_error: str | None = None,
    current_error: str | None = None,
    extra_sync_reads: int = SERVO_LOAD_EXTRA_SYNC_READS,
    frequency: str = SERVO_LOAD_READ_FREQUENCY,
    provenance: str = "bus.sync_read",
) -> ServoLoadTelemetry:
    """Build the shared persist contract.  Does not interpret values for control."""

    load_map = _joint_map(present_load)
    current_map = _joint_map(present_current)
    valid, reason = assess_physical_load_witness(
        load_map,
        current_map,
        load_error=load_error,
        current_error=current_error,
    )
    return ServoLoadTelemetry(
        round_id=str(round_id),
        phase=phase,
        timestamp_s=float(timestamp_s),
        wall_time_unix_s=float(wall_time_unix_s),
        present_load=load_map,
        present_current=current_map,
        load_error=None if load_error is None else str(load_error),
        current_error=None if current_error is None else str(current_error),
        extra_sync_reads=int(extra_sync_reads),
        read_duration_ms=read_duration_ms,
        frequency=frequency,
        provenance=str(provenance),
        physical_witness_valid=valid,
        physical_witness_reason=reason,
        present_load_decoded=_decoded_map(load_map, LOAD_REGISTER),
        present_current_decoded=_decoded_map(current_map, CURRENT_REGISTER),
        register_encoding=encoding_provenance(),
    )


def formal_screen_physical_witness_precheck(
    telemetry: Mapping[str, Any] | ServoLoadTelemetry,
) -> dict[str, Any]:
    """Formal-screen exposure of load-witness validity.  Not a safety bypass."""

    payload = (
        telemetry.to_dict()
        if isinstance(telemetry, ServoLoadTelemetry)
        else dict(telemetry)
    )
    reason = payload.get("physical_witness_reason")
    return {
        "physical_witness_valid": bool(payload.get("physical_witness_valid")),
        "physical_witness_reason": None if reason is None else str(reason),
        "bypasses_hardware_safety": False,
        "used_as_contact_trigger": False,
        "control_influence": False,
    }


def absent_record(
    *,
    round_id: str,
    phase: str | None,
    timestamp_s: float,
    wall_time_unix_s: float,
    error: str,
    extra_sync_reads: int = 0,
    read_duration_ms: float | None = None,
) -> ServoLoadTelemetry:
    return make_servo_load_telemetry(
        present_load=None,
        present_current=None,
        timestamp_s=float(timestamp_s),
        wall_time_unix_s=float(wall_time_unix_s),
        read_duration_ms=read_duration_ms,
        round_id=round_id,
        phase=phase,
        load_error=error,
        current_error=error,
        extra_sync_reads=int(extra_sync_reads),
        provenance="absent",
    )


def read_servo_load_registers(bus: Any) -> dict[str, Any]:
    """Read both registers.  Never raises.  Does not interpret the values."""

    import time

    started = time.perf_counter()
    present_load, load_error = _read_register(bus, LOAD_REGISTER)
    present_current, current_error = _read_register(bus, CURRENT_REGISTER)
    return {
        "present_load": present_load,
        "present_current": present_current,
        "load_error": load_error,
        "current_error": current_error,
        "extra_sync_reads": SERVO_LOAD_EXTRA_SYNC_READS,
        "read_duration_ms": (time.perf_counter() - started) * 1000.0,
        "timestamp_s": started,
        "wall_time_unix_s": time.time(),
    }


def collect_servo_load_telemetry(
    io: Any,
    *,
    plan: Any | None = None,
    round_id: str | None = None,
    phase: str | None = None,
) -> ServoLoadTelemetry:
    """Sample the bus behind ``io``.  Missing hardware becomes an absent record."""

    import time

    resolved_round = (
        str(round_id) if round_id is not None else _round_id_of(plan)
    )
    resolved_phase = phase if phase is not None else _phase_of(plan)
    stamp = time.perf_counter()
    wall = time.time()

    try:
        reader = getattr(io, "read_servo_load_registers", None)
        if callable(reader):
            raw = reader()
        else:
            bus = getattr(getattr(io, "follower", None), "bus", None)
            if bus is None:
                return absent_record(
                    round_id=resolved_round,
                    phase=resolved_phase,
                    timestamp_s=stamp,
                    wall_time_unix_s=wall,
                    error="no_bus",
                )
            raw = read_servo_load_registers(bus)
    except Exception as exc:
        return absent_record(
            round_id=resolved_round,
            phase=resolved_phase,
            timestamp_s=stamp,
            wall_time_unix_s=wall,
            error=f"{type(exc).__name__}: {exc}",
        )

    if not isinstance(raw, Mapping):
        return absent_record(
            round_id=resolved_round,
            phase=resolved_phase,
            timestamp_s=stamp,
            wall_time_unix_s=wall,
            error="invalid_sample",
        )
    return make_servo_load_telemetry(
        present_load=raw.get("present_load"),
        present_current=raw.get("present_current"),
        timestamp_s=float(raw.get("timestamp_s", stamp)),
        wall_time_unix_s=float(raw.get("wall_time_unix_s", wall)),
        read_duration_ms=_finite_or_none(raw.get("read_duration_ms")),
        round_id=resolved_round,
        phase=resolved_phase,
        load_error=(
            None if raw.get("load_error") is None else str(raw.get("load_error"))
        ),
        current_error=(
            None
            if raw.get("current_error") is None
            else str(raw.get("current_error"))
        ),
        extra_sync_reads=int(
            raw.get("extra_sync_reads", SERVO_LOAD_EXTRA_SYNC_READS)
        ),
        provenance=str(raw.get("provenance", "bus.sync_read")),
    )


def attach_servo_load_telemetry(plan: Any, io: Any) -> dict[str, Any]:
    """Record telemetry on ``plan`` after ``plan()`` has already returned.

    Sets ``plan.servo_load_telemetry``.  Does not read or write
    ``selected_chunk``, ``executable``, phase, or any gate field.  Never
    raises.
    """

    try:
        record = collect_servo_load_telemetry(io, plan=plan)
        payload = record.to_dict()
    except Exception as exc:
        import time

        payload = absent_record(
            round_id=_round_id_of(plan),
            phase=_phase_of(plan),
            timestamp_s=time.perf_counter(),
            wall_time_unix_s=time.time(),
            error=f"unexpected:{type(exc).__name__}: {exc}",
        ).to_dict()
    try:
        plan.servo_load_telemetry = payload
    except Exception:
        pass
    return payload
