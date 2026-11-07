"""Cold-start safe energization for the SO-101 follower.

WHY THIS EXISTS -- a hazard measured on the real rig, not a hypothetical.

At cold start this arm's ``Goal_Position`` registers read **raw tick 0 on all six
motors** while ``Present_Position`` reads 863..3152 ticks.  ``Goal_Position`` is
written only by :meth:`SO101Follower.send_action`; ``SO101Follower.connect()``
ends in ``configure()``, whose ``with self.bus.torque_disabled():`` block
re-enables torque in its ``finally:`` clause and never writes a goal.  So a cold
autonomous startup that calls ``connect()`` energizes every joint against a
stale zero target and drives it toward tick 0 -- into the table and the hard
stops.

THE HAZARD IS INVISIBLE THROUGH NORMALIZED READS.  ``MotorsBus._normalize``
clamps with ``bounded_val = min(max_, max(min_, val))``, so raw 0 surfaces as
each joint's calibrated floor.  On this rig that reads back as
``[-100, -100, -100, -100, -100, 0]`` -- exactly ``SO101_POSITION_LOWER``, i.e.
indistinguishable from a legitimate in-range command.  Every classification in
this module therefore reasons on RAW register values read with
``normalize=False``.

WHY THE SEED IS WRITTEN RAW.  ``_unnormalize`` converts with ``int(...)``, which
truncates rather than rounds, so a raw -> normalized -> raw seed can land a tick
away from where the arm actually is.  Reading raw and writing that same raw
value back is exact: ``sync_read`` applies ``_decode_sign`` then optionally
``_normalize``, and ``sync_write`` applies optionally ``_unnormalize`` then
``_encode_sign``, so with ``normalize=False`` on both sides the sign coding
composes to identity.  That makes step 7 of the startup contract an integer
equality check instead of a float tolerance.

This module is deliberately bus-level and duck-typed: it takes any object
exposing ``sync_read``/``sync_write``/``write``/``disable_torque``/
``enable_torque``/``is_connected``, so the cold-start regression test can drive
it with a fake bus and no hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Protocol, Sequence

from fcpg.so101_kinematics import SO101_MUJOCO_JOINT_NAMES

#: Register write value used by the rig's established safe-startup precedent
#: (``scripts/recover_so101_task_view.py``): a low acceleration so that the
#: first commanded motion after energizing cannot be violent.
DEFAULT_ACCELERATION = 20

#: Retry counts matching the rig's existing conventions.
DEFAULT_READ_RETRY = 4
DEFAULT_WRITE_RETRY = 4

#: Seconds to wait after enabling torque before measuring whether the arm moved.
#: Long enough that a servo driving at ``DEFAULT_ACCELERATION`` toward a wrong
#: goal would have produced visible displacement; short enough that it cannot
#: travel far if it is.
DEFAULT_POST_ENERGIZE_SETTLE_S = 0.3

#: Maximum permitted displacement, in NORMALIZED units, between the state read
#: immediately before enabling torque and the state read just after.
#:
#: This is not a new scientific threshold.  It reuses the semantics of
#: ``POSE_DISTINCT_TOLERANCE_DEG`` in
#: ``scripts/phase_2p5_validate_calibration_physical.py``: two joint states
#: within 1.0 L-inf are *the same pose*.  So "the arm did not move while
#: energizing" is exactly "the post-energize state is not a distinct pose".
DEFAULT_MAX_STARTUP_JUMP = 1.0

#: A stale-goal classification is only meaningful against raw values.  Raw 0 is
#: the specific cold-start signature measured on this rig, but the guard seeds
#: unconditionally, so this constant only labels the report.
COLD_START_RAW_GOAL_SENTINEL = 0


class MotorBusLike(Protocol):
    """The subset of ``MotorsBus`` this guard uses."""

    is_connected: bool

    def connect(self) -> None: ...
    def sync_read(
        self,
        data_name: str,
        motors: Any = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Any]: ...
    def sync_write(
        self,
        data_name: str,
        values: Any,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None: ...
    def write(
        self, data_name: str, motor: str, value: Any, *, num_retry: int = 0
    ) -> None: ...
    def disable_torque(self, motors: Any = None, num_retry: int = 0) -> None: ...
    def enable_torque(self, motors: Any = None, num_retry: int = 0) -> None: ...


class SafeEnergizeError(RuntimeError):
    """Raised when the startup contract cannot be verified.  Always fail closed."""


@dataclass(frozen=True)
class SafeEnergizeReport:
    """Everything the live pre-session check must be able to report."""

    motor_names: tuple[str, ...]
    present_raw_before: dict[str, int]
    goal_raw_before: dict[str, int]
    stale_goal_detected: bool
    stale_goal_reason: str
    seeded_goal_raw: dict[str, int]
    goal_raw_readback: dict[str, int]
    seed_verified: bool
    state_before_torque_enable: dict[str, float]
    state_after_torque_enable: dict[str, float]
    present_raw_after: dict[str, int]
    energize_jump: float
    energize_jump_raw_ticks: int
    max_startup_jump: float
    acceleration: int
    torque_enabled: bool
    steps: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.seed_verified
            and self.torque_enabled
            and self.energize_jump <= self.max_startup_jump
        )

    @property
    def verdict(self) -> str:
        return "SAFE_ENERGIZE = PASS" if self.passed else "SAFE_ENERGIZE = FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "motor_names": list(self.motor_names),
            "present_raw_before_energize": dict(self.present_raw_before),
            "goal_raw_before_seed": dict(self.goal_raw_before),
            "stale_goal_detected": self.stale_goal_detected,
            "stale_goal_reason": self.stale_goal_reason,
            "seeded_goal_raw": dict(self.seeded_goal_raw),
            "goal_raw_readback_after_seed": dict(self.goal_raw_readback),
            "seed_verified": self.seed_verified,
            "state_immediately_before_torque_enable": dict(
                self.state_before_torque_enable
            ),
            "state_immediately_after_torque_enable": dict(
                self.state_after_torque_enable
            ),
            "present_raw_after_energize": dict(self.present_raw_after),
            "energize_jump": self.energize_jump,
            "energize_jump_raw_ticks": self.energize_jump_raw_ticks,
            "max_startup_jump": self.max_startup_jump,
            "acceleration": self.acceleration,
            "torque_enabled": self.torque_enabled,
            "steps": list(self.steps),
            "reasoned_on": "raw registers (normalize=False)",
        }


def classify_stale_goal(
    present_raw: Mapping[str, int], goal_raw: Mapping[str, int]
) -> tuple[bool, str]:
    """Label the raw goal state.  Reasons on RAW ticks, never normalized.

    Returns ``(stale, reason)``.  ``stale`` True means the goal registers cannot
    be trusted as a holding target.  The guard seeds regardless -- this exists so
    the report says which condition the rig was actually in.
    """

    missing = [name for name in present_raw if name not in goal_raw]
    if missing:
        return True, f"goal register unreadable for {sorted(missing)}"

    all_zero = all(int(goal_raw[name]) == COLD_START_RAW_GOAL_SENTINEL
                   for name in present_raw)
    if all_zero:
        return True, (
            "cold-start signature: raw Goal_Position is 0 on every motor while "
            "Present_Position is not; energizing would drive all joints toward "
            "tick 0"
        )

    any_zero = [
        name for name in present_raw
        if int(goal_raw[name]) == COLD_START_RAW_GOAL_SENTINEL
    ]
    if any_zero:
        return True, f"raw Goal_Position is 0 on {sorted(any_zero)}"

    deltas = {
        name: abs(int(goal_raw[name]) - int(present_raw[name]))
        for name in present_raw
    }
    worst = max(deltas, key=lambda k: deltas[k])
    return False, (
        "raw goal is non-zero on every motor; largest goal-vs-present "
        f"divergence is {deltas[worst]} ticks on {worst}"
    )


def _read_map(
    bus: MotorBusLike,
    register: str,
    names: Sequence[str],
    *,
    normalize: bool,
    num_retry: int,
) -> dict[str, Any]:
    values = bus.sync_read(register, num_retry=num_retry, normalize=normalize)
    missing = [name for name in names if name not in values]
    if missing:
        raise SafeEnergizeError(
            f"{register} read did not return {sorted(missing)}; cannot verify "
            "the startup contract, failing closed"
        )
    return {name: values[name] for name in names}


def safe_energize(
    bus: MotorBusLike,
    *,
    motor_names: Sequence[str] = SO101_MUJOCO_JOINT_NAMES,
    acceleration: int = DEFAULT_ACCELERATION,
    max_startup_jump: float = DEFAULT_MAX_STARTUP_JUMP,
    post_energize_settle_s: float = DEFAULT_POST_ENERGIZE_SETTLE_S,
    read_retry: int = DEFAULT_READ_RETRY,
    write_retry: int = DEFAULT_WRITE_RETRY,
    sleep: Any = time.sleep,
) -> SafeEnergizeReport:
    """Energize the arm without ever driving it toward a stale goal.

    Order is the contract and is not negotiable:

    1. low-level bus connected (connects it if the caller has not)
    2. torque OFF
    3. read raw ``Present_Position``
    4. read raw ``Goal_Position``
    5. classify the stale-goal state
    6. seed ``Goal_Position := Present_Position`` (RAW)
    7. read the goal back and require exact equality with what was seeded
    8. only then configure registers and enable torque
    9. measure the state immediately before and after enabling torque
    10. enforce the maximum startup jump

    Any step that cannot be verified raises :class:`SafeEnergizeError`.  This
    never calls ``SO101Follower.connect()``: that path re-enables torque without
    seeding a goal, which is the entire hazard.
    """

    names = tuple(motor_names)
    if not names:
        raise SafeEnergizeError("motor_names cannot be empty")
    if max_startup_jump <= 0:
        raise SafeEnergizeError("max_startup_jump must be positive")

    steps: list[str] = []

    # 1 -- bus only.
    if not getattr(bus, "is_connected", False):
        bus.connect()
        steps.append("bus.connect")
    else:
        steps.append("bus already connected")

    # 2 -- torque OFF before anything else is read or written.
    bus.disable_torque(num_retry=write_retry)
    steps.append("disable_torque")

    # 3 / 4 -- RAW reads.  Normalized reads would clamp raw 0 to the calibrated
    # floor and hide the hazard completely.
    present_raw = {
        name: int(value)
        for name, value in _read_map(
            bus, "Present_Position", names, normalize=False, num_retry=read_retry
        ).items()
    }
    steps.append("read raw Present_Position")
    goal_raw = {
        name: int(value)
        for name, value in _read_map(
            bus, "Goal_Position", names, normalize=False, num_retry=read_retry
        ).items()
    }
    steps.append("read raw Goal_Position")

    # 5 -- classify, for the report.  The seed happens either way.
    stale, stale_reason = classify_stale_goal(present_raw, goal_raw)
    steps.append(f"classify stale goal: {stale}")

    # 6 -- seed RAW, so no truncation can shift the target off the arm.
    seed = dict(present_raw)
    bus.sync_write("Goal_Position", seed, normalize=False, num_retry=write_retry)
    steps.append("seed Goal_Position := Present_Position (raw)")

    # 7 -- verify by readback.  Exact integer equality: the raw round-trip is
    # lossless, so any mismatch is a real failure rather than rounding.
    readback = {
        name: int(value)
        for name, value in _read_map(
            bus, "Goal_Position", names, normalize=False, num_retry=read_retry
        ).items()
    }
    mismatched = {
        name: (seed[name], readback[name])
        for name in names
        if readback[name] != seed[name]
    }
    if mismatched:
        raise SafeEnergizeError(
            "seeded Goal_Position did not read back exactly; refusing to enable "
            f"torque against an unverified goal: {mismatched}"
        )
    steps.append("verify goal readback == seeded (exact)")

    # State just before energizing, in normalized units for readability.
    state_before = {
        name: float(value)
        for name, value in _read_map(
            bus, "Present_Position", names, normalize=True, num_retry=read_retry
        ).items()
    }

    # 8 -- configure, then enable torque.  Acceleration is limited first so the
    # very first motion after energizing cannot be violent either.
    for motor in names:
        bus.write("Acceleration", motor, acceleration, num_retry=write_retry)
    steps.append(f"write Acceleration={acceleration}")
    bus.enable_torque(num_retry=write_retry)
    steps.append("enable_torque")

    # 9 -- measure whether energizing moved anything.
    sleep(post_energize_settle_s)
    state_after = {
        name: float(value)
        for name, value in _read_map(
            bus, "Present_Position", names, normalize=True, num_retry=read_retry
        ).items()
    }
    present_raw_after = {
        name: int(value)
        for name, value in _read_map(
            bus, "Present_Position", names, normalize=False, num_retry=read_retry
        ).items()
    }
    jump = max(abs(state_after[name] - state_before[name]) for name in names)
    jump_raw = max(
        abs(present_raw_after[name] - present_raw[name]) for name in names
    )
    steps.append(f"measure energize jump: {jump:.4f} ({jump_raw} raw ticks)")

    report = SafeEnergizeReport(
        motor_names=names,
        present_raw_before=present_raw,
        goal_raw_before=goal_raw,
        stale_goal_detected=stale,
        stale_goal_reason=stale_reason,
        seeded_goal_raw=seed,
        goal_raw_readback=readback,
        seed_verified=True,
        state_before_torque_enable=state_before,
        state_after_torque_enable=state_after,
        present_raw_after=present_raw_after,
        energize_jump=float(jump),
        energize_jump_raw_ticks=int(jump_raw),
        max_startup_jump=float(max_startup_jump),
        acceleration=int(acceleration),
        torque_enabled=True,
        steps=steps,
    )

    # 10 -- enforce.  The arm is energized at this point, so a breach is
    # reported with the full record attached rather than losing it in a bare
    # exception: the caller must be able to log why it stopped.
    if jump > max_startup_jump:
        raise SafeEnergizeError(
            "arm moved while energizing despite a verified seeded goal "
            f"(jump={jump:.4f} > {max_startup_jump:.4f}, {jump_raw} raw ticks); "
            f"refusing to hand over a moving arm. report={report.to_dict()}"
        )
    return report


def preseed_goal_position(
    bus: MotorBusLike,
    *,
    motor_names: Sequence[str] = SO101_MUJOCO_JOINT_NAMES,
    read_retry: int = DEFAULT_READ_RETRY,
    write_retry: int = DEFAULT_WRITE_RETRY,
) -> dict[str, Any]:
    """Make a later ``SO101Follower.connect()`` safe, without replacing it.

    Steps 1-7 of the startup contract only: bus open, torque off, raw reads,
    stale classification, raw seed, exact readback verification.  Torque is NOT
    enabled here.

    This exists because the two production entry points that energize this arm
    (``scripts/serve_so101_realtime_fcpg_lewm.py:1348`` and
    ``scripts/collect_error_field_grid.py:1505``) call
    ``follower.connect(calibrate=False)``, and that call is load-bearing for
    more than energizing: ``configure()`` also writes ``Operating_Mode`` and the
    P/I/D coefficients, which nothing else in this repo writes.  Replacing it
    wholesale would silently drop that configuration.  So instead the goal
    register is seeded FIRST, and ``connect()`` is then allowed to re-enable
    torque against a target that already equals the arm's measured position.

    The caller must release the bus before calling ``follower.connect()``,
    because ``connect()`` raises ``DeviceAlreadyConnectedError`` on an open bus.
    The seed survives that release: ``Goal_Position`` lives in the servo, not in
    the host process.

    Fails closed via :class:`SafeEnergizeError` if the seed cannot be verified.
    """

    names = tuple(motor_names)
    if not names:
        raise SafeEnergizeError("motor_names cannot be empty")

    steps: list[str] = []
    if not getattr(bus, "is_connected", False):
        bus.connect()
        steps.append("bus.connect")
    else:
        steps.append("bus already connected")

    bus.disable_torque(num_retry=write_retry)
    steps.append("disable_torque")

    present_raw = {
        name: int(value)
        for name, value in _read_map(
            bus, "Present_Position", names, normalize=False, num_retry=read_retry
        ).items()
    }
    goal_raw = {
        name: int(value)
        for name, value in _read_map(
            bus, "Goal_Position", names, normalize=False, num_retry=read_retry
        ).items()
    }
    steps.append("read raw Present_Position and raw Goal_Position")

    stale, stale_reason = classify_stale_goal(present_raw, goal_raw)
    steps.append(f"classify stale goal: {stale}")

    bus.sync_write(
        "Goal_Position", dict(present_raw), normalize=False, num_retry=write_retry
    )
    steps.append("seed Goal_Position := Present_Position (raw)")

    readback = {
        name: int(value)
        for name, value in _read_map(
            bus, "Goal_Position", names, normalize=False, num_retry=read_retry
        ).items()
    }
    mismatched = {
        name: (present_raw[name], readback[name])
        for name in names
        if readback[name] != present_raw[name]
    }
    if mismatched:
        raise SafeEnergizeError(
            "seeded Goal_Position did not read back exactly; refusing to let "
            f"connect() enable torque against an unverified goal: {mismatched}"
        )
    steps.append("verify goal readback == seeded (exact)")

    return {
        "present_raw_before_energize": present_raw,
        "goal_raw_before_seed": goal_raw,
        "stale_goal_detected": stale,
        "stale_goal_reason": stale_reason,
        "seeded_goal_raw": dict(present_raw),
        "goal_raw_readback_after_seed": readback,
        "seed_verified": True,
        "torque_enabled_here": False,
        "steps": steps,
        "reasoned_on": "raw registers (normalize=False)",
    }


def safe_deenergize(
    bus: MotorBusLike, *, write_retry: int = DEFAULT_WRITE_RETRY,
    read_retry: int = DEFAULT_READ_RETRY,
) -> dict[str, Any]:
    """Leave the arm de-energized and PROVE it, per the rig's own teardown.

    Mirrors ``scripts/recover_so101_task_view.py``: disable, read
    ``Torque_Enable`` back, and disconnect with ``disable_torque=False`` --
    that flag's True path is itself a motor write, pointless once torque is
    already down.
    """

    result: dict[str, Any] = {"torque_disabled_on_exit": False}
    if not getattr(bus, "is_connected", False):
        result["note"] = "bus was not connected"
        return result
    bus.disable_torque(num_retry=write_retry)
    torque = bus.sync_read("Torque_Enable", num_retry=read_retry)
    result["torque_enable_readback"] = {k: int(v) for k, v in torque.items()}
    result["torque_disabled_on_exit"] = all(
        int(value) == 0 for value in torque.values()
    )
    return result


__all__ = [
    "COLD_START_RAW_GOAL_SENTINEL",
    "DEFAULT_ACCELERATION",
    "DEFAULT_MAX_STARTUP_JUMP",
    "MotorBusLike",
    "SafeEnergizeError",
    "SafeEnergizeReport",
    "classify_stale_goal",
    "safe_deenergize",
    "safe_energize",
]
