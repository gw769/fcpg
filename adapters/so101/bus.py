"""SO-101 motor bus adapter.

Wraps FeetechMotorsBus (LeRobot) behind RobotBusInterface.
The retry count and all hardware params come from adapter config — not hardcoded.
"""

from __future__ import annotations

from interfaces.hardware import RobotBusInterface, MotorReading


class FeetechBusAdapter(RobotBusInterface):
    """RobotBusInterface implementation for Feetech STS series via LeRobot.

    Config keys
    -----------
    port          : serial port path, e.g. /dev/ttyUSB0
    baudrate      : int, default 1000000
    motor_ids     : list of motor IDs present on the bus
    write_retries : int, default 0 (upstream default)
                    Set > 0 to tolerate transient ACK failures.
    """

    def __init__(
        self,
        port: str,
        baudrate: int,
        motor_ids: list[int],
        write_retries: int = 0,
    ) -> None:
        from lerobot.motors.feetech import FeetechMotorsBus

        self._retries = write_retries
        self._motor_ids = motor_ids

        if write_retries > 0:
            self._patch_retries(FeetechMotorsBus, write_retries)

        self._bus = FeetechMotorsBus(port=port, baudrate=baudrate)

    @staticmethod
    def _patch_retries(cls, retries: int) -> None:
        """Raise default num_retry in enable/disable_torque without touching upstream."""
        for name in ("enable_torque", "disable_torque"):
            original = getattr(cls, name)
            def patched(self, motors=None, num_retry=retries, _orig=original):
                return _orig(self, motors, num_retry=num_retry)
            patched.__name__ = original.__name__
            setattr(cls, name, patched)

    def read(self, motor_ids: list[int]) -> MotorReading:
        positions = self._bus.read("Present_Position", motor_ids)
        return MotorReading(
            motor_ids=tuple(motor_ids),
            positions=tuple(float(p) for p in positions),
        )

    def write(self, motor_ids: list[int], positions: list[float]) -> None:
        self._bus.write("Goal_Position", positions, motor_ids)

    def enable_torque(self, motor_ids: list[int] | None = None) -> None:
        self._bus.enable_torque(motor_ids or self._motor_ids)

    def disable_torque(self, motor_ids: list[int] | None = None) -> None:
        self._bus.disable_torque(motor_ids or self._motor_ids)

    def ping(self, motor_id: int) -> bool:
        try:
            result = self._bus.ping(motor_id)
            return bool(result)
        except Exception:
            return False

    def close(self) -> None:
        self._bus.close()
