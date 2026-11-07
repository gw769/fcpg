"""Layer 1 — Hardware adapters.

Abstracts motor buses and camera devices so all upper layers are
device-agnostic.  Concrete implementations live in adapters/<vendor>/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MotorReading:
    """Raw motor state returned by a bus read."""
    motor_ids: tuple[int, ...]
    positions: tuple[float, ...]     # normalised to driver units
    velocities: tuple[float, ...] | None = None
    loads: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Frame:
    """Single camera frame with metadata."""
    data: Any                         # np.ndarray (H,W,C) uint8
    width: int
    height: int
    timestamp_s: float
    camera_id: str
    role: str                         # e.g. "wrist", "column", "overhead"
    encoding: str = "bgr"


# ── interfaces ────────────────────────────────────────────────────────────────

class RobotBusInterface(ABC):
    """Low-level motor bus: read state and write commands to drive hardware.

    Implementations are allowed to perform retries internally.  Callers
    should not assume a fixed latency budget.
    """

    @abstractmethod
    def read(self, motor_ids: list[int]) -> MotorReading:
        """Return current positions (and optionally velocities/loads)."""

    @abstractmethod
    def write(self, motor_ids: list[int], positions: list[float]) -> None:
        """Send goal positions.  Blocks until ACK or raises BusError."""

    @abstractmethod
    def enable_torque(self, motor_ids: list[int] | None = None) -> None:
        """Enable torque for the given motors (all if None)."""

    @abstractmethod
    def disable_torque(self, motor_ids: list[int] | None = None) -> None:
        """Disable torque for the given motors (all if None)."""

    @abstractmethod
    def ping(self, motor_id: int) -> bool:
        """Return True if the motor responds."""

    @abstractmethod
    def close(self) -> None:
        """Release bus resources."""


class CameraInterface(ABC):
    """Single-camera frame source.

    One instance per physical camera; the serve layer coordinates multiple
    instances.
    """

    @property
    @abstractmethod
    def role(self) -> str:
        """Logical role: 'wrist', 'column', 'overhead', etc."""

    @abstractmethod
    def start(self) -> None:
        """Open the device and begin capturing."""

    @abstractmethod
    def capture(self) -> Frame:
        """Return the latest frame.  May block briefly."""

    @abstractmethod
    def stop(self) -> None:
        """Release the device."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Return False if the device has disconnected or errored."""
