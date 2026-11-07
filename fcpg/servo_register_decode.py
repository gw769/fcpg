"""Authoritative STS3215 / SMS-STS register decode for SO-101.

LeRobot ``FeetechMotorsBus.sync_read`` returns:

* ``Present_Position`` / ``Goal_Position`` — normalized, sign-magnitude bit 15
* ``Present_Velocity`` — **already** sign-magnitude bit 15 (signed ticks)
* ``Present_Load`` / ``Present_Current`` — **raw unsigned words**

``STS_SMS_SERIES_ENCODINGS_TABLE`` in
``lerobot/src/lerobot/motors/feetech/tables.py`` lists velocity/position but
omits load and current.  Official Feetech SMS-STS encoding (and the SDK
primitive ``SCS_TOHOST``) is sign-magnitude: bit *N* is the direction bit,
bits below it are the magnitude.

Safety decisions in this repository must use **decoded magnitude**.  Disk
payloads keep the raw word so a historical capture can be reinterpreted.
This module never opens a bus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


#: Feetech STS3215 / SMS-STS (SO-101 follower).
SERVO_FAMILY = "feetech_sts3215_sms_sts"
LOAD_REGISTER = "Present_Load"
CURRENT_REGISTER = "Present_Current"
VELOCITY_REGISTER = "Present_Velocity"

#: Official STS/SMS Present_Load: bit 10 = CW, bits 0–9 = PWM duty magnitude.
#: One LSB is 0.1% output duty; this is not measured joint torque.
LOAD_SIGN_BIT = 10
LOAD_ADDRESS = 60
LOAD_UNIT = "pwm_duty_percent_x10"

#: SMS_STS ReadCurrent / SCS_TOHOST: bit 15 = direction. Unit 6.5 mA / LSB.
CURRENT_SIGN_BIT = 15
CURRENT_ADDRESS = 69
CURRENT_UNIT = "6.5mA"

#: Official Present_Speed and LeRobot Present_Velocity: bit 15. Unit 0.732 rpm / LSB.
VELOCITY_SIGN_BIT = 15
VELOCITY_ADDRESS = 58
VELOCITY_UNIT = "0.732rpm"

#: LeRobot already applied bit-15 decode on sync_read("Present_Velocity").
LEROBOT_DECODES = {
    LOAD_REGISTER: False,
    CURRENT_REGISTER: False,
    VELOCITY_REGISTER: True,
}

SOURCE_LEROBOT_SYNC_READ = "lerobot_sync_read"
SOURCE_RAW_REGISTER = "raw_register"
SOURCE_ALREADY_SIGNED = "already_signed"


__all__ = [
    "CURRENT_REGISTER",
    "CURRENT_SIGN_BIT",
    "CURRENT_UNIT",
    "DecodedServoRegister",
    "LEROBOT_DECODES",
    "LOAD_REGISTER",
    "LOAD_SIGN_BIT",
    "LOAD_UNIT",
    "SERVO_FAMILY",
    "SOURCE_ALREADY_SIGNED",
    "SOURCE_LEROBOT_SYNC_READ",
    "SOURCE_RAW_REGISTER",
    "VELOCITY_REGISTER",
    "VELOCITY_SIGN_BIT",
    "VELOCITY_UNIT",
    "decode_joint_map",
    "decode_present_current",
    "decode_present_load",
    "decode_present_velocity",
    "decode_sign_magnitude",
    "encoding_provenance",
    "reinterpret_register_series",
]


@dataclass(frozen=True)
class DecodedServoRegister:
    """One register word plus the signed / magnitude interpretation."""

    raw: float
    decoded: float
    magnitude: float
    direction: int
    sign_bit: int
    encoding: str
    unit: str
    source: str
    lerobot_decodes: bool
    already_signed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def encoding_provenance() -> dict[str, Any]:
    """Self-describing units for sidecars.  Does not rewrite historical files."""

    return {
        "servo_family": SERVO_FAMILY,
        "disk_values": {
            LOAD_REGISTER: "raw_unsigned",
            CURRENT_REGISTER: "raw_unsigned",
            VELOCITY_REGISTER: "signed_after_lerobot_bit15",
        },
        "safety_uses": "decoded_magnitude",
        "registers": {
            LOAD_REGISTER: {
                "address": LOAD_ADDRESS,
                "encoding": f"sign_magnitude_bit{LOAD_SIGN_BIT}",
                "sign_bit": LOAD_SIGN_BIT,
                "unit": LOAD_UNIT,
                "lerobot_decodes": False,
                "note": (
                    "Official SMS-STS Present_Load: bit 10 is direction "
                    "(0=CCW, 1=CW); bits 0-9 are output PWM duty magnitude, "
                    "0-1000 at 0.1% per LSB. Raw 1244 is 1024+220, not a "
                    "1244 amplitude. This reports drive output, not measured "
                    "joint torque; it cannot prove joint motion or contact."
                ),
            },
            CURRENT_REGISTER: {
                "address": CURRENT_ADDRESS,
                "encoding": f"sign_magnitude_bit{CURRENT_SIGN_BIT}",
                "sign_bit": CURRENT_SIGN_BIT,
                "unit": CURRENT_UNIT,
                "lerobot_decodes": False,
                "note": (
                    "LeRobot does not decode this register.  SMS_STS "
                    "ReadCurrent uses SCS_TOHOST / bit 15.  Unit 6.5 mA/LSB."
                ),
            },
            VELOCITY_REGISTER: {
                "address": VELOCITY_ADDRESS,
                "encoding": f"sign_magnitude_bit{VELOCITY_SIGN_BIT}",
                "sign_bit": VELOCITY_SIGN_BIT,
                "unit": VELOCITY_UNIT,
                "lerobot_decodes": True,
                "note": (
                    "LeRobot FeetechMotorsBus already applies bit-15 "
                    "sign-magnitude.  Live sidecars store the signed tick."
                ),
            },
        },
    }


def decode_sign_magnitude(encoded_value: int, sign_bit_index: int) -> int:
    """Same convention as LeRobot ``encoding_utils`` and SDK ``SCS_TOHOST``."""

    if sign_bit_index < 0 or sign_bit_index > 15:
        raise ValueError("sign_bit_index must lie in 0..15")
    encoded = int(encoded_value)
    if encoded < 0:
        raise ValueError("sign-magnitude decode expects an unsigned raw word")
    direction_bit = (encoded >> sign_bit_index) & 1
    magnitude = encoded & ((1 << sign_bit_index) - 1)
    return -magnitude if direction_bit else magnitude


def _finite_number(value: Any) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError("register value must be finite")
    return parsed


def _decode_word(
    value: Any,
    *,
    sign_bit: int,
    encoding: str,
    unit: str,
    source: str,
    lerobot_decodes: bool,
    already_signed_if_negative: bool,
) -> DecodedServoRegister:
    raw = _finite_number(value)
    if already_signed_if_negative and raw < 0.0:
        decoded = raw
        already_signed = True
        used_source = SOURCE_ALREADY_SIGNED
    else:
        decoded = float(decode_sign_magnitude(int(round(raw)), sign_bit))
        already_signed = False
        used_source = source
    magnitude = abs(float(decoded))
    if decoded > 0.0:
        direction = 1
    elif decoded < 0.0:
        direction = -1
    else:
        direction = 0
    return DecodedServoRegister(
        raw=raw,
        decoded=decoded,
        magnitude=magnitude,
        direction=direction,
        sign_bit=sign_bit,
        encoding=encoding,
        unit=unit,
        source=used_source,
        lerobot_decodes=lerobot_decodes,
        already_signed=already_signed,
    )


def decode_present_load(
    value: Any,
    *,
    source: str = SOURCE_LEROBOT_SYNC_READ,
) -> DecodedServoRegister:
    """Decode a Present_Load word.  ``abs(raw)`` is not a magnitude."""

    return _decode_word(
        value,
        sign_bit=LOAD_SIGN_BIT,
        encoding=f"sign_magnitude_bit{LOAD_SIGN_BIT}",
        unit=LOAD_UNIT,
        source=source,
        lerobot_decodes=False,
        already_signed_if_negative=True,
    )


def decode_present_current(
    value: Any,
    *,
    source: str = SOURCE_LEROBOT_SYNC_READ,
) -> DecodedServoRegister:
    """Decode a Present_Current word (bit 15, 6.5 mA/LSB)."""

    return _decode_word(
        value,
        sign_bit=CURRENT_SIGN_BIT,
        encoding=f"sign_magnitude_bit{CURRENT_SIGN_BIT}",
        unit=CURRENT_UNIT,
        source=source,
        lerobot_decodes=False,
        already_signed_if_negative=True,
    )


def decode_present_velocity(
    value: Any,
    *,
    source: str = SOURCE_LEROBOT_SYNC_READ,
) -> DecodedServoRegister:
    """Decode Present_Velocity.

    Live ``lerobot_sync_read`` values are already signed.  Pass
    ``source='raw_register'`` only for an unsigned bus word.
    """

    already_signed = source != SOURCE_RAW_REGISTER
    return _decode_word(
        value,
        sign_bit=VELOCITY_SIGN_BIT,
        encoding=f"sign_magnitude_bit{VELOCITY_SIGN_BIT}",
        unit=VELOCITY_UNIT,
        source=source,
        lerobot_decodes=True,
        already_signed_if_negative=already_signed,
    )


def decode_joint_map(
    values: Mapping[str, Any] | None,
    register: str,
    *,
    source: str = SOURCE_LEROBOT_SYNC_READ,
) -> dict[str, float | None]:
    """Decode a per-joint map.  Missing / non-finite entries stay None."""

    if not isinstance(values, Mapping):
        return {}
    decoder = {
        LOAD_REGISTER: decode_present_load,
        "present_load": decode_present_load,
        CURRENT_REGISTER: decode_present_current,
        "present_current": decode_present_current,
        VELOCITY_REGISTER: decode_present_velocity,
        "present_velocity": decode_present_velocity,
    }.get(register)
    if decoder is None:
        raise ValueError(f"unsupported register {register!r}")
    decoded: dict[str, float | None] = {}
    for name, raw in values.items():
        if raw is None:
            decoded[str(name)] = None
            continue
        try:
            decoded[str(name)] = float(decoder(raw, source=source).decoded)
        except (TypeError, ValueError):
            decoded[str(name)] = None
    return decoded


def reinterpret_register_series(
    values: Sequence[Any] | None,
    register: str,
    *,
    source: str = SOURCE_LEROBOT_SYNC_READ,
) -> list[float | None]:
    """Reinterpret a historical raw series.  Does not mutate the input."""

    decoder = {
        LOAD_REGISTER: decode_present_load,
        "present_load": decode_present_load,
        CURRENT_REGISTER: decode_present_current,
        "present_current": decode_present_current,
        VELOCITY_REGISTER: decode_present_velocity,
        "present_velocity": decode_present_velocity,
    }.get(register)
    if decoder is None:
        raise ValueError(f"unsupported register {register!r}")
    parsed: list[float | None] = []
    for item in values or ():
        if item is None:
            parsed.append(None)
            continue
        try:
            parsed.append(float(decoder(item, source=source).decoded))
        except (TypeError, ValueError):
            parsed.append(None)
    return parsed
