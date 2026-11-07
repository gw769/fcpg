"""Calibration continuity for operator-registered visual targets.

Why this module exists
----------------------
A visual target is registered by an operator CLICKING on one live planning
frame.  What gets persisted is a planning-canvas pixel plus an appearance crop.
Turning that pixel back into a metric base coordinate is done by the projection
calibration, so a registration only means what the calibration in force at
click time makes it mean.

:mod:`fcpg.so101_visual_targets` already refuses a profile whose *planning
geometry* differs from the runtime's (``squash_v0`` vs ``letterbox_v1``).  That
gate is necessary but not sufficient: two DIFFERENT calibration artifacts can
both declare ``letterbox_v1``: the same pixel then backprojects to two
different base coordinates and the geometry gate sees nothing wrong.  Swapping
the active calibration between registration and the collection sessions is
therefore silently undetectable on the geometry stamp alone.

What is recorded, and how it is compared
----------------------------------------
Three plain, explicit, human-readable fields, written next to the geometry
stamp in ``targets.json``:

``registration_calibration_path``
    The absolute, ``resolve()``-d path of the calibration that was in force.
``registration_calibration_name``
    That path's file name, e.g. ``calibration_letterbox_v1.json`` -- the
    artifact's version/name, which is what a human actually reasons about.
``registration_geometry_protocol``
    The geometry protocol the calibration artifact itself declares.

Comparison is ORDINARY STRING EQUALITY on those fields.  There is deliberately
no content hash, checksum or digest of any kind here: the guarantee is "the
pilot is still pointed at the same named artifact it registered against", which
is a provenance question, not an integrity question, and this project forbids
cryptographic binding.  The cost is stated honestly in
:func:`check_calibration_continuity`: editing a calibration file in place,
without renaming it, is NOT detected by this gate.

Failure is closed
-----------------
A profile that records no calibration identity is REJECTED, not assumed to
match -- exactly like the unversioned-geometry case.  A profile whose recorded
identity differs from the one now in force is REJECTED.  In both cases the
pilot continuation is invalid and an operator must re-register by clicking,
which restamps the profile against the calibration actually in force.  Nothing
here rewrites or "migrates" a profile: a re-stamped registration nobody clicked
is not a registration.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

#: Manifest key: absolute path of the calibration in force at registration.
CALIBRATION_PATH_KEY = "registration_calibration_path"

#: Manifest key: that calibration's file name (its version/name).
CALIBRATION_NAME_KEY = "registration_calibration_name"

#: Manifest key: the geometry protocol the calibration artifact declares.
CALIBRATION_PROTOCOL_KEY = "registration_geometry_protocol"

#: Every field an identity carries.  One tuple, so the writer and the reader
#: cannot drift apart the way two literal lists would.
CALIBRATION_IDENTITY_KEYS = (
    CALIBRATION_PATH_KEY,
    CALIBRATION_NAME_KEY,
    CALIBRATION_PROTOCOL_KEY,
)

#: The profile predates calibration-continuity stamping: which calibration its
#: clicks were taken against is unknown, not merely different.
CALIBRATION_UNRECORDED = "calibration_continuity_unrecorded"

#: The profile names a calibration path, and it is not the one in force.
CALIBRATION_PATH_CHANGED = "calibration_continuity_path_changed"

#: The profile names a calibration file name, and it is not the one in force.
CALIBRATION_NAME_CHANGED = "calibration_continuity_name_changed"

#: The profile's calibration declared a different geometry protocol.
CALIBRATION_PROTOCOL_CHANGED = "calibration_continuity_protocol_changed"

#: Reason -> the field whose inequality produced it.
_MISMATCH_REASONS = {
    CALIBRATION_PATH_KEY: CALIBRATION_PATH_CHANGED,
    CALIBRATION_NAME_KEY: CALIBRATION_NAME_CHANGED,
    CALIBRATION_PROTOCOL_KEY: CALIBRATION_PROTOCOL_CHANGED,
}

_REREGISTER = (
    "The pilot continuation is INVALID: stored target pixels backproject "
    "through the calibration that was in force when they were clicked. Select "
    "the intended calibration and have the operator re-register both targets "
    "on a live frame. Stored clicks are never re-pointed at a different "
    "calibration, because a re-pointed click is not a registration."
)

__all__ = [
    "CALIBRATION_PATH_KEY",
    "CALIBRATION_NAME_KEY",
    "CALIBRATION_PROTOCOL_KEY",
    "CALIBRATION_IDENTITY_KEYS",
    "CALIBRATION_UNRECORDED",
    "CALIBRATION_PATH_CHANGED",
    "CALIBRATION_NAME_CHANGED",
    "CALIBRATION_PROTOCOL_CHANGED",
    "calibration_identity",
    "calibration_identity_from_payload",
    "check_calibration_continuity",
]


def calibration_identity(
    calibration: Path | str, *, geometry_protocol: str | None = None
) -> dict[str, str]:
    """Describe the calibration in force, as plain comparable strings.

    Args:
        calibration: path to the projection calibration artifact.  It is
            ``expanduser().resolve()``-d, so a relative and an absolute
            reference to the same file compare equal.  The file need not exist
            when ``geometry_protocol`` is given explicitly.
        geometry_protocol: the protocol the artifact declares.  When ``None``
            it is READ from the artifact, which must therefore exist and
            declare one.  It is never defaulted or inferred: an artifact of
            unknown geometry is exactly what the B1 migration is guarding
            against, so an unknown protocol raises instead of being recorded as
            an empty string.

    Returns:
        The three identity fields, ready to stamp into a manifest.
    """

    if calibration is None:
        raise TypeError("calibration path is required to build an identity")
    resolved = Path(calibration).expanduser().resolve()
    protocol = geometry_protocol
    if protocol is None:
        if not resolved.is_file():
            raise ValueError(
                f"cannot determine the geometry protocol of {resolved}: the "
                "file does not exist and no geometry_protocol was declared. "
                "Geometry is never inferred."
            )
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(
                f"cannot read the calibration artifact {resolved}: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"calibration artifact {resolved} is not a JSON object"
            )
        protocol = payload.get("geometry_protocol")
        if protocol is None:
            raise ValueError(
                f"calibration artifact {resolved} declares no "
                "'geometry_protocol'. An unversioned calibration cannot anchor "
                "a registration: what its pixels mean is unknown."
            )
    return {
        CALIBRATION_PATH_KEY: str(resolved),
        CALIBRATION_NAME_KEY: resolved.name,
        CALIBRATION_PROTOCOL_KEY: str(protocol),
    }


def calibration_identity_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The identity fields actually present in a manifest payload."""

    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")
    return {
        key: payload[key] for key in CALIBRATION_IDENTITY_KEYS if key in payload
    }


def check_calibration_continuity(
    payload: Mapping[str, Any], required: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Decide whether a registered profile is still valid under ``required``.

    Returns ``None`` when the recorded calibration identity matches the one in
    force, otherwise a rejection record (``reason``, ``message``,
    ``declared_calibration``, ``required_calibration``).

    ``required=None`` means the caller makes no claim about which calibration
    is in force, which is the historical behavior: anything loads.  Once a
    caller does declare one, a profile that recorded NO identity is rejected
    rather than assumed to match, because a stored pixel is consistent with
    every calibration that shares its geometry protocol -- assuming is exactly
    the silent reinterpretation this gate exists to stop.

    Known limitation, stated rather than papered over: this compares the
    calibration's PATH, NAME and DECLARED PROTOCOL by string equality.  It
    detects selecting a different artifact.  It does NOT detect an in-place
    edit of the same file under the same name; catching that would require a
    content digest, which is deliberately out of scope here.
    """

    if required is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")
    if not isinstance(required, Mapping):
        raise TypeError(f"required must be a mapping, got {type(required).__name__}")
    missing = [key for key in CALIBRATION_IDENTITY_KEYS if key not in required]
    if missing:
        raise ValueError(
            f"required calibration identity is missing {missing}; build it with "
            "calibration_identity()"
        )

    declared = calibration_identity_from_payload(payload)
    if not declared:
        return {
            "reason": CALIBRATION_UNRECORDED,
            "message": (
                "visual target profile records no calibration identity: the "
                "projection calibration its clicks were taken against is "
                "unknown, so those clicks cannot be shown to belong to "
                f"{required[CALIBRATION_NAME_KEY]!r}. " + _REREGISTER
            ),
            "declared_calibration": declared,
            "required_calibration": dict(required),
        }

    for key in CALIBRATION_IDENTITY_KEYS:
        mine = declared.get(key)
        theirs = required[key]
        if mine is None:
            return {
                "reason": CALIBRATION_UNRECORDED,
                "message": (
                    f"visual target profile records a partial calibration "
                    f"identity: {key!r} is absent, so continuity with "
                    f"{required[CALIBRATION_NAME_KEY]!r} cannot be "
                    "established. " + _REREGISTER
                ),
                "declared_calibration": declared,
                "required_calibration": dict(required),
            }
        if str(mine) != str(theirs):
            return {
                "reason": _MISMATCH_REASONS[key],
                "message": (
                    f"visual target profile was registered with {key}={mine!r} "
                    f"but the collection session uses {theirs!r}. The same "
                    "planning pixel backprojects to a different base "
                    "coordinate under a different calibration, so the stored "
                    "clicks are not valid here. " + _REREGISTER
                ),
                "declared_calibration": declared,
                "required_calibration": dict(required),
            }
    return None
