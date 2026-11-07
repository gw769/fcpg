"""Held-out TEST seal, including sessions a reviewer would treat as TEST.

The sealed TEST physical session is ``v12_1_live_deploy_20260816``.
``v12_1_actor_v12_4_lewm20k_live_20260817`` started 12 minutes later on the
same column USB ``1-6``.  It is registered here as TEST-associated: it must
not enter TRAIN/VAL, must not be extracted as a new session, and must not be
scored as unused live data.

This module does not open cameras, serial ports, or ``*test_sealed*`` NPZ
arrays.  It does not retune planner gates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


SEALED_TEST_TOKEN = "test_sealed"
SEALED_TEST_SESSION = "v12_1_live_deploy_20260816"
SEALED_TEST_ASSOCIATED_SESSIONS = frozenset(
    {
        "v12_1_actor_v12_4_lewm20k_live_20260817",
    }
)
# Path / identity tokens that mean "this is the TEST day or its continuation".
SEALED_TEST_PATH_TOKENS = frozenset(
    {
        SEALED_TEST_TOKEN,
        SEALED_TEST_SESSION,
        *SEALED_TEST_ASSOCIATED_SESSIONS,
    }
)


class DicgTestSealError(ValueError):
    """A TEST or TEST-associated identity was used as TRAIN/VAL or new data."""


def _as_text(token: object) -> str:
    return str(token or "")


def mentions_sealed_test(token: object) -> bool:
    text = _as_text(token)
    return any(marker in text for marker in SEALED_TEST_PATH_TOKENS)


def is_sealed_test_session(name: object) -> bool:
    return _as_text(name) == SEALED_TEST_SESSION


def is_test_associated_session(name: object) -> bool:
    text = _as_text(name)
    return text in SEALED_TEST_ASSOCIATED_SESSIONS or any(
        associated in text for associated in SEALED_TEST_ASSOCIATED_SESSIONS
    )


def refuse_sealed_test_identity(*tokens: object) -> None:
    """Refuse TEST, TEST-associated, or ``test_sealed`` identities."""

    for token in tokens:
        if mentions_sealed_test(token):
            raise DicgTestSealError(
                "refusing TEST or TEST-associated identity in "
                f"{_as_text(token)!r}; v12_1_actor is sealed with "
                f"{SEALED_TEST_SESSION}"
            )


def refuse_test_associated_train_val(*tokens: object) -> None:
    """Refuse TEST-associated sessions in any TRAIN/VAL claim."""

    refuse_sealed_test_identity(*tokens)


def refuse_associated_session_extract(session_dir: Path | str) -> None:
    """Refuse extracting the actor continuation as if it were a new session.

    The original sealed TEST directory may still be walked for historical
    rebuilds.  The associated actor directory may not.
    """

    path = Path(session_dir)
    if is_test_associated_session(path.name) or is_test_associated_session(path):
        raise DicgTestSealError(
            f"refusing to extract TEST-associated session {path.name!r}; "
            f"it is sealed with {SEALED_TEST_SESSION}"
        )


def refuse_associated_motion_session(session: object) -> None:
    """Refuse motion-NPZ rows from the actor continuation in any split."""

    if is_test_associated_session(session):
        raise DicgTestSealError(
            f"refusing motion rows from TEST-associated session {session!r}"
        )


def iter_train_val_identity_tokens(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(_as_text(value) for value in values)
